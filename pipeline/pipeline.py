#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline d'ingestion big data — 8 étapes.

Chemins :
  /mnt/storage/bronze/    — source ET espace de travail. Les sessions sont déposées
                   directement ici par l'opérateur. Tout le traitement se fait
                   sur place, en mode safe (aucune suppression sans confirmation).
 /home/ia/silver    — sortie finale validée. Écriture uniquement si write_mode=True.

Étapes :
  1. DETECT         — Vérifie l'intégrité minimale de la session dans /mnt/storage/bronze/.
  2. ROTATE         — Rotation 180° des vidéos (FFmpeg, idempotente)
  3. TRACKER        — Validation du fichier tracker_positions.csv
  4. VIDEO          — Validation des vidéos et fichiers JSONL
  4b. VERIFY_LABELS — Vérification des labels caméra left/right/head
                      (géométrie 3D trackers + fisheye + cohérence mouvement)
  5. FLUX_CSV       — Génération des flux optiques 1D (signals.py flux, Farneback)
  6. IA_SYNC        — Synchronisation fine par deep learning (sync.py)
  7. VALIDATE       — Validation de cohérence + rollback si score insuffisant
  8. STORE          — Copie vers/home/ia/silver (seulement si write_mode=True)

Chaque session traverse les étapes indépendamment.
L'état de chaque étape est persisté dans /mnt/storage/bronze//<session>/pipeline_state.json.
Tout rollback restaure les .bak créés automatiquement.
Aucune suppression dans /mnt/storage/bronze/ sauf si delete_after_store=True (opt-in explicite).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# Ajoute le dossier parent (racine du projet) au chemin pour trouver utils/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Constantes
# ══════════════════════════════════════════════════════════════════════════════

# ── Chemins fixes ─────────────────────────────────────────────────────────────
INGEST_DIR   = Path("/mnt/storage/bronze/")  # source ET espace de travail (déposé par l'opérateur)
SILVER_DIR   = Path("/home/exoria/silver")     # sortie finale validée (écriture explicite)
MODEL_DIR    = INGEST_DIR / "_sync_ml_model"
# Alias pour compatibilité rétrograde
DATASETS_DIR = INGEST_DIR

PIPELINE_STATE_FILE  = "pipeline_state.json"
PIPELINE_LOCK_FILE   = ".pipeline_lock"

# Seuils validation (alignés avec sync.py)
MIN_RELIABLE_PAIRS       = 2      # paires fiables min pour valider
MAX_SHIFT_MS             = 800.0  # shift max acceptable en ms
MIN_FLUX_CSV_ROWS        = 30     # lignes min dans un flux CSV valide

# Fichiers attendus dans une session complète
REQUIRED_FILES = [
    "metadata.json",
    "tracker_positions.csv",
]
VIDEO_SIDES = ("head", "left", "right")


# ══════════════════════════════════════════════════════════════════════════════
# Modèles d'état
# ══════════════════════════════════════════════════════════════════════════════

class StepStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    SKIPPED  = "skipped"
    FAILED   = "failed"
    ROLLED_BACK = "rolled_back"


STEP_NAMES = [
    "detect",
    "check_sync",
    "verify",
    "rotate",
    "tracker",
    "video",
    "verify_labels",
    "flux_csv",
    "ia_sync",
    "validate",
    "store",
]

# Étapes destructives qui nécessitent un backup + rollback automatique en cas d'échec
ROLLBACK_STEPS = {"rotate", "flux_csv", "ia_sync"}


@dataclass
class StepState:
    name:       str
    status:     StepStatus  = StepStatus.PENDING
    started_at: Optional[str] = None
    ended_at:   Optional[str] = None
    duration_s: float        = 0.0
    message:    str          = ""
    detail:     dict         = field(default_factory=dict)


@dataclass
class SessionPipelineState:
    session_name:   str
    session_path:   str           # chemin dans /mnt/storage/bronze/
    created_at:  str = field(default_factory=lambda: _now())
    updated_at:  str = field(default_factory=lambda: _now())
    current_step: str = "detect"
    finished:    bool = False
    success:     bool = False
    write_mode:  bool = False        # si True, copie vers/home/ia/silver après validation
    delete_after_store: bool = False # si True, supprime de /mnt/storage/bronze/
    error:       Optional[str] = None
    steps:       Dict[str, StepState] = field(default_factory=dict)
    # Résultats pour accès rapide
    n_reliable:  int   = 0
    mean_conf:   float = 0.0
    shifts:      dict  = field(default_factory=dict)
    silver_path: Optional[str] = None
    # Rétrocompatibilité lecture anciens états sauvegardés
    source_path: str = ""

    def __post_init__(self):
        for name in STEP_NAMES:
            if name not in self.steps:
                self.steps[name] = StepState(name=name)

    def to_dict(self) -> dict:
        d = asdict(self)
        # StepStatus enums → str
        for s in d["steps"].values():
            s["status"] = str(s["status"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionPipelineState":
        steps = {}
        for k, v in d.pop("steps", {}).items():
            v["status"] = StepStatus(v["status"])
            steps[k] = StepState(**v)
        obj = cls(**d)
        obj.steps = steps
        return obj

    def save(self):
        p = Path(self.session_path) / PIPELINE_STATE_FILE
        content = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        tmp = p.with_suffix(".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            if p.exists():
                try:
                    p.chmod(0o644)
                except OSError:
                    pass
            tmp.replace(p)
        except Exception:
            tmp.unlink(missing_ok=True)
            p.write_text(content, encoding="utf-8")

    @classmethod
    def load(cls, session_path: Path) -> Optional["SessionPipelineState"]:
        p = session_path / PIPELINE_STATE_FILE
        if not p.exists():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except Exception:
            return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# Logger de pipeline
# ══════════════════════════════════════════════════════════════════════════════

class PipelineLogger:
    """
    Redirige les logs vers un callback (WebSocket, fichier, stdout).
    """
    def __init__(self, session_name: str, callback: Optional[Callable] = None):
        self.session_name = session_name
        self.callback     = callback or (lambda msg, level: print(f"[{level}] {msg}"))

    def __call__(self, msg: str, level: str = "INFO"):
        full = f"[{self.session_name}] {msg}"
        self.callback(full, level)

    def step_start(self, step: str):
        self(f"▶ Étape {step.upper()}", "STEP")

    def step_done(self, step: str, detail: str = ""):
        self(f"✓ {step.upper()} — {detail}", "OK")

    def step_fail(self, step: str, err: str):
        self(f"✗ {step.upper()} — {err}", "ERROR")


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires fichiers
# ══════════════════════════════════════════════════════════════════════════════

def _backup_session(session_dir: Path, step: str) -> Path:
    """Crée un backup complet de la session avant modification."""
    backup_dir = session_dir.parent / f".backup_{session_dir.name}_{step}"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(session_dir, backup_dir, ignore=shutil.ignore_patterns(
        "*.mp4", "*.bak_syncml",  # exclure les gros fichiers vidéo
    ))
    return backup_dir


def _restore_backup(session_dir: Path, step: str) -> bool:
    """Restaure un backup créé avant une étape."""
    backup_dir = session_dir.parent / f".backup_{session_dir.name}_{step}"
    if not backup_dir.exists():
        return False
    # Restaurer uniquement les CSV/JSONL (pas les vidéos)
    for src in backup_dir.rglob("*"):
        if src.is_file() and src.suffix in (".csv", ".json", ".jsonl", ".txt"):
            rel = src.relative_to(backup_dir)
            dst = session_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    shutil.rmtree(backup_dir)
    return True


def _restore_bak_files(session_dir: Path):
    """Restaure tous les .bak_syncml créés par IA.py."""
    for bak in session_dir.rglob("*.bak_syncml"):
        original = bak.with_suffix("")  # retire .bak_syncml
        if original.suffix:  # ex: head_flux.csv.bak_syncml → head_flux.csv
            shutil.copy2(bak, original)
            bak.unlink()


def _count_flux_csv_rows(session_dir: Path) -> Dict[str, int]:
    """Retourne le nombre de lignes de chaque flux CSV."""
    out = {}
    vid_dir = session_dir / "videos"
    if not vid_dir.exists():
        return out
    for side in VIDEO_SIDES:
        p = vid_dir / f"{side}_flux.csv"
        if p.exists():
            try:
                import pandas as pd
                df = pd.read_csv(p)
                out[side] = len(df)
            except Exception:
                out[side] = 0
    return out


def _session_has_flux_csvs(session_dir: Path) -> bool:
    vid = session_dir / "videos"
    return all((vid / f"{s}_flux.csv").exists() for s in VIDEO_SIDES)


def _session_has_jsonls(session_dir: Path) -> bool:
    vid = session_dir / "videos"
    return all((vid / f"{s}.jsonl").exists() for s in VIDEO_SIDES)


def _session_has_videos(session_dir: Path) -> bool:
    vid = session_dir / "videos"
    return all((vid / f"{s}.mp4").exists() for s in VIDEO_SIDES)


# ══════════════════════════════════════════════════════════════════════════════
# Étapes de la pipeline
# ══════════════════════════════════════════════════════════════════════════════

def step_detect(state: SessionPipelineState, log: PipelineLogger) -> dict:
    """
    Étape 1 — Vérifie l'intégrité minimale de la session entrante.
    Résout les chemins, vérifie les fichiers requis.
    """
    sess = Path(state.session_path)
    issues = []

    # Fichiers obligatoires
    for f in REQUIRED_FILES:
        if not (sess / f).exists():
            issues.append(f"Fichier manquant: {f}")

    # Lire metadata
    meta_path = sess / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            log(f"Session {meta.get('session_id','?')} — scénario={meta.get('scenario','?')} "
                f"durée={meta.get('duration_seconds',0):.1f}s", "INFO")
        except Exception as e:
            issues.append(f"metadata.json illisible: {e}")
    else:
        issues.append("metadata.json manquant")

    # Vidéos
    if not _session_has_videos(sess):
        missing = [s for s in VIDEO_SIDES if not (sess / "videos" / f"{s}.mp4").exists()]
        issues.append(f"MP4 manquants: {missing}")

    # JSONL
    if not _session_has_jsonls(sess):
        missing = [s for s in VIDEO_SIDES if not (sess / "videos" / f"{s}.jsonl").exists()]
        log(f"JSONL manquants: {missing} (ancrage temporel dégradé)", "WARN")

    if issues:
        raise ValueError(f"Détection échouée: {'; '.join(issues)}")

    has_flux = _session_has_flux_csvs(sess)
    log(f"Structure OK — flux_csv={'✓' if has_flux else 'absent, sera généré'}", "OK")
    return {
        "has_flux_csv": has_flux,
        "has_jsonl":    _session_has_jsonls(sess),
        "has_videos":   _session_has_videos(sess),
    }


def step_check_sync(state: SessionPipelineState, log: PipelineLogger) -> dict:
    """
    Étape 1b — Détection du décalage vidéo/capteurs.

    Vérifie que les caméras ont démarré dans les 500 ms suivant le trigger.
    Si le délai dépasse ce seuil (~2 s observés sur les sessions décalées),
    la pipeline est bloquée car toutes les données vidéo seraient décalées
    par rapport aux capteurs (trackers, grippers).

    Signature du décalage :
      - GOOD     : 1er frame caméra arrive 40–120 ms après trigger_time_ns
      - DÉCALAGE : 1er frame caméra arrive 2 000–2 500 ms après trigger_time_ns
    """
    sess = Path(state.session_path)

    # Lire le trigger depuis metadata.json
    meta_path = sess / "metadata.json"
    meta = json.loads(meta_path.read_text())
    trigger_ns = meta.get("trigger_time_ns") or meta.get("start_time_ns")
    if trigger_ns is None:
        log("check_sync: trigger_time_ns et start_time_ns absents — vérification ignorée", "WARN")
        return {"status": "skipped", "reason": "no_trigger"}
    trigger_ms = int(trigger_ns) // 1_000_000

    videos_dir = sess / "videos"
    camera_delays: Dict[str, int] = {}
    THRESHOLD_MS = 500

    for cam in VIDEO_SIDES:
        jsonl_path = videos_dir / f"{cam}.jsonl"
        if not jsonl_path.exists():
            log(f"check_sync: {cam}.jsonl absent — caméra ignorée", "WARN")
            continue
        # Lire le premier capture_time (robuste CRLF)
        first_ms: Optional[int] = None
        with open(jsonl_path, "rb") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    try:
                        first_ms = int(json.loads(line)["capture_time"])
                    except Exception:
                        pass
                    break
        if first_ms is not None:
            camera_delays[cam] = first_ms - trigger_ms

    if not camera_delays:
        log("check_sync: aucun JSONL lisible — vérification ignorée", "WARN")
        return {"status": "skipped", "reason": "no_jsonl"}

    max_delay = max(camera_delays.values())
    result = {
        "status":           "ok" if max_delay <= THRESHOLD_MS else "decalage",
        "max_delay_ms":     max_delay,
        "threshold_ms":     THRESHOLD_MS,
        "camera_delays_ms": camera_delays,
    }

    if max_delay > THRESHOLD_MS:
        msg = (
            f"Décalage vidéo détecté : premier frame arrive {max_delay} ms après le trigger "
            f"(seuil={THRESHOLD_MS} ms, attendu < 120 ms). "
            f"Les vidéos sont désynchronisées des capteurs d'environ "
            f"{max_delay / 1000:.2f} s. Rejeter ou corriger cette session."
        )
        log(msg, "ERROR")
        raise ValueError(msg)

    log(
        f"Synchronisation OK — délai max={max_delay} ms "
        f"({', '.join(f'{c}:{d}ms' for c, d in camera_delays.items())})",
        "OK",
    )
    return result


def step_verify(state: SessionPipelineState, log: PipelineLogger) -> dict:
    """
    Étape 1c — Vérification santé de la session : trackers + désynchronisation.

    Contrôles effectués :
      Trackers
        - Monotonie des timestamps (aucun saut vers l'arrière)
        - Gaps temporels anormaux (> 200 ms entre deux échantillons)
        - Trackers gelés (déplacement total < 0.05 m sur la session)
        - Sauts de position impossibles (> 0.5 m en une étape)
        - Présence des zéros de perte de tracking
        - Distance left-right hors plage plausible (< 0.10 m ou > 1.20 m)
      Désynchronisation vidéo/capteurs
        - Délai premier frame vs trigger > 500 ms (doublon check_sync, signalé comme warning)
        - Données tracker disponibles avant le premier frame caméra (overlap utile)

    En cas d'issue : les problèmes sont écrits dans metadata.json["verification"]
    et la session est marquée has_issues=True.  L'étape ne bloque PAS la pipeline
    (elle warning) sauf si le tracker est totalement inutilisable.
    """
    import pandas as pd

    sess = Path(state.session_path)
    meta_path = sess / "metadata.json"
    meta = json.loads(meta_path.read_text())
    trigger_ns = meta.get("trigger_time_ns") or meta.get("start_time_ns")
    trigger_ms = int(trigger_ns) // 1_000_000 if trigger_ns is not None else None

    issues: List[str] = []
    warnings_list: List[str] = []
    stats: dict = {}

    # ── 1. Tracker ───────────────────────────────────────────────────────────
    tracker_csv = sess / "tracker_positions.csv"
    if not tracker_csv.exists():
        issues.append("tracker_positions.csv manquant")
    else:
        df = pd.read_csv(tracker_csv)
        n_rows = len(df)
        stats["tracker_rows"] = n_rows

        if "timestamp_ns" not in df.columns:
            issues.append("tracker_positions.csv: colonne timestamp_ns manquante")
        else:
            ts_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce").dropna().values
            ts_ms = ts_ns / 1e6
            diffs_ms = np.diff(ts_ms)

            # Monotonie
            n_neg = int((diffs_ms < 0).sum())
            if n_neg > 0:
                issues.append(f"Tracker: {n_neg} timestamp(s) non-monotone(s)")

            # Gaps > 200 ms
            n_gaps = int((diffs_ms > 200).sum())
            stats["tracker_gaps_200ms"] = n_gaps
            if n_gaps > 0:
                max_gap = float(diffs_ms.max())
                msg = f"Tracker: {n_gaps} gap(s) > 200 ms (max={max_gap:.0f} ms)"
                if max_gap > 1000:
                    issues.append(msg)
                else:
                    warnings_list.append(msg)

            # Fréquence d'échantillonnage
            dt_med = float(np.median(diffs_ms)) if len(diffs_ms) else 0
            stats["tracker_dt_ms"] = round(dt_med, 1)
            if dt_med > 50:
                warnings_list.append(f"Tracker: fréquence faible ({dt_med:.0f} ms/sample, attendu ~17 ms)")

        # Vérification positions par tracker
        for pos in ("head", "left", "right"):
            xcol, ycol, zcol = f"tracker_{pos}_x", f"tracker_{pos}_y", f"tracker_{pos}_z"
            if not all(c in df.columns for c in (xcol, ycol, zcol)):
                warnings_list.append(f"Tracker {pos}: colonnes XYZ manquantes")
                continue

            xyz = df[[xcol, ycol, zcol]].apply(pd.to_numeric, errors="coerce").dropna().values
            if len(xyz) < 2:
                issues.append(f"Tracker {pos}: pas assez de données ({len(xyz)} lignes)")
                continue

            # Tracker gelé
            disp = float(np.sqrt(((np.diff(xyz, axis=0)) ** 2).sum(axis=1)).sum())
            stats[f"tracker_{pos}_disp_m"] = round(disp, 4)
            if disp < 0.05:
                issues.append(f"Tracker {pos}: déplacement total={disp:.4f} m — tracker gelé ou absent")

            # Sauts impossibles
            step_dist = np.sqrt(((np.diff(xyz, axis=0)) ** 2).sum(axis=1))
            big_jumps = int((step_dist > 0.5).sum())
            stats[f"tracker_{pos}_big_jumps"] = big_jumps
            if big_jumps > 0:
                warnings_list.append(f"Tracker {pos}: {big_jumps} saut(s) > 0.5 m")

            # Zéros (perte tracking)
            zeros = int(((xyz[:, 0] == 0) & (xyz[:, 1] == 0) & (xyz[:, 2] == 0)).sum())
            stats[f"tracker_{pos}_zeros"] = zeros
            if zeros > n_rows * 0.02:
                warnings_list.append(f"Tracker {pos}: {zeros} échantillons à (0,0,0) — perte tracking")

        # Distance left-right plausible
        if all(c in df.columns for c in
               ("tracker_left_x", "tracker_left_y", "tracker_left_z",
                "tracker_right_x", "tracker_right_y", "tracker_right_z")):
            lxyz = df[["tracker_left_x", "tracker_left_y", "tracker_left_z"]].apply(
                pd.to_numeric, errors="coerce").dropna().values
            rxyz = df[["tracker_right_x", "tracker_right_y", "tracker_right_z"]].apply(
                pd.to_numeric, errors="coerce").dropna().values
            n = min(len(lxyz), len(rxyz))
            if n > 10:
                lr_dist = np.sqrt(((lxyz[:n] - rxyz[:n]) ** 2).sum(axis=1))
                lr_mean = float(lr_dist.mean())
                lr_min  = float(lr_dist.min())
                stats["tracker_lr_dist_mean_m"] = round(lr_mean, 3)
                stats["tracker_lr_dist_min_m"]  = round(lr_min, 3)
                if lr_min < 0.10:
                    warnings_list.append(
                        f"Trackers left/right très proches (min={lr_min:.3f} m) — collision ou swap possible"
                    )
                if lr_mean > 1.20:
                    warnings_list.append(
                        f"Distance left-right anormalement grande ({lr_mean:.3f} m) — configuration inhabituelle"
                    )

    # ── 2. Désynchronisation vidéo/capteurs ──────────────────────────────────
    videos_dir = sess / "videos"
    cam_delays: Dict[str, int] = {}
    SYNC_THRESHOLD_MS = 500

    for cam in VIDEO_SIDES:
        jsonl_path = videos_dir / f"{cam}.jsonl"
        if not jsonl_path.exists():
            continue
        first_ms: Optional[int] = None
        with open(jsonl_path, "rb") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    try:
                        first_ms = int(json.loads(line)["capture_time"])
                    except Exception:
                        pass
                    break
        if first_ms is not None and trigger_ms is not None:
            cam_delays[cam] = first_ms - trigger_ms

    stats["video_delays_ms"] = cam_delays

    if cam_delays:
        max_video_delay = max(cam_delays.values())
        stats["video_max_delay_ms"] = max_video_delay
        if max_video_delay > SYNC_THRESHOLD_MS:
            issues.append(
                f"Désynchronisation vidéo : premier frame {max_video_delay} ms après trigger "
                f"(attendu < {SYNC_THRESHOLD_MS} ms) — vidéos décalées de ~{max_video_delay/1000:.2f} s"
            )

    # ── 3. Persistance dans metadata.json ────────────────────────────────────
    has_issues = len(issues) > 0
    verification = {
        "verified_at":  _now(),
        "has_issues":   has_issues,
        "issues":       issues,
        "warnings":     warnings_list,
        "stats":        stats,
    }
    meta["verification"] = verification
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 4. Rapport ───────────────────────────────────────────────────────────
    for w in warnings_list:
        log(f"⚠ {w}", "WARN")

    if issues:
        for iss in issues:
            log(f"✗ {iss}", "ERROR")
        # On ne bloque la pipeline que si le tracker est totalement inutilisable
        if any("gelé" in i or "manquant" in i for i in issues):
            raise ValueError(
                f"Vérification échouée ({len(issues)} problème(s) critique(s)) — "
                f"session marquée. Voir metadata.json[verification]."
            )
        log(
            f"Vérification terminée avec {len(issues)} issue(s) — "
            "session marquée, pipeline continue",
            "WARN",
        )
    else:
        log(
            f"Vérification OK — {len(warnings_list)} avertissement(s)",
            "OK",
        )

    return verification


def step_rotate(state: SessionPipelineState, log: PipelineLogger,
                force: bool = False) -> dict:
    """
    Étape 2 — Rotation 180° des vidéos via FFmpeg (automatique).
    Ignorée si .rotate_done existe déjà dans videos/ (idempotente).
    """
    from utils.data_prep import rotate_session_videos

    sess = Path(state.session_path)
    result = rotate_session_videos(
        session_dir = sess,
        force       = force,
        log         = log,
    )

    if result.get("already_done"):
        log("Rotation déjà effectuée — étape sautée", "INFO")
    elif result["errors"]:
        raise RuntimeError(
            f"Rotation échouée sur : {result['errors']}. "
            "Les backups .bak_rotate ont été restaurés."
        )

    return result


def step_tracker(state: SessionPipelineState, log: PipelineLogger) -> dict:
    """
    Étape 2 — Validation et correction des trackers (sync_fix.py, dry-run).
    On valide la cohérence temporelle du tracker_positions.csv.
    La correction en place est faite à l'étape IA (step 5).
    """
    import pandas as pd

    sess = Path(state.session_path)
    csv  = sess / "tracker_positions.csv"

    df = pd.read_csv(csv)
    issues = []
    stats  = {}

    # Vérification timestamps
    if "timestamp_ns" in df.columns:
        t_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce")
        nulls = t_ns.isna().sum()
        diffs = t_ns.dropna().diff().dropna()
        neg   = (diffs < 0).sum()
        dt_ms = diffs.median() / 1e6

        stats["rows"]     = len(df)
        stats["null_ts"]  = int(nulls)
        stats["neg_ts"]   = int(neg)
        stats["dt_ms_median"] = float(dt_ms) if np.isfinite(dt_ms) else None

        if nulls > len(df) * 0.05:
            issues.append(f"{nulls} timestamps NULL ({100*nulls/len(df):.1f}%)")
        if neg > 0:
            issues.append(f"{neg} timestamps non-monotones")
        if dt_ms is not None and (dt_ms < 1.0 or dt_ms > 100.0):
            log(f"Pas tracker inhabituel: {dt_ms:.2f}ms (attendu 8-20ms)", "WARN")

    elif "time_seconds" in df.columns:
        t_s = pd.to_numeric(df["time_seconds"], errors="coerce")
        stats["rows"] = len(df)
        stats["null_ts"] = int(t_s.isna().sum())
        log("Pas de timestamp_ns — utilisation de time_seconds", "WARN")
    else:
        raise ValueError("tracker_positions.csv: aucune colonne de temps reconnue")

    # Vérification colonnes position
    expected_cols = []
    for pos in ("head", "left", "right"):
        expected_cols += [f"tracker_{pos}_x", f"tracker_{pos}_y", f"tracker_{pos}_z"]

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        log(f"Colonnes de position manquantes: {missing_cols}", "WARN")
        stats["missing_cols"] = missing_cols
    else:
        # Stats de mouvement par tracker
        for pos in ("head", "left", "right"):
            cols = [f"tracker_{pos}_x", f"tracker_{pos}_y", f"tracker_{pos}_z"]
            xyz = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(xyz) > 1:
                disp = float(np.sqrt(((xyz.diff().dropna()**2).sum(axis=1))).sum())
                stats[f"{pos}_total_displacement_m"] = round(disp, 4)

    if issues:
        raise ValueError(f"Tracker invalide: {'; '.join(issues)}")

    # ── Inférence des rôles trackers (head / left / right) ────────────────────
    try:
        from utils.data_prep import (
            find_tracker_blocks, infer_head, infer_left_right,
            _build_mapping, _global_confidence, _is_already_correct,
        )
        blocks    = find_tracker_blocks(df)
        head_info = infer_head(blocks)
        lr_info   = infer_left_right(blocks, head_info)
        mapping   = _build_mapping(head_info, lr_info)
        conf      = _global_confidence(head_info, lr_info)
        labels_ok = _is_already_correct(blocks, mapping)
        swaps     = [
            (b.role_hint, mapping[b.gid])
            for b in blocks
            if b.role_hint is not None and mapping[b.gid] != b.role_hint
        ]

        # Score de confiance rôles : 1 = parfait, 0 = problème
        if conf < 0.65:
            role_score = round(conf * 0.5, 4)
        elif swaps:
            role_score = round(conf * 0.3, 4)
        else:
            role_score = round(conf, 4)

        # Mapping gid → préfixe de colonne
        gid_to_prefix = {}
        for b in blocks:
            if b.col_names:
                prefix = "_".join(b.col_names[0].split("_")[:-1])
            else:
                prefix = b.gid
            gid_to_prefix[b.gid] = prefix
        role_to_gid = {v: k for k, v in mapping.items()}

        stats["tracker_role_score"]    = role_score
        stats["tracker_role_conf"]     = round(conf, 4)
        stats["tracker_role_head"]     = gid_to_prefix.get(role_to_gid.get("head", ""), "?")
        stats["tracker_role_left"]     = gid_to_prefix.get(role_to_gid.get("left", ""), "?")
        stats["tracker_role_right"]    = gid_to_prefix.get(role_to_gid.get("right", ""), "?")
        stats["tracker_labels_match"]  = labels_ok
        stats["tracker_role_swaps"]    = swaps

        axis_names = {0: "X", 1: "Y", 2: "Z"}
        sign_str   = "+" if head_info.up_axis_sign > 0 else "-"
        axis_str   = sign_str + axis_names.get(head_info.up_axis_index, "?")
        log(
            f"Rôles trackers — score={role_score:.3f} conf={conf:.3f} axe_vertical={axis_str} "
            f"head={stats['tracker_role_head']} left={stats['tracker_role_left']} right={stats['tracker_role_right']}"
            + (f" SWAPS={swaps}" if swaps else ""),
            "WARN" if swaps or conf < 0.65 else "OK",
        )
    except Exception as e:
        log(f"Inférence rôles trackers échouée (non bloquant) : {e}", "WARN")
        stats["tracker_role_score"] = 0.0
        stats["tracker_role_conf"]  = 0.0

    log(f"Tracker OK — {stats.get('rows',0)} lignes, dt_médian={stats.get('dt_ms_median','?')}ms", "OK")
    return stats


def step_video(state: SessionPipelineState, log: PipelineLogger) -> dict:
    """
    Étape 3 — Validation des vidéos et fichiers JSONL.
    Vérifie cohérence frame count / timestamps JSONL.
    """
    import pandas as pd

    sess    = Path(state.session_path)
    vid_dir = sess / "videos"
    stats   = {}

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
        log("OpenCV non disponible — validation vidéo limitée", "WARN")

    for side in VIDEO_SIDES:
        mp4   = vid_dir / f"{side}.mp4"
        jsonl = vid_dir / f"{side}.jsonl"

        if not mp4.exists():
            log(f"{side}.mp4 manquant", "WARN")
            continue

        side_stats = {"mp4": str(mp4.name)}

        # Lire metadata MP4 via OpenCV
        if has_cv2:
            cap = cv2.VideoCapture(str(mp4))
            if cap.isOpened():
                fps    = cap.get(cv2.CAP_PROP_FPS)
                n_frm  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                dur_s  = n_frm / fps if fps > 0 else 0
                cap.release()
                side_stats.update({"fps": fps, "frames": n_frm, "w": w, "h": h, "duration_s": round(dur_s,2)})
                log(f"{side}: {n_frm} frames @ {fps:.1f}fps — {w}×{h} — {dur_s:.1f}s", "INFO")

        # Vérifier JSONL
        if jsonl.exists():
            with open(jsonl) as f:
                lines = [l.strip() for l in f if l.strip()]
            valid_ts = []
            for line in lines:
                try:
                    rec = json.loads(line)
                    valid_ts.append(float(rec["capture_time"]))
                except Exception:
                    pass
            side_stats["jsonl_records"] = len(lines)
            side_stats["jsonl_valid_ts"] = len(valid_ts)

            if valid_ts:
                diffs_ms = np.diff(valid_ts)
                dt_med   = float(np.median(diffs_ms))
                side_stats["jsonl_dt_ms_median"] = round(dt_med, 2)
                # Détection de sauts > 3x la médiane
                jumps = int((diffs_ms > dt_med * 3.0).sum())
                if jumps > 0:
                    log(f"{side} JSONL: {jumps} saut(s) temporel(s) détecté(s)", "WARN")
                    side_stats["jsonl_time_jumps"] = jumps

                # Cohérence frame count / JSONL
                if has_cv2 and "frames" in side_stats:
                    diff_frames = abs(side_stats["frames"] - len(valid_ts))
                    if diff_frames > 5:
                        log(f"{side}: {diff_frames} frames de décalage MP4/JSONL", "WARN")
                        side_stats["frame_jsonl_diff"] = diff_frames
        else:
            log(f"{side}.jsonl manquant — ancrage temporel absolu indisponible", "WARN")
            side_stats["jsonl_records"] = 0

        stats[side] = side_stats

    log(f"Vidéos validées: {list(stats.keys())}", "OK")
    return stats


def step_verify_labels(state: SessionPipelineState, log: PipelineLogger,
                       min_confidence: float = 0.90) -> dict:
    """
    Étape 4b — Vérification des labels caméra (left / right / head).

    Utilise verify_video_labels.py avec 3 niveaux de preuve :
      1. Géométrie 3D des trackers VIVE (source de vérité principale)
      2. Détection fisheye grand angle (head)
      3. Cohérence flux de mouvement left ↔ right

    Mode safe uniquement : écrit le résultat dans metadata.json sous
    la clé "camera_label_verification". Aucun fichier n'est renommé.

    Bloque la pipeline si la confiance est insuffisante (< min_confidence)
    ou si un label prédit diffère du label déclaré.
    """
    import utils.data_prep as vvl

    sess    = Path(state.session_path)
    vid_dir = sess / "videos"

    # Résolution des chemins vidéo
    paths: dict = {}
    for label in ("left", "right", "head"):
        for ext in (".mp4", ".MP4"):
            p = vid_dir / f"{label}{ext}"
            if p.exists():
                paths[label] = str(p)
                break

    missing = [l for l in ("left", "right", "head") if l not in paths]
    if missing:
        raise FileNotFoundError(
            f"Vérification labels impossible — vidéos manquantes : {missing}"
        )

    tracker_csv  = sess / "tracker_positions.csv"
    metadata_path = sess / "metadata.json"

    log("Analyse géométrique 3D des trackers…", "INFO")

    # ── Niveau 1 : trackers ───────────────────────────────────────────────
    tracker_result = vvl.analyze_trackers(str(tracker_csv)) \
        if tracker_csv.exists() else vvl.TrackerAssignment()

    if not tracker_result.ok:
        log("Trackers : confiance insuffisante — passage en analyse visuelle seule", "WARN")
    else:
        log(
            f"Trackers OK — head='{tracker_result.head_tracker_id}' "
            f"left='{tracker_result.left_tracker_id}' "
            f"right='{tracker_result.right_tracker_id}' "
            f"conf={tracker_result.confidence:.2f}",
            "OK",
        )

    # ── Niveau 2 : fisheye ────────────────────────────────────────────────
    log("Analyse fisheye des vidéos…", "INFO")
    fisheye: dict = {}
    for label, path in paths.items():
        fi = vvl.analyze_fisheye(label, path)
        fisheye[label] = fi
        tag = "FISHEYE" if fi.is_fisheye else "normal"
        log(f"  {label}: score_fisheye={fi.score:.3f} [{tag}]", "INFO")

    # ── Niveau 3 : mouvement ──────────────────────────────────────────────
    log("Analyse cohérence mouvement…", "INFO")
    motion = vvl.analyze_motion(paths)
    log(
        f"  left↔right={motion.lr_correlation:+.3f}  "
        f"left↔head={motion.lh_correlation:+.3f}  "
        f"right↔head={motion.rh_correlation:+.3f}",
        "OK" if motion.left_right_consistent else "WARN",
    )

    # ── Verdicts ──────────────────────────────────────────────────────────
    verdicts, recommended, confidence = vvl.compute_verdicts(
        paths, tracker_result, fisheye, motion
    )

    # ── Rapport : écriture safe dans metadata.json ────────────────────────
    report = vvl.VerificationReport(
        session_dir         = str(sess),
        global_ok           = all(v.label_correct for v in verdicts),
        confidence          = confidence,
        tracker             = tracker_result,
        fisheye             = fisheye,
        motion              = motion,
        verdicts            = verdicts,
        recommended_mapping = recommended,
        safe_mode           = True,
    )
    report.summary = (
        f"Labels corrects (confiance={confidence:.0%})"
        if report.global_ok
        else ", ".join(
            f"{v.declared_label}→{v.predicted_label}"
            for v in verdicts if not v.label_correct
        )
    )

    if metadata_path.exists():
        vvl.write_safe_report(str(metadata_path), report)

    # ── PNG de vérification ───────────────────────────────────────────────
    png_path = sess / "label_verification.png"
    try:
        vvl.save_png(paths, verdicts, fisheye, motion, tracker_result, str(png_path))
        log(f"Rapport visuel → {png_path.name}", "INFO")
    except Exception as e:
        log(f"PNG non généré : {e}", "WARN")

    # ── Décision pipeline ─────────────────────────────────────────────────
    bad = [v for v in verdicts if not v.label_correct]
    if bad:
        corrections = ", ".join(
            f"'{v.declared_label}'→'{v.predicted_label}'" for v in bad
        )
        raise ValueError(
            f"Labels caméra incorrects détectés : {corrections}. "
            f"Corrigez les fichiers avant de relancer la pipeline. "
            f"Le rapport est disponible dans metadata.json "
            f"(clé 'camera_label_verification')."
        )

    if confidence < min_confidence:
        raise ValueError(
            f"Confiance insuffisante pour les labels caméra : {confidence:.0%} "
            f"< {min_confidence:.0%} requis. "
            f"Vérifiez manuellement les vidéos."
        )

    log(
        f"Labels vérifiés ✓ — head/left/right corrects "
        f"(confiance={confidence:.0%})",
        "OK",
    )

    return {
        "global_ok":         report.global_ok,
        "confidence":        round(confidence, 4),
        "tracker_conf":      round(tracker_result.confidence, 4),
        "head_fisheye_score": round(fisheye.get("head", vvl.FisheyeResult("head")).score, 4),
        "lr_correlation":    round(motion.lr_correlation, 4),
        "recommended_mapping": recommended,
        "verdicts": [
            {
                "label":     v.declared_label,
                "predicted": v.predicted_label,
                "ok":        v.label_correct,
            }
            for v in verdicts
        ],
    }


def _run_video_flux(side: str, mp4: Path, out_csv: Path, jsonl: Path, signals_py: Path) -> tuple:
    """Lance signals.py flux pour une seule caméra — exécuté en parallèle."""
    cmd = [
        sys.executable, str(signals_py), "flux",
        str(mp4),
        "--output-csv",    str(out_csv),
        "--resize-width",  "640",
        "--smooth-window", "5",
    ]
    if jsonl.exists():
        cmd += ["--jsonl", str(jsonl)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return side, result.returncode, result.stderr.strip()[:400]
    except subprocess.TimeoutExpired:
        return side, -1, "timeout 600s"
    except Exception as e:
        return side, -2, str(e)


def step_flux_csv(state: SessionPipelineState, log: PipelineLogger,
                  force: bool = False) -> dict:
    """
    Étape 4 — Génération des flux CSV optiques via signals.py flux (Farneback).
    Les 3 caméras sont traitées en parallèle (ProcessPoolExecutor).
    Passe en mode skipped si les flux existent déjà et force=False.
    """
    sess        = Path(state.session_path)
    vid_dir     = sess / "videos"
    signals_py  = _ROOT / "utils" / "signals.py"

    if not signals_py.exists():
        raise FileNotFoundError(f"signals.py introuvable: {signals_py}")

    generated = []
    skipped   = []
    errors    = []

    # Déterminer quelles caméras ont besoin d'être (re)générées
    to_generate = []
    for side in VIDEO_SIDES:
        mp4     = vid_dir / f"{side}.mp4"
        out_csv = vid_dir / f"{side}_flux.csv"
        jsonl   = vid_dir / f"{side}.jsonl"

        if not mp4.exists():
            log(f"{side}.mp4 absent — flux skipped", "WARN")
            continue

        if out_csv.exists() and not force:
            rows = _count_flux_csv_rows(sess).get(side, 0)
            if rows >= MIN_FLUX_CSV_ROWS:
                skipped.append(side)
                log(f"{side}_flux.csv existant ({rows} lignes) — skip", "INFO")
                continue
            else:
                log(f"{side}_flux.csv trop court ({rows} lignes) — régénération", "WARN")

        log(f"Génération flux {side}…", "INFO")
        to_generate.append((side, mp4, out_csv, jsonl))

    # Lancer les conversions en parallèle (une vidéo par worker)
    if to_generate:
        n_workers = min(len(to_generate), os.cpu_count() or 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(_run_video_flux, side, mp4, out_csv, jsonl, signals_py): side
                for side, mp4, out_csv, jsonl in to_generate
            }
            for fut in concurrent.futures.as_completed(futures):
                side, returncode, stderr = fut.result()
                if returncode != 0:
                    errors.append(f"{side}: {stderr}")
                    log(f"video.py {side} ERREUR: {stderr}", "ERROR")
                else:
                    rows = _count_flux_csv_rows(sess).get(side, 0)
                    if rows < MIN_FLUX_CSV_ROWS:
                        errors.append(f"{side}: flux CSV trop court ({rows} lignes)")
                    else:
                        generated.append(side)
                        log(f"{side}_flux.csv généré — {rows} lignes", "OK")

    if errors and not generated and not skipped:
        raise RuntimeError(f"Génération flux CSV échouée: {'; '.join(errors)}")

    total_ok = len(generated) + len(skipped)
    if total_ok < len(VIDEO_SIDES):
        log(f"Seulement {total_ok}/{len(VIDEO_SIDES)} flux CSV disponibles", "WARN")

    return {
        "generated": generated,
        "skipped":   skipped,
        "errors":    errors,
        "total_ok":  total_ok,
    }


def step_ia_sync(state: SessionPipelineState, log: PipelineLogger,
                 model_dir: Path, params: dict) -> dict:
    """
    Étape 5 — Synchronisation fine par IA (IA.py, CrossModalAligner).
    Applique les offsets uniquement sur les paires fiables.
    Crée des backups .bak_syncml automatiquement.
    """
    import utils.sync as ia

    sess = Path(state.session_path)

    if not (model_dir / "model.pt").exists():
        raise RuntimeError(
            f"Modèle IA absent ({model_dir}/model.pt). "
            "Entraînez le modèle d'abord depuis l'interface."
        )

    model  = ia.load_model(model_dir)
    fluxes = ia.load_all_fluxes(sess)

    # Lancer toutes les estimations de paires en parallèle (I/O + inférence IA)
    _valid_pairs = [
        (ref_name, tgt_name)
        for ref_name, tgt_name in ia.PAIRS
        if ref_name in fluxes and tgt_name in fluxes
    ]
    for ref_name, tgt_name in ia.PAIRS:
        if ref_name not in fluxes or tgt_name not in fluxes:
            log(f"Paire {ref_name}↔{tgt_name} ignorée (flux manquant)", "WARN")

    def _estimate_one(ref_name: str, tgt_name: str) -> dict:
        est = ia.estimate_pair_offset(
            model       = model,
            ref         = fluxes[ref_name],
            tgt         = fluxes[tgt_name],
            resample_ms = params["resample_ms"],
            max_lag_ms  = params["max_lag_ms"],
            window_ms   = params["window_ms"],
        )
        return {
            "ref_name":          est.ref_name,
            "tgt_name":          est.tgt_name,
            "delta_start_ms":    float(est.delta_start_ms),
            "residual_ms":       float(est.residual_ms),
            "shift_to_apply_ms": float(est.shift_to_apply_ms),
            "confidence":        float(est.confidence),
            "peak_margin":       float(est.peak_margin),
            "is_reliable":       bool(est.is_reliable),
            "method":            est.method,
        }

    estimates = []
    _n_workers = min(len(_valid_pairs), os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_n_workers, thread_name_prefix="ia-pair"
    ) as _tex:
        _futs = {
            _tex.submit(_estimate_one, ref, tgt): (ref, tgt)
            for ref, tgt in _valid_pairs
        }
        for _fut in concurrent.futures.as_completed(_futs):
            d = _fut.result()  # propagera l'exception si une paire échoue
            estimates.append(d)
            log(
                f"{d['ref_name']} ↔ {d['tgt_name']}  shift={d['shift_to_apply_ms']:+.1f}ms  "
                f"conf={d['confidence']:.3f}  reliable={d['is_reliable']}",
                "OK" if d["is_reliable"] else "WARN",
            )
    # Remettre dans l'ordre PAIRS pour la reproductibilité
    _pair_order = {(r, t): i for i, (r, t) in enumerate(ia.PAIRS)}
    estimates.sort(key=lambda d: _pair_order.get((d["ref_name"], d["tgt_name"]), 99))

    # Appliquer les offsets fiables
    applied = set()
    shifts_applied = {}
    for ed in estimates:
        if not ed["is_reliable"]:
            continue
        tgt = ed["tgt_name"]
        if tgt in applied:
            continue
        shift = ed["shift_to_apply_ms"]
        if abs(shift) > MAX_SHIFT_MS:
            log(f"Shift {shift:+.1f}ms sur {tgt} dépasse MAX_SHIFT_MS={MAX_SHIFT_MS} — ignoré", "WARN")
            continue
        ia.apply_shift_to_target(sess, tgt, shift, dry_run=False)
        applied.add(tgt)
        shifts_applied[tgt] = shift
        log(f"Offset appliqué: {tgt}  {shift:+.1f}ms", "OK")

    # Sauvegarder les résultats
    result_payload = {
        "generated_at": _now(),
        "pairs":        estimates,
        "applied":      shifts_applied,
    }
    (sess / ia.RESULTS_JSON).write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    n_reliable  = sum(1 for e in estimates if e["is_reliable"])
    mean_conf   = float(np.mean([e["confidence"] for e in estimates])) if estimates else 0.0

    return {
        "estimates":   estimates,
        "n_reliable":  n_reliable,
        "n_total":     len(estimates),
        "mean_conf":   mean_conf,
        "applied":     shifts_applied,
    }


def step_validate(state: SessionPipelineState, log: PipelineLogger,
                  ia_result: dict) -> dict:
    """
    Étape 6 — Validation déterministe post-sync.
    Utilise le score d'alignement (AlignReport) plutôt que la confiance IA.
    Si la qualité est insuffisante, déclenche un rollback automatique.
    """
    import utils.sync as _sync

    sess = Path(state.session_path)

    n_reliable = ia_result.get("n_reliable", 0)
    n_total    = ia_result.get("n_total", 0)
    estimates  = ia_result.get("estimates", [])

    issues   = []
    warnings = []

    # Règle 1 : assez de paires fiables
    if n_total == 0:
        issues.append("Aucune paire évaluée")
    elif n_reliable < MIN_RELIABLE_PAIRS:
        issues.append(
            f"Seulement {n_reliable}/{n_total} paires fiables "
            f"(minimum requis: {MIN_RELIABLE_PAIRS})"
        )

    # Règle 2 : score déterministe post-application sur paires majeures
    major_scores = []
    for ref_name, tgt_name in _sync.MAJOR_PAIRS:
        result = _sync._score_pair_deterministic(sess, ref_name, tgt_name)
        if result is not None and result.get("score") is not None:
            major_scores.append(result["score"])
            log(f"Score post-sync {ref_name}↔{tgt_name}: {result['score']:.1f}/100", "INFO")

    if major_scores:
        avg_major = float(np.mean(major_scores))
        if avg_major < _sync.MIN_MAJOR_SCORE_POST:
            issues.append(
                f"Score déterministe moyen {avg_major:.1f} < {_sync.MIN_MAJOR_SCORE_POST} requis"
            )
    elif not issues:  # pas de scores mais pas déjà en erreur
        warnings.append("Score déterministe post-sync indisponible (flux manquants?)")

    # Règle 3 : shifts cohérents (pas de valeurs aberrantes entre paires liées)
    shifts = {e["tgt_name"]: e["shift_to_apply_ms"] for e in estimates}
    cam_shifts = [v for k, v in shifts.items() if k.startswith("cam_")]
    if len(cam_shifts) >= 2:
        std_cam = float(np.std(cam_shifts))
        if std_cam > 200.0:
            warnings.append(f"Dispersion élevée des shifts caméra: σ={std_cam:.1f}ms")
            log(f"WARN: dispersion shifts caméra σ={std_cam:.1f}ms", "WARN")

    # Règle 4 : flux CSV toujours valides après modification
    flux_rows = _count_flux_csv_rows(sess)
    for side, rows in flux_rows.items():
        if rows < MIN_FLUX_CSV_ROWS:
            issues.append(f"Flux CSV {side} dégradé après sync ({rows} lignes)")

    # ── Décision ──
    if issues:
        log(f"Validation ÉCHOUÉE: {'; '.join(issues)}", "ERROR")
        # Rollback
        log("Rollback en cours…", "WARN")
        ok = _restore_backup(sess, "ia_sync")
        if ok:
            log("Rollback réussi — données restaurées", "OK")
        else:
            _restore_bak_files(sess)
            log("Rollback via .bak_syncml", "WARN")
        raise ValueError(f"Validation échouée (rollback effectué): {'; '.join(issues)}")

    for w in warnings:
        log(f"Warning: {w}", "WARN")

    avg_score = float(np.mean(major_scores)) if major_scores else None

    # Écrire marqueur de validation dans metadata.json
    meta_path = sess / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            meta["ia_sync_validated"]    = _now()
            meta["ia_sync_n_reliable"]   = n_reliable
            meta["ia_sync_avg_score"]    = round(avg_score, 2) if avg_score is not None else None
            meta["ia_sync_shifts_ms"]    = shifts
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            log(f"Impossible d'écrire le marqueur metadata: {e}", "WARN")

    log(
        f"Validation OK — {n_reliable}/{n_total} paires fiables "
        f"score_moy={avg_score:.1f}/100" if avg_score is not None else
        f"Validation OK — {n_reliable}/{n_total} paires fiables",
        "OK",
    )
    return {
        "valid":      True,
        "n_reliable": n_reliable,
        "avg_score":  avg_score,
        "warnings":   warnings,
        "shifts":     shifts,
    }


def step_store(state: SessionPipelineState, log: PipelineLogger) -> dict:
    """
    Étape 8 — Copie la session traitée de /mnt/storage/bronze/ vers/home/ia/silver.
    N'est appelée que si write_mode=True dans l'état de session.
    Exclut les fichiers temporaires (.bak, locks, backups).
    Si delete_after_store=True, supprime la session de /mnt/storage/bronze/ après copie.
    """
    sess        = Path(state.session_path)   # dans /mnt/storage/bronze/
    silver_path = SILVER_DIR / sess.name

    if not SILVER_DIR.exists():
        raise FileNotFoundError(
            f"/home/ia/silver non accessible : {SILVER_DIR}\n"
            "Vérifiez que le montage est actif avant d'activer le mode écriture."
        )

    log(f"Écriture vers {silver_path}…", "INFO")
    silver_path.mkdir(parents=True, exist_ok=True)

    copied      = []
    skipped     = []
    total_bytes = 0

    for src in sess.rglob("*"):
        if not src.is_file():
            continue
        # Exclure les artefacts de traitement
        if src.suffix == ".bak_syncml":
            continue
        if src.name == PIPELINE_LOCK_FILE:
            continue
        if ".backup_" in str(src):
            continue

        rel = src.relative_to(sess)
        dst = silver_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Pas de réécriture si le fichier est identique (taille + mtime)
        if dst.exists():
            ss, ds = src.stat(), dst.stat()
            if ss.st_size == ds.st_size and abs(ss.st_mtime - ds.st_mtime) < 1.0:
                skipped.append(str(rel))
                continue

        shutil.copy2(src, dst)
        copied.append(str(rel))
        total_bytes += src.stat().st_size

    # Nettoyer les répertoires de backup temporaires dans ingest
    for bk in sess.parent.glob(f".backup_{sess.name}_*"):
        shutil.rmtree(bk, ignore_errors=True)

    log(
        f"Silver OK — {len(copied)} fichiers copiés "
        f"({total_bytes / 1024 / 1024:.1f} Mo) vers {silver_path}",
        "OK",
    )

    # Suppression de /mnt/storage/bronze/ seulement si opt-in explicite
    deleted = False
    if state.delete_after_store:
        log(f"Suppression de {sess} (delete_after_store=True)…", "WARN")
        shutil.rmtree(sess, ignore_errors=True)
        deleted = True
        log("Session supprimée de /mnt/storage/bronze/", "OK")
    else:
        log("Session conservée dans /mnt/storage/bronze/ (delete_after_store=False)", "INFO")

    return {
        "silver_path": str(silver_path),
        "copied":      len(copied),
        "skipped":     len(skipped),
        "total_mb":    round(total_bytes / 1024 / 1024, 2),
        "deleted_from_ingest": deleted,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrateur principal
# ══════════════════════════════════════════════════════════════════════════════

class PipelineRunner:
    """
    Exécute la pipeline complète pour une session.
    Thread-safe, callbacks pour les logs et updates.
    """

    def __init__(
        self,
        source_path:       str,
        params:            dict,
        write_mode:        bool = False,
        delete_after_store: bool = False,
        log_callback:      Optional[Callable] = None,
        step_callback:     Optional[Callable] = None,
        force_flux:        bool = False,
        resume:            bool = True,
        steps_whitelist:   Optional[List[str]] = None,
    ):
        self.session_path  = Path(source_path)
        self.session_name  = self.session_path.name
        self.model_dir     = MODEL_DIR
        self.write_mode    = write_mode
        self.delete_after_store = delete_after_store
        self.params        = params
        self.log_callback  = log_callback
        self.step_callback = step_callback
        self.force_flux      = force_flux
        self.resume          = resume
        self.steps_whitelist = set(steps_whitelist) if steps_whitelist else None
        self.log = PipelineLogger(self.session_name, log_callback)

    def _step_ctx(self, state: SessionPipelineState, name: str):
        """Retourne un context manager qui enregistre durée + statut."""
        return _StepContext(state, name, self.log, self.step_callback)

    def run(self) -> SessionPipelineState:
        sess = self.session_path

        if not sess.exists():
            raise FileNotFoundError(
                f"Session introuvable dans /mnt/storage/bronze/: {sess}\n"
                "Déposez la session dans /mnt/storage/bronze/ avant de lancer la pipeline."
            )

        # Charger ou créer l'état dans le dossier session
        state = SessionPipelineState.load(sess) if self.resume else None
        if state is None:
            state = SessionPipelineState(
                session_name        = self.session_name,
                session_path        = str(sess),
                write_mode          = self.write_mode,
                delete_after_store  = self.delete_after_store,
            )
        else:
            state.write_mode         = self.write_mode
            state.delete_after_store = self.delete_after_store
        state.save()

        if state.finished and state.success and not self.force_flux:
            self.log("Session déjà traitée avec succès — skip", "INFO")
            return state

        lock = sess / PIPELINE_LOCK_FILE
        if lock.exists():
            self.log("Session verrouillée par un autre processus", "WARN")
            return state
        lock.write_text(_now())

        try:
            ia_result = {}

            # ── Étape 1 : Détection ──
            if self._should_run(state, "detect"):
                with self._step_ctx(state, "detect") as ctx:
                    ctx.result = step_detect(state, self.log)

            # ── Étape 1b : Vérification synchronisation vidéo/capteurs ──
            if self._should_run(state, "check_sync"):
                with self._step_ctx(state, "check_sync") as ctx:
                    ctx.result = step_check_sync(state, self.log)

            # ── Étape 1c : Vérification santé (trackers + désynchronisation) ──
            if self._should_run(state, "verify"):
                with self._step_ctx(state, "verify") as ctx:
                    ctx.result = step_verify(state, self.log)

            # ── Étape 2 : Rotation 180° ──
            if self._should_run(state, "rotate"):
                with self._step_ctx(state, "rotate") as ctx:
                    ctx.result = step_rotate(state, self.log, force=self.force_flux)

            # ── Étapes 3 / 4 / 4b : Tracker + Vidéo + Labels (parallèle) ──
            _parallel_steps = [
                ("tracker",       lambda: step_tracker(state, self.log)),
                ("video",         lambda: step_video(state, self.log)),
                ("verify_labels", lambda: step_verify_labels(
                    state, self.log,
                    min_confidence=self.params.get("label_min_confidence", 0.90),
                )),
            ]
            _to_run = [(n, fn) for n, fn in _parallel_steps if self._should_run(state, n)]
            if _to_run:
                _par_errors: Dict[str, Exception] = {}

                def _run_step(name: str, fn: Callable) -> None:
                    with self._step_ctx(state, name) as ctx:
                        ctx.result = fn()

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(_to_run), thread_name_prefix="pipe-step"
                ) as _tex:
                    _futs = {_tex.submit(_run_step, n, fn): n for n, fn in _to_run}
                    for _fut in concurrent.futures.as_completed(_futs):
                        _n = _futs[_fut]
                        exc = _fut.exception()
                        if exc:
                            _par_errors[_n] = exc
                if _par_errors:
                    raise RuntimeError(
                        "Étapes parallèles échouées: "
                        + "; ".join(f"{n}: {e}" for n, e in _par_errors.items())
                    )

            # ── Étape 5 : Flux CSV ──
            if self._should_run(state, "flux_csv"):
                with self._step_ctx(state, "flux_csv") as ctx:
                    ctx.result = step_flux_csv(state, self.log, force=self.force_flux)

            # ── Étape 6 : IA Sync ──
            if self._should_run(state, "ia_sync"):
                with self._step_ctx(state, "ia_sync") as ctx:
                    ctx.result = step_ia_sync(state, self.log, self.model_dir, self.params)
                    ia_result  = ctx.result

            # Récupérer ia_result si reprise depuis validate
            if not ia_result and state.steps.get("ia_sync"):
                ia_result = state.steps["ia_sync"].detail

            # ── Étape 7 : Validation ──
            if self._should_run(state, "validate"):
                with self._step_ctx(state, "validate") as ctx:
                    ctx.result = step_validate(state, self.log, ia_result)
                    state.n_reliable = ctx.result.get("n_reliable", 0)
                    state.mean_conf  = ctx.result.get("avg_score", 0.0) or 0.0
                    state.shifts     = ctx.result.get("shifts", {})

            # ── Étape 8 : Store vers/home/ia/silver ──
            if self._should_run(state, "store"):
                if state.write_mode:
                    with self._step_ctx(state, "store") as ctx:
                        ctx.result = step_store(state, self.log)
                        state.silver_path = ctx.result.get("silver_path")
                else:
                    state.steps["store"].status  = StepStatus.SKIPPED
                    state.steps["store"].message = "write_mode désactivé — aucune écriture vers/home/ia/silver"
                    self.log("Store ignoré (write_mode=False) — données disponibles dans /mnt/storage/bronze/", "INFO")

            state.finished = True
            state.success  = True
            state.current_step = "done"
            self.log("Pipeline terminée avec succès ✓", "OK")

        except Exception as e:
            state.error    = traceback.format_exc()
            state.finished = True
            state.success  = False
            self.log(f"Pipeline échouée: {e}", "ERROR")

        finally:
            state.updated_at = _now()
            state.save()
            lock.unlink(missing_ok=True)

        if self.step_callback:
            self.step_callback(state)

        return state

    def _should_run(self, state: SessionPipelineState, name: str) -> bool:
        # Exclure si la whitelist est définie et que l'étape n'y figure pas
        if self.steps_whitelist is not None and name not in self.steps_whitelist:
            return False
        s = state.steps.get(name)
        if s is None:
            return True
        if s.status in (StepStatus.DONE, StepStatus.SKIPPED):
            return False
        return True


class _StepContext:
    """Context manager pour une étape : chrono + statut + sauvegarde + rollback automatique."""
    def __init__(self, state: SessionPipelineState, name: str,
                 log: PipelineLogger, callback: Optional[Callable]):
        self.state    = state
        self.name     = name
        self.log      = log
        self.callback = callback
        self.result   = {}
        self._t0      = 0.0

    def __enter__(self):
        step = self.state.steps[self.name]
        step.status     = StepStatus.RUNNING
        step.started_at = _now()
        self.state.current_step = self.name
        self.state.updated_at   = _now()
        self.state.save()
        self._t0 = time.time()
        # Backup automatique avant les étapes destructives
        if self.name in ROLLBACK_STEPS:
            sess = Path(self.state.session_path)
            try:
                _backup_session(sess, self.name)
                self.log(f"Backup créé avant étape {self.name.upper()}", "INFO")
            except Exception as e:
                self.log(f"Avertissement: backup impossible pour {self.name}: {e}", "WARN")
        self.log.step_start(self.name)
        if self.callback:
            self.callback(self.state)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        step = self.state.steps[self.name]
        step.ended_at   = _now()
        step.duration_s = round(time.time() - self._t0, 2)
        if exc_type is None:
            step.status  = StepStatus.DONE
            step.detail  = self.result if isinstance(self.result, dict) else {}
            step.message = f"OK ({step.duration_s}s)"
            self.log.step_done(self.name, f"{step.duration_s}s")
        else:
            step.status  = StepStatus.FAILED
            step.message = str(exc_val)[:200]
            self.log.step_fail(self.name, str(exc_val)[:200])
            # Rollback automatique pour les étapes destructives
            if self.name in ROLLBACK_STEPS:
                sess = Path(self.state.session_path)
                self.log(f"Rollback automatique après échec de {self.name.upper()}…", "WARN")
                try:
                    ok = _restore_backup(sess, self.name)
                    if ok:
                        step.status  = StepStatus.ROLLED_BACK
                        step.message = f"ROLLBACK ({str(exc_val)[:150]})"
                        self.log(f"Rollback {self.name.upper()} réussi — état restauré", "OK")
                    else:
                        self.log(f"Rollback {self.name.upper()} impossible (backup introuvable)", "ERROR")
                except Exception as rb_err:
                    self.log(f"Erreur rollback {self.name.upper()}: {rb_err}", "ERROR")
        self.state.updated_at = _now()
        self.state.save()
        if self.callback:
            self.callback(self.state)
        return False  # propager l'exception


# ══════════════════════════════════════════════════════════════════════════════
# Watcher d'ingestion (polling)
# ══════════════════════════════════════════════════════════════════════════════

class IngestionWatcher:
    """
    Surveille /mnt/storage/bronze/ et lance la pipeline automatiquement
    sur les nouvelles sessions déposées par l'opérateur.
    Si write_mode=True, les sessions validées sont copiées vers/home/ia/silver.
    """

    def __init__(
        self,
        params:             dict,
        write_mode:         bool  = False,
        delete_after_store: bool  = False,
        log_callback:       Optional[Callable] = None,
        step_callback:      Optional[Callable] = None,
        poll_interval:      float = 10.0,
        auto_start:         bool  = False,
        watch_dir:          Optional[Path] = None,
    ):
        self.watch_dir          = Path(watch_dir) if watch_dir else INGEST_DIR
        self.write_mode         = write_mode
        self.delete_after_store = delete_after_store
        self.params             = params
        self.log_callback       = log_callback
        self.step_callback      = step_callback
        self.poll_interval      = poll_interval
        self.auto_start         = auto_start

        self._running      = False
        self._thread       = None
        self._seen:  set   = set()
        self._queue: List[Path] = []
        self._lock         = threading.Lock()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _log(self, msg: str, level: str = "INFO"):
        if self.log_callback:
            self.log_callback(f"[Watcher] {msg}", level)

    def _loop(self):
        self._log(f"Surveillance de {self.watch_dir} (interval={self.poll_interval}s)", "INFO")
        while self._running:
            try:
                self._scan()
                if self.auto_start:
                    self._process_queue()
            except Exception as e:
                self._log(f"Erreur watcher: {e}", "ERROR")
            time.sleep(self.poll_interval)

    def _scan(self):
        """Scanne /mnt/storage/bronze/ pour de nouvelles sessions déposées."""
        if not self.watch_dir.exists():
            return
        for child in sorted(self.watch_dir.iterdir()):
            if not child.is_dir():
                continue
            # Ignorer le répertoire du modèle ML
            if child.name.startswith("_"):
                continue
            if str(child) in self._seen:
                continue
            # Ignorer si déjà traitée avec succès
            existing = SessionPipelineState.load(child)
            if existing and existing.finished and existing.success:
                self._seen.add(str(child))
                continue
            if (child / "metadata.json").exists():
                self._log(f"Nouvelle session détectée: {child.name}", "INFO")
                with self._lock:
                    self._queue.append(child)
                self._seen.add(str(child))

    def _run_one_session(self, sess: Path):
        self._log(f"Lancement pipeline: {sess.name}", "INFO")
        runner = PipelineRunner(
            source_path        = str(sess),
            params             = self.params,
            write_mode         = self.write_mode,
            delete_after_store = self.delete_after_store,
            log_callback       = self.log_callback,
            step_callback      = self.step_callback,
        )
        runner.run()

    def _process_queue(self):
        with self._lock:
            queue = list(self._queue)
            self._queue.clear()

        if not queue:
            return

        n_workers = min(len(queue), os.cpu_count() or 1)
        if n_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                list(ex.map(self._run_one_session, queue))
        else:
            self._run_one_session(queue[0])

    def get_queue(self) -> List[str]:
        with self._lock:
            return [str(s) for s in self._queue]

    def get_seen(self) -> List[str]:
        return list(self._seen)

    def enqueue(self, session_path: str):
        """Enfile manuellement une session (chemin dans /mnt/storage/bronze/)."""
        with self._lock:
            self._queue.append(Path(session_path))

    def run_session_now(self, session_path: str) -> SessionPipelineState:
        """Lance la pipeline immédiatement sur une session (bloquant)."""
        runner = PipelineRunner(
            source_path        = session_path,
            params             = self.params,
            write_mode         = self.write_mode,
            delete_after_store = self.delete_after_store,
            log_callback       = self.log_callback,
            step_callback      = self.step_callback,
        )
        return runner.run()


# ══════════════════════════════════════════════════════════════════════════════
# Prétraitement dataset local (absorbé depuis preprocess_and_train.py)
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = _ROOT / "utils"
SIDES = ("head", "left", "right")


def _run_subprocess(cmd: list, label: str) -> bool:
    """Lance une commande et affiche sa sortie en temps réel. Retourne True si succès."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[ERREUR] Code retour {result.returncode} pour : {label}", file=sys.stderr)
        return False
    return True


def _find_dataset_sessions(dataset_dir: Path) -> list:
    return sorted(
        p for p in dataset_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_") and not p.name.endswith("__FAILED")
    )


def preprocess_dataset(dataset_dir: Path, force: bool) -> int:
    """
    Pour chaque session du dataset, génère les flux optiques manquants via signals.py flux.
    Retourne le nombre d'erreurs.
    """
    signals_py = _SCRIPT_DIR / "signals.py"
    if not signals_py.exists():
        print(f"[ERREUR] signals.py introuvable: {signals_py}", file=sys.stderr)
        return 1

    sessions = _find_dataset_sessions(dataset_dir)
    print(f"\n[PREPROCESS] {len(sessions)} sessions trouvées dans {dataset_dir}")

    generated = skipped = errors = 0

    for session in sessions:
        videos_dir = session / "videos"
        if not videos_dir.exists():
            print(f"  [SKIP] {session.name} — dossier videos/ absent")
            skipped += 1
            continue

        for side in SIDES:
            mp4  = videos_dir / f"{side}.mp4"
            jsonl = videos_dir / f"{side}.jsonl"
            flux = videos_dir / f"{side}_flux.csv"

            if not mp4.exists():
                print(f"  [SKIP] {session.name}/{side} — vidéo absente")
                skipped += 1
                continue

            if flux.exists() and not force:
                print(f"  [OK]   {session.name}/{side}_flux.csv déjà présent")
                skipped += 1
                continue

            cmd = [
                sys.executable, str(signals_py), "flux",
                str(mp4),
                "--output-csv", str(flux),
            ]
            if jsonl.exists():
                cmd += ["--jsonl", str(jsonl)]

            ok = _run_subprocess(cmd, f"{session.name} — flux {side}")
            if ok:
                generated += 1
            else:
                errors += 1

    print(f"\n[PREPROCESS] Terminé — {generated} générés, {skipped} ignorés, {errors} erreurs")
    return errors


def train_dataset(dataset_dir: Path, extra_args: list) -> bool:
    """
    Lance sync.py train sur le dataset local.
    Surcharge ROOT_DIR de sync via wrapper inline.
    """
    sessions = _find_dataset_sessions(dataset_dir)
    ready = []
    for s in sessions:
        flux_files = list((s / "videos").glob("*_flux.csv")) if (s / "videos").exists() else []
        if len(flux_files) >= 3:
            ready.append(s.name)
        else:
            print(f"  [SKIP entraînement] {s.name} — seulement {len(flux_files)}/3 flux CSV")

    print(f"\n[TRAIN] {len(ready)} sessions prêtes pour l'entraînement")
    if not ready:
        print("[TRAIN] Aucune session utilisable. Lancez d'abord le prétraitement.")
        return False

    # Wrapper inline : importe sync, patch ROOT_DIR + _session_is_clean, appelle main()
    wrapper = (
        f"import sys; sys.path.insert(0, {str(_SCRIPT_DIR)!r}); "
        f"import sync; from pathlib import Path; "
        f"sync.ROOT_DIR = Path({str(dataset_dir)!r}); "
        f"sync._session_is_clean = lambda d: len(list((d / 'videos').glob('*_flux.csv'))) >= 3 "
        f"if (d / 'videos').exists() else False; "
        f"sys.exit(sync.main())"
    )
    cmd = [sys.executable, "-c", wrapper, "train"] + extra_args
    return _run_subprocess(cmd, "Entraînement IA (sync.py train)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI standalone
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    top = argparse.ArgumentParser(
        description=(
            "Pipeline d'ingestion big data\n"
            "\nSous-commandes :\n"
            f"  run        — traite une ou toutes les sessions de {INGEST_DIR}\n"
            "  preprocess — génère les flux optiques sur un dataset local\n"
            "  train      — entraîne le modèle IA sur un dataset local\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = top.add_subparsers(dest="cmd", required=True)

    # ── Sous-commande : run ────────────────────────────────────────────────
    p_run = sub.add_parser("run", help=f"Traiter des sessions dans {INGEST_DIR}")
    p_run.add_argument("--bronze-dir", type=Path, default=Path("/mnt/storage/bronze/"),
                       metavar="DIR",
                       help="Dossier de travail bronze (défaut: /mnt/storage/bronze/)")
    grp = p_run.add_mutually_exclusive_group(required=True)
    grp.add_argument("--session", metavar="NAME",
                     help="Traiter une session spécifique dans --bronze-dir")
    grp.add_argument("--all", action="store_true",
                     help="Traiter toutes les sessions non encore traitées")
    p_run.add_argument("--write", action="store_true", default=False,
                       help=f"Copier vers {SILVER_DIR} après validation")
    p_run.add_argument("--delete-after-store", action="store_true", default=False,
                       help=f"Supprimer de {INGEST_DIR} après copie (requiert --write)")
    p_run.add_argument("--resample", type=float, default=5.0,    help="Rééchantillonnage (ms)")
    p_run.add_argument("--max-lag",  type=float, default=400.0,  help="Lag max recherché (ms)")
    p_run.add_argument("--window",   type=float, default=2200.0, help="Fenêtre d'analyse (ms)")
    p_run.add_argument("--force-flux", action="store_true",
                       help="Régénérer les flux CSV même s'ils existent")
    p_run.add_argument("--no-resume", action="store_true",
                       help="Ignorer l'état sauvegardé et recommencer depuis le début")
    p_run.add_argument("--label-min-confidence", type=float, default=0.90,
                       help="Confiance minimale labels caméra (défaut: 0.90)")

    # ── Sous-commande : preprocess ─────────────────────────────────────────
    p_pre = sub.add_parser("preprocess", help="Génère les flux optiques sur un dataset local")
    p_pre.add_argument("--dataset", type=Path,
                       default=_ROOT / "dataset",
                       help="Dossier dataset (défaut: ./dataset)")
    p_pre.add_argument("--force", action="store_true",
                       help="Recalcule les flux même s'ils existent déjà")

    # ── Sous-commande : train ──────────────────────────────────────────────
    p_tr = sub.add_parser("train", help="Entraîne le modèle IA sur un dataset local")
    p_tr.add_argument("--dataset", type=Path,
                      default=_ROOT / "dataset",
                      help="Dossier dataset (défaut: ./dataset)")
    p_tr.add_argument("--only-preprocess", action="store_true",
                      help="Lance uniquement le prétraitement, pas l'entraînement")
    p_tr.add_argument("--only-train", action="store_true",
                      help="Lance uniquement l'entraînement (flux déjà présents)")
    p_tr.add_argument("--force",  action="store_true",
                      help="Recalcule les flux même s'ils existent déjà")
    p_tr.add_argument("--epochs",     type=int,   default=None)
    p_tr.add_argument("--batch-size", type=int,   default=None)
    p_tr.add_argument("--lr",         type=float, default=None)

    args = top.parse_args()

    # ── Dossier de travail dynamique (run uniquement) ───────────────────────
    if args.cmd == "run":
        import sys as _sys
        _mod = _sys.modules[__name__]
        _mod.INGEST_DIR   = args.bronze_dir
        _mod.MODEL_DIR    = args.bronze_dir / "_sync_ml_model"
        _mod.DATASETS_DIR = args.bronze_dir
        INGEST_DIR   = args.bronze_dir
        MODEL_DIR    = _mod.MODEL_DIR
        DATASETS_DIR = _mod.DATASETS_DIR

    # ── preprocess ─────────────────────────────────────────────────────────
    if args.cmd == "preprocess":
        if not args.dataset.exists():
            print(f"[ERREUR] Dataset introuvable : {args.dataset}", file=sys.stderr)
            sys.exit(1)
        errors_n = preprocess_dataset(args.dataset, force=args.force)
        sys.exit(1 if errors_n > 0 else 0)

    # ── train ──────────────────────────────────────────────────────────────
    if args.cmd == "train":
        if not args.dataset.exists():
            print(f"[ERREUR] Dataset introuvable : {args.dataset}", file=sys.stderr)
            sys.exit(1)
        do_pre   = not args.only_train
        do_train = not args.only_preprocess
        if do_pre:
            errors_n = preprocess_dataset(args.dataset, force=args.force)
            if errors_n > 0:
                print(f"\n[ATTENTION] {errors_n} flux non générés.", file=sys.stderr)
        if do_train:
            extra: list = []
            if args.epochs     is not None: extra += ["--epochs",     str(args.epochs)]
            if args.batch_size is not None: extra += ["--batch-size", str(args.batch_size)]
            if args.lr         is not None: extra += ["--lr",         str(args.lr)]
            ok = train_dataset(args.dataset, extra)
            if not ok:
                sys.exit(1)
        print("\n[DONE] Terminé.")
        sys.exit(0)

    # ── run ────────────────────────────────────────────────────────────────
    run_errors = []
    if not INGEST_DIR.exists():
        run_errors.append(f"{INGEST_DIR} non accessible")
    if args.write and not SILVER_DIR.exists():
        run_errors.append(f"/home/ia/silver non accessible : {SILVER_DIR}  (requis par --write)")
    if args.delete_after_store and not args.write:
        run_errors.append("--delete-after-store requiert --write")
    if run_errors:
        for e in run_errors:
            print(f"ERREUR : {e}", file=sys.stderr)
        sys.exit(1)

    params = {
        "resample_ms":          args.resample,
        "max_lag_ms":           args.max_lag,
        "window_ms":            args.window,
        "label_min_confidence": args.label_min_confidence,
    }

    # ── Résolution des sessions à traiter (depuis /mnt/storage/bronze/) ────────────
    if args.session:
        source_paths = [INGEST_DIR / args.session]
        if not source_paths[0].is_dir():
            print(f"ERREUR : session introuvable dans {INGEST_DIR} : {args.session}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        all_sessions = sorted(
            s for s in INGEST_DIR.iterdir()
            if s.is_dir()
            and not s.name.startswith("_")
            and (s / "metadata.json").exists()
        )
        if not all_sessions:
            print(f"Aucune session trouvée dans {INGEST_DIR}", file=sys.stderr)
            sys.exit(1)
        pending = []
        for s in all_sessions:
            existing = SessionPipelineState.load(s)
            if existing and existing.finished and existing.success:
                continue
            pending.append(s)
        print(
            f"  {len(all_sessions)} sessions dans {INGEST_DIR} — "
            f"{len(all_sessions) - len(pending)} déjà traitées — "
            f"{len(pending)} à traiter"
        )
        source_paths = pending
        if not source_paths:
            print("  Rien à faire.")
            sys.exit(0)

    # ── Affichage du mode ──────────────────────────────────────────────────
    write_label  = f"OUI → {SILVER_DIR}" if args.write else f"NON (résultats dans {INGEST_DIR})"
    delete_label = "OUI (suppression après store)" if args.delete_after_store else "NON"
    print(f"\n  Travail   : {INGEST_DIR}")
    print(f"  Écriture  : {write_label}")
    print(f"  Suppression ingest : {delete_label}")
    print(f"  Sessions  : {len(source_paths)}\n")

    _print_lock = threading.Lock()

    def _cli_log(msg, level="INFO"):
        icons = {"OK": "✓", "ERROR": "✗", "WARN": "⚠", "STEP": "▶", "TRAIN": "◉"}
        with _print_lock:
            print(f"  {icons.get(level, '·')} {msg}")

    def _run_session_cli(source_path: Path) -> SessionPipelineState:
        with _print_lock:
            print(f"\n{'═'*60}")
            print(f"  {source_path.name}")
            print(f"  {source_path}")
            print(f"{'═'*60}")
        runner = PipelineRunner(
            source_path        = str(source_path),
            params             = params,
            write_mode         = args.write,
            delete_after_store = args.delete_after_store,
            log_callback       = _cli_log,
            force_flux         = args.force_flux,
            resume             = not args.no_resume,
        )
        state = runner.run()
        icon = "✓" if state.success else "✗"
        silver_info = f"  silver={state.silver_path}" if state.silver_path else ""
        score_info = f"{state.mean_conf:.1f}/100" if state.mean_conf else "?"
        with _print_lock:
            print(f"\n  {icon} {'SUCCÈS' if state.success else 'ÉCHEC'}  "
                  f"score={score_info}  fiables={state.n_reliable}{silver_info}")
            for name, step in state.steps.items():
                s_icon = "✓" if step.status == StepStatus.DONE   else \
                         "✗" if step.status == StepStatus.FAILED  else \
                         "○" if step.status == StepStatus.SKIPPED else "·"
                print(f"    {s_icon} {name:<15} {step.duration_s:5.1f}s  {step.message[:55]}")
        return state

    n_ok    = 0
    n_fail  = 0
    results: List[SessionPipelineState] = []

    n_workers = min(len(source_paths), os.cpu_count() or 1)
    if n_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_run_session_cli, source_paths))
    else:
        results = [_run_session_cli(source_paths[0])]

    for state in results:
        if state.success:
            n_ok += 1
        else:
            n_fail += 1

    if len(source_paths) > 1:
        print(f"\n{'═'*60}")
        print(f"  Bilan : {n_ok} succès, {n_fail} échec(s) sur {len(source_paths)} sessions")

    last_state = results[-1] if results else None
    sys.exit(0 if (last_state and last_state.success) else 1)
