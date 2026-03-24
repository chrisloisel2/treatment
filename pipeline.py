#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline d'ingestion big data — 8 étapes.

Chemins :
  /mnt/ingest    — source ET espace de travail. Les sessions sont déposées
                   directement ici par l'opérateur. Tout le traitement se fait
                   sur place, en mode safe (aucune suppression sans confirmation).
  /mnt/silver    — sortie finale validée. Écriture uniquement si write_mode=True.

Étapes :
  1. DETECT         — Vérifie l'intégrité minimale de la session dans /mnt/ingest.
  2. ROTATE         — Rotation 180° des vidéos (FFmpeg, idempotente)
  3. TRACKER        — Validation du fichier tracker_positions.csv
  4. VIDEO          — Validation des vidéos et fichiers JSONL
  4b. VERIFY_LABELS — Vérification des labels caméra left/right/head
                      (géométrie 3D trackers + fisheye + cohérence mouvement)
  5. FLUX_CSV       — Génération des flux optiques 1D (video.py, Farneback)
  6. IA_SYNC        — Synchronisation fine par deep learning (IA.py)
  7. VALIDATE       — Validation de cohérence + rollback si confiance insuffisante
  8. STORE          — Copie vers /mnt/silver (seulement si write_mode=True)

Chaque session traverse les étapes indépendamment.
L'état de chaque étape est persisté dans /mnt/ingest/<session>/pipeline_state.json.
Tout rollback restaure les .bak créés automatiquement.
Aucune suppression dans /mnt/ingest sauf si delete_after_store=True (opt-in explicite).
"""

from __future__ import annotations

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


# ══════════════════════════════════════════════════════════════════════════════
# Constantes
# ══════════════════════════════════════════════════════════════════════════════

# ── Chemins fixes ─────────────────────────────────────────────────────────────
INGEST_DIR   = Path("/mnt/ingest")     # source ET espace de travail (déposé par l'opérateur)
SILVER_DIR   = Path("/mnt/silver")     # sortie finale validée (écriture explicite)
MODEL_DIR    = INGEST_DIR / "_sync_ml_model"
# Alias pour compatibilité rétrograde
DATASETS_DIR = INGEST_DIR

PIPELINE_STATE_FILE  = "pipeline_state.json"
PIPELINE_LOCK_FILE   = ".pipeline_lock"

# Seuils validation
MIN_RELIABLE_PAIRS       = 2      # paires fiables min pour valider
MIN_MEAN_CONFIDENCE      = 0.62   # confiance moyenne min
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
    "rotate",
    "tracker",
    "video",
    "verify_labels",
    "flux_csv",
    "ia_sync",
    "validate",
    "store",
]


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
    session_path:   str           # chemin dans /mnt/ingest (source et espace de travail)
    created_at:  str = field(default_factory=lambda: _now())
    updated_at:  str = field(default_factory=lambda: _now())
    current_step: str = "detect"
    finished:    bool = False
    success:     bool = False
    write_mode:  bool = False        # si True, copie vers /mnt/silver après validation
    delete_after_store: bool = False # si True, supprime de /mnt/ingest après store
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
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

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


def step_rotate(state: SessionPipelineState, log: PipelineLogger,
                force: bool = False) -> dict:
    """
    Étape 2 — Rotation 180° des vidéos via FFmpeg (automatique).
    Ignorée si .rotate_done existe déjà dans videos/ (idempotente).
    """
    from rotate_videos import rotate_session_videos

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
    import verify_video_labels as vvl

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


def step_flux_csv(state: SessionPipelineState, log: PipelineLogger,
                  force: bool = False) -> dict:
    """
    Étape 4 — Génération des flux CSV optiques via video.py (Farneback).
    Passe en mode skipped si les flux existent déjà et force=False.
    """
    sess    = Path(state.session_path)
    vid_dir = sess / "videos"
    video_py = Path(__file__).parent / "video.py"

    if not video_py.exists():
        raise FileNotFoundError(f"video.py introuvable: {video_py}")

    generated = []
    skipped   = []
    errors    = []

    for side in VIDEO_SIDES:
        mp4      = vid_dir / f"{side}.mp4"
        out_csv  = vid_dir / f"{side}_flux.csv"
        jsonl    = vid_dir / f"{side}.jsonl"

        if not mp4.exists():
            log(f"{side}.mp4 absent — flux skipped", "WARN")
            continue

        if out_csv.exists() and not force:
            # Vérifier que le CSV existant est valide
            rows = _count_flux_csv_rows(sess).get(side, 0)
            if rows >= MIN_FLUX_CSV_ROWS:
                skipped.append(side)
                log(f"{side}_flux.csv existant ({rows} lignes) — skip", "INFO")
                continue
            else:
                log(f"{side}_flux.csv trop court ({rows} lignes) — régénération", "WARN")

        log(f"Génération flux {side}…", "INFO")
        cmd = [
            sys.executable, str(video_py),
            str(mp4),
            "--output-csv",    str(out_csv),
            "--resize-width",  "640",
            "--smooth-window", "5",
        ]
        if jsonl.exists():
            cmd += ["--jsonl", str(jsonl)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max par vidéo
            )
            if result.returncode != 0:
                err = result.stderr.strip()[:400]
                errors.append(f"{side}: {err}")
                log(f"video.py {side} ERREUR: {err}", "ERROR")
            else:
                rows = _count_flux_csv_rows(sess).get(side, 0)
                if rows < MIN_FLUX_CSV_ROWS:
                    errors.append(f"{side}: flux CSV trop court ({rows} lignes)")
                else:
                    generated.append(side)
                    log(f"{side}_flux.csv généré — {rows} lignes", "OK")
        except subprocess.TimeoutExpired:
            errors.append(f"{side}: timeout 600s")
            log(f"video.py {side} TIMEOUT", "ERROR")
        except Exception as e:
            errors.append(f"{side}: {e}")

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
    import IA as ia

    sess = Path(state.session_path)

    if not (model_dir / "model.pt").exists():
        raise RuntimeError(
            f"Modèle IA absent ({model_dir}/model.pt). "
            "Entraînez le modèle d'abord depuis l'interface."
        )

    # Backup avant modification
    _backup_session(sess, "ia_sync")
    log("Backup créé avant synchronisation IA", "INFO")

    model  = ia.load_model(model_dir)
    fluxes = ia.load_all_fluxes(sess)

    estimates = []
    for ref_name, tgt_name in ia.PAIRS:
        if ref_name not in fluxes or tgt_name not in fluxes:
            log(f"Paire {ref_name}↔{tgt_name} ignorée (flux manquant)", "WARN")
            continue

        est = ia.estimate_pair_offset(
            model       = model,
            ref         = fluxes[ref_name],
            tgt         = fluxes[tgt_name],
            resample_ms = params["resample_ms"],
            max_lag_ms  = params["max_lag_ms"],
            window_ms   = params["window_ms"],
        )

        d = {
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
        estimates.append(d)
        log(
            f"{ref_name} ↔ {tgt_name}  shift={est.shift_to_apply_ms:+.1f}ms  "
            f"conf={est.confidence:.3f}  reliable={est.is_reliable}",
            "OK" if est.is_reliable else "WARN",
        )

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
    Étape 6 — Validation finale.
    Si la qualité est insuffisante, déclenche un rollback automatique.
    """
    sess = Path(state.session_path)

    n_reliable = ia_result.get("n_reliable", 0)
    n_total    = ia_result.get("n_total", 0)
    mean_conf  = ia_result.get("mean_conf", 0.0)
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

    # Règle 2 : confiance moyenne
    if mean_conf < MIN_MEAN_CONFIDENCE:
        issues.append(
            f"Confiance moyenne {mean_conf:.3f} < {MIN_MEAN_CONFIDENCE}"
        )

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

    # Écrire marqueur de validation dans metadata.json
    meta_path = sess / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            meta["ia_sync_validated"]    = _now()
            meta["ia_sync_n_reliable"]   = n_reliable
            meta["ia_sync_mean_conf"]    = round(mean_conf, 4)
            meta["ia_sync_shifts_ms"]    = shifts
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            log(f"Impossible d'écrire le marqueur metadata: {e}", "WARN")

    log(
        f"Validation OK — {n_reliable}/{n_total} paires fiables "
        f"conf_moy={mean_conf:.3f}",
        "OK",
    )
    return {
        "valid":      True,
        "n_reliable": n_reliable,
        "mean_conf":  mean_conf,
        "warnings":   warnings,
        "shifts":     shifts,
    }


def step_store(state: SessionPipelineState, log: PipelineLogger) -> dict:
    """
    Étape 8 — Copie la session traitée de /mnt/ingest vers /mnt/silver.
    N'est appelée que si write_mode=True dans l'état de session.
    Exclut les fichiers temporaires (.bak, locks, backups).
    Si delete_after_store=True, supprime la session de /mnt/ingest après copie.
    """
    sess        = Path(state.session_path)   # dans /mnt/ingest
    silver_path = SILVER_DIR / sess.name

    if not SILVER_DIR.exists():
        raise FileNotFoundError(
            f"/mnt/silver non accessible : {SILVER_DIR}\n"
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

    # Suppression de /mnt/ingest seulement si opt-in explicite
    deleted = False
    if state.delete_after_store:
        log(f"Suppression de {sess} (delete_after_store=True)…", "WARN")
        shutil.rmtree(sess, ignore_errors=True)
        deleted = True
        log("Session supprimée de /mnt/ingest", "OK")
    else:
        log("Session conservée dans /mnt/ingest (delete_after_store=False)", "INFO")

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
        source_path:       str,          # chemin de la session dans /mnt/ingest
        params:            dict,
        write_mode:        bool = False, # si True, copie vers /mnt/silver après validation
        delete_after_store: bool = False, # si True, supprime de /mnt/ingest après store
        log_callback:      Optional[Callable] = None,
        step_callback:     Optional[Callable] = None,
        force_flux:        bool = False,
        resume:            bool = True,
    ):
        self.session_path  = Path(source_path)
        self.session_name  = self.session_path.name
        self.model_dir     = MODEL_DIR
        self.write_mode    = write_mode
        self.delete_after_store = delete_after_store
        self.params        = params
        self.log_callback  = log_callback
        self.step_callback = step_callback
        self.force_flux    = force_flux
        self.resume        = resume

        self.log = PipelineLogger(self.session_name, log_callback)

    def _step_ctx(self, state: SessionPipelineState, name: str):
        """Retourne un context manager qui enregistre durée + statut."""
        return _StepContext(state, name, self.log, self.step_callback)

    def run(self) -> SessionPipelineState:
        sess = self.session_path

        if not sess.exists():
            raise FileNotFoundError(
                f"Session introuvable dans /mnt/ingest : {sess}\n"
                "Déposez la session dans /mnt/ingest avant de lancer la pipeline."
            )

        # Charger ou créer l'état (dans /mnt/ingest/<session>/)
        state = SessionPipelineState.load(sess) if self.resume else None
        if state is None:
            state = SessionPipelineState(
                session_name        = self.session_name,
                session_path        = str(sess),
                write_mode          = self.write_mode,
                delete_after_store  = self.delete_after_store,
            )
            state.save()
        else:
            # Mettre à jour les options de la session courante
            state.write_mode         = self.write_mode
            state.delete_after_store = self.delete_after_store

        if state.finished and state.success and not self.force_flux:
            self.log("Session déjà traitée avec succès — skip", "INFO")
            return state

        # Verrou fichier pour éviter double-traitement concurrent
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

            # ── Étape 2 : Rotation 180° ──
            if self._should_run(state, "rotate"):
                with self._step_ctx(state, "rotate") as ctx:
                    ctx.result = step_rotate(state, self.log, force=self.force_flux)

            # ── Étape 3 : Tracker ──
            if self._should_run(state, "tracker"):
                with self._step_ctx(state, "tracker") as ctx:
                    ctx.result = step_tracker(state, self.log)

            # ── Étape 4 : Vidéo ──
            if self._should_run(state, "video"):
                with self._step_ctx(state, "video") as ctx:
                    ctx.result = step_video(state, self.log)

            # ── Étape 4b : Vérification labels caméra ──
            if self._should_run(state, "verify_labels"):
                with self._step_ctx(state, "verify_labels") as ctx:
                    ctx.result = step_verify_labels(
                        state, self.log,
                        min_confidence=self.params.get("label_min_confidence", 0.90),
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
                    state.mean_conf  = ctx.result.get("mean_conf", 0.0)
                    state.shifts     = ctx.result.get("shifts", {})

            # ── Étape 8 : Store vers /mnt/silver ──
            if self._should_run(state, "store"):
                if state.write_mode:
                    with self._step_ctx(state, "store") as ctx:
                        ctx.result = step_store(state, self.log)
                        state.silver_path = ctx.result.get("silver_path")
                else:
                    state.steps["store"].status  = StepStatus.SKIPPED
                    state.steps["store"].message = "write_mode désactivé — aucune écriture vers /mnt/silver"
                    self.log("Store ignoré (write_mode=False) — données disponibles dans /mnt/ingest", "INFO")

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
        s = state.steps.get(name)
        if s is None:
            return True
        if s.status in (StepStatus.DONE, StepStatus.SKIPPED):
            return False
        return True


class _StepContext:
    """Context manager pour une étape : chrono + statut + sauvegarde."""
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
    Surveille /mnt/ingest et lance la pipeline automatiquement
    sur les nouvelles sessions déposées par l'opérateur.
    Si write_mode=True, les sessions validées sont copiées vers /mnt/silver.
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
    ):
        self.watch_dir          = INGEST_DIR
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
        """Scanne /mnt/ingest pour de nouvelles sessions déposées."""
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

    def _process_queue(self):
        with self._lock:
            queue = list(self._queue)
            self._queue.clear()

        for sess in queue:
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

    def get_queue(self) -> List[str]:
        with self._lock:
            return [str(s) for s in self._queue]

    def get_seen(self) -> List[str]:
        return list(self._seen)

    def enqueue(self, session_path: str):
        """Enfile manuellement une session (chemin dans /mnt/ingest)."""
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
# CLI standalone
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Pipeline d'ingestion big data\n"
            f"  Travail   : {INGEST_DIR}  (déposez vos sessions ici)\n"
            f"  Sortie    : {SILVER_DIR}  (--write requis pour écrire)\n"
            "\nLe traitement est non-destructif par défaut (mode safe).\n"
            "Utilisez --delete-after-store pour supprimer de /mnt/ingest après store."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Sélection de sessions ──────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--session", metavar="NAME",
        help=f"Traiter une session spécifique dans {INGEST_DIR}",
    )
    grp.add_argument(
        "--all", action="store_true",
        help=f"Traiter toutes les sessions non encore traitées dans {INGEST_DIR}",
    )

    # ── Mode écriture ──────────────────────────────────────────────────────
    p.add_argument(
        "--write", action="store_true", default=False,
        help=f"Copier vers {SILVER_DIR} après validation (désactivé par défaut)",
    )
    p.add_argument(
        "--delete-after-store", action="store_true", default=False,
        help="Supprimer la session de /mnt/ingest après copie vers silver (requiert --write)",
    )

    # ── Paramètres IA ──────────────────────────────────────────────────────
    p.add_argument("--resample", type=float, default=5.0,    help="Rééchantillonnage (ms)")
    p.add_argument("--max-lag",  type=float, default=400.0,  help="Lag max recherché (ms)")
    p.add_argument("--window",   type=float, default=2200.0, help="Fenêtre d'analyse (ms)")

    # ── Options de contrôle ────────────────────────────────────────────────
    p.add_argument("--force-flux",           action="store_true",
                   help="Régénérer les flux CSV même s'ils existent déjà")
    p.add_argument("--no-resume",            action="store_true",
                   help="Ignorer l'état sauvegardé et recommencer depuis le début")
    p.add_argument("--label-min-confidence", type=float, default=0.90,
                   help="Confiance minimale pour la vérification des labels (défaut: 0.90)")

    args = p.parse_args()

    # ── Validation ────────────────────────────────────────────────────────
    errors = []
    if not INGEST_DIR.exists():
        errors.append(f"/mnt/ingest non accessible : {INGEST_DIR}")
    if args.write and not SILVER_DIR.exists():
        errors.append(f"/mnt/silver non accessible : {SILVER_DIR}  (requis par --write)")
    if args.delete_after_store and not args.write:
        errors.append("--delete-after-store requiert --write")
    if errors:
        for e in errors:
            print(f"ERREUR : {e}", file=sys.stderr)
        sys.exit(1)

    params = {
        "resample_ms":          args.resample,
        "max_lag_ms":           args.max_lag,
        "window_ms":            args.window,
        "label_min_confidence": args.label_min_confidence,
    }

    # ── Résolution des sessions à traiter (depuis /mnt/ingest) ────────────
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
        # Filtrer celles déjà traitées avec succès
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

    def _cli_log(msg, level="INFO"):
        icons = {"OK": "✓", "ERROR": "✗", "WARN": "⚠", "STEP": "▶", "TRAIN": "◉"}
        print(f"  {icons.get(level, '·')} {msg}")

    n_ok   = 0
    n_fail = 0
    state  = None

    for source_path in source_paths:
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
        print(f"\n  {icon} {'SUCCÈS' if state.success else 'ÉCHEC'}  "
              f"conf={state.mean_conf:.3f}  fiables={state.n_reliable}{silver_info}")
        for name, step in state.steps.items():
            s_icon = "✓" if step.status == StepStatus.DONE   else \
                     "✗" if step.status == StepStatus.FAILED  else \
                     "○" if step.status == StepStatus.SKIPPED else "·"
            print(f"    {s_icon} {name:<15} {step.duration_s:5.1f}s  {step.message[:55]}")

        if state.success:
            n_ok += 1
        else:
            n_fail += 1

    if len(source_paths) > 1:
        print(f"\n{'═'*60}")
        print(f"  Bilan : {n_ok} succès, {n_fail} échec(s) sur {len(source_paths)} sessions")

    sys.exit(0 if (state and state.success) else 1)
