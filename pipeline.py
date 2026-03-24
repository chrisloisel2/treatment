#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline d'ingestion big data — 7 étapes.

Étapes :
  1. DETECT     — Détection de nouvelles sessions dans le dossier d'ingestion
  2. TRACKER    — Validation & correction de position des trackers (sync_fix.py)
  3. VIDEO      — Validation & positionnement des vidéos (sync_fix.py + jsonl)
  4. FLUX_CSV   — Génération des flux optiques 1D (video.py, Farneback)
  5. IA_SYNC    — Synchronisation fine par deep learning (IA.py)
  6. VALIDATE   — Validation de cohérence + rollback si confiance insuffisante
  7. STORE      — Copie vers NAS / stockage distant

Chaque session traverse les étapes indépendamment.
L'état de chaque étape est persisté dans session_dir/pipeline_state.json.
Tout rollback restaure les .bak créés automatiquement.
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
    session_name: str
    session_path: str
    ingestion_root: str
    created_at:  str = field(default_factory=lambda: _now())
    updated_at:  str = field(default_factory=lambda: _now())
    current_step: str = "detect"
    finished:    bool = False
    success:     bool = False
    error:       Optional[str] = None
    steps:       Dict[str, StepState] = field(default_factory=dict)
    # Résultats pour accès rapide
    n_reliable:  int   = 0
    mean_conf:   float = 0.0
    shifts:      dict  = field(default_factory=dict)
    nas_path:    Optional[str] = None

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


def step_store(state: SessionPipelineState, log: PipelineLogger,
               nas_root: str) -> dict:
    """
    Étape 7 — Copie vers NAS / stockage.
    Copie la session complète (sauf vidéos brutes si demandé).
    """
    sess     = Path(state.session_path)
    nas_path = Path(nas_root) / sess.name

    if not Path(nas_root).exists():
        raise FileNotFoundError(
            f"NAS non accessible: {nas_root}\n"
            "Montez le partage réseau avant de lancer le stockage."
        )

    log(f"Copie vers {nas_path}…", "INFO")
    nas_path.mkdir(parents=True, exist_ok=True)

    copied = []
    skipped = []
    total_bytes = 0

    for src in sess.rglob("*"):
        if not src.is_file():
            continue
        # Exclure les .bak et locks
        if src.suffix in (".bak_syncml",) or src.name == PIPELINE_LOCK_FILE:
            continue
        # Exclure les backups temporaires
        if ".backup_" in str(src):
            continue

        rel = src.relative_to(sess)
        dst = nas_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Ne pas réécrire si identique (même taille + mtime)
        if dst.exists():
            src_stat = src.stat()
            dst_stat = dst.stat()
            if src_stat.st_size == dst_stat.st_size and abs(src_stat.st_mtime - dst_stat.st_mtime) < 1.0:
                skipped.append(str(rel))
                continue

        shutil.copy2(src, dst)
        copied.append(str(rel))
        total_bytes += src.stat().st_size

    # Nettoyer backups temporaires locaux
    for bk in sess.parent.glob(f".backup_{sess.name}_*"):
        shutil.rmtree(bk, ignore_errors=True)

    log(
        f"Stockage OK — {len(copied)} fichiers copiés "
        f"({total_bytes/1024/1024:.1f} Mo) vers {nas_path}",
        "OK",
    )
    return {
        "nas_path":    str(nas_path),
        "copied":      len(copied),
        "skipped":     len(skipped),
        "total_mb":    round(total_bytes / 1024 / 1024, 2),
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
        session_path:   str,
        ingestion_root: str,
        model_dir:      str,
        nas_root:       str,
        params:         dict,
        log_callback:   Optional[Callable] = None,
        step_callback:  Optional[Callable] = None,  # fn(state) appelée après chaque étape
        force_flux:     bool = False,
        resume:         bool = True,
    ):
        self.session_path   = Path(session_path)
        self.ingestion_root = ingestion_root
        self.model_dir      = Path(model_dir)
        self.nas_root       = nas_root
        self.params         = params
        self.log_callback   = log_callback
        self.step_callback  = step_callback
        self.force_flux     = force_flux
        self.resume         = resume

        self.log = PipelineLogger(self.session_path.name, log_callback)

    def _step_ctx(self, state: SessionPipelineState, name: str):
        """Retourne un context manager qui enregistre durée + statut."""
        return _StepContext(state, name, self.log, self.step_callback)

    def run(self) -> SessionPipelineState:
        sess = self.session_path

        # Charger ou créer l'état
        state = SessionPipelineState.load(sess) if self.resume else None
        if state is None:
            state = SessionPipelineState(
                session_name   = sess.name,
                session_path   = str(sess),
                ingestion_root = self.ingestion_root,
            )
            state.save()

        if state.finished and state.success and not self.force_flux:
            self.log("Session déjà traitée avec succès — skip", "INFO")
            return state

        # Verrou fichier pour éviter double-traitement
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

            # ── Étape 4 : Flux CSV ──
            if self._should_run(state, "flux_csv"):
                with self._step_ctx(state, "flux_csv") as ctx:
                    ctx.result = step_flux_csv(state, self.log, force=self.force_flux)

            # ── Étape 5 : IA Sync ──
            if self._should_run(state, "ia_sync"):
                with self._step_ctx(state, "ia_sync") as ctx:
                    ctx.result = step_ia_sync(state, self.log, self.model_dir, self.params)
                    ia_result  = ctx.result

            # Récupérer ia_result si reprise depuis validate
            if not ia_result and state.steps.get("ia_sync"):
                ia_result = state.steps["ia_sync"].detail

            # ── Étape 6 : Validation ──
            if self._should_run(state, "validate"):
                with self._step_ctx(state, "validate") as ctx:
                    ctx.result = step_validate(state, self.log, ia_result)
                    state.n_reliable = ctx.result.get("n_reliable", 0)
                    state.mean_conf  = ctx.result.get("mean_conf", 0.0)
                    state.shifts     = ctx.result.get("shifts", {})

            # ── Étape 7 : Stockage ──
            if self.nas_root and self._should_run(state, "store"):
                with self._step_ctx(state, "store") as ctx:
                    ctx.result = step_store(state, self.log, self.nas_root)
                    state.nas_path = ctx.result.get("nas_path")
            elif not self.nas_root:
                state.steps["store"].status  = StepStatus.SKIPPED
                state.steps["store"].message = "NAS non configuré"

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
    Surveille un répertoire d'ingestion et lance la pipeline
    automatiquement sur les nouvelles sessions.
    """

    def __init__(
        self,
        watch_dir:      str,
        model_dir:      str,
        nas_root:       str,
        params:         dict,
        log_callback:   Optional[Callable] = None,
        step_callback:  Optional[Callable] = None,
        poll_interval:  float = 10.0,
        auto_start:     bool  = False,
    ):
        self.watch_dir     = Path(watch_dir)
        self.model_dir     = model_dir
        self.nas_root      = nas_root
        self.params        = params
        self.log_callback  = log_callback
        self.step_callback = step_callback
        self.poll_interval = poll_interval
        self.auto_start    = auto_start

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
        if not self.watch_dir.exists():
            return
        for child in sorted(self.watch_dir.iterdir()):
            if not child.is_dir():
                continue
            if not child.name.startswith("session_"):
                continue
            if str(child) in self._seen:
                continue
            # Vérifier que c'est une session minimalement valide
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
            state = SessionPipelineState.load(sess)
            if state and state.finished and state.success:
                continue
            self._log(f"Lancement pipeline: {sess.name}", "INFO")
            runner = PipelineRunner(
                session_path   = str(sess),
                ingestion_root = str(self.watch_dir),
                model_dir      = self.model_dir,
                nas_root       = self.nas_root,
                params         = self.params,
                log_callback   = self.log_callback,
                step_callback  = self.step_callback,
            )
            runner.run()

    def get_queue(self) -> List[str]:
        with self._lock:
            return [str(s) for s in self._queue]

    def get_seen(self) -> List[str]:
        return list(self._seen)

    def enqueue(self, session_path: str):
        """Enfile manuellement une session."""
        with self._lock:
            self._queue.append(Path(session_path))

    def run_session_now(self, session_path: str) -> SessionPipelineState:
        """Lance la pipeline immédiatement sur une session (bloquant)."""
        runner = PipelineRunner(
            session_path   = session_path,
            ingestion_root = str(self.watch_dir),
            model_dir      = self.model_dir,
            nas_root       = self.nas_root,
            params         = self.params,
            log_callback   = self.log_callback,
            step_callback  = self.step_callback,
        )
        return runner.run()


# ══════════════════════════════════════════════════════════════════════════════
# CLI standalone
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Pipeline d'ingestion big data")
    p.add_argument("session",    help="Chemin vers la session à traiter")
    p.add_argument("--model-dir",default="_sync_ml_model", help="Répertoire du modèle IA")
    p.add_argument("--nas",      default="",    help="Répertoire NAS de destination")
    p.add_argument("--resample", type=float, default=5.0)
    p.add_argument("--max-lag",  type=float, default=400.0)
    p.add_argument("--window",   type=float, default=2200.0)
    p.add_argument("--force-flux", action="store_true")
    p.add_argument("--no-resume",  action="store_true")
    args = p.parse_args()

    params = {
        "resample_ms": args.resample,
        "max_lag_ms":  args.max_lag,
        "window_ms":   args.window,
    }

    def _cli_log(msg, level="INFO"):
        icons = {"OK": "✓", "ERROR": "✗", "WARN": "⚠", "STEP": "▶", "TRAIN": "◉"}
        print(f"  {icons.get(level,'·')} {msg}")

    runner = PipelineRunner(
        session_path   = args.session,
        ingestion_root = str(Path(args.session).parent),
        model_dir      = args.model_dir,
        nas_root       = args.nas,
        params         = params,
        log_callback   = _cli_log,
        force_flux     = args.force_flux,
        resume         = not args.no_resume,
    )
    state = runner.run()
    print(f"\n{'═'*50}")
    print(f"  Résultat: {'✓ SUCCÈS' if state.success else '✗ ÉCHEC'}")
    print(f"  Session:  {state.session_name}")
    print(f"  Conf moy: {state.mean_conf:.3f}")
    print(f"  Fiables:  {state.n_reliable}")
    for name, step in state.steps.items():
        icon = "✓" if step.status == StepStatus.DONE else "✗" if step.status == StepStatus.FAILED else "○"
        print(f"    {icon} {name:<12} {step.duration_s:.1f}s  {step.message[:60]}")
    sys.exit(0 if state.success else 1)
