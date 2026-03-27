#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_align_pro.py — Alignement temporel professionnel multi-résolution.

Stratégie en 4 passes :
  1. Gross search  : ±2000 ms, pas 20 ms  — localise la zone
  2. Fine search   : ±120 ms autour du pic, pas 1 ms — précision ms
  3. Sub-sample    : affinage parabolique  — précision < 1 ms
  4. Consensus     : vote pondéré inter-paires + rejet outliers

Signaux trackers enrichis (vitesse + accélération + jerk + énergie rotationnelle).
Détection de drift temporel par analyse glissante.
Validation post-correction via cross-corrélation avant écriture.
Backup atomique .bak_alignpro avant toute modification.

Usage :
    python3 pipeline_align_pro.py <session_path>
    python3 pipeline_align_pro.py <root_dir> --batch
    python3 pipeline_align_pro.py <root_dir> --batch --dry-run
    python3 pipeline_align_pro.py <session_path> --force
    python3 pipeline_align_pro.py <session_path> --validate-only

Dépendances :
    pip install pandas numpy scipy scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks
from scipy.stats import entropy as scipy_entropy

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn non disponible — pas de raffinement GBR.")

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Constantes
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.0.0"
MARKER_KEY = "align_pro_applied"

CAMERAS = ["head", "left", "right"]

PAIRS: List[Tuple[str, str]] = [
    ("tracker_head",  "cam_head"),
    ("tracker_left",  "cam_left"),
    ("tracker_right", "cam_right"),
    ("tracker_left",  "gripper_left"),
    ("tracker_right", "gripper_right"),
]

# Paires "majeures" — utilisées pour le consensus final
MAJOR_PAIRS = [("tracker_head", "cam_head"),
               ("tracker_left",  "cam_left"),
               ("tracker_right", "cam_right")]

# Paramètres de recherche multi-résolution
GROSS_MAX_LAG_MS   = 2000.0   # plage grossière ±ms
GROSS_STEP_MS      = 20.0     # pas grossier
FINE_HALF_MS       = 120.0    # demi-plage fine autour du pic grossier
FINE_STEP_MS       = 1.0      # pas fin

# Paramètres signal
RESAMPLE_MS        = 5.0      # rééchantillonnage grille commune
SMOOTH_SIGMA_COARSE = 60.0    # sigma gaussien pour enveloppe (ms, converti en samples)
SMOOTH_SIGMA_FINE   = 20.0
MIN_OVERLAP_MS     = 800.0    # chevauchement temporel minimum requis

# Métriques
N_BINS_MI          = 24       # bins pour info mutuelle
PEAK_TOL_MS        = 40.0     # tolérance pour peak alignment

METRIC_WEIGHTS = {
    "pearson":    0.28,
    "cosine":     0.20,
    "mutual_inf": 0.18,
    "peak_align": 0.14,
    "spectral":   0.12,
    "dtw_proxy":  0.08,   # nouveau : proxy DTW via corrélation normalisée glissante
}

# Consensus
MIN_CONFIDENCE_APPLY  = 0.30   # seuil pour qu'une paire vote dans le consensus
OUTLIER_SIGMA         = 2.0    # seuil rejet outlier (en écarts-type pondérés)
MIN_PAIRS_CONSENSUS   = 1      # nb minimum de paires valides pour appliquer
MAX_DELTA_START_MS    = 5000.0 # Δstart > N ms → paire ignorée (référentiels incompatibles)

# Validation post-correction
VALIDATION_MAX_LAG_MS = 2000.0  # fenêtre de validation (doit couvrir tout le domaine)
VALIDATION_MIN_CORR   = 0.35  # corrélation minimale acceptable

# Drift
DRIFT_WINDOW_MS    = 3000.0   # fenêtre pour analyse de drift
DRIFT_STRIDE_MS    = 1000.0
DRIFT_THRESHOLD_MS = 15.0     # drift > N ms → signal dans rapport


# ══════════════════════════════════════════════════════════════════════════════
# Structures de données
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    name: str
    t_ms_rel: np.ndarray       # timestamps relatifs (ms depuis debut)
    signal: np.ndarray         # signal normalisé
    t_start_abs_ms: float      # timestamp absolu du 1er échantillon (ms)
    source: str = "unknown"
    activity: float = 0.0      # std du signal (indicateur de richesse)


@dataclass
class PairResult:
    ref_name: str
    tgt_name: str
    delta_start_ms: float = 0.0
    gross_lag_ms: float = 0.0
    fine_lag_ms: float = 0.0
    subpixel_lag_ms: float = 0.0
    total_offset_ms: float = 0.0   # Δstart + résidu
    offset_rec_ms: float = 0.0     # à appliquer (−total)
    confidence: float = 0.0
    gross_score: float = 0.0
    fine_score: float = 0.0
    method: str = "multiresolution"
    has_drift: bool = False
    drift_range_ms: float = 0.0
    n_windows_valid: int = 0
    # arrays pour plot
    gross_candidates: np.ndarray = field(default_factory=lambda: np.array([]))
    gross_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    fine_candidates: np.ndarray = field(default_factory=lambda: np.array([]))
    fine_scores: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ConsensusResult:
    weighted_offset_ms: float
    std_ms: float
    n_votes: int
    pairs_used: List[str]
    pairs_rejected: List[str]
    confidence: float


@dataclass
class SessionReport:
    session: str
    status: str          # "ok" | "corrected" | "needs_review" | "skipped" | "error"
    consensus: Optional[ConsensusResult]
    pairs: List[PairResult]
    pre_score: Optional[float]
    post_score: Optional[float]
    offset_applied_ms: float
    has_drift: bool
    message: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Chargement des signaux
# ══════════════════════════════════════════════════════════════════════════════

def _robust_zscore(x: np.ndarray) -> np.ndarray:
    """Z-score robuste basé sur médiane + MAD."""
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-10:
        return np.zeros_like(x)
    return np.clip((x - med) / scale, -5.0, 5.0)


def _soft_clip(x: np.ndarray, q: float = 98.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lim = max(float(np.percentile(np.abs(x[finite]), q)), 1e-8)
    return np.clip(x, -lim, lim)


def _smooth(x: np.ndarray, sigma_ms: float, resample_ms: float = RESAMPLE_MS) -> np.ndarray:
    sigma_samples = sigma_ms / resample_ms
    if sigma_samples < 0.5:
        return x
    return gaussian_filter1d(x.astype(np.float64), sigma=sigma_samples)


def _quat_angular_velocity(qw, qx, qy, qz, dt_s: np.ndarray) -> np.ndarray:
    """Vitesse angulaire approchée depuis quaternions successifs (rad/s).
    dt_s a taille N-1 (diff des timestamps), quat ont taille N.
    """
    # dot produit entre q[i] et q[i+1] → taille N-1
    dot = qw[:-1]*qw[1:] + qx[:-1]*qx[1:] + qy[:-1]*qy[1:] + qz[:-1]*qz[1:]
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2.0 * np.arccos(np.abs(dot))  # taille N-1
    dt = np.where(dt_s > 1e-6, dt_s, 1e-3)  # dt_s est déjà N-1
    ang_vel = angle / dt                      # taille N-1
    return np.concatenate([[0.0], ang_vel])   # taille N


def _build_tracker_signal_rich(df: pd.DataFrame, pos: str,
                                t_ms: np.ndarray) -> np.ndarray:
    """
    Signal tracker enrichi : vitesse + accélération + jerk + énergie rotationnelle.
    Beaucoup plus discriminant que la vitesse seule.
    """
    xyz_cols = [f"tracker_{pos}_x", f"tracker_{pos}_y", f"tracker_{pos}_z"]
    quat_cols = [f"tracker_{pos}_qw", f"tracker_{pos}_qx",
                 f"tracker_{pos}_qy", f"tracker_{pos}_qz"]

    has_xyz  = all(c in df.columns for c in xyz_cols)
    has_quat = all(c in df.columns for c in quat_cols)

    if not has_xyz:
        return np.zeros(len(t_ms))

    x = pd.to_numeric(df[xyz_cols[0]], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(df[xyz_cols[1]], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(df[xyz_cols[2]], errors="coerce").to_numpy(np.float64)

    dt_s = np.diff(t_ms) / 1000.0
    valid_dt = dt_s[dt_s > 1e-6]
    median_dt_s = float(np.median(valid_dt)) if len(valid_dt) else 1e-3
    dt_s = np.where(dt_s > 1e-6, dt_s, median_dt_s)

    dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    speed = np.concatenate([[0.0], dist / dt_s])

    # Accélération et jerk — utilise le pas médian en ms pour le lissage
    median_dt_ms = median_dt_s * 1000.0
    speed_sm = _smooth(speed, 15.0, max(median_dt_ms, 0.5))
    accel = np.abs(np.gradient(speed_sm, median_dt_s))
    jerk  = np.abs(np.gradient(accel,   median_dt_s))

    # Énergie rotationnelle (si quaternions disponibles)
    ang_vel = np.zeros(len(t_ms))
    if has_quat:
        qw = pd.to_numeric(df[quat_cols[0]], errors="coerce").to_numpy(np.float64)
        qx = pd.to_numeric(df[quat_cols[1]], errors="coerce").to_numpy(np.float64)
        qy = pd.to_numeric(df[quat_cols[2]], errors="coerce").to_numpy(np.float64)
        qz = pd.to_numeric(df[quat_cols[3]], errors="coerce").to_numpy(np.float64)
        ang_vel = _quat_angular_velocity(qw, qx, qy, qz, dt_s)

    # Composition pondérée des composantes
    sig  = 0.40 * _robust_zscore(_soft_clip(speed))
    sig += 0.25 * _robust_zscore(_soft_clip(accel))
    sig += 0.15 * _robust_zscore(_soft_clip(jerk))
    sig += 0.20 * _robust_zscore(_soft_clip(ang_vel))

    sig = _smooth(sig, 10.0)
    return sig


def _build_video_signal_rich(df: pd.DataFrame) -> np.ndarray:
    """
    Signal vidéo enrichi : flux optique p90 + mean + diff + dérivée.
    Pondéré par disponibilité des colonnes.
    """
    # Priorité décroissante des colonnes de mouvement
    motion_cols_priority = [
        "motion_p90_smooth", "motion_p90",
        "motion_mean_smooth", "motion_mean",
        "diff_p90_smooth", "diff_p90",
        "diff_mean_smooth", "diff_mean",
    ]

    parts = []
    weights_used = []
    col_weights = {
        "motion_p90_smooth": 0.30, "motion_p90": 0.30,
        "motion_mean_smooth": 0.25, "motion_mean": 0.25,
        "diff_p90_smooth": 0.25, "diff_p90": 0.25,
        "diff_mean_smooth": 0.20, "diff_mean": 0.20,
    }
    seen_base = set()
    for col in motion_cols_priority:
        base = col.replace("_smooth", "")
        if base in seen_base:
            continue
        if col not in df.columns:
            alt = col.replace("_smooth", "")
            if alt in df.columns:
                col = alt
            else:
                continue
        seen_base.add(base)
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
        v = np.nan_to_num(v, nan=0.0)
        parts.append(_robust_zscore(_soft_clip(v)))
        weights_used.append(col_weights.get(col, 0.2))

    if not parts:
        return np.zeros(len(df))

    # Combinaison pondérée
    w = np.array(weights_used)
    w = w / w.sum()
    sig = sum(w[i] * parts[i] for i in range(len(parts)))
    sig = _smooth(sig, 8.0)

    # Dérivée (indicateur de transition)
    dsig = np.abs(np.gradient(sig))
    energy = _smooth(sig**2, 5.0)

    out  = 0.55 * _robust_zscore(sig)
    out += 0.25 * _robust_zscore(_soft_clip(dsig))
    out += 0.20 * _robust_zscore(_soft_clip(energy))
    out  = _smooth(out, 6.0)
    return out


def _jsonl_first_ts(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    return float(json.loads(line)["capture_time"])
                except (KeyError, ValueError, json.JSONDecodeError):
                    pass
    return None


def _load_jsonl_capture_unit(session_path: Path) -> str:
    meta = session_path / "metadata.json"
    if meta.exists():
        try:
            d = json.loads(meta.read_text())
            return d.get("jsonl_capture_time_unit", "milliseconds")
        except Exception:
            pass
    return "milliseconds"


def load_trackers(session_path: Path) -> Dict[str, Signal]:
    csv_path = session_path / "tracker_positions.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)

    # Temps absolu
    t_ns = pd.to_numeric(df.get("timestamp_ns", pd.Series(dtype=float)), errors="coerce")
    if t_ns.notna().sum() >= 2:
        t0_ns = float(t_ns.dropna().iloc[0])
        t_start_abs_ms = t0_ns / 1e6
        t_ms_rel = ((t_ns - t0_ns) / 1e6).to_numpy(np.float64)
    else:
        t_s = pd.to_numeric(df.get("time_seconds", pd.Series(dtype=float)), errors="coerce")
        t_start_abs_ms = float(t_s.iloc[0]) * 1000.0
        t_ms_rel = ((t_s - t_s.iloc[0]) * 1000.0).to_numpy(np.float64)

    # Tri par temps
    order = np.argsort(t_ms_rel)
    df = df.iloc[order].reset_index(drop=True)
    t_arr = t_ms_rel[order]

    out: Dict[str, Signal] = {}
    for pos in CAMERAS:
        sig = _build_tracker_signal_rich(df, pos, t_arr)
        out[f"tracker_{pos}"] = Signal(
            name=f"tracker_{pos}",
            t_ms_rel=t_arr.copy(),
            signal=sig,
            t_start_abs_ms=t_start_abs_ms,
            source="tracker_csv",
            activity=float(np.std(sig)),
        )
    return out


def load_cameras(session_path: Path) -> Dict[str, Signal]:
    out: Dict[str, Signal] = {}
    capture_unit = _load_jsonl_capture_unit(session_path)

    for cam in CAMERAS:
        # Cherche le flux CSV en priorité (videos/ puis racine)
        flux_csv = next((p for p in [
            session_path / "videos" / f"{cam}_flux.csv",
            session_path / f"{cam}_flux.csv",
        ] if p.exists()), None)
        # JSONL : racine d'abord, puis videos/
        jsonl_path = next((p for p in [
            session_path / f"{cam}.jsonl",
            session_path / "videos" / f"{cam}.jsonl",
        ] if p.exists()), None)

        if flux_csv is None and jsonl_path is None:
            continue

        if flux_csv is not None:
            df = pd.read_csv(flux_csv)
            sig = _build_video_signal_rich(df)

            # Ancrage temporel absolu
            t_abs_col = pd.to_numeric(
                df.get("timestamp_abs_ms", pd.Series(dtype=float)), errors="coerce"
            ) if "timestamp_abs_ms" in df.columns else None

            if t_abs_col is not None and t_abs_col.notna().sum() >= 2:
                first_v = float(t_abs_col.dropna().iloc[0])
                t_ms_rel = (t_abs_col.ffill() - first_v).to_numpy(np.float64)
                t_start = first_v
                source = "flux_csv_abs"
            else:
                # Fallback : ancrage via première capture_time du JSONL
                t_start_j = _jsonl_first_ts(jsonl_path) if jsonl_path is not None else None
                if t_start_j is not None and capture_unit == "nanoseconds":
                    t_start_j = t_start_j / 1e6
                elif t_start_j is not None and capture_unit == "microseconds":
                    t_start_j = t_start_j / 1e3
                t_s = pd.to_numeric(df.get("time_seconds", pd.Series(dtype=float)),
                                    errors="coerce").to_numpy(np.float64)
                t_ms_rel = (t_s - t_s[0]) * 1000.0
                t_start = t_start_j if t_start_j is not None else 0.0
                source = "flux_csv_jsonl"

            out[f"cam_{cam}"] = Signal(
                name=f"cam_{cam}", t_ms_rel=t_ms_rel,
                signal=sig, t_start_abs_ms=t_start,
                source=source, activity=float(np.std(sig)),
            )

        elif jsonl_path is not None:
            # Fallback : inter-frame intervals comme proxy de mouvement
            recs = []
            with open(jsonl_path, "r", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if len(recs) < 2:
                continue
            t_abs = np.array([float(r["capture_time"]) for r in recs], dtype=np.float64)

            # Conversion en ms
            if capture_unit == "nanoseconds":
                t_abs_ms = t_abs / 1e6
            elif capture_unit == "microseconds":
                t_abs_ms = t_abs / 1e3
            else:
                t_abs_ms = t_abs

            t_start = float(t_abs_ms[0])
            t_ms_rel = t_abs_ms - t_abs_ms[0]
            iff = np.diff(t_ms_rel)
            med_iff = float(np.median(iff[iff > 0])) if (iff > 0).any() else 33.333
            iff = np.where(iff > 0, iff, med_iff)
            sig = np.concatenate([[med_iff], iff])
            sig = _robust_zscore(sig)

            out[f"cam_{cam}"] = Signal(
                name=f"cam_{cam}", t_ms_rel=t_ms_rel,
                signal=sig, t_start_abs_ms=t_start,
                source="jsonl_only", activity=float(np.std(sig)),
            )

    return out


def load_grippers(session_path: Path) -> Dict[str, Signal]:
    out: Dict[str, Signal] = {}
    for side in ("left", "right"):
        csv_path = session_path / f"gripper_{side}_data.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        t_ns = pd.to_numeric(df.get("timestamp_ns", pd.Series(dtype=float)), errors="coerce")
        if t_ns.notna().sum() >= 2:
            t0 = float(t_ns.dropna().iloc[0])
            t_start = t0 / 1e6
            t_ms_rel = ((t_ns - t0) / 1e6).to_numpy(np.float64)
        else:
            t_s = pd.to_numeric(df.get("time_seconds", pd.Series(dtype=float)), errors="coerce")
            t_start = float(t_s.iloc[0]) * 1000.0
            t_ms_rel = ((t_s - t_s.iloc[0]) * 1000.0).to_numpy(np.float64)

        # Signal gripper — ouverture + vitesse d'ouverture
        if "opening_mm" in df.columns:
            op = pd.to_numeric(df["opening_mm"], errors="coerce").to_numpy(np.float64)
            d_op = np.abs(np.concatenate([[0.0], np.diff(op)]))
            sig = 0.6 * _robust_zscore(_soft_clip(d_op)) + \
                  0.4 * _robust_zscore(_soft_clip(op))
        elif "angle_deg" in df.columns:
            ang = pd.to_numeric(df["angle_deg"], errors="coerce").to_numpy(np.float64)
            d_ang = np.abs(np.concatenate([[0.0], np.diff(ang)]))
            sig = _robust_zscore(_soft_clip(d_ang))
        else:
            sig = np.ones(len(t_ms_rel))

        out[f"gripper_{side}"] = Signal(
            name=f"gripper_{side}", t_ms_rel=t_ms_rel,
            signal=sig, t_start_abs_ms=t_start,
            source="gripper_csv", activity=float(np.std(sig)),
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Métriques d'alignement
# ══════════════════════════════════════════════════════════════════════════════

def _resamp(t_src: np.ndarray, sig: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    sig_clean = np.where(np.isfinite(sig), sig, 0.0)
    return np.interp(t_grid, t_src, sig_clean, left=0.0, right=0.0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8:
        return 0.0
    nb = N_BINS_MI
    a_min, a_max = a.min(), a.max()
    b_min, b_max = b.min(), b.max()
    if (a_max - a_min) < 1e-10 or (b_max - b_min) < 1e-10:
        return 0.0
    ai = np.clip(((a - a_min) / (a_max - a_min) * (nb - 1)).astype(int), 0, nb - 1)
    bi = np.clip(((b - b_min) / (b_max - b_min) * (nb - 1)).astype(int), 0, nb - 1)
    joint = np.zeros((nb, nb), dtype=np.float64)
    np.add.at(joint, (ai, bi), 1)
    joint /= joint.sum() + 1e-12
    pa, pb = joint.sum(1), joint.sum(0)
    ha  = float(scipy_entropy(pa + 1e-12))
    hb  = float(scipy_entropy(pb + 1e-12))
    hab = float(scipy_entropy(joint.flatten() + 1e-12))
    mi  = ha + hb - hab
    return float(np.clip(mi / (min(ha, hb) + 1e-12), 0.0, 1.0))


def _peak_align(a: np.ndarray, b: np.ndarray) -> float:
    tol = max(1, int(PEAK_TOL_MS / RESAMPLE_MS))
    if len(a) < 10:
        return 0.0
    pa, _ = find_peaks(a, height=np.percentile(a, 65), distance=max(1, tol // 2))
    pb, _ = find_peaks(b, height=np.percentile(b, 65), distance=max(1, tol // 2))
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    fwd = sum(1 for p in pa if np.any(np.abs(pb - p) <= tol)) / len(pa)
    bwd = sum(1 for p in pb if np.any(np.abs(pa - p) <= tol)) / len(pb)
    return float((fwd + bwd) / 2.0)  # F1-like bidirectionnel


def _spectral(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 16:
        return 0.0
    fa = np.abs(np.fft.rfft(a[:n]))
    fb = np.abs(np.fft.rfft(b[:n]))
    if np.std(fa) < 1e-12 or np.std(fb) < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(fa, fb)[0, 1], 0.0, 1.0))


def _dtw_proxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    Proxy DTW via corrélation glissante normalisée.
    Évite le coût O(n²) du DTW exact tout en capturant l'élasticité locale.
    """
    if len(a) < 16 or len(b) < 16:
        return 0.0
    n = min(len(a), len(b), 400)
    a_n = a[:n] / (np.std(a[:n]) + 1e-10)
    b_n = b[:n] / (np.std(b[:n]) + 1e-10)
    # Corrélation glissante sur fenêtres courtes
    win = max(8, n // 6)
    scores = []
    for i in range(0, n - win, win // 2):
        wa = a_n[i:i+win]
        wb = b_n[i:i+win]
        c = _pearson(wa, wb)
        scores.append(max(c, 0.0))
    return float(np.mean(scores)) if scores else 0.0


def score_overlap(a: np.ndarray, b: np.ndarray,
                  sigma_coarse_samples: float = 12.0) -> float:
    """Score composite sur un overlap (a, b) déjà sur grille commune."""
    env_a = gaussian_filter1d(a ** 2, sigma=max(sigma_coarse_samples, 0.5))
    env_b = gaussian_filter1d(b ** 2, sigma=max(sigma_coarse_samples, 0.5))

    m = {
        "pearson":    _pearson(a, b),
        "cosine":     _cosine(env_a, env_b),
        "mutual_inf": _mutual_info(a, b),
        "peak_align": _peak_align(a, b),
        "spectral":   _spectral(a, b),
        "dtw_proxy":  _dtw_proxy(a, b),
    }
    return float(sum(METRIC_WEIGHTS[k] * m[k] for k in METRIC_WEIGHTS))


# ══════════════════════════════════════════════════════════════════════════════
# Analyse de drift temporel
# ══════════════════════════════════════════════════════════════════════════════

def analyze_drift(ref: Signal, tgt: Signal, nominal_offset_ms: float) -> Tuple[bool, float, int]:
    """
    Détecte un drift (décalage variable dans le temps) en analysant le lag
    optimal par fenêtre glissante.

    Returns: (has_drift, drift_range_ms, n_windows_valid)
    """
    tgt_t = tgt.t_ms_rel + (tgt.t_start_abs_ms - ref.t_start_abs_ms) + nominal_offset_ms

    win_ms = DRIFT_WINDOW_MS
    stride_ms = DRIFT_STRIDE_MS
    max_lag_samples = int(80.0 / RESAMPLE_MS)

    # Grille commune
    t0 = max(float(ref.t_ms_rel[0]), float(tgt_t[0]))
    t1 = min(float(ref.t_ms_rel[-1]), float(tgt_t[-1]))
    if t1 - t0 < win_ms + 100.0:
        return False, 0.0, 0

    grid = np.arange(t0, t1, RESAMPLE_MS)
    a_full = _resamp(ref.t_ms_rel, ref.signal, grid)
    b_full = _resamp(tgt_t, tgt.signal, grid)

    win_s = int(win_ms / RESAMPLE_MS)
    stride_s = int(stride_ms / RESAMPLE_MS)

    local_lags: List[float] = []
    pos = 0
    while pos + win_s <= len(grid):
        wa = a_full[pos:pos+win_s]
        wb = b_full[pos:pos+win_s]
        if np.std(wa) > 0.1 and np.std(wb) > 0.1:
            corr = correlate(wa, wb, mode="full", method="fft")
            lags = np.arange(-(len(wb)-1), len(wa))
            keep = np.abs(lags) <= max_lag_samples
            best = int(np.argmax(corr[keep]))
            local_lags.append(float(lags[keep][best]) * RESAMPLE_MS)
        pos += stride_s

    if len(local_lags) < 3:
        return False, 0.0, len(local_lags)

    lags_arr = np.array(local_lags)
    drift_range = float(np.max(lags_arr) - np.min(lags_arr))
    has_drift = drift_range > DRIFT_THRESHOLD_MS
    return has_drift, drift_range, len(local_lags)


# ══════════════════════════════════════════════════════════════════════════════
# Recherche multi-résolution
# ══════════════════════════════════════════════════════════════════════════════

def _sweep(ref: Signal, tgt: Signal, delta_start_ms: float,
           candidates: np.ndarray,
           sigma_coarse: float) -> np.ndarray:
    """
    Balayage vectorisé : pour chaque candidat, calcule le score d'overlap.
    Retourne le vecteur de scores.
    """
    scores = np.zeros(len(candidates))
    sigma_samples = sigma_coarse / RESAMPLE_MS

    for i, cand in enumerate(candidates):
        tgt_t = tgt.t_ms_rel + delta_start_ms + cand
        t0 = max(float(ref.t_ms_rel[0]), float(tgt_t[0]))
        t1 = min(float(ref.t_ms_rel[-1]), float(tgt_t[-1]))
        if t1 - t0 < MIN_OVERLAP_MS:
            continue
        grid = np.arange(t0, t1, RESAMPLE_MS)
        if len(grid) < 16:
            continue
        a = _resamp(ref.t_ms_rel, ref.signal, grid)
        b = _resamp(tgt_t,        tgt.signal, grid)
        scores[i] = score_overlap(a, b, sigma_coarse_samples=sigma_samples)

    return scores


def _parabolic_refine(candidates: np.ndarray, scores: np.ndarray,
                      best_idx: int) -> float:
    """Affinage sub-sample parabolique autour du pic."""
    if 0 < best_idx < len(scores) - 1:
        y0, y1, y2 = scores[best_idx-1], scores[best_idx], scores[best_idx+1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        step = float(candidates[1] - candidates[0]) if len(candidates) > 1 else 1.0
        if abs(denom) > 1e-12:
            offset = (y0 - y2) / denom * step
            # Ne pas dépasser un demi-pas
            return float(candidates[best_idx]) + float(np.clip(offset, -step, step))
    return float(candidates[best_idx])


def _gbr_refine(ref: Signal, tgt: Signal, delta_start_ms: float,
                candidates: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Raffinement GradientBoostingRegressor auto-supervisé.
    Prédit le décalage optimal comme régression sur les 6 métriques.
    Ne s'active que si sklearn est disponible et données suffisantes.
    """
    if not SKLEARN_AVAILABLE:
        return float(candidates[np.argmax(scores)]), float(np.max(scores))

    # Features par candidat
    X_all, valid_idx, best_scores = [], [], []
    sigma_samples = SMOOTH_SIGMA_FINE / RESAMPLE_MS

    for i, cand in enumerate(candidates):
        tgt_t = tgt.t_ms_rel + delta_start_ms + cand
        t0 = max(float(ref.t_ms_rel[0]), float(tgt_t[0]))
        t1 = min(float(ref.t_ms_rel[-1]), float(tgt_t[-1]))
        if t1 - t0 < MIN_OVERLAP_MS:
            continue
        grid = np.arange(t0, t1, RESAMPLE_MS)
        if len(grid) < 16:
            continue
        a = _resamp(ref.t_ms_rel, ref.signal, grid)
        b = _resamp(tgt_t,        tgt.signal, grid)
        env_a = gaussian_filter1d(a**2, sigma=max(sigma_samples, 0.5))
        env_b = gaussian_filter1d(b**2, sigma=max(sigma_samples, 0.5))

        feat = [
            _pearson(a, b),
            _cosine(env_a, env_b),
            _mutual_info(a, b),
            _peak_align(a, b),
            _spectral(a, b),
            _dtw_proxy(a, b),
            abs(float(cand)) / (float(np.max(np.abs(candidates))) + 1e-6),
            len(grid) * RESAMPLE_MS / 1000.0,
        ]
        X_all.append(feat)
        valid_idx.append(i)
        best_scores.append(scores[i])

    if len(X_all) < 10:
        return float(candidates[np.argmax(scores)]), float(np.max(scores))

    X = np.array(X_all, dtype=np.float64)
    y = np.array(best_scores, dtype=np.float64)

    # Pseudo-labels : top 10% → bon, bottom 30% → mauvais
    q_good = np.percentile(y, 90)
    q_bad  = np.percentile(y, 30)
    mask_good = y >= q_good
    mask_bad  = y <= q_bad

    if mask_good.sum() < 3 or mask_bad.sum() < 3:
        return float(candidates[np.argmax(scores)]), float(np.max(scores))

    # Entraîner un GBR sur les offsets candidats (régression directe)
    try:
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        c_vals = np.array([float(candidates[vi]) for vi in valid_idx])

        gbr = GradientBoostingRegressor(
            n_estimators=80, max_depth=4, learning_rate=0.1,
            min_samples_leaf=2, random_state=42,
        )
        gbr.fit(Xs, c_vals)
        pred = gbr.predict(Xs)

        # Meilleur score original dans ±20 ms du GBR prédit
        gbr_pred = float(np.median(pred[mask_good]))
        dists = np.abs(c_vals - gbr_pred)
        local_mask = dists < 20.0
        if local_mask.sum() == 0:
            local_mask = dists < 40.0
        if local_mask.sum() > 0:
            local_scores = y * (1.0 / (dists + 1.0))
            best_local = int(np.argmax(local_scores * local_mask.astype(float)))
            return float(c_vals[best_local]), float(y[best_local])
    except Exception:
        pass

    return float(candidates[np.argmax(scores)]), float(np.max(scores))


def analyze_pair(ref: Signal, tgt: Signal) -> PairResult:
    """
    Analyse complète d'une paire (ref, tgt) en 3 passes :
    1. Gross search ± GROSS_MAX_LAG_MS
    2. Fine search autour du pic (± FINE_HALF_MS)
    3. Affinage parabolique + GBR optionnel
    """
    res = PairResult(ref_name=ref.name, tgt_name=tgt.name)
    res.delta_start_ms = tgt.t_start_abs_ms - ref.t_start_abs_ms

    # Pour les paires gripper : Δstart > MAX_DELTA_START_MS = horloge incompatible → skip
    # Pour les paires caméra : le Δstart peut être grand (caméra démarre avant tracker) → OK
    is_gripper_pair = "gripper" in tgt.name
    if is_gripper_pair and abs(res.delta_start_ms) > MAX_DELTA_START_MS:
        print(f"    [SKIP Δstart={res.delta_start_ms:+.0f}ms > ±{MAX_DELTA_START_MS:.0f}ms]", end=" ")
        res.confidence = 0.0
        return res

    # ── PASSE 1 : Gross search ──────────────────────────────────────────────
    gross_cands = np.arange(-GROSS_MAX_LAG_MS, GROSS_MAX_LAG_MS + GROSS_STEP_MS, GROSS_STEP_MS)
    gross_scores = _sweep(ref, tgt, res.delta_start_ms, gross_cands, SMOOTH_SIGMA_COARSE)
    res.gross_candidates = gross_cands
    res.gross_scores = gross_scores

    if gross_scores.max() < 1e-6:
        res.confidence = 0.0
        return res

    best_gross_idx = int(np.argmax(gross_scores))
    best_gross_ms = float(gross_cands[best_gross_idx])
    res.gross_lag_ms = best_gross_ms
    res.gross_score = float(gross_scores[best_gross_idx])

    # ── PASSE 2 : Fine search ───────────────────────────────────────────────
    fine_min = best_gross_ms - FINE_HALF_MS
    fine_max = best_gross_ms + FINE_HALF_MS
    fine_cands = np.arange(fine_min, fine_max + FINE_STEP_MS, FINE_STEP_MS)
    fine_scores = _sweep(ref, tgt, res.delta_start_ms, fine_cands, SMOOTH_SIGMA_FINE)
    res.fine_candidates = fine_cands
    res.fine_scores = fine_scores

    best_fine_idx = int(np.argmax(fine_scores))
    best_fine_ms  = float(fine_cands[best_fine_idx])
    res.fine_lag_ms  = best_fine_ms
    res.fine_score   = float(fine_scores[best_fine_idx])

    # ── PASSE 3 : Affinage parabolique + GBR ────────────────────────────────
    subpixel_ms = _parabolic_refine(fine_cands, fine_scores, best_fine_idx)
    res.subpixel_lag_ms = subpixel_ms

    # GBR sur la plage fine
    if SKLEARN_AVAILABLE and res.fine_score > 0.25:
        gbr_ms, gbr_conf = _gbr_refine(ref, tgt, res.delta_start_ms, fine_cands, fine_scores)
        # N'accepte le GBR que s'il est cohérent avec le parabolique (± 15 ms)
        if abs(gbr_ms - subpixel_ms) < 15.0:
            final_lag_ms = (gbr_ms + subpixel_ms) / 2.0
            res.method = "multiresolution+gbr"
        else:
            final_lag_ms = subpixel_ms
    else:
        final_lag_ms = subpixel_ms

    # ── Drift analysis ──────────────────────────────────────────────────────
    has_drift, drift_range, n_wins = analyze_drift(ref, tgt, res.delta_start_ms + final_lag_ms)
    res.has_drift = has_drift
    res.drift_range_ms = drift_range
    res.n_windows_valid = n_wins

    res.total_offset_ms = res.delta_start_ms + final_lag_ms
    res.offset_rec_ms   = -res.total_offset_ms
    res.confidence = float(fine_scores[best_fine_idx])

    return res


# ══════════════════════════════════════════════════════════════════════════════
# Consensus inter-paires
# ══════════════════════════════════════════════════════════════════════════════

def compute_consensus(results: List[PairResult],
                      major_only: bool = False) -> Optional[ConsensusResult]:
    """
    Vote pondéré sur les offsets des paires majeures.
    Rejet des outliers par distance pondérée.
    """
    pairs_to_use = [r for r in results
                    if r.confidence >= MIN_CONFIDENCE_APPLY
                    and not (major_only and
                             (r.ref_name, r.tgt_name) not in MAJOR_PAIRS)]

    if not pairs_to_use:
        # Essai sans filtre major
        pairs_to_use = [r for r in results if r.confidence >= MIN_CONFIDENCE_APPLY * 0.6]

    if len(pairs_to_use) < MIN_PAIRS_CONSENSUS:
        return None

    # Vote sur le résidu seul (subpixel_lag_ms), pas sur le total_offset.
    # Le Δstart est déjà encodé dans les timestamps absolus — on ne corrige
    # que l'erreur résiduelle de synchronisation.
    residuals = np.array([r.subpixel_lag_ms for r in pairs_to_use])
    weights = np.array([r.confidence for r in pairs_to_use])
    weights = weights / weights.sum()

    # Moyenne pondérée initiale
    w_mean = float(np.dot(weights, residuals))
    w_std  = float(np.sqrt(np.dot(weights, (residuals - w_mean)**2)))

    # Rejet des outliers
    kept, rejected = [], []
    for r, res_val, w in zip(pairs_to_use, residuals, weights * weights.sum()):
        z = abs(res_val - w_mean) / (w_std + 1e-6)
        if z <= OUTLIER_SIGMA:
            kept.append(r)
        else:
            rejected.append(r.ref_name + "↔" + r.tgt_name)

    if not kept:
        kept = pairs_to_use  # garde tout si tout rejeté

    offsets_k = np.array([r.subpixel_lag_ms for r in kept])
    weights_k = np.array([r.confidence for r in kept])
    weights_k = weights_k / weights_k.sum()

    final_offset = float(np.dot(weights_k, offsets_k))
    final_std    = float(np.sqrt(np.dot(weights_k, (offsets_k - final_offset)**2)))
    final_conf   = float(np.mean([r.confidence for r in kept]))

    return ConsensusResult(
        weighted_offset_ms=final_offset,
        std_ms=final_std,
        n_votes=len(kept),
        pairs_used=[r.ref_name + "↔" + r.tgt_name for r in kept],
        pairs_rejected=rejected,
        confidence=final_conf,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Validation post-correction
# ══════════════════════════════════════════════════════════════════════════════

def validate_alignment(ref: Signal, tgt: Signal, offset_ms: float) -> float:
    """
    Mesure le lag résiduel après application de l'offset.
    Retourne un score [0, 100] — 100 = parfaitement aligné.
    """
    tgt_t_corrected = tgt.t_ms_rel + (tgt.t_start_abs_ms - ref.t_start_abs_ms) + offset_ms
    t0 = max(float(ref.t_ms_rel[0]), float(tgt_t_corrected[0]))
    t1 = min(float(ref.t_ms_rel[-1]), float(tgt_t_corrected[-1]))
    if t1 - t0 < 500.0:
        return 0.0

    grid = np.arange(t0, t1, RESAMPLE_MS)
    a = _resamp(ref.t_ms_rel, ref.signal, grid)
    b = _resamp(tgt_t_corrected, tgt.signal, grid)

    if np.std(a) < 0.05 or np.std(b) < 0.05:
        return 50.0  # signal trop plat : indéterminé

    max_lag_s = int(VALIDATION_MAX_LAG_MS * 2 / RESAMPLE_MS)
    corr = correlate(a, b, mode="full", method="fft")
    lags = np.arange(-(len(b)-1), len(a))
    denom = max(np.linalg.norm(a) * np.linalg.norm(b), 1e-10)
    corr_norm = corr / denom
    keep = np.abs(lags) <= max_lag_s
    best_idx = int(np.argmax(corr_norm[keep]))
    best_lag_ms = float(lags[keep][best_idx]) * RESAMPLE_MS
    best_corr   = float(corr_norm[keep][best_idx])

    lag_score  = max(0.0, 1.0 - abs(best_lag_ms) / (VALIDATION_MAX_LAG_MS * 2))
    corr_score = max(0.0, min(1.0, (best_corr - VALIDATION_MIN_CORR) /
                              (0.85 - VALIDATION_MIN_CORR)))
    return float(0.5 * lag_score + 0.5 * corr_score) * 100.0


def compute_session_score(signals_all: Dict[str, Signal],
                          offset_rec_ms: float) -> float:
    """Score moyen d'alignement sur les 3 paires majeures."""
    scores = []
    for ref_name, tgt_name in MAJOR_PAIRS:
        if ref_name in signals_all and tgt_name in signals_all:
            s = validate_alignment(signals_all[ref_name], signals_all[tgt_name], offset_rec_ms)
            scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Application des offsets en place
# ══════════════════════════════════════════════════════════════════════════════

def _shift_iso(series: pd.Series, delta_ns: int) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    shifted = ts + pd.to_timedelta(delta_ns, unit="ns")
    return shifted.dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _backup_session(session_path: Path) -> None:
    """Copie atomique des fichiers modifiables en .bak_alignpro."""
    files_to_backup = [
        session_path / "tracker_positions.csv",
        *(session_path / "videos" / f"{cam}.jsonl" for cam in CAMERAS),
        *(session_path / "videos" / f"{cam}_flux.csv" for cam in CAMERAS),
        *(session_path / f"{cam}_flux.csv" for cam in CAMERAS),
        *(session_path / f"gripper_{s}_data.csv" for s in ("left", "right")),
    ]
    for f in files_to_backup:
        if f.exists():
            bak = f.with_suffix(f.suffix + ".bak_alignpro")
            if not bak.exists():
                shutil.copy2(f, bak)


def _restore_backup(session_path: Path) -> None:
    """Restaure les fichiers depuis .bak_alignpro."""
    for bak in session_path.rglob("*.bak_alignpro"):
        original = bak.with_suffix("")  # retire .bak_alignpro
        shutil.copy2(bak, original)
    print(f"  [RESTORE] Backup restauré pour {session_path.name}")


def apply_offset(session_path: Path, offset_ms: float, dry_run: bool,
                 capture_unit: str = "milliseconds") -> None:
    """
    Applique offset_ms aux caméras JSONL et aux grippers uniquement.
    Le tracker est la référence absolue — il n'est JAMAIS modifié.

    Sémantique : subpixel_lag_ms = décalage residuel de la cible (cam/gripper)
    par rapport au tracker. Si subpixel_lag_ms = -24 ms, la caméra est 24 ms en
    avance. Pour corriger : on ajoute +24 ms aux timestamps caméra.
    cam_delta = -offset_ms (car offset_ms = subpixel_lag_ms < 0 → cam_delta > 0).
    """
    if abs(offset_ms) < 0.05:
        print(f"  [SKIP] offset {offset_ms:+.3f} ms < 0.05 ms — rien à faire")
        return

    # cam_delta : valeur à AJOUTER aux capture_time (unités caméra)
    # Sémantique de subpixel_lag_ms = cand optimal dans _sweep :
    #   tgt_t = tgt.t_ms_rel + Δstart + cand
    # Si cand = +X est optimal, la caméra a besoin d'être décalée de +X ms.
    # Donc on AJOUTE offset_ms aux timestamps caméra (cam_delta = +offset_ms).
    # Tracker est la référence → on ne le touche pas.
    cam_delta_ms = offset_ms  # en ms

    # ── Caméras JSONL ────────────────────────────────────────────────────────
    for cam in CAMERAS:
        for jsonl_path in [
            session_path / "videos" / f"{cam}.jsonl",
            session_path / f"{cam}.jsonl",
        ]:
            if not jsonl_path.exists():
                continue
            lines_out = []
            with open(jsonl_path, "r", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if capture_unit == "nanoseconds":
                            # capture_time est en ns → ajouter -offset_ms en ns
                            rec["capture_time"] = rec["capture_time"] + int(round(cam_delta_ms * 1_000_000))
                        elif capture_unit == "microseconds":
                            # capture_time est en µs → ajouter -offset_ms en µs
                            rec["capture_time"] = rec["capture_time"] + int(round(cam_delta_ms * 1_000.0))
                        else:  # milliseconds
                            rec["capture_time"] = rec["capture_time"] + cam_delta_ms
                        lines_out.append(json.dumps(rec))
                    except (json.JSONDecodeError, KeyError):
                        lines_out.append(line)
            if not dry_run:
                jsonl_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
            print(f"  [cam_{cam}] {cam_delta_ms:+.3f} ms ({capture_unit})")
            break  # premier trouvé suffit

    # ── Flux CSV (timestamp_abs_ms) ───────────────────────────────────────────
    # Les flux CSV contiennent timestamp_abs_ms dérivé des capture_time JSONL.
    # On les met à jour dans le même sens que les caméras.
    for cam in CAMERAS:
        for flux_path in [
            session_path / "videos" / f"{cam}_flux.csv",
            session_path / f"{cam}_flux.csv",
        ]:
            if not flux_path.exists():
                continue
            df = pd.read_csv(flux_path)
            if "timestamp_abs_ms" in df.columns:
                df["timestamp_abs_ms"] = (
                    pd.to_numeric(df["timestamp_abs_ms"], errors="coerce") + cam_delta_ms
                )
                if not dry_run:
                    df.to_csv(flux_path, index=False)
                print(f"  [flux_{cam}] {cam_delta_ms:+.3f} ms (timestamp_abs_ms)")
            break  # premier trouvé suffit

    # ── Grippers ─────────────────────────────────────────────────────────────
    # Même logique : grippers alignés sur tracker → décaler dans le même sens
    # que les caméras (cam_delta_ms = -offset_ms)
    grip_delta_ns = int(round(cam_delta_ms * 1_000_000))
    grip_delta_s  = cam_delta_ms / 1000.0
    for side in ("left", "right"):
        grip = session_path / f"gripper_{side}_data.csv"
        if not grip.exists():
            continue
        df = pd.read_csv(grip)
        for col in ("timestamp_ns", "t_ms_corrected_ns"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") + grip_delta_ns
        if "time_seconds" in df.columns:
            df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce") + grip_delta_s
        if "t_ms" in df.columns:
            df["t_ms"] = pd.to_numeric(df["t_ms"], errors="coerce") + cam_delta_ms
        if "timestamp" in df.columns:
            df["timestamp"] = _shift_iso(df["timestamp"], grip_delta_ns)
        if not dry_run:
            df.to_csv(grip, index=False)
        print(f"  [gripper_{side}] {cam_delta_ms:+.3f} ms")


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation diagnostique
# ══════════════════════════════════════════════════════════════════════════════

def plot_diagnostic(results: List[PairResult], consensus: Optional[ConsensusResult],
                    signals_all: Dict[str, Signal], session_path: Path,
                    pre_score: float, post_score: float) -> None:
    """
    Génère un PNG avec :
    - Courbes de score gross + fine par paire
    - Superposition des signaux avant/après correction
    - Synthèse du consensus
    """
    n_pairs = len(results)
    if n_pairs == 0:
        return

    fig = plt.figure(figsize=(18, 4 * n_pairs + 4), constrained_layout=True)
    gs = fig.add_gridspec(n_pairs + 1, 3)

    colors = ["#1565C0", "#C62828", "#2E7D32", "#6A1B9A", "#E65100"]

    for i, r in enumerate(results):
        # Courbe grossière
        ax_gross = fig.add_subplot(gs[i, 0])
        if len(r.gross_candidates):
            ax_gross.plot(r.gross_candidates, r.gross_scores,
                          lw=1.2, color=colors[i % len(colors)], alpha=0.7)
            ax_gross.axvline(r.gross_lag_ms, color="#C62828", lw=1.5,
                             linestyle="--", label=f"pic={r.gross_lag_ms:+.0f}ms")
        ax_gross.set_title(f"{r.ref_name}↔{r.tgt_name} — Gross", fontsize=8)
        ax_gross.set_xlabel("Décalage (ms)", fontsize=7)
        if ax_gross.get_legend_handles_labels()[0]:
            ax_gross.legend(fontsize=7)
        ax_gross.grid(True, alpha=0.2)

        # Courbe fine
        ax_fine = fig.add_subplot(gs[i, 1])
        if len(r.fine_candidates):
            ax_fine.plot(r.fine_candidates, r.fine_scores,
                         lw=1.5, color=colors[i % len(colors)])
            ax_fine.axvline(r.subpixel_lag_ms, color="#C62828", lw=2.0,
                            linestyle="--",
                            label=f"résidu={r.subpixel_lag_ms:+.2f}ms conf={r.confidence:.3f}")
            ax_fine.axvline(0.0, color="#aaa", lw=0.8, linestyle=":")
        ax_fine.set_title(f"{r.ref_name}↔{r.tgt_name} — Fine", fontsize=8)
        ax_fine.set_xlabel("Décalage résiduel (ms)", fontsize=7)
        if ax_fine.get_legend_handles_labels()[0]:
            ax_fine.legend(fontsize=7)
        ax_fine.grid(True, alpha=0.2)

        # Superposition des signaux après correction
        ax_sig = fig.add_subplot(gs[i, 2])
        ref_s = signals_all.get(r.ref_name)
        tgt_s = signals_all.get(r.tgt_name)
        if ref_s is not None and tgt_s is not None:
            t0 = max(float(ref_s.t_ms_rel[0]),
                     float(tgt_s.t_ms_rel[0]) + (tgt_s.t_start_abs_ms - ref_s.t_start_abs_ms)
                     + r.offset_rec_ms)
            t1 = min(float(ref_s.t_ms_rel[-1]),
                     float(tgt_s.t_ms_rel[-1]) + (tgt_s.t_start_abs_ms - ref_s.t_start_abs_ms)
                     + r.offset_rec_ms)
            if t1 - t0 > 200.0:
                grid = np.arange(t0, t1, RESAMPLE_MS)[:600]
                a = _resamp(ref_s.t_ms_rel, ref_s.signal, grid)
                tgt_t = tgt_s.t_ms_rel + (tgt_s.t_start_abs_ms - ref_s.t_start_abs_ms) + r.offset_rec_ms
                b = _resamp(tgt_t, tgt_s.signal, grid)
                t_plot = (grid - grid[0]) / 1000.0
                ax_sig.plot(t_plot, a, lw=0.9, color="#1565C0", alpha=0.85, label=r.ref_name)
                ax_sig.plot(t_plot, b, lw=0.9, color="#C62828", alpha=0.85, label=r.tgt_name)
                if r.has_drift:
                    ax_sig.set_facecolor("#fff8f0")
        ax_sig.set_title(f"Signaux superposés (après correction)", fontsize=8)
        ax_sig.set_xlabel("Temps (s)", fontsize=7)
        if ax_sig.get_legend_handles_labels()[0]:
            ax_sig.legend(fontsize=6, loc="upper right")
        ax_sig.grid(True, alpha=0.2)

    # Ligne de synthèse
    ax_sum = fig.add_subplot(gs[n_pairs, :])
    ax_sum.axis("off")
    summary_lines = [
        f"Session : {session_path.name}   |   Version : {VERSION}",
        f"Score pré-correction : {pre_score:.1f}/100   →   Score post-correction : {post_score:.1f}/100",
    ]
    if consensus:
        summary_lines.append(
            f"Consensus : {consensus.weighted_offset_ms:+.2f} ms ± {consensus.std_ms:.2f} ms "
            f"({consensus.n_votes} vote(s), conf={consensus.confidence:.3f})"
        )
        if consensus.pairs_rejected:
            summary_lines.append(f"Paires rejetées : {', '.join(consensus.pairs_rejected)}")
    for j, r in enumerate(results):
        drift_tag = f" [DRIFT ±{r.drift_range_ms:.0f}ms]" if r.has_drift else ""
        summary_lines.append(
            f"{r.ref_name}↔{r.tgt_name} : Δstart={r.delta_start_ms:+.1f}ms  "
            f"résidu={r.subpixel_lag_ms:+.2f}ms  total={r.total_offset_ms:+.2f}ms  "
            f"conf={r.confidence:.3f} [{r.method}]{drift_tag}"
        )

    ax_sum.text(0.01, 0.99, "\n".join(summary_lines),
                transform=ax_sum.transAxes, fontsize=7.5,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#f0f4ff", alpha=0.8))

    fig.suptitle(f"pipeline_align_pro — {session_path.name}", fontsize=11, fontweight="bold")
    out = session_path / "align_pro_diagnostic.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Traitement d'une session
# ══════════════════════════════════════════════════════════════════════════════

def align_session(session_path: Path,
                  dry_run: bool = False,
                  validate_only: bool = False,
                  force: bool = False,
                  make_plot: bool = True) -> SessionReport:

    session_path = Path(session_path)
    print(f"\n{'─'*70}")
    print(f"  Session : {session_path.name}")

    # ── Vérification prérequis ───────────────────────────────────────────────
    meta_path = session_path / "metadata.json"
    if not meta_path.exists():
        print(f"  [SKIP] metadata.json absent")
        return SessionReport(session_path.name, "skipped", None, [], None, None,
                             0.0, False, "metadata.json manquant")

    if (session_path / "tracker_positions.csv").exists() is False:
        print(f"  [SKIP] tracker_positions.csv absent")
        return SessionReport(session_path.name, "skipped", None, [], None, None,
                             0.0, False, "tracker_positions.csv manquant")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    capture_unit = metadata.get("jsonl_capture_time_unit", "milliseconds")

    if not force and metadata.get(MARKER_KEY):
        print(f"  [SKIP] déjà traitée le {metadata[MARKER_KEY]} (--force pour re-traiter)")
        return SessionReport(session_path.name, "skipped", None, [], None, None,
                             0.0, False, f"déjà traitée: {metadata[MARKER_KEY]}")

    # ── Chargement des signaux ───────────────────────────────────────────────
    signals_all: Dict[str, Signal] = {}
    signals_all.update(load_trackers(session_path))
    signals_all.update(load_cameras(session_path))
    signals_all.update(load_grippers(session_path))

    if not signals_all:
        print(f"  [SKIP] aucun signal chargé")
        return SessionReport(session_path.name, "skipped", None, [], None, None,
                             0.0, False, "aucun signal")

    # ── Pré-score ────────────────────────────────────────────────────────────
    pre_score = compute_session_score(signals_all, 0.0)
    print(f"  Score pré-correction : {pre_score:.1f}/100")

    if validate_only:
        print(f"  [VALIDATE-ONLY] pas d'écriture")
        return SessionReport(session_path.name, "ok", None, [], pre_score, None,
                             0.0, False, "validate_only")

    # ── Analyse par paire ────────────────────────────────────────────────────
    results: List[PairResult] = []
    for ref_name, tgt_name in PAIRS:
        ref = signals_all.get(ref_name)
        tgt = signals_all.get(tgt_name)
        if ref is None or tgt is None:
            continue
        if ref.activity < 0.05 or tgt.activity < 0.05:
            print(f"  [SKIP pair] {ref_name}↔{tgt_name} : activité trop faible")
            continue

        print(f"  Analyse paire : {ref_name} ↔ {tgt_name} ...", end=" ", flush=True)
        r = analyze_pair(ref, tgt)
        results.append(r)
        drift_tag = f" DRIFT±{r.drift_range_ms:.0f}ms" if r.has_drift else ""
        print(f"Δstart={r.delta_start_ms:+.1f}ms  résidu={r.subpixel_lag_ms:+.2f}ms  "
              f"total={r.total_offset_ms:+.2f}ms  conf={r.confidence:.3f}{drift_tag}")

    if not results:
        print(f"  [ERROR] aucune paire analysable")
        return SessionReport(session_path.name, "error", None, [], pre_score, None,
                             0.0, False, "aucune paire analysable")

    # ── Consensus ────────────────────────────────────────────────────────────
    consensus = compute_consensus(results, major_only=True)
    if consensus is None:
        consensus = compute_consensus(results, major_only=False)

    if consensus is None:
        print(f"  [ERROR] impossible de calculer le consensus (confidence insuffisante)")
        return SessionReport(session_path.name, "needs_review", None, results,
                             pre_score, None, 0.0,
                             any(r.has_drift for r in results),
                             "consensus impossible")

    print(f"\n  Consensus : {consensus.weighted_offset_ms:+.3f} ms "
          f"± {consensus.std_ms:.2f} ms  ({consensus.n_votes} vote(s), "
          f"conf={consensus.confidence:.3f})")
    if consensus.pairs_rejected:
        print(f"  Outliers rejetés : {', '.join(consensus.pairs_rejected)}")

    # ── Décision d'application ───────────────────────────────────────────────
    offset_to_apply = consensus.weighted_offset_ms
    has_drift = any(r.has_drift for r in results)

    if has_drift:
        print(f"  ⚠ Drift temporel détecté — offset moyen appliqué (correction partielle)")

    # Backup avant écriture
    if not dry_run:
        _backup_session(session_path)

    apply_offset(session_path, offset_to_apply, dry_run=dry_run,
                 capture_unit=capture_unit)

    # ── Post-score (recharge les signaux après correction) ───────────────────
    signals_post: Dict[str, Signal] = {}
    if not dry_run:
        signals_post.update(load_trackers(session_path))
        signals_post.update(load_cameras(session_path))
        post_score = compute_session_score(signals_post, 0.0)
    else:
        # En dry-run, on simule le décalage des caméras (cam_delta = +offset_to_apply)
        # validate_alignment ajoute offset_ms au timeline caméra → passer +offset_to_apply
        post_score = compute_session_score(signals_all, offset_to_apply)
    print(f"  Score post-correction : {post_score:.1f}/100")

    # Rollback si dégradation significative
    if not dry_run and post_score < pre_score - 5.0:
        print(f"  ⚠ Score dégradé ({pre_score:.1f} → {post_score:.1f}) — ROLLBACK")
        _restore_backup(session_path)
        status = "needs_review"
        msg = f"rollback: score {pre_score:.1f}→{post_score:.1f}"
    else:
        status = "corrected" if abs(offset_to_apply) > 2.0 else "ok"
        msg = f"offset={offset_to_apply:+.3f}ms"

    # ── Rapport JSON ──────────────────────────────────────────────────────────
    report_data = {
        "session": session_path.name,
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dry_run": dry_run,
        "pre_score": round(pre_score, 2),
        "post_score": round(post_score, 2),
        "consensus": asdict(consensus) if consensus else None,
        "has_drift": has_drift,
        "pairs": [
            {
                "ref": r.ref_name, "tgt": r.tgt_name,
                "delta_start_ms": round(r.delta_start_ms, 3),
                "gross_lag_ms": round(r.gross_lag_ms, 3),
                "fine_lag_ms": round(r.fine_lag_ms, 3),
                "subpixel_lag_ms": round(r.subpixel_lag_ms, 4),
                "total_offset_ms": round(r.total_offset_ms, 4),
                "offset_rec_ms": round(r.offset_rec_ms, 4),
                "confidence": round(r.confidence, 4),
                "method": r.method,
                "has_drift": r.has_drift,
                "drift_range_ms": round(r.drift_range_ms, 2),
                "n_windows_valid": r.n_windows_valid,
            }
            for r in results
        ],
    }
    if not dry_run:
        report_path = session_path / "align_pro_report.json"
        report_path.write_text(
            json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  [REPORT] {report_path.name}")

        # Marquer metadata.json
        metadata[MARKER_KEY] = datetime.now(timezone.utc).isoformat()
        metadata["align_pro_offset_ms"] = round(offset_to_apply, 4)
        metadata["align_pro_pre_score"] = round(pre_score, 2)
        metadata["align_pro_post_score"] = round(post_score, 2)
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Graphe diagnostique ───────────────────────────────────────────────────
    if make_plot and not dry_run:
        sig_for_plot = signals_post if signals_post else signals_all
        try:
            plot_diagnostic(results, consensus, sig_for_plot, session_path,
                            pre_score, post_score)
        except Exception as e:
            print(f"  [WARN] plot échoué: {e}")

    return SessionReport(
        session=session_path.name,
        status=status,
        consensus=consensus,
        pairs=results,
        pre_score=pre_score,
        post_score=post_score,
        offset_applied_ms=offset_to_apply if status != "needs_review" else 0.0,
        has_drift=has_drift,
        message=msg,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Batch
# ══════════════════════════════════════════════════════════════════════════════

def upload_session_to_s3(session_path: Path, bucket: str, prefix: str) -> None:
    """Upload les fichiers de sortie d'une session vers S3."""
    if not BOTO3_AVAILABLE:
        print("  [S3] boto3 non disponible — pip install boto3", file=sys.stderr)
        return

    s3 = boto3.client("s3")
    files_to_upload = [
        session_path / "tracker_positions.csv",
        session_path / "align_pro_report.json",
        session_path / "align_pro_diagnostic.png",
        session_path / "metadata.json",
        *(session_path / "videos" / f"{cam}.jsonl" for cam in CAMERAS),
        *(session_path / f"gripper_{s}_data.csv" for s in ("left", "right")),
    ]

    uploaded = 0
    for local_path in files_to_upload:
        if not local_path.exists():
            continue
        key = f"{prefix}/{session_path.name}/{local_path.name}".lstrip("/")
        try:
            s3.upload_file(str(local_path), bucket, key)
            print(f"  [S3] s3://{bucket}/{key}")
            uploaded += 1
        except (BotoCoreError, ClientError) as e:
            print(f"  [S3] ERREUR upload {local_path.name}: {e}", file=sys.stderr)

    print(f"  [S3] {uploaded} fichier(s) uploadé(s) → s3://{bucket}/{prefix}/{session_path.name}/")


def find_sessions(root: Path) -> List[Path]:
    """Trouve les sessions (dossiers avec tracker_positions.csv)."""
    sessions = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        if (item / "tracker_positions.csv").exists():
            sessions.append(item)
        else:
            for sub in sorted(item.iterdir()):
                if sub.is_dir() and (sub / "tracker_positions.csv").exists():
                    sessions.append(sub)
    return sessions


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"pipeline_align_pro v{VERSION} — alignement temporel professionnel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python3 pipeline_align_pro.py /chemin/session_20260322_133839
  python3 pipeline_align_pro.py ~/Desktop/sessions --batch
  python3 pipeline_align_pro.py ~/Desktop/sessions --batch --dry-run
  python3 pipeline_align_pro.py /chemin/session --validate-only
  python3 pipeline_align_pro.py /chemin/session --force
        """,
    )
    p.add_argument("path",           help="Chemin de session ou répertoire racine (avec --batch)")
    p.add_argument("--batch",        action="store_true", help="Traiter toutes les sessions")
    p.add_argument("--dry-run",      action="store_true", help="Calcule sans écrire")
    p.add_argument("--validate-only",action="store_true", help="Score sans correction")
    p.add_argument("--force",        action="store_true", help="Re-traite sessions déjà corrigées")
    p.add_argument("--no-plot",      action="store_true", help="Désactive les graphes PNG")
    p.add_argument("--s3-bucket",    default=None,        help="Bucket S3 de destination (ex: mon-bucket)")
    p.add_argument("--s3-prefix",    default="",          help="Préfixe clé S3 (ex: sessions/2026)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.path).expanduser().resolve()

    if not root.exists():
        print(f"ERREUR : chemin introuvable : {root}", file=sys.stderr)
        return 1

    print(f"\n{'═'*70}")
    print(f"  pipeline_align_pro v{VERSION}")
    if SKLEARN_AVAILABLE:
        print(f"  Mode : multi-résolution + GradientBoosting")
    else:
        print(f"  Mode : multi-résolution (scikit-learn non disponible)")
    print(f"  Dry-run      : {args.dry_run}")
    print(f"  Validate-only: {args.validate_only}")
    print(f"  Force        : {args.force}")
    print(f"{'═'*70}")

    make_plot = not args.no_plot

    if args.batch:
        sessions = find_sessions(root)
        if not sessions:
            print(f"Aucune session trouvée dans {root}", file=sys.stderr)
            return 1

        print(f"\n  {len(sessions)} session(s) trouvée(s)\n")
        reports: List[SessionReport] = []
        for s in sessions:
            r = align_session(s, dry_run=args.dry_run,
                              validate_only=args.validate_only,
                              force=args.force, make_plot=make_plot)
            reports.append(r)
            if args.s3_bucket and not args.dry_run:
                upload_session_to_s3(s, args.s3_bucket, args.s3_prefix)

        # Résumé global
        print(f"\n{'═'*70}")
        print(f"  RÉSUMÉ BATCH — {len(sessions)} session(s)")
        print(f"{'═'*70}")
        status_icon = {
            "ok": "✓", "corrected": "~", "needs_review": "!", "skipped": "—", "error": "✗"
        }
        for r in reports:
            icon = status_icon.get(r.status, "?")
            pre  = f"pré={r.pre_score:.0f}" if r.pre_score is not None else "pré=?"
            post = f"post={r.post_score:.0f}" if r.post_score is not None else ""
            off  = f"off={r.offset_applied_ms:+.1f}ms" if r.offset_applied_ms != 0 else ""
            drift = " [drift]" if r.has_drift else ""
            print(f"  {icon} {r.session:<45} {r.status:<12} {pre} {post} {off}{drift}")

        # Rapport global JSON
        if not args.dry_run and reports:
            global_report = {
                "version": VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_sessions": len(sessions),
                "sessions": [
                    {
                        "session": r.session,
                        "status": r.status,
                        "pre_score": r.pre_score,
                        "post_score": r.post_score,
                        "offset_ms": r.offset_applied_ms,
                        "has_drift": r.has_drift,
                        "message": r.message,
                    }
                    for r in reports
                ],
            }
            out = root / "align_pro_batch_report.json"
            out.write_text(json.dumps(global_report, indent=2, ensure_ascii=False))
            print(f"\n  Rapport global : {out}")
            if args.s3_bucket:
                key = f"{args.s3_prefix}/align_pro_batch_report.json".lstrip("/")
                try:
                    boto3.client("s3").upload_file(str(out), args.s3_bucket, key)
                    print(f"  [S3] s3://{args.s3_bucket}/{key}")
                except Exception as e:
                    print(f"  [S3] ERREUR rapport batch: {e}", file=sys.stderr)

    else:
        align_session(root, dry_run=args.dry_run,
                      validate_only=args.validate_only,
                      force=args.force, make_plot=make_plot)
        if args.s3_bucket and not args.dry_run:
            upload_session_to_s3(root, args.s3_bucket, args.s3_prefix)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
