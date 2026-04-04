#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sync.py — Moteur de synchronisation temporelle complet.

Regroupe :
  - sync_fix.py        : corrections heuristiques multi-métriques + Random Forest (in-place)
  - IA.py              : alignement deep learning (CrossModalAligner, entraînement + inférence)
  - apply_corrections.py : application sécurisée avec validation pré/post (copie vers corrected/)
  - score_sessions.py  : scoring déterministe toutes sessions (tableau ASCII + JSON)

Usage standalone :
    # Correction heuristique en place
    python sync.py heuristic /path/to/sessions [--dry-run] [--max-lag-ms 500]

    # Entraînement modèle IA
    python sync.py train [--epochs 5] [--batch-size 256]

    # Estimation IA + application
    python sync.py apply [--session session_xxx] [--dry-run]

    # Score de synchronisation
    python sync.py score --dataset ./dataset [--json-out results.json]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import entropy as scipy_entropy, linregress
from scipy.interpolate import interp1d as _interp1d

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn non disponible — scoring heuristique uniquement.")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import du moteur de notation (signals.py)
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
from signals import (
    AlignReport,
    build_tracker_signal,
    build_video_signal as _build_video_signal_notation,
    compute_alignment_report,
    filter_existing_cols,
    get_time_ms,
    RESAMPLE_MS as _NOTATION_RESAMPLE_MS,
    MAX_LAG_MS as _NOTATION_MAX_LAG_MS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Config globale
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

torch.set_float32_matmul_precision("medium")

_N_CPU = os.cpu_count() or 1
if DEVICE == "mps":
    torch.set_num_threads(max(1, min(4, _N_CPU // 2)))
    torch.set_num_interop_threads(1)
else:
    torch.set_num_threads(_N_CPU)
    torch.set_num_interop_threads(max(1, _N_CPU // 2))

_DATALOADER_WORKERS = 0 if DEVICE == "mps" else min(2, max(0, _N_CPU - 1))

PAIRS: List[Tuple[str, str]] = [
    ("tracker_head",  "cam_head"),
    ("tracker_left",  "cam_left"),
    ("tracker_right", "cam_right"),
    ("tracker_left",  "gripper_left"),
    ("tracker_right", "gripper_right"),
]

# Constantes IA
RESAMPLE_MS        = 10.0
MAX_LAG_MS         = 250.0
WINDOW_MS          = 1800.0
WINDOW_STRIDE_MS   = 900.0
MIN_OVERLAP_MS     = 1200.0

PSEUDO_POS_THR     = 0.72
PSEUDO_NEG_THR     = 0.30
EDGE_MARGIN_MS     = 20.0

MIN_CONFIDENCE_TO_APPLY = 0.70
MIN_PEAK_MARGIN    = 0.005
MIN_PAIR_WINDOWS   = 10

TRAIN_EPOCHS       = 5
BATCH_SIZE         = 256
LR                 = 1e-3
WEIGHT_DECAY       = 1e-4

TRAIN_MAX_LAG_MS   = 250.0
INFER_MAX_LAG_MS   = 1000.0

MODEL_DIRNAME      = "_sync_ml_model"
RESULTS_JSON       = "sync_ml_advanced_results.json"

ROOT_DIR = Path("/Users/christopher/Downloads/sync_test_1/treatment/data/")

# Constantes validation déterministe
MIN_MAJOR_SCORE_TO_APPLY  = 45.0
MIN_MAJOR_SCORE_POST      = 50.0
MIN_MAJOR_PAIRS_SCORABLE  = 2
MAJOR_PAIRS = {("tracker_head","cam_head"), ("tracker_left","cam_left"), ("tracker_right","cam_right")}
INDETERMINATE_ACTIVITY_THR = 0.12

CORRECTION_REPORT = "correction_report.json"
SCRIPT_DIR        = _SCRIPT_DIR
_PROJECT_ROOT     = _SCRIPT_DIR.parent
DATASET_DIR       = _PROJECT_ROOT / "dataset"
OUTPUT_DIR        = _PROJECT_ROOT / "corrected"
MODEL_DIR         = DATASET_DIR / "_sync_ml_model"

# Constantes heuristique (sync_fix)
_SF_RESAMPLE_MS     = 5.0
_SF_SMOOTH_SIGMA_MS = 80.0
_SF_N_BINS_MI       = 20
_SF_PEAK_TOL_MS     = 50.0
_SF_MIN_OVERLAP_MS  = 500.0
_SF_MARKER_KEY      = "sync_fix_applied"

_SF_METRIC_WEIGHTS = {
    "pearson":    0.30,
    "cosine":     0.25,
    "mutual_inf": 0.20,
    "peak_align": 0.15,
    "spectral":   0.10,
}


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — Structures de données communes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Flux:
    name: str
    t_ms_rel: np.ndarray
    signal: np.ndarray
    t_start_abs_ms: float
    source: str = "unknown"


@dataclass
class PairEstimate:
    ref_name: str
    tgt_name: str
    delta_start_ms: float
    residual_ms: float
    total_offset_ms: float
    shift_to_apply_ms: float
    confidence: float
    peak_margin: float
    best_score: float
    second_score: float
    sharpness: float
    is_reliable: bool
    method: str
    lags_ms: np.ndarray
    scores: np.ndarray


class PairResult:
    """Résultat d'une analyse heuristique (sync_fix)."""
    def __init__(self, ref_name: str, tgt_name: str):
        self.ref_name         = ref_name
        self.tgt_name         = tgt_name
        self.delta_start_ms   = 0.0
        self.residual_ms      = 0.0
        self.total_offset_ms  = 0.0
        self.offset_rec_ms    = 0.0
        self.confidence       = 0.0
        self.method           = "heuristic"
        self.scores_arr       = np.array([])
        self.candidates_arr   = np.array([])


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — Utils signal
# ══════════════════════════════════════════════════════════════════════════════

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    xf = x[finite]
    m, s = float(np.mean(xf)), float(np.std(xf))
    out = np.zeros_like(x, dtype=np.float32)
    out[finite] = (xf - m) / s if s >= 1e-8 else xf - m
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def robust_clip(x: np.ndarray, q: float = 99.0) -> np.ndarray:
    lim = np.percentile(np.abs(x[np.isfinite(x)]), q) if np.isfinite(x).any() else 1.0
    lim = max(lim, 1e-6)
    return np.clip(x, -lim, lim)


def moving_derivative(sig: np.ndarray, dt_ms: float) -> np.ndarray:
    dt_s = dt_ms / 1000.0
    out = np.gradient(sig) / max(dt_s, 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def resample_to_grid(t_src: np.ndarray, sig_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    sig_src = np.nan_to_num(sig_src, nan=0.0)
    return np.interp(t_grid, t_src, sig_src, left=0.0, right=0.0)


def _smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0:
        return x
    return gaussian_filter1d(x, sigma=sigma_samples)


def first_jsonl_capture_time(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                return float(json.loads(line)["capture_time"])
            except Exception:
                continue
    return None


def load_jsonl_capture_times(path: Path) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out[int(obj["index"])] = float(obj["capture_time"])
            except Exception:
                continue
    return out


def _shift_iso(series: pd.Series, delta_ns: int) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    shifted = ts + pd.to_timedelta(delta_ns, unit="ns")
    return shifted.dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 3 — Chargement des flux
# ══════════════════════════════════════════════════════════════════════════════

def _tracker_speed_features(df: pd.DataFrame, pos: str, dt_ms_nominal: float,
                             custom_cols: Optional[List[str]] = None) -> np.ndarray:
    if custom_cols:
        available = [c for c in custom_cols if c in df.columns]
        if not available:
            return np.zeros(len(df), dtype=np.float32)
        parts = [zscore(robust_clip(np.nan_to_num(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float32))))
                 for c in available]
        return _smooth(np.mean(parts, axis=0).astype(np.float32), 2.0)

    cols = [f"tracker_{pos}_x", f"tracker_{pos}_y", f"tracker_{pos}_z"]
    if not all(c in df.columns for c in cols):
        return np.zeros(len(df), dtype=np.float32)

    x = pd.to_numeric(df[cols[0]], errors="coerce").to_numpy(np.float32)
    y = pd.to_numeric(df[cols[1]], errors="coerce").to_numpy(np.float32)
    z = pd.to_numeric(df[cols[2]], errors="coerce").to_numpy(np.float32)
    dx, dy, dz = np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]), np.diff(z, prepend=z[0])

    speed = np.sqrt(dx*dx + dy*dy + dz*dz) / max(dt_ms_nominal / 1000.0, 1e-6)
    speed = _smooth(speed, 2.0)
    accel = moving_derivative(speed, dt_ms_nominal)
    jerk  = moving_derivative(accel, dt_ms_nominal)

    feat = (0.55 * zscore(robust_clip(speed))
            + 0.30 * zscore(robust_clip(np.abs(accel)))
            + 0.15 * zscore(robust_clip(np.abs(jerk))))
    return _smooth(feat, 2.0).astype(np.float32)


def load_trackers(session_dir: Path, signal_config: Optional[Dict] = None) -> Dict[str, Flux]:
    path = session_dir / "tracker_positions.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "timestamp_ns" in df.columns and pd.to_numeric(df["timestamp_ns"], errors="coerce").notna().any():
        t_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce").to_numpy(np.float64)
        t_start_abs_ms = float(t_ns[0] / 1e6)
        t_ms_rel = (t_ns - t_ns[0]) / 1e6
    else:
        t_s = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy(np.float64)
        t_start_abs_ms = 0.0
        t_ms_rel = (t_s - t_s[0]) * 1000.0

    order = np.argsort(t_ms_rel)
    t_ms_rel = t_ms_rel[order]
    df = df.iloc[order].reset_index(drop=True)

    dt_nom = float(np.median(np.diff(t_ms_rel[np.isfinite(t_ms_rel)]))) if len(t_ms_rel) > 1 else 17.0
    out = {}
    for pos in ("head", "left", "right"):
        custom_cols = (signal_config or {}).get(f"tracker_{pos}")
        sig = _tracker_speed_features(df, pos, dt_nom, custom_cols=custom_cols)
        out[f"tracker_{pos}"] = Flux(
            name=f"tracker_{pos}",
            t_ms_rel=t_ms_rel.astype(np.float32),
            signal=sig,
            t_start_abs_ms=float(t_start_abs_ms),
            source="tracker_positions.csv",
        )
    return out


def _build_video_signal_ia(df: pd.DataFrame, custom_cols: Optional[List[str]] = None) -> Optional[np.ndarray]:
    if custom_cols:
        available = [c for c in custom_cols if c in df.columns]
        if not available:
            return None
        parts = [zscore(robust_clip(np.nan_to_num(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float32))))
                 for c in available]
        sig = _smooth(np.mean(parts, axis=0), 1.5)
        ds = moving_derivative(sig, 33.0)
        return (0.8 * zscore(sig) + 0.2 * zscore(np.abs(ds))).astype(np.float32)

    candidates = [
        ("motion_p90_smooth", 0.40), ("motion_mean_smooth", 0.25),
        ("diff_mean_smooth", 0.20),  ("motion_p90", 0.10),
        ("motion_mean", 0.03),       ("diff_mean", 0.02),
    ]
    parts = []
    for col, w in candidates:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float32)
            parts.append(w * zscore(robust_clip(x)))
    if not parts:
        return None
    sig = _smooth(np.sum(parts, axis=0), 1.5)
    ds = moving_derivative(sig, 33.0)
    return (0.8 * zscore(sig) + 0.2 * zscore(np.abs(ds))).astype(np.float32)


def load_cameras(session_dir: Path, signal_config: Optional[Dict] = None) -> Dict[str, Flux]:
    out = {}
    for cam in ("head", "left", "right"):
        flux_csv   = session_dir / "videos" / f"{cam}_flux.csv"
        jsonl_path = session_dir / "videos" / f"{cam}.jsonl"
        if not flux_csv.exists():
            continue

        df = pd.read_csv(flux_csv)
        custom_cols = (signal_config or {}).get(f"cam_{cam}")
        sig = _build_video_signal_ia(df, custom_cols=custom_cols)
        if sig is None:
            continue

        if "timestamp_abs_ms" in df.columns and pd.to_numeric(df["timestamp_abs_ms"], errors="coerce").notna().any():
            t_abs  = pd.to_numeric(df["timestamp_abs_ms"], errors="coerce").to_numpy(np.float64)
            valid  = np.isfinite(t_abs)
            if valid.sum() < 2:
                continue
            first  = float(t_abs[valid][0])
            t_ms_rel = t_abs - first
            t_start_abs_ms = first
        else:
            t_s = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy(np.float64)
            t_ms_rel = (t_s - t_s[0]) * 1000.0
            anchor   = first_jsonl_capture_time(jsonl_path)
            t_start_abs_ms = float(anchor) if anchor is not None else 0.0

        valid    = np.isfinite(t_ms_rel) & np.isfinite(sig)
        t_ms_rel = t_ms_rel[valid]
        sig      = sig[valid]
        if len(t_ms_rel) < 10:
            continue

        out[f"cam_{cam}"] = Flux(
            name=f"cam_{cam}",
            t_ms_rel=t_ms_rel.astype(np.float32),
            signal=sig.astype(np.float32),
            t_start_abs_ms=float(t_start_abs_ms),
            source=f"videos/{cam}_flux.csv",
        )
    return out


def load_grippers(session_dir: Path, signal_config: Optional[Dict] = None) -> Dict[str, Flux]:
    out = {}
    for side in ("left", "right"):
        path = session_dir / f"gripper_{side}_data.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if "timestamp_ns" in df.columns and pd.to_numeric(df["timestamp_ns"], errors="coerce").notna().any():
            t_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce").to_numpy(np.float64)
            t_start_abs_ms = float(t_ns[0] / 1e6)
            t_ms_rel = (t_ns - t_ns[0]) / 1e6
        else:
            t_s = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy(np.float64)
            t_start_abs_ms = 0.0
            t_ms_rel = (t_s - t_s[0]) * 1000.0

        custom_cols = (signal_config or {}).get(f"gripper_{side}")
        if custom_cols:
            available = [c for c in custom_cols if c in df.columns]
            if not available:
                continue
            parts = [zscore(robust_clip(np.nan_to_num(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float32))))
                     for c in available]
            sig = _smooth(np.mean(parts, axis=0), 1.5)
        else:
            if "angle_deg" not in df.columns:
                continue
            angle = pd.to_numeric(df["angle_deg"], errors="coerce").to_numpy(np.float32)
            d1 = np.diff(angle, prepend=angle[0])
            d2 = np.diff(d1, prepend=d1[0])
            sig = _smooth(0.75 * zscore(np.abs(d1)) + 0.25 * zscore(np.abs(d2)), 1.5)

        valid    = np.isfinite(t_ms_rel) & np.isfinite(sig)
        t_ms_rel = t_ms_rel[valid]
        sig      = sig[valid]
        if len(t_ms_rel) < 10:
            continue

        out[f"gripper_{side}"] = Flux(
            name=f"gripper_{side}",
            t_ms_rel=t_ms_rel.astype(np.float32),
            signal=sig.astype(np.float32),
            t_start_abs_ms=float(t_start_abs_ms),
            source=path.name,
        )
    return out


def load_all_fluxes(session_dir: Path, signal_config: Optional[Dict] = None) -> Dict[str, Flux]:
    out = {}
    out.update(load_trackers(session_dir, signal_config=signal_config))
    out.update(load_cameras(session_dir, signal_config=signal_config))
    out.update(load_grippers(session_dir, signal_config=signal_config))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 4 — Métriques heuristiques (sync_fix)
# ══════════════════════════════════════════════════════════════════════════════

def _resamp(t_src: np.ndarray, sig: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.interp(t_grid, t_src, np.where(np.isfinite(sig), sig, 0.0), left=0.0, right=0.0)


def _envelope(arr: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(arr ** 2, sigma=max(sigma, 0.5))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def _mi(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8:
        return 0.0
    a_min, a_max = np.min(a), np.max(a)
    b_min, b_max = np.min(b), np.max(b)
    a_rng, b_rng = a_max - a_min, b_max - b_min
    if a_rng < 1e-12 or b_rng < 1e-12:
        return 0.0
    nb = _SF_N_BINS_MI
    ai = np.clip(((a - a_min) / a_rng * (nb - 1)).astype(int), 0, nb - 1)
    bi = np.clip(((b - b_min) / b_rng * (nb - 1)).astype(int), 0, nb - 1)
    joint = np.zeros((nb, nb))
    np.add.at(joint, (ai, bi), 1)
    joint /= joint.sum() + 1e-12
    pa, pb = joint.sum(1), joint.sum(0)
    ha  = float(scipy_entropy(pa + 1e-12))
    hb  = float(scipy_entropy(pb + 1e-12))
    hab = float(scipy_entropy(joint.flatten() + 1e-12))
    return float(np.clip((ha + hb - hab) / (min(ha, hb) + 1e-12), 0.0, 1.0))


def _peak_align_sf(a: np.ndarray, b: np.ndarray, resample_ms: float) -> float:
    tol = int(_SF_PEAK_TOL_MS / resample_ms)
    if len(a) < 10:
        return 0.0
    pa, _ = find_peaks(a, height=np.percentile(a, 70), distance=max(1, tol // 2))
    pb, _ = find_peaks(b, height=np.percentile(b, 70), distance=max(1, tol // 2))
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    return sum(1 for p in pa if np.any(np.abs(pb - p) <= tol)) / len(pa)


def _spectral_sf(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8:
        return 0.0
    n = min(len(a), len(b))
    fa, fb = np.abs(np.fft.rfft(a[:n])), np.abs(np.fft.rfft(b[:n]))
    if np.std(fa) < 1e-12 or np.std(fb) < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(fa, fb)[0, 1], 0.0, 1.0))


def _score_overlap_sf(a, b, env_a, env_b, resample_ms: float) -> float:
    m = {
        "pearson":    _pearson(a, b),
        "cosine":     _cosine(env_a, env_b),
        "mutual_inf": _mi(a, b),
        "peak_align": _peak_align_sf(a, b, resample_ms),
        "spectral":   _spectral_sf(a, b),
    }
    return sum(_SF_METRIC_WEIGHTS[k] * m[k] for k in _SF_METRIC_WEIGHTS)


# ── Random Forest auto-supervisé ──────────────────────────────────────────────

def _features_at_sf(ref_t, ref_sig, tgt_t, tgt_sig, offset_ms, resample_ms, sigma):
    tgt_ts = tgt_t + offset_ms
    t0 = max(ref_t[0], tgt_ts[0])
    t1 = min(ref_t[-1], tgt_ts[-1])
    if t1 - t0 < _SF_MIN_OVERLAP_MS:
        return None, 0
    grid = np.arange(t0, t1, resample_ms)
    if len(grid) < 8:
        return None, 0
    a  = _resamp(ref_t,  ref_sig, grid)
    b  = _resamp(tgt_ts, tgt_sig, grid)
    ea = _envelope(a, sigma)
    eb = _envelope(b, sigma)
    return [_pearson(a, b), _cosine(ea, eb), _mi(a, b),
            _peak_align_sf(a, b, resample_ms), _spectral_sf(a, b)], len(grid)


def build_rf(ref_t, ref_sig, tgt_t, tgt_sig, candidates, resample_ms, sigma):
    if not SKLEARN_AVAILABLE:
        return None
    tol = max(resample_ms * 3, 20.0)
    X, y = [], []
    for cand in candidates:
        feat, n = _features_at_sf(ref_t, ref_sig, tgt_t, tgt_sig, cand, resample_ms, sigma)
        if feat is None:
            continue
        X.append(feat + [abs(cand) / (max(np.abs(candidates)) + 1e-6), n * resample_ms / 1000.0])
        y.append(1 if abs(cand) <= tol else 0)
    X, y = np.array(X, dtype=float), np.array(y, dtype=int)
    if y.sum() < 3 or (y == 0).sum() < 3:
        return None
    try:
        sc  = StandardScaler()
        Xs  = sc.fit_transform(X)
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(Xs, y)
        return clf, sc
    except Exception as e:
        warnings.warn(f"RF training failed: {e}")
        return None


def analyze_pair_sf(ref: Flux, tgt: Flux, candidates: np.ndarray,
                    resample_ms: float, sigma: float, use_ml: bool) -> PairResult:
    res = PairResult(ref.name, tgt.name)
    res.delta_start_ms = tgt.t_start_abs_ms - ref.t_start_abs_ms
    tgt_t_aligned = tgt.t_ms_rel + res.delta_start_ms

    hscores = np.zeros(len(candidates))
    for i, cand in enumerate(candidates):
        shifted = tgt_t_aligned + cand
        t0 = max(ref.t_ms_rel[0], shifted[0])
        t1 = min(ref.t_ms_rel[-1], shifted[-1])
        if t1 - t0 < _SF_MIN_OVERLAP_MS:
            continue
        grid = np.arange(t0, t1, resample_ms)
        if len(grid) < 8:
            continue
        a = _resamp(ref.t_ms_rel, ref.signal, grid)
        b = _resamp(shifted, tgt.signal, grid)
        hscores[i] = _score_overlap_sf(a, b, _envelope(a, sigma), _envelope(b, sigma), resample_ms)

    res.scores_arr     = hscores
    res.candidates_arr = candidates

    best_h = int(np.argmax(hscores))
    resid_h = float(candidates[best_h])
    if 0 < best_h < len(hscores) - 1:
        y0, y1, y2 = hscores[best_h-1], hscores[best_h], hscores[best_h+1]
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-12:
            resid_h += (y0 - y2) / denom * (candidates[1] - candidates[0])

    res.residual_ms = resid_h
    res.confidence  = float(hscores[best_h])
    res.method      = "heuristic"

    if use_ml:
        rf = build_rf(ref.t_ms_rel, ref.signal, tgt_t_aligned, tgt.signal,
                      candidates, resample_ms, sigma)
        if rf is not None:
            clf, sc = rf
            X_pred, valid = [], []
            for i, cand in enumerate(candidates):
                feat, n = _features_at_sf(ref.t_ms_rel, ref.signal, tgt_t_aligned, tgt.signal,
                                          cand, resample_ms, sigma)
                if feat is None:
                    continue
                X_pred.append(feat + [abs(cand) / (max(np.abs(candidates)) + 1e-6), n * resample_ms / 1000.0])
                valid.append(i)
            if X_pred:
                proba = clf.predict_proba(sc.transform(np.array(X_pred, dtype=float)))
                pi    = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
                best_local  = int(np.argmax(proba[:, pi]))
                best_global = valid[best_local]
                res.residual_ms = float(candidates[best_global])
                res.confidence  = float(proba[best_local, pi])
                res.method      = "ml"

    res.total_offset_ms = res.delta_start_ms + res.residual_ms
    res.offset_rec_ms   = -res.total_offset_ms
    return res


def apply_offsets_inplace(session_dir: Path, offsets: Dict[str, float], dry_run: bool) -> None:
    for flux_name, offset_ms in offsets.items():
        if abs(offset_ms) < 0.1:
            print(f"    {flux_name:<20} : offset {offset_ms:+.2f} ms  → ignoré (< 0.1 ms)")
            continue

        delta_ns = int(round(offset_ms * 1_000_000))
        delta_s  = offset_ms / 1000.0

        if flux_name == "tracker":
            path = session_dir / "tracker_positions.csv"
            if path.exists():
                df = pd.read_csv(path)
                if "timestamp_ns" in df.columns:
                    df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce") - delta_ns
                if "time_seconds" in df.columns:
                    df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce") - delta_s
                if "timestamp" in df.columns:
                    df["timestamp"] = _shift_iso(df["timestamp"], -delta_ns)
                if not dry_run:
                    df.to_csv(path, index=False)
                print(f"    tracker              : {offset_ms:+.2f} ms  → {path.name}")

        elif flux_name.startswith("cam_"):
            cam  = flux_name[4:]
            path = session_dir / "videos" / f"{cam}.jsonl"
            if path.exists():
                lines_out = []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            rec["capture_time"] = rec["capture_time"] - offset_ms
                            lines_out.append(json.dumps(rec))
                        except (json.JSONDecodeError, KeyError):
                            lines_out.append(line)
                if not dry_run:
                    path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
                print(f"    {flux_name:<20} : {offset_ms:+.2f} ms  → {path.name}")

        elif flux_name.startswith("gripper_"):
            side = flux_name[8:]
            path = session_dir / f"gripper_{side}_data.csv"
            if path.exists():
                df = pd.read_csv(path)
                for col in ("timestamp_ns", "t_ms_corrected_ns"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce") - delta_ns
                if "time_seconds" in df.columns:
                    df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce") - delta_s
                if "t_ms" in df.columns:
                    df["t_ms"] = pd.to_numeric(df["t_ms"], errors="coerce") - offset_ms
                if "timestamp" in df.columns:
                    df["timestamp"] = _shift_iso(df["timestamp"], -delta_ns)
                if not dry_run:
                    df.to_csv(path, index=False)
                print(f"    {flux_name:<20} : {offset_ms:+.2f} ms  → {path.name}")


def plot_session_sf(results: List[PairResult], session_dir: Path) -> None:
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i, 0]
        if len(r.candidates_arr):
            ax.plot(r.candidates_arr, r.scores_arr, lw=1.5, color="#1565C0", alpha=0.8)
            ax.axvline(r.residual_ms, color="#C62828", lw=2, linestyle="--",
                       label=f"résidu={r.residual_ms:+.1f} ms")
            ax.axvline(0.0, color="#999", lw=0.8, linestyle=":")
        ax.set_title(
            f"{r.ref_name} ↔ {r.tgt_name}  |  "
            f"Δstart={r.delta_start_ms:+.1f} ms  résidu={r.residual_ms:+.1f} ms  "
            f"offset_rec={r.offset_rec_ms:+.1f} ms  conf={r.confidence:.3f}  [{r.method}]",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Décalage résiduel (ms)")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"ML Sync — {session_dir.name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = session_dir / "sync_fix_plot.png"
    plt.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"    Graphe : {out}")


def process_session_sf(session_dir: Path, candidates: np.ndarray, resample_ms: float,
                       sigma: float, use_ml: bool, dry_run: bool, force: bool,
                       make_plot: bool) -> Optional[Dict]:
    print(f"\n{'─'*60}")
    print(f"  Session : {session_dir.name}")

    meta_path = session_dir / "metadata.json"
    metadata  = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    session_id = metadata.get("session_id", session_dir.name)

    if not force and metadata.get(_SF_MARKER_KEY):
        print(f"  → Déjà corrigée le {metadata[_SF_MARKER_KEY]}. Utilisez --force pour re-traiter.")
        return None

    all_fluxes: Dict[str, Flux] = {}
    all_fluxes.update(load_trackers(session_dir))
    all_fluxes.update(load_cameras(session_dir))
    all_fluxes.update(load_grippers(session_dir))
    if not all_fluxes:
        print("  → Aucun flux trouvé, session ignorée.")
        return None

    results: List[PairResult] = []
    for ref_name, tgt_name in PAIRS:
        if ref_name not in all_fluxes or tgt_name not in all_fluxes:
            continue
        r = analyze_pair_sf(all_fluxes[ref_name], all_fluxes[tgt_name],
                            candidates, resample_ms, sigma, use_ml)
        results.append(r)
        print(f"  {ref_name:<16} ↔ {tgt_name:<16} "
              f" Δstart={r.delta_start_ms:+.1f}  résidu={r.residual_ms:+.1f}"
              f"  offset_rec={r.offset_rec_ms:+.1f} ms  [{r.method} conf={r.confidence:.3f}]")

    if not results:
        print("  → Aucune paire analysable.")
        return None

    offsets: Dict[str, float] = {}
    for r in results:
        if r.tgt_name not in offsets:
            offsets[r.tgt_name] = r.offset_rec_ms

    print(f"\n  Offsets appliqués ({'DRY-RUN' if dry_run else 'ÉCRITURE EN PLACE'}) :")
    apply_offsets_inplace(session_dir, offsets, dry_run)

    if make_plot:
        plot_session_sf(results, session_dir)

    ml_json = {
        "session_id": session_id,
        "generator":  "sync.py (heuristic)",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "dry_run":    dry_run,
        "pairs": [
            {
                "ref": r.ref_name, "target": r.tgt_name,
                "delta_start_ms": r.delta_start_ms, "residual_ms": r.residual_ms,
                "total_offset_ms": r.total_offset_ms, "offset_rec_ms": r.offset_rec_ms,
                "confidence": r.confidence, "method": r.method,
            }
            for r in results
        ],
    }
    if not dry_run:
        (session_dir / "sync_fix_results.json").write_text(
            json.dumps(ml_json, indent=2, ensure_ascii=False), encoding="utf-8")
        if meta_path.exists():
            metadata[_SF_MARKER_KEY] = datetime.now(timezone.utc).isoformat()
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"session_id": session_id, "offsets": offsets, "n_pairs": len(results)}


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 5 — Modèle deep learning CrossModalAligner (IA.py)
# ══════════════════════════════════════════════════════════════════════════════

def make_common_grid(ref: Flux, tgt: Flux, delta_start_ms: float, extra_shift_ms: float,
                     resample_ms: float):
    tgt_t = tgt.t_ms_rel + delta_start_ms + extra_shift_ms
    t0 = max(float(ref.t_ms_rel[0]), float(tgt_t[0]))
    t1 = min(float(ref.t_ms_rel[-1]), float(tgt_t[-1]))
    if t1 - t0 < MIN_OVERLAP_MS:
        return None, None, None
    grid = np.arange(t0, t1, resample_ms, dtype=np.float32)
    if len(grid) < 16:
        return None, None, None
    a = resample_to_grid(ref.t_ms_rel, ref.signal, grid)
    b = resample_to_grid(tgt_t, tgt.signal, grid)
    return grid, a.astype(np.float32), b.astype(np.float32)


def window_slices(n: int, win: int, stride: int) -> List[Tuple[int, int]]:
    out = []
    s = 0
    while s + win <= n:
        out.append((s, s + win))
        s += stride
    return out


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if not np.isfinite(x).any() or not np.isfinite(y).any():
        return 0.0
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(np.clip(c, -1.0, 1.0)) if np.isfinite(c) else 0.0


def peak_sharpness(scores: np.ndarray, best_idx: int) -> float:
    if len(scores) < 5:
        return 0.0
    best   = float(scores[best_idx])
    median = float(np.median(scores))
    std    = float(np.std(scores))
    if std < 1e-6:
        return 0.0
    return float((best - median) / std)


def heuristic_alignment_score(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8:
        return 0.0
    a1 = zscore(a)
    b1 = zscore(b)
    corr  = safe_corrcoef(a1, b1)
    ea    = _smooth(a1 * a1, 2.0)
    eb    = _smooth(b1 * b1, 2.0)
    ecorr = safe_corrcoef(ea, eb)
    pa, _ = find_peaks(a1, height=np.percentile(a1, 75))
    pb, _ = find_peaks(b1, height=np.percentile(b1, 75))
    match = sum(1 for p in pa if np.any(np.abs(pb - p) <= 4)) / max(len(pa), 1) if len(pa) and len(pb) else 0.0
    fa    = np.abs(np.fft.rfft(a1))
    fb    = np.abs(np.fft.rfft(b1))
    scorr = safe_corrcoef(fa, fb)
    return float(np.clip(0.42 * max(corr, 0.0) + 0.28 * max(ecorr, 0.0) + 0.20 * match + 0.10 * max(scorr, 0.0), 0.0, 1.0))


def build_pseudo_examples_for_pair(
    ref: Flux, tgt: Flux, resample_ms: float, max_lag_ms: float, window_ms: float,
) -> List[Tuple[np.ndarray, np.ndarray, int, str]]:
    examples = []
    delta_start_ms = tgt.t_start_abs_ms - ref.t_start_abs_ms
    candidate_lags = np.arange(-max_lag_ms, max_lag_ms + resample_ms, resample_ms, dtype=np.float32)
    win    = int(window_ms / resample_ms)
    stride = max(4, int(WINDOW_STRIDE_MS / resample_ms))

    for lag in candidate_lags:
        if abs(float(lag)) >= (max_lag_ms - EDGE_MARGIN_MS):
            continue
        grid, a, b = make_common_grid(ref, tgt, delta_start_ms, float(lag), resample_ms)
        if grid is None:
            continue
        for s, e in window_slices(len(grid), win, stride):
            wa, wb = a[s:e], b[s:e]
            score  = heuristic_alignment_score(wa, wb)
            if score >= PSEUDO_POS_THR:
                examples.append((wa, wb, 1, f"{ref.name}|{tgt.name}"))
            elif score <= PSEUDO_NEG_THR:
                examples.append((wa, wb, 0, f"{ref.name}|{tgt.name}"))
    return examples


class PairWindowDataset(Dataset):
    def __init__(self, examples: List[Tuple[np.ndarray, np.ndarray, int, str]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        a, b, y, _ = self.examples[idx]
        a = zscore(robust_clip(a))
        b = zscore(robust_clip(b))
        da = moving_derivative(a, RESAMPLE_MS)
        db = moving_derivative(b, RESAMPLE_MS)
        ea = _smooth(a * a, 2.0)
        eb = _smooth(b * b, 2.0)
        xa = np.stack([a, da, ea], axis=0).astype(np.float32)
        xb = np.stack([b, db, eb], axis=0).astype(np.float32)
        return torch.from_numpy(xa), torch.from_numpy(xb), torch.tensor(y, dtype=torch.float32)


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, s=1, p=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=s, padding=p),
            nn.BatchNorm1d(c_out), nn.GELU(),
            nn.Conv1d(c_out, c_out, k, stride=1, padding=p),
            nn.BatchNorm1d(c_out), nn.GELU(),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class Encoder1D(nn.Module):
    def __init__(self, in_ch=3, emb=64):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_ch, 16), nn.MaxPool1d(2),
            ConvBlock(16, 32),   nn.MaxPool1d(2),
            ConvBlock(32, 48),   nn.MaxPool1d(2),
            ConvBlock(48, 64),   nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Linear(64, emb), nn.GELU(), nn.Linear(emb, emb),
        )

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=-1)


class CrossModalAligner(nn.Module):
    def __init__(self, emb=64):
        super().__init__()
        self.enc_ref = Encoder1D(in_ch=3, emb=emb)
        self.enc_tgt = Encoder1D(in_ch=3, emb=emb)
        self.head = nn.Sequential(
            nn.Linear(emb * 4, 128), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, xa, xb):
        ea = self.enc_ref(xa)
        eb = self.enc_tgt(xb)
        feat  = torch.cat([ea, eb, torch.abs(ea - eb), ea * eb], dim=-1)
        logit = self.head(feat).squeeze(-1)
        return logit, ea, eb


def contrastive_margin(ea, eb, y, margin=0.6):
    dist = torch.norm(ea - eb, dim=-1)
    pos  = y * dist.pow(2)
    neg  = (1 - y) * F.relu(margin - dist).pow(2)
    return (pos + neg).mean()


def train_model(model, loader, epochs, lr, weight_decay):
    import time
    try:
        from tqdm import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs, pct_start=0.3
    )
    best_loss = float("inf")
    history   = []

    print(f"\n{'='*70}")
    print(f"[train] epochs={epochs}  samples={len(loader.dataset)}  device={DEVICE}")
    print(f"{'='*70}\n")

    t_train_start = time.time()
    for epoch in range(epochs):
        t_e = time.time()
        model.train()
        losses, n_correct, n_seen = [], 0, 0

        iter_loader = _tqdm(loader, desc=f"  epoch {epoch+1:02d}/{epochs}", leave=False) if _has_tqdm else loader
        for xa, xb, y in iter_loader:
            xa, xb, y = xa.to(DEVICE), xb.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logit, ea, eb = model(xa, xb)
            bce  = F.binary_cross_entropy_with_logits(logit, y)
            ctr  = contrastive_margin(ea, eb, y)
            loss = 0.75 * bce + 0.25 * ctr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            losses.append(float(loss.item()))
            with torch.no_grad():
                n_correct += int((torch.sigmoid(logit) >= 0.5).float().eq(y).sum().item())
                n_seen    += int(y.size(0))

        mean_loss = float(np.mean(losses))
        acc       = n_correct / max(n_seen, 1) * 100.0
        improved  = "*" if mean_loss < best_loss else " "
        if mean_loss < best_loss:
            best_loss = mean_loss
        history.append({"epoch": epoch + 1, "loss": mean_loss, "acc": acc})
        print(f"[train] epoch {epoch+1:02d}/{epochs}{improved}  loss={mean_loss:.4f}  acc={acc:.1f}%  t={time.time()-t_e:.1f}s")

    print(f"\n[train] terminé en {time.time()-t_train_start:.1f}s  best_loss={best_loss:.4f}")


def _session_is_clean(session_dir: Path) -> bool:
    meta_path  = session_dir / "metadata.json"
    state_path = session_dir / "pipeline_state.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    lv = meta.get("camera_label_verification")
    if not isinstance(lv, dict) or not lv.get("global_ok", False):
        return False
    if lv.get("confidence", 0.0) < 0.90:
        return False
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    steps = state.get("steps", {})
    for step in ("tracker", "video", "verify_labels"):
        if steps.get(step, {}).get("status") != "done":
            return False
    return True


def discover_sessions(root: Path, only_session: Optional[str]) -> List[Path]:
    all_sessions = sorted(
        p.parent for p in root.rglob("metadata.json")
        if p.parent.name.startswith("session_") and "__FAILED" not in p.parent.name
    )
    if only_session:
        all_sessions = [s for s in all_sessions if s.name == only_session]
    clean    = [s for s in all_sessions if _session_is_clean(s)]
    excluded = len(all_sessions) - len(clean)
    if excluded > 0:
        print(f"[discover] {len(all_sessions)} sessions — {excluded} exclues — {len(clean)} utilisées")
    else:
        print(f"[discover] {len(clean)} sessions prêtes")
    return clean


def rebalance_examples(examples, neg_pos_ratio=3, max_total=120000):
    pos = [ex for ex in examples if ex[2] == 1]
    neg = [ex for ex in examples if ex[2] == 0]
    random.shuffle(pos)
    random.shuffle(neg)
    neg = neg[:min(len(neg), len(pos) * neg_pos_ratio)]
    balanced = pos + neg
    random.shuffle(balanced)
    return balanced[:max_total] if len(balanced) > max_total else balanced


def _build_examples_for_session(args_tuple):
    session_dir, resample_ms, max_lag_ms, window_ms, signal_config = args_tuple
    fluxes = load_all_fluxes(session_dir, signal_config=signal_config)
    session_examples = []
    for ref_name, tgt_name in PAIRS:
        if ref_name not in fluxes or tgt_name not in fluxes:
            continue
        ex = build_pseudo_examples_for_pair(
            fluxes[ref_name], fluxes[tgt_name],
            resample_ms=resample_ms, max_lag_ms=max_lag_ms, window_ms=window_ms,
        )
        if ex:
            session_examples.append((session_dir.name, ref_name, tgt_name, ex))
    return session_examples


def build_training_examples(sessions: List[Path], resample_ms: float, max_lag_ms: float,
                             window_ms: float, signal_config: Optional[Dict] = None):
    worker_args = [(s, resample_ms, max_lag_ms, window_ms, signal_config) for s in sessions]
    n_workers = min(_N_CPU, len(sessions))
    all_examples = []
    if n_workers > 1:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for session_results in pool.imap_unordered(_build_examples_for_session, worker_args):
                for sess_name, ref_name, tgt_name, ex in session_results:
                    all_examples.extend(ex)
                    print(f"[pseudo] {sess_name}  {ref_name} ↔ {tgt_name}  examples={len(ex)}")
    else:
        for result in map(_build_examples_for_session, worker_args):
            for sess_name, ref_name, tgt_name, ex in result:
                all_examples.extend(ex)
                print(f"[pseudo] {sess_name}  {ref_name} ↔ {tgt_name}  examples={len(ex)}")
    return all_examples


def save_model(model: nn.Module, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")


def load_model(model_dir: Path) -> CrossModalAligner:
    model = CrossModalAligner().to(DEVICE)
    state = torch.load(model_dir / "model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def _prepare_window_tensors(windows_a: List[np.ndarray], windows_b: List[np.ndarray]):
    batch_xa, batch_xb = [], []
    for a, b in zip(windows_a, windows_b):
        a = zscore(robust_clip(a))
        b = zscore(robust_clip(b))
        batch_xa.append(np.stack([a, moving_derivative(a, RESAMPLE_MS), _smooth(a*a, 2.0)], axis=0).astype(np.float32))
        batch_xb.append(np.stack([b, moving_derivative(b, RESAMPLE_MS), _smooth(b*b, 2.0)], axis=0).astype(np.float32))
    return np.stack(batch_xa), np.stack(batch_xb)


@torch.no_grad()
def score_windows_with_model(model: CrossModalAligner, windows_a: List[np.ndarray],
                              windows_b: List[np.ndarray], chunk_size: int = 128) -> np.ndarray:
    if not windows_a:
        return np.array([], dtype=np.float32)
    xa_np, xb_np = _prepare_window_tensors(windows_a, windows_b)
    all_proba = []
    for i in range(0, len(xa_np), chunk_size):
        xa = torch.from_numpy(xa_np[i:i + chunk_size]).to(DEVICE)
        xb = torch.from_numpy(xb_np[i:i + chunk_size]).to(DEVICE)
        logit, _, _ = model(xa, xb)
        all_proba.append(torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(all_proba)


def scan_lags_for_delta(model, ref, tgt, delta_start_ms, resample_ms, max_lag_ms, window_ms):
    candidate_lags = np.arange(-max_lag_ms, max_lag_ms + resample_ms, resample_ms, dtype=np.float32)
    win    = int(window_ms / resample_ms)
    stride = max(4, int(WINDOW_STRIDE_MS / resample_ms))
    lag_scores: List[float] = []
    valid_lags: List[float] = []
    for lag in candidate_lags:
        if abs(float(lag)) >= (max_lag_ms - EDGE_MARGIN_MS):
            continue
        grid, a, b = make_common_grid(ref, tgt, delta_start_ms, float(lag), resample_ms)
        if grid is None:
            continue
        slices = window_slices(len(grid), win, stride)
        if len(slices) < MIN_PAIR_WINDOWS:
            continue
        proba = score_windows_with_model(model, [a[s:e] for s, e in slices], [b[s:e] for s, e in slices])
        if len(proba) == 0:
            continue
        score = 0.65 * float(np.percentile(proba, 75)) + 0.35 * float(np.mean(proba))
        lag_scores.append(score)
        valid_lags.append(float(lag))
    return np.array(valid_lags, dtype=np.float32), np.array(lag_scores, dtype=np.float32)


def estimate_pair_offset(model, ref, tgt, resample_ms, max_lag_ms, window_ms) -> PairEstimate:
    delta_start_ms = tgt.t_start_abs_ms - ref.t_start_abs_ms
    lags1, scores1 = scan_lags_for_delta(model, ref, tgt, float(delta_start_ms), resample_ms, max_lag_ms, window_ms)
    lags2, scores2 = scan_lags_for_delta(model, ref, tgt, 0.0, resample_ms, max_lag_ms, window_ms)

    candidates = []
    if len(lags1):
        candidates.append(("abs-prior", float(delta_start_ms), lags1, scores1, int(np.argmax(scores1))))
    if len(lags2):
        candidates.append(("no-prior", 0.0, lags2, scores2, int(np.argmax(scores2))))

    if not candidates:
        return PairEstimate(
            ref_name=ref.name, tgt_name=tgt.name,
            delta_start_ms=float(delta_start_ms), residual_ms=0.0,
            total_offset_ms=float(delta_start_ms), shift_to_apply_ms=float(-delta_start_ms),
            confidence=0.0, peak_margin=0.0, best_score=0.0, second_score=0.0,
            sharpness=0.0, is_reliable=False, method="deep-no-valid-lag",
            lags_ms=np.array([], dtype=np.float32), scores=np.array([], dtype=np.float32),
        )

    best_method, best_delta, lags, scores, best_idx = max(candidates, key=lambda x: float(x[3][x[4]]))
    best_score   = float(scores[best_idx])
    tmp          = scores.copy()
    tmp[best_idx] = -1e9
    second_score = float(np.max(tmp)) if len(tmp) > 1 else 0.0
    peak_margin  = best_score - second_score
    sharpness    = peak_sharpness(scores, best_idx)
    best_residual_ms = float(lags[best_idx])
    total_offset_ms  = float(best_delta + best_residual_ms)
    confidence       = float(np.clip(best_score, 0.0, 1.0))
    reliable = (confidence >= MIN_CONFIDENCE_TO_APPLY and peak_margin >= MIN_PEAK_MARGIN and sharpness >= 1.5)

    return PairEstimate(
        ref_name=ref.name, tgt_name=tgt.name,
        delta_start_ms=float(delta_start_ms), residual_ms=best_residual_ms,
        total_offset_ms=total_offset_ms, shift_to_apply_ms=float(-total_offset_ms),
        confidence=confidence, peak_margin=float(peak_margin),
        best_score=best_score, second_score=second_score, sharpness=sharpness,
        is_reliable=bool(reliable), method=f"deep-contrastive-{best_method}",
        lags_ms=lags, scores=scores,
    )


def backup_file(path: Path):
    backup = path.with_suffix(path.suffix + ".bak_syncml")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())


def apply_shift_to_target(session_dir: Path, target_name: str, shift_ms: float, dry_run: bool):
    if abs(shift_ms) < 0.1:
        print(f"    {target_name:<20}  shift={shift_ms:+.2f} ms  ignoré")
        return

    delta_ns = int(round(shift_ms * 1_000_000))
    delta_s  = shift_ms / 1000.0

    if target_name.startswith("cam_"):
        cam       = target_name.replace("cam_", "")
        jsonl_path = session_dir / "videos" / f"{cam}.jsonl"
        flux_path  = session_dir / "videos" / f"{cam}_flux.csv"
        if jsonl_path.exists():
            if not dry_run:
                backup_file(jsonl_path)
            lines = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        rec["capture_time"] = float(rec["capture_time"]) + shift_ms
                        lines.append(json.dumps(rec, ensure_ascii=False))
                    except Exception:
                        lines.append(line)
            if not dry_run:
                jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if flux_path.exists():
            df = pd.read_csv(flux_path)
            if "timestamp_abs_ms" in df.columns:
                df["timestamp_abs_ms"] = pd.to_numeric(df["timestamp_abs_ms"], errors="coerce") + shift_ms
            if not dry_run:
                backup_file(flux_path)
                df.to_csv(flux_path, index=False)

    elif target_name.startswith("gripper_"):
        side = target_name.replace("gripper_", "")
        path = session_dir / f"gripper_{side}_data.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "timestamp_ns" in df.columns:
                df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce") + delta_ns
            if "time_seconds" in df.columns:
                df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce") + delta_s
            if "timestamp" in df.columns:
                df["timestamp"] = _shift_iso(df["timestamp"], delta_ns)
            if not dry_run:
                backup_file(path)
                df.to_csv(path, index=False)


def plot_pair_estimate(session_dir: Path, est: PairEstimate):
    if len(est.lags_ms) == 0:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(est.lags_ms, est.scores, lw=2)
    plt.axvline(est.residual_ms, color="red", linestyle="--", lw=2,
                label=f"best={est.residual_ms:+.1f} ms")
    plt.axhline(est.confidence, color="gray", linestyle=":", lw=1)
    plt.title(
        f"{est.ref_name} ↔ {est.tgt_name} | Δstart={est.delta_start_ms:+.1f} ms | "
        f"residual={est.residual_ms:+.1f} ms | shift={est.shift_to_apply_ms:+.1f} ms | "
        f"conf={est.confidence:.3f} | reliable={est.is_reliable}"
    )
    plt.xlabel("Résidu lag (ms)")
    plt.ylabel("Score deep")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out = session_dir / f"plot_{est.ref_name}_to_{est.tgt_name}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def save_session_results(session_dir: Path, estimates: List[PairEstimate], dry_run: bool):
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run":       dry_run,
        "pairs": [
            {
                "ref": e.ref_name, "target": e.tgt_name,
                "delta_start_ms": e.delta_start_ms, "residual_ms": e.residual_ms,
                "total_offset_ms": e.total_offset_ms, "shift_to_apply_ms": e.shift_to_apply_ms,
                "confidence": e.confidence, "peak_margin": e.peak_margin,
                "second_score": e.second_score, "sharpness": e.sharpness,
                "is_reliable": e.is_reliable, "method": e.method,
            }
            for e in estimates
        ],
    }
    (session_dir / RESULTS_JSON).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def train_pipeline(root: Path, sessions: List[Path], args):
    import time
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"[pipeline] {len(sessions)} sessions  resample_ms={args.resample_ms}  window_ms={args.window_ms}")
    print(f"{'='*70}")

    examples = build_training_examples(
        sessions=sessions, resample_ms=args.resample_ms,
        max_lag_ms=TRAIN_MAX_LAG_MS, window_ms=args.window_ms,
    )
    if len(examples) < 200:
        raise RuntimeError(f"Pas assez d'exemples pseudo-labelisés : {len(examples)}")

    pos_raw = sum(y for _, _, y, _ in examples)
    neg_raw = len(examples) - pos_raw
    print(f"\n[pseudo] avant rééquilibrage : total={len(examples)}  pos={pos_raw}  neg={neg_raw}")
    examples = rebalance_examples(examples, neg_pos_ratio=3, max_total=120000)
    pos = sum(y for _, _, y, _ in examples)
    neg = len(examples) - pos
    print(f"[pseudo] après rééquilibrage : total={len(examples)}  pos={pos}  neg={neg}  t={time.time()-t0:.1f}s")

    ds = PairWindowDataset(examples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
                    num_workers=_DATALOADER_WORKERS, persistent_workers=False, pin_memory=False)

    model    = CrossModalAligner().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Paramètres entraînables : {n_params:,}")
    train_model(model, dl, epochs=args.epochs, lr=args.lr, weight_decay=WEIGHT_DECAY)

    model_dir = root / MODEL_DIRNAME
    save_model(model, model_dir)
    print(f"[model] saved to {model_dir}")


def estimate_session(root: Path, session_dir: Path, args):
    model_dir = root / MODEL_DIRNAME
    if not (model_dir / "model.pt").exists():
        raise RuntimeError("Modèle absent. Lance d'abord avec `python sync.py train`")

    model     = load_model(model_dir)
    fluxes    = load_all_fluxes(session_dir)
    estimates = []

    for ref_name, tgt_name in PAIRS:
        if ref_name not in fluxes or tgt_name not in fluxes:
            continue
        est = estimate_pair_offset(
            model=model, ref=fluxes[ref_name], tgt=fluxes[tgt_name],
            resample_ms=args.resample_ms, max_lag_ms=INFER_MAX_LAG_MS, window_ms=args.window_ms,
        )
        estimates.append(est)
        print(
            f"{session_dir.name} | {ref_name:<16} ↔ {tgt_name:<16} "
            f"Δstart={est.delta_start_ms:+7.1f}  resid={est.residual_ms:+7.1f}  "
            f"shift={est.shift_to_apply_ms:+7.1f} ms  conf={est.confidence:.3f}  "
            f"margin={est.peak_margin:.3f}  sharp={est.sharpness:.2f}  reliable={est.is_reliable}"
        )
        if getattr(args, "plot", False):
            plot_pair_estimate(session_dir, est)

    save_session_results(session_dir, estimates, getattr(args, "dry_run", False))

    if getattr(args, "apply", False) and not getattr(args, "dry_run", False):
        applied_targets = set()
        for est in estimates:
            if not est.is_reliable or est.tgt_name in applied_targets:
                continue
            apply_shift_to_target(session_dir, est.tgt_name, est.shift_to_apply_ms, dry_run=False)
            applied_targets.add(est.tgt_name)

    return estimates


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 6 — Validation déterministe (apply_corrections.py)
# ══════════════════════════════════════════════════════════════════════════════

def _score_pair_deterministic(session_dir: Path, ref_name: str, tgt_name: str) -> Optional[dict]:
    try:
        if ref_name.startswith("tracker_"):
            tracker_csv = session_dir / "tracker_positions.csv"
            if not tracker_csv.exists():
                return None
            df_t  = pd.read_csv(tracker_csv)
            body  = ref_name.replace("tracker_", "")
            t_cols = [c for c in [f"tracker_{body}_x", f"tracker_{body}_y", f"tracker_{body}_z"] if c in df_t.columns]
            if not t_cols:
                return None
            if "timestamp_ns" in df_t.columns:
                t_ns = pd.to_numeric(df_t["timestamp_ns"], errors="coerce").to_numpy(np.float64)
                ref_t = (t_ns - t_ns[0]) / 1e6
            else:
                t_s = pd.to_numeric(df_t["time_seconds"], errors="coerce").to_numpy(np.float64)
                ref_t = (t_s - t_s[0]) * 1000.0
            ref_sig = build_tracker_signal(df_t, t_cols)
        else:
            return None

        if tgt_name.startswith("cam_"):
            cam  = tgt_name.replace("cam_", "")
            flux = session_dir / "videos" / f"{cam}_flux.csv"
            if not flux.exists():
                return None
            df_v   = pd.read_csv(flux)
            v_cols = filter_existing_cols(df_v, ["motion_mean_smooth","diff_mean_smooth","motion_mean","diff_mean"])
            if "timestamp_abs_ms" in df_v.columns:
                t_abs = pd.to_numeric(df_v["timestamp_abs_ms"], errors="coerce").to_numpy(np.float64)
                valid = np.isfinite(t_abs)
                if valid.sum() < 2:
                    return None
                t_abs = t_abs[valid]
                tgt_t = t_abs - t_abs[0]
                df_v  = df_v.iloc[np.where(valid)[0]].reset_index(drop=True)
            else:
                t_s2  = pd.to_numeric(df_v["time_seconds"], errors="coerce").to_numpy(np.float64)
                tgt_t = (t_s2 - t_s2[0]) * 1000.0
            tgt_sig = _build_video_signal_notation(df_v, v_cols)

        elif tgt_name.startswith("gripper_"):
            side = tgt_name.replace("gripper_", "")
            path = session_dir / f"gripper_{side}_data.csv"
            if not path.exists():
                return None
            df_g   = pd.read_csv(path)
            g_cols = filter_existing_cols(df_g, ["opening_mm","angle_deg"])
            if "timestamp_ns" in df_g.columns:
                t_ns2 = pd.to_numeric(df_g["timestamp_ns"], errors="coerce").to_numpy(np.float64)
                tgt_t = (t_ns2 - t_ns2[0]) / 1e6
            else:
                t_s3  = pd.to_numeric(df_g["time_seconds"], errors="coerce").to_numpy(np.float64)
                tgt_t = (t_s3 - t_s3[0]) * 1000.0
            tgt_sig = _build_video_signal_notation(df_g, g_cols)
        else:
            return None

        report = compute_alignment_report(ref_t, ref_sig, tgt_t, tgt_sig)
        return {
            "score_100":        report.score_100,
            "estimated_lag_ms": report.estimated_lag_ms,
            "verdict":          report.verdict,
            "tracker_activity": report.tracker_activity,
            "video_activity":   report.video_activity,
            "peak_corr":        report.global_peak_corr,
            "mad_ms":           report.mad_window_lag_ms,
            "n_windows":        report.n_windows,
        }
    except Exception as exc:
        return {"error": str(exc), "score_100": None}


def validate_pre_application(session_dir: Path, estimates: list) -> dict:
    major_scores = {}
    poor_signal_count = 0
    for ref_name, tgt_name in MAJOR_PAIRS:
        label  = f"{ref_name} ↔ {tgt_name}"
        result = _score_pair_deterministic(session_dir, ref_name, tgt_name)
        if result is None or result.get("score_100") is None:
            major_scores[label] = None
            continue
        act_t = result.get("tracker_activity", 1.0)
        act_v = result.get("video_activity",   1.0)
        if act_t < INDETERMINATE_ACTIVITY_THR or act_v < INDETERMINATE_ACTIVITY_THR:
            poor_signal_count += 1
            major_scores[label] = None
        else:
            major_scores[label] = result["score_100"]

    scorable  = {k: v for k, v in major_scores.items() if v is not None}
    n_scorable = len(scorable)

    if n_scorable < MIN_MAJOR_PAIRS_SCORABLE:
        return {
            "status":       "indeterminate",
            "reason":       f"Seulement {n_scorable}/{len(MAJOR_PAIRS)} paires scorables (seuil={MIN_MAJOR_PAIRS_SCORABLE})",
            "major_scores": major_scores,
            "n_major_ok":   n_scorable,
            "poor_signal":  poor_signal_count > 0,
        }

    failing = {k: v for k, v in scorable.items() if v < MIN_MAJOR_SCORE_TO_APPLY}
    if failing:
        worst_pair  = min(failing, key=failing.get)
        worst_score = failing[worst_pair]
        return {
            "status":       "rejected",
            "reason":       f"{worst_pair} → {worst_score:.1f}/100 (seuil={MIN_MAJOR_SCORE_TO_APPLY})",
            "major_scores": major_scores,
            "n_major_ok":   n_scorable,
            "poor_signal":  False,
        }

    return {
        "status":       "ok",
        "reason":       f"{n_scorable} paires majeures scorables, toutes ≥ {MIN_MAJOR_SCORE_TO_APPLY}",
        "major_scores": major_scores,
        "n_major_ok":   n_scorable,
        "poor_signal":  False,
    }


def validate_post_application(session_dst: Path) -> dict:
    major_scores_post = {}
    for ref_name, tgt_name in MAJOR_PAIRS:
        label  = f"{ref_name} ↔ {tgt_name}"
        result = _score_pair_deterministic(session_dst, ref_name, tgt_name)
        major_scores_post[label] = result["score_100"] if result and result.get("score_100") is not None else None

    scorable = {k: v for k, v in major_scores_post.items() if v is not None}
    failing  = {k: v for k, v in scorable.items() if v < MIN_MAJOR_SCORE_POST}

    if len(scorable) < MIN_MAJOR_PAIRS_SCORABLE:
        return {"status": "indeterminate", "major_scores_post": major_scores_post}
    if failing:
        worst = min(failing, key=failing.get)
        return {
            "status":            "rejected_post",
            "reason_post":       f"Score post-application insuffisant : {worst} → {failing[worst]:.1f}/100",
            "major_scores_post": major_scores_post,
        }
    return {"status": "confirmed", "major_scores_post": major_scores_post}


def apply_shift_cam(session_copy: Path, cam: str, shift_ms: float, dry_run: bool) -> List[str]:
    modified   = []
    jsonl_path = session_copy / "videos" / f"{cam}.jsonl"
    if jsonl_path.exists():
        if not dry_run:
            lines = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        rec["capture_time"] = float(rec["capture_time"]) + shift_ms
                        lines.append(json.dumps(rec, ensure_ascii=False))
                    except Exception:
                        lines.append(line)
            jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        modified.append(f"videos/{cam}.jsonl")

    flux_path = session_copy / "videos" / f"{cam}_flux.csv"
    if flux_path.exists():
        df = pd.read_csv(flux_path)
        if "timestamp_abs_ms" in df.columns:
            df["timestamp_abs_ms"] = pd.to_numeric(df["timestamp_abs_ms"], errors="coerce") + shift_ms
            if not dry_run:
                df.to_csv(flux_path, index=False)
            modified.append(f"videos/{cam}_flux.csv")

    return modified


def apply_shift_gripper(session_copy: Path, side: str, shift_ms: float, dry_run: bool) -> List[str]:
    modified = []
    path     = session_copy / f"gripper_{side}_data.csv"
    if not path.exists():
        return modified
    delta_ns = int(round(shift_ms * 1_000_000))
    delta_s  = shift_ms / 1000.0
    df       = pd.read_csv(path)
    changed  = False
    if "timestamp_ns" in df.columns:
        df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce") + delta_ns
        changed = True
    if "time_seconds" in df.columns:
        df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce") + delta_s
        changed = True
    if "timestamp" in df.columns:
        df["timestamp"] = _shift_iso(df["timestamp"], delta_ns)
        changed = True
    if changed:
        if not dry_run:
            df.to_csv(path, index=False)
        modified.append(f"gripper_{side}_data.csv")
    return modified


def find_sessions(dataset_dir: Path, only_session: Optional[str]) -> List[Path]:
    return sorted(
        p.parent for p in dataset_dir.rglob("metadata.json")
        if p.parent.name.startswith("session_") and "__FAILED" not in str(p.parent)
        and (only_session is None or p.parent.name == only_session)
    )


def session_has_enough_flux(session: Path) -> bool:
    videos = session / "videos"
    return videos.exists() and len(list(videos.glob("*_flux.csv"))) >= 3


def process_session_apply(
    session_src: Path,
    output_dir: Path,
    model,
    dry_run: bool,
    force: bool,
) -> dict:
    session_dst = output_dir / session_src.name

    if session_dst.exists():
        if not force:
            print(f"  [SKIP] {session_src.name} — déjà dans corrected/ (--force pour écraser)")
            return {"session": session_src.name, "status": "skipped"}
        shutil.rmtree(session_dst)

    print(f"\n  [INFER] {session_src.name} — estimation des offsets …", end=" ", flush=True)
    fluxes    = load_all_fluxes(session_src)
    estimates = []
    for ref_name, tgt_name in PAIRS:
        if ref_name not in fluxes or tgt_name not in fluxes:
            continue
        estimates.append(estimate_pair_offset(
            model=model, ref=fluxes[ref_name], tgt=fluxes[tgt_name],
            resample_ms=RESAMPLE_MS, max_lag_ms=INFER_MAX_LAG_MS, window_ms=WINDOW_MS,
        ))
    print(f"{len(estimates)} paires estimées")

    reliable = [e for e in estimates if e.is_reliable]

    print(f"  [VALIDATE-PRE] score déterministe sur les sources …", end=" ", flush=True)
    pre_validation = validate_pre_application(session_src, estimates)
    print(f"status={pre_validation['status']}  reason={pre_validation['reason']}")

    if pre_validation["status"] == "indeterminate":
        return {"session": session_src.name, "status": "indeterminate",
                "reason": pre_validation["reason"], "pre_validation": pre_validation,
                "n_pairs": len(estimates), "n_reliable": len(reliable)}

    if pre_validation["status"] == "rejected":
        return {"session": session_src.name, "status": "rejected",
                "reason": pre_validation["reason"], "pre_validation": pre_validation,
                "n_pairs": len(estimates), "n_reliable": len(reliable)}

    print(f"  [COPY] {session_src.name} …", end=" ", flush=True)
    if not dry_run:
        shutil.copytree(session_src, session_dst)
    print("OK")

    report_pairs    = []
    applied_targets = set()
    files_modified  = []

    for e in estimates:
        row = {
            "ref": e.ref_name, "target": e.tgt_name,
            "delta_start_ms": round(e.delta_start_ms, 3),
            "residual_ms": round(e.residual_ms, 3),
            "shift_to_apply_ms": round(e.shift_to_apply_ms, 3),
            "confidence": round(e.confidence, 4),
            "peak_margin": round(e.peak_margin, 5),
            "sharpness": round(e.sharpness, 3),
            "is_reliable": e.is_reliable,
            "is_major": (e.ref_name, e.tgt_name) in MAJOR_PAIRS,
            "method": e.method, "applied": False,
        }
        report_pairs.append(row)

        if not e.is_reliable:
            print(f"    {e.ref_name:<16} ↔ {e.tgt_name:<16}  conf={e.confidence:.3f}  → non fiable, ignoré")
            continue
        if e.tgt_name in applied_targets:
            continue

        shift = e.shift_to_apply_ms
        print(f"    {e.ref_name:<16} ↔ {e.tgt_name:<16}  shift={shift:+.1f} ms  conf={e.confidence:.3f}  → APPLIQUÉ")

        if not dry_run:
            if e.tgt_name.startswith("cam_"):
                files_modified.extend(apply_shift_cam(session_dst, e.tgt_name.replace("cam_", ""), shift, False))
            elif e.tgt_name.startswith("gripper_"):
                files_modified.extend(apply_shift_gripper(session_dst, e.tgt_name.replace("gripper_", ""), shift, False))

        applied_targets.add(e.tgt_name)
        row["applied"] = True

    final_status   = "corrected"
    post_validation: dict = {}

    if not dry_run and applied_targets:
        print(f"  [VALIDATE-POST] score déterministe sur les fichiers corrigés …", end=" ", flush=True)
        post_validation = validate_post_application(session_dst)
        print(f"status={post_validation['status']}")

        if post_validation["status"] in ("rejected_post", "indeterminate"):
            print(f"  [ROLLBACK] score post insuffisant — suppression de {session_dst}")
            shutil.rmtree(session_dst)
            return {
                "session": session_src.name, "status": "rejected_post",
                "reason": post_validation.get("reason_post", "score post-application insuffisant"),
                "pre_validation": pre_validation, "post_validation": post_validation,
                "n_pairs": len(estimates), "n_reliable": len(reliable), "n_applied": len(applied_targets),
            }

    report = {
        "session": session_src.name, "corrected_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run, "status": final_status,
        "n_pairs": len(estimates), "n_reliable": len(reliable), "n_applied": len(applied_targets),
        "files_modified": files_modified,
        "pre_validation": pre_validation, "post_validation": post_validation,
        "pairs": report_pairs,
    }
    if not dry_run:
        (session_dst / CORRECTION_REPORT).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 7 — Score sessions (score_sessions.py)
# ══════════════════════════════════════════════════════════════════════════════

_TRACKER_PAIRS_SCORE = {
    "head":  ["tracker_head_x",  "tracker_head_y",  "tracker_head_z"],
    "left":  ["tracker_left_x",  "tracker_left_y",  "tracker_left_z"],
    "right": ["tracker_right_x", "tracker_right_y", "tracker_right_z"],
}
_VIDEO_COLS_CANDIDATES  = ["motion_mean_smooth","diff_mean_smooth","motion_mean","diff_mean"]
_GRIPPER_COLS_CANDIDATES = ["opening_mm","angle_deg"]


def _tracker_time_ms(df: pd.DataFrame) -> np.ndarray:
    if "timestamp_ns" in df.columns:
        return get_time_ms(df, ["timestamp_ns"], scale_to_ms=1e-6)
    return get_time_ms(df, ["time_seconds"], scale_to_ms=1000.0)


def _video_time_ms(df: pd.DataFrame) -> np.ndarray:
    if "timestamp_abs_ms" in df.columns:
        return get_time_ms(df, ["timestamp_abs_ms"], scale_to_ms=1.0)
    return get_time_ms(df, ["time_seconds"], scale_to_ms=1000.0)


def _gripper_time_ms(df: pd.DataFrame) -> np.ndarray:
    if "t_ms_corrected_ns" in df.columns:
        return get_time_ms(df, ["t_ms_corrected_ns"], scale_to_ms=1e-6)
    if "timestamp_ns" in df.columns:
        return get_time_ms(df, ["timestamp_ns"], scale_to_ms=1e-6)
    return get_time_ms(df, ["time_seconds"], scale_to_ms=1000.0)


def _score_tracker_video(df_t: pd.DataFrame, tracker_body: str, df_v: pd.DataFrame) -> Tuple[Optional[AlignReport], str]:
    try:
        t_cols  = filter_existing_cols(df_t, _TRACKER_PAIRS_SCORE[tracker_body])
        v_cols  = filter_existing_cols(df_v, _VIDEO_COLS_CANDIDATES)
        t_ms    = _tracker_time_ms(df_t)
        v_ms    = _video_time_ms(df_v)
        t_sig   = build_tracker_signal(df_t.iloc[:len(t_ms)].copy(), t_cols)
        v_sig   = _build_video_signal_notation(df_v.iloc[:len(v_ms)].copy(), v_cols)
        report  = compute_alignment_report(t_ms, t_sig, v_ms, v_sig)
        return report, ""
    except Exception as exc:
        return None, str(exc)


def _score_tracker_gripper(df_t: pd.DataFrame, tracker_body: str,
                            df_g: pd.DataFrame, gripper_side: str) -> Tuple[Optional[AlignReport], str]:
    try:
        t_cols  = filter_existing_cols(df_t, _TRACKER_PAIRS_SCORE[tracker_body])
        g_cols  = filter_existing_cols(df_g, _GRIPPER_COLS_CANDIDATES)
        t_ms    = _tracker_time_ms(df_t)
        g_ms    = _gripper_time_ms(df_g)
        t_sig   = build_tracker_signal(df_t.iloc[:len(t_ms)].copy(), t_cols)
        g_sig   = _build_video_signal_notation(df_g.iloc[:len(g_ms)].copy(), g_cols)
        report  = compute_alignment_report(t_ms, t_sig, g_ms, g_sig)
        return report, ""
    except Exception as exc:
        return None, str(exc)


def score_session(session_dir: Path) -> Dict:
    from dataclasses import asdict
    name = session_dir.name
    result: Dict = {
        "session": name, "pairs": {}, "session_score": None,
        "worst_pair": None, "best_pair": None, "n_pairs_tested": 0,
        "n_pairs_ok": 0, "verdict_session": "indéterminé", "estimated_lags_ms": {},
    }

    tracker_csv = session_dir / "tracker_positions.csv"
    if not tracker_csv.exists():
        result["error"] = "tracker_positions.csv manquant"
        return result
    df_t = pd.read_csv(tracker_csv)

    pair_scores: List[Tuple[str, float, float]] = []

    for cam in ("head", "left", "right"):
        flux_csv = session_dir / "videos" / f"{cam}_flux.csv"
        if not flux_csv.exists():
            continue
        df_v      = pd.read_csv(flux_csv)
        pair_name = f"tracker_{cam} ↔ cam_{cam}"
        result["n_pairs_tested"] += 1
        report, err = _score_tracker_video(df_t, cam, df_v)
        if err:
            result["pairs"][pair_name] = {"error": err, "score": None}
        else:
            result["pairs"][pair_name] = asdict(report)
            result["n_pairs_ok"] += 1
            pair_scores.append((pair_name, report.score_100, report.estimated_lag_ms))
            result["estimated_lags_ms"][pair_name] = report.estimated_lag_ms

    for side in ("left", "right"):
        gripper_csv = session_dir / f"gripper_{side}_data.csv"
        if not gripper_csv.exists():
            continue
        df_g      = pd.read_csv(gripper_csv)
        pair_name = f"tracker_{side} ↔ gripper_{side}"
        result["n_pairs_tested"] += 1
        report, err = _score_tracker_gripper(df_t, side, df_g, side)
        if err:
            result["pairs"][pair_name] = {"error": err, "score": None}
        else:
            result["pairs"][pair_name] = asdict(report)
            result["n_pairs_ok"] += 1
            pair_scores.append((pair_name, report.score_100, report.estimated_lag_ms))
            result["estimated_lags_ms"][pair_name] = report.estimated_lag_ms

    if not pair_scores:
        result["verdict_session"] = "pas de données"
        return result

    scores_only      = [s for _, s, _ in pair_scores]
    pair_scores_sorted = sorted(pair_scores, key=lambda x: x[1])
    worst = pair_scores_sorted[0]
    best  = pair_scores_sorted[-1]

    result["session_score"]  = round(worst[1], 2)
    result["worst_pair"]     = worst[0]
    result["best_pair"]      = best[0]
    result["mean_score"]     = round(float(np.mean(scores_only)), 2)
    result["median_score"]   = round(float(np.median(scores_only)), 2)
    result["scores_by_pair"] = {p: round(s, 2) for p, s, _ in pair_scores_sorted}

    ws = worst[1]
    result["verdict_session"] = ("parfaite" if ws >= 80 else "bonne" if ws >= 60
                                  else "moyenne" if ws >= 35 else "décalée")
    return result


_VERDICT_COLORS = {
    "parfaite":       "\033[92m",
    "bonne":          "\033[96m",
    "moyenne":        "\033[93m",
    "décalée":        "\033[91m",
    "indéterminé":    "\033[90m",
    "pas de données": "\033[90m",
}
_RESET = "\033[0m"


def _color_verdict(v: str) -> str:
    c = _VERDICT_COLORS.get(v, "")
    return f"{c}{v}{_RESET}" if c else v


def _score_bar(score: Optional[float], width: int = 20) -> str:
    if score is None:
        return "─" * width
    filled = int(round(score / 100.0 * width))
    return "█" * filled + "░" * (width - filled)


def print_score_summary(sessions_results: List[Dict], no_color: bool = False) -> None:
    def cv(v):
        return v if no_color else _color_verdict(v)

    sorted_res = sorted(sessions_results, key=lambda r: r.get("session_score") or -1.0)

    col_w  = [30, 7, 7, 7, 20, 11, 40]
    header = (
        f"{'SESSION':<{col_w[0]}} {'WORST':>{col_w[1]}} {'MEAN':>{col_w[2]}} "
        f"{'BEST':>{col_w[3]}} {'BAR':<{col_w[4]}} {'VERDICT':<{col_w[5]}} {'WORST PAIR':<{col_w[6]}}"
    )
    sep = "─" * len(header)

    print()
    print("╔" + "═" * (len(header) + 2) + "╗")
    print("║  RAPPORT DE SYNCHRONISATION — TOUTES SESSIONS" + " " * (len(header) - 45) + "  ║")
    print("╚" + "═" * (len(header) + 2) + "╝")
    print()
    print(header)
    print(sep)

    for r in sorted_res:
        name    = r["session"][:col_w[0]]
        worst   = f"{r['session_score']:.1f}" if r["session_score"] is not None else "N/A"
        mean_s  = f"{r.get('mean_score', 0.0):.1f}"
        best_s  = f"{r['scores_by_pair'][r['best_pair']]:.1f}" if r.get("best_pair") and r.get("scores_by_pair") else "N/A"
        bar     = _score_bar(r["session_score"])
        verdict = cv(r["verdict_session"])
        wpair   = (r.get("worst_pair") or "")[:col_w[6]]
        print(f"{name:<{col_w[0]}} {worst:>{col_w[1]}} {mean_s:>{col_w[2]}} {best_s:>{col_w[3]}} "
              f"{bar:<{col_w[4]}} {verdict:<{col_w[5]}} {wpair:<{col_w[6]}}")

    print(sep)
    valid_scores = [r["session_score"] for r in sessions_results if r["session_score"] is not None]
    if valid_scores:
        print()
        print(f"  Sessions analysées : {len(sessions_results)}")
        print(f"  Sessions avec score : {len(valid_scores)}")
        print(f"  Score moyen  : {np.mean(valid_scores):.1f} / 100")
        print(f"  Score médian : {np.median(valid_scores):.1f} / 100")
        print(f"  Score min    : {min(valid_scores):.1f}")
        print(f"  Score max    : {max(valid_scores):.1f}")
        n_parfaite = sum(1 for r in sessions_results if r["verdict_session"] == "parfaite")
        n_bonne    = sum(1 for r in sessions_results if r["verdict_session"] == "bonne")
        n_moyenne  = sum(1 for r in sessions_results if r["verdict_session"] == "moyenne")
        n_decalee  = sum(1 for r in sessions_results if r["verdict_session"] == "décalée")
        print()
        print(f"  Parfaite  : {n_parfaite:3d}  Bonne : {n_bonne:3d}  Moyenne : {n_moyenne:3d}  Décalée : {n_decalee:3d}")
    print()


def print_detail(r: Dict) -> None:
    print(f"\n{'─'*60}")
    print(f"  Détail session : {r['session']}")
    print(f"  Verdict global : {r['verdict_session']}")
    print(f"  Score pire paire : {r.get('session_score', 'N/A')}")
    print(f"  Score moyen      : {r.get('mean_score', 'N/A')}")
    print()
    for pair, score in (r.get("scores_by_pair") or {}).items():
        lag     = r.get("estimated_lags_ms", {}).get(pair, "?")
        detail  = r["pairs"].get(pair, {})
        verdict = detail.get("verdict", "?") if isinstance(detail, dict) else "erreur"
        err_flag = "  ⚠" if isinstance(detail, dict) and "error" in detail else ""
        print(f"    {pair}")
        print(f"      score={score:.1f}  lag={lag} ms  verdict={verdict}{err_flag}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_heuristic_parser(sub):
    p = sub.add_parser("heuristic", help="Correction heuristique multi-métriques en place.")
    p.add_argument("root_dir",     type=str, help="Répertoire contenant les dossiers session_*.")
    p.add_argument("--max-lag-ms", type=float, default=500.0)
    p.add_argument("--step-ms",    type=float, default=5.0)
    p.add_argument("--no-ml",      action="store_true")
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--force",      action="store_true")
    p.add_argument("--no-plot",    action="store_true")
    return p


def _build_train_parser(sub):
    p = sub.add_parser("train", help="Entraîner le modèle CrossModalAligner.")
    p.add_argument("--root",       type=Path, default=ROOT_DIR)
    p.add_argument("--session",    type=str, default=None)
    p.add_argument("--epochs",     type=int,   default=TRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--resample-ms",type=float, default=RESAMPLE_MS)
    p.add_argument("--window-ms",  type=float, default=WINDOW_MS)
    return p


def _build_apply_parser(sub):
    p = sub.add_parser("apply", help="Appliquer les corrections IA (copie vers corrected/).")
    p.add_argument("--dataset",  type=Path, default=DATASET_DIR)
    p.add_argument("--output",   type=Path, default=OUTPUT_DIR)
    p.add_argument("--session",  type=str, default=None)
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--force",    action="store_true")
    return p


def _build_score_parser(sub):
    p = sub.add_parser("score", help="Score de synchronisation sur toutes les sessions.")
    p.add_argument("--dataset",   type=Path, default=Path("./dataset"))
    p.add_argument("--session",   type=str,  default=None)
    p.add_argument("--json-out",  type=Path, default=None)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--detail",    action="store_true")
    p.add_argument("--no-color",  action="store_true")
    return p


def _cmd_heuristic(args) -> int:
    root = Path(args.root_dir).resolve()
    if not root.exists():
        print(f"ERREUR : répertoire introuvable : {root}", file=sys.stderr)
        return 1

    sessions = sorted(p.parent for p in root.rglob("metadata.json")
                      if p.parent.name.startswith("session_"))
    if not sessions:
        print(f"Aucun dossier session_* trouvé dans {root}.", file=sys.stderr)
        return 1

    print(f"Répertoire racine : {root}")
    print(f"Sessions trouvées : {len(sessions)}")
    print(f"Plage ±{args.max_lag_ms} ms  |  Pas {args.step_ms} ms")
    print(f"Mode : {'DRY-RUN' if args.dry_run else 'ÉCRITURE EN PLACE'}")

    candidates  = np.arange(-args.max_lag_ms, args.max_lag_ms + args.step_ms, args.step_ms)
    resample_ms = _SF_RESAMPLE_MS
    sigma       = _SF_SMOOTH_SIGMA_MS / resample_ms
    use_ml      = (not args.no_ml) and SKLEARN_AVAILABLE
    make_plot   = not args.no_plot

    summary = []
    for s in sessions:
        res = process_session_sf(s, candidates, resample_ms, sigma, use_ml,
                                 args.dry_run, args.force, make_plot)
        if res:
            summary.append(res)

    print(f"\n{'═'*60}")
    print(f"  RÉSUMÉ  —  {len(summary)}/{len(sessions)} sessions traitées")
    print(f"{'═'*60}")
    for s in summary:
        print(f"  {s['session_id']}  ({s['n_pairs']} paires)")
        for flux, off in s["offsets"].items():
            print(f"    {flux:<20} : {off:>+8.2f} ms")
    return 0


def _cmd_train(args) -> int:
    set_seed()
    root = args.root
    if not root.exists():
        print(f"ERREUR: ROOT_DIR introuvable: {root}", file=sys.stderr)
        return 1
    sessions = discover_sessions(root, getattr(args, "session", None))
    if not sessions:
        print("ERREUR: aucune session trouvée", file=sys.stderr)
        return 1
    print(f"device={DEVICE}  root={root}  sessions={len(sessions)}")
    train_pipeline(root, sessions, args)
    return 0


def _cmd_apply(args) -> int:
    dataset_dir: Path = args.dataset
    output_dir:  Path = args.output
    model_dir = dataset_dir / "_sync_ml_model"

    if not dataset_dir.exists():
        print(f"[ERREUR] Dataset introuvable : {dataset_dir}", file=sys.stderr)
        return 1
    if not (model_dir / "model.pt").exists():
        print(f"[ERREUR] Modèle introuvable : {model_dir / 'model.pt'}", file=sys.stderr)
        print("         Lance d'abord : python sync.py train", file=sys.stderr)
        return 1

    print(f"[INIT] Chargement du modèle depuis {model_dir} …", end=" ", flush=True)
    model = load_model(model_dir)
    print(f"OK  (device={DEVICE})")

    sessions = find_sessions(dataset_dir, args.session)
    usable   = [s for s in sessions if session_has_enough_flux(s)]
    skipped  = [s for s in sessions if not session_has_enough_flux(s)]

    print(f"\n[INFO] {len(usable)} sessions utilisables, {len(skipped)} ignorées")
    for s in skipped:
        print(f"  [SKIP] {s.name} — flux CSV insuffisants")

    if not usable:
        print("[ERREUR] Aucune session utilisable.", file=sys.stderr)
        return 1

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = []
    for session in usable:
        report = process_session_apply(
            session_src=session, output_dir=output_dir,
            model=model, dry_run=args.dry_run, force=args.force,
        )
        all_reports.append(report)

    n_corrected     = sum(1 for r in all_reports if r.get("status") == "corrected")
    n_indeterminate = sum(1 for r in all_reports if r.get("status") == "indeterminate")
    n_rejected      = sum(1 for r in all_reports if r.get("status") in ("rejected","rejected_post"))
    n_skipped       = sum(1 for r in all_reports if r.get("status") == "skipped")
    n_reliable_tot  = sum(r.get("n_reliable", 0) for r in all_reports if r.get("status") == "corrected")

    print(f"\n{'═'*60}")
    print(f"  Corrigées : {n_corrected}  Indéterminées : {n_indeterminate}  "
          f"Rejetées : {n_rejected}  Ignorées : {n_skipped}")
    print(f"  Paires fiables totales : {n_reliable_tot}")
    if not args.dry_run:
        print(f"  Sortie : {output_dir}")
    print(f"{'═'*60}")

    if not args.dry_run and n_corrected > 0:
        global_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_dir), "output": str(output_dir),
            "dry_run": args.dry_run, "sessions": all_reports,
        }
        (output_dir / "correction_summary.json").write_text(
            json.dumps(global_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


def _cmd_score(args) -> int:
    root = args.dataset.resolve()
    if not root.is_dir():
        print(f"Erreur : répertoire introuvable : {root}", file=sys.stderr)
        return 1

    if args.session:
        session_dirs = [root / args.session]
        if not session_dirs[0].is_dir():
            print(f"Erreur : session introuvable : {session_dirs[0]}", file=sys.stderr)
            return 1
    else:
        session_dirs = sorted([d for d in root.iterdir()
                                if d.is_dir() and d.name.startswith("session_")])

    print(f"  → {len(session_dirs)} session(s) détectées dans {root}", file=sys.stderr)

    results = []
    for i, sd in enumerate(session_dirs, 1):
        print(f"  [{i:2d}/{len(session_dirs)}] {sd.name} ...", end=" ", flush=True, file=sys.stderr)
        r = score_session(sd)
        results.append(r)
        score_str = f"{r['session_score']:.1f}" if r["session_score"] is not None else "N/A"
        print(f"score={score_str}  verdict={r['verdict_session']}", file=sys.stderr)

    display = results
    if args.min_score is not None:
        display = [r for r in results if r["session_score"] is None or r["session_score"] < args.min_score]

    print_score_summary(display, no_color=args.no_color)

    if args.detail:
        for r in sorted(display, key=lambda r: r.get("session_score") or 0):
            print_detail(r)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  JSON sauvegardé → {args.json_out}")

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 7 — Vérification alignement pinces (session_pinces.py)
# ══════════════════════════════════════════════════════════════════════════════

_NOMINAL_FPS_PINCES      = 30.0
_NOMINAL_PERIOD_S_PINCES = 1.0 / _NOMINAL_FPS_PINCES
_DROP_THRESHOLD_S_PINCES = 2.5 * _NOMINAL_PERIOD_S_PINCES


@dataclass
class _PincesThresholds:
    offset_ms:      float = 200.0
    latency_max_ms: float = 25.0
    jitter_std_ms:  float = 15.0
    max_drops:      int   = 5
    max_vel_mm_s:   float = 2000.0
    min_overlap_s:  float = 3.0


@dataclass
class _VidTsMetrics:
    n_frames:        int
    duration_s:      float
    dt_mean_ms:      float
    dt_std_ms:       float
    dt_min_ms:       float
    dt_max_ms:       float
    frame_drops:     int
    missing_indices: int
    jitter_std_ms:   float
    ts_ns:           np.ndarray
    indices:         np.ndarray


@dataclass
class _SensorMetricsPinces:
    n_samples:      int
    duration_s:     float
    dt_mean_ms:     float
    dt_std_ms:      float
    neg_dt_count:   int
    neg_dt_details: list
    vel_max_mm_s:   float
    vel_p99_mm_s:   float
    vel_anomalies:  int
    opening_range:  tuple


@dataclass
class _AlignMetrics:
    dur_vid_s:               float
    dur_sensor_s:            float
    overlap_s:               float
    offset_start_ms:         float
    latency_mean_ms:         float
    latency_std_ms:          float
    latency_max_abs_ms:      float
    latency_p95_abs_ms:      float
    frames_no_sensor:        int
    sensor_gap_max_ms:       float
    sensor_gap_count:        int
    linfit_slope:            float
    linfit_r2:               float
    linfit_residual_std_ms:  float
    linfit_residual_max_ms:  float
    opening_at_frames:       np.ndarray
    frame_ts_ns:             np.ndarray


@dataclass
class _PhysCoherence:
    n_frames_with_sensor: int
    n_frames_no_sensor:   int
    opening_mean_mm:      float
    opening_std_mm:       float
    opening_range:        tuple
    d_opening_max_mm_s:   float
    d_opening_p99_mm_s:   float
    impossible_jumps:     int


@dataclass
class _PincesAlert:
    code:      str
    level:     str
    message:   str
    value:     float
    threshold: float


@dataclass
class _SideResult:
    session_name: str
    side:         str
    success:      bool
    error:        str = ""
    video:        Optional[_VidTsMetrics]       = None
    sensor:       Optional[_SensorMetricsPinces] = None
    alignment:    Optional[_AlignMetrics]        = None
    physical:     Optional[_PhysCoherence]       = None
    alerts:       List[_PincesAlert] = None
    has_errors:   bool = False
    sensor_df:    Optional[pd.DataFrame] = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []

    @property
    def status(self) -> str:
        if not self.success:
            return "FAILED"
        if self.has_errors:
            return "ERROR"
        if self.alerts:
            return "WARNING"
        return "OK"

    @property
    def n_errors(self) -> int:
        return sum(1 for a in self.alerts if a.level == "ERROR")

    @property
    def n_warnings(self) -> int:
        return sum(1 for a in self.alerts if a.level == "WARNING")


def _load_jsonl_timestamps_pinces(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = open(path, "rb").read()
    indices, ts_ns = [], []
    for line in raw.split(b"\r\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
            indices.append(int(obj["index"]))
            ts_ns.append(int(obj["capture_time"]) * 1_000_000)
        except Exception:
            continue
    if not indices:
        raise RuntimeError(f"Aucune entrée valide dans {path}")
    indices = np.array(indices, dtype=np.int32)
    ts_ns   = np.array(ts_ns,   dtype=np.int64)
    order   = np.argsort(indices)
    return indices[order], ts_ns[order]


def _load_sensor_pinces(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("timestamp_ns", "opening_mm"):
        if col not in df.columns:
            raise RuntimeError(f"Colonne manquante '{col}' dans {path}")
    df = df[["timestamp_ns", "opening_mm"]].copy()
    df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce")
    df["opening_mm"]   = pd.to_numeric(df["opening_mm"],   errors="coerce")
    df = df.dropna().sort_values("timestamp_ns").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"Capteur vide après nettoyage : {path}")
    dt = np.diff(df["timestamp_ns"].values, prepend=df["timestamp_ns"].values[0]) / 1e6
    df["dt_ms"] = dt
    return df


def _fill_sensor_gaps_pinces(ts_ns, opening_mm, dt_nominal_ms=17.0, max_gap_ms=1500.0):
    if len(ts_ns) < 2:
        return ts_ns.copy(), opening_mm.copy()
    gap_thresh_ns = 4.0 * dt_nominal_ms * 1e6
    max_gap_ns    = max_gap_ms * 1e6
    filled_ts = list(ts_ns)
    filled_op = list(opening_mm.astype(float))
    diffs_ns  = np.diff(ts_ns)
    for i, gap_ns in enumerate(diffs_ns):
        if gap_ns <= gap_thresh_ns:
            continue
        gap_ms        = gap_ns / 1e6
        op_before     = float(opening_mm[i])
        op_after      = float(opening_mm[i + 1])
        delta_opening = abs(op_after - op_before)
        if gap_ms > max_gap_ms or delta_opening > 2.0:
            continue
        n = max(1, int(round(gap_ms / dt_nominal_ms)) - 1)
        synth_ts = np.linspace(ts_ns[i] + dt_nominal_ms * 1e6,
                               ts_ns[i+1] - dt_nominal_ms * 1e6, n, dtype=np.int64)
        synth_op = np.linspace(op_before, op_after, n)
        filled_ts.extend(synth_ts.tolist())
        filled_op.extend(synth_op.tolist())
    order = np.argsort(filled_ts)
    return np.array(filled_ts, dtype=np.int64)[order], np.array(filled_op)[order]


def _extend_sensor_backward_pinces(ts_ns, opening_mm, vid_start_ns, dt_nominal_ms=17.0):
    gap_ms = (ts_ns[0] - vid_start_ns) / 1e6
    if gap_ms <= 0 or gap_ms > 200.0:
        return ts_ns, opening_mm
    dt_ns    = int(dt_nominal_ms * 1e6)
    synth_ts = np.arange(vid_start_ns, ts_ns[0], dt_ns, dtype=np.int64)
    if len(synth_ts) == 0:
        return ts_ns, opening_mm
    synth_op = np.full(len(synth_ts), float(opening_mm[0]))
    return np.concatenate([synth_ts, ts_ns]), np.concatenate([synth_op, opening_mm.astype(float)])


def _apply_sensor_fixes_pinces(sensor_df: pd.DataFrame, vid_start_ns: int, dt_nominal_ms: float) -> pd.DataFrame:
    ts = sensor_df["timestamp_ns"].values.astype(np.int64)
    op = sensor_df["opening_mm"].values.astype(np.float64)
    ts, op = _fill_sensor_gaps_pinces(ts, op, dt_nominal_ms=dt_nominal_ms)
    ts, op = _extend_sensor_backward_pinces(ts, op, vid_start_ns, dt_nominal_ms=dt_nominal_ms)
    dt = np.diff(ts, prepend=ts[0]).astype(float) / 1e6
    return pd.DataFrame({"timestamp_ns": ts, "opening_mm": op, "dt_ms": dt})


def _analyze_vid_ts(indices, ts_ns) -> _VidTsMetrics:
    dt_ms = np.diff(ts_ns) / 1e6
    drops = int((dt_ms > _DROP_THRESHOLD_S_PINCES * 1000).sum())
    expected = np.arange(indices[0], indices[-1] + 1)
    missing  = int(len(expected) - len(indices))
    residuals = dt_ms - np.median(dt_ms)
    mad  = np.median(np.abs(residuals))
    return _VidTsMetrics(
        n_frames=len(ts_ns), duration_s=float((ts_ns[-1]-ts_ns[0])/1e9),
        dt_mean_ms=float(dt_ms.mean()), dt_std_ms=float(dt_ms.std()),
        dt_min_ms=float(dt_ms.min()), dt_max_ms=float(dt_ms.max()),
        frame_drops=drops, missing_indices=missing,
        jitter_std_ms=float(mad * 1.4826), ts_ns=ts_ns, indices=indices,
    )


def _analyze_sensor_pinces(df: pd.DataFrame, max_vel: float) -> _SensorMetricsPinces:
    ts  = df["timestamp_ns"].values
    op  = df["opening_mm"].values
    dt  = df["dt_ms"].values[1:]
    neg_idx = np.where(dt < 0)[0]
    neg_details = [{"idx": int(i), "dt_ms": float(dt[i]),
                    "opening_before": float(op[i]), "opening_after": float(op[i+1])}
                   for i in neg_idx]
    safe_dt_s = np.maximum(dt, 1.0) / 1000.0
    vel = np.abs(np.diff(op)) / safe_dt_s
    return _SensorMetricsPinces(
        n_samples=len(df), duration_s=float((ts[-1]-ts[0])/1e9),
        dt_mean_ms=float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.0,
        dt_std_ms=float(dt[dt > 0].std()) if (dt > 0).sum() > 1 else 0.0,
        neg_dt_count=len(neg_idx), neg_dt_details=neg_details,
        vel_max_mm_s=float(vel.max()) if len(vel) else 0.0,
        vel_p99_mm_s=float(np.percentile(vel, 99)) if len(vel) else 0.0,
        vel_anomalies=int((vel > max_vel).sum()),
        opening_range=(float(op.min()), float(op.max())),
    )


def _compute_alignment_pinces(vid: _VidTsMetrics, sensor_df: pd.DataFrame) -> _AlignMetrics:
    tv = vid.ts_ns.astype(np.float64)
    ts = sensor_df["timestamp_ns"].values.astype(np.float64)
    opening = sensor_df["opening_mm"].values

    dur_vid_s    = float((tv[-1] - tv[0]) / 1e9)
    dur_sensor_s = float((ts[-1] - ts[0]) / 1e9)
    t_ov0 = max(tv[0], ts[0]);  t_ov1 = min(tv[-1], ts[-1])
    overlap_s = float((t_ov1 - t_ov0) / 1e9) if t_ov1 > t_ov0 else 0.0
    offset_start_ms = float((tv[0] - ts[0]) / 1e6)

    nearest_hi = np.clip(np.searchsorted(ts, tv), 0, len(ts)-1)
    nearest_lo = np.clip(nearest_hi - 1, 0, len(ts)-1)
    d_hi = np.abs(tv - ts[nearest_hi]);  d_lo = np.abs(tv - ts[nearest_lo])
    nearest_ts = np.where(d_hi <= d_lo, ts[nearest_hi], ts[nearest_lo])
    latency_ms = (tv - nearest_ts) / 1e6

    in_range = (tv >= ts[0]) & (tv <= ts[-1])
    n_no_sensor = int((~in_range).sum())
    lat_valid = latency_ms[in_range]
    if len(lat_valid) > 0:
        lat_mean = float(lat_valid.mean());  lat_std = float(lat_valid.std())
        lat_max  = float(np.abs(lat_valid).max())
        lat_p95  = float(np.percentile(np.abs(lat_valid), 95))
    else:
        lat_mean = lat_std = lat_max = lat_p95 = np.nan

    mask_sen = (ts >= tv[0]) & (ts <= tv[-1])
    ts_in_vid = ts[mask_sen]
    dt_nom_ms = float(np.median(np.diff(ts)) / 1e6)
    if len(ts_in_vid) > 1:
        gaps_ms = np.diff(ts_in_vid) / 1e6
        sensor_gap_max_ms = float(gaps_ms.max())
        sensor_gap_count  = int((gaps_ms > 4.0 * dt_nom_ms).sum())
    else:
        sensor_gap_max_ms = float('inf');  sensor_gap_count = 0

    if len(tv) > 10:
        idx = np.arange(len(tv), dtype=np.float64)
        slope_ns, intercept_ns, r, _, _ = linregress(idx, tv)
        r2 = r ** 2
        fitted = slope_ns * idx + intercept_ns
        residuals_ms = (tv - fitted) / 1e6
        lf_std = float(residuals_ms.std());  lf_max = float(np.abs(residuals_ms).max())
    else:
        slope_ns = (tv[-1] - tv[0]) / max(len(tv)-1, 1)
        r2 = np.nan;  lf_std = lf_max = np.nan

    f_opening = _interp1d(ts, opening, kind="linear", bounds_error=False, fill_value=np.nan)
    opening_at_frames = f_opening(tv)

    return _AlignMetrics(
        dur_vid_s=dur_vid_s, dur_sensor_s=dur_sensor_s, overlap_s=overlap_s,
        offset_start_ms=offset_start_ms,
        latency_mean_ms=lat_mean, latency_std_ms=lat_std,
        latency_max_abs_ms=lat_max, latency_p95_abs_ms=lat_p95,
        frames_no_sensor=n_no_sensor,
        sensor_gap_max_ms=sensor_gap_max_ms, sensor_gap_count=sensor_gap_count,
        linfit_slope=float(slope_ns), linfit_r2=float(r2),
        linfit_residual_std_ms=lf_std, linfit_residual_max_ms=lf_max,
        opening_at_frames=opening_at_frames, frame_ts_ns=vid.ts_ns,
    )


def _compute_phys_coherence_pinces(aln: _AlignMetrics, max_vel: float) -> _PhysCoherence:
    op    = aln.opening_at_frames
    ts_ns = aln.frame_ts_ns.astype(np.float64)
    valid = np.isfinite(op)
    n_ok  = int(valid.sum());  n_miss = int((~valid).sum())
    if n_ok < 2:
        return _PhysCoherence(n_ok, n_miss, np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, 0)
    op_v = op[valid];  ts_v = ts_ns[valid]
    dt_s = np.diff(ts_v) / 1e9
    vel  = np.abs(np.diff(op_v)) / np.maximum(dt_s, 1e-6)
    return _PhysCoherence(
        n_frames_with_sensor=n_ok, n_frames_no_sensor=n_miss,
        opening_mean_mm=float(op_v.mean()), opening_std_mm=float(op_v.std()),
        opening_range=(float(op_v.min()), float(op_v.max())),
        d_opening_max_mm_s=float(vel.max()), d_opening_p99_mm_s=float(np.percentile(vel, 99)),
        impossible_jumps=int((vel > max_vel).sum()),
    )


def _generate_pinces_alerts(vid, sen, aln, phy, thr) -> List[_PincesAlert]:
    alerts = []

    def add(code, level, msg, value, threshold):
        alerts.append(_PincesAlert(code=code, level=level, message=msg,
                                   value=value, threshold=threshold))

    ONE_FRAME_MS = 1000.0 / _NOMINAL_FPS_PINCES
    abs_offset = abs(aln.offset_start_ms)
    if aln.offset_start_ms < -ONE_FRAME_MS:
        add("OFFSET_START", "ERROR",
            f"Vidéo démarre {aln.offset_start_ms:+.1f}ms avant le capteur — capteur manquant au début",
            abs_offset, ONE_FRAME_MS)
    elif abs_offset > thr.offset_ms:
        add("OFFSET_START", "ERROR",
            f"Offset démarrage {aln.offset_start_ms:+.1f}ms > seuil {thr.offset_ms:.0f}ms",
            abs_offset, thr.offset_ms)

    if np.isfinite(aln.latency_max_abs_ms) and aln.latency_max_abs_ms > thr.latency_max_ms:
        has_gap = aln.sensor_gap_count > 0
        level = "WARNING" if has_gap else ("ERROR" if aln.latency_max_abs_ms > thr.latency_max_ms * 2 else "WARNING")
        note  = f"  (causée par gap capteur {aln.sensor_gap_max_ms:.0f}ms)" if has_gap else ""
        add("LATENCY_MAX", level,
            f"Latence max frame→capteur {aln.latency_max_abs_ms:.1f}ms > seuil {thr.latency_max_ms:.0f}ms"
            f"  (P95={aln.latency_p95_abs_ms:.1f}ms){note}",
            aln.latency_max_abs_ms, thr.latency_max_ms)

    if aln.frames_no_sensor > 1:
        frac  = aln.frames_no_sensor / max(vid.n_frames, 1)
        level = "ERROR" if frac > 0.05 else "WARNING"
        add("FRAMES_NO_SENSOR", level,
            f"{aln.frames_no_sensor}/{vid.n_frames} frames ({frac*100:.1f}%) hors plage capteur",
            aln.frames_no_sensor, 0.0)

    if aln.sensor_gap_count > 0:
        level = "ERROR" if aln.sensor_gap_max_ms > 100.0 else "WARNING"
        add("SENSOR_GAP", level,
            f"{aln.sensor_gap_count} trou(s) capteur — gap max={aln.sensor_gap_max_ms:.1f}ms",
            aln.sensor_gap_max_ms, 0.0)

    if aln.overlap_s < thr.min_overlap_s:
        add("OVERLAP_SHORT", "ERROR",
            f"Recouvrement {aln.overlap_s:.1f}s < minimum {thr.min_overlap_s:.0f}s",
            aln.overlap_s, thr.min_overlap_s)

    if vid.jitter_std_ms > thr.jitter_std_ms:
        add("JITTER", "WARNING",
            f"Jitter timestamps vidéo {vid.jitter_std_ms:.2f}ms > seuil {thr.jitter_std_ms:.0f}ms",
            vid.jitter_std_ms, thr.jitter_std_ms)

    if vid.frame_drops > thr.max_drops:
        level = "ERROR" if vid.frame_drops > thr.max_drops * 3 else "WARNING"
        add("FRAME_DROPS", level,
            f"{vid.frame_drops} frame drops > seuil {thr.max_drops}",
            vid.frame_drops, thr.max_drops)

    if vid.missing_indices > 0:
        add("MISSING_FRAMES", "WARNING",
            f"{vid.missing_indices} indices manquants dans le flux JSONL",
            vid.missing_indices, 0.0)

    if sen.neg_dt_count > 0:
        level = "ERROR" if sen.neg_dt_count > 2 else "WARNING"
        add("SENSOR_NEG_DT", level,
            f"{sen.neg_dt_count} saut(s) temporel(s) négatif(s) dans le capteur",
            sen.neg_dt_count, 0.0)

    if sen.vel_anomalies > 0:
        add("SENSOR_VEL_ANOMALY", "WARNING",
            f"{sen.vel_anomalies} saut(s) impossible(s) dans capteur (vel > {thr.max_vel_mm_s:.0f}mm/s)",
            sen.vel_max_mm_s, thr.max_vel_mm_s)

    if phy.impossible_jumps > 0:
        add("INTERP_VEL_ANOMALY", "WARNING",
            f"{phy.impossible_jumps} saut(s) impossible(s) dans le capteur interpolé aux frames",
            phy.impossible_jumps, 0.0)

    return alerts


def _process_side_pinces(session_path: Path, side: str, thr: _PincesThresholds) -> _SideResult:
    session_name = session_path.name
    jsonl_path   = session_path / "videos" / f"{side}.jsonl"
    sensor_path  = session_path / f"gripper_{side}_data.csv"

    missing = [p for p in [jsonl_path, sensor_path] if not p.exists()]
    if missing:
        names = [p.name for p in missing]
        return _SideResult(session_name=session_name, side=side, success=False,
                           error=f"Fichiers absents : {names}")
    try:
        indices, ts_ns = _load_jsonl_timestamps_pinces(jsonl_path)
        sensor_df      = _load_sensor_pinces(sensor_path)
        dt_nominal_ms  = float(np.median(np.diff(sensor_df["timestamp_ns"].values)) / 1e6)
        sensor_df      = _apply_sensor_fixes_pinces(sensor_df, int(ts_ns[0]), dt_nominal_ms)

        vid = _analyze_vid_ts(indices, ts_ns)
        sen = _analyze_sensor_pinces(sensor_df, thr.max_vel_mm_s)
        aln = _compute_alignment_pinces(vid, sensor_df)
        phy = _compute_phys_coherence_pinces(aln, thr.max_vel_mm_s)
        als = _generate_pinces_alerts(vid, sen, aln, phy, thr)

        return _SideResult(
            session_name=session_name, side=side, success=True,
            video=vid, sensor=sen, alignment=aln, physical=phy,
            alerts=als, has_errors=any(a.level == "ERROR" for a in als),
            sensor_df=sensor_df,
        )
    except Exception as exc:
        import traceback
        return _SideResult(session_name=session_name, side=side, success=False,
                           error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def _print_pinces_summary(all_results: List[_SideResult]) -> None:
    ok_list   = [r for r in all_results if r.success and r.status == "OK"]
    warn_list = [r for r in all_results if r.success and r.status == "WARNING"]
    err_list  = [r for r in all_results if r.success and r.status == "ERROR"]
    fail_list = [r for r in all_results if not r.success]

    print()
    print("═" * 64)
    print(f"  Total    : {len(all_results):4d}")
    print(f"  OK       : {len(ok_list):4d}")
    print(f"  WARNING  : {len(warn_list):4d}")
    print(f"  ERROR    : {len(err_list):4d}")
    print(f"  FAILED   : {len(fail_list):4d}")

    success_list = [r for r in all_results if r.success and r.alignment]
    if success_list:
        offsets = np.array([r.alignment.offset_start_ms for r in success_list])
        lats    = np.array([r.alignment.latency_max_abs_ms for r in success_list
                            if np.isfinite(r.alignment.latency_max_abs_ms)])
        drops   = np.array([r.video.frame_drops for r in success_list])
        neg_dts = np.array([r.sensor.neg_dt_count for r in success_list])
        print()
        print(f"  Offset démarrage (ms)    : moy={offsets.mean():+.1f}  "
              f"med={np.median(offsets):+.1f}  max|.|={np.abs(offsets).max():.1f}")
        if len(lats):
            print(f"  Latence max frame→capteur: moy={lats.mean():.1f}ms  "
                  f"P95={np.percentile(lats, 95):.1f}ms  max={lats.max():.1f}ms")
        print(f"  Frame drops total        : {drops.sum():.0f}")
        print(f"  Sensor neg_dt total      : {neg_dts.sum():.0f}")

    print("═" * 64)

    if err_list or fail_list:
        print("\nPROBLÈMES CRITIQUES :")
        for r in fail_list:
            print(f"  [FAILED] {r.session_name}/{r.side} — {r.error[:100]}")
        for r in err_list:
            print(f"  [ERROR]  {r.session_name}/{r.side}")
            for a in r.alerts:
                if a.level == "ERROR":
                    print(f"    [{a.code}] {a.message}")


def _build_pinces_parser(sub) -> None:
    p = sub.add_parser(
        "pinces",
        help="Vérifier l'alignement pince/vidéo (timestamps absolus) sur toutes les sessions",
    )
    p.add_argument("root", type=Path,
                   help="Répertoire racine (les sessions session_* sont cherchées récursivement)")
    p.add_argument("--session", default=None,
                   help="Limiter à une seule session (nom du dossier)")
    p.add_argument("--tolerance-offset-ms",  type=float, default=200.0)
    p.add_argument("--tolerance-latency-ms", type=float, default=25.0)
    p.add_argument("--tolerance-jitter-ms",  type=float, default=15.0)
    p.add_argument("--max-frame-drops",      type=int,   default=5)
    p.add_argument("--max-vel-mm-s",         type=float, default=2000.0)
    p.add_argument("--min-overlap-s",        type=float, default=3.0)


def _cmd_pinces(args) -> int:
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[ERREUR] Répertoire introuvable : {root}", file=sys.stderr)
        return 1

    thr = _PincesThresholds(
        offset_ms      = args.tolerance_offset_ms,
        latency_max_ms = args.tolerance_latency_ms,
        jitter_std_ms  = args.tolerance_jitter_ms,
        max_drops      = args.max_frame_drops,
        max_vel_mm_s   = args.max_vel_mm_s,
        min_overlap_s  = args.min_overlap_s,
    )

    # Découverte récursive
    all_session_paths = sorted(
        p.parent for p in root.rglob("metadata.json")
        if p.parent.name.startswith("session_") and "__FAILED" not in str(p.parent)
        and (args.session is None or p.parent.name == args.session)
    )

    if not all_session_paths:
        print(f"[ERREUR] Aucune session trouvée dans {root}", file=sys.stderr)
        return 1

    print(f"Sessions trouvées : {len(all_session_paths)}")
    all_results: List[_SideResult] = []

    for i, spath in enumerate(all_session_paths):
        sname = spath.name
        print(f"  [{i+1:02d}/{len(all_session_paths)}] {sname}", end="", flush=True)

        for side in ("left", "right"):
            r = _process_side_pinces(spath, side, thr)
            all_results.append(r)

            if r.success:
                aln = r.alignment
                sym = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(r.status, "?")
                lat_str = f"{aln.latency_max_abs_ms:.1f}" if np.isfinite(aln.latency_max_abs_ms) else "N/A"
                print(f"  {side}:{sym} off={aln.offset_start_ms:+.0f}ms lat_max={lat_str}ms", end="")
                if r.alerts:
                    codes = " ".join(a.code for a in r.alerts[:2])
                    print(f" [{codes}]", end="")
            else:
                print(f"  {side}:FAILED", end="")

        print()

    _print_pinces_summary(all_results)

    err_list  = [r for r in all_results if r.success and r.status == "ERROR"]
    fail_list = [r for r in all_results if not r.success]
    warn_list = [r for r in all_results if r.success and r.status == "WARNING"]
    if err_list or fail_list:
        return 2
    if warn_list:
        return 1
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 8 — Fix camera offset (fix_camera_offset.py)
# ══════════════════════════════════════════════════════════════════════════════

_CAMERAS_FIX         = ["head", "left", "right"]
_CAM_SYNC_MARKER_KEY = "camera_tracker_sync_applied"
_CAM_OFFSET_THRESHOLD_MS = 500.0


def _read_jsonl_cam(path: Path) -> list:
    with open(path, "rb") as f:
        raw = f.read()
    frames = []
    for part in raw.split(b"\r\n"):
        part = part.strip()
        if len(part) > 5:
            try:
                frames.append(json.loads(part))
            except json.JSONDecodeError:
                pass
    return frames


def _write_jsonl_cam(path: Path, frames: list) -> None:
    lines = [json.dumps(frame, separators=(",", ":")) + "\r\n" for frame in frames]
    with open(path, "wb") as f:
        f.write("".join(lines).encode("utf-8"))


def _read_tracker_window_ms(session_path: Path) -> Tuple[Optional[float], Optional[float]]:
    import csv as _csv
    tracker_path = session_path / "tracker_positions.csv"
    if not tracker_path.exists():
        return None, None
    t_first = t_last = None
    with open(tracker_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            ns_str = row.get("timestamp_ns", "").strip()
            if not ns_str:
                continue
            try:
                t_ms = int(ns_str) / 1_000_000
            except ValueError:
                continue
            if t_first is None:
                t_first = t_ms
            t_last = t_ms
    return t_first, t_last


def _fix_session_camera(session_path: Path, dry_run: bool = False, force: bool = False) -> dict:
    session_name = session_path.name
    meta_path    = session_path / "metadata.json"

    if not meta_path.exists():
        return {"session": session_name, "status": "skipped", "reason": "metadata.json absent"}

    with open(meta_path, "rb") as f:
        meta = json.loads(f.read())

    if not force and meta.get(_CAM_SYNC_MARKER_KEY):
        return {"session": session_name, "status": "skipped",
                "reason": f"déjà corrigée ({_CAM_SYNC_MARKER_KEY}=true)"}

    trk_t0, trk_t1 = _read_tracker_window_ms(session_path)
    if trk_t0 is None:
        return {"session": session_name, "status": "error",
                "reason": "tracker_positions.csv introuvable ou sans timestamp_ns valide"}

    offsets: Dict[str, float] = {}
    cam_frames: Dict[str, list] = {}

    for cam in _CAMERAS_FIX:
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        if not jsonl_path.exists():
            continue
        frames = _read_jsonl_cam(jsonl_path)
        if not frames:
            continue
        cam_frames[cam] = frames
        offsets[cam]    = frames[0]["capture_time"] - trk_t0

    if not offsets:
        return {"session": session_name, "status": "error",
                "reason": "aucun fichier .jsonl trouvé"}

    max_offset = max(abs(v) for v in offsets.values())
    if max_offset < _CAM_OFFSET_THRESHOLD_MS and not force:
        return {
            "session": session_name, "status": "ok",
            "reason": f"offset max={max_offset:.1f}ms < seuil {_CAM_OFFSET_THRESHOLD_MS}ms — déjà alignées",
            "offsets_ms": offsets, "tracker_t0_ms": trk_t0, "tracker_t1_ms": trk_t1,
        }

    report = {
        "session": session_name,
        "status": "dry-run" if dry_run else "corrected",
        "tracker_t0_ms": trk_t0, "tracker_t1_ms": trk_t1,
        "tracker_duration_s": (trk_t1 - trk_t0) / 1000,
        "offsets_ms": offsets, "cameras_fixed": [],
    }

    for cam, offset_ms in offsets.items():
        jsonl_path  = session_path / "videos" / f"{cam}.jsonl"
        frames      = cam_frames[cam]
        offset_int  = round(offset_ms)
        recaled     = [{**fr, "capture_time": fr["capture_time"] - offset_int} for fr in frames]
        truncated   = [fr for fr in recaled if trk_t0 <= fr["capture_time"] <= trk_t1]
        n_removed   = len(recaled) - len(truncated)
        overlap_s   = ((truncated[-1]["capture_time"] - truncated[0]["capture_time"]) / 1000
                       if truncated else 0)

        report["cameras_fixed"].append({
            "camera": cam, "offset_ms": offset_ms, "offset_applied_ms": offset_int,
            "frames_original": len(frames), "frames_kept": len(truncated),
            "frames_removed": n_removed, "overlap_s": round(overlap_s, 2),
            "first_original_ms": frames[0]["capture_time"],
            "first_corrected_ms": truncated[0]["capture_time"] if truncated else None,
            "last_corrected_ms":  truncated[-1]["capture_time"] if truncated else None,
        })

        if not dry_run and truncated:
            bak_path = jsonl_path.with_suffix(".jsonl.bak")
            if not bak_path.exists():
                shutil.copy2(jsonl_path, bak_path)
            _write_jsonl_cam(jsonl_path, truncated)

    if not dry_run:
        meta[_CAM_SYNC_MARKER_KEY]                        = True
        meta["camera_tracker_sync_offsets_ms"]            = offsets
        meta["camera_tracker_sync_tracker_t0_ms"]         = trk_t0
        meta["camera_tracker_sync_tracker_t1_ms"]         = trk_t1
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return report


def _print_fix_camera_report(report: dict) -> None:
    status  = report["status"]
    session = report["session"]
    reason  = report.get("reason", "")
    if status in ("skipped", "ok"):
        print(f"  [{status.upper()}] {session} — {reason}")
        return
    if status == "error":
        print(f"  [ERROR] {session} — {reason}")
        return
    trk_dur = report.get("tracker_duration_s", 0)
    print(f"  [{status.upper()}] {session}  (tracker window={trk_dur:.1f}s)")
    for c in report.get("cameras_fixed", []):
        cam     = c["camera"]
        off     = c["offset_ms"]
        kept    = c["frames_kept"]
        removed = c["frames_removed"]
        total   = c["frames_original"]
        overlap = c["overlap_s"]
        pct     = kept / total * 100 if total else 0
        print(f"    {cam}: offset={off:+.0f}ms  "
              f"kept={kept}/{total} ({pct:.0f}%)  "
              f"removed={removed}  overlap={overlap:.1f}s")


def _build_fix_camera_parser(sub) -> None:
    p = sub.add_parser(
        "fix-camera",
        help="Recaler et tronquer les capture_time caméras sur la fenêtre tracker",
    )
    p.add_argument("root", type=Path,
                   help="Répertoire racine (sessions cherchées récursivement)")
    p.add_argument("--dry-run", action="store_true",
                   help="Afficher les corrections sans modifier les fichiers")
    p.add_argument("--force", action="store_true",
                   help="Forcer la correction même si déjà appliquée")
    p.add_argument("--session", default=None,
                   help="Limiter à une seule session (nom du dossier)")


def _cmd_fix_camera(args) -> int:
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[ERREUR] Répertoire introuvable : {root}", file=sys.stderr)
        return 1

    sessions = sorted(
        p.parent for p in root.rglob("metadata.json")
        if (p.parent / "videos").exists()
        and (args.session is None or p.parent.name == args.session)
    )

    if not sessions:
        print(f"[ERREUR] Aucune session trouvée dans {root}", file=sys.stderr)
        return 1

    prefix = "[DRY-RUN] " if args.dry_run else ""
    print(f"{prefix}Traitement de {len(sessions)} session(s)...\n")

    n_corrected = n_skipped = n_errors = 0
    for session_path in sessions:
        report = _fix_session_camera(session_path, dry_run=args.dry_run, force=args.force)
        _print_fix_camera_report(report)
        if report["status"] in ("corrected", "dry-run"):
            n_corrected += 1
        elif report["status"] == "error":
            n_errors += 1
        else:
            n_skipped += 1

    print(f"\n  Corrigées : {n_corrected}  Ignorées : {n_skipped}  Erreurs : {n_errors}")
    if args.dry_run:
        print("  Mode dry-run : aucun fichier modifié.")
    return 0 if n_errors == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="sync.py — moteur de synchronisation temporelle complet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _build_heuristic_parser(sub)
    _build_train_parser(sub)
    _build_apply_parser(sub)
    _build_score_parser(sub)
    _build_pinces_parser(sub)
    _build_fix_camera_parser(sub)

    args = parser.parse_args()
    if args.cmd == "heuristic":
        return _cmd_heuristic(args)
    elif args.cmd == "train":
        return _cmd_train(args)
    elif args.cmd == "apply":
        return _cmd_apply(args)
    elif args.cmd == "score":
        return _cmd_score(args)
    elif args.cmd == "pinces":
        return _cmd_pinces(args)
    elif args.cmd == "fix-camera":
        return _cmd_fix_camera(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
