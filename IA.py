#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alignement inter-flux avancé par deep learning auto-supervisé.

Idée :
- chaque flux est transformé en signal 1D robuste
- on découpe en fenêtres
- on entraîne un encodeur cross-modal à reconnaître les fenêtres alignées
- à l'inférence, on balaie les lags et on prend celui qui maximise le score modèle
- on refuse les cas peu fiables

Paires :
  tracker_head   ↔ cam_head
  tracker_left   ↔ cam_left
  tracker_right  ↔ cam_right
  tracker_left   ↔ gripper_left
  tracker_right  ↔ gripper_right

Entrées attendues :
  session_xxx/
    metadata.json
    tracker_positions.csv
    gripper_left_data.csv
    gripper_right_data.csv
    videos/
      head.jsonl
      left.jsonl
      right.jsonl
      head_flux.csv
      left_flux.csv
      right_flux.csv

Le fichier *_flux.csv doit contenir au moins :
  time_seconds
  timestamp_abs_ms   (fortement recommandé)
  motion_mean_smooth ou motion_mean
  diff_mean_smooth ou diff_mean

Usage :
  python sync_ml_advanced.py /path/to/root --train
  python sync_ml_advanced.py /path/to/root --train --apply
  python sync_ml_advanced.py /path/to/root --session session_20260323_143810 --plot

Ce script :
- entraîne un modèle global
- estime les offsets
- peut corriger les fichiers en place avec --apply

Important :
- sans pseudo-labels fiables, aucun ML ne sauvera la data
- le script refuse d'appliquer les offsets si la confiance est faible
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import multiprocessing
import os
import sys
sys.path.insert(0, str(Path(__file__).parent))
from check import check_session, MIN_IA_SCORE as _CHECK_MIN_IA_SCORE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

PAIRS: List[Tuple[str, str]] = [
    ("tracker_head",  "cam_head"),
    ("tracker_left",  "cam_left"),
    ("tracker_right", "cam_right"),
    ("tracker_left",  "gripper_left"),
    ("tracker_right", "gripper_right"),
]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Utilise tous les CPU disponibles pour PyTorch (opérations matricielles)
_N_CPU = os.cpu_count() or 1
torch.set_num_threads(_N_CPU)
torch.set_num_interop_threads(max(1, _N_CPU // 2))

# Nombre de workers pour le DataLoader et les pools multiprocessing
_DATALOADER_WORKERS = min(4, max(0, _N_CPU - 1))

RESAMPLE_MS = 5.0
MAX_LAG_MS = 400.0
WINDOW_MS = 2200.0
WINDOW_STRIDE_MS = 350.0
MIN_OVERLAP_MS = 1500.0

PSEUDO_POS_THR = 0.72
PSEUDO_NEG_THR = 0.30
EDGE_MARGIN_MS = 20.0

MIN_CONFIDENCE_TO_APPLY = _CHECK_MIN_IA_SCORE  # aligné sur check.py
MIN_PEAK_MARGIN = 0.06
MIN_PAIR_WINDOWS = 10

TRAIN_EPOCHS = 18
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

MODEL_DIRNAME = "_sync_ml_model"
RESULTS_JSON  = "sync_ml_advanced_results.json"

# Espace de travail fixe — toutes les sessions sont dans ce répertoire
ROOT_DIR = Path("/Users/christopher/Downloads/sync_test_1/treatment/")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Alignement inter-flux par deep learning — espace de travail : {ROOT_DIR}"
    )
    p.add_argument("--session",    type=str,   default=None,        help="Traiter une seule session (nom, pas chemin complet).")
    p.add_argument("--train",      action="store_true",              help="Entraîne le modèle avant estimation.")
    p.add_argument("--apply",      action="store_true",              help="Applique les offsets estimés en place.")
    p.add_argument("--plot",       action="store_true",              help="Génère les graphes.")
    p.add_argument("--force",      action="store_true",              help="Ignore les marqueurs existants.")
    p.add_argument("--max-lag-ms", type=float, default=MAX_LAG_MS)
    p.add_argument("--window-ms",  type=float, default=WINDOW_MS)
    p.add_argument("--resample-ms",type=float, default=RESAMPLE_MS)
    p.add_argument("--epochs",     type=int,   default=TRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--dry-run",    action="store_true",              help="Calcule seulement, n'écrit rien.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Flux:
    name: str
    t_ms_rel: np.ndarray
    signal: np.ndarray
    t_start_abs_ms: float
    source: str

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
    is_reliable: bool
    method: str
    lags_ms: np.ndarray
    scores: np.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-8:
        return np.nan_to_num(x - m, nan=0.0)
    return np.nan_to_num((x - m) / s, nan=0.0)

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

def smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
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
                obj = json.loads(line)
                return float(obj["capture_time"])
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
                idx = int(obj["index"])
                t = float(obj["capture_time"])
                out[idx] = t
            except Exception:
                continue
    return out

def _shift_iso(series: pd.Series, delta_ns: int) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    shifted = ts + pd.to_timedelta(delta_ns, unit="ns")
    return shifted.dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des flux
# ──────────────────────────────────────────────────────────────────────────────

def tracker_speed_features(df: pd.DataFrame, pos: str, dt_ms_nominal: float,
                            custom_cols: Optional[List[str]] = None) -> np.ndarray:
    """Construit le signal d'alignement pour un tracker.

    Si custom_cols est fourni (liste de colonnes du CSV), on combine ces colonnes
    en signal zscore. Sinon, on utilise les colonnes x/y/z pour calculer vitesse/accél/jerk.
    """
    if custom_cols:
        available = [c for c in custom_cols if c in df.columns]
        if not available:
            return np.zeros(len(df), dtype=np.float32)
        parts = []
        for col in available:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float32)
            parts.append(zscore(robust_clip(np.nan_to_num(v))))
        sig = np.mean(parts, axis=0).astype(np.float32)
        sig = smooth(sig, 2.0)
        return sig

    cols = [f"tracker_{pos}_x", f"tracker_{pos}_y", f"tracker_{pos}_z"]
    if not all(c in df.columns for c in cols):
        return np.zeros(len(df), dtype=np.float32)

    x = pd.to_numeric(df[cols[0]], errors="coerce").to_numpy(np.float32)
    y = pd.to_numeric(df[cols[1]], errors="coerce").to_numpy(np.float32)
    z = pd.to_numeric(df[cols[2]], errors="coerce").to_numpy(np.float32)

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dz = np.diff(z, prepend=z[0])

    speed = np.sqrt(dx * dx + dy * dy + dz * dz) / max(dt_ms_nominal / 1000.0, 1e-6)
    speed = smooth(speed, 2.0)
    accel = moving_derivative(speed, dt_ms_nominal)
    jerk = moving_derivative(accel, dt_ms_nominal)

    feat = (
        0.55 * zscore(robust_clip(speed)) +
        0.30 * zscore(robust_clip(np.abs(accel))) +
        0.15 * zscore(robust_clip(np.abs(jerk)))
    )
    feat = smooth(feat, 2.0)
    return feat.astype(np.float32)

def load_trackers(session_dir: Path, signal_config: Optional[Dict] = None) -> Dict[str, Flux]:
    """Charge les trackers.

    signal_config peut contenir des clés "tracker_head", "tracker_left", "tracker_right"
    avec une liste de colonnes CSV à utiliser comme signal (ex: ["tracker_head_x", "tracker_head_z"]).
    Si absent, comportement par défaut (speed/accel/jerk depuis x/y/z).
    """
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
        sig = tracker_speed_features(df, pos, dt_nom, custom_cols=custom_cols)
        out[f"tracker_{pos}"] = Flux(
            name=f"tracker_{pos}",
            t_ms_rel=t_ms_rel.astype(np.float32),
            signal=sig,
            t_start_abs_ms=float(t_start_abs_ms),
            source="tracker_positions.csv",
        )
    return out

def build_video_signal(df: pd.DataFrame, custom_cols: Optional[List[str]] = None) -> Optional[np.ndarray]:
    """Construit le signal vidéo.

    Si custom_cols est fourni, on moyenne ces colonnes. Sinon, pondération automatique
    des colonnes motion_*/diff_* disponibles.
    """
    if custom_cols:
        available = [c for c in custom_cols if c in df.columns]
        if not available:
            return None
        parts = []
        for col in available:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float32)
            parts.append(zscore(robust_clip(np.nan_to_num(x))))
        sig = np.mean(parts, axis=0)
        sig = smooth(sig, 1.5)
        ds = moving_derivative(sig, 33.0)
        sig = 0.8 * zscore(sig) + 0.2 * zscore(np.abs(ds))
        return sig.astype(np.float32)

    candidates = [
        ("motion_p90_smooth", 0.40),
        ("motion_mean_smooth", 0.25),
        ("diff_mean_smooth", 0.20),
        ("motion_p90", 0.10),
        ("motion_mean", 0.03),
        ("diff_mean", 0.02),
    ]
    parts = []
    for col, w in candidates:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float32)
            x = zscore(robust_clip(x))
            parts.append(w * x)

    if not parts:
        return None

    sig = np.sum(parts, axis=0)
    sig = smooth(sig, 1.5)
    ds = moving_derivative(sig, 33.0)
    sig = 0.8 * zscore(sig) + 0.2 * zscore(np.abs(ds))
    return sig.astype(np.float32)

def load_cameras(session_dir: Path, signal_config: Optional[Dict] = None) -> Dict[str, Flux]:
    out = {}
    for cam in ("head", "left", "right"):
        flux_csv = session_dir / "videos" / f"{cam}_flux.csv"
        jsonl_path = session_dir / "videos" / f"{cam}.jsonl"
        if not flux_csv.exists():
            continue

        df = pd.read_csv(flux_csv)
        custom_cols = (signal_config or {}).get(f"cam_{cam}")
        sig = build_video_signal(df, custom_cols=custom_cols)
        if sig is None:
            continue

        if "timestamp_abs_ms" in df.columns and pd.to_numeric(df["timestamp_abs_ms"], errors="coerce").notna().any():
            t_abs = pd.to_numeric(df["timestamp_abs_ms"], errors="coerce").to_numpy(np.float64)
            valid = np.isfinite(t_abs)
            if valid.sum() < 2:
                continue
            first = float(t_abs[valid][0])
            t_ms_rel = t_abs - first
            t_start_abs_ms = first
        else:
            t_s = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy(np.float64)
            t_ms_rel = (t_s - t_s[0]) * 1000.0
            anchor = first_jsonl_capture_time(jsonl_path)
            t_start_abs_ms = float(anchor) if anchor is not None else 0.0

        valid = np.isfinite(t_ms_rel) & np.isfinite(sig)
        t_ms_rel = t_ms_rel[valid]
        sig = sig[valid]
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
    """Charge les grippers.

    signal_config peut contenir "gripper_left" / "gripper_right" avec une liste de colonnes.
    Si absent, utilise angle_deg (dérivée 1e+2e ordre).
    """
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
            parts = []
            for col in available:
                v = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float32)
                parts.append(zscore(robust_clip(np.nan_to_num(v))))
            sig = smooth(np.mean(parts, axis=0), 1.5)
        else:
            if "angle_deg" not in df.columns:
                continue
            angle = pd.to_numeric(df["angle_deg"], errors="coerce").to_numpy(np.float32)
            d1 = np.diff(angle, prepend=angle[0])
            d2 = np.diff(d1, prepend=d1[0])
            sig = 0.75 * zscore(np.abs(d1)) + 0.25 * zscore(np.abs(d2))
            sig = smooth(sig, 1.5)

        valid = np.isfinite(t_ms_rel) & np.isfinite(sig)
        t_ms_rel = t_ms_rel[valid]
        sig = sig[valid]
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


# ──────────────────────────────────────────────────────────────────────────────
# Fenêtres et pseudo-labels
# ──────────────────────────────────────────────────────────────────────────────

def make_common_grid(ref: Flux, tgt: Flux, delta_start_ms: float, extra_shift_ms: float, resample_ms: float):
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

def heuristic_alignment_score(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8:
        return 0.0

    a1 = zscore(a)
    b1 = zscore(b)

    corr = float(np.clip(np.corrcoef(a1, b1)[0, 1], -1.0, 1.0)) if np.std(a1) > 1e-6 and np.std(b1) > 1e-6 else 0.0

    ea = smooth(a1 * a1, 2.0)
    eb = smooth(b1 * b1, 2.0)
    ecorr = float(np.clip(np.corrcoef(ea, eb)[0, 1], -1.0, 1.0)) if np.std(ea) > 1e-6 and np.std(eb) > 1e-6 else 0.0

    pa, _ = find_peaks(a1, height=np.percentile(a1, 75))
    pb, _ = find_peaks(b1, height=np.percentile(b1, 75))
    if len(pa) and len(pb):
        match = sum(1 for p in pa if np.any(np.abs(pb - p) <= 4)) / max(len(pa), 1)
    else:
        match = 0.0

    fa = np.abs(np.fft.rfft(a1))
    fb = np.abs(np.fft.rfft(b1))
    scorr = float(np.clip(np.corrcoef(fa, fb)[0, 1], -1.0, 1.0)) if np.std(fa) > 1e-6 and np.std(fb) > 1e-6 else 0.0

    score = 0.42 * max(corr, 0.0) + 0.28 * max(ecorr, 0.0) + 0.20 * match + 0.10 * max(scorr, 0.0)
    return float(np.clip(score, 0.0, 1.0))

def build_pseudo_examples_for_pair(
    ref: Flux,
    tgt: Flux,
    resample_ms: float,
    max_lag_ms: float,
    window_ms: float,
) -> List[Tuple[np.ndarray, np.ndarray, int, str]]:
    examples = []
    delta_start_ms = tgt.t_start_abs_ms - ref.t_start_abs_ms

    candidate_lags = np.arange(-max_lag_ms, max_lag_ms + resample_ms, resample_ms, dtype=np.float32)
    win = int(window_ms / resample_ms)
    stride = max(4, int(WINDOW_STRIDE_MS / resample_ms))

    for lag in candidate_lags:
        grid, a, b = make_common_grid(ref, tgt, delta_start_ms, float(lag), resample_ms)
        if grid is None:
            continue

        for s, e in window_slices(len(grid), win, stride):
            wa = a[s:e]
            wb = b[s:e]
            score = heuristic_alignment_score(wa, wb)

            is_edge = abs(float(lag)) >= (max_lag_ms - EDGE_MARGIN_MS)
            if is_edge:
                continue

            if score >= PSEUDO_POS_THR:
                examples.append((wa, wb, 1, f"{ref.name}|{tgt.name}"))
            elif score <= PSEUDO_NEG_THR:
                examples.append((wa, wb, 0, f"{ref.name}|{tgt.name}"))

    return examples


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PairWindowDataset(Dataset):
    def __init__(self, examples: List[Tuple[np.ndarray, np.ndarray, int, str]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        a, b, y, pair_name = self.examples[idx]
        a = zscore(robust_clip(a))
        b = zscore(robust_clip(b))

        da = moving_derivative(a, RESAMPLE_MS)
        db = moving_derivative(b, RESAMPLE_MS)

        ea = smooth(a * a, 2.0)
        eb = smooth(b * b, 2.0)

        xa = np.stack([a, da, ea], axis=0).astype(np.float32)
        xb = np.stack([b, db, eb], axis=0).astype(np.float32)

        return torch.from_numpy(xa), torch.from_numpy(xb), torch.tensor(y, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Modèle
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, s=1, p=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=s, padding=p),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, k, stride=1, padding=p),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)

class Encoder1D(nn.Module):
    def __init__(self, in_ch=3, emb=128):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_ch, 32),
            nn.MaxPool1d(2),
            ConvBlock(32, 64),
            nn.MaxPool1d(2),
            ConvBlock(64, 96),
            nn.MaxPool1d(2),
            ConvBlock(96, 128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, emb),
            nn.GELU(),
            nn.Linear(emb, emb),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        return x

class CrossModalAligner(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.enc_ref = Encoder1D(in_ch=3, emb=emb)
        self.enc_tgt = Encoder1D(in_ch=3, emb=emb)
        self.head = nn.Sequential(
            nn.Linear(emb * 4, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, xa, xb):
        ea = self.enc_ref(xa)
        eb = self.enc_tgt(xb)
        feat = torch.cat([ea, eb, torch.abs(ea - eb), ea * eb], dim=-1)
        logit = self.head(feat).squeeze(-1)
        return logit, ea, eb


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def contrastive_margin(ea, eb, y, margin=0.6):
    dist = torch.norm(ea - eb, dim=-1)
    pos = y * dist.pow(2)
    neg = (1 - y) * F.relu(margin - dist).pow(2)
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

    n_batches = len(loader)
    n_samples = len(loader.dataset)
    best_loss = float("inf")
    history = []

    print(f"\n{'='*70}")
    print(f"[train] Démarrage entraînement")
    print(f"[train]   epochs       : {epochs}")
    print(f"[train]   batches/epoch: {n_batches}")
    print(f"[train]   samples      : {n_samples}")
    print(f"[train]   batch_size   : {loader.batch_size}")
    print(f"[train]   lr_max       : {lr}")
    print(f"[train]   device       : {DEVICE}")
    print(f"{'='*70}\n")

    t_train_start = time.time()

    for epoch in range(epochs):
        t_epoch_start = time.time()
        model.train()

        losses_total = []
        losses_bce   = []
        losses_ctr   = []
        n_correct = 0
        n_seen    = 0

        iter_loader = _tqdm(loader, desc=f"  epoch {epoch+1:02d}/{epochs}", leave=False) if _has_tqdm else loader

        for batch_idx, (xa, xb, y) in enumerate(iter_loader):
            xa = xa.to(DEVICE)
            xb = xb.to(DEVICE)
            y  = y.to(DEVICE)

            opt.zero_grad()
            logit, ea, eb = model(xa, xb)

            bce = F.binary_cross_entropy_with_logits(logit, y)
            ctr = contrastive_margin(ea, eb, y)
            loss = 0.75 * bce + 0.25 * ctr

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            losses_total.append(float(loss.item()))
            losses_bce.append(float(bce.item()))
            losses_ctr.append(float(ctr.item()))

            with torch.no_grad():
                preds = (torch.sigmoid(logit) >= 0.5).float()
                n_correct += int((preds == y).sum().item())
                n_seen    += int(y.size(0))

        t_epoch = time.time() - t_epoch_start
        lr_now  = scheduler.get_last_lr()[0]

        mean_loss = float(np.mean(losses_total))
        mean_bce  = float(np.mean(losses_bce))
        mean_ctr  = float(np.mean(losses_ctr))
        acc       = n_correct / max(n_seen, 1) * 100.0
        improved  = "*" if mean_loss < best_loss else " "
        if mean_loss < best_loss:
            best_loss = mean_loss

        history.append({
            "epoch": epoch + 1,
            "loss": mean_loss,
            "bce": mean_bce,
            "ctr": mean_ctr,
            "acc": acc,
            "lr": lr_now,
        })

        print(
            f"[train] epoch {epoch+1:02d}/{epochs}{improved} "
            f"loss={mean_loss:.4f}  bce={mean_bce:.4f}  ctr={mean_ctr:.4f}  "
            f"acc={acc:5.1f}%  lr={lr_now:.2e}  t={t_epoch:.1f}s"
        )

    t_total = time.time() - t_train_start
    best_epoch = min(history, key=lambda h: h["loss"])
    print(f"\n{'='*70}")
    print(f"[train] Entraînement terminé en {t_total:.1f}s")
    print(f"[train] Meilleure époch : {best_epoch['epoch']:02d}  "
          f"loss={best_epoch['loss']:.4f}  acc={best_epoch['acc']:.1f}%")
    print(f"[train] Acc finale      : {history[-1]['acc']:.1f}%")
    print(f"[train] Loss finale     : {history[-1]['loss']:.4f}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entraînement global
# ──────────────────────────────────────────────────────────────────────────────

def _session_is_clean(session_dir: Path) -> bool:
    """Délègue à check_session — exclut les sessions avec score < 40% (bloquées ou très mauvaises)."""
    report = check_session(session_dir, model=None)
    return report.score >= 40.0 or not report.is_blocked()


def discover_sessions(root: Path, only_session: Optional[str]) -> List[Path]:
    """
    Découvre les sessions dans ROOT_DIR.
    Seules les sessions nettoyées (labels vérifiés + tracker validé) sont retournées.
    """
    all_sessions = sorted(
        p.parent
        for p in root.rglob("metadata.json")
        if p.parent.name.startswith("session_")
    )

    if only_session:
        all_sessions = [s for s in all_sessions if s.name == only_session]

    clean     = [s for s in all_sessions if _session_is_clean(s)]
    excluded  = len(all_sessions) - len(clean)

    if excluded > 0:
        print(
            f"[discover] {len(all_sessions)} sessions trouvées — "
            f"{excluded} exclues (non nettoyées) — "
            f"{len(clean)} utilisées pour l'entraînement"
        )
    else:
        print(f"[discover] {len(clean)} sessions prêtes pour l'entraînement")

    return clean

def load_all_fluxes(session_dir: Path, signal_config: Optional[Dict] = None) -> Dict[str, Flux]:
    """signal_config : dict optionnel {flux_name: [col1, col2, ...]} pour chaque flux."""
    out = {}
    out.update(load_trackers(session_dir, signal_config=signal_config))
    out.update(load_cameras(session_dir, signal_config=signal_config))
    out.update(load_grippers(session_dir, signal_config=signal_config))
    return out

def _build_examples_for_session(args_tuple):
    """Worker multiprocessing pour une session (doit être top-level pour pickle)."""
    session_dir, resample_ms, max_lag_ms, window_ms, signal_config = args_tuple
    fluxes = load_all_fluxes(session_dir, signal_config=signal_config)
    session_examples = []
    for ref_name, tgt_name in PAIRS:
        if ref_name not in fluxes or tgt_name not in fluxes:
            continue
        ex = build_pseudo_examples_for_pair(
            fluxes[ref_name],
            fluxes[tgt_name],
            resample_ms=resample_ms,
            max_lag_ms=max_lag_ms,
            window_ms=window_ms,
        )
        if ex:
            session_examples.append((session_dir.name, ref_name, tgt_name, ex))
    return session_examples


def build_training_examples(sessions: List[Path], resample_ms: float, max_lag_ms: float,
                             window_ms: float, signal_config: Optional[Dict] = None):
    worker_args = [
        (session_dir, resample_ms, max_lag_ms, window_ms, signal_config)
        for session_dir in sessions
    ]

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


# ──────────────────────────────────────────────────────────────────────────────
# Inférence offset
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_window_tensors(windows_a: List[np.ndarray], windows_b: List[np.ndarray]):
    """Pré-calcul numpy des features (zscore, dérivée, énergie) — parallélisable."""
    batch_xa = []
    batch_xb = []
    for a, b in zip(windows_a, windows_b):
        a = zscore(robust_clip(a))
        b = zscore(robust_clip(b))
        da = moving_derivative(a, RESAMPLE_MS)
        db = moving_derivative(b, RESAMPLE_MS)
        ea = smooth(a * a, 2.0)
        eb = smooth(b * b, 2.0)
        batch_xa.append(np.stack([a, da, ea], axis=0).astype(np.float32))
        batch_xb.append(np.stack([b, db, eb], axis=0).astype(np.float32))
    return np.stack(batch_xa), np.stack(batch_xb)


@torch.no_grad()
def score_windows_with_model(model: CrossModalAligner, windows_a: List[np.ndarray],
                              windows_b: List[np.ndarray], chunk_size: int = 256) -> np.ndarray:
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

def estimate_pair_offset(
    model: CrossModalAligner,
    ref: Flux,
    tgt: Flux,
    resample_ms: float,
    max_lag_ms: float,
    window_ms: float,
) -> PairEstimate:
    delta_start_ms = tgt.t_start_abs_ms - ref.t_start_abs_ms
    candidate_lags = np.arange(-max_lag_ms, max_lag_ms + resample_ms, resample_ms, dtype=np.float32)

    win = int(window_ms / resample_ms)
    stride = max(4, int(WINDOW_STRIDE_MS / resample_ms))

    lag_scores = []
    valid_lags = []

    for lag in candidate_lags:
        if abs(float(lag)) >= (max_lag_ms - EDGE_MARGIN_MS):
            continue

        grid, a, b = make_common_grid(ref, tgt, delta_start_ms, float(lag), resample_ms)
        if grid is None:
            continue

        slices = window_slices(len(grid), win, stride)
        if len(slices) < MIN_PAIR_WINDOWS:
            continue

        wa_list = [a[s:e] for s, e in slices]
        wb_list = [b[s:e] for s, e in slices]

        proba = score_windows_with_model(model, wa_list, wb_list)
        if len(proba) == 0:
            continue

        q75 = float(np.percentile(proba, 75))
        mean = float(np.mean(proba))
        score = 0.65 * q75 + 0.35 * mean

        lag_scores.append(score)
        valid_lags.append(float(lag))

    if not valid_lags:
        return PairEstimate(
            ref_name=ref.name,
            tgt_name=tgt.name,
            delta_start_ms=float(delta_start_ms),
            residual_ms=0.0,
            total_offset_ms=float(delta_start_ms),
            shift_to_apply_ms=float(-delta_start_ms),
            confidence=0.0,
            peak_margin=0.0,
            best_score=0.0,
            second_score=0.0,
            is_reliable=False,
            method="deep-no-valid-lag",
            lags_ms=np.array([], dtype=np.float32),
            scores=np.array([], dtype=np.float32),
        )

    scores = np.array(lag_scores, dtype=np.float32)
    lags = np.array(valid_lags, dtype=np.float32)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    tmp = scores.copy()
    tmp[best_idx] = -1e9
    second_score = float(np.max(tmp)) if len(tmp) > 1 else 0.0
    peak_margin = best_score - second_score

    best_residual_ms = float(lags[best_idx])
    total_offset_ms = float(delta_start_ms + best_residual_ms)

    confidence = float(np.clip(best_score, 0.0, 1.0))
    reliable = (confidence >= MIN_CONFIDENCE_TO_APPLY) and (peak_margin >= MIN_PEAK_MARGIN)

    return PairEstimate(
        ref_name=ref.name,
        tgt_name=tgt.name,
        delta_start_ms=float(delta_start_ms),
        residual_ms=best_residual_ms,
        total_offset_ms=total_offset_ms,
        shift_to_apply_ms=float(-total_offset_ms),
        confidence=confidence,
        peak_margin=float(peak_margin),
        best_score=best_score,
        second_score=second_score,
        is_reliable=bool(reliable),
        method="deep-contrastive",
        lags_ms=lags,
        scores=scores,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Écriture offsets
# ──────────────────────────────────────────────────────────────────────────────

def backup_file(path: Path):
    backup = path.with_suffix(path.suffix + ".bak_syncml")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())

def apply_shift_to_target(session_dir: Path, target_name: str, shift_ms: float, dry_run: bool):
    """
    Convention unique :
      new_time = old_time + shift_ms
    """

    if abs(shift_ms) < 0.1:
        print(f"    {target_name:<20}  shift={shift_ms:+.2f} ms  ignoré")
        return

    delta_ns = int(round(shift_ms * 1_000_000))
    delta_s = shift_ms / 1000.0

    if target_name.startswith("cam_"):
        cam = target_name.replace("cam_", "")
        jsonl_path = session_dir / "videos" / f"{cam}.jsonl"
        flux_path = session_dir / "videos" / f"{cam}_flux.csv"

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


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_pair_estimate(session_dir: Path, est: PairEstimate):
    if len(est.lags_ms) == 0:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(est.lags_ms, est.scores, lw=2)
    plt.axvline(est.residual_ms, color="red", linestyle="--", lw=2, label=f"best={est.residual_ms:+.1f} ms")
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
        "dry_run": dry_run,
        "pairs": [
            {
                "ref": e.ref_name,
                "target": e.tgt_name,
                "delta_start_ms": e.delta_start_ms,
                "residual_ms": e.residual_ms,
                "total_offset_ms": e.total_offset_ms,
                "shift_to_apply_ms": e.shift_to_apply_ms,
                "confidence": e.confidence,
                "peak_margin": e.peak_margin,
                "is_reliable": e.is_reliable,
                "method": e.method,
            }
            for e in estimates
        ],
    }
    (session_dir / RESULTS_JSON).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def train_pipeline(root: Path, sessions: List[Path], args):
    import time
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"[pipeline] Construction des pseudo-exemples")
    print(f"[pipeline]   sessions    : {len(sessions)}")
    print(f"[pipeline]   resample_ms : {args.resample_ms}")
    print(f"[pipeline]   max_lag_ms  : {args.max_lag_ms}")
    print(f"[pipeline]   window_ms   : {args.window_ms}")
    print(f"{'='*70}")

    examples = build_training_examples(
        sessions=sessions,
        resample_ms=args.resample_ms,
        max_lag_ms=args.max_lag_ms,
        window_ms=args.window_ms,
    )
    if len(examples) < 200:
        raise RuntimeError(f"Pas assez d'exemples pseudo-labelisés : {len(examples)}")

    pos = sum(y for _, _, y, _ in examples)
    neg = len(examples) - pos
    ratio = pos / max(neg, 1)
    print(f"\n[pseudo] total={len(examples)}  pos={pos}  neg={neg}  ratio_pos/neg={ratio:.2f}")
    if ratio < 0.3 or ratio > 3.0:
        print(f"[pseudo] AVERTISSEMENT : déséquilibre important pos/neg={ratio:.2f} — l'entraînement peut être biaisé")
    t_build = time.time() - t0
    print(f"[pseudo] Construction en {t_build:.1f}s")

    ds = PairWindowDataset(examples)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=_DATALOADER_WORKERS,
        persistent_workers=_DATALOADER_WORKERS > 0,
        pin_memory=False,
    )

    model = CrossModalAligner().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Paramètres entraînables : {n_params:,}")

    train_model(model, dl, epochs=args.epochs, lr=args.lr, weight_decay=WEIGHT_DECAY)

    model_dir = root / MODEL_DIRNAME
    save_model(model, model_dir)
    print(f"[model] saved to {model_dir}")

def estimate_session(root: Path, session_dir: Path, args):
    model_dir = root / MODEL_DIRNAME
    if not (model_dir / "model.pt").exists():
        raise RuntimeError("Modèle absent. Lance d'abord avec --train")

    model = load_model(model_dir)
    fluxes = load_all_fluxes(session_dir)

    estimates: List[PairEstimate] = []
    for ref_name, tgt_name in PAIRS:
        if ref_name not in fluxes or tgt_name not in fluxes:
            continue
        est = estimate_pair_offset(
            model=model,
            ref=fluxes[ref_name],
            tgt=fluxes[tgt_name],
            resample_ms=args.resample_ms,
            max_lag_ms=args.max_lag_ms,
            window_ms=args.window_ms,
        )
        estimates.append(est)

        print(
            f"{session_dir.name} | {ref_name:<16} ↔ {tgt_name:<16} "
            f"Δstart={est.delta_start_ms:+7.1f}  resid={est.residual_ms:+7.1f}  "
            f"shift={est.shift_to_apply_ms:+7.1f} ms  conf={est.confidence:.3f}  "
            f"margin={est.peak_margin:.3f}  reliable={est.is_reliable}"
        )

        if args.plot:
            plot_pair_estimate(session_dir, est)

    save_session_results(session_dir, estimates, args.dry_run)

    if args.apply and not args.dry_run:
        pre_report = check_session(session_dir, model=None)
        if pre_report.verdict == "FAIL":
            failed = [g for g in pre_report.gates if not g.passed]
            details = "\n".join(f"         ✗ {g.name}: {g.message}" for g in failed)
            print(
                f"[apply] {session_dir.name} — BLOQUÉ (score={pre_report.score:.0f}%)\n"
                + (details if details else f"         {pre_report.blocking_reason}")
            )
        else:
            applied_targets = set()
            for est in estimates:
                if not est.is_reliable:
                    continue
                if est.tgt_name in applied_targets:
                    continue
                apply_shift_to_target(session_dir, est.tgt_name, est.shift_to_apply_ms, dry_run=False)
                applied_targets.add(est.tgt_name)

    return estimates

def main() -> int:
    set_seed()
    args = parse_args()

    root = ROOT_DIR
    if not root.exists():
        print(f"ERREUR: ROOT_DIR introuvable: {root}", file=sys.stderr)
        return 1

    sessions = discover_sessions(root, args.session)
    if not sessions:
        print("ERREUR: aucune session trouvée", file=sys.stderr)
        return 1

    print(f"device      : {DEVICE}")
    print(f"root        : {root}")
    print(f"sessions    : {len(sessions)}")
    print(f"resample_ms : {args.resample_ms}")
    print(f"max_lag_ms  : {args.max_lag_ms}")
    print(f"window_ms   : {args.window_ms}")

    if args.train:
        train_pipeline(root, sessions, args)

    for session_dir in sessions:
        estimate_session(root, session_dir, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
