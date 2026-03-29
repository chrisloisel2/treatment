#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check.py — Script central de vérification de synchronisation temporelle.

S'entraîne sur les datasets sync/ (ground truth alignés) et desync/ (désalignés),
puis vérifie toute session en produisant un score de synchronisation 0–100%.

Score = 0% si une vérification critique échoue (fichiers manquants, quaternions
        corrompus, coverage insuffisante, trop de gaps).
      = score IA × pénalités qualité (continuité, offset) sinon.

Usage :
    # Entraîner le modèle sur sync/ et desync/
    python check.py train

    # Vérifier une session unique
    python check.py check /path/to/session

    # Vérifier toutes les sessions d'un dossier
    python check.py check /path/to/root --batch

    # Générer un rapport JSON
    python check.py check /path/to/root --batch --json-out results.json

    # Tout en une passe (train + check)
    python check.py all
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Dépendances optionnelles ───────────────────────────────────────────────────
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from scipy.ndimage import gaussian_filter1d
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from scipy.signal import find_peaks as _find_peaks
    def find_peaks(x, height=None):
        return _find_peaks(x, height=height)
except ImportError:
    def find_peaks(x, height=None):
        peaks = [i for i in range(1, len(x) - 1) if x[i] > x[i-1] and x[i] > x[i+1]]
        if height is not None:
            peaks = [p for p in peaks if x[p] >= height]
        return np.array(peaks), {}

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
SYNC_DIR      = ROOT / "sync"
DESYNC_DIR    = ROOT / "desync"
MODEL_DIR     = ROOT / "_check_model"
MODEL_PATH    = MODEL_DIR / "check_model.pt"

# ── Hyper-paramètres signal ────────────────────────────────────────────────────
RESAMPLE_MS       = 10.0
MAX_LAG_MS        = 500.0
WINDOW_MS         = 2000.0       # fenêtre plus large pour mieux capturer la corrélation
WINDOW_STRIDE_MS  = 500.0        # stride plus serré = plus d'exemples
MIN_OVERLAP_MS    = 800.0
EDGE_MARGIN_MS    = 30.0

PSEUDO_POS_THR    = 0.72         # seuil heuristique pour pseudo-positifs (non utilisé si gt_label fourni)
PSEUDO_NEG_THR    = 0.22         # seuil heuristique pour pseudo-négatifs (non utilisé si gt_label fourni)

# ── Hyper-paramètres entraînement ──────────────────────────────────────────────
TRAIN_EPOCHS      = 24           # plus d'epochs pour converger
BATCH_SIZE        = 64
LR                = 8e-4
WEIGHT_DECAY      = 1e-4
EMB               = 128          # embedding plus grand = plus expressif

# ── Seuils structurels (portes binaires) ───────────────────────────────────────
# Ces seuils déterminent si la session est utilisable (score = 0 si bloquée)
OFFSET_THRESHOLD_MS    = 250.0   # décalage cam-trigger au-delà → pénalité offset
QUAT_CORRUPT_FRAC      = 0.05    # fraction NaN/inf quaternions → score = 0
TRACKER_GAP_MS         = 60.0    # seuil gap tracker
TRACKER_GAP_FAIL_N     = 3       # nb gaps tracker → score = 0
CAMERA_GAP_MS          = 120.0   # seuil gap caméra
CAMERA_GAP_FAIL_N      = 5       # nb gaps caméra → score = 0
MIN_COVERAGE_RATIO     = 0.65    # couverture min → score = 0

# ── Seuils score IA ────────────────────────────────────────────────────────────
MIN_IA_SCORE           = 0.55    # score IA → session considérée désynchronisée

PAIRS = [
    ("tracker_head", "cam_head"),
    ("tracker_left", "cam_left"),
    ("tracker_right", "cam_right"),
]

MAJOR_PAIRS = set(PAIRS)

# ── Device ────────────────────────────────────────────────────────────────────
def _get_device():
    if not _TORCH:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _get_device()

# ══════════════════════════════════════════════════════════════════════════════
# Structures de données
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Flux:
    name: str
    t_ms_rel: np.ndarray
    signal: np.ndarray
    t_start_abs_ms: float
    source: str = ""


@dataclass
class GateResult:
    """Résultat d'une vérification structurelle (porte binaire)."""
    name: str
    passed: bool       # True = ok, False = bloque le score
    message: str = ""
    value: Optional[float] = None


@dataclass
class SessionReport:
    """Rapport complet pour une session."""
    session_path: str
    session_id: str = ""
    gates: List[GateResult] = field(default_factory=list)
    ia_scores: Dict[str, float] = field(default_factory=dict)
    ia_score: float = 0.0          # score IA brut [0.0, 1.0]
    score: float = 0.0             # score final [0.0, 100.0]
    blocking_reason: str = ""      # raison si score = 0
    errors: List[str] = field(default_factory=list)

    def add_gate(self, g: GateResult):
        self.gates.append(g)

    def is_blocked(self) -> bool:
        return any(not g.passed for g in self.gates)

    def first_failure(self) -> Optional[GateResult]:
        for g in self.gates:
            if not g.passed:
                return g
        return None


# ── Compat: ancien verdict pour IA.py ─────────────────────────────────────────
# IA.py appelle check_session et vérifie report.verdict != "FAIL"
# On expose une propriété verdict pour ne pas casser l'intégration.
@property
def _verdict(self) -> str:
    if self.score >= 70.0:
        return "PASS"
    elif self.score >= 40.0:
        return "WARN"
    else:
        return "FAIL"

SessionReport.verdict = _verdict  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires signal
# ══════════════════════════════════════════════════════════════════════════════

def zscore(x: np.ndarray) -> np.ndarray:
    mu, sigma = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sigma + 1e-8)


def robust_clip(x: np.ndarray) -> np.ndarray:
    p1, p99 = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    return np.clip(x, p1, p99)


def moving_derivative(x: np.ndarray, dt: float) -> np.ndarray:
    d = np.diff(x, prepend=x[0]) / (dt + 1e-8)
    return d.astype(np.float32)


def smooth(x: np.ndarray, sigma_ms: float) -> np.ndarray:
    sigma_samples = max(sigma_ms / RESAMPLE_MS, 0.1)
    if _SCIPY:
        return gaussian_filter1d(x.astype(np.float64), sigma=sigma_samples).astype(np.float32)
    k = max(1, int(sigma_samples * 2))
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(x.astype(np.float64), kernel, mode="same").astype(np.float32)


def resample_to_grid(t: np.ndarray, sig: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, t, sig).astype(np.float32)


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(np.clip(c, -1.0, 1.0)) if np.isfinite(c) else 0.0


def heuristic_alignment_score(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8:
        return 0.0
    a1, b1 = zscore(a), zscore(b)
    corr  = safe_corrcoef(a1, b1)
    ea    = smooth(a1 * a1, 2.0)
    eb    = smooth(b1 * b1, 2.0)
    ecorr = safe_corrcoef(ea, eb)
    pa, _ = find_peaks(a1, height=float(np.percentile(a1, 75)))
    pb, _ = find_peaks(b1, height=float(np.percentile(b1, 75)))
    match = (sum(1 for p in pa if np.any(np.abs(pb - p) <= 4)) / max(len(pa), 1)
             if len(pa) > 0 and len(pb) > 0 else 0.0)
    fa = np.abs(np.fft.rfft(a1))
    fb = np.abs(np.fft.rfft(b1))
    scorr = safe_corrcoef(fa, fb)
    return float(np.clip(
        0.42 * max(corr, 0.0) + 0.28 * max(ecorr, 0.0) +
        0.20 * match + 0.10 * max(scorr, 0.0),
        0.0, 1.0,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

def _load_jsonl_times(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    times = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "capture_time" in obj:
                        times.append(float(obj["capture_time"]))
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        return None
    return np.array(times, dtype=np.float64) if times else None


def load_tracker_flux(session_dir: Path) -> Optional[Flux]:
    if not _PANDAS:
        return None
    path = session_dir / "tracker_positions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "timestamp_ns" in df.columns:
        t_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce").to_numpy(np.float64)
        t_ms_rel = (t_ns - t_ns[0]) / 1e6
        t_start_abs_ms = float(t_ns[0] / 1e6)
    elif "time_seconds" in df.columns:
        t_s = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy(np.float64)
        t_ms_rel = (t_s - t_s[0]) * 1000.0
        t_start_abs_ms = 0.0
    else:
        return None

    sig = None
    for prefix in ("tracker_head", "tracker_left", "tracker_right"):
        cols = [f"{prefix}_{ax}" for ax in ("x", "y", "z")]
        if all(c in df.columns for c in cols):
            xyz = np.stack([pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64) for c in cols], axis=1)
            speed = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[:1]), axis=1)
            sig = zscore(robust_clip(speed.astype(np.float32)))
            break

    if sig is None:
        return None

    valid = np.isfinite(t_ms_rel) & np.isfinite(sig)
    t_ms_rel, sig = t_ms_rel[valid], sig[valid]
    if len(t_ms_rel) < 20:
        return None

    return Flux(
        name="tracker_head",
        t_ms_rel=t_ms_rel.astype(np.float32),
        signal=sig.astype(np.float32),
        t_start_abs_ms=t_start_abs_ms,
        source=str(path),
    )


def load_camera_flux(session_dir: Path, cam: str) -> Optional[Flux]:
    """Construit un signal caméra à partir des timestamps JSONL (densité de frames)."""
    jsonl_path = session_dir / "videos" / f"{cam}.jsonl"
    times = _load_jsonl_times(jsonl_path)
    if times is None or len(times) < 20:
        return None

    t_start = float(times[0])
    t_ms_rel = (times - t_start).astype(np.float32)
    ifi = np.diff(times, prepend=times[0]).astype(np.float32)
    sig = zscore(robust_clip(np.abs(ifi - np.median(ifi))))

    valid = np.isfinite(t_ms_rel) & np.isfinite(sig)
    t_ms_rel, sig = t_ms_rel[valid], sig[valid]
    if len(t_ms_rel) < 20:
        return None

    return Flux(
        name=f"cam_{cam}",
        t_ms_rel=t_ms_rel,
        signal=sig,
        t_start_abs_ms=t_start,
        source=str(jsonl_path),
    )


def load_all_fluxes(session_dir: Path) -> Dict[str, Flux]:
    out: Dict[str, Flux] = {}
    trk = load_tracker_flux(session_dir)
    if trk:
        for prefix in ("tracker_head", "tracker_left", "tracker_right"):
            out[prefix] = Flux(
                name=prefix,
                t_ms_rel=trk.t_ms_rel.copy(),
                signal=trk.signal.copy(),
                t_start_abs_ms=trk.t_start_abs_ms,
                source=trk.source,
            )
    for cam in ("head", "left", "right"):
        fx = load_camera_flux(session_dir, cam)
        if fx:
            out[f"cam_{cam}"] = fx
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Pseudo-labels et Dataset
# ══════════════════════════════════════════════════════════════════════════════

def make_common_grid(ref: Flux, tgt: Flux, delta_start_ms: float,
                     extra_shift_ms: float, resample_ms: float):
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
    return grid, a, b


def window_slices(n: int, win: int, stride: int) -> List[Tuple[int, int]]:
    out = []
    s = 0
    while s + win <= n:
        out.append((s, s + win))
        s += stride
    return out


def build_pseudo_examples(session_dir: Path,
                          gt_label: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Construit des exemples d'entraînement pour une session.

    Si gt_label est fourni (0 ou 1), on utilise directement cette étiquette :
      - gt_label=1 (sync/) : on génère des fenêtres à lag≈0 → positif
      - gt_label=0 (desync/) : on génère des fenêtres à lags variés → négatif
    Sinon, on utilise l'heuristique de corrélation (fallback).
    """
    fluxes = load_all_fluxes(session_dir)
    examples = []
    win    = int(WINDOW_MS / RESAMPLE_MS)
    stride = max(4, int(WINDOW_STRIDE_MS / RESAMPLE_MS))

    for ref_name, tgt_name in PAIRS:
        if ref_name not in fluxes or tgt_name not in fluxes:
            continue
        ref = fluxes[ref_name]
        tgt = fluxes[tgt_name]
        delta = tgt.t_start_abs_ms - ref.t_start_abs_ms

        if gt_label is not None:
            if gt_label == 1:
                # Session alignée : fenêtres à lag=0 → toutes positives
                grid, a, b = make_common_grid(ref, tgt, delta, 0.0, RESAMPLE_MS)
                if grid is None:
                    continue
                for s, e in window_slices(len(grid), win, stride):
                    examples.append((a[s:e], b[s:e], 1))
            else:
                # Session désalignée : fenêtres à plusieurs lags décalés → toutes négatives
                # On prend quelques lags espacés pour couvrir la variété des décalages
                # (~12 lags × 3 paires × quelques fenêtres = ~200-600 exemples/session)
                lags_to_sample = np.arange(-MAX_LAG_MS + 50, MAX_LAG_MS - 50,
                                           80.0, dtype=np.float32)
                pair_ex_count = 0
                MAX_NEG_PER_PAIR = 200
                for lag in lags_to_sample:
                    if pair_ex_count >= MAX_NEG_PER_PAIR:
                        break
                    grid, a, b = make_common_grid(ref, tgt, delta, float(lag), RESAMPLE_MS)
                    if grid is None:
                        continue
                    for s, e in window_slices(len(grid), win, stride):
                        examples.append((a[s:e], b[s:e], 0))
                        pair_ex_count += 1
                        if pair_ex_count >= MAX_NEG_PER_PAIR:
                            break
        else:
            # Fallback heuristique (si pas d'étiquette GT)
            candidate_lags = np.arange(-MAX_LAG_MS, MAX_LAG_MS + RESAMPLE_MS,
                                       RESAMPLE_MS, dtype=np.float32)
            for lag in candidate_lags:
                if abs(float(lag)) >= (MAX_LAG_MS - EDGE_MARGIN_MS):
                    continue
                grid, a, b = make_common_grid(ref, tgt, delta, float(lag), RESAMPLE_MS)
                if grid is None:
                    continue
                for s, e in window_slices(len(grid), win, stride):
                    wa, wb = a[s:e], b[s:e]
                    score = heuristic_alignment_score(wa, wb)
                    if score >= PSEUDO_POS_THR:
                        examples.append((wa, wb, 1))
                    elif score <= PSEUDO_NEG_THR:
                        examples.append((wa, wb, 0))
    return examples


if _TORCH:
    class PairWindowDataset(Dataset):
        def __init__(self, examples: List[Tuple[np.ndarray, np.ndarray, int]]):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            a, b, y = self.examples[idx]
            a = zscore(robust_clip(a))
            b = zscore(robust_clip(b))
            da = moving_derivative(a, RESAMPLE_MS)
            db = moving_derivative(b, RESAMPLE_MS)
            ea = smooth(a * a, 2.0)
            eb = smooth(b * b, 2.0)
            xa = np.stack([a, da, ea], axis=0).astype(np.float32)
            xb = np.stack([b, db, eb], axis=0).astype(np.float32)
            return (torch.from_numpy(xa), torch.from_numpy(xb),
                    torch.tensor(float(y), dtype=torch.float32))


# ══════════════════════════════════════════════════════════════════════════════
# Modèle CrossModalAligner (amélioré)
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:
    class ConvBlock(nn.Module):
        def __init__(self, c_in, c_out, k=5, p=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c_in, c_out, k, padding=p),
                nn.BatchNorm1d(c_out), nn.GELU(),
                nn.Conv1d(c_out, c_out, k, padding=p),
                nn.BatchNorm1d(c_out), nn.GELU(),
            )
            self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class Encoder1D(nn.Module):
        def __init__(self, in_ch=3, emb=EMB):
            super().__init__()
            self.backbone = nn.Sequential(
                ConvBlock(in_ch, 32),  nn.MaxPool1d(2),
                ConvBlock(32, 64),     nn.MaxPool1d(2),
                ConvBlock(64, 96),     nn.MaxPool1d(2),
                ConvBlock(96, emb),    nn.AdaptiveAvgPool1d(1),
            )
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(emb, emb), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(emb, emb),
            )

        def forward(self, x):
            return F.normalize(self.proj(self.backbone(x)), dim=-1)

    class CrossModalAligner(nn.Module):
        def __init__(self, emb=EMB):
            super().__init__()
            self.enc_ref = Encoder1D(emb=emb)
            self.enc_tgt = Encoder1D(emb=emb)
            self.head = nn.Sequential(
                nn.Linear(emb * 4, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, 64),      nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, xa, xb):
            ea = self.enc_ref(xa)
            eb = self.enc_tgt(xb)
            feat = torch.cat([ea, eb, torch.abs(ea - eb), ea * eb], dim=-1)
            return self.head(feat).squeeze(-1), ea, eb


# ══════════════════════════════════════════════════════════════════════════════
# Entraînement
# ══════════════════════════════════════════════════════════════════════════════

def _collect_sessions(root: Path) -> List[Path]:
    return sorted(
        p.parent for p in root.rglob("metadata.json")
        if p.parent != root
    )


def train(sync_dir: Path = SYNC_DIR, desync_dir: Path = DESYNC_DIR,
          epochs: int = TRAIN_EPOCHS, batch_size: int = BATCH_SIZE,
          model_dir: Path = MODEL_DIR):
    if not _TORCH:
        print("[train] PyTorch non disponible — entraînement ignoré.")
        return
    if not _PANDAS:
        print("[train] Pandas non disponible — entraînement ignoré.")
        return

    sessions: List[Path] = []
    for root in (sync_dir, desync_dir):
        if root.exists():
            found = _collect_sessions(root)
            print(f"[train] {root.name}/ : {len(found)} sessions")
            sessions.extend(found)

    if not sessions:
        print("[train] Aucune session trouvée.")
        return

    # Associer chaque session à son étiquette GT (1 = sync, 0 = desync)
    gt_map: Dict[Path, int] = {}
    for root, label in ((sync_dir, 1), (desync_dir, 0)):
        if root.exists():
            for sess in _collect_sessions(root):
                gt_map[sess] = label

    print(f"[train] Collecte des exemples sur {len(sessions)} sessions "
          f"(sync={sum(1 for v in gt_map.values() if v==1)} "
          f"desync={sum(1 for v in gt_map.values() if v==0)})...")
    all_examples: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for i, sess in enumerate(sessions, 1):
        gt = gt_map.get(sess, None)
        ex = build_pseudo_examples(sess, gt_label=gt)
        if ex:
            all_examples.extend(ex)
            print(f"  [{i:3d}/{len(sessions)}] {sess.name:<40}  +{len(ex):5d} exemples "
                  f"(pos={sum(1 for _,_,y in ex if y==1)} "
                  f"neg={sum(1 for _,_,y in ex if y==0)})")
        else:
            print(f"  [{i:3d}/{len(sessions)}] {sess.name:<40}  (aucun flux disponible)")

    n_pos = sum(1 for _, _, y in all_examples if y == 1)
    n_neg = sum(1 for _, _, y in all_examples if y == 0)
    print(f"\n[train] Total : {len(all_examples)} exemples  pos={n_pos}  neg={n_neg}")

    if len(all_examples) < 50:
        print("[train] Pas assez d'exemples pour entraîner (< 50).")
        return

    # Équilibrage pos/neg
    pos_ex = [(a, b, y) for a, b, y in all_examples if y == 1]
    neg_ex = [(a, b, y) for a, b, y in all_examples if y == 0]
    n_min = min(len(pos_ex), len(neg_ex))
    if n_min > 0 and max(len(pos_ex), len(neg_ex)) > n_min * 3:
        # Sur-échantillonnage léger pour réduire le déséquilibre
        rng = np.random.default_rng(42)
        if len(pos_ex) < len(neg_ex):
            idx = rng.choice(len(pos_ex), size=min(n_min * 2, len(neg_ex)), replace=True)
            pos_ex = [pos_ex[i] for i in idx]
        else:
            idx = rng.choice(len(neg_ex), size=min(n_min * 2, len(pos_ex)), replace=True)
            neg_ex = [neg_ex[i] for i in idx]
        all_examples = pos_ex + neg_ex
        rng.shuffle(all_examples)
        print(f"[train] Après équilibrage : {len(all_examples)} exemples  "
              f"pos={sum(1 for _,_,y in all_examples if y==1)}  "
              f"neg={sum(1 for _,_,y in all_examples if y==0)}")

    dataset = PairWindowDataset(all_examples)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=(str(DEVICE) == "cuda"))

    model = CrossModalAligner().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(loader), epochs=epochs, pct_start=0.2,
        anneal_strategy="cos",
    )

    print(f"\n[train] Démarrage — {epochs} epochs  device={DEVICE}  batch={batch_size}  emb={EMB}")
    t0 = time.time()
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        losses, n_correct, n_seen = [], 0, 0
        for xa, xb, y in loader:
            xa, xb, y = xa.to(DEVICE), xb.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logit, ea, eb = model(xa, xb)
            bce  = F.binary_cross_entropy_with_logits(logit, y)
            dist = torch.norm(ea - eb, dim=-1)
            # Contrastive loss avec marge 0.7
            ctr  = (y * dist.pow(2) + (1 - y) * F.relu(0.7 - dist).pow(2)).mean()
            loss = 0.70 * bce + 0.30 * ctr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.item()))
            with torch.no_grad():
                n_correct += int(((torch.sigmoid(logit) >= 0.5).float() == y).sum())
                n_seen    += int(y.size(0))

        acc = n_correct / max(n_seen, 1) * 100
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  epoch {epoch+1:02d}/{epochs}  loss={np.mean(losses):.4f}  "
              f"acc={acc:.1f}%  best={best_acc:.1f}%  t={time.time()-t0:.0f}s")

    # Restaurer le meilleur modèle
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n[train] Meilleur modèle restauré (acc={best_acc:.1f}%)")

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "check_model.pt")
    meta = {
        "n_sessions": len(sessions),
        "n_examples": len(all_examples),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "epochs": epochs,
        "emb": EMB,
        "best_acc": round(best_acc, 2),
    }
    (model_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] Modèle sauvegardé → {model_dir / 'check_model.pt'}")


def load_model() -> Optional["CrossModalAligner"]:
    if not _TORCH:
        return None
    path = MODEL_DIR / "check_model.pt"
    if not path.exists():
        return None
    model = CrossModalAligner().to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Score IA par paire
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _ia_score_pair(model: "CrossModalAligner", ref: Flux, tgt: Flux) -> float:
    delta = tgt.t_start_abs_ms - ref.t_start_abs_ms
    win     = int(WINDOW_MS / RESAMPLE_MS)
    stride  = max(4, int(WINDOW_STRIDE_MS / RESAMPLE_MS))

    lag_scores = []
    cands = np.arange(-MAX_LAG_MS, MAX_LAG_MS + RESAMPLE_MS, RESAMPLE_MS, dtype=np.float32)
    for lag in cands:
        if abs(float(lag)) >= (MAX_LAG_MS - EDGE_MARGIN_MS):
            continue
        grid, a, b = make_common_grid(ref, tgt, delta, float(lag), RESAMPLE_MS)
        if grid is None:
            continue
        slices = window_slices(len(grid), win, stride)
        if not slices:
            continue
        batch_xa, batch_xb = [], []
        for s, e in slices:
            wa = zscore(robust_clip(a[s:e]))
            wb = zscore(robust_clip(b[s:e]))
            da = moving_derivative(wa, RESAMPLE_MS)
            db = moving_derivative(wb, RESAMPLE_MS)
            ea = smooth(wa * wa, 2.0)
            eb = smooth(wb * wb, 2.0)
            batch_xa.append(np.stack([wa, da, ea], axis=0))
            batch_xb.append(np.stack([wb, db, eb], axis=0))
        xa_t = torch.from_numpy(np.stack(batch_xa).astype(np.float32)).to(DEVICE)
        xb_t = torch.from_numpy(np.stack(batch_xb).astype(np.float32)).to(DEVICE)
        logit, _, _ = model(xa_t, xb_t)
        proba = torch.sigmoid(logit).cpu().numpy()
        lag_scores.append(0.65 * float(np.percentile(proba, 75)) + 0.35 * float(np.mean(proba)))

    if not lag_scores:
        return 0.0
    scores = np.array(lag_scores)
    return float(np.max(scores))


def score_session_ia(model: "CrossModalAligner", session_dir: Path) -> Dict[str, float]:
    fluxes = load_all_fluxes(session_dir)
    out = {}
    for ref_name, tgt_name in PAIRS:
        if ref_name not in fluxes or tgt_name not in fluxes:
            continue
        s = _ia_score_pair(model, fluxes[ref_name], fluxes[tgt_name])
        out[f"{ref_name}|{tgt_name}"] = round(s, 4)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Vérifications structurelles (portes binaires)
# ══════════════════════════════════════════════════════════════════════════════

def _gate_structure(session_dir: Path, report: SessionReport) -> bool:
    """Vérifie que tous les fichiers requis existent. Bloquant si manquant."""
    required = [
        "metadata.json",
        "tracker_positions.csv",
        "videos/head.jsonl",
        "videos/left.jsonl",
        "videos/right.jsonl",
    ]
    missing = []
    for rel in required:
        p = session_dir / rel
        if not p.exists() or p.stat().st_size == 0:
            missing.append(rel)

    if missing:
        report.add_gate(GateResult(
            name="structure",
            passed=False,
            message=f"Fichiers manquants : {', '.join(missing)}",
        ))
        return False
    report.add_gate(GateResult(name="structure", passed=True))
    return True


def _gate_metadata(session_dir: Path, report: SessionReport) -> Optional[dict]:
    """Lit metadata.json. Bloquant si illisible."""
    path = session_dir / "metadata.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        report.session_id = meta.get("session_id", "")
        report.add_gate(GateResult(name="metadata", passed=True))
        return meta
    except Exception as e:
        report.add_gate(GateResult(name="metadata", passed=False, message=str(e)))
        return None


def _gate_quaternions(session_dir: Path, report: SessionReport) -> bool:
    """Vérifie l'intégrité des quaternions tracker. Bloquant si > QUAT_CORRUPT_FRAC NaN/inf."""
    if not _PANDAS:
        return True
    path = session_dir / "tracker_positions.csv"
    if not path.exists():
        return True
    try:
        df = pd.read_csv(path)
    except Exception:
        return True

    worst_frac = 0.0
    for prefix in ("tracker_head", "tracker_left", "tracker_right"):
        q_cols = [f"{prefix}_q{c}" for c in ("w", "x", "y", "z")]
        if not all(c in df.columns for c in q_cols):
            continue
        Q = np.stack([pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
                      for c in q_cols], axis=1)
        frac = float(np.mean(~np.all(np.isfinite(Q), axis=1)))
        worst_frac = max(worst_frac, frac)

    if worst_frac > QUAT_CORRUPT_FRAC:
        report.add_gate(GateResult(
            name="quaternions",
            passed=False,
            message=f"{worst_frac*100:.1f}% quaternions NaN/inf (seuil {QUAT_CORRUPT_FRAC*100:.0f}%)",
            value=round(worst_frac, 4),
        ))
        return False
    report.add_gate(GateResult(name="quaternions", passed=True,
                               message=f"{worst_frac*100:.1f}% invalides", value=round(worst_frac, 4)))
    return True


def _gate_tracker_continuity(session_dir: Path, report: SessionReport) -> bool:
    """Vérifie les gaps tracker. Bloquant si trop de gaps."""
    if not _PANDAS:
        return True
    path = session_dir / "tracker_positions.csv"
    if not path.exists():
        return True
    try:
        df = pd.read_csv(path)
    except Exception:
        return True

    if "timestamp_ns" in df.columns:
        t_ms = pd.to_numeric(df["timestamp_ns"], errors="coerce").to_numpy(np.float64) / 1e6
    elif "time_seconds" in df.columns:
        t_ms = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy(np.float64) * 1000.0
    else:
        return True

    t_ms = t_ms[np.isfinite(t_ms)]
    if len(t_ms) < 2:
        return True

    dt = np.diff(t_ms)
    n_gaps = int(np.sum(dt > TRACKER_GAP_MS))
    max_gap = float(np.max(dt))

    if n_gaps >= TRACKER_GAP_FAIL_N:
        report.add_gate(GateResult(
            name="tracker_continuity",
            passed=False,
            message=f"{n_gaps} gaps > {TRACKER_GAP_MS:.0f} ms (max={max_gap:.0f} ms)",
            value=float(n_gaps),
        ))
        return False
    report.add_gate(GateResult(
        name="tracker_continuity", passed=True,
        message=f"{n_gaps} gaps (max={max_gap:.0f} ms)", value=float(n_gaps),
    ))
    return True


def _gate_camera_continuity(session_dir: Path, report: SessionReport) -> bool:
    """Vérifie les gaps caméra. Bloquant si trop de gaps sur n'importe quelle cam."""
    worst_n = 0
    worst_cam = ""
    for cam in ("head", "left", "right"):
        times = _load_jsonl_times(session_dir / "videos" / f"{cam}.jsonl")
        if times is None or len(times) < 2:
            continue
        dt = np.diff(times)
        n_gaps = int(np.sum(dt > CAMERA_GAP_MS))
        if n_gaps > worst_n:
            worst_n = n_gaps
            worst_cam = cam

    if worst_n >= CAMERA_GAP_FAIL_N:
        report.add_gate(GateResult(
            name="camera_continuity",
            passed=False,
            message=f"{worst_n} gaps > {CAMERA_GAP_MS:.0f} ms sur cam_{worst_cam}",
            value=float(worst_n),
        ))
        return False
    report.add_gate(GateResult(
        name="camera_continuity", passed=True,
        message=f"max {worst_n} gaps", value=float(worst_n),
    ))
    return True


def _gate_camera_coverage(session_dir: Path, report: SessionReport) -> bool:
    """Vérifie la couverture caméra sur la fenêtre tracker. Bloquant si insuffisante."""
    if not _PANDAS:
        return True
    trk_path = session_dir / "tracker_positions.csv"
    if not trk_path.exists():
        return True
    try:
        df = pd.read_csv(trk_path)
    except Exception:
        return True

    if "timestamp_ns" not in df.columns:
        return True

    t_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce").dropna().to_numpy(np.float64)
    if len(t_ns) < 2:
        return True
    trk_t0_ms = float(t_ns[0] / 1e6)
    trk_t1_ms = float(t_ns[-1] / 1e6)
    trk_dur_ms = trk_t1_ms - trk_t0_ms
    if trk_dur_ms <= 0:
        return True

    worst_ratio = 1.0
    worst_cam = ""
    for cam in ("head", "left", "right"):
        times = _load_jsonl_times(session_dir / "videos" / f"{cam}.jsonl")
        if times is None or len(times) < 2:
            continue
        cam_t0, cam_t1 = float(times[0]), float(times[-1])
        overlap_ms = max(0.0, min(cam_t1, trk_t1_ms) - max(cam_t0, trk_t0_ms))
        ratio = overlap_ms / (trk_dur_ms + 1e-6)
        if ratio < worst_ratio:
            worst_ratio = ratio
            worst_cam = cam

    if worst_ratio < MIN_COVERAGE_RATIO:
        report.add_gate(GateResult(
            name="camera_coverage",
            passed=False,
            message=f"Couverture {worst_ratio*100:.0f}% sur cam_{worst_cam} (seuil {MIN_COVERAGE_RATIO*100:.0f}%)",
            value=round(worst_ratio, 3),
        ))
        return False
    report.add_gate(GateResult(
        name="camera_coverage", passed=True,
        message=f"min {worst_ratio*100:.0f}%", value=round(worst_ratio, 3),
    ))
    return True


def _compute_penalties(session_dir: Path, meta: Optional[dict]) -> float:
    """Calcule un facteur de pénalité [0.5, 1.0] pour les défauts non bloquants.

    Les défauts non bloquants (offset cam élevé, quelques gaps) réduisent légèrement
    le score final sans le mettre à zéro.
    """
    penalty = 1.0

    # Pénalité offset caméra-trigger (session avec offset = signal potentiellement décalé)
    if meta is not None:
        trigger_ns = meta.get("trigger_time_ns")
        if trigger_ns is not None:
            trigger_ms = float(trigger_ns) / 1e6
            max_offset = 0.0
            for cam in ("head", "left", "right"):
                times = _load_jsonl_times(session_dir / "videos" / f"{cam}.jsonl")
                if times is not None and len(times) > 0:
                    offset = abs(float(times[0]) - trigger_ms)
                    max_offset = max(max_offset, offset)
            if max_offset >= OFFSET_THRESHOLD_MS:
                # Offset significatif : pénalité proportionnelle, max -20%
                penalty *= max(0.80, 1.0 - (max_offset - OFFSET_THRESHOLD_MS) / 5000.0)

    # Pénalité gaps tracker (quelques gaps mais sous le seuil bloquant)
    if _PANDAS:
        path = session_dir / "tracker_positions.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "timestamp_ns" in df.columns:
                    t_ms = pd.to_numeric(df["timestamp_ns"], errors="coerce").to_numpy(np.float64) / 1e6
                    t_ms = t_ms[np.isfinite(t_ms)]
                    if len(t_ms) >= 2:
                        n_gaps = int(np.sum(np.diff(t_ms) > TRACKER_GAP_MS))
                        if n_gaps > 0:
                            penalty *= max(0.90, 1.0 - n_gaps * 0.03)
            except Exception:
                pass

    return float(np.clip(penalty, 0.50, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# Vérification complète d'une session — retourne un score 0–100%
# ══════════════════════════════════════════════════════════════════════════════

def check_session(session_dir: Path,
                  model: Optional["CrossModalAligner"] = None) -> "SessionReport":
    """Vérifie une session et retourne un SessionReport avec score 0–100.

    score = 0   si une porte structurelle est bloquée (fichier manquant, corruption, etc.)
    score = IA × pénalités × 100   sinon
    """
    report = SessionReport(session_path=str(session_dir))

    # ── Portes structurelles (bloquantes) ─────────────────────────────────────
    if not _gate_structure(session_dir, report):
        report.blocking_reason = report.first_failure().message  # type: ignore[union-attr]
        report.score = 0.0
        return report

    meta = _gate_metadata(session_dir, report)
    if meta is None:
        report.blocking_reason = "metadata.json illisible"
        report.score = 0.0
        return report

    if not _gate_quaternions(session_dir, report):
        report.blocking_reason = report.first_failure().message  # type: ignore[union-attr]
        report.score = 0.0
        return report

    if not _gate_tracker_continuity(session_dir, report):
        report.blocking_reason = report.first_failure().message  # type: ignore[union-attr]
        report.score = 0.0
        return report

    if not _gate_camera_continuity(session_dir, report):
        report.blocking_reason = report.first_failure().message  # type: ignore[union-attr]
        report.score = 0.0
        return report

    if not _gate_camera_coverage(session_dir, report):
        report.blocking_reason = report.first_failure().message  # type: ignore[union-attr]
        report.score = 0.0
        return report

    # ── Score IA ──────────────────────────────────────────────────────────────
    if model is None:
        # Sans modèle : heuristique uniquement
        fluxes = load_all_fluxes(session_dir)
        heuristic_scores = []
        for ref_name, tgt_name in PAIRS:
            if ref_name not in fluxes or tgt_name not in fluxes:
                continue
            ref = fluxes[ref_name]
            tgt = fluxes[tgt_name]
            delta = tgt.t_start_abs_ms - ref.t_start_abs_ms
            grid, a, b = make_common_grid(ref, tgt, delta, 0.0, RESAMPLE_MS)
            if grid is not None:
                heuristic_scores.append(heuristic_alignment_score(a, b))
        ia_raw = float(np.mean(heuristic_scores)) if heuristic_scores else 0.0
    else:
        ia_scores_dict = score_session_ia(model, session_dir)
        report.ia_scores = ia_scores_dict
        if ia_scores_dict:
            major = [v for k, v in ia_scores_dict.items()
                     if any(f"{r}|{t}" == k for r, t in MAJOR_PAIRS)]
            ia_raw = float(np.mean(major)) if major else float(np.mean(list(ia_scores_dict.values())))
        else:
            ia_raw = 0.0

    report.ia_score = round(ia_raw, 4)

    # ── Pénalités qualité (non bloquantes) ────────────────────────────────────
    penalty = _compute_penalties(session_dir, meta)

    # Score final
    report.score = round(float(np.clip(ia_raw * penalty * 100.0, 0.0, 100.0)), 1)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# Affichage
# ══════════════════════════════════════════════════════════════════════════════

def _score_color(score: float) -> str:
    if score >= 70:
        return "\033[92m"   # vert
    elif score >= 40:
        return "\033[93m"   # jaune
    else:
        return "\033[91m"   # rouge

_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _color(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{_RESET}"
    return text


def print_report(report: SessionReport, verbose: bool = False):
    color = _score_color(report.score)
    score_str = _color(f"{report.score:.0f}%", color)
    sid = f"  ({report.session_id})" if report.session_id else ""
    ia_str = f"  ia={report.ia_score:.3f}" if report.ia_score > 0 else ""
    print(f"\n{_BOLD}[{score_str}]  {Path(report.session_path).name}{sid}{ia_str}{_RESET}")

    failed_gates = [g for g in report.gates if not g.passed]
    if failed_gates:
        for g in failed_gates:
            print(f"  ✗ {g.name}: {g.message}")
    elif report.blocking_reason:
        print(f"  ✗ {report.blocking_reason}")
    elif verbose:
        passed = [g for g in report.gates if g.passed]
        for g in passed:
            msg = f" ({g.message})" if g.message else ""
            print(f"  ✓ {g.name}{msg}")

    if verbose and report.ia_scores:
        print("  Scores IA par paire :")
        for k, v in sorted(report.ia_scores.items()):
            bar = "█" * int(v * 20)
            print(f"    {k:<35} {v:.3f}  {bar}")


def print_summary(reports: List["SessionReport"]):
    scores = [r.score for r in reports]
    n_high = sum(1 for s in scores if s >= 70)
    n_mid  = sum(1 for s in scores if 40 <= s < 70)
    n_low  = sum(1 for s in scores if s < 40)

    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ  {len(reports)} sessions")
    print(f"  {_color(f'≥70% (bon)   : {n_high}', chr(27)+'[92m')}")
    print(f"  {_color(f'40–70% (moyen): {n_mid}',  chr(27)+'[93m')}")
    print(f"  {_color(f'<40% (mauvais): {n_low}',  chr(27)+'[91m')}")

    if scores:
        print(f"\n  Score moyen : {np.mean(scores):.1f}%  "
              f"min={min(scores):.1f}%  max={max(scores):.1f}%")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    train(
        sync_dir=Path(args.sync_dir),
        desync_dir=Path(args.desync_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


def cmd_check(args):
    model = load_model()
    if model is None:
        print("[check] ⚠ Aucun modèle trouvé — scores heuristiques uniquement.\n"
              "         Lancez d'abord : python check.py train")

    target = Path(args.path)
    reports: List[SessionReport] = []

    if args.batch or (target.is_dir() and not (target / "metadata.json").exists()):
        sessions = _collect_sessions(target)
        if not sessions:
            print(f"Aucune session trouvée dans {target}")
            return
        print(f"[check] {len(sessions)} sessions à vérifier...")
        for sess in sessions:
            r = check_session(sess, model)
            reports.append(r)
            print_report(r, verbose=args.verbose)
        print_summary(reports)
    else:
        r = check_session(target, model)
        reports.append(r)
        print_report(r, verbose=True)

    if args.json_out:
        def _serial(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(type(obj))

        out = []
        for r in reports:
            d = asdict(r)
            # Ajouter le verdict compat pour les outils qui l'utilisent encore
            d["verdict"] = r.verdict
            out.append(d)
        Path(args.json_out).write_text(
            json.dumps(out, indent=2, default=_serial), encoding="utf-8"
        )
        print(f"[check] Rapport JSON → {args.json_out}")


def cmd_all(args):
    class TrainArgs:
        sync_dir   = args.sync_dir
        desync_dir = args.desync_dir
        epochs     = args.epochs
        batch_size = args.batch_size
    cmd_train(TrainArgs())

    class CheckArgs:
        path     = args.desync_dir
        batch    = True
        verbose  = getattr(args, "verbose", False)
        json_out = getattr(args, "json_out", None)
    print("\n[all] Vérification desync/ :")
    cmd_check(CheckArgs())

    class CheckArgs2(CheckArgs):
        path = args.sync_dir
    print("\n[all] Vérification sync/ :")
    cmd_check(CheckArgs2())


def main():
    parser = argparse.ArgumentParser(
        description="check.py — Vérification et scoring de synchronisation temporelle (0–100%)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sync-dir",   default=str(SYNC_DIR),   help="Dossier sync/")
    parser.add_argument("--desync-dir", default=str(DESYNC_DIR), help="Dossier desync/")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Entraîner le modèle sur sync/ + desync/")
    p_train.add_argument("--epochs",     type=int, default=TRAIN_EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p_train.set_defaults(func=cmd_train)

    p_check = sub.add_parser("check", help="Vérifier une ou plusieurs sessions")
    p_check.add_argument("path", help="Chemin vers la session ou le dossier racine")
    p_check.add_argument("--batch",    action="store_true",
                         help="Vérifier toutes les sessions dans path/")
    p_check.add_argument("--verbose",  action="store_true")
    p_check.add_argument("--json-out", metavar="FILE",
                         help="Sauvegarder le rapport en JSON")
    p_check.set_defaults(func=cmd_check)

    p_all = sub.add_parser("all", help="Train puis check sur sync/ + desync/")
    p_all.add_argument("--epochs",     type=int, default=TRAIN_EPOCHS)
    p_all.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p_all.add_argument("--verbose",    action="store_true")
    p_all.add_argument("--json-out",   metavar="FILE")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
