#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Détecte et corrige les décalages temporels de toutes les sessions d'un répertoire.

Pour chaque session trouvée :
  1. Calcule les offsets optimaux (multi-métriques + Random Forest auto-supervisé)
  2. Réécrit les fichiers DATA DIRECTEMENT EN PLACE (pas de sous-dossier)
     - tracker_positions.csv
     - gripper_left_data.csv / gripper_right_data.csv
     - videos/head.jsonl / left.jsonl / right.jsonl
  3. Génère un rapport par session + rapport global

Usage :
    python sync_fix.py .                        # traite toutes les sessions du répertoire courant
    python sync_fix.py /chemin/vers/sessions/   # répertoire arbitraire
    python sync_fix.py . --max-lag-ms 1000      # plage de recherche élargie
    python sync_fix.py . --step-ms 2            # précision 2 ms
    python sync_fix.py . --dry-run              # aperçu sans écrire
    python sync_fix.py . --force                # re-traite même les sessions déjà corrigées

Dépendances :
    pip install pandas numpy scipy scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy as scipy_entropy

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn non disponible — scoring heuristique uniquement.")


# ──────────────────────────────────────────────────────────────────────────────
# Paramètres
# ──────────────────────────────────────────────────────────────────────────────

PAIRS: List[Tuple[str, str]] = [
    ("tracker_head",  "cam_head"),
    ("tracker_left",  "cam_left"),
    ("tracker_right", "cam_right"),
    ("tracker_left",  "gripper_left"),
    ("tracker_right", "gripper_right"),
]

RESAMPLE_MS      = 5.0
SMOOTH_SIGMA_MS  = 80.0
N_BINS_MI        = 20
PEAK_TOL_MS      = 50.0
MIN_OVERLAP_MS   = 500.0

METRIC_WEIGHTS = {
    "pearson":    0.30,
    "cosine":     0.25,
    "mutual_inf": 0.20,
    "peak_align": 0.15,
    "spectral":   0.10,
}

MARKER_KEY = "sync_fix_applied"   # clé écrite dans metadata.json pour éviter double-traitement


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Corrige les décalages temporels de toutes les sessions en place."
    )
    p.add_argument("root_dir",     type=str, help="Répertoire contenant les dossiers session_*.")
    p.add_argument("--max-lag-ms", type=float, default=500.0,
                   help="Plage de recherche ±ms. Défaut: 500")
    p.add_argument("--step-ms",    type=float, default=5.0,
                   help="Pas de balayage en ms. Défaut: 5")
    p.add_argument("--no-ml",      action="store_true",
                   help="Scoring heuristique uniquement (sans Random Forest).")
    p.add_argument("--dry-run",    action="store_true",
                   help="Calcule les offsets mais n'écrit rien.")
    p.add_argument("--force",      action="store_true",
                   help="Re-traite même les sessions déjà corrigées.")
    p.add_argument("--no-plot",    action="store_true",
                   help="Ne génère pas les graphes PNG.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Structures
# ──────────────────────────────────────────────────────────────────────────────

class Flux:
    def __init__(self, name: str, t_ms_rel: np.ndarray, signal: np.ndarray,
                 t_start_abs_ms: float, source: str = "unknown"):
        self.name           = name
        self.t_ms_rel       = t_ms_rel
        self.signal         = signal
        self.t_start_abs_ms = t_start_abs_ms
        self.source         = source


class PairResult:
    def __init__(self, ref_name: str, tgt_name: str):
        self.ref_name         = ref_name
        self.tgt_name         = tgt_name
        self.delta_start_ms   = 0.0
        self.residual_ms      = 0.0   # résidu détecté par ML au-delà du Δstart
        self.total_offset_ms  = 0.0   # Δstart + résidu  (positif = target en avance)
        self.offset_rec_ms    = 0.0   # = -total_offset  (à soustraire du target)
        self.confidence       = 0.0
        self.method           = "heuristic"
        self.scores_arr       = np.array([])
        self.candidates_arr   = np.array([])


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des flux
# ──────────────────────────────────────────────────────────────────────────────

def _speed_signal(df: pd.DataFrame, t_ms: np.ndarray, pos: str) -> np.ndarray:
    cols = [f"tracker_{pos}_x", f"tracker_{pos}_y", f"tracker_{pos}_z"]
    if not all(c in df.columns for c in cols):
        return np.zeros(len(t_ms))
    x = pd.to_numeric(df[cols[0]], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[cols[1]], errors="coerce").to_numpy(float)
    z = pd.to_numeric(df[cols[2]], errors="coerce").to_numpy(float)
    dt = np.diff(t_ms) / 1000.0
    dt = np.where(dt <= 1e-6, np.nan, dt)
    dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    spd  = np.where(np.isfinite(dt) & (dt > 0), dist / dt, 0.0)
    return np.concatenate([[0.0], spd])


def _jsonl_first_ts(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    return float(json.loads(line)["capture_time"])
                except (KeyError, ValueError, json.JSONDecodeError):
                    pass
    return None


def load_trackers(session_dir: Path) -> Dict[str, Flux]:
    csv_path = session_dir / "tracker_positions.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    t_ns = pd.to_numeric(df.get("timestamp_ns", pd.Series(dtype=float)), errors="coerce")
    if t_ns.notna().any():
        t0  = float(t_ns.dropna().iloc[0])
        t_start = t0 / 1e6
        t_ms_rel = ((t_ns - t_ns.iloc[0]) / 1e6).to_numpy(float)
    else:
        t_s = pd.to_numeric(df["time_seconds"], errors="coerce")
        t_start = float(t_s.iloc[0]) * 1000.0
        t_ms_rel = ((t_s - t_s.iloc[0]) * 1000.0).to_numpy(float)
    df["_t"] = t_ms_rel
    df = df.sort_values("_t").reset_index(drop=True)
    t_arr = df["_t"].to_numpy(float)
    out: Dict[str, Flux] = {}
    for pos in ("head", "left", "right"):
        out[f"tracker_{pos}"] = Flux(
            name=f"tracker_{pos}", t_ms_rel=t_arr.copy(),
            signal=_speed_signal(df, t_arr, pos),
            t_start_abs_ms=t_start, source="tracker_csv")
    return out


def load_cameras(session_dir: Path) -> Dict[str, Flux]:
    out: Dict[str, Flux] = {}
    for cam in ("head", "left", "right"):
        flux_csv   = session_dir / "videos" / f"{cam}_flux.csv"
        jsonl_path = session_dir / "videos" / f"{cam}.jsonl"
        if flux_csv.exists():
            df = pd.read_csv(flux_csv)
            col = next((c for c in ("motion_mean_smooth", "motion_mean",
                                    "diff_mean_smooth", "diff_mean") if c in df.columns), None)
            if col is None:
                continue
            signal = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)
            t_abs_col = pd.to_numeric(df.get("timestamp_abs_ms"), errors="coerce") \
                        if "timestamp_abs_ms" in df.columns else None
            if t_abs_col is not None and t_abs_col.notna().any():
                first_v  = float(t_abs_col.dropna().iloc[0])
                t_ms_rel = (t_abs_col - first_v).ffill().to_numpy(float)
                t_start  = first_v
                source   = "motion_csv"
            else:
                t_start_j = _jsonl_first_ts(jsonl_path)
                t_s = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy(float)
                t_ms_rel = (t_s - t_s[0]) * 1000.0
                t_start  = t_start_j if t_start_j is not None else 0.0
                source   = "motion_csv_jsonl_anchor"
            out[f"cam_{cam}"] = Flux(f"cam_{cam}", t_ms_rel, signal, t_start, source)
        elif jsonl_path.exists():
            recs = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            if not recs:
                continue
            t_abs    = pd.to_numeric(pd.DataFrame(recs)["capture_time"], errors="coerce").to_numpy(float)
            t_start  = float(t_abs[0])
            t_ms_rel = t_abs - t_abs[0]
            iff      = np.diff(t_ms_rel)
            mean_if  = float(np.mean(iff)) if len(iff) > 0 else 33.333
            iff      = np.where(iff > 0, iff, mean_if)
            signal   = np.concatenate([[mean_if], iff])
            out[f"cam_{cam}"] = Flux(f"cam_{cam}", t_ms_rel, signal, t_start, "jsonl_only")
    return out


def load_grippers(session_dir: Path) -> Dict[str, Flux]:
    out: Dict[str, Flux] = {}
    for side in ("left", "right"):
        csv_path = session_dir / f"gripper_{side}_data.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        t_ns = pd.to_numeric(df.get("timestamp_ns", pd.Series(dtype=float)), errors="coerce")
        if t_ns.notna().any():
            t_start  = float(t_ns.dropna().iloc[0]) / 1e6
            t_ms_rel = ((t_ns - t_ns.iloc[0]) / 1e6).to_numpy(float)
        else:
            t_s = pd.to_numeric(df["time_seconds"], errors="coerce")
            t_start  = float(t_s.iloc[0]) * 1000.0
            t_ms_rel = ((t_s - t_s.iloc[0]) * 1000.0).to_numpy(float)
        df["_t"] = t_ms_rel
        df = df.sort_values("_t").reset_index(drop=True)
        t_arr = df["_t"].to_numpy(float)
        if "opening_mm" in df.columns:
            op  = pd.to_numeric(df["opening_mm"], errors="coerce").to_numpy(float)
            sig = np.concatenate([[0.0], np.abs(np.diff(op))])
        else:
            sig = np.ones(len(t_arr))
        out[f"gripper_{side}"] = Flux(f"gripper_{side}", t_arr, sig, t_start, "gripper_csv")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Signal utils
# ──────────────────────────────────────────────────────────────────────────────

def resamp(t_src: np.ndarray, sig: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.interp(t_grid, t_src, np.where(np.isfinite(sig), sig, 0.0), left=0.0, right=0.0)


def envelope(arr: np.ndarray, sigma: float) -> np.ndarray:
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
    def _rng(x):
        mn, mx = np.min(x), np.max(x)
        return mn, mx, mx - mn
    a_min, _, a_rng = _rng(a)
    b_min, _, b_rng = _rng(b)
    if a_rng < 1e-12 or b_rng < 1e-12:
        return 0.0
    nb = N_BINS_MI
    ai = np.clip(((a - a_min) / a_rng * (nb - 1)).astype(int), 0, nb - 1)
    bi = np.clip(((b - b_min) / b_rng * (nb - 1)).astype(int), 0, nb - 1)
    joint = np.zeros((nb, nb))
    np.add.at(joint, (ai, bi), 1)
    joint /= joint.sum() + 1e-12
    pa, pb = joint.sum(1), joint.sum(0)
    ha  = float(scipy_entropy(pa + 1e-12))
    hb  = float(scipy_entropy(pb + 1e-12))
    hab = float(scipy_entropy(joint.flatten() + 1e-12))
    mi  = ha + hb - hab
    return float(np.clip(mi / (min(ha, hb) + 1e-12), 0.0, 1.0))


def _peak_align(a: np.ndarray, b: np.ndarray, resample_ms: float) -> float:
    tol = int(PEAK_TOL_MS / resample_ms)
    if len(a) < 10:
        return 0.0
    pa, _ = find_peaks(a, height=np.percentile(a, 70), distance=max(1, tol // 2))
    pb, _ = find_peaks(b, height=np.percentile(b, 70), distance=max(1, tol // 2))
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    return sum(1 for p in pa if np.any(np.abs(pb - p) <= tol)) / len(pa)


def _spectral(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8:
        return 0.0
    n = min(len(a), len(b))
    fa, fb = np.abs(np.fft.rfft(a[:n])), np.abs(np.fft.rfft(b[:n]))
    if np.std(fa) < 1e-12 or np.std(fb) < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(fa, fb)[0, 1], 0.0, 1.0))


def score_overlap(a: np.ndarray, b: np.ndarray, env_a: np.ndarray,
                  env_b: np.ndarray, resample_ms: float) -> float:
    m = {
        "pearson":    _pearson(a, b),
        "cosine":     _cosine(env_a, env_b),
        "mutual_inf": _mi(a, b),
        "peak_align": _peak_align(a, b, resample_ms),
        "spectral":   _spectral(a, b),
    }
    return sum(METRIC_WEIGHTS[k] * m[k] for k in METRIC_WEIGHTS)


# ──────────────────────────────────────────────────────────────────────────────
# ML auto-supervisé
# ──────────────────────────────────────────────────────────────────────────────

def _features_at(ref_t, ref_sig, tgt_t, tgt_sig, offset_ms, resample_ms, sigma):
    tgt_ts = tgt_t + offset_ms
    t0 = max(ref_t[0], tgt_ts[0])
    t1 = min(ref_t[-1], tgt_ts[-1])
    if t1 - t0 < MIN_OVERLAP_MS:
        return None, 0
    grid = np.arange(t0, t1, resample_ms)
    if len(grid) < 8:
        return None, 0
    a = resamp(ref_t,  ref_sig,  grid)
    b = resamp(tgt_ts, tgt_sig,  grid)
    ea, eb = envelope(a, sigma), envelope(b, sigma)
    return [_pearson(a, b), _cosine(ea, eb), _mi(a, b),
            _peak_align(a, b, resample_ms), _spectral(a, b)], len(grid)


def build_rf(ref_t, ref_sig, tgt_t, tgt_sig, candidates, resample_ms, sigma):
    if not SKLEARN_AVAILABLE:
        return None
    tol = max(resample_ms * 3, 20.0)
    X, y = [], []
    for cand in candidates:
        feat, n = _features_at(ref_t, ref_sig, tgt_t, tgt_sig, cand, resample_ms, sigma)
        if feat is None:
            continue
        label = 1 if abs(cand) <= tol else 0
        X.append(feat + [abs(cand) / (max(np.abs(candidates)) + 1e-6),
                         n * resample_ms / 1000.0])
        y.append(label)
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


# ──────────────────────────────────────────────────────────────────────────────
# Analyse d'une paire
# ──────────────────────────────────────────────────────────────────────────────

def analyze_pair(ref: Flux, tgt: Flux, candidates: np.ndarray,
                 resample_ms: float, sigma: float, use_ml: bool) -> PairResult:
    res = PairResult(ref.name, tgt.name)
    res.delta_start_ms = tgt.t_start_abs_ms - ref.t_start_abs_ms

    # Aligner sur le Δstart absolu, puis chercher le résidu
    tgt_t_aligned = tgt.t_ms_rel + res.delta_start_ms

    # ── Balayage heuristique ──
    hscores = np.zeros(len(candidates))
    for i, cand in enumerate(candidates):
        shifted = tgt_t_aligned + cand
        t0 = max(ref.t_ms_rel[0], shifted[0])
        t1 = min(ref.t_ms_rel[-1], shifted[-1])
        if t1 - t0 < MIN_OVERLAP_MS:
            continue
        grid = np.arange(t0, t1, resample_ms)
        if len(grid) < 8:
            continue
        a = resamp(ref.t_ms_rel, ref.signal, grid)
        b = resamp(shifted,      tgt.signal, grid)
        ea, eb = envelope(a, sigma), envelope(b, sigma)
        hscores[i] = score_overlap(a, b, ea, eb, resample_ms)

    res.scores_arr     = hscores
    res.candidates_arr = candidates

    best_h = int(np.argmax(hscores))
    # Affinage parabolique
    resid_h = float(candidates[best_h])
    if 0 < best_h < len(hscores) - 1:
        y0, y1, y2 = hscores[best_h-1], hscores[best_h], hscores[best_h+1]
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-12:
            resid_h += (y0 - y2) / denom * (candidates[1] - candidates[0])

    res.residual_ms  = resid_h
    res.confidence   = float(hscores[best_h])
    res.method       = "heuristic"

    # ── ML ──
    if use_ml:
        rf = build_rf(ref.t_ms_rel, ref.signal,
                      tgt_t_aligned, tgt.signal,
                      candidates, resample_ms, sigma)
        if rf is not None:
            clf, sc = rf
            X_pred, valid = [], []
            for i, cand in enumerate(candidates):
                feat, n = _features_at(ref.t_ms_rel, ref.signal,
                                       tgt_t_aligned, tgt.signal,
                                       cand, resample_ms, sigma)
                if feat is None:
                    continue
                X_pred.append(feat + [abs(cand) / (max(np.abs(candidates)) + 1e-6),
                                      n * resample_ms / 1000.0])
                valid.append(i)
            if X_pred:
                proba = clf.predict_proba(sc.transform(np.array(X_pred, dtype=float)))
                pi    = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
                best_local = int(np.argmax(proba[:, pi]))
                best_global = valid[best_local]
                res.residual_ms = float(candidates[best_global])
                res.confidence  = float(proba[best_local, pi])
                res.method      = "ml"

    res.total_offset_ms = res.delta_start_ms + res.residual_ms
    res.offset_rec_ms   = -res.total_offset_ms
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Correction en place des fichiers
# ──────────────────────────────────────────────────────────────────────────────

def _shift_iso(series: pd.Series, delta_ns: int) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    shifted = ts + pd.to_timedelta(delta_ns, unit="ns")
    return shifted.dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def apply_offsets_inplace(session_dir: Path, offsets: Dict[str, float], dry_run: bool) -> None:
    """
    offsets : {flux_name: offset_rec_ms}  (positif = flux était en avance = on soustrait)
    """
    for flux_name, offset_ms in offsets.items():
        if abs(offset_ms) < 0.1:
            print(f"    {flux_name:<20} : offset {offset_ms:+.2f} ms  → ignoré (< 0.1 ms)")
            continue

        delta_ns = int(round(offset_ms * 1_000_000))
        delta_s  = offset_ms / 1000.0

        # ── tracker ──
        if flux_name == "tracker":
            path = session_dir / "tracker_positions.csv"
            if path.exists():
                df = pd.read_csv(path)
                if "timestamp_ns"  in df.columns:
                    df["timestamp_ns"]  = pd.to_numeric(df["timestamp_ns"],  errors="coerce") - delta_ns
                if "time_seconds"  in df.columns:
                    df["time_seconds"]  = pd.to_numeric(df["time_seconds"],  errors="coerce") - delta_s
                if "timestamp"     in df.columns:
                    df["timestamp"]     = _shift_iso(df["timestamp"], -delta_ns)
                if not dry_run:
                    df.to_csv(path, index=False)
                print(f"    tracker              : {offset_ms:+.2f} ms  → {path.name}")

        # ── caméras ──
        elif flux_name.startswith("cam_"):
            cam = flux_name[4:]  # head / left / right
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

        # ── grippers ──
        elif flux_name.startswith("gripper_"):
            side = flux_name[8:]  # left / right
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


# ──────────────────────────────────────────────────────────────────────────────
# Graphe
# ──────────────────────────────────────────────────────────────────────────────

def plot_session(results: List[PairResult], session_dir: Path) -> None:
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i, 0]
        if len(r.candidates_arr):
            ax.plot(r.candidates_arr, r.scores_arr, lw=1.5, color="#1565C0", alpha=0.8)
            ax.axvline(r.residual_ms, color="#C62828", lw=2,
                       linestyle="--", label=f"résidu={r.residual_ms:+.1f} ms")
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


# ──────────────────────────────────────────────────────────────────────────────
# Traitement d'une session
# ──────────────────────────────────────────────────────────────────────────────

def process_session(session_dir: Path, candidates: np.ndarray, resample_ms: float,
                    sigma: float, use_ml: bool, dry_run: bool, force: bool,
                    make_plot: bool) -> Optional[Dict]:
    print(f"\n{'─'*60}")
    print(f"  Session : {session_dir.name}")

    # Métadonnées
    meta_path = session_dir / "metadata.json"
    metadata  = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    session_id = metadata.get("session_id", session_dir.name)

    # Déjà corrigée ?
    if not force and metadata.get(MARKER_KEY):
        print(f"  → Déjà corrigée le {metadata[MARKER_KEY]}. Utilisez --force pour re-traiter.")
        return None

    # Charger les flux
    all_fluxes: Dict[str, Flux] = {}
    all_fluxes.update(load_trackers(session_dir))
    all_fluxes.update(load_cameras(session_dir))
    all_fluxes.update(load_grippers(session_dir))
    if not all_fluxes:
        print("  → Aucun flux trouvé, session ignorée.")
        return None

    # Analyser les paires
    results: List[PairResult] = []
    for ref_name, tgt_name in PAIRS:
        if ref_name not in all_fluxes or tgt_name not in all_fluxes:
            continue
        r = analyze_pair(all_fluxes[ref_name], all_fluxes[tgt_name],
                         candidates, resample_ms, sigma, use_ml)
        results.append(r)
        print(f"  {ref_name:<16} ↔ {tgt_name:<16} "
              f" Δstart={r.delta_start_ms:+.1f}  résidu={r.residual_ms:+.1f}"
              f"  offset_rec={r.offset_rec_ms:+.1f} ms  [{r.method} conf={r.confidence:.3f}]")

    if not results:
        print("  → Aucune paire analysable.")
        return None

    # Construire le dict d'offsets (un offset par target unique)
    offsets: Dict[str, float] = {}
    for r in results:
        if r.tgt_name not in offsets:
            offsets[r.tgt_name] = r.offset_rec_ms

    print(f"\n  Offsets appliqués ({'DRY-RUN' if dry_run else 'ÉCRITURE EN PLACE'}) :")
    apply_offsets_inplace(session_dir, offsets, dry_run)

    # Graphe
    if make_plot:
        plot_session(results, session_dir)

    # Sauvegarde JSON des résultats ML
    ml_json = {
        "session_id":  session_id,
        "generator":   "sync_fix.py",
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "dry_run":     dry_run,
        "pairs": [
            {
                "ref":             r.ref_name,
                "target":          r.tgt_name,
                "delta_start_ms":  r.delta_start_ms,
                "residual_ms":     r.residual_ms,
                "total_offset_ms": r.total_offset_ms,
                "offset_rec_ms":   r.offset_rec_ms,
                "confidence":      r.confidence,
                "method":          r.method,
            }
            for r in results
        ],
    }
    if not dry_run:
        (session_dir / "sync_fix_results.json").write_text(
            json.dumps(ml_json, indent=2, ensure_ascii=False), encoding="utf-8")

        # Marquer la session comme corrigée
        if meta_path.exists():
            metadata[MARKER_KEY] = datetime.now(timezone.utc).isoformat()
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    return {"session_id": session_id, "offsets": offsets, "n_pairs": len(results)}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    root = Path(args.root_dir).resolve()
    if not root.exists():
        print(f"ERREUR : répertoire introuvable : {root}", file=sys.stderr)
        return 1

    # Découverte des sessions
    sessions = sorted(p.parent for p in root.rglob("metadata.json")
                      if p.parent.name.startswith("session_"))
    if not sessions:
        print(f"Aucun dossier session_* trouvé dans {root}.", file=sys.stderr)
        return 1

    print(f"Répertoire racine : {root}")
    print(f"Sessions trouvées : {len(sessions)}")
    print(f"Plage             : ±{args.max_lag_ms} ms  |  Pas : {args.step_ms} ms")
    print(f"ML                : {'Non (--no-ml)' if args.no_ml else 'Oui (Random Forest)'}")
    print(f"Mode              : {'DRY-RUN' if args.dry_run else 'ÉCRITURE EN PLACE'}")
    if args.force:
        print("Force             : Oui (re-traite toutes les sessions)")

    candidates   = np.arange(-args.max_lag_ms, args.max_lag_ms + args.step_ms, args.step_ms)
    resample_ms  = RESAMPLE_MS
    sigma        = SMOOTH_SIGMA_MS / resample_ms
    use_ml       = (not args.no_ml) and SKLEARN_AVAILABLE
    make_plot    = not args.no_plot

    summary = []
    for s in sessions:
        res = process_session(s, candidates, resample_ms, sigma, use_ml,
                              args.dry_run, args.force, make_plot)
        if res:
            summary.append(res)

    # Rapport global
    print(f"\n{'═'*60}")
    print(f"  RÉSUMÉ  —  {len(summary)}/{len(sessions)} sessions traitées")
    print(f"{'═'*60}")
    for s in summary:
        print(f"  {s['session_id']}  ({s['n_pairs']} paires)")
        for flux, off in s["offsets"].items():
            print(f"    {flux:<20} : {off:>+8.2f} ms")

    if not summary:
        print("  Aucune session traitée.")
    elif args.dry_run:
        print("\n  [DRY-RUN] Aucun fichier modifié.")
    else:
        print(f"\n  Données corrigées en place dans {root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
