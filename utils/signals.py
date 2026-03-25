#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
signals.py — Extraction de flux de mouvement + Score d'alignement temporel.

Regroupe :
  - video.py   : extraction Farneback (flux optique) depuis une vidéo MP4
  - notation.py : score de synchronisation tracker ↔ vidéo (cross-corrélation, fenêtres, spectral)

Usage standalone :
    # Extraction de flux
    python signals.py flux video.mp4 --output-csv head_flux.csv --jsonl head.jsonl

    # Score d'alignement
    python signals.py score --tracker-csv tracker_positions.csv --video-csv head_flux.csv --pair tracker_head cam_head
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

# OpenCV utilise ses propres threads pour Farneback — lui allouer tous les CPU
cv2.setNumThreads(os.cpu_count() or 1)


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — Extraction de flux optique (video.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl_timestamps(jsonl_path: str) -> Dict[int, float]:
    """Charge un fichier JSONL et retourne un dict {frame_index: capture_time_ms}."""
    result: Dict[int, float] = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                result[int(rec["index"])] = float(rec["capture_time"])
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return result


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def preprocess_frame(frame: np.ndarray, resize_width: int, roi):
    h, w = frame.shape[:2]

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(x1, w - 1))
        x2 = max(1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(1, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI invalide.")
        frame = frame[y1:y2, x1:x2]

    if resize_width and resize_width > 0:
        h, w = frame.shape[:2]
        new_h = int(round(h * (resize_width / float(w))))
        frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def extract_optical_flow(
    video_path: str,
    output_csv: str,
    output_plot: str = "video_flux.png",
    output_npy: Optional[str] = None,
    resize_width: int = 640,
    smooth_window: int = 1,
    start_sec: float = 0.0,
    end_sec: float = -1.0,
    roi=None,
    farneback_pyr_scale: float = 0.5,
    farneback_levels: int = 3,
    farneback_winsize: int = 15,
    farneback_iterations: int = 3,
    farneback_poly_n: int = 5,
    farneback_poly_sigma: float = 1.2,
    jsonl_path: Optional[str] = None,
    show: bool = False,
) -> int:
    """
    Génère un flux 1D de mouvement à partir d'une vidéo (Farneback optical flow).
    Retourne 0 si succès, 1 si erreur.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"ERREUR : vidéo introuvable : {video_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERREUR : impossible d'ouvrir la vidéo.", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("ERREUR : FPS invalide.", file=sys.stderr)
        return 1

    duration_sec = frame_count / fps
    start_sec = max(0.0, start_sec)
    end_sec = duration_sec if end_sec < 0 else min(end_sec, duration_sec)

    if end_sec <= start_sec:
        print("ERREUR : intervalle temporel invalide.", file=sys.stderr)
        return 1

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    jsonl_timestamps: Optional[Dict[int, float]] = None
    if jsonl_path:
        jsonl_path_obj = Path(jsonl_path)
        if not jsonl_path_obj.exists():
            print(f"ERREUR : JSONL introuvable : {jsonl_path_obj}", file=sys.stderr)
            return 1
        jsonl_timestamps = load_jsonl_timestamps(str(jsonl_path_obj))
        print(f"JSONL chargé : {len(jsonl_timestamps)} entrées")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame = cap.read()
    if not ok:
        print("ERREUR : impossible de lire la première frame.", file=sys.stderr)
        return 1

    prev_gray = preprocess_frame(frame, resize_width, roi)
    rows = []
    current_frame_idx = start_frame

    while True:
        if current_frame_idx + 1 >= end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break

        current_frame_idx += 1
        time_sec = current_frame_idx / fps
        gray = preprocess_frame(frame, resize_width, roi)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            farneback_pyr_scale, farneback_levels, farneback_winsize,
            farneback_iterations, farneback_poly_n, farneback_poly_sigma, 0,
        )

        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        mag_flat = mag.ravel()
        mag_sorted = np.sort(mag_flat)
        p90_idx = int(0.90 * len(mag_sorted))

        abs_diff = cv2.absdiff(prev_gray, gray).astype(np.float32)
        diff_flat = abs_diff.ravel()
        diff_sorted = np.sort(diff_flat)
        dp90_idx = int(0.90 * len(diff_sorted))

        row = {
            "frame_index":   current_frame_idx,
            "time_seconds":  time_sec,
            "motion_mean":   float(mag_flat.mean()),
            "motion_median": float(mag_sorted[len(mag_sorted) // 2]),
            "motion_p90":    float(mag_sorted[p90_idx]),
            "motion_max":    float(mag_sorted[-1]),
            "diff_mean":     float(diff_flat.mean()),
            "diff_median":   float(diff_sorted[len(diff_sorted) // 2]),
            "diff_p90":      float(diff_sorted[dp90_idx]),
            "diff_max":      float(diff_sorted[-1]),
        }

        if jsonl_timestamps is not None:
            t_abs = jsonl_timestamps.get(current_frame_idx)
            row["timestamp_abs_ms"] = float(t_abs) if t_abs is not None else float("nan")

        rows.append(row)
        prev_gray = gray

    cap.release()

    if not rows:
        print("ERREUR : aucune donnée générée.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    numeric_cols = [
        "motion_mean", "motion_median", "motion_p90", "motion_max",
        "diff_mean", "diff_median", "diff_p90", "diff_max",
    ]
    for col in numeric_cols:
        df[col + "_smooth"] = moving_average(df[col].to_numpy(), smooth_window)

    df.to_csv(output_csv, index=False)

    if output_npy:
        np.save(
            output_npy,
            {
                "time_seconds":    df["time_seconds"].to_numpy(),
                "motion_mean":     df["motion_mean_smooth"].to_numpy(),
                "motion_median":   df["motion_median_smooth"].to_numpy(),
                "motion_p90":      df["motion_p90_smooth"].to_numpy(),
                "diff_mean":       df["diff_mean_smooth"].to_numpy(),
                "diff_median":     df["diff_median_smooth"].to_numpy(),
            },
            allow_pickle=True,
        )

    plt.figure(figsize=(16, 8))
    plt.plot(df["time_seconds"], df["motion_mean_smooth"],   label="motion_mean")
    plt.plot(df["time_seconds"], df["motion_median_smooth"], label="motion_median")
    plt.plot(df["time_seconds"], df["motion_p90_smooth"],    label="motion_p90")
    plt.plot(df["time_seconds"], df["diff_mean_smooth"],     label="diff_mean")
    plt.plot(df["time_seconds"], df["diff_median_smooth"],   label="diff_median")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.title("Flux de mouvement extraits de la vidéo")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=160)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Vidéo           : {video_path}")
    print(f"FPS             : {fps:.6f}")
    print(f"Frames totales  : {frame_count}")
    print(f"Début           : {start_sec:.3f} s")
    print(f"Fin             : {end_sec:.3f} s")
    print(f"CSV             : {output_csv}")
    print(f"PNG             : {output_plot}")
    if output_npy:
        print(f"NPY             : {output_npy}")

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — Score d'alignement temporel (notation.py)
# ══════════════════════════════════════════════════════════════════════════════

# Config stricte
RESAMPLE_MS       = 10.0
MAX_LAG_MS        = 400.0
WINDOW_MS         = 2000.0
WINDOW_STRIDE_MS  = 500.0
MIN_OVERLAP_MS    = 1500.0

PERFECT_LAG_MS   = 8.0
GOOD_LAG_MS      = 20.0
BAD_LAG_MS       = 60.0
PERFECT_MAD_MS   = 6.0
BAD_MAD_MS       = 30.0
MIN_ACTIVITY_STD = 0.15
MIN_WINDOWS      = 6
EPS              = 1e-8


@dataclass
class AlignReport:
    score_100: float
    verdict: str
    estimated_lag_ms: float
    is_shifted: bool

    global_peak_corr: float
    zero_lag_corr: float
    peak_prominence: float
    spectral_coherence: float

    median_window_lag_ms: float
    mad_window_lag_ms: float
    inlier_ratio_10ms: float
    inlier_ratio_20ms: float
    n_windows: int

    tracker_activity: float
    video_activity: float
    quality_cap: float

    reason: str


# ── Utils ─────────────────────────────────────────────────────────────────────

def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < EPS:
        return np.zeros_like(x, dtype=np.float64)
    return (x - med) / scale


def soft_clip(x: np.ndarray, q: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lim = np.percentile(np.abs(x[finite]), q)
    lim = max(float(lim), 1e-6)
    return np.clip(x, -lim, lim)


def smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0:
        return x
    return gaussian_filter1d(x, sigma=sigma_samples)


def derivative(x: np.ndarray, dt_ms: float) -> np.ndarray:
    dt_s = max(dt_ms / 1000.0, 1e-6)
    d = np.gradient(x) / dt_s
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def build_tracker_signal(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    parts = []
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Colonne tracker absente: {c}")
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
        parts.append(v)

    X = np.stack(parts, axis=1)
    if X.shape[1] >= 3:
        d = np.diff(X, axis=0, prepend=X[:1])
        speed = np.linalg.norm(d, axis=1)
    else:
        speed = np.mean(np.abs(np.diff(X, axis=0, prepend=X[:1])), axis=1)

    speed = smooth(speed, 2.0)
    accel = np.abs(derivative(speed, 10.0))
    energy = smooth(speed**2, 2.0)

    sig = 0.55 * robust_zscore(soft_clip(speed))
    sig += 0.25 * robust_zscore(soft_clip(accel))
    sig += 0.20 * robust_zscore(soft_clip(energy))
    sig = smooth(sig, 1.5)
    return sig.astype(np.float64)


def build_video_signal(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    parts = []
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Colonne vidéo absente: {c}")
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
        parts.append(robust_zscore(soft_clip(v)))

    sig = np.mean(np.stack(parts, axis=0), axis=0)
    sig = smooth(sig, 1.5)
    ds = np.abs(derivative(sig, 33.0))
    energy = smooth(sig**2, 2.0)

    out = 0.55 * robust_zscore(sig)
    out += 0.25 * robust_zscore(ds)
    out += 0.20 * robust_zscore(energy)
    out = smooth(out, 1.5)
    return out.astype(np.float64)


def get_time_ms(df: pd.DataFrame, candidates: List[str], scale_to_ms: float = 1.0) -> np.ndarray:
    for c in candidates:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64) * scale_to_ms
            valid = np.isfinite(t)
            if valid.sum() >= 2:
                t = t[valid]
                t = t - t[0]
                return t
    raise ValueError(f"Aucune colonne temps trouvée parmi {candidates}")


def interp_on_common_grid(
    t1: np.ndarray, x1: np.ndarray,
    t2: np.ndarray, x2: np.ndarray,
    resample_ms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0 = max(float(t1[0]), float(t2[0]))
    t1_end = min(float(t1[-1]), float(t2[-1]))
    if t1_end - t0 < MIN_OVERLAP_MS:
        raise ValueError("Pas assez d'overlap temporel")
    grid = np.arange(t0, t1_end, resample_ms, dtype=np.float64)
    if len(grid) < 64:
        raise ValueError("Grille commune trop courte")
    a = np.interp(grid, t1, x1)
    b = np.interp(grid, t2, x2)
    return grid, a, b


def norm_corr_for_lags(a: np.ndarray, b: np.ndarray, max_lag_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    a = robust_zscore(soft_clip(a))
    b = robust_zscore(soft_clip(b))
    corr_full = correlate(a, b, mode="full", method="fft")
    lags_full = np.arange(-len(b) + 1, len(a), dtype=int)
    denom = max(np.linalg.norm(a) * np.linalg.norm(b), EPS)
    corr_full = corr_full / denom
    keep = np.abs(lags_full) <= max_lag_samples
    return lags_full[keep], corr_full[keep]


def median_abs_deviation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def piecewise_desc_score(x: float, good: float, bad: float) -> float:
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return 1.0 - (x - good) / max(bad - good, EPS)


def piecewise_asc_score(x: float, bad: float, good: float) -> float:
    if x <= bad:
        return 0.0
    if x >= good:
        return 1.0
    return (x - bad) / max(good - bad, EPS)


def spectral_coherence_score(a: np.ndarray, b: np.ndarray) -> float:
    a = robust_zscore(a)
    b = robust_zscore(b)
    fa = np.abs(np.fft.rfft(a))
    fb = np.abs(np.fft.rfft(b))
    if len(fa) < 4 or len(fb) < 4:
        return 0.0
    fa = robust_zscore(fa)
    fb = robust_zscore(fb)
    num = float(np.dot(fa, fb))
    den = float(np.linalg.norm(fa) * np.linalg.norm(fb))
    if den < EPS:
        return 0.0
    c = num / den
    return clamp01((c + 1.0) / 2.0)


def sliding_windows(n: int, win: int, stride: int) -> List[Tuple[int, int]]:
    out = []
    s = 0
    while s + win <= n:
        out.append((s, s + win))
        s += stride
    return out


def filter_existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    out = [c for c in cols if c in df.columns]
    if not out:
        raise ValueError(f"Aucune colonne valide parmi: {cols}")
    return out


# ── Score principal ───────────────────────────────────────────────────────────

def compute_alignment_report(
    tracker_t_ms: np.ndarray,
    tracker_sig: np.ndarray,
    video_t_ms: np.ndarray,
    video_sig: np.ndarray,
    resample_ms: float = RESAMPLE_MS,
    max_lag_ms: float = MAX_LAG_MS,
    make_strict: bool = True,
) -> AlignReport:
    _, a_raw, b_raw = interp_on_common_grid(
        tracker_t_ms, tracker_sig, video_t_ms, video_sig, resample_ms
    )

    a = robust_zscore(soft_clip(a_raw))
    b = robust_zscore(soft_clip(b_raw))

    tracker_activity = float(np.std(a))
    video_activity   = float(np.std(b))

    max_lag_samples = int(round(max_lag_ms / resample_ms))
    lags_s, corr = norm_corr_for_lags(a, b, max_lag_samples=max_lag_samples)
    lags_ms = lags_s.astype(np.float64) * resample_ms

    if len(corr) < 3:
        return AlignReport(
            score_100=0.0, verdict="indéterminé",
            estimated_lag_ms=0.0, is_shifted=True,
            global_peak_corr=0.0, zero_lag_corr=0.0,
            peak_prominence=0.0, spectral_coherence=0.0,
            median_window_lag_ms=0.0, mad_window_lag_ms=999.0,
            inlier_ratio_10ms=0.0, inlier_ratio_20ms=0.0, n_windows=0,
            tracker_activity=tracker_activity, video_activity=video_activity,
            quality_cap=0.0, reason="corrélation insuffisante",
        )

    best_idx     = int(np.argmax(corr))
    best_lag_ms  = float(lags_ms[best_idx])
    best_corr    = float(corr[best_idx])

    zero_idx  = int(np.argmin(np.abs(lags_ms)))
    zero_corr = float(corr[zero_idx])

    tmp = corr.copy()
    tmp[best_idx] = -1e9
    second_best  = float(np.max(tmp)) if len(tmp) > 1 else -1.0
    peak_prominence = float(best_corr - second_best)

    win    = int(round(WINDOW_MS / resample_ms))
    stride = int(round(WINDOW_STRIDE_MS / resample_ms))
    win_ranges = sliding_windows(len(a), win, stride)

    local_lags  = []
    local_corrs = []

    for s, e in win_ranges:
        wa = a[s:e]
        wb = b[s:e]
        if len(wa) < 32:
            continue
        if np.std(wa) < MIN_ACTIVITY_STD or np.std(wb) < MIN_ACTIVITY_STD:
            continue
        wl_s, wc = norm_corr_for_lags(wa, wb, max_lag_samples=max_lag_samples)
        if len(wc) == 0:
            continue
        idx = int(np.argmax(wc))
        local_lags.append(float(wl_s[idx] * resample_ms))
        local_corrs.append(float(wc[idx]))

    local_lags  = np.asarray(local_lags,  dtype=np.float64)
    local_corrs = np.asarray(local_corrs, dtype=np.float64)

    if len(local_lags) >= 1:
        median_window_lag_ms = float(np.median(local_lags))
        mad_window_lag_ms    = float(median_abs_deviation(local_lags))
        inlier_ratio_10ms    = float(np.mean(np.abs(local_lags) <= 10.0))
        inlier_ratio_20ms    = float(np.mean(np.abs(local_lags) <= 20.0))
    else:
        median_window_lag_ms = best_lag_ms
        mad_window_lag_ms    = 999.0
        inlier_ratio_10ms    = 0.0
        inlier_ratio_20ms    = 0.0

    spec = spectral_coherence_score(a, b)

    lag_abs          = abs(best_lag_ms)
    lag_score_strict = piecewise_desc_score(lag_abs, PERFECT_LAG_MS, BAD_LAG_MS)
    lag_score_soft   = piecewise_desc_score(lag_abs, GOOD_LAG_MS, 120.0)
    lag_score        = 0.75 * lag_score_strict + 0.25 * lag_score_soft

    stability_score  = piecewise_desc_score(mad_window_lag_ms, PERFECT_MAD_MS, BAD_MAD_MS)
    inlier_score     = 0.6 * inlier_ratio_10ms + 0.4 * inlier_ratio_20ms
    global_corr_score    = piecewise_asc_score(best_corr, 0.20, 0.75)
    prominence_score     = piecewise_asc_score(peak_prominence, 0.02, 0.18)
    zero_gap             = max(best_corr - zero_corr, 0.0)
    zero_consistency_score = piecewise_desc_score(zero_gap, 0.015, 0.18)
    spectral_score       = spec
    tracker_quality      = piecewise_asc_score(tracker_activity, 0.08, 0.8)
    video_quality        = piecewise_asc_score(video_activity, 0.08, 0.8)
    signal_quality_score = 0.5 * tracker_quality + 0.5 * video_quality
    windows_score        = piecewise_asc_score(len(local_lags), 2, 10)

    raw  = 0.34 * lag_score
    raw += 0.16 * stability_score
    raw += 0.10 * inlier_score
    raw += 0.12 * global_corr_score
    raw += 0.10 * prominence_score
    raw += 0.08 * zero_consistency_score
    raw += 0.05 * spectral_score
    raw += 0.03 * signal_quality_score
    raw += 0.02 * windows_score
    raw  = clamp01(raw)

    quality_cap = 1.0
    if len(local_lags) < MIN_WINDOWS:
        quality_cap = min(quality_cap, 0.65)
    if tracker_activity < MIN_ACTIVITY_STD or video_activity < MIN_ACTIVITY_STD:
        quality_cap = min(quality_cap, 0.45)
    if best_corr < 0.30:
        quality_cap = min(quality_cap, 0.40)
    if peak_prominence < 0.03:
        quality_cap = min(quality_cap, 0.50)
    if mad_window_lag_ms > 40.0:
        quality_cap = min(quality_cap, 0.35)
    if make_strict:
        if lag_abs > BAD_LAG_MS:
            quality_cap = min(quality_cap, 0.10)
        elif lag_abs > 100.0:
            quality_cap = min(quality_cap, 0.05)

    final_score = 100.0 * min(raw, quality_cap)

    if quality_cap < 0.25 and final_score < 25:
        verdict = "décalée"
        reason  = "preuve forte de décalage ou signal trop faible"
    elif (lag_abs <= PERFECT_LAG_MS and mad_window_lag_ms <= PERFECT_MAD_MS
          and best_corr >= 0.65 and peak_prominence >= 0.08):
        verdict = "parfaite"
        final_score = max(final_score, 95.0)
        reason  = "lag ~ 0 ms, stable, pic net, corrélation forte"
    elif lag_abs <= GOOD_LAG_MS and mad_window_lag_ms <= 15.0 and best_corr >= 0.50:
        verdict = "bonne"
        reason  = "alignement global cohérent"
    elif lag_abs <= BAD_LAG_MS:
        verdict = "moyenne"
        reason  = "alignement partiel ou preuve insuffisante"
    else:
        verdict = "décalée"
        reason  = "meilleur lag trop éloigné de 0 ms"

    is_shifted = bool(final_score < 50.0 or abs(best_lag_ms) > GOOD_LAG_MS)

    return AlignReport(
        score_100=round(final_score, 2),
        verdict=verdict,
        estimated_lag_ms=round(best_lag_ms, 3),
        is_shifted=is_shifted,
        global_peak_corr=round(best_corr, 4),
        zero_lag_corr=round(zero_corr, 4),
        peak_prominence=round(peak_prominence, 4),
        spectral_coherence=round(spec, 4),
        median_window_lag_ms=round(median_window_lag_ms, 3),
        mad_window_lag_ms=round(mad_window_lag_ms, 3),
        inlier_ratio_10ms=round(inlier_ratio_10ms, 4),
        inlier_ratio_20ms=round(inlier_ratio_20ms, 4),
        n_windows=int(len(local_lags)),
        tracker_activity=round(tracker_activity, 4),
        video_activity=round(video_activity, 4),
        quality_cap=round(100.0 * quality_cap, 2),
        reason=reason,
    )


# ── I/O helpers ───────────────────────────────────────────────────────────────

def default_tracker_cols(pair_name: str) -> List[str]:
    mapping = {
        "tracker_head":  ["tracker_head_x",  "tracker_head_y",  "tracker_head_z"],
        "tracker_left":  ["tracker_left_x",  "tracker_left_y",  "tracker_left_z"],
        "tracker_right": ["tracker_right_x", "tracker_right_y", "tracker_right_z"],
    }
    if pair_name not in mapping:
        raise ValueError(f"Pair tracker inconnu: {pair_name}")
    return mapping[pair_name]


def default_video_cols(pair_name: str) -> List[str]:
    return ["motion_mean_smooth", "diff_mean_smooth", "motion_mean", "diff_mean"]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_flux_parser(sub):
    p = sub.add_parser("flux", help="Extraire un flux optique depuis une vidéo.")
    p.add_argument("video_path", type=str, help="Chemin de la vidéo.")
    p.add_argument("--output-csv",           type=str, default="video_flux.csv")
    p.add_argument("--output-plot",          type=str, default="video_flux.png")
    p.add_argument("--output-npy",           type=str, default=None)
    p.add_argument("--resize-width",         type=int, default=640)
    p.add_argument("--smooth-window",        type=int, default=1)
    p.add_argument("--start-sec",            type=float, default=0.0)
    p.add_argument("--end-sec",              type=float, default=-1.0)
    p.add_argument("--roi",                  type=int, nargs=4, metavar=("X1","Y1","X2","Y2"), default=None)
    p.add_argument("--farneback-pyr-scale",  type=float, default=0.5)
    p.add_argument("--farneback-levels",     type=int,   default=3)
    p.add_argument("--farneback-winsize",    type=int,   default=15)
    p.add_argument("--farneback-iterations", type=int,   default=3)
    p.add_argument("--farneback-poly-n",     type=int,   default=5)
    p.add_argument("--farneback-poly-sigma", type=float, default=1.2)
    p.add_argument("--jsonl",                type=str, default=None)
    p.add_argument("--show",                 action="store_true")
    return p


def _build_score_parser(sub):
    p = sub.add_parser("score", help="Score d'alignement tracker ↔ vidéo.")
    p.add_argument("--tracker-csv", type=Path, required=True)
    p.add_argument("--video-csv",   type=Path, required=True)
    p.add_argument("--pair", nargs=2, metavar=("TRACKER_NAME", "VIDEO_NAME"), default=None)
    p.add_argument("--tracker-cols", nargs="+", default=None)
    p.add_argument("--video-cols",   nargs="+", default=None)
    p.add_argument("--resample-ms",  type=float, default=RESAMPLE_MS)
    p.add_argument("--max-lag-ms",   type=float, default=MAX_LAG_MS)
    p.add_argument("--json-out",     type=Path,  default=None)
    return p


def _cmd_flux(args) -> int:
    return extract_optical_flow(
        video_path=args.video_path,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
        output_npy=args.output_npy,
        resize_width=args.resize_width,
        smooth_window=args.smooth_window,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        roi=args.roi,
        farneback_pyr_scale=args.farneback_pyr_scale,
        farneback_levels=args.farneback_levels,
        farneback_winsize=args.farneback_winsize,
        farneback_iterations=args.farneback_iterations,
        farneback_poly_n=args.farneback_poly_n,
        farneback_poly_sigma=args.farneback_poly_sigma,
        jsonl_path=args.jsonl,
        show=args.show,
    )


def _cmd_score(args) -> int:
    df_t = pd.read_csv(args.tracker_csv)
    df_v = pd.read_csv(args.video_csv)

    tracker_cols = args.tracker_cols
    video_cols   = args.video_cols

    if tracker_cols is None:
        if args.pair is None:
            raise ValueError("Sans --tracker-cols, il faut fournir --pair")
        tracker_cols = default_tracker_cols(args.pair[0])
    if video_cols is None:
        video_cols = default_video_cols(args.pair[1] if args.pair else "cam")

    tracker_cols = filter_existing_cols(df_t, tracker_cols)
    video_cols   = filter_existing_cols(df_v, video_cols)

    if "timestamp_ns" in df_t.columns:
        t_tracker_ms = get_time_ms(df_t, ["timestamp_ns"], scale_to_ms=1e-6)
    else:
        t_tracker_ms = get_time_ms(df_t, ["time_seconds"], scale_to_ms=1000.0)

    if "timestamp_abs_ms" in df_v.columns:
        t_video_ms = get_time_ms(df_v, ["timestamp_abs_ms"], scale_to_ms=1.0)
    else:
        t_video_ms = get_time_ms(df_v, ["time_seconds"], scale_to_ms=1000.0)

    tracker_sig = build_tracker_signal(df_t, tracker_cols)
    video_sig   = build_video_signal(df_v, video_cols)

    report  = compute_alignment_report(
        tracker_t_ms=t_tracker_ms,
        tracker_sig=tracker_sig,
        video_t_ms=t_video_ms,
        video_sig=video_sig,
        resample_ms=args.resample_ms,
        max_lag_ms=args.max_lag_ms,
        make_strict=True,
    )

    payload = asdict(report)
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="signals.py — extraction de flux optique + score d'alignement temporel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _build_flux_parser(sub)
    _build_score_parser(sub)

    args = parser.parse_args()
    if args.cmd == "flux":
        return _cmd_flux(args)
    elif args.cmd == "score":
        return _cmd_score(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
