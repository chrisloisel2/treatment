#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_gripper.py — Alignement temporel précis vidéo ↔ capteur par NCC kernel.

ALGORITHME
──────────
1. EXTRACTION MULTI-FEATURES
   16 features géométriques candidates extraites par frame :
   span (10/20/30%), dark_frac (10/20/30%), bright_center (10/20%),
   col_var (10%), mean_gray (10%), mid_dark, bot_dark, blue (10/20%)

2. AUTO-SÉLECTION DE FEATURE
   Sur sous-échantillon de 100 frames, NCC de chaque feature vs capteur.
   Score = |NCC_peak| × SNR. Feature avec meilleur score sélectionnée.
   Consensus vérifié sur top-3 (lag concordant à ±200ms).

3. NCC PAR CONVOLUTION FFT COMPLÈTE
   Signaux z-scorés robustes (MAD). Grille 1ms.
   NCC(τ) = fftconvolve(ṽ, flip(s̃)) / N

4. PRÉCISION SUB-MS PAR INTERPOLATION PARABOLIQUE
   δ = (NCC[k-1] - NCC[k+1]) / (2·(NCC[k-1] - 2·NCC[k] + NCC[k+1]))

5. CORRECTION + VÉRIFICATION FRAME PAR FRAME
   Pour chaque frame : ouverture capteur à t_frame - τ*
   Résidu = |vis_norm - sen_norm|  → flags SUSPECT si > μ + 3σ

Usage :
  python align_gripper.py --session /path/to/session
  python align_gripper.py --sessions_dir /path/to/all/sessions
  python align_gripper.py --session /path ... --side left
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve


# ═════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AlignConfig:
    grid_ms:            float = 1.0     # résolution grille commune (ms)
    max_lag_s:          float = 3.0     # ± secondes recherche NCC
    residual_sigma_thr: float = 3.0     # seuil SUSPECT (multiple de σ)
    n_calib_frames:     int   = 120     # frames pour auto-sélection feature
    consensus_tol_ms:   float = 200.0   # tolérance consensus top-3 features


# ═════════════════════════════════════════════════════════════════════════════
# RÉSULTAT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AlignResult:
    session:   str
    side:      str
    success:   bool
    error:     str = ""

    offset_ms:           float = float("nan")
    offset_ms_subpx:     float = float("nan")

    best_feature:        str   = ""
    feature_ncc_scores:  Dict  = field(default_factory=dict)   # {feature: ncc_peak}
    consensus_ok:        bool  = False
    consensus_spread_ms: float = float("nan")

    ncc_peak:            float = float("nan")
    ncc_peak_snr:        float = float("nan")

    residual_mean:       float = float("nan")
    residual_std:        float = float("nan")
    residual_p95:        float = float("nan")
    residual_max:        float = float("nan")
    residual_threshold:  float = float("nan")
    residual_sigma_thr:  float = 3.0

    n_frames:            int   = 0
    n_suspect:           int   = 0

    pearson_r:           float = float("nan")
    vis_polarity:        float = float("nan")  # NCC peak sign: <0 = feature inversée

    frame_data: Optional[pd.DataFrame] = field(default=None, repr=False)


# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def load_jsonl_abs(jsonl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    raw = open(jsonl_path, "rb").read()
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
        raise RuntimeError(f"JSONL vide : {jsonl_path}")
    idx = np.array(indices, dtype=np.int64)
    ts  = np.array(ts_ns,   dtype=np.int64)
    order = np.argsort(idx)
    idx, ts = idx[order], ts[order]
    idx = idx - idx[0]
    return idx, ts


def load_sensor_ns(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce")
    df["opening_mm"]   = pd.to_numeric(df["opening_mm"],   errors="coerce")
    df = df.dropna(subset=["timestamp_ns", "opening_mm"])
    df = df.sort_values("timestamp_ns").drop_duplicates("timestamp_ns")
    return df["timestamp_ns"].values.astype(np.int64), df["opening_mm"].values.astype(np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTION MULTI-FEATURES
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_DEFS = [
    # (nom, band_frac, type)
    # ── Features globales ────────────────────────────────────────────────────
    ("span_10",       0.10, "span"),
    ("span_20",       0.20, "span"),
    ("span_30",       0.30, "span"),
    ("dark_10",       0.10, "dark_frac"),
    ("dark_20",       0.20, "dark_frac"),
    ("bright_c_10",   0.10, "bright_center"),   # gap brillant au centre
    ("bright_c_20",   0.20, "bright_center"),
    ("mean_g_10",     0.10, "mean_gray"),        # inversé : sombre=ouvert
    ("col_var_10",    0.10, "col_var"),
    ("mid_dark",      None, "mid_dark"),
    # ── Features directionnelles (asymétrie gauche/droite) ───────────────────
    ("tl_dark",       0.10, "quadrant_tl"),      # top-left 1/3, dark frac
    ("tr_dark",       0.10, "quadrant_tr"),      # top-right 1/3, dark frac
    ("left_dark_10",  0.10, "half_left_dark"),
    ("right_dark_10", 0.10, "half_right_dark"),
    ("left_dark_20",  0.20, "half_left_dark"),
    ("left_bright_10",0.10, "half_left_bright"), # inversé : plus bright=plus fermé
    ("sym_diff",      0.10, "sym_diff"),          # |dark_left - dark_right|, inversé
    ("span_asym",     0.10, "span_asym"),         # position horizontale du centre de masse sombre
    # ── Features de gap ──────────────────────────────────────────────────────
    ("gap_10",        0.10, "inner_gap"),
    ("gap_20",        0.20, "inner_gap"),
]

DARK_THR = 60   # seuil binarisation fixe (pixel < DARK_THR → bras)


def _extract_frame_features(gray: np.ndarray,
                              frame_bgr: np.ndarray) -> Dict[str, float]:
    """Extrait toutes les features d'une frame. Retourne dict {nom: valeur}."""
    H, W = gray.shape
    blue_ch = frame_bgr[:, :, 0].astype(np.float32)   # BGR
    result: Dict[str, float] = {}

    for name, band_frac, ftype in FEATURE_DEFS:

        if ftype in ("span", "dark_frac", "bright_center",
                     "mean_gray", "col_var", "blue_frac", "inner_gap"):
            bh = max(1, int(H * band_frac))
            bg = gray[:bh, :]
            bb = blue_ch[:bh, :]
            col_min = bg.min(axis=0)

        if ftype == "span":
            dc = np.where(col_min < DARK_THR)[0]
            val = float(dc[-1] - dc[0]) / W if len(dc) >= 2 else 0.0

        elif ftype == "dark_frac":
            val = float((col_min < DARK_THR).mean())

        elif ftype == "bright_center":
            cw0 = W // 2 - W // 6
            cw1 = W // 2 + W // 6
            val = float((bg[:, cw0:cw1] > 100).mean())

        elif ftype == "mean_gray":
            # Inversé : plus sombre = plus ouvert pour certaines caméras
            val = 1.0 - float(bg.mean()) / 255.0

        elif ftype == "col_var":
            val = float(bg.mean(axis=0).std()) / 128.0

        elif ftype == "blue_frac":
            # Inversé : moins de bleu = plus ouvert (bras couvrent le fond bleu)
            val = 1.0 - float((bb > 100).any(axis=0).mean())

        elif ftype == "inner_gap":
            dc = np.where(col_min < DARK_THR)[0]
            if len(dc) >= 2:
                not_d = col_min >= DARK_THR
                mg, in_g, gs = 0, False, 0
                for c in range(int(dc[0]), int(dc[-1]) + 1):
                    if not_d[c]:
                        if not in_g:
                            gs = c; in_g = True
                    else:
                        if in_g:
                            mg = max(mg, c - gs); in_g = False
                val = float(mg) / W
            else:
                val = 0.0

        elif ftype == "mid_dark":
            bh0, bh1 = int(H * 0.20), int(H * 0.40)
            val = float((gray[bh0:bh1, :] < DARK_THR).mean())

        elif ftype == "bot_dark":
            bh0, bh1 = int(H * 0.40), int(H * 0.60)
            val = float((gray[bh0:bh1, :] < DARK_THR).mean())

        elif ftype == "quadrant_tl":
            bh = max(1, int(H * band_frac))
            q  = W // 3
            val = float((gray[:bh, :q] < DARK_THR).mean())

        elif ftype == "quadrant_tr":
            bh = max(1, int(H * band_frac))
            q  = W // 3
            val = float((gray[:bh, 2*q:] < DARK_THR).mean())

        elif ftype == "half_left_dark":
            bh = max(1, int(H * band_frac))
            val = float((gray[:bh, :W//2] < DARK_THR).mean())

        elif ftype == "half_right_dark":
            bh = max(1, int(H * band_frac))
            val = float((gray[:bh, W//2:] < DARK_THR).mean())

        elif ftype == "half_left_bright":
            bh = max(1, int(H * band_frac))
            # inversé : moins de bright → bras présent → gripper fermé
            val = 1.0 - float((gray[:bh, :W//2] > 100).mean())

        elif ftype == "sym_diff":
            bh = max(1, int(H * band_frac))
            dl = float((gray[:bh, :W//2] < DARK_THR).mean())
            dr = float((gray[:bh, W//2:] < DARK_THR).mean())
            # symétrique quand ouvert → inversé pour que plus ouvert = plus grand
            val = 1.0 - abs(dl - dr)

        elif ftype == "span_asym":
            bh = max(1, int(H * band_frac))
            col_min = gray[:bh, :].min(axis=0)
            dc = np.where(col_min < DARK_THR)[0]
            # position du centre de masse sombre (normalisée, 0=gauche, 1=droite)
            val = (float(np.median(dc)) - W / 2) / W if len(dc) >= 2 else 0.0

        else:
            val = 0.0

        result[name] = val

    return result


def extract_all_features(video_path: str,
                          frame_positions: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extrait toutes les features pour chaque frame de frame_positions.
    Retourne {feature_name: np.ndarray}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = len(frame_positions)
    raw: Dict[str, List] = {d[0]: [] for d in FEATURE_DEFS}

    current = -1
    for fi in frame_positions:
        fi = int(fi)
        if fi >= total:
            for k in raw:
                raw[k].append(np.nan)
            continue
        if fi != current + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            for k in raw:
                raw[k].append(np.nan)
            continue
        current = fi

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feats = _extract_frame_features(gray, frame)
        for k in raw:
            raw[k].append(feats.get(k, np.nan))

    cap.release()

    # Interpoler NaN + normalisation min-max robuste [2%–98%]
    out: Dict[str, np.ndarray] = {}
    idx = np.arange(n)
    for name, vals in raw.items():
        arr = np.array(vals, dtype=np.float64)
        valid = np.isfinite(arr)
        if valid.sum() < 5:
            out[name] = np.zeros(n)
            continue
        arr = np.interp(idx, idx[valid], arr[valid])   # combler NaN
        # Normalisation percentile robuste
        lo, hi = np.percentile(arr, [2, 98])
        if hi - lo < 1e-6:
            out[name] = np.zeros(n)
        else:
            out[name] = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    return out


def smooth_signal(x: np.ndarray, half_win: int = 5) -> np.ndarray:
    if half_win <= 0:
        return x.copy()
    k = np.ones(2 * half_win + 1) / (2 * half_win + 1)
    return np.convolve(x, k, mode="same")


# ═════════════════════════════════════════════════════════════════════════════
# NCC KERNEL
# ═════════════════════════════════════════════════════════════════════════════

def ncc_kernel(v: np.ndarray, s: np.ndarray,
               dt_ms: float, max_lag_s: float
               ) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    NCC normalisée par std → valeurs strictement dans [-1, 1].
    Utilise la corrélation croisée de Pearson à chaque lag.
    """
    N = len(v)
    assert len(s) == N

    # Z-score par std (pas MAD) : garantit output dans [-1, 1]
    v_std = float(v.std())
    s_std = float(s.std())
    if v_std < 0.02 or s_std < 1e-9:   # Signal sans dynamique
        lags_ms = (np.arange(2*N-1) - (N-1)) * dt_ms
        return lags_ms, np.zeros(2*N-1), N-1, 0.0

    v_z = (v - v.mean()) / v_std
    s_z = (s - s.mean()) / s_std

    xcorr   = fftconvolve(v_z, s_z[::-1], mode="full") / N
    # Clip pour garantir [-1, 1] malgré erreurs numériques
    xcorr   = np.clip(xcorr, -1.0, 1.0)
    k_vec   = np.arange(2 * N - 1) - (N - 1)
    lags_ms = k_vec * dt_ms

    max_lag_ms = max_lag_s * 1000.0
    mask = np.abs(lags_ms) <= max_lag_ms
    xm, lm = xcorr[mask], lags_ms[mask]

    best_w     = int(np.argmax(np.abs(xm)))
    global_idx = np.where(mask)[0][best_w]

    return lags_ms, xcorr, int(global_idx), float(xm[best_w])


def parabolic_interp(xcorr: np.ndarray, peak: int, dt_ms: float) -> float:
    if peak <= 0 or peak >= len(xcorr) - 1:
        N_half = (len(xcorr) + 1) // 2
        return float((peak - (N_half - 1)) * dt_ms)
    ym, y0, yp = float(xcorr[peak - 1]), float(xcorr[peak]), float(xcorr[peak + 1])
    denom = ym - 2.0 * y0 + yp
    delta = np.clip((ym - yp) / (2.0 * denom), -1.0, 1.0) if abs(denom) > 1e-12 else 0.0
    N_half = (len(xcorr) + 1) // 2
    return float((peak + delta - (N_half - 1)) * dt_ms)


def ncc_snr(xcorr: np.ndarray, peak: int, excl_ms: float, dt_ms: float) -> float:
    excl = max(1, int(excl_ms / dt_ms))
    lo, hi = max(0, peak - excl), min(len(xcorr), peak + excl + 1)
    bg = np.concatenate([xcorr[:lo], xcorr[hi:]])
    if len(bg) < 5:
        return float("nan")
    std = bg.std()
    return float(abs(xcorr[peak]) / std) if std > 1e-12 else float("inf")


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-SÉLECTION DE FEATURE
# ═════════════════════════════════════════════════════════════════════════════

def select_best_feature(
    all_features: Dict[str, np.ndarray],   # {name: signal normalisé [0,1] N frames}
    ts_video_ns:  np.ndarray,
    ts_sensor_ns: np.ndarray,
    op_sensor:    np.ndarray,
    cfg:          AlignConfig,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Pour chaque feature, calcule la NCC avec le capteur sur sous-échantillon.
    Retourne (best_feature_name, best_ncc_score, {feature: score}).

    Score = |NCC_peak| * min(SNR/10, 1.0) pour favoriser les pics nets.
    """
    dt_ns   = int(cfg.grid_ms * 1e6)
    t_min   = max(float(ts_video_ns[0]),  float(ts_sensor_ns[0]))
    t_max   = min(float(ts_video_ns[-1]), float(ts_sensor_ns[-1]))
    grid_ns = np.arange(t_min, t_max, dt_ns, dtype=np.float64)
    N_grid  = len(grid_ns)

    if N_grid < 200:
        raise RuntimeError("Grille trop courte pour auto-sélection.")

    # Capteur sur grille
    f_sen = interp1d(ts_sensor_ns.astype(np.float64), op_sensor,
                     kind="linear", bounds_error=False, fill_value=np.nan)
    sen_raw = f_sen(grid_ns)
    valid_g = np.isfinite(sen_raw)
    if valid_g.sum() < 200:
        raise RuntimeError("Capteur hors plage sur la grille.")
    sen_raw = np.interp(np.arange(N_grid), np.where(valid_g)[0], sen_raw[valid_g])
    lo98s, hi98s = np.percentile(sen_raw, [2, 98])
    if hi98s - lo98s < 1e-6:
        raise RuntimeError("Capteur sans dynamique.")
    sen_grid = np.clip((sen_raw - lo98s) / (hi98s - lo98s), 0.0, 1.0)

    scores: Dict[str, float] = {}
    lags_dict: Dict[str, float] = {}

    for name, vis_arr in all_features.items():
        # Filtre : dynamique minimale (std ≥ 5% de la plage normalisée)
        if vis_arr.std() < 0.05:
            scores[name] = 0.0
            lags_dict[name] = 0.0
            continue
        # Filtre : pas plus de 70% de valeurs identiques (signal trop sparse)
        hist, _ = np.histogram(vis_arr, bins=20)
        if hist.max() / len(vis_arr) > 0.70:
            scores[name] = 0.0
            lags_dict[name] = 0.0
            continue

        # Lisser légèrement (supprime bruit frame-à-frame)
        vis_smooth = smooth_signal(vis_arr, half_win=3)

        # Interpoler sur grille
        f_vis = interp1d(ts_video_ns.astype(np.float64), vis_smooth,
                         kind="linear", bounds_error=False, fill_value=np.nan)
        vis_raw = f_vis(grid_ns)
        valid_v = np.isfinite(vis_raw)
        if valid_v.sum() < 200:
            scores[name] = 0.0; lags_dict[name] = 0.0; continue
        vis_grid = np.interp(np.arange(N_grid), np.where(valid_v)[0], vis_raw[valid_v])

        try:
            lags_ms, xcorr, peak_idx, ncc_val = ncc_kernel(
                vis_grid, sen_grid, cfg.grid_ms, cfg.max_lag_s)
            # Score = |NCC_peak| ∈ [0,1], sans multiplication SNR
            # (le SNR peut être biaisé sur des signaux non-gaussiens)
            lag_ms = parabolic_interp(xcorr, peak_idx, cfg.grid_ms)
            scores[name] = round(float(abs(ncc_val)), 5)
            lags_dict[name] = round(float(lag_ms), 2)
        except Exception:
            scores[name] = 0.0; lags_dict[name] = 0.0

    if not scores or max(scores.values()) < 0.01:
        return "span_20", 0.0, scores

    best_name = max(scores, key=lambda k: scores[k])

    # Consensus : top-3 features d'accord sur le lag?
    top3 = sorted(scores, key=lambda k: scores[k], reverse=True)[:3]
    top_lags = [lags_dict[k] for k in top3]
    spread = max(top_lags) - min(top_lags) if top_lags else 0.0

    # ── Fallback consensus : si le meilleur feature est isolé (consensus échoue),
    # chercher la paire de features qui s'accordent le mieux sur le lag,
    # puis utiliser le feature de la paire ayant le score le plus élevé.
    if spread > cfg.consensus_tol_ms:
        top8 = sorted(scores, key=lambda k: scores[k], reverse=True)[:8]
        best_pair_score = -1.0
        best_consensus_candidate = best_name   # fallback
        for i in range(len(top8)):
            for j in range(i + 1, len(top8)):
                ki, kj = top8[i], top8[j]
                if abs(lags_dict[ki] - lags_dict[kj]) <= cfg.consensus_tol_ms:
                    pair_score = scores[ki] + scores[kj]
                    if pair_score > best_pair_score:
                        best_pair_score = pair_score
                        best_consensus_candidate = ki if scores[ki] >= scores[kj] else kj

        if best_consensus_candidate != best_name:
            # Une paire d'accord trouvée — elle est plus fiable que le feature isolé
            best_name = best_consensus_candidate

    # ── Garde-fou offset excessif ────────────────────────────────────────────
    # Si le meilleur feature donne un offset > 1500ms ET consensus échoue,
    # c'est probablement une corrélation spurieuse (feature sensible à la scène,
    # pas au gripper). Fallback vers les features de base fiables.
    if abs(lags_dict.get(best_name, 0.0)) > 1500.0 and spread > cfg.consensus_tol_ms:
        baseline = ["span_20", "dark_20", "span_10", "dark_10", "span_30", "dark_frac",
                    "bright_c_10", "mean_g_10"]
        for bf in baseline:
            if bf in scores and scores[bf] > 0.02:
                best_name = bf
                break

    return best_name, float(scores[best_name]), scores


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def align_side(session_path: str, side: str, cfg: AlignConfig, output_dir: Path
               ) -> Tuple["AlignResult", Optional[np.ndarray], Optional[np.ndarray]]:

    sname  = Path(session_path).name
    result = AlignResult(session=sname, side=side, success=False,
                         residual_sigma_thr=cfg.residual_sigma_thr)

    jsonl_p  = Path(session_path) / "videos" / f"{side}.jsonl"
    video_p  = Path(session_path) / "videos" / f"{side}.mp4"
    sensor_p = Path(session_path) / f"gripper_{side}_data.csv"

    for p in [jsonl_p, video_p, sensor_p]:
        if not p.exists():
            result.error = f"Fichier manquant : {p}"
            return result, None, None

    # ── Chargement ──────────────────────────────────────────────────────────
    try:
        frame_pos, ts_video_ns = load_jsonl_abs(str(jsonl_p))
        ts_sensor_ns, op_sensor = load_sensor_ns(str(sensor_p))
    except Exception as e:
        result.error = f"Chargement : {e}"
        return result, None, None

    n_frames = len(frame_pos)
    result.n_frames = n_frames
    print(f"  [{side}] {n_frames} frames  |  {len(ts_sensor_ns)} éch. capteur")

    # ── Extraction TOUTES les features ──────────────────────────────────────
    print(f"  [{side}] Extraction multi-features ({len(FEATURE_DEFS)} features)...")
    try:
        all_features = extract_all_features(str(video_p), frame_pos)
    except Exception as e:
        result.error = f"Extraction : {e}"
        return result, None, None

    # ── Auto-sélection feature ───────────────────────────────────────────────
    print(f"  [{side}] Auto-sélection feature...")
    try:
        best_feat, best_score, all_scores = select_best_feature(
            all_features, ts_video_ns, ts_sensor_ns, op_sensor, cfg)
    except Exception as e:
        result.error = f"Auto-sélection : {e}"
        return result, None, None

    result.best_feature      = best_feat
    result.feature_ncc_scores = {k: round(v, 5) for k, v in
                                   sorted(all_scores.items(), key=lambda x: -x[1])[:8]}

    top3 = sorted(all_scores, key=lambda k: all_scores[k], reverse=True)[:3]
    print(f"  [{side}] Top features : " +
          " | ".join(f"{k}={all_scores[k]:.4f}" for k in top3))
    print(f"  [{side}] Feature sélectionnée : {best_feat}  (score={best_score:.4f})")

    # ── Grille commune ───────────────────────────────────────────────────────
    dt_ns   = int(cfg.grid_ms * 1e6)
    t_min   = max(float(ts_video_ns[0]),  float(ts_sensor_ns[0]))
    t_max   = min(float(ts_video_ns[-1]), float(ts_sensor_ns[-1]))
    grid_ns = np.arange(t_min, t_max, dt_ns, dtype=np.float64)
    N_grid  = len(grid_ns)

    if N_grid < 200:
        result.error = "Grille trop courte."
        return result, None, None

    # Capteur sur grille — normalisé
    f_sen_raw = interp1d(ts_sensor_ns.astype(np.float64), op_sensor,
                         kind="linear", bounds_error=False, fill_value=np.nan)
    sen_raw = f_sen_raw(grid_ns)
    valid_s = np.isfinite(sen_raw)
    sen_raw = np.interp(np.arange(N_grid), np.where(valid_s)[0], sen_raw[valid_s])
    lo98s, hi98s = np.percentile(sen_raw, [2, 98])
    if hi98s - lo98s < 1e-6:
        result.error = "Capteur plat."
        return result, None, None
    sen_grid = np.clip((sen_raw - lo98s) / (hi98s - lo98s), 0.0, 1.0)

    # Signal visuel sélectionné sur grille
    vis_smooth = smooth_signal(all_features[best_feat], half_win=3)
    f_vis = interp1d(ts_video_ns.astype(np.float64), vis_smooth,
                     kind="linear", bounds_error=False, fill_value=np.nan)
    vis_raw = f_vis(grid_ns)
    valid_v = np.isfinite(vis_raw)
    if valid_v.sum() < 200:
        result.error = "Signal visuel insuffisant sur grille."
        return result, None, None
    vis_grid = np.interp(np.arange(N_grid), np.where(valid_v)[0], vis_raw[valid_v])

    # ── NCC KERNEL ───────────────────────────────────────────────────────────
    print(f"  [{side}] NCC kernel sur {N_grid} pts ({N_grid*cfg.grid_ms/1000:.1f}s)...")
    lags_ms, xcorr, peak_idx, ncc_val = ncc_kernel(
        vis_grid, sen_grid, cfg.grid_ms, cfg.max_lag_s)
    offset_ms_sub = parabolic_interp(xcorr, peak_idx, cfg.grid_ms)
    snr           = ncc_snr(xcorr, peak_idx, 500.0, cfg.grid_ms)

    # ── RAFFINEMENT PAR NCC DÉRIVÉE (robuste aux dérives lentes) ────────────
    # La NCC standard peut être attirée par des dérives lentes (bruit basse fréquence).
    # La NCC sur les dérivées (δvis/δt, δsen/δt) est sensible uniquement aux
    # TRANSITIONS RAPIDES (fermetures). Si les deux offsets divergent de > 150ms,
    # on préfère l'offset dérivée (plus précis pour les événements).
    try:
        vis_d_g = np.diff(vis_grid)
        sen_d_g = np.diff(sen_grid)
        vd_std  = float(vis_d_g.std())
        sd_std  = float(sen_d_g.std())
        if vd_std > 1e-5 and sd_std > 1e-5:
            vd_z = (vis_d_g - vis_d_g.mean()) / vd_std
            sd_z = (sen_d_g - sen_d_g.mean()) / sd_std
            N_d  = len(vd_z)
            xc_d = fftconvolve(vd_z, sd_z[::-1], mode="full") / N_d
            xc_d = np.clip(xc_d, -1.0, 1.0)
            # Chercher le pic de dérivée dans la même fenêtre de lag
            k_d  = np.arange(2 * N_d - 1) - (N_d - 1)
            lags_d_ms = k_d * cfg.grid_ms
            mask_d = np.abs(lags_d_ms) <= cfg.max_lag_s * 1000.0
            xm_d, lm_d = xc_d[mask_d], lags_d_ms[mask_d]
            best_d = int(np.argmax(np.abs(xm_d)))
            peak_idx_d = np.where(mask_d)[0][best_d]
            offset_d_ms = parabolic_interp(xc_d, peak_idx_d, cfg.grid_ms)
            ncc_d_val   = float(xm_d[best_d])

            drift = abs(offset_ms_sub - offset_d_ms)
            # Si la NCC dérivée donne un offset très différent (> 150ms)
            # ET son NCC est significatif (> 0.3), on utilise la NCC dérivée
            if drift > 150.0 and abs(ncc_d_val) > 0.30:
                print(f"  [{side}] ⚠ Dérive détectée : NCC_full={offset_ms_sub:.1f}ms"
                      f" vs NCC_derivée={offset_d_ms:.1f}ms (diff={drift:.0f}ms)"
                      f" → utilisation offset dérivée")
                offset_ms_sub = offset_d_ms
                # Recalcul du pic et SNR pour reporting
                lags_ms_eff = lags_d_ms
                peak_idx = peak_idx_d
                ncc_val  = ncc_d_val
    except Exception:
        pass

    # Vérification consensus top-3
    top3_names = sorted(all_scores, key=lambda k: all_scores[k], reverse=True)[:3]
    top3_lags = []
    for tn in top3_names:
        if all_features[tn].std() < 1e-6:
            continue
        vs = smooth_signal(all_features[tn], 3)
        fv = interp1d(ts_video_ns.astype(np.float64), vs,
                      kind="linear", bounds_error=False, fill_value=np.nan)
        vg = fv(grid_ns)
        valid_vt = np.isfinite(vg)
        if valid_vt.sum() < 200:
            continue
        vg = np.interp(np.arange(N_grid), np.where(valid_vt)[0], vg[valid_vt])
        try:
            _, xc_t, pi_t, _ = ncc_kernel(vg, sen_grid, cfg.grid_ms, cfg.max_lag_s)
            top3_lags.append(parabolic_interp(xc_t, pi_t, cfg.grid_ms))
        except Exception:
            pass

    consensus_spread = (max(top3_lags) - min(top3_lags)) if len(top3_lags) >= 2 else float("nan")
    consensus_ok = np.isfinite(consensus_spread) and consensus_spread <= cfg.consensus_tol_ms

    result.offset_ms           = float(lags_ms[peak_idx])
    result.offset_ms_subpx     = offset_ms_sub
    result.ncc_peak            = round(abs(ncc_val), 5)
    result.ncc_peak_snr        = round(snr, 2) if np.isfinite(snr) else float("nan")
    result.consensus_ok        = bool(consensus_ok)
    result.consensus_spread_ms = round(float(consensus_spread), 2) if np.isfinite(consensus_spread) else float("nan")

    print(f"  [{side}] Offset τ*={offset_ms_sub:.3f}ms  |  "
          f"NCC={abs(ncc_val):.4f}  |  SNR={snr:.1f}  |  "
          f"consensus={'OK' if consensus_ok else 'FAIL'} "
          f"(spread={consensus_spread:.1f}ms)")

    # ── CORRECTION + RÉSIDUS PAR FRAME ──────────────────────────────────────
    offset_ns     = int(offset_ms_sub * 1e6)
    t_corr_ns     = ts_video_ns.astype(np.float64) - float(offset_ns)

    # Capteur brut aligné
    f_sen_mm = interp1d(ts_sensor_ns.astype(np.float64), op_sensor,
                        kind="linear", bounds_error=False, fill_value=np.nan)
    opening_aligned = f_sen_mm(t_corr_ns)

    # Capteur normalisé aligné
    sen_norm_func = interp1d(
        ts_sensor_ns.astype(np.float64),
        np.clip((op_sensor - lo98s) / (hi98s - lo98s), 0.0, 1.0),
        kind="linear", bounds_error=False, fill_value=np.nan)
    sen_norm_at_frames = sen_norm_func(t_corr_ns)

    # Signal visuel normalisé par frame (signal brut, non lissé sur grille)
    vis_at_frames = all_features[best_feat]   # déjà normalisé

    # Résidus
    residuals = np.abs(vis_at_frames - sen_norm_at_frames)
    valid_frames = np.isfinite(residuals)

    if valid_frames.sum() < 5:
        result.error = "Résidus insuffisants."
        return result, None, None

    res_v   = residuals[valid_frames]
    res_mean = float(res_v.mean())
    res_std  = float(res_v.std())
    threshold = res_mean + cfg.residual_sigma_thr * res_std

    result.residual_mean      = round(res_mean, 6)
    result.residual_std       = round(res_std,  6)
    result.residual_p95       = round(float(np.percentile(res_v, 95)), 6)
    result.residual_max       = round(float(res_v.max()), 6)
    result.residual_threshold = round(float(threshold), 6)
    result.n_suspect          = int((residuals > threshold).sum())

    # Pearson après alignement
    v_ok = vis_at_frames[valid_frames]
    s_ok = sen_norm_at_frames[valid_frames]
    if v_ok.std() > 1e-9 and s_ok.std() > 1e-9:
        result.pearson_r = round(float(np.corrcoef(v_ok, s_ok)[0, 1]), 6)

    # NOTE: si ncc_val < 0, la feature est inversée (vis monte quand gripper ferme).
    # Le CSV garde vis_norm tel quel. validate_align.py détecte et gère ce cas.
    result.vis_polarity = round(float(ncc_val), 4)  # <0 = feature inversée

    # DataFrame par frame
    t_rel_s = (ts_video_ns - ts_video_ns[0]) / 1e9
    result.frame_data = pd.DataFrame({
        "frame_idx":          frame_pos.astype(int),
        "timestamp_ns":       ts_video_ns,
        "t_rel_s":            t_rel_s,
        "opening_mm_aligned": opening_aligned,
        "vis_norm":           vis_at_frames,
        "sen_norm_aligned":   sen_norm_at_frames,
        "residual":           residuals,
        "suspect":            (residuals > threshold).astype(int),
    })

    result.success = True
    return result, lags_ms, xcorr


# ═════════════════════════════════════════════════════════════════════════════
# GRAPHE
# ═════════════════════════════════════════════════════════════════════════════

def make_diagnostic_plot(result: AlignResult, output_path: Path,
                          lags_ms: Optional[np.ndarray] = None,
                          xcorr: Optional[np.ndarray] = None) -> None:
    df = result.frame_data
    if df is None or len(df) == 0:
        return

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), dpi=100)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    t = df["t_rel_s"].values

    # ── Panneau 1 : signaux ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, df["vis_norm"],         color="#58a6ff", lw=1.0, alpha=0.9,
            label=f"vision [{result.best_feature}]")
    ax.plot(t, df["sen_norm_aligned"], color="#ff7b72", lw=1.0, alpha=0.9,
            label=f"capteur aligné (τ*={result.offset_ms_subpx:.2f}ms)")
    suspect_t = t[df["suspect"].values == 1]
    if len(suspect_t):
        ax.vlines(suspect_t, 0, 1, color="#f0e68c", alpha=0.3, lw=0.7,
                  label=f"{result.n_suspect} frames SUSPECT")
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("Signal normalisé", color="#c9d1d9", fontsize=9)
    status = "OK" if result.consensus_ok else "⚠ NO CONSENSUS"
    ax.set_title(
        f"{result.session}  |  {result.side}  |  "
        f"r={result.pearson_r:.4f}  NCC={result.ncc_peak:.4f}  SNR={result.ncc_peak_snr:.1f}  "
        f"consensus={status}  spread={result.consensus_spread_ms:.1f}ms",
        color="#c9d1d9", fontsize=9, pad=6)
    ax.legend(loc="upper right", fontsize=7, facecolor="#161b22",
              labelcolor="#c9d1d9", framealpha=0.8)
    ax.grid(True, alpha=0.12, color="#c9d1d9")

    # ── Panneau 2 : résidus ──────────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(t, df["residual"].values, color="#3fb950", alpha=0.6)
    ax.axhline(result.residual_threshold, color="#f0e68c", lw=1.5, ls="--",
               label=f"seuil {result.residual_sigma_thr}σ={result.residual_threshold:.3f}")
    ax.axhline(result.residual_mean, color="#58a6ff", lw=1.0, ls=":",
               label=f"μ={result.residual_mean:.4f}  σ={result.residual_std:.4f}")
    pct = 100.0 * result.n_suspect / max(result.n_frames, 1)
    ax.set_title(f"Résidu |vision − capteur| par frame  |  "
                 f"{result.n_suspect} SUSPECT ({pct:.1f}%)",
                 color="#c9d1d9", fontsize=9)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Résidu norm.", color="#c9d1d9", fontsize=9)
    ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.8)
    ax.grid(True, alpha=0.12, color="#c9d1d9")

    # ── Panneau 3 : courbe NCC ───────────────────────────────────────────────
    ax = axes[2]
    if lags_ms is not None and xcorr is not None:
        mask = np.abs(lags_ms) <= 3000.0
        ax.plot(lags_ms[mask], xcorr[mask], color="#58a6ff", lw=0.8, alpha=0.8)
        ax.fill_between(lags_ms[mask], xcorr[mask], alpha=0.12, color="#58a6ff")
    ax.axvline(result.offset_ms_subpx, color="#f0e68c", lw=2.0, ls="--",
               label=f"τ*={result.offset_ms_subpx:.3f}ms")
    ax.axvline(0.0, color="#c9d1d9", lw=0.7, ls="-", alpha=0.3)
    ax.set_title("NCC kernel — corrélation croisée normalisée",
                 color="#c9d1d9", fontsize=9)
    ax.set_xlabel("Lag τ (ms)", color="#c9d1d9", fontsize=9)
    ax.set_ylabel("NCC", color="#c9d1d9", fontsize=9)
    ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.8)
    ax.text(0.02, 0.82,
            f"feature: {result.best_feature}\n"
            f"NCC={result.ncc_peak:.4f}  SNR={result.ncc_peak_snr:.1f}",
            transform=ax.transAxes, color="#c9d1d9", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#1c2128", alpha=0.7))
    ax.grid(True, alpha=0.12, color="#c9d1d9")

    # ── Panneau 4 : scores features ──────────────────────────────────────────
    ax = axes[3]
    score_data = result.feature_ncc_scores
    names = list(score_data.keys())
    vals  = [score_data[n] for n in names]
    colors = ["#f0e68c" if n == result.best_feature else "#58a6ff" for n in names]
    bars = ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Score NCC × SNR", color="#c9d1d9", fontsize=9)
    ax.set_title("Scores d'auto-sélection des features",
                 color="#c9d1d9", fontsize=9)
    ax.tick_params(colors="#c9d1d9")
    ax.grid(True, alpha=0.12, axis="x", color="#c9d1d9")

    plt.tight_layout(pad=1.5)
    plt.savefig(str(output_path), dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ═════════════════════════════════════════════════════════════════════════════

def save_results(result: AlignResult, output_dir: Path, side: str,
                  lags_ms=None, xcorr=None) -> None:
    if result.frame_data is not None:
        result.frame_data.to_csv(
            str(output_dir / f"aligned_{side}.csv"), index=False, float_format="%.6f")

    summary = {k: v for k, v in asdict(result).items() if k != "frame_data"}
    with open(str(output_dir / f"align_report_{side}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False,
                  default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    try:
        make_diagnostic_plot(result, output_dir / f"align_diag_{side}.png",
                              lags_ms=lags_ms, xcorr=xcorr)
    except Exception as e:
        print(f"  [WARN] Graphe : {e}")


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════

def run_session(session_path: str, cfg: AlignConfig,
                sides: List[str], base_output: Optional[Path]) -> Dict:
    sname = Path(session_path).name
    out   = (base_output or Path(session_path)) / "align_gripper"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for side in sides:
        print(f"\n── {sname}  côté {side} ─────────────────────────────")
        result, lags_ms, xcorr = align_side(session_path, side, cfg, out)
        results[side] = result

        if result.success:
            pct = 100.0 * result.n_suspect / max(result.n_frames, 1)
            ok  = (result.ncc_peak >= 0.35 and result.consensus_ok
                   and pct < 10 and result.pearson_r >= 0.6)
            status = "OK" if ok else "WARNING"
            print(f"  → {status}  r={result.pearson_r:.4f}  "
                  f"NCC={result.ncc_peak:.4f}  SNR={result.ncc_peak_snr:.1f}  "
                  f"suspect={result.n_suspect}({pct:.1f}%)")
            save_results(result, out, side, lags_ms, xcorr)
        else:
            print(f"  → ERREUR : {result.error}")

    return results


def main():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--session")
    grp.add_argument("--sessions_dir")
    p.add_argument("--side",       default="both", choices=["left","right","both"])
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_lag_s",  type=float, default=3.0)
    p.add_argument("--grid_ms",    type=float, default=1.0)
    args = p.parse_args()

    cfg   = AlignConfig(max_lag_s=args.max_lag_s, grid_ms=args.grid_ms)
    sides = ["left","right"] if args.side == "both" else [args.side]
    base  = Path(args.output_dir) if args.output_dir else None

    sessions = ([args.session] if args.session else
                sorted(str(d) for d in Path(args.sessions_dir).iterdir()
                       if d.is_dir() and not d.name.startswith(".")))
    if len(sessions) > 1:
        print(f"Sessions : {len(sessions)}")

    ok = True
    for sess in sessions:
        try:
            run_session(sess, cfg, sides, base)
        except Exception as e:
            print(f"[ERREUR] {sess}: {e}"); traceback.print_exc(); ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
