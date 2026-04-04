#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
session_pinces.py  — Vérification d'alignement temporel pince/vidéo

PRINCIPE
--------
La caméra et le capteur tournent sur la même machine Windows et partagent
la même horloge Unix. L'alignement se mesure DIRECTEMENT par comparaison
des timestamps absolus :

  - Vidéo   : videos/{side}.jsonl  → capture_time (ms) × 1 000 000 = ns
  - Capteur : gripper_{side}_data.csv → timestamp_ns (ns)

Il n'y a pas de cross-corrélation signal. On mesure des grandeurs précises :

  [M1] offset_start_ms  : décalage vidéo − capteur au premier instant commun
  [M2] drift_ms         : dérive totale sur la durée de la session
  [M3] drift_rate_ms_s  : taux de dérive (ms par seconde)
  [M4] jitter_std_ms    : stabilité des intervalles inter-frames (écart-type)
  [M5] frame_drops      : nombre de sauts > 2 × période nominale
  [M6] sensor_neg_dt    : sauts temporels négatifs dans le capteur
  [M7] sensor_vel_max   : vitesse max d'ouverture (mm/s) — anomalie si > MAX_VEL
  [M8] overlap_s        : durée du recouvrement vidéo/capteur

Les seuils d'alerte sont configurables (voir DEFAULT_THRESHOLDS).

USAGE
-----
  python session_pinces.py --sessions_dir /path/to/sessions [options]

  --test           Active le mode rapport complet (fichiers + graphiques)
  --output_dir     Répertoire de sortie (défaut : pinces_results)
  --tolerance_offset_ms   Seuil alerte offset début (défaut : 200 ms)
  --tolerance_drift_ms    Seuil alerte dérive totale (défaut : 100 ms)
  --tolerance_drift_rate  Seuil alerte taux dérive ms/s (défaut : 2.0)
  --tolerance_jitter_ms   Seuil alerte jitter std (défaut : 15 ms)
  --max_frame_drops       Seuil alerte nombre drops (défaut : 5)
  --max_vel_mm_s          Vitesse max physique pince mm/s (défaut : 2000)
  --min_overlap_s         Recouvrement minimum requis (défaut : 5 s)
"""

import os
import sys
import json
import argparse
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

NOMINAL_FPS      = 30.0          # FPS nominal des caméras
NOMINAL_PERIOD_S = 1.0 / NOMINAL_FPS   # 33.33 ms
DROP_THRESHOLD_S = 2.5 * NOMINAL_PERIOD_S   # >83 ms = saut probable


# ─────────────────────────────────────────────────────────────────────────────
# Seuils d'alerte (tous modifiables via argparse)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Thresholds:
    offset_ms:      float = 200.0  # |offset démarrage vidéo vs capteur| max (ms)
    latency_max_ms: float = 25.0   # latence max frame→capteur (ms)
    #  À 60Hz dt≈16.3ms, latence théorique max ≈ 8ms.
    #  25ms = tolérance pour un gap capteur pouvant aller jusqu'à 3× dt (49ms)
    #  Au-delà, le capteur interpolé sera trop imprécis pour être exploitable.
    jitter_std_ms:  float = 15.0   # jitter MAD timestamps vidéo (ms)
    max_drops:      int   = 5      # nb de frame drops autorisés
    max_vel_mm_s:   float = 2000.0 # vitesse ouverture physique max (mm/s)
    min_overlap_s:  float = 3.0    # recouvrement vidéo/capteur minimum (s)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement timestamps JSONL
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_timestamps(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse videos/{side}.jsonl (format CRLF Windows).

    Retourne:
        indices   : np.ndarray int32  — index de frame
        ts_ns     : np.ndarray int64  — timestamp en nanosecondes (horloge Unix)

    capture_time est en **millisecondes** → × 1 000 000 pour obtenir ns.
    """
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

    # Trier par index croissant
    order   = np.argsort(indices)
    return indices[order], ts_ns[order]


# ─────────────────────────────────────────────────────────────────────────────
# Chargement signal capteur
# ─────────────────────────────────────────────────────────────────────────────

def load_sensor(path: str) -> pd.DataFrame:
    """
    Parse gripper_{side}_data.csv.

    Retourne un DataFrame trié par timestamp_ns avec :
        timestamp_ns  : int64  — horloge Unix absolue (ns)
        opening_mm    : float  — ouverture mesurée (mm)
        dt_ms         : float  — intervalle depuis échantillon précédent (ms)
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Correction capteur : comblement de gaps et extrapolation arrière
# ─────────────────────────────────────────────────────────────────────────────

def fill_sensor_gaps(
    ts_ns: np.ndarray,
    opening_mm: np.ndarray,
    dt_nominal_ms: float = 17.0,
    max_opening_change_mm: float = 2.0,
    max_gap_ms: float = 1500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Comble les gaps dans le signal capteur par interpolation.

    Règle de remplissage :
    - Gap < 4 × dt_nominal_ms → ignoré (jitter normal)
    - Gap entre 4 × dt_nominal et max_gap_ms et |Δopening| ≤ max_opening_change_mm
      → interpolation linéaire (≈ constante si pince immobile)
    - Gap > max_gap_ms ou |Δopening| > max_opening_change_mm
      → gap conservé (signal dynamique ou trop long pour interpoler sans risque)

    Retourne (ts_ns_filled, opening_mm_filled) triés par timestamp.
    """
    if len(ts_ns) < 2:
        return ts_ns.copy(), opening_mm.copy()

    gap_thresh_ns = 4.0 * dt_nominal_ms * 1e6   # ns
    max_gap_ns    = max_gap_ms * 1e6              # ns

    filled_ts: list = list(ts_ns)
    filled_op: list = list(opening_mm.astype(float))

    diffs_ns = np.diff(ts_ns)

    for i, gap_ns in enumerate(diffs_ns):
        if gap_ns <= gap_thresh_ns:
            continue

        gap_ms        = gap_ns / 1e6
        op_before     = float(opening_mm[i])
        op_after      = float(opening_mm[i + 1])
        delta_opening = abs(op_after - op_before)

        if gap_ms > max_gap_ms or delta_opening > max_opening_change_mm:
            # Gap trop long ou signal dynamique → ne pas interpoler
            continue

        # Nombre de points synthétiques à insérer
        n = max(1, int(round(gap_ms / dt_nominal_ms)) - 1)

        t_start = ts_ns[i]
        t_end   = ts_ns[i + 1]
        synth_ts = np.linspace(
            t_start + dt_nominal_ms * 1e6,
            t_end   - dt_nominal_ms * 1e6,
            n,
            dtype=np.int64,
        )
        synth_op = np.linspace(op_before, op_after, n)

        filled_ts.extend(synth_ts.tolist())
        filled_op.extend(synth_op.tolist())

    order = np.argsort(filled_ts)
    return (
        np.array(filled_ts, dtype=np.int64)[order],
        np.array(filled_op, dtype=np.float64)[order],
    )


def extend_sensor_backward(
    ts_ns: np.ndarray,
    opening_mm: np.ndarray,
    vid_start_ns: int,
    dt_nominal_ms: float = 17.0,
    max_extrap_ms: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Étend le signal capteur vers l'arrière si la vidéo commence avant le capteur.

    Utilisé pour résoudre FRAMES_NO_SENSOR quand le capteur démarre quelques ms
    après la première frame vidéo (trigger légèrement anticipé).

    Limite max_extrap_ms : au-delà, on ne sait pas ce que la pince faisait.
    On extrapole à valeur constante = première valeur connue du capteur.
    """
    sensor_start_ns = ts_ns[0]
    gap_ms = (sensor_start_ns - vid_start_ns) / 1e6

    if gap_ms <= 0 or gap_ms > max_extrap_ms:
        return ts_ns, opening_mm

    dt_ns = int(dt_nominal_ms * 1e6)
    synth_ts = np.arange(vid_start_ns, sensor_start_ns, dt_ns, dtype=np.int64)

    if len(synth_ts) == 0:
        return ts_ns, opening_mm

    synth_op = np.full(len(synth_ts), float(opening_mm[0]))

    ts_out  = np.concatenate([synth_ts, ts_ns])
    op_out  = np.concatenate([synth_op, opening_mm.astype(float)])
    return ts_out, op_out


def apply_sensor_fixes(
    sensor_df: pd.DataFrame,
    vid_start_ns: int,
    dt_nominal_ms: float = 17.0,
) -> pd.DataFrame:
    """
    Applique les deux corrections au DataFrame capteur :
    1. Comblement des gaps statiques (fill_sensor_gaps)
    2. Extrapolation arrière si la vidéo commence avant le capteur (extend_sensor_backward)

    Retourne un nouveau DataFrame avec les mêmes colonnes que sensor_df.
    """
    ts     = sensor_df["timestamp_ns"].values.astype(np.int64)
    op     = sensor_df["opening_mm"].values.astype(np.float64)

    ts, op = fill_sensor_gaps(ts, op, dt_nominal_ms=dt_nominal_ms)
    ts, op = extend_sensor_backward(ts, op, vid_start_ns, dt_nominal_ms=dt_nominal_ms)

    dt = np.diff(ts, prepend=ts[0]).astype(float) / 1e6
    return pd.DataFrame({"timestamp_ns": ts, "opening_mm": op, "dt_ms": dt})


# ─────────────────────────────────────────────────────────────────────────────
# Métriques timestamps vidéo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoTimestampMetrics:
    n_frames:       int
    duration_s:     float
    dt_mean_ms:     float
    dt_std_ms:      float
    dt_min_ms:      float
    dt_max_ms:      float
    frame_drops:    int        # nb d'intervalles > DROP_THRESHOLD_S × 1000
    missing_indices: int       # indices manquants dans la séquence
    jitter_std_ms:  float      # robuste : mad × 1.4826
    ts_ns:          np.ndarray = field(repr=False)
    indices:        np.ndarray = field(repr=False)


def analyze_video_timestamps(indices: np.ndarray, ts_ns: np.ndarray) -> VideoTimestampMetrics:
    dt_ms = np.diff(ts_ns) / 1e6

    drops = int((dt_ms > DROP_THRESHOLD_S * 1000).sum())

    # Indices manquants
    expected = np.arange(indices[0], indices[-1] + 1)
    missing  = int(len(expected) - len(indices))

    # Jitter robuste (MAD)
    residuals = dt_ms - np.median(dt_ms)
    mad       = np.median(np.abs(residuals))
    jitter    = float(mad * 1.4826)

    return VideoTimestampMetrics(
        n_frames       = len(ts_ns),
        duration_s     = float((ts_ns[-1] - ts_ns[0]) / 1e9),
        dt_mean_ms     = float(dt_ms.mean()),
        dt_std_ms      = float(dt_ms.std()),
        dt_min_ms      = float(dt_ms.min()),
        dt_max_ms      = float(dt_ms.max()),
        frame_drops    = drops,
        missing_indices= missing,
        jitter_std_ms  = jitter,
        ts_ns          = ts_ns,
        indices        = indices,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Métriques capteur
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorMetrics:
    n_samples:      int
    duration_s:     float
    dt_mean_ms:     float
    dt_std_ms:      float
    neg_dt_count:   int        # sauts temporels négatifs (reset horloge)
    neg_dt_details: List[Dict] # liste des sauts {idx, dt_ms, delta_opening_mm}
    vel_max_mm_s:   float      # vitesse max absolue d'ouverture
    vel_p99_mm_s:   float      # 99e percentile vitesse
    vel_anomalies:  int        # nb de sauts > max_vel
    opening_range:  Tuple[float, float]


def analyze_sensor(df: pd.DataFrame, max_vel: float) -> SensorMetrics:
    ts   = df["timestamp_ns"].values
    op   = df["opening_mm"].values
    dt   = df["dt_ms"].values[1:]        # intervalles (n-1 valeurs)

    neg_idx = np.where(dt < 0)[0]
    neg_details = []
    for i in neg_idx:
        neg_details.append({
            "idx":             int(i),
            "dt_ms":           float(dt[i]),
            "opening_before":  float(op[i]),
            "opening_after":   float(op[i + 1]),
        })

    # Vitesse d'ouverture — ignorer les intervalles négatifs/nuls
    safe_dt_s = np.maximum(dt, 1.0) / 1000.0   # clamp à 1ms minimum
    vel       = np.abs(np.diff(op)) / safe_dt_s
    vel_max   = float(vel.max()) if len(vel) else 0.0
    vel_p99   = float(np.percentile(vel, 99)) if len(vel) else 0.0
    vel_anom  = int((vel > max_vel).sum())

    return SensorMetrics(
        n_samples      = len(df),
        duration_s     = float((ts[-1] - ts[0]) / 1e9),
        dt_mean_ms     = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.0,
        dt_std_ms      = float(dt[dt > 0].std()) if (dt > 0).sum() > 1 else 0.0,
        neg_dt_count   = int(len(neg_idx)),
        neg_dt_details = neg_details,
        vel_max_mm_s   = vel_max,
        vel_p99_mm_s   = vel_p99,
        vel_anomalies  = vel_anom,
        opening_range  = (float(op.min()), float(op.max())),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alignement horloge vidéo / capteur
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlignmentMetrics:
    # ── Durées ────────────────────────────────────────────────────────────────
    dur_vid_s:          float   # durée enregistrement vidéo (s)
    dur_sensor_s:       float   # durée enregistrement capteur (s)
    overlap_s:          float   # durée du recouvrement commun (s)

    # ── Offset de démarrage ───────────────────────────────────────────────────
    # t_video[0] − t_sensor[0] : délai entre démarrage capteur et démarrage vidéo
    # Positif = vidéo commence après capteur (normal, capteur démarre en premier)
    # Négatif = vidéo commence avant capteur (anormal)
    offset_start_ms:    float

    # ── Latence par frame : distance temporelle frame → échantillon capteur le plus proche ──
    # Mesure la précision d'alignement pour CHAQUE frame.
    # Les deux clocks étant sur la même machine Unix, cette latence mesure
    # uniquement le "temps de quantification" dû aux fréquences différentes.
    latency_mean_ms:    float   # moyenne signée (devrait être ≈ 0)
    latency_std_ms:     float   # std (devrait être ≈ dt_sensor/2 ≈ 8ms à 60Hz)
    latency_max_abs_ms: float   # |latence| max sur toutes les frames
    latency_p95_abs_ms: float   # 95e percentile de |latence|
    frames_no_sensor:   int     # frames hors plage capteur (interpolation impossible)

    # ── Continuité capteur dans la fenêtre vidéo ─────────────────────────────
    # Max gap entre deux échantillons capteur pendant la fenêtre vidéo
    # Seuil : > 2× dt_nominal (33ms à 60Hz) = trou problématique
    sensor_gap_max_ms:  float   # max gap capteur dans fenêtre vidéo (ms)
    sensor_gap_count:   int     # nb de gaps > 2× dt_nominal

    # ── Jitter vidéo (linfit) ─────────────────────────────────────────────────
    # Modèle linéaire sur les timestamps vidéo pour détecter jitter non-uniforme
    linfit_slope:       float   # ≈ 1.0 si clock uniforme
    linfit_r2:          float   # R² (≈ 1.0 si très régulier)
    linfit_residual_std_ms: float   # std des résidus (ms) - jitter autour du modèle linéaire
    linfit_residual_max_ms: float   # max résidu

    # ── Interpolation capteur aux instants vidéo ──────────────────────────────
    opening_at_frames:  np.ndarray = field(repr=False)
    frame_ts_ns:        np.ndarray = field(repr=False)


def compute_alignment(
    vid: VideoTimestampMetrics,
    sensor_df: pd.DataFrame,
) -> AlignmentMetrics:
    tv = vid.ts_ns.astype(np.float64)           # ns — horloge Unix Windows
    ts = sensor_df["timestamp_ns"].values.astype(np.float64)
    opening = sensor_df["opening_mm"].values

    # ── Durées et overlap ────────────────────────────────────────────────────
    dur_vid_s    = float((tv[-1] - tv[0]) / 1e9)
    dur_sensor_s = float((ts[-1] - ts[0]) / 1e9)
    t_ov0 = max(tv[0], ts[0])
    t_ov1 = min(tv[-1], ts[-1])
    overlap_s = float((t_ov1 - t_ov0) / 1e9) if t_ov1 > t_ov0 else 0.0

    # ── Offset de démarrage ──────────────────────────────────────────────────
    # Les deux clocks sont Unix sur la même machine Windows → même référence absolue.
    # offset_start = combien la vidéo démarre après le capteur.
    offset_start_ms = float((tv[0] - ts[0]) / 1e6)

    # ── Latence frame → capteur ──────────────────────────────────────────────
    # Pour chaque frame vidéo, on trouve l'échantillon capteur le plus proche dans le temps.
    # La latence = t_frame - t_sensor_nearest  (positif = frame légèrement en avance)
    # Cette mesure est exacte car les deux timestamps sont dans le même référentiel Unix.
    nearest_hi = np.searchsorted(ts, tv)   # index du premier ts >= tv[i]
    nearest_hi = np.clip(nearest_hi, 0, len(ts) - 1)
    nearest_lo = np.clip(nearest_hi - 1, 0, len(ts) - 1)
    d_hi = np.abs(tv - ts[nearest_hi])
    d_lo = np.abs(tv - ts[nearest_lo])
    nearest_ts = np.where(d_hi <= d_lo, ts[nearest_hi], ts[nearest_lo])
    latency_ms = (tv - nearest_ts) / 1e6  # ms

    # Frames hors plage capteur (pas d'interpolation possible)
    in_range = (tv >= ts[0]) & (tv <= ts[-1])
    n_no_sensor = int((~in_range).sum())

    # Stats latence (uniquement sur frames avec capteur)
    lat_valid = latency_ms[in_range]
    if len(lat_valid) > 0:
        lat_mean   = float(lat_valid.mean())
        lat_std    = float(lat_valid.std())
        lat_max    = float(np.abs(lat_valid).max())
        lat_p95    = float(np.percentile(np.abs(lat_valid), 95))
    else:
        lat_mean = lat_std = lat_max = lat_p95 = np.nan

    # ── Continuité capteur dans la fenêtre vidéo ──────────────────────────────
    # Chercher des trous dans le flux capteur pendant l'enregistrement vidéo
    mask_sen_in_vid = (ts >= tv[0]) & (ts <= tv[-1])
    ts_in_vid = ts[mask_sen_in_vid]
    dt_nominal_ms = float(np.median(np.diff(ts)) / 1e6)  # période nominale capteur
    if len(ts_in_vid) > 1:
        gaps_ms = np.diff(ts_in_vid) / 1e6
        # Seuil: 3× la période nominale (plutôt que 2×) pour éviter les faux positifs
        # sur jitter normal du capteur. À 60Hz (16.3ms): seuil = 48.9ms
        # Seuil à 4× dt_nominal : à 60Hz (16.3ms) → 65ms.
        # En dessous c'est du jitter normal ou un seul sample manquant, pas un gap problématique.
        gap_thresh = 4.0 * dt_nominal_ms
        sensor_gap_max_ms = float(gaps_ms.max())
        sensor_gap_count  = int((gaps_ms > gap_thresh).sum())
    else:
        sensor_gap_max_ms = float('inf')
        sensor_gap_count  = 0

    # ── Modèle linéaire sur timestamps vidéo ─────────────────────────────────
    # Régression linéaire t_video[i] = a * i + b
    # slope ≈ dt_nominal_ns, R² proche de 1.0 → timestamps très réguliers
    # std résidus mesure le jitter autour du modèle linéaire
    if len(tv) > 10:
        idx = np.arange(len(tv), dtype=np.float64)
        slope_ns, intercept_ns, r, _, _ = linregress(idx, tv)
        r2 = r ** 2
        fitted = slope_ns * idx + intercept_ns
        residuals_ms = (tv - fitted) / 1e6
        lf_std = float(residuals_ms.std())
        lf_max = float(np.abs(residuals_ms).max())
    else:
        slope_ns = (tv[-1] - tv[0]) / max(len(tv) - 1, 1)
        r2 = np.nan
        lf_std = lf_max = np.nan

    # ── Interpolation capteur aux instants vidéo ─────────────────────────────
    from scipy.interpolate import interp1d
    f_opening = interp1d(
        ts, opening,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    opening_at_frames = f_opening(tv)

    return AlignmentMetrics(
        dur_vid_s           = dur_vid_s,
        dur_sensor_s        = dur_sensor_s,
        overlap_s           = overlap_s,
        offset_start_ms     = offset_start_ms,
        latency_mean_ms     = lat_mean,
        latency_std_ms      = lat_std,
        latency_max_abs_ms  = lat_max,
        latency_p95_abs_ms  = lat_p95,
        frames_no_sensor    = n_no_sensor,
        sensor_gap_max_ms   = sensor_gap_max_ms,
        sensor_gap_count    = sensor_gap_count,
        linfit_slope        = float(slope_ns),
        linfit_r2           = float(r2),
        linfit_residual_std_ms = lf_std,
        linfit_residual_max_ms = lf_max,
        opening_at_frames   = opening_at_frames,
        frame_ts_ns         = vid.ts_ns,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cohérence physique : capteur interpolé aux frames
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicalCoherenceMetrics:
    n_frames_with_sensor: int
    n_frames_no_sensor:   int    # interpolation hors plage
    opening_mean_mm:      float
    opening_std_mm:       float
    opening_range:        Tuple[float, float]
    d_opening_max_mm_s:   float  # vitesse max inter-frames (mm/s)
    d_opening_p99_mm_s:   float
    impossible_jumps:     int    # sauts > max_vel entre frames consécutives


def compute_physical_coherence(
    alignment: AlignmentMetrics,
    max_vel: float,
) -> PhysicalCoherenceMetrics:
    op    = alignment.opening_at_frames
    ts_ns = alignment.frame_ts_ns.astype(np.float64)

    valid  = np.isfinite(op)
    n_ok   = int(valid.sum())
    n_miss = int((~valid).sum())

    if n_ok < 2:
        return PhysicalCoherenceMetrics(
            n_frames_with_sensor=n_ok,
            n_frames_no_sensor=n_miss,
            opening_mean_mm=np.nan,
            opening_std_mm=np.nan,
            opening_range=(np.nan, np.nan),
            d_opening_max_mm_s=np.nan,
            d_opening_p99_mm_s=np.nan,
            impossible_jumps=0,
        )

    op_valid   = op[valid]
    ts_valid   = ts_ns[valid]
    dt_s       = np.diff(ts_valid) / 1e9
    d_op       = np.abs(np.diff(op_valid))
    vel        = d_op / np.maximum(dt_s, 1e-6)

    return PhysicalCoherenceMetrics(
        n_frames_with_sensor = n_ok,
        n_frames_no_sensor   = n_miss,
        opening_mean_mm      = float(op_valid.mean()),
        opening_std_mm       = float(op_valid.std()),
        opening_range        = (float(op_valid.min()), float(op_valid.max())),
        d_opening_max_mm_s   = float(vel.max()),
        d_opening_p99_mm_s   = float(np.percentile(vel, 99)),
        impossible_jumps     = int((vel > max_vel).sum()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alertes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    code:    str
    level:   str   # "ERROR" ou "WARNING"
    message: str
    value:   float
    threshold: float


def generate_alerts(
    vid:   VideoTimestampMetrics,
    sen:   SensorMetrics,
    aln:   AlignmentMetrics,
    phy:   PhysicalCoherenceMetrics,
    thr:   Thresholds,
) -> List[Alert]:
    alerts: List[Alert] = []

    def add(code, level, msg, value, threshold):
        alerts.append(Alert(code=code, level=level, message=msg,
                            value=value, threshold=threshold))

    # ── Offset de démarrage ──────────────────────────────────────────────────
    # Mesure t_video[0] - t_gripper[0].
    # Note: fix_camera_offset aligne la vidéo sur t_tracker[0], pas t_gripper[0].
    # Si tracker et gripper ne démarrent pas exactement au même instant (écart ≤ 1 frame ≈ 33ms),
    # l'offset peut légèrement diverger après fix. Tolérance: ±1 frame vidéo nominale.
    ONE_FRAME_MS = 1000.0 / NOMINAL_FPS   # ≈ 33.3ms
    abs_offset = abs(aln.offset_start_ms)
    if aln.offset_start_ms < -ONE_FRAME_MS:
        # Vidéo démarre significativement avant le capteur — anormal
        add("OFFSET_START", "ERROR",
            f"Vidéo démarre {aln.offset_start_ms:+.1f}ms avant le capteur — capteur manquant au début",
            abs_offset, ONE_FRAME_MS)
    elif abs_offset > thr.offset_ms:
        add("OFFSET_START", "ERROR",
            f"Offset démarrage {aln.offset_start_ms:+.1f}ms > seuil {thr.offset_ms:.0f}ms  "
            f"[corrigeable par fix_camera_offset]",
            abs_offset, thr.offset_ms)

    # ── Latence frame→capteur ────────────────────────────────────────────────
    # Mesure la précision de l'alignement : pour chaque frame, à quel point
    # l'échantillon capteur le plus proche est-il éloigné dans le temps.
    # À 60Hz (dt≈16.3ms), la latence max théorique est ≈ 8ms.
    # > 25ms = problème de synchronisation ou gap capteur.
    # Note: si SENSOR_GAP est déjà détecté (gap > 4×dt), LATENCY_MAX est conséquence
    # directe du gap — on l'élève en WARNING pour ne pas doubler l'erreur.
    if np.isfinite(aln.latency_max_abs_ms):
        if aln.latency_max_abs_ms > thr.latency_max_ms:
            has_gap = aln.sensor_gap_count > 0
            if has_gap:
                # Latence élevée causée par un gap capteur déjà signalé → WARNING seulement
                level = "WARNING"
                note = f"  (causée par gap capteur {aln.sensor_gap_max_ms:.0f}ms)"
            else:
                level = "ERROR" if aln.latency_max_abs_ms > thr.latency_max_ms * 2 else "WARNING"
                note = ""
            add("LATENCY_MAX", level,
                f"Latence max frame→capteur {aln.latency_max_abs_ms:.1f}ms > seuil {thr.latency_max_ms:.0f}ms  "
                f"(P95={aln.latency_p95_abs_ms:.1f}ms){note}",
                aln.latency_max_abs_ms, thr.latency_max_ms)

    # ── Frames sans capteur ──────────────────────────────────────────────────
    # Seuil : > 1 frame ET > 0.1% pour éviter les faux positifs sur 1 frame limite
    if aln.frames_no_sensor > 1:
        total_f = vid.n_frames
        frac = aln.frames_no_sensor / max(total_f, 1)
        level = "ERROR" if frac > 0.05 else "WARNING"
        add("FRAMES_NO_SENSOR", level,
            f"{aln.frames_no_sensor}/{total_f} frames ({frac*100:.1f}%) hors plage capteur",
            aln.frames_no_sensor, 0.0)

    # ── Trous capteur dans la fenêtre vidéo ──────────────────────────────────
    if aln.sensor_gap_count > 0:
        level = "ERROR" if aln.sensor_gap_max_ms > 100.0 else "WARNING"
        add("SENSOR_GAP", level,
            f"{aln.sensor_gap_count} trou(s) capteur dans la fenêtre vidéo — "
            f"gap max={aln.sensor_gap_max_ms:.1f}ms",
            aln.sensor_gap_max_ms, 0.0)

    # ── Recouvrement insuffisant ─────────────────────────────────────────────
    if aln.overlap_s < thr.min_overlap_s:
        add("OVERLAP_SHORT", "ERROR",
            f"Recouvrement vidéo/capteur {aln.overlap_s:.1f}s < minimum {thr.min_overlap_s:.0f}s",
            aln.overlap_s, thr.min_overlap_s)

    # ── Jitter vidéo (robuste MAD) ───────────────────────────────────────────
    if vid.jitter_std_ms > thr.jitter_std_ms:
        add("JITTER", "WARNING",
            f"Jitter timestamps vidéo {vid.jitter_std_ms:.2f}ms > seuil {thr.jitter_std_ms:.0f}ms",
            vid.jitter_std_ms, thr.jitter_std_ms)

    # ── Jitter linfit désactivé ───────────────────────────────────────────────
    # linfit_residual_std est systématiquement élevé en présence de gaps vidéo normaux
    # (pauses d'encodage, frame drops) car les résidus s'accumulent après chaque gap.
    # Le MAD jitter (ci-dessus) mesure la vraie stabilité de l'horloge et est suffisant.
    # LINFIT_JITTER génèrerait systématiquement des faux positifs sur données réelles.

    # ── Frame drops ──────────────────────────────────────────────────────────
    if vid.frame_drops > thr.max_drops:
        level = "ERROR" if vid.frame_drops > thr.max_drops * 3 else "WARNING"
        add("FRAME_DROPS", level,
            f"{vid.frame_drops} frame drops > seuil {thr.max_drops}",
            vid.frame_drops, thr.max_drops)

    # ── Indices manquants ────────────────────────────────────────────────────
    if vid.missing_indices > 0:
        add("MISSING_FRAMES", "WARNING",
            f"{vid.missing_indices} indices manquants dans le flux JSONL",
            vid.missing_indices, 0.0)

    # ── Timestamps négatifs capteur ──────────────────────────────────────────
    if sen.neg_dt_count > 0:
        level = "ERROR" if sen.neg_dt_count > 2 else "WARNING"
        add("SENSOR_NEG_DT", level,
            f"{sen.neg_dt_count} saut(s) temporel(s) négatif(s) dans le capteur (reset horloge ?)",
            sen.neg_dt_count, 0.0)

    # ── Anomalies de vitesse capteur ─────────────────────────────────────────
    if sen.vel_anomalies > 0:
        add("SENSOR_VEL_ANOMALY", "WARNING",
            f"{sen.vel_anomalies} saut(s) impossible(s) dans capteur (vel > {thr.max_vel_mm_s:.0f}mm/s)",
            sen.vel_max_mm_s, thr.max_vel_mm_s)

    # ── Sauts impossibles dans les données interpolées ───────────────────────
    if phy.impossible_jumps > 0:
        add("INTERP_VEL_ANOMALY", "WARNING",
            f"{phy.impossible_jumps} saut(s) impossible(s) dans le capteur interpolé aux frames",
            phy.impossible_jumps, 0.0)

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Résultat complet d'un côté
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SideResult:
    session_name: str
    side:         str
    success:      bool
    error:        str                        = ""
    video:        Optional[VideoTimestampMetrics]    = None
    sensor:       Optional[SensorMetrics]            = None
    alignment:    Optional[AlignmentMetrics]         = None
    physical:     Optional[PhysicalCoherenceMetrics] = None
    alerts:       List[Alert]                        = field(default_factory=list)
    has_errors:   bool                               = False
    sensor_df:    Optional[pd.DataFrame]             = field(default=None, repr=False)

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


# ─────────────────────────────────────────────────────────────────────────────
# Traitement d'un côté dans une session
# ─────────────────────────────────────────────────────────────────────────────

def process_side(
    session_path: str,
    side: str,
    thr: Thresholds,
) -> SideResult:
    session_name = os.path.basename(session_path)
    vdir         = os.path.join(session_path, "videos")
    jsonl_path   = os.path.join(vdir, f"{side}.jsonl")
    sensor_path  = os.path.join(session_path, f"gripper_{side}_data.csv")

    missing = [p for p in [jsonl_path, sensor_path] if not os.path.exists(p)]
    if missing:
        names = [os.path.basename(p) for p in missing]
        # Distinguer capture incomplète (fichiers jamais créés) de corruption
        error_msg = f"Session incomplète — fichiers absents : {names}"
        return SideResult(
            session_name=session_name, side=side, success=False,
            error=error_msg,
        )

    try:
        indices, ts_ns = load_jsonl_timestamps(jsonl_path)
        sensor_df      = load_sensor(sensor_path)

        # Calcul dt_nominal avant les corrections (sur données brutes triées)
        dt_nominal_ms = float(np.median(np.diff(sensor_df["timestamp_ns"].values)) / 1e6)

        # Corrections capteur : comblement gaps statiques + extrapolation arrière
        sensor_df = apply_sensor_fixes(sensor_df, int(ts_ns[0]), dt_nominal_ms)

        vid = analyze_video_timestamps(indices, ts_ns)
        sen = analyze_sensor(sensor_df, thr.max_vel_mm_s)
        aln = compute_alignment(vid, sensor_df)
        phy = compute_physical_coherence(aln, thr.max_vel_mm_s)
        als = generate_alerts(vid, sen, aln, phy, thr)

        return SideResult(
            session_name = session_name,
            side         = side,
            success      = True,
            video        = vid,
            sensor       = sen,
            alignment    = aln,
            physical     = phy,
            alerts       = als,
            has_errors   = any(a.level == "ERROR" for a in als),
            sensor_df    = sensor_df,
        )

    except Exception as exc:
        return SideResult(
            session_name=session_name, side=side, success=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rapport texte par côté
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, fmt=".3f", unit=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:{fmt}}{unit}"


def write_side_report(result: SideResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:

        def w(line=""):
            f.write(line + "\n")

        w(f"{'='*70}")
        w(f"RAPPORT ALIGNEMENT PINCE")
        w(f"Session : {result.session_name}")
        w(f"Côté    : {result.side.upper()}")
        w(f"Statut  : {result.status}")
        w(f"{'='*70}")
        w()

        if not result.success:
            w(f"ECHEC : {result.error}")
            return

        # ── Alertes ──────────────────────────────────────────────────────────
        if result.alerts:
            w(f"{'─'*70}")
            w(f"ALERTES ({result.n_errors} erreur(s), {result.n_warnings} avertissement(s))")
            w(f"{'─'*70}")
            for a in result.alerts:
                w(f"  [{a.level:7s}] [{a.code}]")
                w(f"           {a.message}")
            w()

        # ── Timestamps vidéo ─────────────────────────────────────────────────
        vid = result.video
        w(f"{'─'*70}")
        w("TIMESTAMPS VIDÉO")
        w(f"{'─'*70}")
        w(f"  Frames analysées       : {vid.n_frames}")
        w(f"  Durée couverte         : {vid.duration_s:.3f} s")
        w(f"  Intervalle inter-frames: moy={vid.dt_mean_ms:.2f}ms, "
          f"std={vid.dt_std_ms:.2f}ms, "
          f"[{vid.dt_min_ms:.1f}ms – {vid.dt_max_ms:.1f}ms]")
        w(f"  Jitter robuste (MAD)   : {vid.jitter_std_ms:.3f} ms")
        w(f"  Frame drops (>83ms)    : {vid.frame_drops}")
        w(f"  Indices manquants      : {vid.missing_indices}")
        w()

        # ── Capteur ───────────────────────────────────────────────────────────
        sen = result.sensor
        w(f"{'─'*70}")
        w("SIGNAL CAPTEUR")
        w(f"{'─'*70}")
        w(f"  Échantillons           : {sen.n_samples}")
        w(f"  Durée couverte         : {sen.duration_s:.3f} s")
        w(f"  Fréquence médiane      : {1000/max(sen.dt_mean_ms,1e-6):.1f} Hz  "
          f"(dt médian = {sen.dt_mean_ms:.2f} ms)")
        w(f"  Sauts dt négatifs      : {sen.neg_dt_count}")
        for d in sen.neg_dt_details:
            w(f"    idx={d['idx']}: dt={d['dt_ms']:.1f}ms, "
              f"opening={d['opening_before']:.1f}→{d['opening_after']:.1f}mm")
        w(f"  Ouverture plage        : [{sen.opening_range[0]:.1f}, {sen.opening_range[1]:.1f}] mm")
        w(f"  Vitesse max            : {sen.vel_max_mm_s:.0f} mm/s  "
          f"(P99 = {sen.vel_p99_mm_s:.0f} mm/s)")
        w(f"  Anomalies vitesse      : {sen.vel_anomalies}")
        w()

        # ── Alignement ────────────────────────────────────────────────────────
        aln = result.alignment
        w(f"{'─'*70}")
        w("ALIGNEMENT HORLOGE")
        w(f"{'─'*70}")
        w(f"  Durée vidéo            : {aln.dur_vid_s:.3f} s")
        w(f"  Durée capteur          : {aln.dur_sensor_s:.3f} s")
        w(f"  Recouvrement           : {aln.overlap_s:.3f} s")
        w()
        w(f"  Offset démarrage       : {aln.offset_start_ms:+.3f} ms  "
          f"(t_video[0] − t_sensor[0])")
        w(f"    → vidéo démarre {aln.offset_start_ms:+.0f}ms après le capteur")
        w()
        w(f"  Latence frame→capteur  :")
        w(f"    moy                  : {_fmt(aln.latency_mean_ms, '+.3f', ' ms')}")
        w(f"    std                  : {_fmt(aln.latency_std_ms, '.3f', ' ms')}")
        w(f"    max |latence|        : {_fmt(aln.latency_max_abs_ms, '.3f', ' ms')}")
        w(f"    P95 |latence|        : {_fmt(aln.latency_p95_abs_ms, '.3f', ' ms')}")
        w(f"    frames hors capteur  : {aln.frames_no_sensor}")
        w()
        w(f"  Continuité capteur (fenêtre vidéo) :")
        w(f"    gap max              : {_fmt(aln.sensor_gap_max_ms, '.1f', ' ms')}")
        w(f"    nb gaps >2×dt_nom    : {aln.sensor_gap_count}")
        w()
        w(f"  Régularité horloge vidéo (modèle linéaire) :")
        w(f"    slope                : {aln.linfit_slope:.3f} ns/frame")
        w(f"    R²                   : {_fmt(aln.linfit_r2, '.6f')}")
        w(f"    résidus std          : {_fmt(aln.linfit_residual_std_ms, '.3f', ' ms')}")
        w(f"    résidus max          : {_fmt(aln.linfit_residual_max_ms, '.3f', ' ms')}")
        w()

        # ── Cohérence physique ────────────────────────────────────────────────
        phy = result.physical
        w(f"{'─'*70}")
        w("COHÉRENCE PHYSIQUE (capteur interpolé aux instants vidéo)")
        w(f"{'─'*70}")
        w(f"  Frames avec capteur    : {phy.n_frames_with_sensor}")
        w(f"  Frames sans capteur    : {phy.n_frames_no_sensor}")
        w(f"  Ouverture moy/std      : {_fmt(phy.opening_mean_mm, '.2f')} ± "
          f"{_fmt(phy.opening_std_mm, '.2f')} mm")
        w(f"  Plage ouverture        : [{_fmt(phy.opening_range[0], '.2f')}, "
          f"{_fmt(phy.opening_range[1], '.2f')}] mm")
        w(f"  Vitesse max inter-fr.  : {_fmt(phy.d_opening_max_mm_s, '.0f', ' mm/s')}")
        w(f"  Vitesse P99 inter-fr.  : {_fmt(phy.d_opening_p99_mm_s, '.0f', ' mm/s')}")
        w(f"  Sauts impossibles      : {phy.impossible_jumps}")
        w()


# ─────────────────────────────────────────────────────────────────────────────
# Rapport global
# ─────────────────────────────────────────────────────────────────────────────

def write_global_report(
    all_results: List[SideResult],
    path: str,
    thr: Thresholds,
) -> None:
    ok       = [r for r in all_results if r.success and r.status == "OK"]
    warnings = [r for r in all_results if r.success and r.status == "WARNING"]
    errors   = [r for r in all_results if r.success and r.status == "ERROR"]
    failed   = [r for r in all_results if not r.success]

    with open(path, "w", encoding="utf-8") as f:
        def w(line=""):
            f.write(line + "\n")

        w("=" * 72)
        w("RAPPORT GLOBAL — VÉRIFICATION ALIGNEMENT PINCES")
        w("=" * 72)
        w(f"  Total analysés  : {len(all_results)}")
        w(f"  OK              : {len(ok)}")
        w(f"  WARNINGS        : {len(warnings)}")
        w(f"  ERRORS          : {len(errors)}")
        w(f"  FAILED          : {len(failed)}")
        w()
        w("Seuils utilisés :")
        w(f"  offset_start_ms  : {thr.offset_ms:.0f} ms  (délai démarrage vidéo/capteur)")
        w(f"  latency_max_ms   : {thr.latency_max_ms:.0f} ms  (distance max frame→capteur, théorique ≈8ms à 60Hz)")
        w(f"  jitter_std_ms    : {thr.jitter_std_ms:.0f} ms  (stabilité timestamps vidéo)")
        w(f"  max_frame_drops  : {thr.max_drops}")
        w(f"  max_vel_mm_s     : {thr.max_vel_mm_s:.0f} mm/s")
        w(f"  min_overlap_s    : {thr.min_overlap_s:.0f} s  (durée minimale session valide)")
        w()

        # Statistiques globales sur les sessions réussies
        success_list = [r for r in all_results if r.success and r.alignment]
        if success_list:
            offsets   = [r.alignment.offset_start_ms    for r in success_list]
            latencies = [r.alignment.latency_max_abs_ms for r in success_list
                         if np.isfinite(r.alignment.latency_max_abs_ms)]
            lat_p95s  = [r.alignment.latency_p95_abs_ms for r in success_list
                         if np.isfinite(r.alignment.latency_p95_abs_ms)]
            jitters   = [r.video.jitter_std_ms          for r in success_list]
            overlaps  = [r.alignment.overlap_s           for r in success_list]
            gap_maxs  = [r.alignment.sensor_gap_max_ms  for r in success_list
                         if np.isfinite(r.alignment.sensor_gap_max_ms)]

            w("─" * 72)
            w("STATISTIQUES GLOBALES")
            w("─" * 72)

            def stat_row(label, values, fmt=".2f", unit="ms"):
                if not values:
                    w(f"  {label:<32} (aucune donnée)")
                    return
                arr = np.array(values)
                mean_ = np.mean(arr)
                std_  = np.std(arr)
                med_  = np.median(arr)
                p95_  = np.percentile(np.abs(arr), 95)
                outliers = [v for v in values if abs(v - mean_) > 3 * std_ + 1e-9]
                line = (f"  {label:<32} moy={mean_:{fmt}}{unit}, "
                        f"med={med_:{fmt}}{unit}, "
                        f"P95|.|={p95_:{fmt}}{unit}, "
                        f"[{np.min(arr):{fmt}}, {np.max(arr):{fmt}}]{unit}")
                if outliers:
                    line += f"  ⚠ {len(outliers)} outlier(s)"
                w(line)

            stat_row("Offset démarrage",     offsets)
            stat_row("Latence max |frame→capteur|", latencies)
            stat_row("Latence P95 |frame→capteur|", lat_p95s)
            stat_row("Jitter vidéo (MAD)",   jitters)
            stat_row("Gap capteur max",       gap_maxs)
            stat_row("Recouvrement",          overlaps, ".1f", "s")
            w()

        # Détail par session
        w("─" * 72)
        w("DÉTAIL PAR SESSION")
        w("─" * 72)

        sessions: Dict[str, List[SideResult]] = {}
        for r in all_results:
            sessions.setdefault(r.session_name, []).append(r)

        for sname, res_list in sorted(sessions.items()):
            w(f"\n{sname}")
            for r in res_list:
                if not r.success:
                    w(f"  [{r.side:5s}] FAILED  {r.error[:80]}")
                    continue
                aln = r.alignment
                vid = r.video
                alerts_str = (
                    f"{r.n_errors}E/{r.n_warnings}W "
                    + " ".join(f"[{a.code}]" for a in r.alerts[:3])
                    + ("..." if len(r.alerts) > 3 else "")
                ) if r.alerts else "—"
                lat_str = f"{aln.latency_max_abs_ms:.1f}" if np.isfinite(aln.latency_max_abs_ms) else "N/A"
                gap_str = f"{aln.sensor_gap_max_ms:.1f}" if np.isfinite(aln.sensor_gap_max_ms) else "N/A"
                w(
                    f"  [{r.side:5s}] {r.status:<8} "
                    f"off={aln.offset_start_ms:+7.1f}ms  "
                    f"lat_max={lat_str:>6}ms  "
                    f"gap_max={gap_str:>6}ms  "
                    f"jit={vid.jitter_std_ms:5.2f}ms  "
                    f"drops={vid.frame_drops:2d}  "
                    f"neg_dt={r.sensor.neg_dt_count}  "
                    f"alerts={alerts_str}"
                )

        # Section des sessions avec problèmes graves
        if failed or errors:
            w()
            w("─" * 72)
            w("PROBLÈMES CRITIQUES")
            w("─" * 72)
            if failed:
                w()
                w("  SESSIONS INCOMPLÈTES (fichiers manquants) :")
                for r in failed:
                    w(f"    {r.session_name} / {r.side} — {r.error}")
            if errors:
                w()
                w("  SESSIONS AVEC ERREURS DE DONNÉES (hardware/capture) :")
                for r in errors:
                    w(f"    {r.session_name} / {r.side}")
                    for a in r.alerts:
                        if a.level == "ERROR":
                            w(f"      [{a.code}] {a.message}")


# ─────────────────────────────────────────────────────────────────────────────
# Graphiques
# ─────────────────────────────────────────────────────────────────────────────

def save_plot(result: SideResult, path: str) -> None:
    if not result.success or result.alignment is None:
        return

    aln = result.alignment
    vid = result.video
    sen = result.sensor
    sensor_df = result.sensor_df

    t_ref_ns   = float(aln.frame_ts_ns[0])
    t_video_s  = (aln.frame_ts_ns.astype(float) - t_ref_ns) / 1e9
    op_frames  = aln.opening_at_frames

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
    fig.suptitle(
        f"{result.session_name}  |  côté {result.side.upper()}  |  {result.status}\n"
        f"offset={aln.offset_start_ms:+.1f}ms  "
        f"drift={aln.drift_ms:+.1f}ms  "
        f"rate={aln.drift_rate_ms_s:+.3f}ms/s  "
        f"jitter={vid.jitter_std_ms:.2f}ms  "
        f"drops={vid.frame_drops}",
        fontsize=11
    )

    # ── Axe 1 : intervalles inter-frames ─────────────────────────────────────
    ax = axes[0]
    dt_ms = np.diff(aln.frame_ts_ns) / 1e6
    t_mid = (t_video_s[:-1] + t_video_s[1:]) / 2
    ax.plot(t_mid, dt_ms, linewidth=0.8, color="steelblue", label="Δt frames (ms)")
    ax.axhline(1000 / NOMINAL_FPS, color="green", linestyle="--", linewidth=1,
               label=f"Nominal {1000/NOMINAL_FPS:.1f}ms")
    ax.axhline(DROP_THRESHOLD_S * 1000, color="red", linestyle=":", linewidth=1,
               label=f"Drop seuil {DROP_THRESHOLD_S*1000:.0f}ms")
    ax.set_ylabel("Δt (ms)")
    ax.set_xlabel("Temps vidéo (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Axe 2 : signal capteur + interpolation sur frames ────────────────────
    ax = axes[1]
    if sensor_df is not None:
        t_sensor_s = (sensor_df["timestamp_ns"].values.astype(float) - t_ref_ns) / 1e9
        ax.plot(t_sensor_s, sensor_df["opening_mm"].values,
                color="orange", linewidth=0.8, alpha=0.7, label="Capteur brut")
    valid = np.isfinite(op_frames)
    ax.scatter(t_video_s[valid], op_frames[valid],
               s=3, color="royalblue", alpha=0.5, label="Capteur @ frame", zorder=3)
    ax.set_ylabel("Ouverture (mm)")
    ax.set_xlabel("Temps (s relatif à frame 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Axe 3 : drift horloge (résidus cumulés) ───────────────────────────────
    ax = axes[2]
    # Résidu = t_video_mesuré - t_video_ideal (basé sur FPS nominal)
    t_ideal = t_video_s[0] + np.arange(len(t_video_s)) * NOMINAL_PERIOD_S
    residual_ms = (t_video_s - t_ideal) * 1000
    ax.plot(t_video_s, residual_ms, linewidth=1.0, color="purple", label="Dérive cumulée vidéo (ms)")
    ax.axhline(0, color="black", linewidth=0.5)
    # Modèle linéaire
    slope_drift = (residual_ms[-1] - residual_ms[0]) / (t_video_s[-1] - t_video_s[0] + 1e-9)
    t_fit = np.array([t_video_s[0], t_video_s[-1]])
    y_fit = residual_ms[0] + slope_drift * (t_fit - t_video_s[0])
    ax.plot(t_fit, y_fit, color="red", linestyle="--", linewidth=1,
            label=f"Tendance {slope_drift:.3f}ms/s")
    ax.set_ylabel("Dérive cumulée (ms)")
    ax.set_xlabel("Temps vidéo (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Collecte des sessions
# ─────────────────────────────────────────────────────────────────────────────

def collect_session_paths(sessions_dir: str, pattern: str) -> List[str]:
    import glob as _glob
    root = sessions_dir
    paths = []
    for meta in sorted(_glob.glob(os.path.join(root, "**", "metadata.json"), recursive=True)):
        d = os.path.dirname(meta)
        if os.path.basename(d).startswith(pattern):
            paths.append(d)
    return sorted(paths)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Vérification alignement pince/vidéo par analyse des timestamps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sessions_dir", required=True,
                   help="Répertoire contenant les dossiers session_*")
    p.add_argument("--output_dir", default="pinces_results",
                   help="Répertoire de sortie")
    p.add_argument("--test", action="store_true",
                   help="Mode test : rapports détaillés + graphiques")
    p.add_argument("--session_pattern", default="session_",
                   help="Préfixe des dossiers session")

    # Seuils
    p.add_argument("--tolerance_offset_ms",  type=float, default=200.0,
                   help="Offset démarrage max vidéo/capteur (ms)")
    p.add_argument("--tolerance_latency_ms", type=float, default=20.0,
                   help="Latence max frame→capteur (ms), théorique ≈8ms à 60Hz")
    p.add_argument("--tolerance_jitter_ms",  type=float, default=15.0,
                   help="Jitter MAD timestamps vidéo (ms)")
    p.add_argument("--max_frame_drops",      type=int,   default=5)
    p.add_argument("--max_vel_mm_s",         type=float, default=2000.0)
    p.add_argument("--min_overlap_s",        type=float, default=3.0)

    return p


def main():
    parser = build_argparser()
    args   = parser.parse_args()

    thr = Thresholds(
        offset_ms      = args.tolerance_offset_ms,
        latency_max_ms = args.tolerance_latency_ms,
        jitter_std_ms  = args.tolerance_jitter_ms,
        max_drops      = args.max_frame_drops,
        max_vel_mm_s   = args.max_vel_mm_s,
        min_overlap_s  = args.min_overlap_s,
    )

    session_paths = collect_session_paths(args.sessions_dir, args.session_pattern)
    if not session_paths:
        print(f"Aucune session trouvée dans : {args.sessions_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Sessions trouvées : {len(session_paths)}")
    if args.test:
        os.makedirs(args.output_dir, exist_ok=True)

    all_results: List[SideResult] = []

    for i, spath in enumerate(session_paths):
        sname = os.path.basename(spath)
        print(f"  [{i+1:02d}/{len(session_paths)}] {sname}", end="")

        for side in ("left", "right"):
            r = process_side(spath, side, thr)
            all_results.append(r)

            if r.success:
                aln = r.alignment
                sym = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(r.status, "?")
                lat_str = f"{aln.latency_max_abs_ms:.1f}" if np.isfinite(aln.latency_max_abs_ms) else "N/A"
                print(
                    f"  {side}:{sym} "
                    f"off={aln.offset_start_ms:+.0f}ms "
                    f"lat_max={lat_str}ms",
                    end="",
                )
                if r.alerts:
                    codes = " ".join(a.code for a in r.alerts[:2])
                    print(f" [{codes}]", end="")
            else:
                print(f"  {side}:FAILED", end="")

            if args.test:
                prefix = os.path.join(args.output_dir, f"{sname}_{side}")
                write_side_report(r, prefix + "_report.txt")
                if r.success:
                    try:
                        save_plot(r, prefix + "_plot.png")
                    except Exception as exc:
                        print(f"\n    [WARN] Graphique {sname}/{side}: {exc}", end="")

        print()  # newline après la session

    # ── Résumé console ────────────────────────────────────────────────────────
    ok_list   = [r for r in all_results if r.success and r.status == "OK"]
    warn_list = [r for r in all_results if r.success and r.status == "WARNING"]
    err_list  = [r for r in all_results if r.success and r.status == "ERROR"]
    fail_list = [r for r in all_results if not r.success]

    print()
    print("=" * 64)
    print(f"Total    : {len(all_results):4d}")
    print(f"OK       : {len(ok_list):4d}")
    print(f"WARNING  : {len(warn_list):4d}")
    print(f"ERROR    : {len(err_list):4d}")
    print(f"FAILED   : {len(fail_list):4d}")

    success_list = [r for r in all_results if r.success and r.alignment]
    if success_list:
        offsets  = np.array([r.alignment.offset_start_ms    for r in success_list])
        lats     = np.array([r.alignment.latency_max_abs_ms for r in success_list
                             if np.isfinite(r.alignment.latency_max_abs_ms)])
        drops    = np.array([r.video.frame_drops             for r in success_list])
        neg_dts  = np.array([r.sensor.neg_dt_count           for r in success_list])

        print()
        print(f"Offset démarrage (ms)   : moy={offsets.mean():+.1f}  "
              f"med={np.median(offsets):+.1f}  "
              f"max|.|={np.abs(offsets).max():.1f}")
        if len(lats):
            print(f"Latence max frame→capteur: moy={lats.mean():.1f}ms  "
                  f"P95={np.percentile(lats,95):.1f}ms  "
                  f"max={lats.max():.1f}ms")
        print(f"Frame drops total       : {drops.sum():.0f}")
        print(f"Sensor neg_dt total     : {neg_dts.sum():.0f}")

    print("=" * 64)

    # ── Rapport global ────────────────────────────────────────────────────────
    if args.test:
        global_path = os.path.join(args.output_dir, "gripper_alignment_report.txt")
        write_global_report(all_results, global_path, thr)
        print(f"\nRapport global  : {global_path}")
        print(f"Rapports détail : {args.output_dir}/")

    # Code de retour : 2 si erreurs, 1 si seulement warnings
    if err_list or fail_list:
        sys.exit(2)
    if warn_list:
        sys.exit(1)


if __name__ == "__main__":
    main()
