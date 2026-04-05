#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
session_check.py — Autorité de vérification des sessions robot.

Fichier UNIQUE et autonome : tout le code de vérification est intégré ici.
Aucune dépendance sur les autres modules du dossier verification/.

Orchestre 5 dimensions d'analyse indépendantes et produit un rapport
d'autorité structuré avec score global, grade, diagnostics fins et
recommandations de réparation.

Dimensions (et poids) :
  1. gripper_timestamp_sync  (25%) — alignement horloge gripper↔vidéo (8 métriques)
  2. video_tracker_sync      (25%) — sync IA vidéo↔tracker (6 portes + score IA)
  3. tracker_placement       (15%) — head/left/right corrects (règle xyzw/0/-1)
  4. video_quality           (20%) — qualité timestamps JSONL (jitter, drops, gaps)
  5. gripper_frame_sync      (15%) — cohérence visuelle CV2 frame par frame

Score global = somme pondérée 0–100
Session "parfaite" si score ≥ PERFECT_THRESHOLD et aucune porte bloquante.

Grade:
  A  ≥ 90   Exemplaire
  B  ≥ 75   Bonne session
  C  ≥ 60   Acceptable
  D  ≥ 45   À corriger
  F  < 45   Non-récupérable / rejet

Usage CLI :
  python session_check.py <session>           # rapport texte
  python session_check.py <session> --json    # JSON brut
  python session_check.py <session> --full    # diagnostics complets
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Chemins ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Dépendances optionnelles ──────────────────────────────────────────────────
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import cv2 as _cv2
    _CV2 = True
except ImportError:
    _cv2 = None
    _CV2 = False

try:
    from scipy.interpolate import interp1d as _interp1d
    from scipy.signal import correlate as _correlate, savgol_filter as _savgol
    from scipy.stats import pearsonr as _pearsonr, linregress as _linregress
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    _TORCH = True
except ImportError:
    _TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
# Seuils et configuration globaux
# ══════════════════════════════════════════════════════════════════════════════

# Poids des 5 dimensions (doit sommer à 1.0)
W_GRIPPER_TS    = 0.25
W_VIDEO_SYNC    = 0.25
W_TRACKER_PLAC  = 0.15
W_VIDEO_QUAL    = 0.20
W_GRIP_FRAME    = 0.15

PERFECT_THRESHOLD = 75.0   # B ou mieux requis pour "parfaite"
GRADE_A = 90.0
GRADE_B = 75.0
GRADE_C = 60.0
GRADE_D = 45.0

# Seuils qualité vidéo (resserrés)
VQ_JITTER_WARN_MS   = 3.0   # était 5.0
VQ_JITTER_ERR_MS    = 10.0  # était 15.0
VQ_DROPS_WARN       = 2     # était 3
VQ_DROPS_ERR        = 5     # était 10
VQ_COVERAGE_WARN    = 0.95  # était 0.92
VQ_COVERAGE_ERR     = 0.90  # était 0.85

# Seuils gripper timestamp sync
GT_OFFSET_ERR_MS    = 200.0
GT_LATENCY_ERR_MS   = 25.0
GT_JITTER_ERR_MS    = 15.0
GT_OVERLAP_MIN_S    = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# Structures de résultats (session_check)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimensionResult:
    name:         str
    score:        float
    weight:       float
    grade:        str
    ok:           bool
    blocking:     bool
    confidence:   float
    summary:      str
    diagnostics:  List[str]
    repairs:      List[str]
    details:      Dict[str, Any]
    error:        Optional[str]


def _grade(score: float) -> str:
    if score >= GRADE_A: return "A"
    if score >= GRADE_B: return "B"
    if score >= GRADE_C: return "C"
    if score >= GRADE_D: return "D"
    return "F"


def _dim_ok(score: float) -> bool:
    return score >= GRADE_C  # D n'est plus acceptable


# ══════════════════════════════════════════════════════════════════════════════
# ════════════════════ MODULE 1 : GRIPPER TIMESTAMP SYNC ═════════════════════
# (code de session_pinces.py intégré directement)
# ══════════════════════════════════════════════════════════════════════════════

_NOMINAL_FPS      = 30.0
_NOMINAL_PERIOD_S = 1.0 / _NOMINAL_FPS
_DROP_THRESHOLD_S = 2.5 * _NOMINAL_PERIOD_S


@dataclass
class _Thresholds:
    offset_ms:      float = 200.0
    latency_max_ms: float = 25.0
    jitter_std_ms:  float = 15.0
    max_drops:      int   = 5
    max_vel_mm_s:   float = 2000.0
    min_overlap_s:  float = 3.0


@dataclass
class _VideoTimestampMetrics:
    n_frames:       int
    duration_s:     float
    dt_mean_ms:     float
    dt_std_ms:      float
    dt_min_ms:      float
    dt_max_ms:      float
    frame_drops:    int
    missing_indices: int
    jitter_std_ms:  float
    ts_ns:          np.ndarray = field(repr=False)
    indices:        np.ndarray = field(repr=False)


@dataclass
class _SensorMetrics:
    n_samples:      int
    duration_s:     float
    dt_mean_ms:     float
    dt_std_ms:      float
    neg_dt_count:   int
    neg_dt_details: List[Dict]
    vel_max_mm_s:   float
    vel_p99_mm_s:   float
    vel_anomalies:  int
    opening_range:  Tuple[float, float]


@dataclass
class _AlignmentMetrics:
    dur_vid_s:          float
    dur_sensor_s:       float
    overlap_s:          float
    offset_start_ms:    float
    latency_mean_ms:    float
    latency_std_ms:     float
    latency_max_abs_ms: float
    latency_p95_abs_ms: float
    frames_no_sensor:   int
    sensor_gap_max_ms:  float
    sensor_gap_count:   int
    linfit_slope:       float
    linfit_r2:          float
    linfit_residual_std_ms: float
    linfit_residual_max_ms: float
    opening_at_frames:  np.ndarray = field(repr=False)
    frame_ts_ns:        np.ndarray = field(repr=False)


@dataclass
class _PhysicalCoherenceMetrics:
    n_frames_with_sensor: int
    n_frames_no_sensor:   int
    opening_mean_mm:      float
    opening_std_mm:       float
    opening_range:        Tuple[float, float]
    d_opening_max_mm_s:   float
    d_opening_p99_mm_s:   float
    impossible_jumps:     int


@dataclass
class _Alert:
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
    error:        str                              = ""
    video:        Optional[_VideoTimestampMetrics] = None
    sensor:       Optional[_SensorMetrics]         = None
    alignment:    Optional[_AlignmentMetrics]      = None
    physical:     Optional[_PhysicalCoherenceMetrics] = None
    alerts:       List[_Alert]                     = field(default_factory=list)
    has_errors:   bool                             = False
    sensor_df:    Any                              = field(default=None, repr=False)

    @property
    def status(self) -> str:
        if not self.success: return "FAILED"
        if self.has_errors:  return "ERROR"
        if self.alerts:      return "WARNING"
        return "OK"

    @property
    def n_errors(self) -> int:
        return sum(1 for a in self.alerts if a.level == "ERROR")

    @property
    def n_warnings(self) -> int:
        return sum(1 for a in self.alerts if a.level == "WARNING")


def _sp_load_jsonl_timestamps(path: str) -> Tuple[np.ndarray, np.ndarray]:
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


def _sp_load_sensor(path: str):
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


def _sp_fill_sensor_gaps(ts_ns, opening_mm, dt_nominal_ms=17.0,
                          max_opening_change_mm=2.0, max_gap_ms=1500.0):
    if len(ts_ns) < 2:
        return ts_ns.copy(), opening_mm.copy()
    gap_thresh_ns = 4.0 * dt_nominal_ms * 1e6
    max_gap_ns    = max_gap_ms * 1e6
    filled_ts = list(ts_ns)
    filled_op = list(opening_mm.astype(float))
    diffs_ns = np.diff(ts_ns)
    for i, gap_ns in enumerate(diffs_ns):
        if gap_ns <= gap_thresh_ns:
            continue
        gap_ms        = gap_ns / 1e6
        op_before     = float(opening_mm[i])
        op_after      = float(opening_mm[i + 1])
        delta_opening = abs(op_after - op_before)
        if gap_ms > max_gap_ms or delta_opening > max_opening_change_mm:
            continue
        n = max(1, int(round(gap_ms / dt_nominal_ms)) - 1)
        synth_ts = np.linspace(
            ts_ns[i] + dt_nominal_ms * 1e6,
            ts_ns[i + 1] - dt_nominal_ms * 1e6,
            n, dtype=np.int64,
        )
        synth_op = np.linspace(op_before, op_after, n)
        filled_ts.extend(synth_ts.tolist())
        filled_op.extend(synth_op.tolist())
    order = np.argsort(filled_ts)
    return np.array(filled_ts, dtype=np.int64)[order], np.array(filled_op, dtype=np.float64)[order]


def _sp_extend_sensor_backward(ts_ns, opening_mm, vid_start_ns,
                                dt_nominal_ms=17.0, max_extrap_ms=200.0):
    sensor_start_ns = ts_ns[0]
    gap_ms = (sensor_start_ns - vid_start_ns) / 1e6
    if gap_ms <= 0 or gap_ms > max_extrap_ms:
        return ts_ns, opening_mm
    dt_ns    = int(dt_nominal_ms * 1e6)
    synth_ts = np.arange(vid_start_ns, sensor_start_ns, dt_ns, dtype=np.int64)
    if len(synth_ts) == 0:
        return ts_ns, opening_mm
    synth_op = np.full(len(synth_ts), float(opening_mm[0]))
    return np.concatenate([synth_ts, ts_ns]), np.concatenate([synth_op, opening_mm.astype(float)])


def _sp_apply_sensor_fixes(sensor_df, vid_start_ns, dt_nominal_ms=17.0):
    ts = sensor_df["timestamp_ns"].values.astype(np.int64)
    op = sensor_df["opening_mm"].values.astype(np.float64)
    ts, op = _sp_fill_sensor_gaps(ts, op, dt_nominal_ms=dt_nominal_ms)
    ts, op = _sp_extend_sensor_backward(ts, op, vid_start_ns, dt_nominal_ms=dt_nominal_ms)
    dt = np.diff(ts, prepend=ts[0]).astype(float) / 1e6
    return pd.DataFrame({"timestamp_ns": ts, "opening_mm": op, "dt_ms": dt})


def _sp_analyze_video_timestamps(indices, ts_ns) -> _VideoTimestampMetrics:
    dt_ms = np.diff(ts_ns) / 1e6
    drops   = int((dt_ms > _DROP_THRESHOLD_S * 1000).sum())
    expected = np.arange(indices[0], indices[-1] + 1)
    missing  = int(len(expected) - len(indices))
    residuals = dt_ms - np.median(dt_ms)
    mad       = np.median(np.abs(residuals))
    jitter    = float(mad * 1.4826)
    return _VideoTimestampMetrics(
        n_frames        = len(ts_ns),
        duration_s      = float((ts_ns[-1] - ts_ns[0]) / 1e9),
        dt_mean_ms      = float(dt_ms.mean()),
        dt_std_ms       = float(dt_ms.std()),
        dt_min_ms       = float(dt_ms.min()),
        dt_max_ms       = float(dt_ms.max()),
        frame_drops     = drops,
        missing_indices = missing,
        jitter_std_ms   = jitter,
        ts_ns           = ts_ns,
        indices         = indices,
    )


def _sp_analyze_sensor(df, max_vel) -> _SensorMetrics:
    ts  = df["timestamp_ns"].values
    op  = df["opening_mm"].values
    dt  = df["dt_ms"].values[1:]
    neg_idx = np.where(dt < 0)[0]
    neg_details = [{"idx": int(i), "dt_ms": float(dt[i]),
                    "opening_before": float(op[i]), "opening_after": float(op[i+1])}
                   for i in neg_idx]
    safe_dt_s = np.maximum(dt, 1.0) / 1000.0
    vel       = np.abs(np.diff(op)) / safe_dt_s
    return _SensorMetrics(
        n_samples      = len(df),
        duration_s     = float((ts[-1] - ts[0]) / 1e9),
        dt_mean_ms     = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.0,
        dt_std_ms      = float(dt[dt > 0].std()) if (dt > 0).sum() > 1 else 0.0,
        neg_dt_count   = len(neg_idx),
        neg_dt_details = neg_details,
        vel_max_mm_s   = float(vel.max()) if len(vel) else 0.0,
        vel_p99_mm_s   = float(np.percentile(vel, 99)) if len(vel) else 0.0,
        vel_anomalies  = int((vel > max_vel).sum()),
        opening_range  = (float(op.min()), float(op.max())),
    )


def _sp_compute_alignment(vid: _VideoTimestampMetrics, sensor_df) -> _AlignmentMetrics:
    tv      = vid.ts_ns.astype(np.float64)
    ts      = sensor_df["timestamp_ns"].values.astype(np.float64)
    opening = sensor_df["opening_mm"].values

    dur_vid_s    = float((tv[-1] - tv[0]) / 1e9)
    dur_sensor_s = float((ts[-1] - ts[0]) / 1e9)
    t_ov0 = max(tv[0], ts[0])
    t_ov1 = min(tv[-1], ts[-1])
    overlap_s = float((t_ov1 - t_ov0) / 1e9) if t_ov1 > t_ov0 else 0.0
    offset_start_ms = float((tv[0] - ts[0]) / 1e6)

    nearest_hi = np.searchsorted(ts, tv)
    nearest_hi = np.clip(nearest_hi, 0, len(ts) - 1)
    nearest_lo = np.clip(nearest_hi - 1, 0, len(ts) - 1)
    d_hi = np.abs(tv - ts[nearest_hi])
    d_lo = np.abs(tv - ts[nearest_lo])
    nearest_ts = np.where(d_hi <= d_lo, ts[nearest_hi], ts[nearest_lo])
    latency_ms = (tv - nearest_ts) / 1e6
    in_range   = (tv >= ts[0]) & (tv <= ts[-1])
    n_no_sensor = int((~in_range).sum())
    lat_valid   = latency_ms[in_range]
    if len(lat_valid) > 0:
        lat_mean = float(lat_valid.mean())
        lat_std  = float(lat_valid.std())
        lat_max  = float(np.abs(lat_valid).max())
        lat_p95  = float(np.percentile(np.abs(lat_valid), 95))
    else:
        lat_mean = lat_std = lat_max = lat_p95 = np.nan

    mask_sen_in_vid = (ts >= tv[0]) & (ts <= tv[-1])
    ts_in_vid       = ts[mask_sen_in_vid]
    dt_nominal_ms   = float(np.median(np.diff(ts)) / 1e6)
    if len(ts_in_vid) > 1:
        gaps_ms        = np.diff(ts_in_vid) / 1e6
        gap_thresh     = 4.0 * dt_nominal_ms
        sensor_gap_max = float(gaps_ms.max())
        sensor_gap_cnt = int((gaps_ms > gap_thresh).sum())
    else:
        sensor_gap_max = float('inf')
        sensor_gap_cnt = 0

    if len(tv) > 10:
        idx = np.arange(len(tv), dtype=np.float64)
        slope_ns, intercept_ns, r, _, _ = _linregress(idx, tv)
        r2      = r ** 2
        fitted  = slope_ns * idx + intercept_ns
        res_ms  = (tv - fitted) / 1e6
        lf_std  = float(res_ms.std())
        lf_max  = float(np.abs(res_ms).max())
    else:
        slope_ns = (tv[-1] - tv[0]) / max(len(tv) - 1, 1)
        r2 = lf_std = lf_max = np.nan

    f_opening = _interp1d(ts, opening, kind="linear", bounds_error=False, fill_value=np.nan)
    opening_at_frames = f_opening(tv)

    return _AlignmentMetrics(
        dur_vid_s               = dur_vid_s,
        dur_sensor_s            = dur_sensor_s,
        overlap_s               = overlap_s,
        offset_start_ms         = offset_start_ms,
        latency_mean_ms         = lat_mean,
        latency_std_ms          = lat_std,
        latency_max_abs_ms      = lat_max,
        latency_p95_abs_ms      = lat_p95,
        frames_no_sensor        = n_no_sensor,
        sensor_gap_max_ms       = sensor_gap_max,
        sensor_gap_count        = sensor_gap_cnt,
        linfit_slope            = float(slope_ns),
        linfit_r2               = float(r2),
        linfit_residual_std_ms  = lf_std,
        linfit_residual_max_ms  = lf_max,
        opening_at_frames       = opening_at_frames,
        frame_ts_ns             = vid.ts_ns,
    )


def _sp_compute_physical(alignment: _AlignmentMetrics, max_vel) -> _PhysicalCoherenceMetrics:
    op    = alignment.opening_at_frames
    ts_ns = alignment.frame_ts_ns.astype(np.float64)
    valid = np.isfinite(op)
    n_ok  = int(valid.sum())
    n_miss = int((~valid).sum())
    if n_ok < 2:
        return _PhysicalCoherenceMetrics(n_ok, n_miss, np.nan, np.nan,
                                          (np.nan, np.nan), np.nan, np.nan, 0)
    op_v   = op[valid]
    ts_v   = ts_ns[valid]
    dt_s   = np.diff(ts_v) / 1e9
    d_op   = np.abs(np.diff(op_v))
    vel    = d_op / np.maximum(dt_s, 1e-6)
    return _PhysicalCoherenceMetrics(
        n_frames_with_sensor = n_ok,
        n_frames_no_sensor   = n_miss,
        opening_mean_mm      = float(op_v.mean()),
        opening_std_mm       = float(op_v.std()),
        opening_range        = (float(op_v.min()), float(op_v.max())),
        d_opening_max_mm_s   = float(vel.max()),
        d_opening_p99_mm_s   = float(np.percentile(vel, 99)),
        impossible_jumps     = int((vel > max_vel).sum()),
    )


def _sp_generate_alerts(vid, sen, aln, phy, thr) -> List[_Alert]:
    alerts: List[_Alert] = []

    def add(code, level, msg, value, threshold):
        alerts.append(_Alert(code=code, level=level, message=msg,
                             value=value, threshold=threshold))

    ONE_FRAME_MS = 1000.0 / _NOMINAL_FPS
    abs_offset = abs(aln.offset_start_ms)
    if aln.offset_start_ms < -ONE_FRAME_MS:
        add("OFFSET_START", "ERROR",
            f"Vidéo démarre {aln.offset_start_ms:+.1f}ms avant le capteur — capteur manquant au début",
            abs_offset, ONE_FRAME_MS)
    elif abs_offset > thr.offset_ms:
        add("OFFSET_START", "ERROR",
            f"Offset démarrage {aln.offset_start_ms:+.1f}ms > seuil {thr.offset_ms:.0f}ms  "
            f"[corrigeable par fix_camera_offset]",
            abs_offset, thr.offset_ms)

    if np.isfinite(aln.latency_max_abs_ms):
        if aln.latency_max_abs_ms > thr.latency_max_ms:
            has_gap = aln.sensor_gap_count > 0
            if has_gap:
                level = "WARNING"
                note  = f"  (causée par gap capteur {aln.sensor_gap_max_ms:.0f}ms)"
            else:
                level = "ERROR" if aln.latency_max_abs_ms > thr.latency_max_ms * 2 else "WARNING"
                note  = ""
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
            f"Recouvrement vidéo/capteur {aln.overlap_s:.1f}s < minimum {thr.min_overlap_s:.0f}s",
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
            f"{phy.impossible_jumps} saut(s) impossible(s) dans le capteur interpolé",
            phy.impossible_jumps, 0.0)

    return alerts


def _sp_process_side(session_path: str, side: str, thr: _Thresholds) -> _SideResult:
    if not _PANDAS:
        return _SideResult(session_name=Path(session_path).name, side=side,
                           success=False, error="pandas non disponible")
    if not _SCIPY:
        return _SideResult(session_name=Path(session_path).name, side=side,
                           success=False, error="scipy non disponible")

    session_name = os.path.basename(session_path)
    vdir         = os.path.join(session_path, "videos")
    jsonl_path   = os.path.join(vdir, f"{side}.jsonl")
    sensor_path  = os.path.join(session_path, f"gripper_{side}_data.csv")

    missing = [p for p in [jsonl_path, sensor_path] if not os.path.exists(p)]
    if missing:
        names = [os.path.basename(p) for p in missing]
        return _SideResult(session_name=session_name, side=side, success=False,
                           error=f"Session incomplète — fichiers absents : {names}")

    try:
        indices, ts_ns = _sp_load_jsonl_timestamps(jsonl_path)
        sensor_df      = _sp_load_sensor(sensor_path)
        dt_nominal_ms  = float(np.median(np.diff(sensor_df["timestamp_ns"].values)) / 1e6)
        sensor_df      = _sp_apply_sensor_fixes(sensor_df, int(ts_ns[0]), dt_nominal_ms)
        vid = _sp_analyze_video_timestamps(indices, ts_ns)
        sen = _sp_analyze_sensor(sensor_df, thr.max_vel_mm_s)
        aln = _sp_compute_alignment(vid, sensor_df)
        phy = _sp_compute_physical(aln, thr.max_vel_mm_s)
        als = _sp_generate_alerts(vid, sen, aln, phy, thr)
        return _SideResult(
            session_name = session_name, side = side, success = True,
            video = vid, sensor = sen, alignment = aln, physical = phy,
            alerts = als, has_errors = any(a.level == "ERROR" for a in als),
            sensor_df = sensor_df,
        )
    except Exception as exc:
        return _SideResult(session_name=session_name, side=side, success=False,
                           error=f"{type(exc).__name__}: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# ════════════════════ MODULE 2 : VIDEO↔TRACKER SYNC (check.py) ══════════════
# Code intégré directement
# ══════════════════════════════════════════════════════════════════════════════

# ── Constantes check ──────────────────────────────────────────────────────────
_RESAMPLE_MS        = 10.0
_MAX_LAG_MS         = 500.0
_WINDOW_MS          = 2000.0
_WINDOW_STRIDE_MS   = 500.0
_MIN_OVERLAP_MS     = 800.0
_EDGE_MARGIN_MS     = 30.0
_QUAT_CORRUPT_FRAC  = 0.05
_TRACKER_GAP_WARN_MS  = 500.0
_TRACKER_GAP_FAIL_MS  = 2000.0
_TRACKER_GAP_FAIL_N   = 3
_CAM_CONTINUITY_MAX_GAP_MS  = 500.0
_CAM_CONTINUITY_MIN_COV     = 0.80
# Décalage max toléré entre le démarrage du tracker et celui des caméras/gripper
_STREAM_ALIGN_WARN_MS       = 200.0   # warning
_STREAM_ALIGN_FAIL_MS       = 1000.0  # gate échoue (session non récupérable sans trim)

_MODEL_DIR  = _ROOT / "_check_model"
_MODEL_PATH = _MODEL_DIR / "check_model.pt"
_PAIRS = [("tracker", "left"), ("tracker", "right"), ("tracker", "head"),
          ("left", "right"), ("left", "head"), ("right", "head")]
_MAJOR_PAIRS = [("tracker", "left"), ("tracker", "right")]


@dataclass
class _GateResult:
    name:    str
    passed:  bool
    message: str  = ""
    value:   Any  = None


@dataclass
class _SessionReport:
    session_path:    str
    session_id:      str               = ""
    gates:           List[_GateResult] = field(default_factory=list)
    ia_scores:       Dict[str, float]  = field(default_factory=dict)
    ia_score:        float             = 0.0
    score:           float             = 0.0
    blocking_reason: str               = ""
    errors:          List[str]         = field(default_factory=list)

    def add_gate(self, g: _GateResult):
        self.gates.append(g)

    def is_blocked(self) -> bool:
        return any(not g.passed for g in self.gates)

    def first_failure(self) -> Optional[_GateResult]:
        return next((g for g in self.gates if not g.passed), None)


def _chk_read_jsonl(path: Path):
    """Lit un .jsonl et retourne list de dict."""
    if not _PANDAS:
        return []
    rows = []
    raw = path.read_bytes()
    for line in raw.split(b"\r\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _chk_read_tracker(path: Path):
    """Lit tracker_positions.csv, retourne DataFrame."""
    if not _PANDAS:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _chk_gate_structure(session_dir: Path, report: _SessionReport) -> bool:
    needed = [
        session_dir / "metadata.json",
        session_dir / "tracker_positions.csv",
        session_dir / "videos" / "head.jsonl",
        session_dir / "videos" / "left.jsonl",
        session_dir / "videos" / "right.jsonl",
    ]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        report.add_gate(_GateResult("structure", False,
                                    f"Fichiers manquants: {missing}"))
        return False
    report.add_gate(_GateResult("structure", True))
    return True


def _chk_gate_metadata(session_dir: Path, report: _SessionReport):
    try:
        meta = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        report.session_id = meta.get("session_id", "")
        report.add_gate(_GateResult("metadata", True))
        return meta
    except Exception as e:
        report.add_gate(_GateResult("metadata", False, str(e)))
        return None


def _chk_gate_quaternions(session_dir: Path, report: _SessionReport) -> bool:
    if not _PANDAS:
        report.add_gate(_GateResult("quaternions", True, "pandas absent — ignoré"))
        return True
    try:
        df = _chk_read_tracker(session_dir / "tracker_positions.csv")
        if df is None or df.empty:
            report.add_gate(_GateResult("quaternions", False, "tracker CSV illisible"))
            return False
        # Ne sélectionner que les colonnes quaternion strictement (qw/qx/qy/qz)
        # Exclure les colonnes de position (_x/_y/_z) dont la norme n'est pas ~1
        quat_cols = [c for c in df.columns
                     if any(c.lower().endswith(s) for s in ["qw", "qx", "qy", "qz"])
                     or any(f"_{s}" in c.lower() for s in ["qw", "qx", "qy", "qz"])]
        if not quat_cols:
            report.add_gate(_GateResult("quaternions", True, "colonnes quat non trouvées — ignoré"))
            return True
        # Vérifier par groupe de 4 (un quaternion = 4 composantes)
        n_groups = len(quat_cols) // 4
        if n_groups == 0:
            report.add_gate(_GateResult("quaternions", True, "moins de 4 colonnes quat — ignoré"))
            return True
        total_bad = 0
        total_rows = len(df)
        for i in range(n_groups):
            group = quat_cols[i*4:(i+1)*4]
            vals  = df[group].values.astype(float)
            norms = np.linalg.norm(vals, axis=1)
            total_bad += int(np.sum((norms < 0.5) | (norms > 1.5)))
        bad = total_bad / max(total_rows * n_groups, 1)
        if bad > _QUAT_CORRUPT_FRAC:
            report.add_gate(_GateResult("quaternions", False,
                                        f"{bad*100:.1f}% quaternions invalides"))
            return False
        report.add_gate(_GateResult("quaternions", True, f"{bad*100:.1f}% invalides"))
        return True
    except Exception as e:
        report.add_gate(_GateResult("quaternions", True, f"erreur ignorée: {e}"))
        return True


def _chk_gate_tracker_continuity(session_dir: Path, report: _SessionReport) -> bool:
    if not _PANDAS:
        report.add_gate(_GateResult("tracker_continuity", True, "pandas absent"))
        return True
    try:
        df = _chk_read_tracker(session_dir / "tracker_positions.csv")
        if df is None or df.empty:
            report.add_gate(_GateResult("tracker_continuity", False, "CSV vide"))
            return False
        ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
        if ts_col is None:
            report.add_gate(_GateResult("tracker_continuity", True, "colonne ts introuvable"))
            return True
        ts = df[ts_col].values.astype(float)
        dt = np.diff(ts)
        dt_ms = dt / 1e6 if ts.max() > 1e12 else dt * 1000.0
        gaps_fail = int((dt_ms > _TRACKER_GAP_FAIL_MS).sum())
        gaps_warn = int((dt_ms > _TRACKER_GAP_WARN_MS).sum())
        if gaps_fail >= _TRACKER_GAP_FAIL_N:
            report.add_gate(_GateResult("tracker_continuity", False,
                                        f"{gaps_fail} gap(s) > {_TRACKER_GAP_FAIL_MS:.0f}ms"))
            return False
        report.add_gate(_GateResult("tracker_continuity", True,
                                    f"gaps warn={gaps_warn} fail={gaps_fail}"))
        return True
    except Exception as e:
        report.add_gate(_GateResult("tracker_continuity", True, f"erreur ignorée: {e}"))
        return True


def _chk_gate_camera_continuity(session_dir: Path, report: _SessionReport) -> bool:
    worst_frac = 0.0
    for cam in ("head", "left", "right"):
        rows = _chk_read_jsonl(session_dir / "videos" / f"{cam}.jsonl")
        if not rows:
            continue
        ts = np.array([r.get("capture_time", 0) for r in rows], dtype=np.float64)
        dt = np.diff(ts)
        nominal = float(np.median(dt))
        if nominal <= 0:
            continue
        large_gaps = (dt > _CAM_CONTINUITY_MAX_GAP_MS).sum()
        frac = large_gaps / max(len(dt), 1)
        worst_frac = max(worst_frac, frac)
    if worst_frac > (1.0 - _CAM_CONTINUITY_MIN_COV):
        report.add_gate(_GateResult("camera_continuity", False,
                                    f"{worst_frac*100:.1f}% frames avec grand gap"))
        return False
    report.add_gate(_GateResult("camera_continuity", True))
    return True


def _chk_gate_camera_coverage(session_dir: Path, report: _SessionReport) -> bool:
    if not _PANDAS:
        report.add_gate(_GateResult("camera_coverage", True, "pandas absent"))
        return True
    try:
        trk = _chk_read_tracker(session_dir / "tracker_positions.csv")
        if trk is None or trk.empty:
            report.add_gate(_GateResult("camera_coverage", True, "tracker absent"))
            return True
        ts_col = next((c for c in trk.columns if "timestamp" in c.lower()), None)
        if ts_col is None:
            report.add_gate(_GateResult("camera_coverage", True))
            return True
        ts = trk[ts_col].values.astype(float)
        trk_t0 = float(ts.min())
        trk_t1 = float(ts.max())
        trk_dur = (trk_t1 - trk_t0) / (1e6 if ts.max() > 1e12 else 1.0)

        worst_cov = 1.0
        for cam in ("head", "left", "right"):
            rows = _chk_read_jsonl(session_dir / "videos" / f"{cam}.jsonl")
            if not rows:
                continue
            cam_ts = np.array([r.get("capture_time", 0) for r in rows], dtype=np.float64)
            cam_t0 = float(cam_ts.min())
            cam_t1 = float(cam_ts.max())
            cam_dur = cam_t1 - cam_t0
            if trk_dur <= 0:
                continue
            cov = cam_dur / trk_dur
            worst_cov = min(worst_cov, cov)

        if worst_cov < _CAM_CONTINUITY_MIN_COV:
            report.add_gate(_GateResult("camera_coverage", False,
                                        f"Coverage caméra {worst_cov*100:.0f}% < {_CAM_CONTINUITY_MIN_COV*100:.0f}%"))
            return False
        report.add_gate(_GateResult("camera_coverage", True,
                                    f"coverage min={worst_cov*100:.0f}%"))
        return True
    except Exception as e:
        report.add_gate(_GateResult("camera_coverage", True, f"erreur ignorée: {e}"))
        return True


def _chk_gate_stream_alignment(session_dir: Path, report: _SessionReport) -> bool:
    """
    Vérifie que tous les flux (tracker, caméras, gripper) démarrent au même moment.
    Un décalage > _STREAM_ALIGN_FAIL_MS signifie que des données sont perdues
    au début et que la session nécessite un trim avant tout traitement.
    """
    try:
        trk_path = session_dir / "tracker_positions.csv"
        if not trk_path.exists():
            report.add_gate(_GateResult("stream_alignment", True, "tracker absent — ignoré"))
            return True

        # Tracker t0 — via pandas helper
        trk_df = _chk_read_tracker(trk_path)
        if trk_df is None or trk_df.empty:
            report.add_gate(_GateResult("stream_alignment", True, "tracker vide — ignoré"))
            return True
        ts_col = next((c for c in trk_df.columns if "timestamp_ns" in c.lower()), None)
        if ts_col is None:
            report.add_gate(_GateResult("stream_alignment", True, "timestamp_ns absent — ignoré"))
            return True
        ts_vals_trk = trk_df[ts_col].dropna().astype(float)
        if ts_vals_trk.empty:
            report.add_gate(_GateResult("stream_alignment", True, "timestamp_ns vide — ignoré"))
            return True
        t_trk = float(ts_vals_trk.iloc[0]) / 1_000_000

        # Caméras t0
        t_cams = {}
        for cam in ("head", "left", "right"):
            rows = _chk_read_jsonl(session_dir / "videos" / f"{cam}.jsonl")
            if rows:
                ts_c = [r.get("capture_time", 0) for r in rows if r.get("capture_time")]
                if ts_c:
                    t_cams[cam] = float(min(ts_c))

        # Gripper t0 — via pandas helper
        t_grps = {}
        for side in ("left", "right"):
            gp = session_dir / f"gripper_{side}_data.csv"
            if gp.exists():
                gdf = _chk_read_tracker(gp)  # même format timestamp_ns
                if gdf is not None and not gdf.empty:
                    ts_g = next((c for c in gdf.columns if "timestamp_ns" in c.lower()), None)
                    if ts_g:
                        vals_g = gdf[ts_g].dropna().astype(float)
                        if not vals_g.empty:
                            t_grps[side] = float(vals_g.iloc[0]) / 1_000_000

        if not t_cams and not t_grps:
            report.add_gate(_GateResult("stream_alignment", True, "aucun flux caméra/gripper — ignoré"))
            return True

        all_t0s = list(t_cams.values()) + list(t_grps.values())
        latest_other = max(all_t0s)    # le flux non-tracker qui démarre le plus tard
        earliest_other = min(all_t0s)  # le flux non-tracker qui démarre le plus tôt

        # Décalage tracker vs autres flux
        tracker_lead_ms = latest_other - t_trk   # >0 = tracker démarre avant les autres
        cam_spread_ms   = latest_other - earliest_other  # écart entre caméras/grippers

        offsets_str = ", ".join(
            f"{k}:{v - t_trk:+.0f}ms"
            for k, v in {**t_cams, **t_grps}.items()
        )

        if tracker_lead_ms > _STREAM_ALIGN_FAIL_MS:
            report.add_gate(_GateResult(
                "stream_alignment", False,
                f"tracker démarre {tracker_lead_ms:.0f}ms avant les caméras — trim requis "
                f"({offsets_str})"
            ))
            return False
        elif tracker_lead_ms > _STREAM_ALIGN_WARN_MS:
            report.add_gate(_GateResult(
                "stream_alignment", True,
                f"⚠ tracker en avance de {tracker_lead_ms:.0f}ms — trim recommandé "
                f"({offsets_str})"
            ))
            return True
        else:
            report.add_gate(_GateResult(
                "stream_alignment", True,
                f"flux alignés (écart max={tracker_lead_ms:.0f}ms)"
            ))
            return True
    except Exception as e:
        report.add_gate(_GateResult("stream_alignment", True, f"erreur ignorée: {e}"))
        return True


def _chk_math_score(session_dir: Path) -> Tuple[float, dict]:
    """
    Score mathématique pur de synchronisation vidéo↔tracker — sans IA.

    Principe : le flux optique de la caméra HEAD mesure le mouvement apparent
    de la scène. La vitesse angulaire du tracker HEAD mesure le mouvement réel
    de la tête. Si les deux flux sont synchronisés, leur corrélation croisée
    doit présenter un pic net à lag ≈ 0.

    Pipeline :
      1. Flux optique dense (Farneback) frame→frame sur la caméra HEAD.
      2. Vitesse angulaire du tracker HEAD (norme de dquat/dt).
      3. Cross-corrélation ±500ms pour trouver le lag optimal.
      4. r au lag optimal = score de sync brut.
      5. Pénalités : lag > 1 frame, overlap insuffisant.

    Retourne (score_0_1, details_dict).
    """
    if not _CV2 or not _PANDAS or not _SCIPY:
        missing = []
        if not _CV2:    missing.append("opencv-python")
        if not _PANDAS: missing.append("pandas")
        if not _SCIPY:  missing.append("scipy")
        return 0.70, {"method": "unavailable", "reason": f"dépendances manquantes: {missing}"}

    LAG_SEARCH_MS = 500.0   # plage de recherche du lag ±ms
    FRAME_MS      = 33.3    # 1 frame @ 30fps (lag naturel flux optique)
    MIN_OVERLAP_S = 5.0     # recouvrement minimum pour un score fiable
    MIN_FRAMES    = 50      # frames minimum pour calculer le flux

    details: Dict[str, Any] = {"method": "optical_flow_xcorr", "cameras": {}}

    try:
        # ── Tracker HEAD : vitesse angulaire ──────────────────────────────────
        trk = _chk_read_tracker(session_dir / "tracker_positions.csv")
        if trk is None or trk.empty:
            return 0.70, {**details, "reason": "tracker vide"}

        ts_col = next((c for c in trk.columns if c.lower() == "timestamp_ns"), None)
        if ts_col is None:
            ts_col = next((c for c in trk.columns if "timestamp_ns" in c.lower()), None)
        if ts_col is None:
            return 0.70, {**details, "reason": "timestamp_ns introuvable"}

        ts_ns = trk[ts_col].values.astype(np.float64)
        ts_ms = ts_ns / 1e6 if ts_ns.max() > 1e12 else ts_ns * 1000.0

        # Chercher les colonnes quaternion du tracker head
        qcols = [c for c in trk.columns
                 if "head" in c.lower() and any(q in c.lower() for q in ["qw","qx","qy","qz"])]
        if len(qcols) < 4:
            # Fallback : vitesse de position head
            pcols = [c for c in trk.columns if "head" in c.lower()
                     and any(a in c.lower() for a in ["_x","_y","_z"])]
            if len(pcols) < 3:
                return 0.70, {**details, "reason": "colonnes head tracker introuvables"}
            pos = trk[pcols[:3]].values.astype(np.float64)
            dpos = np.diff(pos, axis=0)
            dt_s = np.diff(ts_ms) / 1000.0
            trk_motion = np.linalg.norm(dpos, axis=1) / np.maximum(dt_s, 1e-6)
        else:
            # Vitesse angulaire = norme(dquat/dt)
            qcols_sorted = sorted(qcols, key=lambda c: next(
                i for i, q in enumerate(["qw","qx","qy","qz"]) if q in c.lower()
            ))
            quat = trk[qcols_sorted[:4]].values.astype(np.float64)
            dq   = np.diff(quat, axis=0)
            dt_s = np.diff(ts_ms) / 1000.0
            trk_motion = np.linalg.norm(dq, axis=1) / np.maximum(dt_s, 1e-6)

        trk_ts = ts_ms[1:]  # timestamps des deltas

        # ── Caméra HEAD : flux optique ────────────────────────────────────────
        # On préfère HEAD car elle est solidaire du casque (= même mouvement que le tracker HEAD)
        cam_scores = {}

        for cam in ("head", "left", "right"):
            video_path = session_dir / "videos" / f"{cam}.mp4"
            jsonl_path = session_dir / "videos" / f"{cam}.jsonl"
            if not video_path.exists() or not jsonl_path.exists():
                continue

            rows = _chk_read_jsonl(jsonl_path)
            if not rows or len(rows) < MIN_FRAMES:
                continue
            cam_ts = np.array([r.get("capture_time", 0) for r in rows
                               if r.get("capture_time", 0) > 0], dtype=np.float64)
            if len(cam_ts) < MIN_FRAMES:
                continue

            # Vérifier le chevauchement avant de lire la vidéo
            t_ov0 = max(ts_ms[0], cam_ts[0])
            t_ov1 = min(ts_ms[-1], cam_ts[-1])
            overlap_s = max(0.0, (t_ov1 - t_ov0) / 1000.0)
            if overlap_s < MIN_OVERLAP_S:
                details["cameras"][cam] = {"error": f"overlap insuffisant ({overlap_s:.1f}s)", "score": 0.0}
                continue

            # Flux optique dense sur la vidéo
            cap = _cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                details["cameras"][cam] = {"error": "vidéo inaccessible", "score": 0.0}
                continue

            prev_gray = None
            of_norms: List[float] = []
            fi = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # Réduire la résolution pour accélérer (factor 4)
                small = _cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
                gray  = _cv2.cvtColor(small, _cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = _cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=2, winsize=15,
                        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
                    )
                    of_norms.append(float(np.linalg.norm(flow, axis=2).mean()))
                prev_gray = gray
                fi += 1
            cap.release()

            if len(of_norms) < MIN_FRAMES:
                details["cameras"][cam] = {"error": f"trop peu de frames ({len(of_norms)})", "score": 0.0}
                continue

            of_norms_arr = np.array(of_norms, dtype=np.float64)
            of_ts        = cam_ts[1:len(of_norms_arr) + 1]  # timestamps des deltas

            # Grille d'interpolation commune
            t0_grid = max(of_ts[0], trk_ts[0])
            t1_grid = min(of_ts[-1], trk_ts[-1])
            if t1_grid - t0_grid < MIN_OVERLAP_S * 1000.0:
                details["cameras"][cam] = {"error": "overlap effectif insuffisant", "score": 0.0}
                continue

            grid    = np.arange(t0_grid, t1_grid, FRAME_MS)
            of_g    = np.interp(grid, of_ts,  of_norms_arr)
            trk_g   = np.interp(grid, trk_ts, trk_motion)

            def znorm(x):
                return (x - x.mean()) / (x.std() + 1e-8)

            of_z  = znorm(of_g)
            trk_z = znorm(trk_g)

            # Cross-corrélation sur ±LAG_SEARCH_MS
            xcorr        = _correlate(of_z, trk_z, mode="full")
            n            = len(of_z)
            mid          = len(xcorr) // 2
            max_lag_fr   = max(1, int(LAG_SEARCH_MS / FRAME_MS))
            lo, hi       = max(0, mid - max_lag_fr), min(len(xcorr), mid + max_lag_fr + 1)
            xcorr_win    = xcorr[lo:hi]

            best_i       = int(np.argmax(xcorr_win))
            lag_frames   = best_i - (mid - lo)
            lag_ms_found = float(lag_frames * FRAME_MS)
            r_best       = float(xcorr_win[best_i]) / max(n, 1)
            r_best       = float(np.clip(r_best, -1.0, 1.0))

            # Pearson à lag=0 (référence)
            r_lag0 = float(np.corrcoef(of_z, trk_z)[0, 1])

            # Score de corrélation : r normalisé sur [0,1]
            # r=0.7 → excellent, r=0.3 → limite, r<0.1 → mauvais
            r_score = float(np.clip((r_best - 0.10) / (0.70 - 0.10), 0.0, 1.0))

            # Pénalité si le lag optimal est > 1 frame (anormal sauf décalage systématique)
            # 1 frame de décalage est naturel pour le flux optique
            lag_excess_ms = max(0.0, abs(lag_ms_found) - FRAME_MS * 1.5)
            lag_penalty   = float(np.clip(1.0 - lag_excess_ms / LAG_SEARCH_MS, 0.0, 1.0))

            # Pénalité overlap
            overlap_score = float(np.clip(overlap_s / 30.0, 0.0, 1.0))  # max bonus à 30s

            cam_score = float(np.clip(
                0.70 * r_score + 0.20 * lag_penalty + 0.10 * overlap_score,
                0.0, 1.0
            ))

            details["cameras"][cam] = {
                "r_best":       round(r_best,   4),
                "r_lag0":       round(r_lag0,   4),
                "r_score":      round(r_score,  4),
                "lag_ms":       round(lag_ms_found, 1),
                "lag_penalty":  round(lag_penalty,  4),
                "overlap_s":    round(overlap_s,    2),
                "n_frames_of":  len(of_norms),
                "score":        round(cam_score, 4),
            }
            cam_scores[cam] = cam_score

            # HEAD suffit — c'est le signal le plus direct avec le tracker head
            # On s'arrête après head si disponible
            if cam == "head" and cam_score > 0:
                break

        if not cam_scores:
            return 0.70, {**details, "reason": "aucune caméra utilisable (vidéo manquante ou overlap insuffisant)"}

        # Priorité HEAD > moyenne LEFT+RIGHT
        if "head" in cam_scores:
            final = cam_scores["head"]
        else:
            final = float(np.mean(list(cam_scores.values())))

        details["n_cams_used"] = len(cam_scores)
        details["final_score"] = round(final, 4)
        return final, details

    except Exception as e:
        return 0.70, {**details, "reason": f"erreur: {e}"}

def _chk_load_model():
    if not _TORCH:
        return None
    if not _MODEL_PATH.exists():
        return None
    try:
        model = _torch.load(str(_MODEL_PATH), map_location="cpu")
        model.eval()
        return model
    except Exception:
        return None


def _chk_score_session_ia(model, session_dir: Path) -> Dict[str, float]:
    """Score IA via le modèle entraîné."""
    if model is None:
        return {}
    try:
        # Tente d'importer score_session_ia depuis check.py si disponible
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("check_mod", _ROOT / "check.py")
        if spec is None:
            return {}
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.score_session_ia(model, session_dir)
    except Exception:
        return {}


def _chk_compute_penalties(session_dir: Path, meta) -> float:
    """Calcul des pénalités qualité (non bloquantes)."""
    penalty = 1.0
    # Pénalité si trop de frames drops dans le JSONL
    for cam in ("head", "left", "right"):
        rows = _chk_read_jsonl(session_dir / "videos" / f"{cam}.jsonl")
        if not rows:
            continue
        ts = np.array([r.get("capture_time", 0) for r in rows], dtype=np.float64)
        if len(ts) < 2:
            continue
        dt = np.diff(ts)
        nominal = float(np.median(dt))
        if nominal <= 0:
            continue
        drops = (dt > 2.5 * nominal).sum()
        if drops > 10:
            penalty *= 0.85
        elif drops > 3:
            penalty *= 0.95
    return max(0.1, penalty)


def _chk_check_session(session_dir: Path, model=None) -> _SessionReport:
    report = _SessionReport(session_path=str(session_dir))

    if not _chk_gate_structure(session_dir, report):
        report.blocking_reason = report.first_failure().message
        report.score = 0.0
        return report

    meta = _chk_gate_metadata(session_dir, report)
    if meta is None:
        report.blocking_reason = "metadata.json illisible"
        report.score = 0.0
        return report

    if not _chk_gate_quaternions(session_dir, report):
        report.blocking_reason = report.first_failure().message
        report.score = 0.0
        return report

    if not _chk_gate_tracker_continuity(session_dir, report):
        report.blocking_reason = report.first_failure().message
        report.score = 0.0
        return report

    if not _chk_gate_camera_continuity(session_dir, report):
        report.blocking_reason = report.first_failure().message
        report.score = 0.0
        return report

    if not _chk_gate_camera_coverage(session_dir, report):
        report.blocking_reason = report.first_failure().message
        report.score = 0.0
        return report

    # stream_alignment : on enregistre le résultat mais on ne bloque plus ici.
    # La porte échouée sera visible dans les détails de video_tracker_sync,
    # et le score sera pénalisé proportionnellement au décalage.
    stream_ok = _chk_gate_stream_alignment(session_dir, report)

    # Score mathématique pur — aucun modèle requis
    math_raw, math_details = _chk_math_score(session_dir)

    # Pénalité stream_alignment : décalage > seuil réduit le score
    if not stream_ok:
        # Flux très désalignés → score réduit mais pas à 0 (réparable via fix+trim)
        math_raw = math_raw * 0.55
        report.blocking_reason = report.first_failure().message if report.first_failure() else "stream_alignment"

    penalty = _chk_compute_penalties(session_dir, meta)
    math_raw = float(np.clip(math_raw * penalty, 0.0, 1.0))

    report.ia_score  = round(math_raw, 4)
    report.ia_scores = math_details
    report.score     = round(math_raw * 100.0, 1)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════ MODULE 3 : TRACKER PLACEMENT (trakeur.py) ════════════
# Code intégré directement
# ══════════════════════════════════════════════════════════════════════════════

def _tk_moving_average(arr, window=9):
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(arr[:, j], kernel, mode="same")
    return out


def _tk_rank_points(values, higher_better=True):
    values = np.asarray(values, dtype=float)
    order  = np.argsort(values)
    pts    = np.zeros(len(values), dtype=float)
    if higher_better:
        pts[order[0]] = 0.0; pts[order[1]] = 1.0; pts[order[2]] = 2.0
    else:
        pts[order[0]] = 2.0; pts[order[1]] = 1.0; pts[order[2]] = 0.0
    return pts


def _tk_quat_to_rotmat_xyzw(q):
    q = q.astype(float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = 1 - 2*(y*y + z*z); R[:, 0, 1] = 2*(x*y - z*w); R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w);     R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w);     R[:, 2, 1] = 2*(y*z + x*w);     R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def _tk_quat_to_rotmat_wxyz(q):
    q = q.astype(float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = 1 - 2*(y*y + z*z); R[:, 0, 1] = 2*(x*y - z*w); R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w);     R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w);     R[:, 2, 1] = 2*(y*z + x*w);     R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def _tk_split_blocks(df, meta_cols=3, block_size=7, smooth_window=9):
    data    = df.iloc[:, meta_cols:].to_numpy(dtype=float)
    n_blocks = data.shape[1] // block_size
    if n_blocks != 3:
        raise ValueError(f"Expected 3 tracker blocks, found {n_blocks}")
    blocks = []
    for i in range(n_blocks):
        block = data[:, i*block_size:(i+1)*block_size]
        pos   = _tk_moving_average(block[:, :3], smooth_window)
        quat  = block[:, 3:7]
        blocks.append((i, pos, quat))
    return blocks


def _tk_parse_truth(df, meta_cols=3, block_size=7):
    cols    = list(df.columns)[meta_cols:]
    n_blocks = len(cols) // block_size
    if n_blocks != 3:
        raise ValueError(f"Expected 3 tracker blocks in headers, found {n_blocks}")
    truth = {}
    for i in range(n_blocks):
        block_cols = cols[i*block_size:(i+1)*block_size]
        joined     = " ".join(str(c).lower() for c in block_cols)
        found = [r for r in ("head", "left", "right") if r in joined]
        if len(found) != 1:
            raise ValueError(f"Cannot infer unique label for block {i}: {block_cols}")
        truth[found[0]] = i
    if set(truth.keys()) != {"head", "left", "right"}:
        raise ValueError(f"Incomplete truth mapping: {truth}")
    return truth


def _tk_detect_head(blocks):
    centers = []; motions = []; spreads = []
    for _, pos, _ in blocks:
        center = np.median(pos, axis=0)
        centers.append(center)
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        motions.append(np.median(step))
        radial = np.linalg.norm(pos - center, axis=1)
        spreads.append(np.median(radial))
    centers = np.asarray(centers)
    motions = np.asarray(motions)
    spreads = np.asarray(spreads)

    pair_mean = np.zeros(3, dtype=float)
    for i in range(3):
        d = []
        for j in range(3):
            if i == j: continue
            dij = np.linalg.norm(blocks[i][1] - blocks[j][1], axis=1)
            d.append(np.median(dij))
        pair_mean[i] = np.mean(d)

    h_y = centers[:, 1]
    best_sep_other = -np.inf
    best_h_other   = np.zeros(3, dtype=float)
    for axis in (0, 2):
        for sign in (-1, 1):
            h  = sign * centers[:, axis]
            hs = np.sort(h)
            sep = (hs[-1] - hs[-2]) / (np.ptp(h) + 1e-9)
            if sep > best_sep_other:
                best_sep_other = sep
                best_h_other   = h

    def _sep(vals, higher):
        sv  = np.sort(vals)
        gap = (sv[-1] - sv[-2]) if higher else (sv[1] - sv[0])
        return float(gap / (np.ptp(vals) + 1e-9))

    w_hy   = 6.0 * (1.0 + 2.0 * _sep(h_y,      higher=True))
    w_pair = 3.0 * (1.0 + 2.0 * _sep(pair_mean, higher=False))

    score = (
        1.0    * _tk_rank_points(motions,      higher_better=False) +
        1.0    * _tk_rank_points(spreads,      higher_better=False) +
        w_pair * _tk_rank_points(pair_mean,    higher_better=False) +
        w_hy   * _tk_rank_points(h_y,          higher_better=True)  +
        1.0    * _tk_rank_points(best_h_other, higher_better=True)
    )
    return int(np.argmax(score)), score


def _tk_predict_hands(blocks, head_idx, quat_mode, axis, sign):
    head   = next(b for b in blocks if b[0] == head_idx)
    others = [b for b in blocks if b[0] != head_idx]
    _, head_pos, head_quat = head
    idx_a, pos_a, _ = others[0]
    idx_b, pos_b, _ = others[1]

    if quat_mode == "xyzw":
        R = _tk_quat_to_rotmat_xyzw(head_quat)
    else:
        R = _tk_quat_to_rotmat_wxyz(head_quat)

    basis  = sign * R[:, :, axis]
    proj_a = np.sum((pos_a - head_pos) * basis, axis=1)
    proj_b = np.sum((pos_b - head_pos) * basis, axis=1)
    med_a  = float(np.median(proj_a))
    med_b  = float(np.median(proj_b))

    if med_a <= med_b:
        left, right = idx_a, idx_b
    else:
        left, right = idx_b, idx_a
    return left, right


def _tk_check_single_session(session_path) -> dict:
    """Vérifie le placement des trackers. Règle fixe : xyzw, axis=0, sign=-1."""
    if not _PANDAS:
        return {"ok": False, "error": "pandas non disponible", "pred": {}, "truth": {}, "correct": {}, "rule": None}

    RULE = ("xyzw", 0, -1)
    csv_path = Path(session_path) / "tracker_positions.csv"

    if not csv_path.exists():
        return {"ok": False, "error": "tracker_positions.csv introuvable",
                "pred": {}, "truth": {}, "correct": {}, "rule": None}

    try:
        df         = pd.read_csv(csv_path)
        blocks     = _tk_split_blocks(df)
        truth      = _tk_parse_truth(df)
        pred_head, _ = _tk_detect_head(blocks)
        pred_left, pred_right = _tk_predict_hands(blocks, pred_head, RULE[0], RULE[1], RULE[2])
        pred    = {"head": pred_head, "left": pred_left, "right": pred_right}
        correct = {k: pred[k] == truth[k] for k in ("head", "left", "right")}
        return {"ok": all(correct.values()), "pred": pred, "truth": truth,
                "correct": correct, "rule": RULE, "error": None}
    except Exception as e:
        return {"ok": False, "error": str(e), "pred": {}, "truth": {}, "correct": {}, "rule": None}


# ══════════════════════════════════════════════════════════════════════════════
# ═══════════════════ MODULE 4 : GRIPPER FRAME SYNC (gripper_frame_sync.py) ══
# Code intégré directement
# ══════════════════════════════════════════════════════════════════════════════

_GFS_ARM_Y_TOP      = 400
_GFS_DARK_THR       = 40
_GFS_DENSITY_MIN    = 0.5
_GFS_P_LEFT_INNER   = 70
_GFS_P_RIGHT_INNER  = 25
_GFS_MIN_DARK_COLS  = 4
_GFS_SMOOTH_WINDOW  = 15
_GFS_SMOOTH_POLY    = 3
_GFS_ROLL_WINDOW_S  = 3.0
_GFS_ROLL_STEP_S    = 1.0
_GFS_MOTION_MIN_MM  = 2.0
_GFS_R_MIN_OK       = 0.70
_GFS_R_GOOD         = 0.85
_GFS_LAG_MAX_MS     = 500.0
_GFS_TOL_SIGMA      = 2.0


@dataclass
class _GfsSideSyncResult:
    session:           str
    side:              str
    success:           bool
    error:             str   = ""
    n_frames:          int   = 0
    duration_s:        float = 0.0
    fps:               float = 0.0
    mean_dark_density: float = 0.0
    invalid_frames:    int   = 0
    confidence:        float = 0.0
    fit_a:             float = 0.0
    fit_b:             float = 0.0
    fit_r2:            float = 0.0
    global_r:          float = 0.0
    global_r_p:        float = 1.0
    lag_ms:            float = 0.0
    lag_frames:        int   = 0
    rolling_r_mean:    float = 0.0
    rolling_r_min:     float = 0.0
    rolling_r_max:     float = 0.0
    n_segments:        int   = 0
    bad_segments:      int   = 0
    good_segments:     int   = 0
    residual_std_px:   float = 0.0
    residual_std_mm:   float = 0.0
    px_per_mm:         float = 0.0
    tol_px:            float = 0.0
    frames_in_sync:    float = 0.0
    anomaly_frames:    List[int] = field(default_factory=list)
    score:             float = 0.0
    ok:                bool  = False
    verdict:           str   = ""
    per_frame:         List[dict] = field(default_factory=list)


def _gfs_load_jsonl(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    times = []
    try:
        raw = path.read_bytes()
        for line in raw.split(b'\n'):
            line = line.strip().rstrip(b'\r')
            if not line:
                continue
            try:
                obj = json.loads(line)
                ct  = obj.get("capture_time")
                if ct is not None:
                    times.append(float(ct))
            except Exception:
                continue
    except Exception:
        return None
    return np.array(times, dtype=np.float64) if len(times) > 10 else None


def _gfs_load_gripper_csv(path: Path):
    if not path.exists() or not _PANDAS:
        return None
    try:
        df = pd.read_csv(path)
        if "timestamp_ns" in df.columns:
            t_ms = df["timestamp_ns"].astype(float).to_numpy() / 1e6
        elif "t_ms" in df.columns:
            t_ms = df["t_ms"].astype(float).to_numpy()
        else:
            return None
        if "opening_mm" not in df.columns:
            return None
        opening = df["opening_mm"].astype(float).to_numpy()
        valid   = np.isfinite(t_ms) & np.isfinite(opening)
        if valid.sum() < 20:
            return None
        return t_ms[valid], opening[valid]
    except Exception:
        return None


def _gfs_extract_visual_gap(frame_bgr, arm_y_top=_GFS_ARM_Y_TOP, dark_thr=_GFS_DARK_THR,
                              p_left=_GFS_P_LEFT_INNER, p_right=_GFS_P_RIGHT_INNER,
                              density_min=_GFS_DENSITY_MIN, min_dark_cols=_GFS_MIN_DARK_COLS):
    gray = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    arm  = gray[arm_y_top:, :]
    half = W // 2

    col_dark     = (arm < dark_thr).sum(axis=0).astype(np.float32)
    dark_density = float(col_dark.mean()) / max(arm.shape[0], 1)
    threshold    = col_dark.mean() * density_min

    left_dark_cols  = np.where(col_dark[:half] > threshold)[0]
    right_dark_cols = np.where(col_dark[half:] > threshold)[0]

    if len(left_dark_cols) < min_dark_cols or len(right_dark_cols) < min_dark_cols:
        return None, dark_density

    left_inner  = float(np.percentile(left_dark_cols, p_left))
    right_inner = float(np.percentile(right_dark_cols, p_right)) + half
    gap = right_inner - left_inner
    return gap, dark_density


def _gfs_analyze_side(session_path: Path, side: str,
                       include_per_frame: bool = False) -> _GfsSideSyncResult:
    session_path = Path(session_path)
    sess_name    = session_path.name
    result = _GfsSideSyncResult(session=sess_name, side=side, success=False)

    if not _CV2:
        result.error = "opencv-python non installé"
        return result
    if not _SCIPY:
        result.error = "scipy non installé"
        return result
    if not _PANDAS:
        result.error = "pandas non installé"
        return result

    video_path = session_path / "videos" / f"{side}.mp4"
    jsonl_path = session_path / "videos" / f"{side}.jsonl"
    csv_path   = session_path / f"gripper_{side}_data.csv"

    missing = [p.name for p in [video_path, jsonl_path, csv_path] if not p.exists()]
    if missing:
        result.error = f"Fichiers absents : {missing}"
        return result

    times_ms = _gfs_load_jsonl(jsonl_path)
    if times_ms is None:
        result.error = f"Impossible de lire {jsonl_path.name}"
        return result

    gripper = _gfs_load_gripper_csv(csv_path)
    if gripper is None:
        result.error = f"Impossible de lire {csv_path.name}"
        return result

    ts_ms, opening_mm = gripper
    f_open            = _interp1d(ts_ms, opening_mm, bounds_error=False, fill_value=np.nan)
    sensor_at_frame   = f_open(times_ms)
    valid_mask        = np.isfinite(sensor_at_frame)

    if valid_mask.sum() < 30:
        result.error = f"Trop peu de frames avec données capteur ({valid_mask.sum()})"
        return result

    cap = _cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result.error = f"Impossible d'ouvrir {video_path.name}"
        return result

    fps_cap   = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    gap_raw:   List[float] = []
    s_open:    List[float] = []
    f_times:   List[float] = []
    f_indices: List[int]   = []
    dark_dens: List[float] = []
    n_invalid  = 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi < len(times_ms) and valid_mask[fi]:
            gap, density = _gfs_extract_visual_gap(frame)
            if gap is not None:
                gap_raw.append(gap)
                s_open.append(float(sensor_at_frame[fi]))
                f_times.append(float(times_ms[fi]))
                f_indices.append(fi)
                dark_dens.append(density)
            else:
                n_invalid += 1
        fi += 1
    cap.release()

    n = len(gap_raw)
    if n < 30:
        result.error = f"Trop peu de frames valides ({n}, invalides={n_invalid})"
        return result

    gap_raw   = np.array(gap_raw,   dtype=np.float64)
    s_open    = np.array(s_open,    dtype=np.float64)
    f_times   = np.array(f_times,   dtype=np.float64)
    dark_dens = np.array(dark_dens, dtype=np.float64)

    fps_real = float(n / ((f_times[-1] - f_times[0]) / 1000.0)) if f_times[-1] > f_times[0] else fps_cap

    total_frames = n + n_invalid
    valid_frac   = n / max(total_frames, 1)
    mean_density = float(dark_dens.mean())
    confidence   = float(np.clip(valid_frac * min(1.0, mean_density / 0.05), 0.0, 1.0))

    win  = min(_GFS_SMOOTH_WINDOW, n - 1 if (n - 1) % 2 == 0 else n - 2)
    win  = max(win | 1, 5)
    poly = min(_GFS_SMOOTH_POLY, win - 1)
    gap_smooth = _savgol(gap_raw, window_length=win, polyorder=poly)

    A_mat = np.vstack([s_open, np.ones(n)]).T
    coef, _, _, _ = np.linalg.lstsq(A_mat, gap_smooth, rcond=None)
    fit_a, fit_b  = float(coef[0]), float(coef[1])
    predicted     = fit_a * s_open + fit_b
    residuals     = gap_smooth - predicted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((gap_smooth - gap_smooth.mean())**2))
    r2     = max(0.0, 1.0 - ss_res / (ss_tot + 1e-12))

    px_per_mm    = abs(fit_a) if abs(fit_a) > 0.01 else 1.0
    residuals_mm = residuals / px_per_mm
    res_std      = float(np.std(residuals))
    tol_px       = max(15.0, _GFS_TOL_SIGMA * res_std)
    in_sync      = np.abs(residuals) <= tol_px

    # ── Détection du lag par cross-corrélation ──────────────────────────────
    def _znorm(x): return (x - x.mean()) / (x.std() + 1e-8)
    lag_range_frames = min(int(_GFS_LAG_MAX_MS * fps_real / 1000.0) + 5, n - 1)
    cc = _correlate(_znorm(gap_smooth), _znorm(s_open), mode="full")
    center = n - 1
    cc_masked = cc.copy()
    cc_masked[:center - lag_range_frames] = -np.inf
    cc_masked[center + lag_range_frames:] = -np.inf
    best_idx   = int(np.argmax(cc_masked))
    lag_frames = best_idx - center
    lag_ms     = float(lag_frames * 1000.0 / fps_real)

    # ── global_r calculé APRÈS alignement au lag détecté ─────────────────
    # Sans alignement, r ≈ 0 si le lag est > quelques frames.
    if lag_frames >= 0:
        gap_aligned = gap_smooth[lag_frames:]
        s_aligned   = s_open[:n - lag_frames]
    else:
        gap_aligned = gap_smooth[:n + lag_frames]
        s_aligned   = s_open[-lag_frames:]
    if len(gap_aligned) < 10:
        gap_aligned, s_aligned = gap_smooth, s_open  # fallback
    global_r, global_p = _pearsonr(gap_aligned, s_aligned)
    # |r| : l'orientation de la caméra peut inverser le gap visuellement
    global_r_abs = abs(global_r)

    win_fr  = max(int(_GFS_ROLL_WINDOW_S * fps_real), 15)
    step_fr = max(int(_GFS_ROLL_STEP_S   * fps_real), 5)
    rolling_r_vals: List[float] = []
    for i in range(0, n - win_fr + 1, step_fr):
        seg_g = gap_smooth[i:i + win_fr]
        seg_s = s_open[i:i + win_fr]
        if seg_s.std() < _GFS_MOTION_MIN_MM:
            continue
        r_seg, _ = _pearsonr(seg_g, seg_s)
        rolling_r_vals.append(abs(float(r_seg)))  # |r| par segment

    if rolling_r_vals:
        rr        = np.array(rolling_r_vals)
        roll_mean = float(rr.mean())
        roll_min  = float(rr.min())
        roll_max  = float(rr.max())
        n_seg     = len(rr)
        bad_seg   = int((rr < _GFS_R_MIN_OK).sum())
        good_seg  = int((rr >= _GFS_R_GOOD).sum())
    else:
        roll_mean = roll_min = roll_max = global_r_abs
        n_seg = bad_seg = good_seg = 0

    anomaly_mask   = np.abs(residuals) > 3.0 * tol_px
    anomaly_frames = [int(f_indices[i]) for i in np.where(anomaly_mask)[0]]

    score_r   = float(np.clip((global_r_abs - 0.30) / 0.70, 0.0, 1.0)) * 100.0
    score_seg = (float(n_seg - bad_seg) / n_seg * 100.0) if n_seg > 0 else score_r
    score_roll = float(np.clip((roll_mean - 0.30) / 0.70, 0.0, 1.0)) * 100.0 if rolling_r_vals else score_r
    # Meilleur des deux : global_r aligné ou rolling_r_mean (robustesse)
    score_corr = max(score_r, score_roll)
    lag_abs   = abs(lag_ms)
    if lag_abs <= _GFS_LAG_MAX_MS * 0.20:
        score_lag = 100.0
    elif lag_abs <= _GFS_LAG_MAX_MS * 0.60:
        score_lag = 100.0 - 50.0 * (lag_abs - _GFS_LAG_MAX_MS * 0.20) / (_GFS_LAG_MAX_MS * 0.40)
    else:
        score_lag = max(0.0, 50.0 - 50.0 * (lag_abs - _GFS_LAG_MAX_MS * 0.60) / (_GFS_LAG_MAX_MS * 0.40))
    score_conf = confidence * 100.0
    score = float(np.clip(0.45*score_corr + 0.30*score_seg + 0.15*score_lag + 0.10*score_conf, 0.0, 100.0))

    best_r = max(global_r_abs, roll_mean if rolling_r_vals else 0.0)
    ok = (best_r >= _GFS_R_MIN_OK and bad_seg <= max(1, n_seg // 4) and lag_abs <= _GFS_LAG_MAX_MS)

    if best_r >= _GFS_R_GOOD and bad_seg == 0 and lag_abs <= _GFS_LAG_MAX_MS * 0.20:
        verdict = "SYNC_PARFAITE"
    elif ok:
        verdict = "SYNC_OK"
    elif best_r >= 0.5:
        verdict = "SYNC_PARTIELLE"
    else:
        verdict = "DESYNC"

    per_frame: List[dict] = []
    if include_per_frame:
        for i in range(n):
            per_frame.append({
                "frame_idx": int(f_indices[i]), "t_ms": round(float(f_times[i]), 1),
                "sensor_mm": round(float(s_open[i]), 2), "visual_gap": round(float(gap_raw[i]), 1),
                "visual_smooth": round(float(gap_smooth[i]), 1),
                "predicted_gap": round(float(predicted[i]), 1),
                "residual_px": round(float(residuals[i]), 1),
                "residual_mm": round(float(residuals_mm[i]), 2),
                "dark_density": round(float(dark_dens[i]), 4),
                "in_sync": bool(in_sync[i]),
            })

    result.success           = True
    result.n_frames          = n
    result.duration_s        = round(float((f_times[-1] - f_times[0]) / 1000.0), 2)
    result.fps               = round(fps_real, 2)
    result.mean_dark_density = round(mean_density, 4)
    result.invalid_frames    = n_invalid
    result.confidence        = round(confidence, 3)
    result.fit_a             = round(fit_a, 4)
    result.fit_b             = round(fit_b, 2)
    result.fit_r2            = round(r2, 4)
    result.global_r          = round(float(global_r), 4)
    result.global_r_p        = round(float(global_p), 6)
    result.lag_ms            = round(lag_ms, 1)
    result.lag_frames        = int(lag_frames)
    result.rolling_r_mean    = round(roll_mean, 4)
    result.rolling_r_min     = round(roll_min, 4)
    result.rolling_r_max     = round(roll_max, 4)
    result.n_segments        = int(n_seg)
    result.bad_segments      = int(bad_seg)
    result.good_segments     = int(good_seg)
    result.residual_std_px   = round(res_std, 2)
    result.residual_std_mm   = round(float(res_std / px_per_mm), 2)
    result.px_per_mm         = round(px_per_mm, 4)
    result.tol_px            = round(tol_px, 1)
    result.frames_in_sync    = round(float(in_sync.mean()), 4)
    result.score             = round(score, 1)
    result.ok                = ok
    result.verdict           = verdict
    result.per_frame         = per_frame
    result.anomaly_frames    = anomaly_frames[:50]
    return result


def _gfs_analyze_session(session_path, sides=None, include_per_frame=False) -> dict:
    session_path = Path(session_path)
    if sides is None:
        sides = ["left", "right"]
    results: Dict[str, dict] = {}
    scores:  List[float]     = []
    all_ok   = True
    max_lag  = 0.0

    for side in sides:
        r = _gfs_analyze_side(session_path, side, include_per_frame=include_per_frame)
        d = asdict(r)
        # Nettoyage JSON
        def _clean(obj):
            if isinstance(obj, dict):   return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [_clean(v) for v in obj]
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating):
                v = float(obj)
                return None if not np.isfinite(v) else v
            if isinstance(obj, float) and not np.isfinite(obj): return None
            return obj
        results[side] = _clean(d)
        if r.success:
            scores.append(r.score)
            if not r.ok: all_ok = False
            max_lag = max(max_lag, abs(r.lag_ms))

    global_score = float(np.mean(scores)) if scores else 0.0

    # Si le signal visuel est trop bruité pour être fiable, on neutralise le score.
    # Critères : confidence < 0.75 OU invalid > 40% OU rolling_r_mean < 0.50 (aucune corrélation locale)
    confidences    = [results[s].get("confidence", 1.0) for s in results if results[s].get("success")]
    invalid_ratios = [
        results[s].get("invalid_frames", 0) / max(results[s].get("n_frames", 1), 1)
        for s in results if results[s].get("success")
    ]
    roll_means     = [results[s].get("rolling_r_mean") or 0.0 for s in results if results[s].get("success")]
    mean_conf      = float(np.mean(confidences))    if confidences    else 1.0
    mean_inv_ratio = float(np.mean(invalid_ratios)) if invalid_ratios else 0.0
    mean_roll_r    = float(np.mean(roll_means))     if roll_means     else 1.0
    signal_weak    = mean_conf < 0.75 or mean_inv_ratio > 0.40 or mean_roll_r < 0.50

    if not scores:
        verdict = "NO_DATA"
    elif signal_weak:
        verdict = "SIGNAL_FAIBLE"
        global_score = 50.0   # neutre — on ne pénalise pas un signal non mesurable
        all_ok = True         # ne bloque pas la session
    elif all_ok and global_score >= 80:
        verdict = "SYNC_PARFAITE"
    elif all_ok:
        verdict = "SYNC_OK"
    elif global_score >= 50:
        verdict = "SYNC_PARTIELLE"
    else:
        verdict = "DESYNC"

    return {
        "score":        round(global_score, 1),
        "ok":           all_ok and bool(scores),
        "verdict":      verdict,
        "lag_ms_max":   round(max_lag, 1),
        "n_sides":      len(scores),
        "signal_weak":    signal_weak,
        "mean_conf":      round(mean_conf, 2),
        "mean_inv_ratio": round(mean_inv_ratio, 2),
        "mean_roll_r":    round(mean_roll_r, 2),
        "sides":        results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ════════════════════════════ DIMENSIONS D'ANALYSE ═══════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def _dim_gripper_ts(session_path: Path) -> DimensionResult:
    diags:   List[str] = []
    repairs: List[str] = []
    details: Dict[str, Any] = {}
    missing: List[str] = []

    try:
        thr = _Thresholds(
            offset_ms      = GT_OFFSET_ERR_MS,
            latency_max_ms = GT_LATENCY_ERR_MS,
            jitter_std_ms  = GT_JITTER_ERR_MS,
            min_overlap_s  = GT_OVERLAP_MIN_S,
        )

        side_scores = []
        all_errors  = 0
        all_warns   = 0
        blocking    = False

        for side in ("left", "right"):
            r = _sp_process_side(str(session_path), side, thr)
            side_detail: Dict[str, Any] = {"status": r.status}

            if not r.success:
                diags.append(f"  {side}: FAILED — {r.error}")
                details[side] = side_detail
                missing.append(f"gripper_{side} / videos/{side}.jsonl")
                continue

            errors   = r.n_errors
            warnings = r.n_warnings
            all_errors  += errors
            all_warns   += warnings

            if errors > 0:
                penalty = min(25 * errors + 5 * (errors - 1), 65)
                s_side  = max(0.0, 100.0 - penalty)
            elif warnings > 0:
                s_side  = max(55.0, 100.0 - 10.0 * warnings)
            else:
                s_side  = 100.0

            aln = r.alignment
            if aln:
                side_detail.update({
                    "offset_start_ms":   round(aln.offset_start_ms, 1),
                    "latency_max_ms":    round(aln.latency_max_abs_ms, 1) if np.isfinite(aln.latency_max_abs_ms) else None,
                    "latency_p95_ms":    round(aln.latency_p95_abs_ms, 1) if np.isfinite(aln.latency_p95_abs_ms) else None,
                    "overlap_s":         round(aln.overlap_s, 2),
                    "sensor_gap_count":  aln.sensor_gap_count,
                    "sensor_gap_max_ms": round(aln.sensor_gap_max_ms, 1) if np.isfinite(aln.sensor_gap_max_ms) else None,
                    "frames_no_sensor":  aln.frames_no_sensor,
                })
                if abs(aln.offset_start_ms) > GT_OFFSET_ERR_MS:
                    repairs.append(
                        f"gripper_{side}: offset démarrage {aln.offset_start_ms:+.0f}ms — "
                        f"exécuter fix_camera_offset"
                    )
                    blocking = True
                if aln.overlap_s < GT_OVERLAP_MIN_S:
                    diags.append(f"  {side}: recouvrement critique {aln.overlap_s:.1f}s < {GT_OVERLAP_MIN_S}s")
                    blocking = True

            if r.video:
                side_detail["frame_drops"]   = r.video.frame_drops
                side_detail["jitter_std_ms"] = round(r.video.jitter_std_ms, 2)
                side_detail["n_frames"]      = r.video.n_frames
                side_detail["duration_s"]    = round(r.video.duration_s, 2)

            side_detail["score"]    = round(s_side, 1)
            side_detail["n_errors"] = errors
            side_detail["n_warns"]  = warnings

            for a in r.alerts:
                icon = "✗" if a.level == "ERROR" else "⚠"
                diags.append(f"  {side} {icon} {a.message}")
                if a.level == "ERROR" and "corrigeable" in a.message:
                    repairs.append(f"gripper_{side}: {a.message}")

            side_scores.append(s_side)
            details[side] = side_detail

        if not side_scores:
            return DimensionResult(
                name="gripper_timestamp_sync", score=0.0, weight=W_GRIPPER_TS,
                grade="F", ok=False, blocking=True, confidence=0.0,
                summary="Données gripper/vidéo absentes",
                diagnostics=diags, repairs=repairs, details=details,
                error="Aucune paire gripper/vidéo lisible",
            )

        score = float(np.mean(side_scores))
        conf  = 1.0 if len(side_scores) == 2 else 0.6

        if all_errors == 0:
            summary = f"Alignement gripper/vidéo nominal ({len(side_scores)} côtés)"
        else:
            summary = f"{all_errors} erreur(s), {all_warns} avert. sur {len(side_scores)} côté(s)"

        return DimensionResult(
            name="gripper_timestamp_sync", score=round(score, 1), weight=W_GRIPPER_TS,
            grade=_grade(score), ok=_dim_ok(score), blocking=blocking, confidence=conf,
            summary=summary, diagnostics=diags, repairs=repairs,
            details={"sides": details, "total_errors": all_errors, "total_warns": all_warns},
            error=None,
        )

    except Exception as e:
        return DimensionResult(
            name="gripper_timestamp_sync", score=50.0, weight=W_GRIPPER_TS,
            grade="C", ok=True, blocking=False, confidence=0.1,
            summary=f"Erreur gripper_timestamp_sync: {e}",
            diagnostics=[str(e)], repairs=[], details={}, error=str(e),
        )


def _dim_video_tracker_sync(session_path: Path, model=None) -> DimensionResult:
    diags:   List[str] = []
    repairs: List[str] = []
    details: Dict[str, Any] = {}

    try:
        report = _chk_check_session(session_path, model=None)  # modèle ignoré — math pur
        score  = float(report.score)
        failed = [g for g in report.gates if not g.passed]
        passed = [g for g in report.gates if g.passed]

        math_details = report.ia_scores if isinstance(report.ia_scores, dict) else {}
        cam_details  = math_details.get("cameras", {})

        details = {
            "math_score":      round(report.ia_score, 4) if report.ia_score else None,
            "math_details":    math_details,
            "n_gates_passed":  len(passed),
            "n_gates_failed":  len(failed),
            "failed_gates":    [{"name": g.name, "message": g.message} for g in failed],
            "blocking_reason": report.blocking_reason,
            # Compatibilité interface existante
            "ia_score":        round(report.ia_score, 4) if report.ia_score else None,
            "ia_scores":       {},
            "failed_gates":    [{"name": g.name, "message": g.message} for g in failed],
        }

        # stream_alignment bloquant → pas de score de sync fiable
        stream_failed = any(g.name == "stream_alignment" for g in failed)
        blocking = stream_failed or any(
            g.name not in ("stream_alignment",) for g in failed
        )

        for g in failed:
            diags.append(f"  ✗ porte [{g.name}]: {g.message}")
            if g.name in ("camera_coverage", "camera_continuity"):
                repairs.append(f"porte {g.name}: corriger l'offset caméra (fix_camera_offset.py)")
            if g.name == "stream_alignment":
                repairs.append(f"Flux désalignés au démarrage — exécuter ✂ Trim pour rogner le tracker/gripper")

        # Pénalité si stream_alignment en warning
        align_gate = next((g for g in report.gates if g.name == "stream_alignment"), None)
        if align_gate and align_gate.passed and "⚠" in align_gate.message:
            score = max(0.0, score * 0.80)

        for g in passed:
            diags.append(f"  ✓ porte [{g.name}]")

        # Détail par caméra
        for cam, cd in cam_details.items():
            if cd.get("corr_r") is not None:
                r     = cd["corr_r"]
                lag   = cd["lag_ms"]
                ov    = cd["overlap_s"]
                r2    = cd["clock_r2"]
                cs    = cd["score"]
                diags.append(
                    f"  {cam}: r={r:.3f}  lag_opt={lag:+.0f}ms  "
                    f"overlap={ov:.1f}s  clock_r²={r2:.3f}  → {cs*100:.1f}%"
                )
            else:
                diags.append(f"  {cam}: signal insuffisant")

        math_score_pct = round(report.ia_score * 100, 1) if report.ia_score else 0.0
        diags.append(f"  Score math global: {math_score_pct:.1f}%")

        if not failed:
            summary = f"Toutes portes OK — score math={math_score_pct:.1f}%"
        elif stream_failed:
            summary = f"{len(failed)} porte(s) échouée(s): {[g.name for g in failed]}"
        else:
            summary = f"{len(failed)} porte(s) échouée(s): {[g.name for g in failed]}"

        return DimensionResult(
            name="video_tracker_sync", score=round(score, 1), weight=W_VIDEO_SYNC,
            grade=_grade(score), ok=_dim_ok(score), blocking=blocking, confidence=0.90,
            summary=summary, diagnostics=diags, repairs=repairs, details=details, error=None,
        )

    except Exception as e:
        return DimensionResult(
            name="video_tracker_sync", score=0.0, weight=W_VIDEO_SYNC,
            grade="F", ok=False, blocking=True, confidence=0.0,
            summary=f"Erreur video_tracker_sync: {e}",
            diagnostics=[str(e)], repairs=[], details={}, error=str(e),
        )


def _dim_tracker_placement(session_path: Path) -> DimensionResult:
    diags:   List[str] = []
    repairs: List[str] = []

    try:
        result = _tk_check_single_session(session_path)

        ok      = result.get("ok", False)
        pred    = result.get("pred", {})
        truth   = result.get("truth", {})
        correct = result.get("correct", {})
        rule    = result.get("rule", ())
        err     = result.get("error")

        if err:
            return DimensionResult(
                name="tracker_placement", score=50.0, weight=W_TRACKER_PLAC,
                grade="C", ok=True, blocking=False, confidence=0.2,
                summary=f"Erreur trakeur: {err}",
                diagnostics=[f"  ✗ Erreur: {err}"],
                repairs=["Vérifier que tracker_positions.csv est présent et complet"],
                details={"error": err}, error=err,
            )

        details = {"ok": ok, "rule": list(rule) if rule else [],
                   "pred": pred, "truth": truth, "correct": correct}

        score    = 100.0 if ok else 0.0
        blocking = not ok
        wrong    = [k for k, v in correct.items() if not v] if correct else []

        for role in ("head", "left", "right"):
            p = pred.get(role, "?")
            t = truth.get(role, "?")
            c = correct.get(role, False)
            diags.append(f"  {'✓' if c else '✗'} {role}: prédit={p}  attendu={t}")

        if wrong:
            summary = f"Tracker(s) mal placé(s): {wrong}"
            repairs.append(
                f"Trackers {wrong} inversés — vérifier le port USB et le mapping dans les métadonnées"
            )
        else:
            summary = "Placement head/left/right correct (règle xyzw/0/-1)"

        diags.append(f"  Règle utilisée: {rule}")

        return DimensionResult(
            name="tracker_placement", score=score, weight=W_TRACKER_PLAC,
            grade=_grade(score), ok=not blocking, blocking=blocking, confidence=0.95,
            summary=summary, diagnostics=diags, repairs=repairs, details=details, error=None,
        )

    except Exception as e:
        return DimensionResult(
            name="tracker_placement", score=50.0, weight=W_TRACKER_PLAC,
            grade="C", ok=True, blocking=False, confidence=0.1,
            summary=f"Erreur tracker_placement: {e}",
            diagnostics=[str(e)], repairs=[], details={}, error=str(e),
        )


def _load_jsonl_timestamps_raw(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    times = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ct  = obj.get("capture_time")
                    if ct is not None:
                        times.append(float(ct))
                except Exception:
                    continue
    except Exception:
        return None
    return np.array(times, dtype=np.float64) if len(times) > 10 else None


def _dim_video_quality(session_path: Path) -> DimensionResult:
    diags:   List[str] = []
    repairs: List[str] = []
    details: Dict[str, Any] = {}
    missing: List[str] = []

    videos_dir      = session_path / "videos"
    scores_per_cam: List[float] = []
    all_drops = 0
    max_jitter = 0.0

    for cam in ("head", "left", "right"):
        jsonl = videos_dir / f"{cam}.jsonl"
        times = _load_jsonl_timestamps_raw(jsonl)

        if times is None:
            diags.append(f"  ✗ {cam}.jsonl : absent ou insuffisant")
            missing.append(f"videos/{cam}.jsonl")
            continue

        dt_ms = np.diff(times)
        if len(dt_ms) < 2:
            diags.append(f"  ✗ {cam}: trop peu de frames ({len(times)})")
            continue

        nominal_ms = float(np.median(dt_ms))
        jitter_ms  = float(np.median(np.abs(dt_ms - nominal_ms)) * 1.4826)
        drops      = int((dt_ms > 2.5 * nominal_ms).sum())
        gaps_large = int((dt_ms > 5.0 * nominal_ms).sum())
        duration_s = float((times[-1] - times[0]) / 1000.0)
        fps_est    = 1000.0 / nominal_ms if nominal_ms > 0 else 0.0

        n_expected = max(1, int(duration_s * fps_est))
        coverage   = min(1.0, len(times) / n_expected)

        jitter_pen = 30.0 if jitter_ms > VQ_JITTER_ERR_MS else (10.0 if jitter_ms > VQ_JITTER_WARN_MS else 0.0)
        drops_pen  = 30.0 if drops > VQ_DROPS_ERR else (10.0 if drops > VQ_DROPS_WARN else 0.0)
        cov_pen    = 25.0 if coverage < VQ_COVERAGE_ERR else (8.0 if coverage < VQ_COVERAGE_WARN else 0.0)
        gap_pen    = min(20.0, gaps_large * 7.0)

        cam_score = max(0.0, 100.0 - jitter_pen - drops_pen - cov_pen - gap_pen)
        scores_per_cam.append(cam_score)
        all_drops  += drops
        max_jitter  = max(max_jitter, jitter_ms)

        icon_j = "✗" if jitter_ms > VQ_JITTER_ERR_MS else ("⚠" if jitter_ms > VQ_JITTER_WARN_MS else "✓")
        icon_d = "✗" if drops > VQ_DROPS_ERR else ("⚠" if drops > VQ_DROPS_WARN else "✓")
        diags.append(
            f"  {cam}: {icon_j} jitter={jitter_ms:.1f}ms  "
            f"{icon_d} drops={drops}  dur={duration_s:.1f}s  fps≈{fps_est:.1f}"
        )

        if gaps_large > 0:
            diags.append(f"    ⚠ {cam}: {gaps_large} grand(s) gap(s) (>{5.0*nominal_ms:.0f}ms)")
            repairs.append(f"caméra {cam}: {gaps_large} gap(s) > 5× période — risque de désynchronisation")

        details[cam] = {
            "score": round(cam_score, 1), "jitter_ms": round(jitter_ms, 2),
            "drops": drops, "gaps_large": gaps_large, "coverage": round(coverage, 4),
            "duration_s": round(duration_s, 2), "fps_est": round(fps_est, 2),
            "n_frames": len(times),
        }

    if not scores_per_cam:
        return DimensionResult(
            name="video_quality", score=0.0, weight=W_VIDEO_QUAL,
            grade="F", ok=False, blocking=True, confidence=0.0,
            summary="Aucun fichier JSONL lisible",
            diagnostics=diags, repairs=repairs,
            details={"missing": missing}, error="Aucun JSONL disponible",
        )

    score    = float(np.mean(scores_per_cam))
    blocking = score < GRADE_D  # bloque si qualité vidéo insuffisante (< D)

    if all_drops == 0 and max_jitter < VQ_JITTER_WARN_MS:
        summary = f"Timestamps vidéo excellents ({len(scores_per_cam)} caméras)"
    elif all_drops > VQ_DROPS_ERR * 2:
        summary = f"{all_drops} frame drops total — qualité insuffisante"
    else:
        summary = f"Drops={all_drops} jitter_max={max_jitter:.1f}ms ({len(scores_per_cam)} caméras)"

    return DimensionResult(
        name="video_quality", score=round(score, 1), weight=W_VIDEO_QUAL,
        grade=_grade(score), ok=_dim_ok(score), blocking=blocking, confidence=0.95,
        summary=summary, diagnostics=diags, repairs=repairs,
        details={"cameras": details, "missing": missing}, error=None,
    )


def _dim_gripper_frame_sync(session_path: Path) -> DimensionResult:
    diags:   List[str] = []
    repairs: List[str] = []
    details: Dict[str, Any] = {}

    if not _CV2:
        return DimensionResult(
            name="gripper_frame_sync", score=60.0, weight=W_GRIP_FRAME,
            grade="C", ok=True, blocking=False, confidence=0.0,
            summary="cv2 non disponible — analyse par frame ignorée (score neutre 60%)",
            diagnostics=["  ⚠ pip install opencv-python"],
            repairs=[], details={"skipped": True}, error=None,
        )

    try:
        report = _gfs_analyze_session(session_path, sides=["left", "right"],
                                       include_per_frame=False)

        score       = float(report.get("score", 0.0))
        ok          = bool(report.get("ok", False))
        verdict     = report.get("verdict", "?")
        sides       = report.get("sides", {})
        signal_weak = report.get("signal_weak", False)

        for side, sr in sides.items():
            if not sr.get("success"):
                diags.append(f"  {side}: ✗ {sr.get('error', 'échec')}")
                details[side] = {"success": False, "error": sr.get("error")}
                continue

            r_val   = sr.get("global_r", 0.0) or 0.0
            lag_ms  = sr.get("lag_ms", 0.0) or 0.0
            r_mean  = sr.get("rolling_r_mean", r_val) or r_val
            r_min   = sr.get("rolling_r_min", r_val) or r_val
            n_seg   = sr.get("n_segments", 0) or 0
            bad_seg = sr.get("bad_segments", 0) or 0
            fis     = sr.get("frames_in_sync", 1.0) or 1.0
            anom    = sr.get("anomaly_frames", []) or []
            side_v  = sr.get("verdict", "?")
            side_s  = sr.get("score", 0.0) or 0.0

            if signal_weak:
                diags.append(f"  {side}: signal visuel trop faible pour mesure fiable (r={r_val:.3f}  roll_r={r_mean:.2f})")
            else:
                r_icon   = "✓" if r_val >= 0.70 else ("⚠" if r_val >= 0.50 else "✗")
                lag_icon = "✓" if abs(lag_ms) <= 50 else ("⚠" if abs(lag_ms) <= 100 else "✗")
                diags.append(
                    f"  {side}: {side_v}  score={side_s:.1f}%  "
                    f"{r_icon} r={r_val:.3f}  {lag_icon} lag={lag_ms:+.0f}ms"
                )
                if n_seg > 0:
                    seg_icon = "✓" if bad_seg == 0 else ("⚠" if bad_seg <= n_seg // 4 else "✗")
                    diags.append(f"    {seg_icon} fenêtres {n_seg-bad_seg}/{n_seg} OK  r_moy={r_mean:.3f}  r_min={r_min:.3f}")
                diags.append(f"    frames en sync: {fis*100:.1f}%  anomalies: {len(anom)}")

            if anom:
                diags.append(f"    ⚠ frames anormales: {anom[:10]}{'…' if len(anom) > 10 else ''}")
                if len(anom) > 5:
                    repairs.append(
                        f"gripper {side}: {len(anom)} frames avec désynchronisation — vérifier horloge commune"
                    )

            # Ne générer des repairs que si le signal visuel est fiable
            if not signal_weak:
                if r_val < 0.50:
                    repairs.append(
                        f"gripper {side}: corrélation visuelle faible (r={r_val:.3f}) — "
                        f"vérifier orientation caméra et seuil de segmentation"
                    )
                if abs(lag_ms) > 200:
                    # Ce lag est entre signal visuel et capteur CSV — indépendant de fix_camera_offset
                    repairs.append(
                        f"gripper {side}: lag visuel/capteur={lag_ms:+.0f}ms — "
                        f"vérifier la synchronisation horloge gripper"
                    )

            details[side] = {
                "success": True, "score": side_s, "verdict": side_v,
                "global_r": r_val, "lag_ms": lag_ms,
                "lag_frames": sr.get("lag_frames", 0),
                "rolling_r_mean": r_mean, "rolling_r_min": r_min,
                "rolling_r_max": sr.get("rolling_r_max", r_val),
                "n_segments": n_seg, "bad_segments": bad_seg,
                "frames_in_sync": fis,
                "residual_std_mm": sr.get("residual_std_mm", 0.0),
                "px_per_mm": sr.get("px_per_mm", 0.0),
                "fit_r2": sr.get("fit_r2", 0.0),
                "n_frames": sr.get("n_frames", 0),
                "anomaly_frames": anom[:20],
            }

        n_sides = report.get("n_sides", 0)
        conf    = 0.90 if n_sides == 2 else (0.5 if n_sides == 1 else 0.0)

        if not sides or n_sides == 0:
            return DimensionResult(
                name="gripper_frame_sync", score=0.0, weight=W_GRIP_FRAME,
                grade="F", ok=False, blocking=False, confidence=0.0,
                summary="Aucune donnée gripper/vidéo disponible",
                diagnostics=diags, repairs=repairs, details=details, error=None,
            )

        if verdict == "SYNC_PARFAITE":    summary = f"Sync par frame parfaite ({n_sides} côtés, score={score:.1f}%)"
        elif verdict == "SYNC_OK":        summary = f"Sync par frame OK ({n_sides} côtés, score={score:.1f}%)"
        elif verdict == "SYNC_PARTIELLE": summary = f"Sync par frame partielle ({score:.1f}%) — vérifier le lag"
        elif verdict == "SIGNAL_FAIBLE":  summary = f"Signal visuel insuffisant — mesure non conclusive ({score:.1f}%)"
        else:                             summary = f"DÉSYNC détectée ({score:.1f}%) — corrélation insuffisante"

        return DimensionResult(
            name="gripper_frame_sync", score=round(score, 1), weight=W_GRIP_FRAME,
            grade=_grade(score), ok=ok, blocking=False, confidence=conf,
            summary=summary, diagnostics=diags, repairs=repairs,
            details={"session_verdict": verdict, "lag_ms_max": report.get("lag_ms_max", 0.0), "sides": details},
            error=None,
        )

    except Exception as e:
        return DimensionResult(
            name="gripper_frame_sync", score=50.0, weight=W_GRIP_FRAME,
            grade="C", ok=True, blocking=False, confidence=0.0,
            summary=f"Erreur gripper_frame_sync: {e}",
            diagnostics=[str(e)], repairs=[], details={}, error=str(e),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée principal
# ══════════════════════════════════════════════════════════════════════════════

def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _dim_to_dict(d: DimensionResult) -> dict:
    return {
        "score":       d.score,
        "grade":       d.grade,
        "weight":      d.weight,
        "ok":          d.ok,
        "blocking":    d.blocking,
        "confidence":  d.confidence,
        "summary":     d.summary,
        "diagnostics": d.diagnostics,
        "repairs":     d.repairs,
        "details":     _make_serializable(d.details),
        "error":       d.error,
    }


def _load_session_metadata(session_path: Path) -> Dict[str, Any]:
    meta_path = session_path / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def check_session_full(session_path, model=None) -> dict:
    """
    Point d'entrée principal — appelé par server.py via /api/pipeline/check_score.

    Exécute les 5 dimensions d'analyse et produit un rapport d'autorité complet.
    Tout le code est intégré dans ce fichier, aucune dépendance sur les autres
    modules du dossier verification/.

    Retourne un dict compatible avec l'interface server.py :
      { score, grade, perfect, blocked, blocking_reason, verdict, confidence,
        repairs, missing_data, session_meta, components{5 dimensions},
        + clés de compatibilité : ia_score, failed_gates, tracker_ok, gripper_sync }
    """
    session_path = Path(session_path)
    session_name = session_path.name

    # Charger le modèle IA une seule fois si non fourni
    if model is None:
        model = _chk_load_model()

    # ── Exécuter les 5 dimensions ──────────────────────────────────────────
    dim_gt  = _dim_gripper_ts(session_path)
    dim_vt  = _dim_video_tracker_sync(session_path, model=model)
    dim_tp  = _dim_tracker_placement(session_path)
    dim_vq  = _dim_video_quality(session_path)
    dim_gf  = _dim_gripper_frame_sync(session_path)

    dims = {
        "gripper_timestamp_sync": dim_gt,
        "video_tracker_sync":     dim_vt,
        "tracker_placement":      dim_tp,
        "video_quality":          dim_vq,
        "gripper_frame_sync":     dim_gf,
    }

    # ── Score global pondéré ──────────────────────────────────────────────
    total_w = sum(d.weight for d in dims.values())
    total_s = sum(d.weight * d.score for d in dims.values())
    global_score = float(np.clip(total_s / max(total_w, 1e-6), 0.0, 100.0))

    # ── Portes bloquantes ─────────────────────────────────────────────────
    blocking_reason = ""
    for d in dims.values():
        if d.blocking and not blocking_reason:
            blocking_reason = f"[{d.name}] {d.summary}"

    is_blocked = bool(blocking_reason)

    # Une dimension en dessous de C (60) est rédhibitoire même sans porte bloquante
    weak_dims = [d for d in dims.values() if d.score < GRADE_C]
    if weak_dims and not is_blocked:
        blocking_reason = (
            f"dimension(s) insuffisante(s) : "
            + ", ".join(f"{d.name}={d.score:.0f}%" for d in weak_dims)
        )
        is_blocked = True

    is_perfect = not is_blocked and global_score >= PERFECT_THRESHOLD
    grade      = _grade(global_score)

    # ── Verdict ───────────────────────────────────────────────────────────
    if is_perfect:                 verdict = "PARFAITE"
    elif is_blocked:               verdict = "BLOQUÉE"
    elif global_score >= GRADE_B:  verdict = "ACCEPTABLE"
    elif global_score >= GRADE_D:  verdict = "ISSUES"
    else:                          verdict = "ÉCHEC"

    # ── Confiance globale ─────────────────────────────────────────────────
    conf_global = float(np.average(
        [d.confidence for d in dims.values()],
        weights=[d.weight for d in dims.values()]
    ))

    # ── Réparations consolidées ───────────────────────────────────────────
    all_repairs: List[str] = []
    seen = set()
    for d in dims.values():
        for r in d.repairs:
            if r not in seen:
                all_repairs.append(r)
                seen.add(r)

    # ── Données manquantes ────────────────────────────────────────────────
    missing: List[str] = []
    for d in (dim_gt, dim_vq):
        m = d.details.get("missing", [])
        if isinstance(m, list):
            missing.extend(m)

    # ── Métadonnées session ───────────────────────────────────────────────
    meta = _load_session_metadata(session_path)
    session_meta = {
        "session_name": session_name,
        "task":         meta.get("task", ""),
        "robot":        meta.get("robot", ""),
        "operator":     meta.get("operator", ""),
    }

    # ── Compatibilité server.py ───────────────────────────────────────────
    vt = dim_vt.details
    tp = dim_tp.details
    gv_left  = dim_gt.details.get("sides", {}).get("left", {})
    gv_right = dim_gt.details.get("sides", {}).get("right", {})

    report = {
        # Rapport d'autorité
        "score":           round(global_score, 1),
        "grade":           grade,
        "perfect":         is_perfect,
        "blocked":         is_blocked,
        "blocking_reason": blocking_reason,
        "verdict":         verdict,
        "confidence":      round(conf_global, 3),
        "repairs":         all_repairs,
        "missing_data":    missing,
        "session_meta":    session_meta,

        # Dimensions détaillées
        "components": {
            "gripper_timestamp_sync": _dim_to_dict(dim_gt),
            "video_tracker_sync":     _dim_to_dict(dim_vt),
            "tracker_placement":      _dim_to_dict(dim_tp),
            "video_quality":          _dim_to_dict(dim_vq),
            "gripper_frame_sync":     _dim_to_dict(dim_gf),
        },

        # Clés de compatibilité (ancienne interface)
        "ia_score":       vt.get("ia_score"),
        "ia_scores":      vt.get("ia_scores", {}),
        "failed_gates":   vt.get("failed_gates", []),
        "tracker_ok":     tp.get("ok"),
        "tracker_result": {
            "pred":    tp.get("pred", {}),
            "truth":   tp.get("truth", {}),
            "correct": tp.get("correct", {}),
        },
        "gripper_sync": {
            "left":  {
                "corr":      gv_left.get("latency_max_ms"),
                "lag_ms":    gv_left.get("offset_start_ms"),
                "overlap_s": gv_left.get("overlap_s"),
                "ok":        gv_left.get("n_errors", 1) == 0,
            },
            "right": {
                "corr":      gv_right.get("latency_max_ms"),
                "lag_ms":    gv_right.get("offset_start_ms"),
                "overlap_s": gv_right.get("overlap_s"),
                "ok":        gv_right.get("n_errors", 1) == 0,
            },
        },
    }

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI standalone
# ══════════════════════════════════════════════════════════════════════════════

_GRADE_COLORS = {
    "A": "\033[92m", "B": "\033[92m",
    "C": "\033[93m", "D": "\033[93m",
    "F": "\033[91m",
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _c(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{_RESET}"
    return text


def _print_report(result: dict, full: bool = False) -> None:
    score   = result["score"]
    grade   = result["grade"]
    verdict = result["verdict"]
    perfect = result["perfect"]
    blocked = result["blocked"]
    conf    = result["confidence"]

    vc = _GRADE_COLORS.get(grade, "")
    stars = "★" if perfect else ("✗" if blocked else "●")
    print(f"\n{_c(f'{stars} {verdict}', _BOLD + vc)}  "
          f"score={score:.1f}%  grade={_c(grade, vc)}  confiance={conf:.0%}")

    if blocked:
        print(f"  Blocage : {result['blocking_reason']}")

    print(f"\n{'─'*60}")
    print(f"  {'Dimension':<30}  {'Score':>6}  {'Grade':>5}  Résumé")
    print(f"{'─'*60}")

    dim_order = ["gripper_timestamp_sync", "video_tracker_sync",
                 "tracker_placement", "video_quality", "gripper_frame_sync"]
    labels = {
        "gripper_timestamp_sync": "Gripper TS sync      (25%)",
        "video_tracker_sync":     "Vidéo/tracker IA     (25%)",
        "tracker_placement":      "Placement trackers   (15%)",
        "video_quality":          "Qualité vidéo        (20%)",
        "gripper_frame_sync":     "Sync frame/capteur   (15%)",
    }

    for key in dim_order:
        d = result["components"].get(key)
        if not d:
            continue
        g     = d["grade"]
        gc    = _GRADE_COLORS.get(g, "")
        blk   = " [BLOQUANT]" if d.get("blocking") else ""
        label = labels.get(key, key)
        print(f"  {label:<30}  {d['score']:>5.1f}%  {_c(g, gc):>5}{blk}  {d['summary']}")

    print(f"{'─'*60}")

    if result.get("repairs"):
        print(f"\n  ⚙ Réparations recommandées:")
        for r in result["repairs"]:
            print(f"    • {r}")

    if result.get("missing_data"):
        print(f"\n  ⚠ Données manquantes:")
        for m in result["missing_data"]:
            print(f"    • {m}")

    if full:
        print(f"\n{'═'*60}")
        print("  DIAGNOSTICS DÉTAILLÉS")
        print(f"{'═'*60}")
        for key in dim_order:
            d = result["components"].get(key)
            if not d:
                continue
            label = labels.get(key, key)
            g  = d["grade"]
            gc = _GRADE_COLORS.get(g, "")
            print(f"\n  [{_c(g, gc)}] {label}")
            for line in d.get("diagnostics", []):
                print(f"  {line}")

    print()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Autorité de vérification complète d'une session robot (fichier autonome)"
    )
    p.add_argument("session",  help="Chemin vers la session")
    p.add_argument("--json",   action="store_true", help="Sortie JSON brute")
    p.add_argument("--full",   action="store_true", help="Diagnostics complets")
    args = p.parse_args()

    result = check_session_full(Path(args.session))

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        _print_report(result, full=args.full)
