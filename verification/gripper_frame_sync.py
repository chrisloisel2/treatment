#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gripper_frame_sync.py — Analyse par frame de la synchronisation gripper.

Pour chaque frame vidéo, compare l'ouverture VISUELLE des doigts du gripper
(extraite par analyse de pixels) avec l'ouverture CAPTEUR (interpolée depuis
le CSV à l'instant exact de la frame via l'horloge Unix commune).

Méthode de segmentation visuelle
──────────────────────────────────
Feature "inner-edge percentile" (r ≈ 0.92 validé sur 15 sessions, 30 côtés) :

  1. Bande bras = image[arm_y_top:, :]
  2. Profil de densité sombre par colonne : col_dark[c] = nombre de pixels < dark_thr
  3. Seuil d'appartenance = mean(col_dark) * 0.5 (adaptatif à la luminosité)
  4. left_dark_cols  = colonnes appartenant au doigt gauche (demi-image gauche)
  5. right_dark_cols = colonnes appartenant au doigt droit (demi-image droite)
  6. Bord interne gauche = P70(left_dark_cols)   — robuste aux outliers
  7. Bord interne droit  = P25(right_dark_cols)  — robuste aux outliers
  8. Gap visuel = bord_droit – bord_gauche (px)

Ce choix est supérieur au centroïde de masse brut car :
  - Les bords internes sont invariants aux ombres (qui élargissent le blob)
  - Le percentile filtre les pixels parasites (fond bruité, réflexions)
  - Calibré empiriquement sur 30 (session,côté) : médiane r=0.89, std résidu ≈ 8mm

Pipeline complet
────────────────
  1. Extraire gap_px par frame (inner-edge percentile)
  2. Lisser avec Savitzky-Golay (fenêtre adaptée à la durée)
  3. Interpoler opening_mm CSV à chaque frame (horloge commune Unix)
  4. Ajustement linéaire : gap_prédit = a * opening_mm + b
     → calibration px/mm automatique, robuste aux variations de montage caméra
  5. Résidu par frame = gap_lissé – gap_prédit (en mm via px/mm)
  6. Corrélation de Pearson globale + fenêtres glissantes 3s
  7. Cross-corrélation pour le lag temporel
  8. Score composite : 0–100

Métriques de sortie (par côté)
────────────────────────────────
  global_r          r de Pearson sur toute la session
  lag_ms            décalage temporel optimal
  rolling_r_mean    corrélation moyenne sur fenêtres glissantes 3s
  bad_segments      fenêtres avec r < R_MIN_OK (0.70)
  frames_in_sync    fraction de frames avec |résidu| < tolérance calibrée
  anomaly_frames    indices des frames avec |résidu| > 3× tolérance
  per_frame         [{frame_idx, t_ms, sensor_mm, visual_gap, predicted_gap,
                       residual_px, residual_mm, in_sync}, ...]
  confidence        0–1, fiabilité de la mesure (basée sur dark_density et r2)

Usage CLI
─────────
  python gripper_frame_sync.py <session>
  python gripper_frame_sync.py <session> --side left --verbose
  python gripper_frame_sync.py <session> --json --no-per-frame
  python gripper_frame_sync.py <root> --batch
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── deps optionnels ───────────────────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    cv2 = None
    CV2_OK = False

try:
    from scipy.interpolate import interp1d
    from scipy.signal import correlate, savgol_filter
    from scipy.stats import pearsonr
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

ARM_Y_TOP      = 400     # début de la bande bras (px depuis le haut)
DARK_THR       = 40      # seuil de noirceur (pixel < thr → bras)
DARK_DENSITY_MIN = 0.5   # fraction minimale du profil sombre pour être dans un doigt
P_LEFT_INNER   = 70      # percentile du bord interne du doigt gauche (calibré)
P_RIGHT_INNER  = 25      # percentile du bord interne du doigt droit (calibré)
MIN_DARK_COLS  = 4       # nb minimum de colonnes sombres pour une mesure valide

SMOOTH_WINDOW  = 15      # fenêtre Savitzky-Golay (frames)
SMOOTH_POLY    = 3       # ordre polynôme SG

ROLL_WINDOW_S  = 3.0     # durée fenêtre glissante (s)
ROLL_STEP_S    = 1.0     # pas fenêtre glissante (s)
MOTION_MIN_MM  = 2.0     # std ouverture minimale pour scorer un segment

R_MIN_OK       = 0.70    # corrélation minimale par segment
R_GOOD         = 0.85    # corrélation "bonne"
LAG_MAX_MS     = 100.0   # lag temporel maximal acceptable (ms)
TOL_SIGMA      = 2.0     # nb de sigma pour la tolérance de résidu


# ══════════════════════════════════════════════════════════════════════════════
# Structures de données
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SideSyncResult:
    """Résultat complet d'un côté (left ou right)."""
    session:      str
    side:         str
    success:      bool
    error:        str = ""

    # ── Infos vidéo ──────────────────────────────────────────────────────────
    n_frames:     int   = 0
    duration_s:   float = 0.0
    fps:          float = 0.0

    # ── Qualité de la segmentation visuelle ──────────────────────────────────
    mean_dark_density: float = 0.0   # densité moyenne de pixels sombres dans la bande
    invalid_frames:    int   = 0     # frames sans mesure valide (doigts non détectés)
    confidence:        float = 0.0   # 0–1 : fiabilité de la mesure

    # ── Modèle linéaire gap_px = a * opening_mm + b ──────────────────────────
    fit_a:   float = 0.0   # px/mm (doit être positif, ~4–7)
    fit_b:   float = 0.0   # px (intercept)
    fit_r2:  float = 0.0   # R² de l'ajustement

    # ── Corrélation globale ───────────────────────────────────────────────────
    global_r:   float = 0.0
    global_r_p: float = 1.0

    # ── Lag temporel ─────────────────────────────────────────────────────────
    lag_ms:     float = 0.0
    lag_frames: int   = 0

    # ── Fenêtres glissantes ───────────────────────────────────────────────────
    rolling_r_mean: float = 0.0
    rolling_r_min:  float = 0.0
    rolling_r_max:  float = 0.0
    n_segments:     int   = 0
    bad_segments:   int   = 0
    good_segments:  int   = 0

    # ── Résidus par frame ────────────────────────────────────────────────────
    residual_std_px:  float = 0.0
    residual_std_mm:  float = 0.0
    px_per_mm:        float = 0.0
    tol_px:           float = 0.0
    frames_in_sync:   float = 0.0   # 0–1
    anomaly_frames:   List[int] = field(default_factory=list)

    # ── Score et verdict ─────────────────────────────────────────────────────
    score:   float = 0.0
    ok:      bool  = False
    verdict: str   = ""

    # ── Données par frame ────────────────────────────────────────────────────
    per_frame: List[dict] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

def _load_jsonl(path: Path) -> Optional[np.ndarray]:
    """Charge les capture_time (ms) depuis un .jsonl (CRLF ou LF)."""
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
                ct = obj.get("capture_time")
                if ct is not None:
                    times.append(float(ct))
            except Exception:
                continue
    except Exception:
        return None
    return np.array(times, dtype=np.float64) if len(times) > 10 else None


def _load_gripper_csv(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Charge (timestamp_ms, opening_mm) depuis gripper_*_data.csv."""
    if not path.exists() or not PANDAS_OK:
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
        valid = np.isfinite(t_ms) & np.isfinite(opening)
        if valid.sum() < 20:
            return None
        return t_ms[valid], opening[valid]
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Extraction du gap visuel — méthode inner-edge percentile
# ══════════════════════════════════════════════════════════════════════════════

def _extract_visual_gap(
    frame_bgr: np.ndarray,
    arm_y_top:    int   = ARM_Y_TOP,
    dark_thr:     int   = DARK_THR,
    p_left:       int   = P_LEFT_INNER,
    p_right:      int   = P_RIGHT_INNER,
    density_min:  float = DARK_DENSITY_MIN,
    min_dark_cols: int  = MIN_DARK_COLS,
) -> Tuple[Optional[float], float]:
    """
    Extrait le gap entre les deux doigts du gripper dans une frame.

    Algorithme inner-edge percentile :
    1. Convertir en niveaux de gris
    2. Bande bras = image[arm_y_top:, :]
    3. Profil sombre = sum(pixels < dark_thr) par colonne
    4. Seuil d'appartenance = mean * density_min (adaptatif)
    5. Colonnes du doigt gauche = colonnes demi-gauche au-dessus du seuil
    6. Colonnes du doigt droit  = colonnes demi-droite au-dessus du seuil
    7. Bord interne gauche = percentile(p_left, left_dark_cols)
    8. Bord interne droit  = percentile(p_right, right_dark_cols) + W/2
    9. Gap = bord_droit - bord_gauche

    Returns:
        (gap_px, dark_density) ou (None, density) si doigts non détectés
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    arm  = gray[arm_y_top:, :]
    half = W // 2

    # Profil de densité sombre par colonne
    col_dark = (arm < dark_thr).sum(axis=0).astype(np.float32)

    # Densité normalisée (fraction de la hauteur de bande)
    dark_density = float(col_dark.mean()) / max(arm.shape[0], 1)

    # Seuil adaptatif à la luminosité globale de la bande
    threshold = col_dark.mean() * density_min

    # Colonnes appartenant à chaque doigt
    left_dark_cols  = np.where(col_dark[:half] > threshold)[0]
    right_dark_cols = np.where(col_dark[half:] > threshold)[0]

    if len(left_dark_cols) < min_dark_cols or len(right_dark_cols) < min_dark_cols:
        return None, dark_density

    # Bords internes via percentile (robuste aux outliers de segmentation)
    left_inner  = float(np.percentile(left_dark_cols, p_left))
    right_inner = float(np.percentile(right_dark_cols, p_right)) + half

    gap = right_inner - left_inner
    return gap, dark_density


# ══════════════════════════════════════════════════════════════════════════════
# Analyse d'un côté
# ══════════════════════════════════════════════════════════════════════════════

def analyze_side(
    session_path: Path,
    side: str,
    include_per_frame: bool = True,
    arm_y_top:  int   = ARM_Y_TOP,
    dark_thr:   int   = DARK_THR,
    p_left:     int   = P_LEFT_INNER,
    p_right:    int   = P_RIGHT_INNER,
) -> SideSyncResult:
    """
    Analyse complète de la synchronisation gripper visuel ↔ capteur pour un côté.

    Args:
        session_path:      chemin de la session
        side:              "left" ou "right"
        include_per_frame: inclure la liste par frame dans le résultat
        arm_y_top:         ligne de départ de la bande bras (px)
        dark_thr:          seuil pixel sombre
        p_left:            percentile bord interne doigt gauche (défaut 70)
        p_right:           percentile bord interne doigt droit  (défaut 25)

    Returns:
        SideSyncResult complet
    """
    session_path = Path(session_path)
    sess_name    = session_path.name
    result = SideSyncResult(session=sess_name, side=side, success=False)

    # ── Vérifications préalables ──────────────────────────────────────────────
    if not CV2_OK:
        result.error = "opencv-python non installé (pip install opencv-python)"
        return result
    if not SCIPY_OK:
        result.error = "scipy non installé (pip install scipy)"
        return result
    if not PANDAS_OK:
        result.error = "pandas non installé (pip install pandas)"
        return result

    video_path = session_path / "videos" / f"{side}.mp4"
    jsonl_path = session_path / "videos" / f"{side}.jsonl"
    csv_path   = session_path / f"gripper_{side}_data.csv"

    missing = [p.name for p in [video_path, jsonl_path, csv_path] if not p.exists()]
    if missing:
        result.error = f"Fichiers absents : {missing}"
        return result

    # ── Chargement timestamps ────────────────────────────────────────────────
    times_ms = _load_jsonl(jsonl_path)
    if times_ms is None:
        result.error = f"Impossible de lire {jsonl_path.name}"
        return result

    gripper = _load_gripper_csv(csv_path)
    if gripper is None:
        result.error = f"Impossible de lire {csv_path.name}"
        return result

    ts_ms, opening_mm = gripper

    # ── Interpolation capteur aux instants vidéo ─────────────────────────────
    f_open = interp1d(ts_ms, opening_mm, bounds_error=False, fill_value=np.nan)
    sensor_at_frame = f_open(times_ms)
    valid_mask = np.isfinite(sensor_at_frame)

    if valid_mask.sum() < 30:
        result.error = f"Trop peu de frames avec données capteur ({valid_mask.sum()})"
        return result

    # ── Extraction du gap visuel frame par frame ──────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result.error = f"Impossible d'ouvrir {video_path.name}"
        return result

    fps_cap  = cap.get(cv2.CAP_PROP_FPS) or 30.0

    gap_raw:    List[float]  = []   # gap visuel brut par frame
    s_open:     List[float]  = []   # ouverture capteur interpolée
    f_times:    List[float]  = []   # timestamp ms de chaque frame
    f_indices:  List[int]    = []   # indice original dans times_ms
    dark_dens:  List[float]  = []   # densité sombre (indicateur qualité)
    n_invalid   = 0

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi < len(times_ms) and valid_mask[fi]:
            gap, density = _extract_visual_gap(
                frame, arm_y_top=arm_y_top, dark_thr=dark_thr,
                p_left=p_left, p_right=p_right,
            )
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

    gap_raw  = np.array(gap_raw,  dtype=np.float64)
    s_open   = np.array(s_open,   dtype=np.float64)
    f_times  = np.array(f_times,  dtype=np.float64)
    dark_dens = np.array(dark_dens, dtype=np.float64)

    fps_real = float(n / ((f_times[-1] - f_times[0]) / 1000.0)) if f_times[-1] > f_times[0] else fps_cap

    # ── Confiance de la segmentation ─────────────────────────────────────────
    # Basée sur la densité dark moyenne et la fraction de frames valides
    total_frames = n + n_invalid
    valid_frac   = n / max(total_frames, 1)
    mean_density = float(dark_dens.mean())
    confidence   = float(np.clip(valid_frac * min(1.0, mean_density / 0.05), 0.0, 1.0))

    # ── Lissage Savitzky-Golay ────────────────────────────────────────────────
    win = min(SMOOTH_WINDOW, n - 1 if (n - 1) % 2 == 0 else n - 2)
    win = max(win | 1, 5)
    poly = min(SMOOTH_POLY, win - 1)
    gap_smooth = savgol_filter(gap_raw, window_length=win, polyorder=poly)

    # ── Ajustement linéaire ───────────────────────────────────────────────────
    A_mat = np.vstack([s_open, np.ones(n)]).T
    coef, _, _, _ = np.linalg.lstsq(A_mat, gap_smooth, rcond=None)
    fit_a, fit_b = float(coef[0]), float(coef[1])

    predicted  = fit_a * s_open + fit_b
    residuals  = gap_smooth - predicted

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((gap_smooth - gap_smooth.mean()) ** 2))
    r2     = max(0.0, 1.0 - ss_res / (ss_tot + 1e-12))

    px_per_mm   = abs(fit_a) if abs(fit_a) > 0.01 else 1.0
    residuals_mm = residuals / px_per_mm

    # Tolérance : TOL_SIGMA × écart-type des résidus
    res_std = float(np.std(residuals))
    tol_px  = max(15.0, TOL_SIGMA * res_std)
    in_sync = np.abs(residuals) <= tol_px

    # ── Corrélation globale ───────────────────────────────────────────────────
    global_r, global_p = pearsonr(gap_smooth, s_open)

    # ── Lag temporel (cross-corrélation) ─────────────────────────────────────
    def _znorm(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-8)

    lag_range_frames = min(int(LAG_MAX_MS * fps_real / 1000.0) + 5, n - 1)
    cc = correlate(_znorm(gap_smooth), _znorm(s_open), mode="full")
    center = n - 1
    cc_masked = cc.copy()
    cc_masked[:center - lag_range_frames] = -np.inf
    cc_masked[center + lag_range_frames:] = -np.inf
    best_idx   = int(np.argmax(cc_masked))
    lag_frames = best_idx - center
    lag_ms     = float(lag_frames * 1000.0 / fps_real)

    # ── Fenêtres glissantes ───────────────────────────────────────────────────
    win_fr  = max(int(ROLL_WINDOW_S * fps_real), 15)
    step_fr = max(int(ROLL_STEP_S   * fps_real), 5)

    rolling_r_vals: List[float] = []
    for i in range(0, n - win_fr + 1, step_fr):
        seg_g = gap_smooth[i:i + win_fr]
        seg_s = s_open[i:i + win_fr]
        if seg_s.std() < MOTION_MIN_MM:
            continue
        r_seg, _ = pearsonr(seg_g, seg_s)
        rolling_r_vals.append(float(r_seg))

    if rolling_r_vals:
        rr = np.array(rolling_r_vals)
        roll_mean = float(rr.mean())
        roll_min  = float(rr.min())
        roll_max  = float(rr.max())
        n_seg     = len(rr)
        bad_seg   = int((rr < R_MIN_OK).sum())
        good_seg  = int((rr >= R_GOOD).sum())
    else:
        roll_mean = roll_min = roll_max = float(global_r)
        n_seg = bad_seg = good_seg = 0

    # ── Anomalies ────────────────────────────────────────────────────────────
    anomaly_mask   = np.abs(residuals) > 3.0 * tol_px
    anomaly_frames = [int(f_indices[i]) for i in np.where(anomaly_mask)[0]]

    # ── Score 0–100 ──────────────────────────────────────────────────────────
    # Composante r global (0–100)
    score_r = float(np.clip((global_r - 0.30) / 0.70, 0.0, 1.0)) * 100.0

    # Composante segments OK (0–100)
    score_seg = (float(n_seg - bad_seg) / n_seg * 100.0) if n_seg > 0 else score_r

    # Composante lag (0–100)
    lag_abs = abs(lag_ms)
    if lag_abs <= LAG_MAX_MS * 0.5:
        score_lag = 100.0
    elif lag_abs <= LAG_MAX_MS:
        score_lag = 100.0 - 50.0 * (lag_abs - LAG_MAX_MS * 0.5) / (LAG_MAX_MS * 0.5)
    else:
        score_lag = max(0.0, 50.0 - 50.0 * (lag_abs - LAG_MAX_MS) / LAG_MAX_MS)

    # Composante confiance segmentation (0–100)
    score_conf = confidence * 100.0

    score = float(np.clip(
        0.45 * score_r + 0.30 * score_seg + 0.15 * score_lag + 0.10 * score_conf,
        0.0, 100.0
    ))

    ok = (
        global_r >= R_MIN_OK
        and bad_seg <= max(1, n_seg // 4)
        and lag_abs <= LAG_MAX_MS
    )

    if global_r >= R_GOOD and bad_seg == 0 and lag_abs <= LAG_MAX_MS * 0.5:
        verdict = "SYNC_PARFAITE"
    elif ok:
        verdict = "SYNC_OK"
    elif global_r >= 0.5:
        verdict = "SYNC_PARTIELLE"
    else:
        verdict = "DESYNC"

    # ── Données par frame ────────────────────────────────────────────────────
    per_frame: List[dict] = []
    if include_per_frame:
        for i in range(n):
            per_frame.append({
                "frame_idx":     int(f_indices[i]),
                "t_ms":          round(float(f_times[i]), 1),
                "sensor_mm":     round(float(s_open[i]), 2),
                "visual_gap":    round(float(gap_raw[i]), 1),
                "visual_smooth": round(float(gap_smooth[i]), 1),
                "predicted_gap": round(float(predicted[i]), 1),
                "residual_px":   round(float(residuals[i]), 1),
                "residual_mm":   round(float(residuals_mm[i]), 2),
                "dark_density":  round(float(dark_dens[i]), 4),
                "in_sync":       bool(in_sync[i]),
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


# ══════════════════════════════════════════════════════════════════════════════
# Analyse d'une session complète (les deux côtés)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_session(
    session_path,
    sides: List[str] = None,
    include_per_frame: bool = True,
) -> dict:
    """
    Analyse les deux côtés (left et right) d'une session.

    Returns :
    {
      "score":      float,     # moyenne des scores des deux côtés
      "ok":         bool,
      "verdict":    str,
      "lag_ms_max": float,
      "n_sides":    int,
      "sides":      {"left": {...}, "right": {...}}
    }
    """
    session_path = Path(session_path)
    if sides is None:
        sides = ["left", "right"]

    results: Dict[str, dict] = {}
    scores:  List[float]     = []
    all_ok   = True
    max_lag  = 0.0

    for side in sides:
        r = analyze_side(session_path, side, include_per_frame=include_per_frame)
        results[side] = _clean_dict(asdict(r))
        if r.success:
            scores.append(r.score)
            if not r.ok:
                all_ok = False
            max_lag = max(max_lag, abs(r.lag_ms))

    global_score = float(np.mean(scores)) if scores else 0.0

    if not scores:
        verdict = "NO_DATA"
    elif all_ok and global_score >= 80:
        verdict = "SYNC_PARFAITE"
    elif all_ok:
        verdict = "SYNC_OK"
    elif global_score >= 50:
        verdict = "SYNC_PARTIELLE"
    else:
        verdict = "DESYNC"

    return {
        "score":       round(global_score, 1),
        "ok":          all_ok and bool(scores),
        "verdict":     verdict,
        "lag_ms_max":  round(max_lag, 1),
        "n_sides":     len(scores),
        "sides":       results,
    }


def _clean_dict(d):
    """Rend un dict JSON-sérialisable."""
    if isinstance(d, dict):
        return {k: _clean_dict(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean_dict(v) for v in d]
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating,)):
        v = float(d)
        return None if not np.isfinite(v) else v
    if isinstance(d, float) and not np.isfinite(d):
        return None
    return d


# Interface pour session_check.py
def check_session(session_path, include_per_frame: bool = False) -> dict:
    return analyze_session(session_path, sides=["left", "right"],
                           include_per_frame=include_per_frame)


# ══════════════════════════════════════════════════════════════════════════════
# Affichage console
# ══════════════════════════════════════════════════════════════════════════════

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

VERDICT_COLOR = {
    "SYNC_PARFAITE":  _GREEN + _BOLD,
    "SYNC_OK":        _GREEN,
    "SYNC_PARTIELLE": _YELLOW,
    "DESYNC":         _RED + _BOLD,
    "NO_DATA":        _RED,
}


def _c(text, color):
    return f"{color}{text}{_RESET}" if sys.stdout.isatty() else text


def _print_side_dict(d: dict, side: str, verbose: bool = False) -> None:
    if not d.get("success"):
        print(f"\n  [{side.upper()}] ERREUR — {d.get('error', '?')}")
        return

    verdict   = d.get("verdict", "?")
    vc        = VERDICT_COLOR.get(verdict, "")
    score     = d.get("score", 0.0) or 0.0
    r_val     = d.get("global_r", 0.0) or 0.0
    lag_ms    = d.get("lag_ms", 0.0) or 0.0
    lag_fr    = d.get("lag_frames", 0) or 0
    r2        = d.get("fit_r2", 0.0) or 0.0
    a         = d.get("fit_a", 0.0) or 0.0
    b         = d.get("fit_b", 0.0) or 0.0
    res_mm    = d.get("residual_std_mm", 0.0) or 0.0
    res_px    = d.get("residual_std_px", 0.0) or 0.0
    tol       = d.get("tol_px", 0.0) or 0.0
    fis       = d.get("frames_in_sync", 0.0) or 0.0
    conf      = d.get("confidence", 0.0) or 0.0
    inv       = d.get("invalid_frames", 0) or 0
    density   = d.get("mean_dark_density", 0.0) or 0.0
    n_seg     = d.get("n_segments", 0) or 0
    bad_seg   = d.get("bad_segments", 0) or 0
    roll_mean = d.get("rolling_r_mean", r_val) or r_val
    roll_min  = d.get("rolling_r_min", r_val) or r_val
    anom      = d.get("anomaly_frames", []) or []

    lag_c  = _GREEN if abs(lag_ms) <= 50 else (_YELLOW if abs(lag_ms) <= 100 else _RED)
    r_c    = _GREEN if r_val >= 0.85 else (_YELLOW if r_val >= 0.70 else _RED)
    conf_c = _GREEN if conf >= 0.85 else (_YELLOW if conf >= 0.60 else _RED)

    print(f"\n  [{side.upper()}]  {_c(verdict, vc)}  score={score:.1f}%")
    print(f"    Corrélation    : {_c(f'r={r_val:.4f}', r_c)}  R²(fit)={r2:.4f}")
    print(f"    Modèle         : gap = {a:.2f}px/mm × opening + {b:.1f}px  "
          f"({a:.2f} px/mm → 1px = {1/a:.3f}mm)" if abs(a) > 0.01 else
          f"    Modèle         : a={a:.2f} b={b:.1f}")
    print(f"    Lag temporel   : {_c(f'{lag_ms:+.1f}ms ({lag_fr:+d} fr)', lag_c)}")
    print(f"    Résidu         : std={res_px:.1f}px ({res_mm:.2f}mm)  tol=±{tol:.0f}px ({tol/max(abs(a),1e-3):.1f}mm)")
    print(f"    Frames en sync : {fis*100:.1f}%  anomalies={len(anom)}")
    print(f"    Segmentation   : {_c(f'confiance={conf:.0%}', conf_c)}  "
          f"densité_dark={density:.4f}  invalides={inv}")

    if n_seg > 0:
        seg_c = _GREEN if bad_seg == 0 else (_YELLOW if bad_seg <= n_seg // 4 else _RED)
        print(f"    Fenêtres (3s)  : {_c(f'{n_seg-bad_seg}/{n_seg} OK', seg_c)}"
              f"  r_moy={roll_mean:.4f}  r_min={roll_min:.4f}")

    if anom and verbose:
        print(f"    Anomalies      : frames {anom[:20]}"
              f"{'…' if len(anom) > 20 else ''}")

    if verbose and d.get("per_frame"):
        pf_all = d["per_frame"]
        step = max(1, len(pf_all) // 25)
        print(f"\n    {'frame':>6}  {'t_ms':>14}  {'sensor':>8}  {'visual':>8}  "
              f"{'pred':>8}  {'resid':>9}  {'dark':>6}  sync")
        for pf in pf_all[::step]:
            ok_s = "✓" if pf["in_sync"] else "✗"
            print(f"    {pf['frame_idx']:>6}  {pf['t_ms']:>14.0f}  "
                  f"{pf['sensor_mm']:>7.1f}mm  {pf['visual_gap']:>7.1f}px  "
                  f"{pf['predicted_gap']:>7.1f}px  {pf['residual_mm']:>+8.2f}mm  "
                  f"{pf['dark_density']:>6.4f}  {ok_s}")


def _print_session_report(result: dict, verbose: bool = False) -> None:
    score   = result["score"]
    verdict = result["verdict"]
    vc      = VERDICT_COLOR.get(verdict, "")
    max_lag = result["lag_ms_max"]
    lag_c   = _GREEN if max_lag <= 50 else (_YELLOW if max_lag <= 100 else _RED)

    print(f"\n{'═'*65}")
    print(f"  Sync gripper par frame  |  "
          f"{_c(verdict, vc)}  score={score:.1f}%  "
          f"lag_max={_c(f'{max_lag:.0f}ms', lag_c)}")
    print(f"{'═'*65}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse par frame de la synchronisation gripper visuel ↔ capteur"
    )
    parser.add_argument("path",           help="Session ou dossier racine (--batch)")
    parser.add_argument("--side",         default="both", choices=["left", "right", "both"])
    parser.add_argument("--batch",        action="store_true", help="Toutes les sessions du dossier")
    parser.add_argument("--json",         action="store_true", help="Sortie JSON brute")
    parser.add_argument("--verbose",      action="store_true", help="Détails par frame")
    parser.add_argument("--no-per-frame", action="store_true", help="Sans données par frame")
    args = parser.parse_args()

    sides      = ["left", "right"] if args.side == "both" else [args.side]
    include_pf = not args.no_per_frame

    root = Path(args.path)
    if not root.exists():
        print(f"Chemin introuvable : {root}", file=sys.stderr)
        sys.exit(1)

    if args.batch or (root.is_dir() and not (root / "metadata.json").exists()):
        sessions = sorted(
            p.parent for p in root.rglob("metadata.json")
            if (p.parent / "videos").exists()
        )
        if not sessions:
            print(f"Aucune session trouvée dans {root}", file=sys.stderr)
            sys.exit(1)
        print(f"Traitement de {len(sessions)} session(s)…\n")
        all_results = []
        for sess in sessions:
            r = analyze_session(sess, sides=sides, include_per_frame=include_pf)
            all_results.append({"session": sess.name, **r})
            if not args.json:
                _print_session_report(r)
                for side, sd in r["sides"].items():
                    _print_side_dict(sd, side, verbose=args.verbose)
        if args.json:
            print(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
    else:
        r = analyze_session(root, sides=sides, include_per_frame=include_pf)
        if args.json:
            print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
        else:
            _print_session_report(r)
            for side, sd in r["sides"].items():
                _print_side_dict(sd, side, verbose=args.verbose)
            print()


if __name__ == "__main__":
    main()
