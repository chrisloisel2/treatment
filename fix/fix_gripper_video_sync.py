#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_gripper_video_sync.py — IA de vérification gripper-vidéo.

Vérifie que les valeurs mesurées par les capteurs gripper (opening_mm) correspondent
aux valeurs visuelles réelles observées dans le flux vidéo.

DEUX NIVEAUX D'ANALYSE :
────────────────────────
NIVEAU 1 — Synchronisation temporelle (JSONL uniquement)
    Compare les timestamps de chaque caméra (JSONL) avec les timestamps du gripper
    correspondant (CSV). Vérifie que les deux flux couvrent les mêmes fenêtres
    temporelles et que les pics d'activité gripper tombent dans des fenêtres de
    capture correctes.
    → Ne nécessite pas les fichiers MP4.

NIVEAU 2 — Correspondance visuelle (requiert MP4 ou frames)
    Extrait le signal visuel d'ouverture du gripper depuis les frames vidéo :
    1. Sample N frames par seconde depuis la caméra gauche / droite.
    2. Pour chaque frame, détecte la région du gripper (ROI automatique via
       analyse de luminosité + détection de contours).
    3. Extrait la valeur visuelle d'ouverture (gap entre les mâchoires du gripper)
       en mesurant la largeur du segment le plus lumineux dans la ROI centrale.
    4. Normalise et aligne le signal visuel avec le signal capteur.
    5. Calcule la corrélation croisée (Pearson + cross-correlation temporelle).
    6. Identifie le décalage temporel optimal et la qualité de correspondance.

SCORE DE CORRESPONDANCE :
    r ≥ 0.85 : EXCELLENT — parfaitement synchronisé
    r ≥ 0.70 : BON
    r ≥ 0.50 : MOYEN — synchronisation approximative
    r < 0.50 : FAIBLE — désynchronisé ou signal visuel non détecté

Usage :
    python -m fix.fix_gripper_video_sync /chemin/session
    python -m fix.fix_gripper_video_sync /chemin/session --level 2 --fps 5
    python -m fix.fix_gripper_video_sync /chemin/root --batch --level 1
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import cv2 as _cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ── Seuils ────────────────────────────────────────────────────────────────────

CORR_EXCELLENT = 0.85
CORR_GOOD      = 0.70
CORR_FAIR      = 0.50

# Sampling vidéo
DEFAULT_SAMPLE_FPS   = 3       # 3 frames/s pour l'analyse visuelle
MAX_LAG_SEARCH_MS    = 1000.0  # fenêtre de décalage cherchée (±1s)
RESAMPLE_MS          = 33.0    # grille de rééchantillonnage (~30fps)

# Détection gripper dans la frame (paramètres ROI)
ROI_TOP_FRAC    = 0.10    # ignorer les 10% du haut (ciel, plafond)
ROI_BOTTOM_FRAC = 0.90    # ignorer les 10% du bas
ROI_HORIZ_FRAC  = 0.15    # bandes verticales latérales à exclure

# Seuil de luminosité pour "voir le gripper" (percentile)
GRIP_BRIGHT_PERCENTILE = 85

# ── Paramètres analyse position fermée (convolution) ─────────────────────────

CLOSED_FRAC             = 0.25   # seuil adaptatif : fermé si opening < 25% de la plage
CONV_SAMPLE_FPS         = 10     # fps vidéo pour l'analyse position fermée
CONV_RESAMPLE_MS        = 50.0   # résolution grille de convolution (ms)
CONV_MAX_LAG_MS         = 2000.0 # fenêtre de recherche décalage (±2s)
CONV_MIN_CLOSED_FRAMES  = 10     # événements "fermé" minimum requis
OFFSET_SIGNIFICANT_MS   = 50.0   # décalage > 50ms → significatif (à corriger)
GRIPPER_SYNC_MARKER_KEY = "gripper_closed_sync_applied"


# ── Structures ────────────────────────────────────────────────────────────────

@dataclass
class GripperSyncResult:
    side: str                         # "left" | "right"
    level: int                        # 1 ou 2
    n_sensor_samples: int
    n_camera_frames: int
    # Niveau 1
    temporal_overlap_ms: float
    temporal_delta_start_ms: float    # t_start_camera - t_start_gripper
    temporal_aligned: bool
    # Niveau 2 (None si level=1)
    n_visual_frames_analysed: Optional[int] = None
    pearson_r: Optional[float]         = None
    optimal_lag_ms: Optional[float]    = None
    visual_signal_quality: Optional[float] = None  # fraction de frames où gripper détecté
    status: str = "unknown"
    issues: list[str] = field(default_factory=list)


@dataclass
class GripperVideoSyncReport:
    session: str
    status: str                        # "ok"|"warn"|"error"|"no_data"
    level: int
    results: list[GripperSyncResult]
    issues: list[str]
    summary: dict
    closed_sync: list["ClosedPositionSyncResult"] = field(default_factory=list)


@dataclass
class ClosedPositionSyncResult:
    """
    Résultat de l'analyse de synchronisation par convolution sur la position fermée.

    Convention offset_ms :
        offset_ms > 0  — capteur EN AVANCE sur la vidéo de offset_ms ms
                         (le capteur signale "fermé" avant que la vidéo le montre)
        offset_ms < 0  — capteur EN RETARD sur la vidéo
        offset_ms ≈ 0  — flux synchronisés

    Correction à appliquer sur les timestamps capteur (CSV) :
        timestamp_corrigé_ns = timestamp_ns + offset_ms × 1e6
        (offset > 0 → retarde le capteur ; offset < 0 → l'avance)
    """
    side: str
    n_closed_sensor: int          = 0
    n_closed_visual: int          = 0
    peak_correlation: float       = 0.0   # corrélation normalisée au pic [0-1]
    offset_ms: float              = 0.0   # décalage détecté (ms)
    snr: float                    = 0.0   # rapport signal/bruit du pic (>2 = fiable)
    status: str                   = "unknown"
    issues: list[str]             = field(default_factory=list)


# ── Chargement données ────────────────────────────────────────────────────────

def _load_gripper_df(path: Path) -> Optional[pd.DataFrame]:
    if not _PANDAS or not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        required = {"timestamp_ns", "opening_mm"}
        if not required.issubset(df.columns):
            # Essayer t_ms_corrected_ns
            if "t_ms_corrected_ns" in df.columns and "opening_mm" in df.columns:
                df["timestamp_ns"] = df["t_ms_corrected_ns"]
            else:
                return None
        df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce")
        df["opening_mm"]   = pd.to_numeric(df["opening_mm"],   errors="coerce")
        df = df.dropna(subset=["timestamp_ns", "opening_mm"])
        df = df.sort_values("timestamp_ns").reset_index(drop=True)
        return df if len(df) > 10 else None
    except Exception:
        return None


def _load_jsonl_times(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    times = []
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    times.append(float(json.loads(line)["capture_time"]))
                except Exception:
                    continue
    except Exception:
        return None
    return np.array(times, dtype=np.float64) if len(times) > 5 else None


def _resample_to_grid(t: np.ndarray, v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, t, v).astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    return (x - np.mean(x)) / (s + 1e-8)


# ── Convolution position fermée ───────────────────────────────────────────────

def _compute_closed_offset(
    sensor_t_ms: np.ndarray,
    sensor_opening: np.ndarray,
    visual_t_ms: np.ndarray,
    visual_signal: np.ndarray,
    resample_ms: float = CONV_RESAMPLE_MS,
    max_lag_ms: float  = CONV_MAX_LAG_MS,
) -> "tuple[float, float, float, int, int]":
    """
    Trouve l'offset temporel entre flux capteur et flux vidéo par convolution
    (corrélation croisée) des signaux binaires « pince fermée ».

    Algorithme :
      1. Seuil adaptatif (percentile 25/75) sur chaque signal → binaire fermé/ouvert.
      2. Rééchantillonnage sur une grille commune (resample_ms).
      3. Corrélation croisée : corr = np.correlate(visual_binary, sensor_binary, 'full').
         Le pic à lag > 0 indique que le capteur est EN AVANCE sur la vidéo.
      4. Interpolation parabolique autour du pic pour précision sub-grille.

    Convention retournée :
        offset_ms > 0  → capteur en avance sur la vidéo de offset_ms ms
        offset_ms < 0  → capteur en retard

    Correction : timestamp_corrigé_ns = timestamp_ns + offset_ms × 1e6

    Returns:
        (offset_ms, peak_corr_norm, snr, n_closed_sensor, n_closed_visual)
    """
    # ── 1. Seuils adaptatifs ──────────────────────────────────────────────────
    g_q25 = float(np.percentile(sensor_opening, 25))
    g_q75 = float(np.percentile(sensor_opening, 75))
    if g_q75 - g_q25 < 3.0:
        return 0.0, 0.0, 0.0, 0, 0

    s_thresh = g_q25 + (g_q75 - g_q25) * CLOSED_FRAC
    sensor_bin = (sensor_opening <= s_thresh).astype(np.float32)

    v_q25 = float(np.percentile(visual_signal, 25))
    v_q75 = float(np.percentile(visual_signal, 75))
    if v_q75 - v_q25 < 0.03:
        return 0.0, 0.0, 0.0, 0, 0

    v_thresh = v_q25 + (v_q75 - v_q25) * CLOSED_FRAC
    visual_bin = (visual_signal <= v_thresh).astype(np.float32)

    # ── 2. Grille commune ─────────────────────────────────────────────────────
    t0 = max(float(sensor_t_ms[0]), float(visual_t_ms[0]))
    t1 = min(float(sensor_t_ms[-1]), float(visual_t_ms[-1]))
    if t1 - t0 < 2000.0:
        return 0.0, 0.0, 0.0, 0, 0

    grid = np.arange(t0, t1, resample_ms)
    s_grid = (np.interp(grid, sensor_t_ms, sensor_bin.astype(float)) >= 0.5).astype(np.float32)
    v_grid = (np.interp(grid, visual_t_ms, visual_bin.astype(float)) >= 0.5).astype(np.float32)

    n_s = int(s_grid.sum())
    n_v = int(v_grid.sum())

    if n_s < CONV_MIN_CLOSED_FRAMES or n_v < CONV_MIN_CLOSED_FRAMES:
        return 0.0, 0.0, 0.0, n_s, n_v

    # ── 3. Corrélation croisée ────────────────────────────────────────────────
    # corr = np.correlate(visual, sensor, 'full')
    # corr[k + center] = Σ_n visual[n+k] · sensor[n]
    # Pic à k > 0 → capteur en avance (sensor leads video) → offset_ms > 0
    # center = len(sensor) - 1
    corr   = np.correlate(v_grid, s_grid, mode="full")
    center = len(s_grid) - 1

    max_lag_s = int(max_lag_ms / resample_ms)
    lo = max(0, center - max_lag_s)
    hi = min(len(corr), center + max_lag_s + 1)
    window     = corr[lo:hi]
    lag_idxs   = np.arange(lo, hi) - center   # lag en samples

    if len(window) == 0:
        return 0.0, 0.0, 0.0, n_s, n_v

    pk_local = int(np.argmax(window))
    pk_lag_s = int(lag_idxs[pk_local])
    pk_val   = float(window[pk_local])

    # Normaliser par le max théorique
    max_possible = float(min(s_grid.sum(), v_grid.sum()))
    peak_norm = pk_val / (max_possible + 1e-8)

    # SNR : pic / bruit moyen (hors ±5 samples autour du pic)
    ex_lo = max(0, pk_local - 5)
    ex_hi = min(len(window), pk_local + 6)
    rest  = np.concatenate([window[:ex_lo], window[ex_hi:]])
    noise = float(np.mean(np.abs(rest))) if len(rest) > 0 else 1e-8
    snr   = pk_val / (noise + 1e-8)

    # ── 4. Interpolation parabolique (sub-sample) ─────────────────────────────
    if 0 < pk_local < len(window) - 1:
        y_m = window[pk_local - 1]
        y_0 = window[pk_local]
        y_p = window[pk_local + 1]
        denom = y_m - 2.0 * y_0 + y_p
        delta = 0.5 * (y_m - y_p) / (denom - 1e-10) if abs(denom) > 1e-10 else 0.0
        pk_lag_frac = pk_lag_s + float(np.clip(delta, -0.5, 0.5))
    else:
        pk_lag_frac = float(pk_lag_s)

    offset_ms = pk_lag_frac * resample_ms
    return float(offset_ms), float(peak_norm), float(snr), n_s, n_v


def _analyse_closed_position_sync(
    side: str,
    gripper_df: "pd.DataFrame",
    cam_times_ms: np.ndarray,
    session_path: Path,
    sample_fps: int = CONV_SAMPLE_FPS,
) -> "ClosedPositionSyncResult":
    """
    Analyse la synchronisation gripper-vidéo en ciblant la position fermée.

    Démarche :
      1. Signal capteur opening_mm → binarisé sur "pince fermée" (seuil adaptatif).
      2. Signal visuel extrait frame-par-frame depuis la vidéo → binarisé.
      3. Convolution croisée des deux signaux binaires → offset précis en ms.

    Avantages par rapport à la corrélation sur signal continu :
      - La transition ouvert→fermé est un front net → peu sensible au bruit.
      - Aucune calibration amplitude px→mm requise.
      - Fonctionne même si la qualité vidéo est faible.
    """
    result = ClosedPositionSyncResult(side=side)

    if not _CV2:
        result.status = "no_data"
        result.issues.append("OpenCV non disponible (pip install opencv-python)")
        return result

    video_candidates = [
        session_path / "videos" / f"{side}.mp4",
        session_path / f"{side}.mp4",
        session_path / "videos" / f"cam_{side}.mp4",
    ]
    video_path = next((p for p in video_candidates if p.exists()), None)
    if video_path is None:
        result.status = "no_data"
        result.issues.append(f"Vidéo introuvable pour le côté '{side}'")
        return result

    # ── Extraction visuelle ───────────────────────────────────────────────────
    vis_t_rel, vis_sig = _extract_gripper_brightness_signal(video_path, sample_fps)
    if len(vis_t_rel) < 20:
        result.status = "no_data"
        result.issues.append(f"Extraction vidéo insuffisante ({len(vis_t_rel)} frames)")
        return result

    # Timestamps relatifs → absolus via premier timestamp JSONL
    vis_t_abs = vis_t_rel + float(cam_times_ms[0])

    # ── Signal capteur ────────────────────────────────────────────────────────
    sensor_t_ms = gripper_df["timestamp_ns"].to_numpy() / 1e6
    sensor_v_mm = gripper_df["opening_mm"].to_numpy().astype(np.float32)

    # ── Convolution ───────────────────────────────────────────────────────────
    offset_ms, peak_corr, snr, n_s, n_v = _compute_closed_offset(
        sensor_t_ms, sensor_v_mm, vis_t_abs, vis_sig,
    )

    result.n_closed_sensor  = n_s
    result.n_closed_visual  = n_v
    result.peak_correlation = round(peak_corr, 4)
    result.offset_ms        = round(offset_ms, 1)
    result.snr              = round(snr, 2)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if n_s < CONV_MIN_CLOSED_FRAMES:
        result.status = "warn"
        result.issues.append(
            f"Peu d'événements 'pince fermée' capteur ({n_s} frames, min={CONV_MIN_CLOSED_FRAMES})"
        )
        return result

    if n_v < CONV_MIN_CLOSED_FRAMES:
        result.status = "warn"
        result.issues.append(
            f"Peu d'événements 'pince fermée' visuels ({n_v} frames, min={CONV_MIN_CLOSED_FRAMES})"
        )
        return result

    if snr < 2.0:
        result.status = "warn"
        result.issues.append(f"SNR faible ({snr:.1f}) — pic de corrélation peu fiable (min=2.0)")
        return result

    if abs(offset_ms) > OFFSET_SIGNIFICANT_MS:
        direction = "capteur en avance" if offset_ms > 0 else "capteur en retard"
        result.status = "warn"
        result.issues.append(
            f"Décalage significatif : {offset_ms:+.0f}ms ({direction})"
        )
    else:
        result.status = "ok"

    return result


# ── Fix : correction du décalage capteur/vidéo ────────────────────────────────

def fix_gripper_closed_offset(
    session_path: Path,
    dry_run: bool = False,
    sample_fps: int = CONV_SAMPLE_FPS,
    min_snr: float = 2.0,
    min_offset_ms: float = OFFSET_SIGNIFICANT_MS,
    force: bool = False,
) -> dict:
    """
    Détecte et corrige le décalage entre le flux capteur gripper et la vidéo
    en se basant sur la position fermée (convolution binaire).

    Méthode :
        Convolution des signaux binaires "pince fermée" → offset précis en ms.
        Correction appliquée sur les timestamps du CSV gripper.

    Convention :
        offset_ms > 0 (capteur en avance) → on retarde les timestamps capteur
        timestamp_corrigé_ns = timestamp_ns + offset_ms × 1e6

    Args:
        session_path: chemin de la session
        dry_run:      calcule mais n'écrit rien
        sample_fps:   fps d'extraction vidéo
        min_snr:      SNR minimum pour appliquer la correction
        min_offset_ms: décalage minimum pour corriger
        force:        réappliquer même si le marqueur est déjà présent

    Returns:
        dict { session, status, corrections_ms, details }
    """
    name = session_path.name

    # Vérifier le marqueur (éviter double-application)
    meta_path = session_path / "metadata.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if meta.get(GRIPPER_SYNC_MARKER_KEY) and not force:
        return {"session": name, "status": "already_applied", "corrections_ms": {}, "details": {}}

    corrections: dict = {}
    details:     dict = {}

    for side in ("left", "right"):
        gripper_path = session_path / f"gripper_{side}_data.csv"
        jsonl_path   = session_path / "videos" / f"{side}.jsonl"

        gripper_df = _load_gripper_df(gripper_path)
        cam_times  = _load_jsonl_times(jsonl_path)

        if gripper_df is None or cam_times is None:
            continue

        result = _analyse_closed_position_sync(
            side, gripper_df, cam_times, session_path, sample_fps
        )

        details[side] = {
            "offset_ms":       result.offset_ms,
            "peak_corr":       result.peak_correlation,
            "snr":             result.snr,
            "n_closed_sensor": result.n_closed_sensor,
            "n_closed_visual": result.n_closed_visual,
            "status":          result.status,
            "issues":          result.issues,
        }

        if result.snr < min_snr:
            continue
        if abs(result.offset_ms) < min_offset_ms:
            continue

        corrections[side] = result.offset_ms

        if not dry_run:
            df = gripper_df.copy()
            df["timestamp_ns"] = (
                df["timestamp_ns"].to_numpy(dtype=np.float64)
                + result.offset_ms * 1e6
            ).astype(np.int64)
            df.to_csv(gripper_path, index=False)

    if corrections and not dry_run:
        meta[GRIPPER_SYNC_MARKER_KEY]           = True
        meta["gripper_closed_sync_offsets_ms"]  = {k: round(v, 1) for k, v in corrections.items()}
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    overall = ("corrected"  if corrections and not dry_run else
               "dry-run"    if dry_run and corrections else
               "ok")

    return {
        "session":        name,
        "status":         overall,
        "corrections_ms": {k: round(v, 1) for k, v in corrections.items()},
        "details":        details,
    }


# ── Niveau 1 : synchronisation temporelle ────────────────────────────────────

def _analyse_level1(
    side: str,
    gripper_df: pd.DataFrame,
    cam_times_ms: np.ndarray,
) -> GripperSyncResult:
    """
    Niveau 1 : vérification temporelle pure (timestamps).
    """
    g_t_ms = gripper_df["timestamp_ns"].to_numpy() / 1e6
    g_start, g_end = float(g_t_ms[0]), float(g_t_ms[-1])
    c_start, c_end = float(cam_times_ms[0]), float(cam_times_ms[-1])

    delta_start = c_start - g_start    # ms
    overlap_s   = max(g_start, c_start)
    overlap_e   = min(g_end,   c_end)
    overlap_ms  = max(0.0, overlap_e - overlap_s)
    temporal_ok = abs(delta_start) < 2000.0 and overlap_ms > 5000.0

    issues: list[str] = []
    status = "ok"

    if abs(delta_start) > 2000.0:
        issues.append(f"Δt_start={delta_start:.0f}ms > 2000ms → désynchronisé")
        status = "error"
    elif abs(delta_start) > 500.0:
        issues.append(f"Δt_start={delta_start:.0f}ms > 500ms → décalage modéré")
        status = "warn"

    if overlap_ms < 3000.0:
        issues.append(f"chevauchement={overlap_ms:.0f}ms insuffisant")
        status = "error"

    return GripperSyncResult(
        side=side,
        level=1,
        n_sensor_samples=len(gripper_df),
        n_camera_frames=len(cam_times_ms),
        temporal_overlap_ms=round(overlap_ms, 1),
        temporal_delta_start_ms=round(delta_start, 1),
        temporal_aligned=temporal_ok,
        status=status,
        issues=issues,
    )


# ── Niveau 2 : correspondance visuelle ───────────────────────────────────────

def _extract_gripper_brightness_signal(video_path: Path, sample_fps: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait le signal de luminosité de la zone gripper depuis la vidéo.

    Retourne (timestamps_ms, signal) où signal est la valeur d'ouverture
    visuelle normalisée (0=fermé, 1=ouvert au max).

    Algorithme :
      1. Charger la frame en niveaux de gris.
      2. Découper la ROI centrale (évite les bords).
      3. Dans la ROI : trouver la ligne horizontale avec le plus grand
         contraste vertical (frontière gripper/fond) → c'est l'emplacement
         des mâchoires du gripper.
      4. Mesurer la largeur de la zone lumineuse entre les mâchoires
         (gap = ouverture du gripper).
    """
    if not _CV2:
        return np.array([]), np.array([])
    if not video_path.exists():
        return np.array([]), np.array([])

    cap = _cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.array([]), np.array([])

    fps_vid    = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps_vid / sample_fps))

    timestamps_ms: list[float] = []
    signals:       list[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            ts_ms = cap.get(_cv2.CAP_PROP_POS_MSEC)
            signal = _extract_opening_from_frame(frame)
            timestamps_ms.append(float(ts_ms))
            signals.append(signal)

        frame_idx += 1

    cap.release()
    return np.array(timestamps_ms, dtype=np.float64), np.array(signals, dtype=np.float32)


def _extract_opening_from_frame(frame: "np.ndarray") -> float:
    """
    Extrait la valeur d'ouverture du gripper depuis une frame BGR.

    Méthode : détection par luminosité dans la ROI centrale.
    Le gap entre les mâchoires du gripper apparaît comme une bande
    lumineuse (fond blanc/clair) entre deux zones sombres (les mâchoires).

    Retourne une valeur 0–1 (0=gripper fermé, 1=max ouverture visible).
    """
    h, w = frame.shape[:2]

    # ROI : éviter les bords
    y0 = int(h * ROI_TOP_FRAC)
    y1 = int(h * ROI_BOTTOM_FRAC)
    x0 = int(w * ROI_HORIZ_FRAC)
    x1 = int(w * (1 - ROI_HORIZ_FRAC))
    roi = frame[y0:y1, x0:x1]

    gray = _cv2.cvtColor(roi, _cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    # Colonne centrale (zone d'ouverture du gripper)
    cx0 = int(gray.shape[1] * 0.35)
    cx1 = int(gray.shape[1] * 0.65)
    center_strip = gray[:, cx0:cx1]

    # Profil vertical moyen (1D)
    profile = center_strip.mean(axis=1).astype(np.float32)

    # Lissage léger
    k = 5
    if len(profile) > k:
        profile = np.convolve(profile, np.ones(k)/k, mode="same")

    # Detecter la zone lumineuse (gap entre mâchoires)
    bright_thresh = float(np.percentile(profile, GRIP_BRIGHT_PERCENTILE))
    bright_mask = profile >= bright_thresh

    # Trouver le run le plus long de pixels lumineux (= gap ouverture)
    max_run = 0
    cur_run = 0
    for b in bright_mask:
        if b:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    # Normaliser par la hauteur de la ROI
    opening_frac = max_run / (len(profile) + 1e-6)
    return float(np.clip(opening_frac, 0.0, 1.0))


def _analyse_level2(
    side: str,
    gripper_df: pd.DataFrame,
    cam_times_ms: np.ndarray,
    session_path: Path,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
) -> GripperSyncResult:
    """
    Niveau 2 : correspondance visuelle gripper-capteur.
    Nécessite les fichiers MP4.
    """
    # D'abord le niveau 1
    base = _analyse_level1(side, gripper_df, cam_times_ms)
    base.level = 2

    # Chercher le fichier vidéo
    video_candidates = [
        session_path / "videos" / f"{side}.mp4",
        session_path / f"{side}.mp4",
        session_path / "videos" / f"cam_{side}.mp4",
    ]
    video_path = next((p for p in video_candidates if p.exists()), None)

    if not _CV2:
        base.issues.append("OpenCV non disponible (pip install opencv-python)")
        base.status = "warn" if base.status == "ok" else base.status
        return base

    if video_path is None:
        base.issues.append(
            f"Fichier vidéo introuvable : cherché dans {[str(c) for c in video_candidates]}"
        )
        base.status = "warn" if base.status == "ok" else base.status
        return base

    # ── Extraire le signal visuel ─────────────────────────────────────────────
    vis_t, vis_sig = _extract_gripper_brightness_signal(video_path, sample_fps)
    if len(vis_t) < 10:
        base.issues.append("Extraction visuelle échouée (frames insuffisantes)")
        return base

    base.n_visual_frames_analysed = len(vis_t)

    # Convertir les timestamps vidéo relatifs en absolus (via JSONL)
    # vis_t est en ms depuis le début de la vidéo ; aligner avec cam_times_ms
    vis_t_abs = vis_t + cam_times_ms[0]

    # ── Rééchantillonner les deux signaux sur une grille commune ──────────────
    g_t_ms  = gripper_df["timestamp_ns"].to_numpy() / 1e6
    g_v     = gripper_df["opening_mm"].to_numpy().astype(np.float32)

    t0 = max(float(g_t_ms[0]), float(vis_t_abs[0]))
    t1 = min(float(g_t_ms[-1]), float(vis_t_abs[-1]))
    if t1 - t0 < 2000.0:
        base.issues.append("Chevauchement trop court pour corrélation (<2s)")
        return base

    grid = np.arange(t0, t1, RESAMPLE_MS)
    sig_sensor = _resample_to_grid(g_t_ms, g_v, grid)
    sig_visual = _resample_to_grid(vis_t_abs, vis_sig, grid)

    # Qualité signal visuel (variance non nulle = gripper détecté)
    visual_quality = float(np.std(sig_visual) / (np.std(sig_sensor) + 1e-8))
    visual_quality = float(np.clip(visual_quality, 0.0, 2.0))
    base.visual_signal_quality = round(visual_quality, 3)

    if np.std(sig_visual) < 1e-4:
        base.issues.append("Signal visuel constant (gripper non détecté dans la vidéo)")
        base.status = "warn"
        return base

    # ── Corrélation croisée décalée ────────────────────────────────────────────
    a = _zscore(sig_sensor)
    b = _zscore(sig_visual)

    # Chercher le meilleur lag en ±MAX_LAG_SEARCH_MS
    lag_samples = int(MAX_LAG_SEARCH_MS / RESAMPLE_MS)
    n = len(a)
    best_r   = -2.0
    best_lag = 0

    for lag in range(-lag_samples, lag_samples + 1):
        if lag >= 0:
            sa, sb = a[lag:], b[:n-lag] if lag > 0 else b
        else:
            sa, sb = a[:n+lag], b[-lag:]
        m = min(len(sa), len(sb))
        if m < 20:
            continue
        r = float(np.mean(sa[:m] * sb[:m]))
        if r > best_r:
            best_r   = r
            best_lag = lag

    base.pearson_r    = round(float(best_r), 4)
    base.optimal_lag_ms = round(float(best_lag * RESAMPLE_MS), 1)

    # ── Verdict ───────────────────────────────────────────────────────────────
    r = best_r
    lag_abs = abs(base.optimal_lag_ms)

    if r >= CORR_EXCELLENT and lag_abs < 100.0:
        status = "ok"
    elif r >= CORR_GOOD and lag_abs < 300.0:
        status = "ok"
        if base.status == "ok":
            pass  # déjà ok
    elif r >= CORR_FAIR:
        status = "warn"
        base.issues.append(f"Corrélation modérée r={r:.2f} (seuil bon={CORR_GOOD})")
    else:
        status = "error"
        base.issues.append(f"Corrélation faible r={r:.2f} — désynchronisé ou gripper invisible")

    if lag_abs > 100.0:
        base.issues.append(f"Décalage optimal={base.optimal_lag_ms:.0f}ms (attendu ≈ 0ms)")
        if status == "ok":
            status = "warn"

    # Fusionner status avec niveau 1
    severity = {"ok": 0, "warn": 1, "error": 2}
    worst = max(severity.get(base.status, 0), severity.get(status, 0))
    base.status = ["ok", "warn", "error"][worst]

    return base


# ── Point d'entrée principal ──────────────────────────────────────────────────

def analyse_gripper_video_sync(
    session_path: Path,
    level: int = 1,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
) -> GripperVideoSyncReport:
    """
    Analyse complète gripper ↔ vidéo pour une session.

    level=1 : timestamps uniquement (rapide, sans vidéo)
    level=2 : corrélation visuelle continue + analyse convolution position fermée
              (nécessite MP4 + OpenCV)
    """
    name = session_path.name
    results:     list[GripperSyncResult]         = []
    closed_sync: list[ClosedPositionSyncResult]  = []
    issues:      list[str]                       = []

    for side in ("left", "right"):
        gripper_df = _load_gripper_df(session_path / f"gripper_{side}_data.csv")
        cam_times  = _load_jsonl_times(session_path / "videos" / f"{side}.jsonl")

        if gripper_df is None and cam_times is None:
            continue

        if gripper_df is None:
            issues.append(f"[gripper_{side}] CSV absent ou illisible")
            continue
        if cam_times is None:
            issues.append(f"[cam_{side}] JSONL absent ou illisible")
            continue

        if level == 1:
            res = _analyse_level1(side, gripper_df, cam_times)
        else:
            res = _analyse_level2(side, gripper_df, cam_times, session_path, sample_fps)

            # ── Analyse convolution position fermée (niveau 2 uniquement) ─────
            cl = _analyse_closed_position_sync(
                side, gripper_df, cam_times, session_path,
                sample_fps=CONV_SAMPLE_FPS,
            )
            closed_sync.append(cl)
            if cl.status in ("warn", "error"):
                for iss in cl.issues:
                    issues.append(f"[{side}/conv] {iss}")

        results.append(res)
        issues.extend([f"[{side}] {i}" for i in res.issues])

    if not results:
        return GripperVideoSyncReport(
            session=name, status="no_data", level=level,
            results=[], issues=["Aucune donnée gripper + caméra trouvée"],
            summary={}, closed_sync=[],
        )

    # Statut global (tient compte aussi des résultats de convolution)
    all_statuses = [r.status for r in results] + [c.status for c in closed_sync
                                                   if c.status != "unknown"]
    if "error" in all_statuses:
        global_status = "error"
    elif "warn" in all_statuses:
        global_status = "warn"
    else:
        global_status = "ok"

    summary: dict = {
        "sides_analysed":   [r.side for r in results],
        "global_status":    global_status,
        "level":            level,
    }
    for r in results:
        summary[f"{r.side}_temporal_delta_ms"] = r.temporal_delta_start_ms
        summary[f"{r.side}_overlap_ms"]        = r.temporal_overlap_ms
        if r.pearson_r is not None:
            summary[f"{r.side}_pearson_r"]    = r.pearson_r
            summary[f"{r.side}_lag_ms"]       = r.optimal_lag_ms
    for c in closed_sync:
        summary[f"{c.side}_closed_offset_ms"] = c.offset_ms
        summary[f"{c.side}_closed_snr"]       = c.snr

    return GripperVideoSyncReport(
        session=name,
        status=global_status,
        level=level,
        results=results,
        issues=issues,
        summary=summary,
        closed_sync=closed_sync,
    )


# ── Affichage ─────────────────────────────────────────────────────────────────

_COLORS = {"ok": "\033[92m", "warn": "\033[93m", "error": "\033[91m",
           "reset": "\033[0m", "bold": "\033[1m"}
_USE_COLOR = sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    return f"{_COLORS[color]}{text}{_COLORS['reset']}" if _USE_COLOR else text


def print_report(r: GripperVideoSyncReport) -> None:
    status_color = r.status if r.status in ("ok", "warn", "error") else "ok"
    print(f"\n{_c(r.status.upper(), status_color)} — {_c(r.session, 'bold')}  "
          f"[niveau {r.level}]")

    for res in r.results:
        color = res.status if res.status in ("ok", "warn", "error") else "ok"
        parts = [
            f"gripper_{res.side}",
            f"n_sensor={res.n_sensor_samples}",
            f"n_frames={res.n_camera_frames}",
            f"Δstart={res.temporal_delta_start_ms:+.0f}ms",
            f"overlap={res.temporal_overlap_ms/1000:.1f}s",
        ]
        if res.pearson_r is not None:
            r_grade = ("EXCELLENT" if res.pearson_r >= CORR_EXCELLENT
                       else "BON" if res.pearson_r >= CORR_GOOD
                       else "MOYEN" if res.pearson_r >= CORR_FAIR
                       else "FAIBLE")
            parts.append(f"r={res.pearson_r:.3f}({r_grade})")
            parts.append(f"lag={res.optimal_lag_ms:+.0f}ms")
        print(f"  {_c(f'[{res.status.upper():5s}]', color)}  {'  '.join(parts)}")
        for issue in res.issues:
            print(f"           ↳ {_c(issue, 'warn')}")

    # Résultats convolution position fermée
    for c in r.closed_sync:
        color = c.status if c.status in ("ok", "warn", "error") else "ok"
        parts = [
            f"  conv_{c.side}",
            f"fermé_capteur={c.n_closed_sensor}",
            f"fermé_visuel={c.n_closed_visual}",
            f"offset={c.offset_ms:+.0f}ms",
            f"SNR={c.snr:.1f}",
            f"r={c.peak_correlation:.3f}",
        ]
        print(f"  {_c(f'[{c.status.upper():5s}]', color)}  {'  '.join(parts)}")
        for issue in c.issues:
            print(f"           ↳ {_c(issue, 'warn')}")

    if r.issues:
        for issue in r.issues:
            print(f"  ⚠ {issue}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vérifie la synchronisation gripper-vidéo d'une session."
    )
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--level", type=int, default=1, choices=[1, 2],
                        help="Niveau d'analyse : 1=timestamps, 2=visuel+corrélation")
    parser.add_argument("--fps",   type=int, default=DEFAULT_SAMPLE_FPS,
                        help=f"Frames/s pour l'extraction visuelle (défaut={DEFAULT_SAMPLE_FPS})")
    parser.add_argument("--batch",   action="store_true")
    parser.add_argument("--json",    action="store_true")
    parser.add_argument("--fix",     action="store_true",
                        help="Applique la correction de décalage sur les CSV gripper (convolution)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Avec --fix : calcule mais n'écrit pas")
    args = parser.parse_args()

    if args.level == 2 and not _CV2:
        print("⚠ OpenCV non installé. Installez avec : pip install opencv-python")
        print("  Fallback vers niveau 1.")
        args.level = 1

    sessions: list[Path] = []
    for p in args.sessions:
        p = p.resolve()
        if args.batch and p.is_dir():
            sessions.extend(
                m.parent for m in p.rglob("metadata.json")
                if any((m.parent / f"gripper_{s}_data.csv").exists()
                       for s in ("left", "right"))
            )
        else:
            sessions.append(p)

    results  = []
    fix_rpts = []
    for s in sessions:
        if not s.is_dir():
            continue
        r = analyse_gripper_video_sync(s, level=args.level, sample_fps=args.fps)
        results.append(r)
        if not args.json:
            print_report(r)

        if args.fix:
            if not _CV2:
                if not args.json:
                    print("  [--fix ignoré : OpenCV non disponible]")
            else:
                fix_r = fix_gripper_closed_offset(
                    s, dry_run=args.dry_run, sample_fps=CONV_SAMPLE_FPS
                )
                fix_rpts.append(fix_r)
                if not args.json:
                    tag = "(dry-run)" if args.dry_run else ""
                    if fix_r["corrections_ms"]:
                        print(f"  [FIX {tag}] {fix_r['status']} — "
                              f"corrections={fix_r['corrections_ms']}")
                    else:
                        print(f"  [FIX {tag}] aucune correction nécessaire")

    if args.json:
        out = [asdict(r) for r in results]
        if fix_rpts:
            for i, f in enumerate(fix_rpts):
                if i < len(out):
                    out[i]["fix"] = f
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))

    if not args.json and results:
        n_ok   = sum(1 for r in results if r.status == "ok")
        n_warn = sum(1 for r in results if r.status == "warn")
        n_err  = sum(1 for r in results if r.status == "error")
        print(f"\n{'─'*60}")
        print(f"Total {len(results)} : "
              f"{_c(f'✓ {n_ok} ok', 'ok')}  "
              f"{_c(f'⚠ {n_warn} warn', 'warn')}  "
              f"{_c(f'✗ {n_err} error', 'error')}")


if __name__ == "__main__":
    main()
