#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gripper_video_vs_sensor.py

But:
- Extraire un signal d'ouverture depuis une vidéo de pince
- Aligner ce signal avec une série temporelle capteur
- Comparer les deux et générer des sorties exploitables

Sorties:
- video_signal.csv
- aligned_comparison.csv
- alignment_report.txt
- opening_vs_time.png
- comparison_aligned.png
- annotated_output.mp4 (optionnel)

Usage exemple:
python gripper_video_vs_sensor.py \
    --video input.mp4 \
    --sensor_csv sensor.csv \
    --sensor_time_col time \
    --sensor_open_col opening_mm \
    --output_dir results \
    --roi 0.15 0.55 0.85 1.00 \
    --px_to_mm_slope 0.120 \
    --px_to_mm_intercept 0.0 \
    --annotated_video

Si tu n'as pas encore la calibration mm:
- mets --px_to_mm_slope 1.0
- tu auras le signal en "pseudo-mm" = pixels
- ensuite tu calibres avec 2/5/10/30 mm

Format capteur attendu:
- CSV avec au moins:
  - une colonne temps (secondes ou ms)
  - une colonne ouverture

Important:
- Le script suppose caméra fixe
- ROI à ajuster une fois
- Les mors sont sombres et visibles dans la ROI
"""

import os
import sys
import cv2
import math
import json
import argparse
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks, correlate
from scipy.interpolate import interp1d


# =========================
# Helpers généraux
# =========================

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def robust_read_csv(path: str) -> pd.DataFrame:
    last_exc = None
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Impossible de lire le CSV: {path}. Dernière erreur: {last_exc}")


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad < 1e-12:
        std = np.nanstd(x)
        if std < 1e-12:
            return np.zeros_like(x)
        return (x - np.nanmean(x)) / std
    return (x - med) / (1.4826 * mad)


def fill_nan_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    idx = np.arange(n)
    valid = np.isfinite(y)
    if valid.sum() == 0:
        raise RuntimeError("Signal entièrement NaN.")
    if valid.sum() == 1:
        return np.full_like(y, y[valid][0], dtype=float)
    return np.interp(idx, idx[valid], y[valid])


def rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=k, center=True, min_periods=1).median().to_numpy()


def smooth_signal(x: np.ndarray, fps: float) -> np.ndarray:
    x = fill_nan_1d(x)
    x = rolling_median(x, max(3, int(round(fps * 0.10)) | 1))
    win = max(5, int(round(fps * 0.25)) | 1)
    win = min(win, len(x) - 1 if len(x) % 2 == 0 else len(x))
    if win < 5:
        return x
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return x
    try:
        return savgol_filter(x, window_length=win, polyorder=2, mode="interp")
    except Exception:
        return x


def compute_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "max_abs": np.nan, "corr": np.nan}
    e = a[m] - b[m]
    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e ** 2))
    max_abs = np.max(np.abs(e))
    corr = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 2 else np.nan
    return {"mae": mae, "rmse": rmse, "max_abs": max_abs, "corr": corr}


# =========================
# Config
# =========================

@dataclass
class ROIConfig:
    x0_rel: float
    y0_rel: float
    x1_rel: float
    y1_rel: float

    def clamp(self):
        self.x0_rel = max(0.0, min(1.0, self.x0_rel))
        self.y0_rel = max(0.0, min(1.0, self.y0_rel))
        self.x1_rel = max(0.0, min(1.0, self.x1_rel))
        self.y1_rel = max(0.0, min(1.0, self.y1_rel))
        if self.x1_rel <= self.x0_rel or self.y1_rel <= self.y0_rel:
            raise ValueError("ROI invalide.")

    def to_pixels(self, w: int, h: int) -> Tuple[int, int, int, int]:
        self.clamp()
        x0 = int(round(self.x0_rel * w))
        y0 = int(round(self.y0_rel * h))
        x1 = int(round(self.x1_rel * w))
        y1 = int(round(self.y1_rel * h))
        return x0, y0, x1, y1


@dataclass
class Calibration:
    slope: float
    intercept: float

    def px_to_mm(self, px: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(px, dtype=float) + self.intercept


# =========================
# Extraction ouverture vidéo
# =========================

class GripperVideoExtractor:
    def __init__(self, video_path: str, roi: ROIConfig, calibration: Calibration):
        self.video_path = video_path
        self.roi = roi
        self.calibration = calibration

    @staticmethod
    def _preprocess_roi(roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Lissage léger
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Les mors sont sombres: seuil inverse automatique
        _, mask = cv2.threshold(
            gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Nettoyage morphologique
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return gray_blur, mask

    @staticmethod
    def _find_gripper_components(mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        """
        Retourne les composantes candidates:
        [(x, y, w, h, area), ...]
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comps = []
        H, W = mask.shape[:2]
        min_area = max(150, int(0.001 * W * H))

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            # on veut des objets plutôt bas dans l'image et suffisamment hauts
            if h < 0.12 * H:
                continue
            comps.append((x, y, w, h, area))

        comps.sort(key=lambda c: c[4], reverse=True)
        return comps

    @staticmethod
    def _estimate_opening_from_mask(mask: np.ndarray) -> Tuple[float, Optional[Tuple[int, int]], Optional[Tuple[int, int]], dict]:
        """
        Mesure l'ouverture entre les deux bords internes.
        Méthode:
        - trouver les 2 plus grosses composantes noires
        - prendre leur bord interne à une ligne horizontale proche du bas
        - mesurer distance horizontale
        """
        H, W = mask.shape[:2]
        comps = GripperVideoExtractor._find_gripper_components(mask)

        debug = {"num_components": len(comps), "method": None}

        if len(comps) < 2:
            return np.nan, None, None, debug

        # Prendre 2 plus grosses composantes
        c1, c2 = comps[0], comps[1]

        # Ordonner gauche/droite
        if c1[0] > c2[0]:
            c1, c2 = c2, c1

        x1, y1, w1, h1, a1 = c1
        x2, y2, w2, h2, a2 = c2

        # Ligne de mesure: proche du bas commun visible
        y_candidates = [
            y1 + int(0.70 * h1),
            y1 + int(0.80 * h1),
            y1 + int(0.90 * h1),
            y2 + int(0.70 * h2),
            y2 + int(0.80 * h2),
            y2 + int(0.90 * h2),
        ]

        valid_measurements = []

        for yy in y_candidates:
            if yy < 0 or yy >= H:
                continue

            row = mask[yy, :]
            idx = np.where(row > 0)[0]
            if len(idx) < 2:
                continue

            # points du blob gauche sur cette ligne
            left_idx = idx[(idx >= x1) & (idx < x1 + w1)]
            right_idx = idx[(idx >= x2) & (idx < x2 + w2)]

            if len(left_idx) == 0 or len(right_idx) == 0:
                continue

            # bord interne:
            left_inner = left_idx.max()
            right_inner = right_idx.min()

            opening_px = right_inner - left_inner
            if opening_px > 0:
                valid_measurements.append((opening_px, (left_inner, yy), (right_inner, yy)))

        if not valid_measurements:
            return np.nan, None, None, debug

        # robuste: médiane des lignes valides
        vals = np.array([v[0] for v in valid_measurements], dtype=float)
        med = float(np.median(vals))
        best_idx = int(np.argmin(np.abs(vals - med)))
        opening_px, p_left, p_right = valid_measurements[best_idx]

        debug["method"] = "row_inner_edges"
        return float(opening_px), p_left, p_right, debug

    def process(self, output_dir: str, create_annotated_video: bool = False) -> pd.DataFrame:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-6:
            fps = 30.0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        x0, y0, x1, y1 = self.roi.to_pixels(width, height)

        writer = None
        annotated_path = os.path.join(output_dir, "annotated_output.mp4")
        if create_annotated_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

        records = []
        frame_id = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                t = frame_id / fps

                roi_bgr = frame[y0:y1, x0:x1]
                _, mask = self._preprocess_roi(roi_bgr)
                opening_px, p_left, p_right, dbg = self._estimate_opening_from_mask(mask)
                opening_mm = self.calibration.px_to_mm(np.array([opening_px]))[0] if np.isfinite(opening_px) else np.nan

                records.append({
                    "frame_id": frame_id,
                    "time_s": t,
                    "opening_px_raw": opening_px,
                    "opening_mm_raw": opening_mm,
                    "num_components": dbg.get("num_components", np.nan),
                    "method": dbg.get("method", "")
                })

                if writer is not None:
                    disp = frame.copy()
                    cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 2)

                    if p_left is not None and p_right is not None:
                        pl = (p_left[0] + x0, p_left[1] + y0)
                        pr = (p_right[0] + x0, p_right[1] + y0)
                        cv2.circle(disp, pl, 5, (0, 255, 0), -1)
                        cv2.circle(disp, pr, 5, (0, 0, 255), -1)
                        cv2.line(disp, pl, pr, (255, 255, 0), 2)

                    text1 = f"t={t:7.3f}s  px={opening_px if np.isfinite(opening_px) else -1:.1f}"
                    text2 = f"mm={opening_mm if np.isfinite(opening_mm) else -1:.2f}"
                    cv2.putText(disp, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.putText(disp, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                    writer.write(disp)

                frame_id += 1

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if len(records) == 0:
            raise RuntimeError("Aucune frame traitée.")

        df = pd.DataFrame(records)

        # interpolation des trous
        df["opening_px"] = fill_nan_1d(df["opening_px_raw"].to_numpy())
        df["opening_px"] = np.clip(df["opening_px"], a_min=0.0, a_max=None)
        df["opening_px_smooth"] = smooth_signal(df["opening_px"].to_numpy(), fps)

        df["opening_mm"] = self.calibration.px_to_mm(df["opening_px"].to_numpy())
        df["opening_mm_smooth"] = self.calibration.px_to_mm(df["opening_px_smooth"].to_numpy())

        return df


# =========================
# Signal capteur
# =========================

def load_sensor_signal(csv_path: str, time_col: str, open_col: str, time_unit: str) -> pd.DataFrame:
    df = robust_read_csv(csv_path)

    if time_col not in df.columns:
        raise RuntimeError(f"Colonne temps introuvable: {time_col}. Colonnes dispo: {list(df.columns)}")
    if open_col not in df.columns:
        raise RuntimeError(f"Colonne ouverture introuvable: {open_col}. Colonnes dispo: {list(df.columns)}")

    out = df[[time_col, open_col]].copy()
    out.columns = ["time_raw", "opening_sensor"]

    out["time_raw"] = pd.to_numeric(out["time_raw"], errors="coerce")
    out["opening_sensor"] = pd.to_numeric(out["opening_sensor"], errors="coerce")
    out = out.dropna().sort_values("time_raw").reset_index(drop=True)

    if out.empty:
        raise RuntimeError("Signal capteur vide après nettoyage.")

    if time_unit.lower() == "ms":
        out["time_s"] = out["time_raw"] / 1000.0
    elif time_unit.lower() == "s":
        out["time_s"] = out["time_raw"].astype(float)
    else:
        raise RuntimeError("time_unit doit être 's' ou 'ms'.")

    # normalisation origine temps
    out["time_s"] = out["time_s"] - out["time_s"].iloc[0]

    y = fill_nan_1d(out["opening_sensor"].to_numpy())
    dt = np.median(np.diff(out["time_s"].to_numpy())) if len(out) > 2 else 0.01
    fps_equiv = 1.0 / max(dt, 1e-6)
    out["opening_sensor_smooth"] = smooth_signal(y, fps_equiv)

    return out[["time_s", "opening_sensor", "opening_sensor_smooth"]]


# =========================
# Alignement vidéo vs capteur
# =========================

def estimate_time_offset(
    t_video: np.ndarray,
    y_video: np.ndarray,
    t_sensor: np.ndarray,
    y_sensor: np.ndarray,
    resample_hz: float = 100.0
) -> Tuple[float, pd.DataFrame]:
    """
    Retourne delta_t tel que:
        sensor_aligned(t) = sensor(t - delta_t)
    delta_t > 0 => le capteur est en retard
    """
    t_video = np.asarray(t_video, dtype=float)
    y_video = np.asarray(y_video, dtype=float)
    t_sensor = np.asarray(t_sensor, dtype=float)
    y_sensor = np.asarray(y_sensor, dtype=float)

    t0 = max(np.nanmin(t_video), np.nanmin(t_sensor))
    t1 = min(np.nanmax(t_video), np.nanmax(t_sensor))
    if t1 <= t0:
        raise RuntimeError("Pas de recouvrement temporel entre vidéo et capteur.")

    dt = 1.0 / resample_hz
    t_common = np.arange(t0, t1, dt)
    if len(t_common) < 10:
        raise RuntimeError("Base de temps commune trop courte pour alignement.")

    f_video = interp1d(t_video, y_video, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_sensor = interp1d(t_sensor, y_sensor, kind="linear", bounds_error=False, fill_value="extrapolate")

    yv = f_video(t_common)
    ys = f_sensor(t_common)

    yv_n = normalize_signal(yv)
    ys_n = normalize_signal(ys)

    c = correlate(yv_n, ys_n, mode="full")
    lags = np.arange(-len(yv_n) + 1, len(yv_n)) * dt
    best = int(np.argmax(c))
    best_lag = lags[best]

    # convention documentée
    delta_t = -best_lag

    df_corr = pd.DataFrame({
        "lag_s": lags,
        "xcorr": c
    })
    return float(delta_t), df_corr


def align_and_compare(
    video_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    output_dir: str
) -> Tuple[pd.DataFrame, Dict[str, float], float]:
    delta_t, corr_df = estimate_time_offset(
        t_video=video_df["time_s"].to_numpy(),
        y_video=video_df["opening_mm_smooth"].to_numpy(),
        t_sensor=sensor_df["time_s"].to_numpy(),
        y_sensor=sensor_df["opening_sensor_smooth"].to_numpy(),
        resample_hz=100.0
    )

    corr_df.to_csv(os.path.join(output_dir, "cross_correlation.csv"), index=False)

    sensor_aligned = sensor_df.copy()
    sensor_aligned["time_s_aligned"] = sensor_aligned["time_s"] + delta_t

    f_sensor = interp1d(
        sensor_aligned["time_s_aligned"].to_numpy(),
        sensor_aligned["opening_sensor"].to_numpy(),
        kind="linear",
        bounds_error=False,
        fill_value=np.nan
    )
    f_sensor_s = interp1d(
        sensor_aligned["time_s_aligned"].to_numpy(),
        sensor_aligned["opening_sensor_smooth"].to_numpy(),
        kind="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    cmp_df = video_df.copy()
    cmp_df["opening_sensor_interp"] = f_sensor(cmp_df["time_s"].to_numpy())
    cmp_df["opening_sensor_smooth_interp"] = f_sensor_s(cmp_df["time_s"].to_numpy())
    cmp_df["abs_error_mm"] = np.abs(cmp_df["opening_mm_smooth"] - cmp_df["opening_sensor_smooth_interp"])

    metrics = compute_metrics(
        cmp_df["opening_mm_smooth"].to_numpy(),
        cmp_df["opening_sensor_smooth_interp"].to_numpy()
    )

    return cmp_df, metrics, delta_t


# =========================
# Etats discrets 2/5/10/30 mm
# =========================

def classify_to_nominal_levels(values_mm: np.ndarray, levels=(2, 5, 10, 30)) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values_mm, dtype=float)
    levels = np.asarray(levels, dtype=float)

    assigned = np.full(vals.shape, np.nan)
    distance = np.full(vals.shape, np.nan)

    finite = np.isfinite(vals)
    if finite.sum() == 0:
        return assigned, distance

    d = np.abs(vals[finite, None] - levels[None, :])
    idx = np.argmin(d, axis=1)
    assigned[finite] = levels[idx]
    distance[finite] = d[np.arange(len(idx)), idx]

    return assigned, distance


def add_state_columns(df: pd.DataFrame, tolerance_mm: float) -> pd.DataFrame:
    out = df.copy()

    out["video_state_mm"], out["video_state_dist"] = classify_to_nominal_levels(out["opening_mm_smooth"].to_numpy())
    out["sensor_state_mm"], out["sensor_state_dist"] = classify_to_nominal_levels(out["opening_sensor_smooth_interp"].to_numpy())

    out["video_state_valid"] = out["video_state_dist"] <= tolerance_mm
    out["sensor_state_valid"] = out["sensor_state_dist"] <= tolerance_mm

    out["same_state"] = (
        out["video_state_valid"] &
        out["sensor_state_valid"] &
        (out["video_state_mm"] == out["sensor_state_mm"])
    )

    return out


# =========================
# Détection événements
# =========================

def detect_transition_events(time_s: np.ndarray, signal_mm: np.ndarray, min_prominence: float = 0.5) -> pd.DataFrame:
    y = np.asarray(signal_mm, dtype=float)
    t = np.asarray(time_s, dtype=float)

    dy = np.gradient(y, t)
    dy_s = smooth_signal(dy, fps=max(1.0, len(t) / max(t[-1] - t[0], 1e-6)))

    pos_peaks, _ = find_peaks(dy_s, prominence=min_prominence)
    neg_peaks, _ = find_peaks(-dy_s, prominence=min_prominence)

    rows = []
    for i in pos_peaks:
        rows.append({"time_s": t[i], "event": "opening_transition", "strength": dy_s[i]})
    for i in neg_peaks:
        rows.append({"time_s": t[i], "event": "closing_transition", "strength": -dy_s[i]})

    ev = pd.DataFrame(rows).sort_values("time_s").reset_index(drop=True) if rows else pd.DataFrame(columns=["time_s", "event", "strength"])
    return ev


# =========================
# Plots
# =========================

def save_plot_opening(video_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(video_df["time_s"], video_df["opening_mm_raw"], label="video_raw", alpha=0.5)
    plt.plot(video_df["time_s"], video_df["opening_mm_smooth"], label="video_smooth", linewidth=2)
    plt.xlabel("Temps (s)")
    plt.ylabel("Ouverture (mm)")
    plt.title("Ouverture vidéo vs temps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_plot_comparison(cmp_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(cmp_df["time_s"], cmp_df["opening_mm_smooth"], label="video_smooth", linewidth=2)
    plt.plot(cmp_df["time_s"], cmp_df["opening_sensor_smooth_interp"], label="sensor_aligned_smooth", linewidth=2)
    plt.xlabel("Temps (s)")
    plt.ylabel("Ouverture (mm)")
    plt.title("Comparaison vidéo / capteur après alignement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =========================
# Main
# =========================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extraction ouverture gripper vidéo et comparaison capteur.")

    p.add_argument("--video", required=True, help="Chemin vidéo mp4")
    p.add_argument("--sensor_csv", required=True, help="Chemin CSV capteur")
    p.add_argument("--sensor_time_col", required=True, help="Nom colonne temps capteur")
    p.add_argument("--sensor_open_col", required=True, help="Nom colonne ouverture capteur")
    p.add_argument("--sensor_time_unit", default="s", choices=["s", "ms"], help="Unité de la colonne temps capteur")

    p.add_argument("--output_dir", required=True, help="Répertoire de sortie")

    # ROI relative
    p.add_argument("--roi", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                   default=[0.15, 0.55, 0.85, 1.00],
                   help="ROI relative [x0 y0 x1 y1]")

    # Calibration pixels -> mm
    p.add_argument("--px_to_mm_slope", type=float, default=1.0, help="Pente conversion px->mm")
    p.add_argument("--px_to_mm_intercept", type=float, default=0.0, help="Ordonnée conversion px->mm")

    p.add_argument("--state_tolerance_mm", type=float, default=1.0, help="Tolérance pour valider états 2/5/10/30 mm")
    p.add_argument("--annotated_video", action="store_true", help="Génère une vidéo annotée")
    p.add_argument("--transition_prominence", type=float, default=0.5, help="Prominence min pour détecter transitions")

    return p


def write_report(
    output_dir: str,
    args,
    metrics: Dict[str, float],
    delta_t: float,
    video_df: pd.DataFrame,
    cmp_df: pd.DataFrame,
    events_df: pd.DataFrame
) -> None:
    path = os.path.join(output_dir, "alignment_report.txt")

    state_match_ratio = float(np.nanmean(cmp_df["same_state"].astype(float))) if "same_state" in cmp_df.columns and len(cmp_df) else np.nan
    valid_ratio = float(np.nanmean(np.isfinite(video_df["opening_mm_raw"]).astype(float)))

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== ALIGNMENT REPORT ===\n\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Sensor CSV: {args.sensor_csv}\n")
        f.write(f"Output dir: {args.output_dir}\n\n")

        f.write("ROI relative:\n")
        f.write(f"  x0={args.roi[0]:.4f}, y0={args.roi[1]:.4f}, x1={args.roi[2]:.4f}, y1={args.roi[3]:.4f}\n\n")

        f.write("Calibration px->mm:\n")
        f.write(f"  slope={args.px_to_mm_slope}\n")
        f.write(f"  intercept={args.px_to_mm_intercept}\n\n")

        f.write("Alignement:\n")
        f.write(f"  delta_t_seconds={delta_t:.6f}\n")
        f.write("  Convention: sensor_aligned_time = sensor_time + delta_t\n\n")

        f.write("Qualite extraction video:\n")
        f.write(f"  valid_detection_ratio={valid_ratio:.6f}\n\n")

        f.write("Metrics comparaison:\n")
        for k, v in metrics.items():
            f.write(f"  {k}={v:.6f}\n")
        f.write(f"  same_state_ratio={state_match_ratio:.6f}\n\n")

        f.write("Transitions detectees:\n")
        if len(events_df) == 0:
            f.write("  aucune\n")
        else:
            for _, row in events_df.iterrows():
                f.write(f"  t={row['time_s']:.3f}s | {row['event']} | strength={row['strength']:.4f}\n")


def main():
    parser = build_argparser()
    args = parser.parse_args()

    try:
        safe_makedirs(args.output_dir)

        roi = ROIConfig(*args.roi)
        calib = Calibration(args.px_to_mm_slope, args.px_to_mm_intercept)

        extractor = GripperVideoExtractor(args.video, roi, calib)
        video_df = extractor.process(
            output_dir=args.output_dir,
            create_annotated_video=args.annotated_video
        )
        video_df.to_csv(os.path.join(args.output_dir, "video_signal.csv"), index=False)

        sensor_df = load_sensor_signal(
            csv_path=args.sensor_csv,
            time_col=args.sensor_time_col,
            open_col=args.sensor_open_col,
            time_unit=args.sensor_time_unit
        )
        sensor_df.to_csv(os.path.join(args.output_dir, "sensor_clean.csv"), index=False)

        cmp_df, metrics, delta_t = align_and_compare(video_df, sensor_df, args.output_dir)
        cmp_df = add_state_columns(cmp_df, tolerance_mm=args.state_tolerance_mm)

        events_df = detect_transition_events(
            cmp_df["time_s"].to_numpy(),
            cmp_df["opening_mm_smooth"].to_numpy(),
            min_prominence=args.transition_prominence
        )

        cmp_df.to_csv(os.path.join(args.output_dir, "aligned_comparison.csv"), index=False)
        events_df.to_csv(os.path.join(args.output_dir, "detected_events.csv"), index=False)

        save_plot_opening(video_df, os.path.join(args.output_dir, "opening_vs_time.png"))
        save_plot_comparison(cmp_df, os.path.join(args.output_dir, "comparison_aligned.png"))

        write_report(
            output_dir=args.output_dir,
            args=args,
            metrics=metrics,
            delta_t=delta_t,
            video_df=video_df,
            cmp_df=cmp_df,
            events_df=events_df
        )

        config_dump = {
            "video": args.video,
            "sensor_csv": args.sensor_csv,
            "sensor_time_col": args.sensor_time_col,
            "sensor_open_col": args.sensor_open_col,
            "sensor_time_unit": args.sensor_time_unit,
            "output_dir": args.output_dir,
            "roi": args.roi,
            "px_to_mm_slope": args.px_to_mm_slope,
            "px_to_mm_intercept": args.px_to_mm_intercept,
            "state_tolerance_mm": args.state_tolerance_mm,
            "annotated_video": args.annotated_video,
            "transition_prominence": args.transition_prominence
        }
        with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dump, f, indent=2, ensure_ascii=False)

        print("Terminé.")
        print(f"Résultats dans: {args.output_dir}")

    except Exception as exc:
        print("ERREUR:", str(exc), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
