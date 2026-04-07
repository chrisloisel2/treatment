#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_clock_drift.py — Correction de la dérive linéaire d'horloge caméra.

Problème :
    L'horloge caméra peut dériver progressivement par rapport à l'horloge tracker.
    Ce phénomène se traduit par un offset qui augmente linéairement avec le temps :
        offset(t) ≈ a * t + b
    où :
        b = offset initial (géré par fix_camera_offset)
        a = taux de dérive (µs/s, ppm)

    Un drift de 500 ppm signifie que la caméra accumule 0.5ms de décalage par seconde,
    soit 30ms sur une minute — suffisant pour dégrader la synchronisation.

Algorithme :
  1. Estimer l'offset local caméra/tracker en N points distribués dans le temps
     (méthode : nearest-neighbour entre les deux flux)
  2. Ajustement linéaire : offset = slope * t + intercept
  3. Si R² ≥ seuil et |slope| ≥ seuil_ppm :
     → Appliquer la correction : new_ct = old_ct - (slope * t_rel + intercept)
     → La correction est nulle au début et croissante sur la durée

Validation :
  - Avant/après correction : recalcul de l'offset moyen résiduel
  - Si la correction aggrave les choses → non appliquée

Usage :
    from fix.fix_clock_drift import fix_clock_drift
    report = fix_clock_drift(Path("/path/to/session"))
"""

from __future__ import annotations

import json
import sys
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
    from scipy.stats import linregress as _linregress
    _SCIPY_LINREG = True
except ImportError:
    _SCIPY_LINREG = False


# ── Paramètres ────────────────────────────────────────────────────────────────

CAMERAS              = ("head", "left", "right")
DRIFT_MARKER_KEY     = "camera_clock_drift_applied"

# Nombre de points de mesure de l'offset
N_MEASURE_POINTS     = 30

# Taux de dérive minimum pour appliquer la correction (µs/s = ppm)
DRIFT_PPM_MIN        = 200.0

# R² minimum pour que la régression soit significative
DRIFT_R2_MIN         = 0.40

# Durée minimale de la session pour détecter un drift fiable (ms)
MIN_DURATION_MS      = 3000.0

# Tolérance pour la recherche de la frame caméra la plus proche (ms)
MATCH_TOL_MS         = 500.0


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires I/O
# ══════════════════════════════════════════════════════════════════════════════

def _read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    frames = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if len(line) > 5:
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return frames


def _write_jsonl(path: Path, frames: list[dict]) -> None:
    lines = [json.dumps(f, separators=(",", ":")) + "\r\n" for f in frames]
    path.write_bytes("".join(lines).encode("utf-8"))


def _load_tracker_times_ms(session_path: Path) -> Optional[np.ndarray]:
    if not _PANDAS:
        return None
    csv_path = session_path / "tracker_positions.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if "timestamp_ns" not in df.columns:
            return None
        t_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce").dropna().to_numpy(np.float64)
        return t_ns / 1e6 if len(t_ns) >= 10 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Estimation du drift
# ══════════════════════════════════════════════════════════════════════════════

def _linreg_numpy(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Régression linéaire pure numpy. Retourne (slope, intercept, r_squared)."""
    n = len(x)
    if n < 4:
        return 0.0, 0.0, 0.0
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mx, my = np.mean(x), np.mean(y)
    sx = np.sum((x - mx) ** 2)
    if sx < 1e-12:
        return 0.0, my, 0.0
    slope = np.sum((x - mx) * (y - my)) / sx
    intercept = my - slope * mx
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - my) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(slope), float(intercept), float(np.clip(r2, 0.0, 1.0))


def measure_drift(
    tracker_t_ms: np.ndarray,
    cam_t_ms: np.ndarray,
    n_points: int = N_MEASURE_POINTS,
) -> dict:
    """
    Mesure la dérive d'horloge entre les timestamps caméra et tracker.

    Méthode :
        On distribue N_MEASURE_POINTS instants équidistants dans la zone de chevauchement.
        Pour chaque instant tracker t_trk, on trouve la frame caméra la plus proche.
        L'offset local = cam_nearest - t_trk.
        On ajuste une droite sur (t_trk_rel, offset_local).

    Returns:
        dict avec slope_ms_per_ms, intercept_ms, r2, drift_ppm, valid
    """
    t_trk_t0 = float(tracker_t_ms[0])
    t_trk_t1 = float(tracker_t_ms[-1])
    t_cam_t0 = float(cam_t_ms[0])
    t_cam_t1 = float(cam_t_ms[-1])

    overlap_start = max(t_trk_t0, t_cam_t0)
    overlap_end   = min(t_trk_t1, t_cam_t1)
    overlap_dur   = overlap_end - overlap_start

    if overlap_dur < MIN_DURATION_MS:
        return {"valid": False, "reason": f"overlap trop court ({overlap_dur:.0f}ms)"}

    # Points de mesure uniformément espacés dans la zone commune
    t_measure = np.linspace(overlap_start, overlap_end, n_points)

    # Trouver la frame caméra la plus proche pour chaque point de mesure
    cam_sorted = np.sort(cam_t_ms)
    offsets_ms = []
    t_points   = []

    for t_trk in t_measure:
        idx = np.searchsorted(cam_sorted, t_trk)
        # Vérifier les voisins gauche et droit
        candidates = []
        for i in (idx - 1, idx):
            if 0 <= i < len(cam_sorted):
                candidates.append((abs(cam_sorted[i] - t_trk), cam_sorted[i]))
        if not candidates:
            continue
        dist, cam_nearest = min(candidates)
        if dist > MATCH_TOL_MS:
            continue
        offsets_ms.append(cam_nearest - t_trk)
        t_points.append(t_trk - t_trk_t0)  # temps relatif depuis t0 tracker

    if len(offsets_ms) < 6:
        return {"valid": False, "reason": f"seulement {len(offsets_ms)} points de mesure"}

    t_arr = np.array(t_points,  dtype=np.float64)
    o_arr = np.array(offsets_ms, dtype=np.float64)

    # Robustesse : supprimer les outliers (> 3 sigma)
    mu, sigma = np.mean(o_arr), np.std(o_arr)
    if sigma > 1.0:
        mask = np.abs(o_arr - mu) < 3.0 * sigma
        t_arr = t_arr[mask]
        o_arr = o_arr[mask]

    if len(t_arr) < 4:
        return {"valid": False, "reason": "trop d'outliers"}

    if _SCIPY_LINREG:
        from scipy.stats import linregress
        slope, intercept, r, p, _ = linregress(t_arr, o_arr)
        r2 = float(r ** 2)
    else:
        slope, intercept, r2 = _linreg_numpy(t_arr, o_arr)

    # slope en ms/ms → drift_ppm (µs/s)
    drift_ppm = abs(float(slope)) * 1e6

    return {
        "valid":            True,
        "slope_ms_per_ms":  round(float(slope), 8),
        "intercept_ms":     round(float(intercept), 3),
        "r2":               round(r2, 4),
        "drift_ppm":        round(drift_ppm, 1),
        "n_points":         len(t_arr),
        "offset_mean_ms":   round(float(np.mean(o_arr)), 3),
        "offset_std_ms":    round(float(np.std(o_arr)), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Fix principal
# ══════════════════════════════════════════════════════════════════════════════

def fix_clock_drift(
    session_path: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Détecte et corrige la dérive linéaire d'horloge des caméras.

    La correction est appliquée uniquement si :
      - drift_ppm ≥ DRIFT_PPM_MIN
      - R² ≥ DRIFT_R2_MIN
      - La correction réduit bien la variance des offsets (validation)

    Note : fix_camera_offset doit être appliqué EN PREMIER pour corriger
    l'offset grossier (intercept). Ici on corrige uniquement la pente (slope).

    Args:
        session_path : chemin vers la session
        dry_run      : mesure uniquement, sans modifier
        force        : re-applique même si déjà corrigé

    Returns:
        dict rapport avec drift_ppm, r2, correction appliquée par caméra
    """
    name = session_path.name
    meta_path = session_path / "metadata.json"

    if not meta_path.exists():
        return {"session": name, "status": "error", "reason": "metadata.json absent"}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"session": name, "status": "error", "reason": f"metadata.json illisible: {e}"}

    if not force and meta.get(DRIFT_MARKER_KEY):
        return {"session": name, "status": "skipped", "reason": "déjà corrigé (drift)"}

    # Charger les timestamps tracker
    tracker_t_ms = _load_tracker_times_ms(session_path)
    if tracker_t_ms is None or len(tracker_t_ms) < 20:
        return {"session": name, "status": "error",
                "reason": "tracker_positions.csv insuffisant"}

    trk_t0 = float(tracker_t_ms[0])

    cam_reports: dict[str, dict] = {}
    corrections_applied = 0

    for cam in CAMERAS:
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        if not jsonl_path.exists():
            continue

        frames = _read_jsonl(jsonl_path)
        if len(frames) < 30:
            cam_reports[cam] = {"status": "skip", "reason": "trop peu de frames"}
            continue

        cam_t_ms = np.array([
            float(fr["capture_time"])
            for fr in frames
            if fr.get("capture_time") is not None
        ], dtype=np.float64)

        if len(cam_t_ms) < 20:
            cam_reports[cam] = {"status": "skip", "reason": "timestamps invalides"}
            continue

        # Mesurer le drift
        drift = measure_drift(tracker_t_ms, cam_t_ms)
        cam_reports[cam] = drift.copy()

        if not drift.get("valid", False):
            cam_reports[cam]["status"] = "no_drift"
            continue

        drift_ppm = drift.get("drift_ppm", 0.0)
        r2        = drift.get("r2", 0.0)
        slope     = drift.get("slope_ms_per_ms", 0.0)
        intercept = drift.get("intercept_ms", 0.0)

        # Décision : corriger ?
        if drift_ppm < DRIFT_PPM_MIN or r2 < DRIFT_R2_MIN:
            cam_reports[cam]["status"] = "drift_negligible"
            cam_reports[cam]["decision"] = (
                f"drift={drift_ppm:.0f}ppm R²={r2:.3f} — sous le seuil"
            )
            continue

        cam_reports[cam]["status"] = "drift_detected"
        cam_reports[cam]["decision"] = (
            f"drift={drift_ppm:.0f}ppm R²={r2:.3f} — correction appliquée"
        )

        if dry_run:
            corrections_applied += 1
            continue

        # ── Application de la correction ──────────────────────────────────────
        # correction(t) = slope * (t - trk_t0) + intercept
        # new_ct = old_ct - correction(old_ct)
        # Note : on corrige sur le temps de la frame (pas le temps tracker)
        #        car on veut annuler la dérive progressive

        corrected_frames = []
        for fr in frames:
            ct = fr.get("capture_time")
            if ct is None:
                continue
            t_rel = float(ct) - trk_t0
            correction_ms = slope * t_rel + intercept
            new_ct = float(ct) - correction_ms
            fr_new = dict(fr)
            fr_new["capture_time"] = round(new_ct, 3)
            corrected_frames.append(fr_new)

        # Validation : la correction doit réduire la variance des offsets
        new_cam_t = np.array([
            float(fr["capture_time"])
            for fr in corrected_frames
            if fr.get("capture_time") is not None
        ], dtype=np.float64)
        drift_after = measure_drift(tracker_t_ms, new_cam_t)

        drift_ppm_after = drift_after.get("drift_ppm", 0.0) if drift_after.get("valid") else 0.0
        if drift_ppm_after >= drift_ppm * 0.8:
            # La correction n'a pas amélioré les choses → on annule
            cam_reports[cam]["status"] = "correction_rejected"
            cam_reports[cam]["drift_ppm_after"] = drift_ppm_after
            cam_reports[cam]["decision"] = (
                f"correction rejetée : drift après={drift_ppm_after:.0f}ppm "
                f"≥ avant={drift_ppm:.0f}ppm"
            )
            continue

        # Écrire les frames corrigées
        _write_jsonl(jsonl_path, corrected_frames)
        corrections_applied += 1
        cam_reports[cam]["status"] = "corrected"
        cam_reports[cam]["drift_ppm_after"] = round(drift_ppm_after, 1)

    if corrections_applied == 0:
        return {
            "session":     name,
            "status":      "ok",
            "reason":      "aucun drift significatif détecté",
            "cam_reports": cam_reports,
        }

    if not dry_run:
        meta[DRIFT_MARKER_KEY] = True
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "session":            name,
        "status":             "corrected" if not dry_run else "dry-run",
        "corrections_applied": corrections_applied,
        "cam_reports":        cam_reports,
    }
