#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_camera_offset.py — Recalage précis des timestamps caméra sur le tracker.

Problème :
    Les caméras démarrent 6 à 23 secondes AVANT le tracker.
    Les capture_time sont des timestamps epoch ms corrects mais décalés.
    Résultat : la majorité des frames n'ont pas de correspondance tracker.

Algorithme amélioré (3 passes) :
  1. Offset grossier    : offset = median(cam_head_frames[:5]) - tracker_t0
     → rapide, déplace l'ensemble des frames dans la fenêtre tracker

  2. Affinage par cross-corrélation :
     Sur chaque caméra, on essaie tous les lags [-MAX_LAG, +MAX_LAG] par pas de 10ms.
     On retient le lag qui maximise la corrélation IFI_caméra ↔ vitesse_tracker.
     → précision ≈ 10ms

  3. Raffinement sub-10ms (interpolation parabolique au voisinage du maximum xcorr)
     → précision ≈ 1ms

  4. Troncature :
     Suppression des frames hors fenêtre [tracker_t0, tracker_t1] après correction.

Marqueur anti-double-application :
    metadata.json → "camera_tracker_sync_applied": true
    Utiliser force=True pour ré-appliquer.

Usage :
    from fix.fix_camera_offset import fix_camera_offset
    report = fix_camera_offset(Path("/path/to/session"))
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
    from scipy.signal import correlate as _correlate
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ── Paramètres ────────────────────────────────────────────────────────────────

CAMERAS               = ("head", "left", "right")
MARKER_KEY            = "camera_tracker_sync_applied"

# Seuil en dessous duquel l'offset brut est considéré négligeable
OFFSET_THRESHOLD_MS   = 30.0

# Plage de recherche xcorr (±)
XCORR_MAX_LAG_MS      = 5000.0

# Pas de la grille xcorr (grossier puis fin)
XCORR_STEP_COARSE_MS  = 50.0
XCORR_STEP_FINE_MS    = 5.0

# Résolution de rééchantillonnage pour la xcorr
RESAMPLE_MS           = 20.0

# Overlap minimum pour que la xcorr soit significative
MIN_OVERLAP_MS        = 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires I/O
# ══════════════════════════════════════════════════════════════════════════════

def _read_jsonl(path: Path) -> list[dict]:
    """Parse JSONL robuste."""
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


def _load_tracker_window(session_path: Path) -> tuple[Optional[float], Optional[float]]:
    """Retourne (t0_ms, t1_ms) de la fenêtre tracker depuis timestamp_ns."""
    csv_path = session_path / "tracker_positions.csv"
    if not csv_path.exists():
        return None, None
    t0, t1 = None, None
    try:
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ns_str = row.get("timestamp_ns", "").strip()
                if not ns_str:
                    continue
                try:
                    t_ms = int(ns_str) / 1_000_000
                except ValueError:
                    continue
                if t0 is None:
                    t0 = t_ms
                t1 = t_ms
    except Exception:
        pass
    return t0, t1


def _load_tracker_signal(session_path: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Charge le signal de vitesse de déplacement du tracker (en ms relatifs depuis t0).
    Retourne (t_ms_rel, speed) ou None.
    """
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
        if len(t_ns) < 10:
            return None
        t_ms = t_ns / 1e6
        t_ms_rel = t_ms - t_ms[0]

        # Vitesse du premier tracker trouvé
        speed = None
        for prefix in ("tracker_head", "tracker_left", "tracker_right"):
            cols = [f"{prefix}_{ax}" for ax in ("x", "y", "z")]
            if all(c in df.columns for c in cols):
                xyz = np.stack([
                    pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(np.float64)
                    for c in cols
                ], axis=1)
                s = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[:1]), axis=1)
                speed = s.astype(np.float32)
                break

        if speed is None:
            return None

        valid = np.isfinite(t_ms_rel) & np.isfinite(speed)
        return t_ms_rel[valid], speed[valid]
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Cross-corrélation : recherche du lag optimal
# ══════════════════════════════════════════════════════════════════════════════

def _build_cam_signal(
    times: np.ndarray,
    t0_abs_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit un signal caméra à partir des timestamps JSONL.
    Signal = variation normalisée de l'IFI (inter-frame interval).
    Retourne (t_ms_rel, signal).
    """
    t_rel = times - t0_abs_ms
    ifi = np.diff(times, prepend=times[0]).astype(np.float64)
    med_ifi = float(np.median(ifi))
    if med_ifi < 1e-3:
        return t_rel, np.zeros(len(times))
    ifi_dev = np.abs(ifi - med_ifi)
    win = max(3, int(200.0 / med_ifi))
    if win < len(ifi_dev):
        kernel = np.ones(win, dtype=np.float64) / win
        sig = np.convolve(ifi_dev, kernel, mode="same")
    else:
        sig = ifi_dev
    mu, std = np.mean(sig), np.std(sig)
    sig = (sig - mu) / (std + 1e-8)
    return t_rel, sig.astype(np.float32)


def _xcorr_score(
    trk_t: np.ndarray,
    trk_sig: np.ndarray,
    cam_t_abs: np.ndarray,
    cam_sig: np.ndarray,
    lag_ms: float,
    grid_step: float = RESAMPLE_MS,
) -> float:
    """
    Évalue la qualité d'alignement pour un lag donné.
    Retourne la corrélation de Pearson normalisée sur la zone commune.
    """
    # Appliquer le lag : déplacer les timestamps caméra
    cam_t_shifted = cam_t_abs + lag_ms

    # Zone commune
    t_start = max(float(trk_t[0]), float(cam_t_shifted[0]))
    t_end   = min(float(trk_t[-1]), float(cam_t_shifted[-1]))
    if t_end - t_start < MIN_OVERLAP_MS:
        return -1.0

    grid = np.arange(t_start, t_end, grid_step)
    if len(grid) < 20:
        return -1.0

    # Rééchantillonner les deux signaux sur la grille
    a = np.interp(grid, trk_t, trk_sig).astype(np.float64)
    b = np.interp(grid, cam_t_shifted, cam_sig).astype(np.float64)

    # Corrélation de Pearson
    a -= a.mean()
    b -= b.mean()
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a < 1e-8 or std_b < 1e-8:
        return 0.0
    return float(np.clip(np.mean(a * b) / (std_a * std_b), -1.0, 1.0))


def find_optimal_lag(
    trk_t_ms_abs: np.ndarray,
    trk_speed: np.ndarray,
    cam_t_ms_abs: np.ndarray,
    cam_sig: np.ndarray,
    rough_offset_ms: float = 0.0,
) -> tuple[float, float]:
    """
    Recherche le lag optimal entre caméra et tracker par grid search xcorr.

    Stratégie en 2 passes :
      1. Passe grossière : ±XCORR_MAX_LAG_MS par pas de XCORR_STEP_COARSE_MS
         (autour de rough_offset_ms pour aller plus vite)
      2. Passe fine : ±2 × XCORR_STEP_COARSE_MS autour du meilleur grossier,
         par pas de XCORR_STEP_FINE_MS

    Returns:
        (best_lag_ms, best_corr)
    """
    # Normaliser les signaux
    trk_mu, trk_std = np.mean(trk_speed), np.std(trk_speed)
    trk_n = (trk_speed - trk_mu) / (trk_std + 1e-8)
    cam_mu, cam_std = np.mean(cam_sig), np.std(cam_sig)
    cam_n = (cam_sig - cam_mu) / (cam_std + 1e-8)

    # ── Passe 1 : grossière ────────────────────────────────────────────────────
    search_center = rough_offset_ms
    coarse_lags = np.arange(
        search_center - XCORR_MAX_LAG_MS,
        search_center + XCORR_MAX_LAG_MS + XCORR_STEP_COARSE_MS,
        XCORR_STEP_COARSE_MS,
    )

    best_lag   = rough_offset_ms
    best_score = -2.0

    for lag in coarse_lags:
        s = _xcorr_score(trk_t_ms_abs, trk_n, cam_t_ms_abs, cam_n, float(lag),
                         grid_step=RESAMPLE_MS * 2)
        if s > best_score:
            best_score = s
            best_lag   = float(lag)

    # ── Passe 2 : fine ────────────────────────────────────────────────────────
    fine_lags = np.arange(
        best_lag - XCORR_STEP_COARSE_MS * 2,
        best_lag + XCORR_STEP_COARSE_MS * 2 + XCORR_STEP_FINE_MS,
        XCORR_STEP_FINE_MS,
    )

    for lag in fine_lags:
        s = _xcorr_score(trk_t_ms_abs, trk_n, cam_t_ms_abs, cam_n, float(lag),
                         grid_step=RESAMPLE_MS)
        if s > best_score:
            best_score = s
            best_lag   = float(lag)

    # ── Passe 3 : interpolation parabolique sub-pixel ─────────────────────────
    # Calculer les scores autour du meilleur point pour affiner
    eps = XCORR_STEP_FINE_MS / 2.0
    s_m = _xcorr_score(trk_t_ms_abs, trk_n, cam_t_ms_abs, cam_n, best_lag - eps,
                       grid_step=RESAMPLE_MS)
    s_0 = best_score
    s_p = _xcorr_score(trk_t_ms_abs, trk_n, cam_t_ms_abs, cam_n, best_lag + eps,
                       grid_step=RESAMPLE_MS)
    denom = 2.0 * (s_m - 2.0 * s_0 + s_p)
    if abs(denom) > 1e-10:
        sub_offset = eps * (s_m - s_p) / denom
        sub_offset = float(np.clip(sub_offset, -eps, eps))
        best_lag += sub_offset
        best_score = _xcorr_score(
            trk_t_ms_abs, trk_n, cam_t_ms_abs, cam_n, best_lag,
            grid_step=RESAMPLE_MS,
        )

    return round(best_lag, 1), round(best_score, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Fix principal
# ══════════════════════════════════════════════════════════════════════════════

def fix_camera_offset(
    session_path: Path,
    dry_run: bool = False,
    force: bool = False,
    use_xcorr: bool = True,
) -> dict:
    """
    Recale et tronque les capture_time des caméras sur la fenêtre du tracker.

    Étapes :
      1. Mesure offset brut (cam[median_first_frames] - tracker_t0)
      2. Si use_xcorr : affinage par cross-corrélation (meilleure précision)
      3. Application : capture_time_new = capture_time_old - optimal_lag_ms
      4. Troncature : suppression des frames hors [tracker_t0, tracker_t1]
      5. Marquage metadata

    Args:
        session_path : chemin vers la session
        dry_run      : si True, mesure uniquement sans modifier
        force        : si True, re-applique même si déjà corrigé
        use_xcorr    : si True, utilise la cross-corrélation pour l'affinage

    Returns:
        dict rapport avec status, offsets par caméra, xcorr scores
    """
    name = session_path.name
    meta_path = session_path / "metadata.json"

    # ── Vérifications préliminaires ───────────────────────────────────────────
    if not meta_path.exists():
        return {"session": name, "status": "error", "reason": "metadata.json absent"}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"session": name, "status": "error", "reason": f"metadata.json illisible: {e}"}

    if not force and meta.get(MARKER_KEY):
        return {"session": name, "status": "skipped", "reason": "déjà corrigée"}

    # ── Fenêtre tracker ────────────────────────────────────────────────────────
    trk_t0, trk_t1 = _load_tracker_window(session_path)
    if trk_t0 is None:
        return {"session": name, "status": "error",
                "reason": "tracker_positions.csv introuvable ou sans timestamp_ns"}

    # ── Signal tracker pour xcorr ──────────────────────────────────────────────
    trk_data = None
    trk_t_abs = None
    trk_speed = None
    if use_xcorr:
        trk_data = _load_tracker_signal(session_path)
        if trk_data is not None:
            trk_t_rel, trk_speed = trk_data
            trk_t_abs = trk_t_rel + trk_t0  # timestamps absolus ms

    # ── Lecture JSONL de chaque caméra ─────────────────────────────────────────
    cam_frames: dict[str, list[dict]] = {}
    raw_offsets: dict[str, float] = {}
    cam_times_abs: dict[str, np.ndarray] = {}

    for cam in CAMERAS:
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        if not jsonl_path.exists():
            continue
        frames = _read_jsonl(jsonl_path)
        if not frames:
            continue
        cam_frames[cam] = frames

        # Offset brut (médiane des 5 premières frames)
        head_times = []
        for fr in frames[:5]:
            ct = fr.get("capture_time")
            if ct is not None:
                head_times.append(float(ct))
        if head_times:
            raw_offsets[cam] = float(np.median(head_times)) - trk_t0
        else:
            raw_offsets[cam] = 0.0

        # Tous les timestamps
        all_times = np.array([
            float(fr["capture_time"])
            for fr in frames
            if fr.get("capture_time") is not None
        ], dtype=np.float64)
        if len(all_times) > 0:
            cam_times_abs[cam] = all_times

    if not cam_frames:
        return {"session": name, "status": "error", "reason": "aucun fichier JSONL trouvé"}

    # ── Calcul des lags optimaux ───────────────────────────────────────────────
    optimal_lags:  dict[str, float] = {}
    xcorr_scores:  dict[str, float] = {}
    offset_method: dict[str, str]   = {}

    for cam in cam_frames:
        rough_offset = raw_offsets.get(cam, 0.0)
        times = cam_times_abs.get(cam)

        if times is None or len(times) < 10:
            optimal_lags[cam] = rough_offset
            offset_method[cam] = "rough_only"
            continue

        # Vérifier si l'offset est négligeable
        if abs(rough_offset) < OFFSET_THRESHOLD_MS and not force:
            optimal_lags[cam] = 0.0
            xcorr_scores[cam]  = 1.0
            offset_method[cam] = "negligible"
            continue

        if use_xcorr and trk_data is not None and trk_t_abs is not None:
            # Signal caméra
            _, cam_sig = _build_cam_signal(times, float(times[0]))

            # Recherche xcorr
            lag, corr = find_optimal_lag(
                trk_t_abs=trk_t_abs,
                trk_speed=trk_speed,
                cam_t_ms_abs=times,
                cam_sig=cam_sig,
                rough_offset_ms=rough_offset,
            )

            # Si la xcorr donne un résultat raisonnable (corr > 0), on l'utilise
            if corr > 0.05:
                # Le lag xcorr est le décalage optimal à appliquer :
                # new_cam_t = old_cam_t - lag  → le lag EST la correction à soustraire
                # Mais find_optimal_lag retourne le lag à AJOUTER aux timestamps cam
                # pour maximiser la corrélation avec le tracker.
                # Donc : optimal_correction = -lag
                # Vérifions : si cam_t0 = trk_t0 + 1000, lag_optimal = +1000ms
                # → cam_t_shifted = cam_t + 1000 → cam[0] → trk[0] ✓
                # → correction à appliquer sur capture_time : subtract(-lag) = add(lag)?
                # NON : on veut cam_t_corrected = cam_t - raw_offset
                # mais xcorr trouve le lag tel que cam_t + lag ≈ trk_t
                # donc correction = lag (soustraction d'une valeur négative = addition)
                # Plus précisément : si cam est en avance de 10s sur tracker,
                # raw_offset = cam[0] - trk[0] = +10000ms
                # xcorr trouve lag = -10000ms (il faut décaler cam de -10s pour l'aligner)
                # → on soustrait offset = +10000ms → cam_corrected = cam - 10000 = trk[0]
                # Donc : optimal_correction_to_subtract = -lag
                optimal_lags[cam] = -lag
                xcorr_scores[cam]  = corr
                offset_method[cam] = "xcorr"
            else:
                # xcorr peu fiable → fallback sur l'offset brut
                optimal_lags[cam] = rough_offset
                xcorr_scores[cam]  = corr
                offset_method[cam] = "rough_fallback"
        else:
            optimal_lags[cam] = rough_offset
            offset_method[cam] = "rough"

    # Vérifier si tous les offsets sont négligeables
    max_correction = max((abs(v) for v in optimal_lags.values()), default=0.0)
    if max_correction < OFFSET_THRESHOLD_MS and not force:
        return {
            "session":     name,
            "status":      "ok",
            "reason":      f"offset max = {max_correction:.1f} ms < seuil — déjà aligné",
            "offsets_ms":  raw_offsets,
            "xcorr":       xcorr_scores,
        }

    # ── Application des corrections (si pas dry_run) ──────────────────────────
    report = {
        "session":         name,
        "status":          "corrected" if not dry_run else "dry-run",
        "tracker_t0_ms":   trk_t0,
        "tracker_t1_ms":   trk_t1,
        "raw_offsets_ms":  raw_offsets,
        "applied_ms":      optimal_lags,
        "xcorr_scores":    xcorr_scores,
        "methods":         offset_method,
    }

    if dry_run:
        return report

    frames_before = {cam: len(frames) for cam, frames in cam_frames.items()}
    frames_after  = {}

    for cam, frames in cam_frames.items():
        correction = optimal_lags.get(cam, 0.0)
        if abs(correction) < 1.0 and not force:
            frames_after[cam] = len(frames)
            continue

        # Appliquer la correction et tronquer
        corrected = []
        for fr in frames:
            ct = fr.get("capture_time")
            if ct is None:
                continue
            new_ct = float(ct) - correction
            # Garder uniquement les frames dans la fenêtre tracker (avec marge de 50ms)
            if trk_t0 - 50.0 <= new_ct <= trk_t1 + 50.0:
                fr_new = dict(fr)
                fr_new["capture_time"] = round(new_ct, 3)
                corrected.append(fr_new)

        # Renuméroter les index si présents
        if corrected and "index" in corrected[0]:
            for i, fr in enumerate(corrected):
                fr["index"] = i

        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        _write_jsonl(jsonl_path, corrected)
        frames_after[cam] = len(corrected)

    # Mettre à jour metadata
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta[MARKER_KEY] = True
    meta["camera_offset_corrections_ms"] = {
        cam: round(v, 1) for cam, v in optimal_lags.items()
    }
    meta["camera_offset_xcorr_scores"] = {
        cam: round(v, 4) for cam, v in xcorr_scores.items()
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    report["frames_before"] = frames_before
    report["frames_after"]  = frames_after
    return report
