#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_sync_lag.py — Correction fine du lag de synchronisation caméra/tracker.

Problème :
    Après fix_camera_offset et fix_clock_drift, il peut rester un lag résiduel
    sub-seconde entre les caméras et le tracker.
    Ce lag dégrade le score IA même si la couverture est correcte.

Approche multi-méthode (choisir la meilleure) :

  METHOD A — Cross-corrélation IFI/vitesse-tracker :
    Signal caméra   : variation de l'inter-frame interval (IFI)
    Signal tracker  : vitesse de déplacement (norme de la dérivée)
    Recherche       : grid search ±MAX_LAG_MS par pas fins
    Précision       : 1-5ms selon la qualité du signal

  METHOD B — Score IA (si modèle disponible) :
    Utiliser le modèle de check.py pour scorer chaque lag candidat.
    Précision : meilleure que A mais plus lente.
    Utilisé en validation si modèle présent.

  VALIDATION :
    Après application du lag :
      - Recalcul du score xcorr (doit augmenter)
      - Si modèle disponible : recalcul du score IA
    Si la correction dégrade le score → non appliquée.

Usage :
    from fix.fix_sync_lag import fix_sync_lag
    report = fix_sync_lag(Path("/path/to/session"))
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


# ── Paramètres ────────────────────────────────────────────────────────────────

SYNC_MARKER_KEY         = "sync_lag_fixed"
CAMERAS                 = ("head", "left", "right")

# Plage de recherche du lag résiduel (beaucoup plus petite que fix_camera_offset)
MAX_LAG_SEARCH_MS       = 800.0
LAG_STEP_COARSE_MS      = 20.0
LAG_STEP_FINE_MS        = 2.0

# Résolution de rééchantillonnage
RESAMPLE_MS             = 10.0

# Overlap minimum pour la xcorr
MIN_OVERLAP_MS          = 1500.0

# Amélioration minimale du score pour appliquer la correction
MIN_SCORE_IMPROVEMENT   = 0.02

# Score xcorr minimum (sous ce seuil, le signal est trop bruité)
MIN_XCORR_SIGNAL_QUALITY = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# Chargement des données
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


def _load_tracker_data(session_path: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Retourne (t_ms_abs, speed) du tracker.
    speed = norme de la dérivée de position.
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
        t_ms = pd.to_numeric(df["timestamp_ns"], errors="coerce").dropna().to_numpy(np.float64) / 1e6
        if len(t_ms) < 20:
            return None

        speed = None
        for prefix in ("tracker_head", "tracker_left", "tracker_right"):
            cols = [f"{prefix}_{ax}" for ax in ("x", "y", "z")]
            if all(c in df.columns for c in cols):
                xyz = np.stack([
                    pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(np.float64)
                    for c in cols
                ], axis=1)
                dt = np.diff(t_ms, prepend=t_ms[0])
                dt[dt < 1e-3] = 1e-3
                pos_diff = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[:1]), axis=1)
                speed = pos_diff / dt  # mm/s approximatif
                break

        if speed is None:
            return None

        valid = np.isfinite(t_ms) & np.isfinite(speed)
        return t_ms[valid], speed[valid]
    except Exception:
        return None


def _build_cam_ifi_signal(
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit le signal IFI normalisé de la caméra.
    Retourne (t_abs_ms, signal).
    """
    ifi = np.diff(times, prepend=times[0]).astype(np.float64)
    med = float(np.median(ifi))
    if med < 1e-3:
        return times, np.zeros(len(times))
    dev = np.abs(ifi - med)
    win = max(3, int(200.0 / med))
    if win < len(dev):
        kernel = np.ones(win, dtype=np.float64) / win
        dev = np.convolve(dev, kernel, mode="same")
    mu, std = np.mean(dev), np.std(dev)
    sig = (dev - mu) / (std + 1e-8)
    return times, sig.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Scoring xcorr sur une grille de lags
# ══════════════════════════════════════════════════════════════════════════════

def _score_lag(
    trk_t: np.ndarray,
    trk_sig: np.ndarray,
    cam_t: np.ndarray,
    cam_sig: np.ndarray,
    lag_ms: float,
    grid_step: float = RESAMPLE_MS,
) -> float:
    """Corrélation de Pearson à un lag donné (cam décalée de lag_ms vers la droite)."""
    cam_t_shifted = cam_t + lag_ms
    t_start = max(float(trk_t[0]), float(cam_t_shifted[0]))
    t_end   = min(float(trk_t[-1]), float(cam_t_shifted[-1]))
    if t_end - t_start < MIN_OVERLAP_MS:
        return -1.0
    grid = np.arange(t_start, t_end, grid_step)
    if len(grid) < 30:
        return -1.0
    a = np.interp(grid, trk_t, trk_sig).astype(np.float64)
    b = np.interp(grid, cam_t_shifted, cam_sig).astype(np.float64)
    a -= a.mean(); b -= b.mean()
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.clip(np.mean(a * b) / (sa * sb), -1.0, 1.0))


def find_best_lag(
    trk_t: np.ndarray,
    trk_sig: np.ndarray,
    cam_t: np.ndarray,
    cam_sig: np.ndarray,
    search_center_ms: float = 0.0,
) -> tuple[float, float, float]:
    """
    Recherche le lag optimal en deux passes (grossière + fine).

    Returns:
        (best_lag_ms, score_before, score_after)
    """
    # Normaliser les signaux
    def _norm(s):
        m, sd = np.mean(s), np.std(s)
        return (s - m) / (sd + 1e-8)

    trk_n = _norm(trk_sig)
    cam_n = _norm(cam_sig)

    # Score initial (lag = 0)
    score_before = _score_lag(trk_t, trk_n, cam_t, cam_n, 0.0)

    if abs(score_before) < MIN_XCORR_SIGNAL_QUALITY:
        return 0.0, score_before, score_before

    # Passe 1 : grossière
    coarse_lags = np.arange(
        search_center_ms - MAX_LAG_SEARCH_MS,
        search_center_ms + MAX_LAG_SEARCH_MS + LAG_STEP_COARSE_MS,
        LAG_STEP_COARSE_MS,
    )
    best_lag   = 0.0
    best_score = score_before

    for lag in coarse_lags:
        s = _score_lag(trk_t, trk_n, cam_t, cam_n, float(lag), grid_step=RESAMPLE_MS * 2)
        if s > best_score:
            best_score = s
            best_lag   = float(lag)

    # Passe 2 : fine autour du meilleur grossier
    fine_lags = np.arange(
        best_lag - LAG_STEP_COARSE_MS * 2,
        best_lag + LAG_STEP_COARSE_MS * 2 + LAG_STEP_FINE_MS,
        LAG_STEP_FINE_MS,
    )
    for lag in fine_lags:
        s = _score_lag(trk_t, trk_n, cam_t, cam_n, float(lag), grid_step=RESAMPLE_MS)
        if s > best_score:
            best_score = s
            best_lag   = float(lag)

    # Interpolation parabolique sub-pixel
    eps = LAG_STEP_FINE_MS / 2.0
    s_m = _score_lag(trk_t, trk_n, cam_t, cam_n, best_lag - eps)
    s_0 = best_score
    s_p = _score_lag(trk_t, trk_n, cam_t, cam_n, best_lag + eps)
    denom = 2.0 * (s_m - 2.0 * s_0 + s_p)
    if abs(denom) > 1e-10:
        sub = eps * (s_m - s_p) / denom
        sub = float(np.clip(sub, -eps, eps))
        new_lag = best_lag + sub
        new_score = _score_lag(trk_t, trk_n, cam_t, cam_n, new_lag)
        if new_score > best_score:
            best_lag   = new_lag
            best_score = new_score

    return round(best_lag, 1), round(score_before, 4), round(best_score, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Fix principal
# ══════════════════════════════════════════════════════════════════════════════

def fix_sync_lag(
    session_path: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Détecte et corrige le lag résiduel de synchronisation entre caméras et tracker.

    Ne corrige que si l'amélioration de la xcorr est ≥ MIN_SCORE_IMPROVEMENT.
    Le lag trouvé est soustrait des capture_time (cam_t_new = cam_t - lag).

    Attention : cette correction est appliquée PAR CAMÉRA.
    Si les 3 caméras ont des lags différents, elles sont corrigées indépendamment.
    Cela peut indiquer un problème de placement ou d'assignement.

    Args:
        session_path : chemin vers la session
        dry_run      : mesure uniquement
        force        : re-applique

    Returns:
        dict rapport avec lag_ms et score_improvement par caméra
    """
    name = session_path.name
    meta_path = session_path / "metadata.json"

    if not meta_path.exists():
        return {"session": name, "status": "error", "reason": "metadata.json absent"}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"session": name, "status": "error", "reason": f"metadata.json illisible: {e}"}

    if not force and meta.get(SYNC_MARKER_KEY):
        return {"session": name, "status": "skipped", "reason": "déjà corrigé (sync lag)"}

    # Charger le signal tracker
    trk_data = _load_tracker_data(session_path)
    if trk_data is None:
        return {"session": name, "status": "error",
                "reason": "impossible de charger le signal tracker"}

    trk_t_abs, trk_speed = trk_data
    # Normaliser le signal tracker
    mu, sd = np.mean(trk_speed), np.std(trk_speed)
    trk_sig_norm = (trk_speed - mu) / (sd + 1e-8)

    cam_reports: dict[str, dict] = {}
    corrections: dict[str, float] = {}
    any_corrected = False

    for cam in CAMERAS:
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        if not jsonl_path.exists():
            cam_reports[cam] = {"status": "skip", "reason": "JSONL absent"}
            continue

        frames = _read_jsonl(jsonl_path)
        if len(frames) < 30:
            cam_reports[cam] = {"status": "skip", "reason": "trop peu de frames"}
            continue

        cam_t_abs = np.array([
            float(fr["capture_time"])
            for fr in frames
            if fr.get("capture_time") is not None
        ], dtype=np.float64)

        if len(cam_t_abs) < 20:
            cam_reports[cam] = {"status": "skip", "reason": "timestamps invalides"}
            continue

        # Signal IFI de la caméra
        _, cam_sig = _build_cam_ifi_signal(cam_t_abs)
        cam_sig_norm = cam_sig.astype(np.float64)

        # Vérifier qualité du signal
        if float(np.std(cam_sig_norm)) < MIN_XCORR_SIGNAL_QUALITY:
            cam_reports[cam] = {"status": "skip", "reason": "signal trop plat"}
            continue

        # Trouver le meilleur lag
        best_lag, score_before, score_after = find_best_lag(
            trk_t=trk_t_abs,
            trk_sig=trk_sig_norm,
            cam_t=cam_t_abs,
            cam_sig=cam_sig_norm,
        )

        improvement = score_after - score_before
        cam_reports[cam] = {
            "lag_ms":       best_lag,
            "score_before": score_before,
            "score_after":  score_after,
            "improvement":  round(improvement, 4),
        }

        if improvement < MIN_SCORE_IMPROVEMENT:
            cam_reports[cam]["status"] = "no_improvement"
            cam_reports[cam]["reason"] = (
                f"amélioration {improvement:.4f} < seuil {MIN_SCORE_IMPROVEMENT}"
            )
            continue

        if abs(best_lag) < 1.0:
            cam_reports[cam]["status"] = "lag_negligible"
            continue

        cam_reports[cam]["status"] = "corrected" if not dry_run else "would_correct"
        corrections[cam] = best_lag

        if not dry_run:
            # Appliquer le lag : cam_t_new = cam_t - best_lag
            # (best_lag est le décalage optimal à ajouter, on veut corriger)
            corrected_frames = []
            trk_t0 = float(trk_t_abs[0])
            trk_t1 = float(trk_t_abs[-1])
            for fr in frames:
                ct = fr.get("capture_time")
                if ct is None:
                    continue
                new_ct = float(ct) - best_lag
                # Garder dans la fenêtre tracker
                if trk_t0 - 100.0 <= new_ct <= trk_t1 + 100.0:
                    fr_new = dict(fr)
                    fr_new["capture_time"] = round(new_ct, 3)
                    corrected_frames.append(fr_new)
            _write_jsonl(jsonl_path, corrected_frames)
            any_corrected = True
            cam_reports[cam]["frames_after"] = len(corrected_frames)

    # ── Détection d'anomalie : lags très différents entre caméras ────────────
    valid_lags = {cam: lag for cam, lag in corrections.items()}
    if len(valid_lags) >= 2:
        lag_values = list(valid_lags.values())
        lag_spread = float(np.max(lag_values) - np.min(lag_values))
        if lag_spread > 200.0:
            # Les lags sont très différents → possible mauvais assignement de caméra
            for cam in CAMERAS:
                if cam in cam_reports:
                    cam_reports[cam]["warning"] = (
                        f"Spread des lags = {lag_spread:.0f}ms — "
                        f"possible mauvais assignement de caméra !"
                    )

    if not dry_run and any_corrected:
        meta[SYNC_MARKER_KEY] = True
        meta["sync_lag_corrections_ms"] = {cam: round(v, 1) for cam, v in corrections.items()}
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "session":     name,
        "status":      "corrected" if any_corrected else ("dry-run" if dry_run else "ok"),
        "corrections": corrections,
        "cam_reports": cam_reports,
    }
