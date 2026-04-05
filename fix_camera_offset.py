#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_camera_offset.py — Recale et tronque les capture_time des caméras sur la fenêtre du tracker.

Problème dans les sessions désynchronisées :
    Les caméras démarrent 6 à 23 secondes AVANT le tracker.
    Les capture_time sont des timestamps epoch ms corrects, mais décalés.
    Résultat : la majorité des frames caméra n'ont pas de correspondance tracker
    et la session est inutilisable.

Deux opérations effectuées :

  1. RECALAGE (offset)
     offset_ms = cam[0].capture_time - tracker[0].timestamp_ns/1e6
     capture_time_corrigé = capture_time_original - round(offset_ms)
     => cam[0] s'aligne sur tracker[0]

  2. TRONCATURE
     On supprime du JSONL toutes les frames dont le capture_time corrigé
     est en dehors de [tracker_t0, tracker_t1].
     Ces frames n'ont pas de correspondance proprioceptive et corrompent
     la visualisation / l'export LeRobot.
     La vidéo mp4 n'est PAS modifiée : les frames supprimées du JSONL
     sont simplement ignorées par la pipeline (seek par index).

Marker anti-double-application :
    metadata.json reçoit "camera_tracker_sync_applied": true après correction.
    Utiliser --force pour ré-appliquer.

Usage :
    python3 fix_camera_offset.py <session_path>
    python3 fix_camera_offset.py <root_dir> --batch
    python3 fix_camera_offset.py <root_dir> --batch --dry-run
    python3 fix_camera_offset.py <session_path> --force
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

CAMERAS = ["head", "left", "right"]
MARKER_KEY      = "camera_tracker_sync_applied"
TRIM_MARKER_KEY = "stream_trim_applied"

# En dessous de ce seuil, l'offset est considéré négligeable (bruit d'horloge).
# Au-dessus, la correction est appliquée.
OFFSET_THRESHOLD_MS = 30.0

# Validation par cross-corrélation gripper visuel/capteur
GRIPPER_WINDOW_S   = 5.0    # durée de la fenêtre d'analyse (s)
GRIPPER_LAG_MAX_MS = 150.0  # lag max acceptable entre vidéo et capteur gripper
GRIPPER_WARN_MS    = 50.0   # seuil d'avertissement (lag non nul mais acceptable)


# ──────────────────────────────────────────────────────────────────────────────
# Mesure du lag gripper visuel/capteur par cross-corrélation
# ──────────────────────────────────────────────────────────────────────────────

def _measure_gripper_visual_lag(
    session_path: Path,
    side: str,
    window_s: float = GRIPPER_WINDOW_S,
    lag_max_ms: float = GRIPPER_LAG_MAX_MS,
) -> dict:
    """
    Mesure le décalage temporel entre les timestamps vidéo et les données capteur
    gripper en cross-corrélant le signal d'ouverture visuel (extrait par analyse
    de pixels) avec le signal capteur (CSV) sur une fenêtre de quelques secondes.

    Algorithme :
      1. Charger les N premières secondes du JSONL caméra (capture_time en ms)
      2. Charger le CSV gripper (timestamp_ns → ms, opening_mm)
      3. Pour chaque frame dans la fenêtre :
         a. Lire la frame depuis la vidéo MP4
         b. Extraire le gap visuel avec la méthode inner-edge percentile
      4. Lisser le signal visuel (moyenne glissante sur 5 frames)
      5. Interpoler le signal capteur aux instants des frames
      6. Cross-corrélation normalisée entre les deux signaux
      7. Le pic de corrélation donne le lag optimal (en ms)

    Retourne un dict avec :
      lag_ms         : décalage optimal (positif = vidéo en avance sur capteur)
      xcorr_peak     : valeur du pic de corrélation (0–1)
      n_frames       : nombre de frames analysées
      signal_std_mm  : std du signal capteur (indicateur de mouvement)
      valid          : True si la mesure est fiable
      error          : message d'erreur si valid=False
    """
    result = {
        "side": side,
        "lag_ms": 0.0,
        "xcorr_peak": 0.0,
        "n_frames": 0,
        "signal_std_mm": 0.0,
        "valid": False,
        "error": "",
    }

    # ── Vérification des dépendances ──────────────────────────────────────────
    try:
        import cv2
    except ImportError:
        result["error"] = "cv2 non disponible"
        return result

    try:
        import numpy as np
        from scipy.signal import correlate, savgol_filter
        import pandas as pd
    except ImportError as e:
        result["error"] = f"dépendance manquante : {e}"
        return result

    # ── Chargement JSONL caméra ───────────────────────────────────────────────
    jsonl_path = session_path / "videos" / f"{side}.jsonl"
    mp4_path   = session_path / "videos" / f"{side}.mp4"

    if not jsonl_path.exists() or not mp4_path.exists():
        result["error"] = f"fichiers vidéo manquants pour le côté {side}"
        return result

    # Lire tous les capture_time du JSONL
    frame_times_ms = []
    try:
        raw = jsonl_path.read_bytes()
        for line in raw.split(b"\n"):
            line = line.strip().rstrip(b"\r")
            if not line:
                continue
            try:
                obj = json.loads(line)
                ct = obj.get("capture_time")
                if ct is not None:
                    frame_times_ms.append(float(ct))
            except Exception:
                continue
    except Exception as e:
        result["error"] = f"lecture JSONL échouée : {e}"
        return result

    if len(frame_times_ms) < 20:
        result["error"] = "pas assez de frames dans le JSONL"
        return result

    frame_times_ms = np.array(frame_times_ms)
    t0 = frame_times_ms[0]
    t1 = t0 + window_s * 1000.0

    # Sélectionner les frames dans la fenêtre
    window_mask = frame_times_ms <= t1
    window_indices = np.where(window_mask)[0]

    if len(window_indices) < 15:
        result["error"] = "fenêtre trop courte (< 15 frames)"
        return result

    # ── Chargement CSV gripper ────────────────────────────────────────────────
    csv_name = "gripper_left_data.csv" if side == "left" else "gripper_right_data.csv"
    csv_path = session_path / csv_name

    if not csv_path.exists():
        result["error"] = f"{csv_name} introuvable"
        return result

    try:
        df = pd.read_csv(csv_path)
        if "timestamp_ns" in df.columns:
            sensor_t_ms = df["timestamp_ns"].astype(float).to_numpy() / 1e6
        elif "t_ms" in df.columns:
            sensor_t_ms = df["t_ms"].astype(float).to_numpy()
        else:
            result["error"] = "colonne de temps introuvable dans le CSV gripper"
            return result

        if "opening_mm" not in df.columns:
            result["error"] = "colonne opening_mm introuvable dans le CSV"
            return result

        sensor_opening = df["opening_mm"].astype(float).to_numpy()
        valid_mask = np.isfinite(sensor_t_ms) & np.isfinite(sensor_opening)
        sensor_t_ms  = sensor_t_ms[valid_mask]
        sensor_opening = sensor_opening[valid_mask]
    except Exception as e:
        result["error"] = f"lecture CSV échouée : {e}"
        return result

    if len(sensor_t_ms) < 20:
        result["error"] = "CSV gripper insuffisant"
        return result

    # Vérifier que les timestamps capteur couvrent la fenêtre vidéo
    cam_t_window = frame_times_ms[window_indices]
    if sensor_t_ms[0] > cam_t_window[0] + 200 or sensor_t_ms[-1] < cam_t_window[-1] - 200:
        result["error"] = "timestamps capteur ne couvrent pas la fenêtre vidéo"
        return result

    # Interpoler le signal capteur aux instants des frames
    from scipy.interpolate import interp1d
    interp_fn = interp1d(sensor_t_ms, sensor_opening, kind="linear",
                         bounds_error=False, fill_value=np.nan)
    sensor_at_frames = interp_fn(cam_t_window)

    # Vérifier qu'il y a assez de mouvement (sinon la xcorr est non significative)
    finite_mask = np.isfinite(sensor_at_frames)
    if finite_mask.sum() < 15:
        result["error"] = "pas assez de chevauchement temporal capteur/vidéo"
        return result

    signal_std = float(np.nanstd(sensor_at_frames))
    result["signal_std_mm"] = round(signal_std, 2)
    if signal_std < 1.5:
        result["error"] = f"signal capteur trop statique (std={signal_std:.1f}mm < 1.5mm)"
        return result

    # ── Extraction du gap visuel par frame ───────────────────────────────────
    # Paramètres de segmentation (identiques à gripper_frame_sync.py)
    ARM_Y_TOP     = 400
    DARK_THR      = 40
    P_LEFT_INNER  = 70
    P_RIGHT_INNER = 25
    DENSITY_MIN   = 0.5
    MIN_DARK_COLS = 4

    cap = cv2.VideoCapture(str(mp4_path))
    visual_gaps = []
    valid_frames = []

    for idx in window_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            visual_gaps.append(np.nan)
            valid_frames.append(False)
            continue

        # Extraction inner-edge percentile
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        arm  = gray[ARM_Y_TOP:, :]
        if arm.shape[0] == 0 or arm.shape[1] < 20:
            visual_gaps.append(np.nan)
            valid_frames.append(False)
            continue

        half = arm.shape[1] // 2
        col_dark = (arm < DARK_THR).sum(axis=0).astype(np.float32)
        mean_dark = col_dark.mean()
        if mean_dark < 0.5:
            visual_gaps.append(np.nan)
            valid_frames.append(False)
            continue

        threshold = mean_dark * DENSITY_MIN
        left_dark_cols  = np.where(col_dark[:half] > threshold)[0]
        right_dark_cols = np.where(col_dark[half:] > threshold)[0]

        if len(left_dark_cols) < MIN_DARK_COLS or len(right_dark_cols) < MIN_DARK_COLS:
            visual_gaps.append(np.nan)
            valid_frames.append(False)
            continue

        left_inner  = float(np.percentile(left_dark_cols, P_LEFT_INNER))
        right_inner = float(np.percentile(right_dark_cols, P_RIGHT_INNER)) + half
        gap = right_inner - left_inner

        if gap < 0 or gap > arm.shape[1]:
            visual_gaps.append(np.nan)
            valid_frames.append(False)
            continue

        visual_gaps.append(gap)
        valid_frames.append(True)

    cap.release()

    visual_gaps  = np.array(visual_gaps, dtype=np.float64)
    valid_frames = np.array(valid_frames)

    n_valid = valid_frames.sum()
    result["n_frames"] = int(n_valid)

    if n_valid < 15:
        result["error"] = f"seulement {n_valid} frames valides (< 15)"
        return result

    # ── Lissage du signal visuel ──────────────────────────────────────────────
    # Remplacer les NaN par interpolation linéaire avant le lissage
    x_all = np.arange(len(visual_gaps))
    valid_idx = np.where(valid_frames)[0]
    from scipy.interpolate import interp1d as interp1d_local
    gap_interp_fn = interp1d_local(valid_idx, visual_gaps[valid_idx],
                                   kind="linear", bounds_error=False,
                                   fill_value=(visual_gaps[valid_idx[0]],
                                               visual_gaps[valid_idx[-1]]))
    visual_gaps_filled = gap_interp_fn(x_all)

    # Savitzky-Golay si assez de points
    sg_window = min(15, n_valid // 3 * 2 + 1)
    sg_window = max(5, sg_window if sg_window % 2 == 1 else sg_window - 1)
    try:
        visual_smooth = savgol_filter(visual_gaps_filled, sg_window, 3)
    except Exception:
        visual_smooth = visual_gaps_filled

    # ── Cross-corrélation normalisée ──────────────────────────────────────────
    # Aligner les deux signaux sur les frames où les deux sont valides
    both_valid = finite_mask & valid_frames
    if both_valid.sum() < 15:
        result["error"] = "pas assez de frames valides pour les deux signaux"
        return result

    sig_visual = visual_smooth[both_valid]
    sig_sensor = sensor_at_frames[both_valid]
    t_both_ms  = cam_t_window[both_valid]

    # Normaliser (centrer + réduire)
    sig_v = (sig_visual - sig_visual.mean()) / (sig_visual.std() + 1e-9)
    sig_s = (sig_sensor - sig_sensor.mean()) / (sig_sensor.std() + 1e-9)

    # Xcorr avec restriction à ±lag_max_ms
    fps_est = (len(window_indices) / window_s) if window_s > 0 else 30.0
    lag_max_frames = max(3, int(lag_max_ms / 1000.0 * fps_est))

    xcorr = correlate(sig_v, sig_s, mode="full")
    xcorr /= (len(sig_v) + 1e-9)
    lags = np.arange(-(len(sig_s) - 1), len(sig_v))

    # Restreindre aux lags acceptables
    mask_lag = np.abs(lags) <= lag_max_frames
    xcorr_restricted = xcorr.copy()
    xcorr_restricted[~mask_lag] = -1.0

    best_lag_frames = int(lags[np.argmax(xcorr_restricted)])
    xcorr_peak = float(xcorr[np.argmax(xcorr_restricted)])

    # Convertir en ms (lag moyen entre frames consécutives)
    if len(t_both_ms) > 1:
        mean_dt_ms = float(np.median(np.diff(t_both_ms)))
    else:
        mean_dt_ms = 1000.0 / fps_est

    lag_ms = best_lag_frames * mean_dt_ms

    result["lag_ms"]     = round(lag_ms, 1)
    result["xcorr_peak"] = round(xcorr_peak, 3)
    result["valid"]      = True

    return result


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        raw = f.read()
    frames = []
    for part in raw.split(b"\r\n"):
        part = part.strip()
        if len(part) > 5:
            try:
                frames.append(json.loads(part))
            except json.JSONDecodeError:
                pass
    return frames


def write_jsonl(path: Path, frames: list[dict]) -> None:
    lines = [json.dumps(frame, separators=(",", ":")) + "\r\n" for frame in frames]
    with open(path, "wb") as f:
        f.write("".join(lines).encode("utf-8"))


def read_tracker_window_ms(session_path: Path) -> tuple[float, float] | tuple[None, None]:
    """
    Retourne (tracker_t0_ms, tracker_t1_ms) depuis tracker_positions.csv.
    Utilise timestamp_ns (epoch nanoseconds).
    """
    tracker_path = session_path / "tracker_positions.csv"
    if not tracker_path.exists():
        return None, None

    t_first = None
    t_last = None
    with open(tracker_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ns_str = row.get("timestamp_ns", "").strip()
            if not ns_str:
                continue
            try:
                t_ms = int(ns_str) / 1_000_000
            except ValueError:
                continue
            if t_first is None:
                t_first = t_ms
            t_last = t_ms

    return t_first, t_last


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def fix_session(session_path: Path, dry_run: bool = False, force: bool = False) -> dict:
    """
    Corrige une session :
      - recale les capture_time des caméras sur tracker[0]
      - tronque le JSONL à la fenêtre [tracker_t0, tracker_t1]

    Retourne un dict rapport.
    """
    session_name = session_path.name
    meta_path = session_path / "metadata.json"

    if not meta_path.exists():
        return {"session": session_name, "status": "skipped", "reason": "metadata.json absent"}

    with open(meta_path, "rb") as f:
        meta = json.loads(f.read())

    if not force and meta.get(MARKER_KEY):
        return {
            "session": session_name,
            "status": "skipped",
            "reason": f"déjà corrigée ({MARKER_KEY}=true)",
        }

    trk_t0, trk_t1 = read_tracker_window_ms(session_path)
    if trk_t0 is None:
        return {
            "session": session_name,
            "status": "error",
            "reason": "tracker_positions.csv introuvable ou sans timestamp_ns valide",
        }

    # Calcul des offsets par caméra
    offsets: dict[str, float] = {}
    cam_frames: dict[str, list[dict]] = {}

    for cam in CAMERAS:
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        if not jsonl_path.exists():
            continue
        frames = read_jsonl(jsonl_path)
        if not frames:
            continue
        cam_frames[cam] = frames
        offsets[cam] = frames[0]["capture_time"] - trk_t0

    if not offsets:
        return {"session": session_name, "status": "error", "reason": "aucun fichier .jsonl trouvé"}

    max_offset = max(abs(v) for v in offsets.values())
    if max_offset < OFFSET_THRESHOLD_MS and not force:
        return {
            "session": session_name,
            "status": "ok",
            "reason": (
                f"offset max={max_offset:.1f} ms < seuil {OFFSET_THRESHOLD_MS} ms — "
                f"caméras déjà alignées sur le tracker"
            ),
            "offsets_ms": offsets,
            "tracker_t0_ms": trk_t0,
            "tracker_t1_ms": trk_t1,
        }

    # ── Validation par cross-corrélation gripper visuel/capteur ──────────────
    # Mesure le lag entre les timestamps vidéo et le capteur gripper sur 5s.
    # Cela confirme (ou infirme) que l'offset tracker est cohérent avec
    # l'alignement gripper visuel, signal indépendant du tracker.
    gripper_lag_results = {}
    for side in ("left", "right"):
        lag_result = _measure_gripper_visual_lag(session_path, side)
        if lag_result["valid"]:
            gripper_lag_results[side] = lag_result

    # Évaluer la cohérence entre offset tracker et lag gripper visuel
    gripper_validation = {}
    for side, lag_r in gripper_lag_results.items():
        lag_ms = lag_r["lag_ms"]
        cam_side = side  # "left" ou "right" correspond à la caméra du même nom
        tracker_offset = offsets.get(cam_side, 0.0)

        # Le lag xcorr mesure combien la vidéo est EN AVANCE sur le capteur.
        # Après correction tracker (offset_ms soustrait), le résidu gripper
        # devrait être proche de zéro.
        residual_ms = lag_ms  # Si tracker_offset bien appliqué, lag_ms ≈ 0

        coherent = abs(residual_ms) < GRIPPER_LAG_MAX_MS
        gripper_validation[side] = {
            "lag_ms":       round(lag_ms, 1),
            "xcorr_peak":   lag_r["xcorr_peak"],
            "n_frames":     lag_r["n_frames"],
            "signal_std_mm": lag_r["signal_std_mm"],
            "coherent":     coherent,
            "warning":      (abs(residual_ms) >= GRIPPER_WARN_MS),
        }

    report = {
        "session": session_name,
        "status": "dry-run" if dry_run else "corrected",
        "tracker_t0_ms": trk_t0,
        "tracker_t1_ms": trk_t1,
        "tracker_duration_s": (trk_t1 - trk_t0) / 1000,
        "offsets_ms": offsets,
        "cameras_fixed": [],
        "gripper_lag_validation": gripper_validation,
    }

    for cam, offset_ms in offsets.items():
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        frames = cam_frames[cam]
        offset_int = round(offset_ms)

        # 1. Recalage
        recaled = [
            {**frame, "capture_time": frame["capture_time"] - offset_int}
            for frame in frames
        ]

        # 2. Troncature : ne garder que les frames dans la fenêtre tracker
        truncated = [
            frame for frame in recaled
            if trk_t0 <= frame["capture_time"] <= trk_t1
        ]

        n_removed = len(recaled) - len(truncated)
        overlap_s = (truncated[-1]["capture_time"] - truncated[0]["capture_time"]) / 1000 if truncated else 0

        report["cameras_fixed"].append({
            "camera": cam,
            "offset_ms": offset_ms,
            "offset_applied_ms": offset_int,
            "frames_original": len(frames),
            "frames_kept": len(truncated),
            "frames_removed": n_removed,
            "overlap_s": round(overlap_s, 2),
            "first_original_ms": frames[0]["capture_time"],
            "first_corrected_ms": truncated[0]["capture_time"] if truncated else None,
            "last_corrected_ms": truncated[-1]["capture_time"] if truncated else None,
        })

        if not dry_run and truncated:
            bak_path = jsonl_path.with_suffix(".jsonl.bak")
            if not bak_path.exists():
                shutil.copy2(jsonl_path, bak_path)
            write_jsonl(jsonl_path, truncated)

    if not dry_run:
        meta[MARKER_KEY] = True
        meta["camera_tracker_sync_offsets_ms"] = offsets
        meta["camera_tracker_sync_tracker_t0_ms"] = trk_t0
        meta["camera_tracker_sync_tracker_t1_ms"] = trk_t1
        if gripper_validation:
            meta["gripper_visual_lag_validation"] = gripper_validation
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Trim — rogne le début de chaque flux pour les aligner sur le plus tardif
# ──────────────────────────────────────────────────────────────────────────────

def _read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    """Lit un CSV, retourne (fieldnames, rows)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []) if not rows else list(rows[0].keys()), rows


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def trim_session(session_path: Path, dry_run: bool = False, force: bool = False) -> dict:
    """
    Aligne temporellement tous les flux d'une session en coupant le début
    de chaque flux jusqu'au point de démarrage commun (le flux qui démarre
    le plus tard définit t=0).

    Flux traités :
      - tracker_positions.csv  (colonne timestamp_ns)
      - gripper_left_data.csv  (colonne timestamp_ns)
      - gripper_right_data.csv (colonne timestamp_ns)
      - videos/*.jsonl         (champ capture_time en ms)

    La vidéo .mp4 n'est PAS modifiée — les frames JSONL supprimées
    sont ignorées par la pipeline (seek par index).

    Retourne un dict rapport avec les quantités rognées par flux.
    """
    session_name = session_path.name
    meta_path    = session_path / "metadata.json"

    if not meta_path.exists():
        return {"session": session_name, "status": "error", "reason": "metadata.json absent"}

    with open(meta_path, "rb") as f:
        meta = json.loads(f.read())

    if not force and meta.get(TRIM_MARKER_KEY):
        return {"session": session_name, "status": "skipped",
                "reason": f"déjà rognée ({TRIM_MARKER_KEY}=true)"}

    # ── Collecter les t0 de chaque flux ─────────────────────────────────────
    t0s: dict[str, float] = {}

    # Tracker
    trk_path = session_path / "tracker_positions.csv"
    if trk_path.exists():
        with open(trk_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ns = row.get("timestamp_ns", "").strip()
                if ns:
                    t0s["tracker"] = int(ns) / 1_000_000
                    break

    # Gripper CSVs
    for side in ("left", "right"):
        grp_path = session_path / f"gripper_{side}_data.csv"
        if grp_path.exists():
            with open(grp_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ns = row.get("timestamp_ns", "").strip()
                    if ns:
                        t0s[f"gripper_{side}"] = int(ns) / 1_000_000
                        break

    # JSONL caméras
    for cam in CAMERAS:
        jf = session_path / "videos" / f"{cam}.jsonl"
        if not jf.exists():
            continue
        for line in jf.read_bytes().split(b"\r\n"):
            line = line.strip()
            if len(line) > 5:
                try:
                    t0s[f"camera_{cam}"] = json.loads(line)["capture_time"]
                    break
                except (json.JSONDecodeError, KeyError):
                    pass

    if len(t0s) < 2:
        return {"session": session_name, "status": "error",
                "reason": f"insuffisant de flux détectés ({list(t0s.keys())})"}

    # ── Point de synchronisation = max(tous les t0) ──────────────────────────
    sync_t0 = max(t0s.values())
    trims    = {k: sync_t0 - v for k, v in t0s.items()}  # ms à couper par flux

    # Vérifier que le trim est non-trivial (> 5ms sur au moins un flux)
    if max(trims.values()) < 5.0 and not force:
        return {"session": session_name, "status": "ok",
                "reason": f"tous les flux démarrent dans les 5ms — aucun trim nécessaire",
                "sync_t0_ms": sync_t0, "trims_ms": trims}

    report = {
        "session":    session_name,
        "status":     "dry-run" if dry_run else "trimmed",
        "sync_t0_ms": sync_t0,
        "trims_ms":   trims,
        "streams":    {},
    }

    if dry_run:
        return report

    # ── Tracker CSV ──────────────────────────────────────────────────────────
    if trk_path.exists() and trims.get("tracker", 0) > 0:
        fieldnames, rows = _read_csv_rows(trk_path)
        before = len(rows)
        rows = [r for r in rows
                if int(r.get("timestamp_ns", "0") or "0") / 1_000_000 >= sync_t0]
        kept = len(rows)
        bak = trk_path.with_suffix(".csv.bak")
        if not bak.exists():
            shutil.copy2(trk_path, bak)
        _write_csv_rows(trk_path, fieldnames, rows)
        report["streams"]["tracker"] = {"rows_before": before, "rows_kept": kept,
                                         "rows_removed": before - kept,
                                         "trim_ms": round(trims["tracker"], 1)}

    # ── Gripper CSVs ─────────────────────────────────────────────────────────
    for side in ("left", "right"):
        key = f"gripper_{side}"
        grp_path = session_path / f"gripper_{side}_data.csv"
        if not grp_path.exists() or trims.get(key, 0) <= 0:
            continue
        fieldnames, rows = _read_csv_rows(grp_path)
        before = len(rows)
        rows = [r for r in rows
                if int(r.get("timestamp_ns", "0") or "0") / 1_000_000 >= sync_t0]
        kept = len(rows)
        bak = grp_path.with_suffix(".csv.bak")
        if not bak.exists():
            shutil.copy2(grp_path, bak)
        _write_csv_rows(grp_path, fieldnames, rows)
        report["streams"][key] = {"rows_before": before, "rows_kept": kept,
                                   "rows_removed": before - kept,
                                   "trim_ms": round(trims[key], 1)}

    # ── JSONL caméras ────────────────────────────────────────────────────────
    for cam in CAMERAS:
        key = f"camera_{cam}"
        jf  = session_path / "videos" / f"{cam}.jsonl"
        if not jf.exists() or trims.get(key, 0) <= 0:
            continue
        frames = read_jsonl(jf)
        before = len(frames)
        frames = [fr for fr in frames if fr.get("capture_time", 0) >= sync_t0]
        kept   = len(frames)
        bak    = jf.with_suffix(".jsonl.bak")
        if not bak.exists():
            shutil.copy2(jf, bak)
        write_jsonl(jf, frames)
        report["streams"][key] = {"frames_before": before, "frames_kept": kept,
                                   "frames_removed": before - kept,
                                   "trim_ms": round(trims[key], 1)}

    # ── Marquer et persister ─────────────────────────────────────────────────
    meta[TRIM_MARKER_KEY]             = True
    meta["stream_trim_sync_t0_ms"]    = sync_t0
    meta["stream_trim_trims_ms"]      = {k: round(v, 1) for k, v in trims.items()}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI output
# ──────────────────────────────────────────────────────────────────────────────

def print_report(report: dict) -> None:
    status = report["status"]
    session = report["session"]
    reason = report.get("reason", "")

    if status in ("skipped", "ok"):
        print(f"  [{status.upper()}] {session} — {reason}")
        return
    if status == "error":
        print(f"  [ERROR] {session} — {reason}")
        return

    trk_dur = report.get("tracker_duration_s", 0)
    print(f"  [{status.upper()}] {session}  (tracker window={trk_dur:.1f}s)")
    for c in report.get("cameras_fixed", []):
        cam     = c["camera"]
        off     = c["offset_ms"]
        kept    = c["frames_kept"]
        removed = c["frames_removed"]
        total   = c["frames_original"]
        overlap = c["overlap_s"]
        pct     = kept / total * 100 if total else 0
        print(
            f"    {cam}: offset={off:+.0f}ms  "
            f"kept={kept}/{total} ({pct:.0f}%)  "
            f"removed={removed}  overlap={overlap:.1f}s"
        )

    glv = report.get("gripper_lag_validation", {})
    if glv:
        print("    ── Validation gripper visuel/capteur ──")
        for side, v in glv.items():
            lag     = v["lag_ms"]
            peak    = v["xcorr_peak"]
            n       = v["n_frames"]
            std     = v["signal_std_mm"]
            ok_flag = "OK" if v["coherent"] else "WARN"
            warn_flag = " ⚠" if v["warning"] else ""
            print(
                f"    {side}: lag={lag:+.0f}ms  xcorr={peak:.2f}  "
                f"frames={n}  signal_std={std:.1f}mm  [{ok_flag}]{warn_flag}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recale et tronque les capture_time caméras sur la fenêtre du tracker."
    )
    parser.add_argument("path", help="Chemin vers la session ou le dossier racine")
    parser.add_argument("--batch", action="store_true", help="Traiter toutes les sessions du dossier")
    parser.add_argument("--dry-run", action="store_true", help="Afficher sans modifier les fichiers")
    parser.add_argument("--force", action="store_true", help="Forcer même si déjà corrigée")
    args = parser.parse_args()

    root = Path(args.path)

    if args.batch:
        sessions = sorted(
            p.parent for p in root.rglob("metadata.json")
            if (p.parent / "videos").exists()
        )
        if not sessions:
            print(f"Aucune session trouvée dans {root}")
            sys.exit(1)
        print(f"{'[DRY-RUN] ' if args.dry_run else ''}Traitement de {len(sessions)} session(s)...\n")
        for session_path in sessions:
            report = fix_session(session_path, dry_run=args.dry_run, force=args.force)
            print_report(report)
    else:
        if not root.exists():
            print(f"Chemin introuvable : {root}")
            sys.exit(1)
        report = fix_session(root, dry_run=args.dry_run, force=args.force)
        print_report(report)

    print()
    if args.dry_run:
        print("Mode dry-run : aucun fichier modifié.")


if __name__ == "__main__":
    main()
