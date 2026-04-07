#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_camera_gaps.py — Correction des gaps dans le flux caméra.

Problème :
    Les fichiers JSONL peuvent contenir des interruptions de la capture vidéo :
      - Décrochages matériels (câble USB, overrun buffer)
      - Redémarrages du processus de capture
      - Frames corrompues (capture_time manquant)

    Ces gaps génèrent une erreur camera_continuity dans check.py si N_gaps > seuil.

Stratégies disponibles (choisies automatiquement selon le type de gap) :

  1. TRIM_TO_SEGMENT (prioritaire) :
     Trouver le segment continu le plus long qui couvre la fenêtre tracker.
     Tronquer le JSONL à ce segment.
     Avantage : simple, robuste, conserve les meilleures données.

  2. GAP_INTERPOLATION (complément) :
     Pour les petits gaps (< MAX_INTERP_GAP_MS), insérer des frames synthétiques
     avec des timestamps interpolés linéairement.
     Ces frames n'ont pas de vidéo correspondante → ignorées par la pipeline vidéo.
     Avantage : préserve la continuité temporelle du JSONL pour la synchronisation.

  3. DROP_SHORT_SEGMENTS :
     Si un segment est trop court (< MIN_SEGMENT_DURATION_MS), le supprimer.

Décision :
  - Si le segment principal couvre ≥ MIN_COVERAGE_AFTER_TRIM de la fenêtre tracker
    → TRIM_TO_SEGMENT
  - Sinon → non récupérable (signal trop fragmenté)

Usage :
    from fix.fix_camera_gaps import fix_camera_gaps
    report = fix_camera_gaps(Path("/path/to/session"))
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

CAMERAS                    = ("head", "left", "right")
GAPS_MARKER_KEY            = "camera_gaps_fixed"

# Seuil de détection de gap (ms) — cohérent avec check.py
CAMERA_GAP_THRESHOLD_MS    = 120.0

# Gap en dessous duquel on peut interpoler des frames synthétiques
MAX_INTERP_GAP_MS          = 500.0

# Durée minimale d'un segment pour qu'il soit conservé
MIN_SEGMENT_DURATION_MS    = 500.0

# Couverture minimale du segment principal après trim
MIN_COVERAGE_AFTER_TRIM    = 0.60


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


def _load_tracker_window(session_path: Path) -> tuple[Optional[float], Optional[float]]:
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


# ══════════════════════════════════════════════════════════════════════════════
# Analyse des gaps
# ══════════════════════════════════════════════════════════════════════════════

def _find_segments(
    frames: list[dict],
    gap_threshold_ms: float = CAMERA_GAP_THRESHOLD_MS,
) -> list[tuple[int, int, float, float]]:
    """
    Découpe la liste de frames en segments continus.

    Returns:
        Liste de (idx_start, idx_end, t_start_ms, t_end_ms) — indices inclus.
    """
    times = []
    valid_idx = []
    for i, fr in enumerate(frames):
        ct = fr.get("capture_time")
        if ct is not None:
            try:
                times.append(float(ct))
                valid_idx.append(i)
            except (ValueError, TypeError):
                pass

    if not times:
        return []

    segments = []
    seg_start_frame = valid_idx[0]
    seg_start_t     = times[0]
    prev_t          = times[0]
    prev_frame      = valid_idx[0]

    for i in range(1, len(times)):
        dt = times[i] - prev_t
        if dt > gap_threshold_ms:
            # Fin du segment courant
            segments.append((seg_start_frame, prev_frame, seg_start_t, prev_t))
            # Nouveau segment
            seg_start_frame = valid_idx[i]
            seg_start_t     = times[i]
        prev_t     = times[i]
        prev_frame = valid_idx[i]

    # Dernier segment
    segments.append((seg_start_frame, prev_frame, seg_start_t, prev_t))
    return segments


def _best_segment_for_tracker(
    segments: list[tuple[int, int, float, float]],
    trk_t0: float,
    trk_t1: float,
) -> Optional[tuple[int, int, float]]:
    """
    Trouve le segment qui couvre le mieux la fenêtre tracker.

    Returns:
        (idx_start, idx_end, coverage_ratio) ou None
    """
    trk_dur = trk_t1 - trk_t0
    if trk_dur <= 0:
        return None

    best = None
    best_coverage = -1.0

    for seg_start, seg_end, seg_t0, seg_t1 in segments:
        dur = seg_t1 - seg_t0
        if dur < MIN_SEGMENT_DURATION_MS:
            continue
        overlap = max(0.0, min(seg_t1, trk_t1) - max(seg_t0, trk_t0))
        coverage = overlap / (trk_dur + 1e-6)
        if coverage > best_coverage:
            best_coverage = coverage
            best = (seg_start, seg_end, coverage)

    return best


# ══════════════════════════════════════════════════════════════════════════════
# Fix principal
# ══════════════════════════════════════════════════════════════════════════════

def fix_camera_gaps(
    session_path: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Corrige les gaps dans le flux caméra en trimmant aux segments continus.

    Pour chaque caméra :
      1. Détecter les segments continus (gaps > CAMERA_GAP_THRESHOLD_MS)
      2. Trouver le segment qui couvre le mieux la fenêtre tracker
      3. Si coverage ≥ MIN_COVERAGE_AFTER_TRIM → tronquer au segment
      4. Sinon → non récupérable

    Args:
        session_path : chemin vers la session
        dry_run      : mesure uniquement, sans modifier
        force        : re-applique même si déjà corrigé

    Returns:
        dict rapport par caméra
    """
    name = session_path.name
    meta_path = session_path / "metadata.json"

    if not meta_path.exists():
        return {"session": name, "status": "error", "reason": "metadata.json absent"}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"session": name, "status": "error", "reason": f"metadata.json illisible: {e}"}

    if not force and meta.get(GAPS_MARKER_KEY):
        return {"session": name, "status": "skipped", "reason": "déjà corrigé (gaps)"}

    # Fenêtre tracker
    trk_t0, trk_t1 = _load_tracker_window(session_path)
    if trk_t0 is None:
        return {"session": name, "status": "error",
                "reason": "tracker_positions.csv introuvable"}

    trk_dur = (trk_t1 or 0.0) - (trk_t0 or 0.0)

    cam_reports: dict[str, dict] = {}
    any_corrected = False
    all_ok = True

    for cam in CAMERAS:
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        if not jsonl_path.exists():
            cam_reports[cam] = {"status": "skip", "reason": "JSONL absent"}
            continue

        frames = _read_jsonl(jsonl_path)
        if len(frames) < 10:
            cam_reports[cam] = {"status": "skip", "reason": "trop peu de frames"}
            continue

        # Analyser les segments
        segments = _find_segments(frames, CAMERA_GAP_THRESHOLD_MS)
        n_gaps = max(0, len(segments) - 1)

        if n_gaps == 0:
            cam_reports[cam] = {
                "status": "ok",
                "n_gaps": 0,
                "n_frames": len(frames),
            }
            continue

        # Trouver le meilleur segment
        best = _best_segment_for_tracker(segments, trk_t0, trk_t1 or trk_t0 + 60000)
        if best is None:
            cam_reports[cam] = {
                "status": "unrecoverable",
                "n_gaps": n_gaps,
                "reason": "aucun segment suffisamment long",
            }
            all_ok = False
            continue

        idx_start, idx_end, coverage = best

        if coverage < MIN_COVERAGE_AFTER_TRIM:
            cam_reports[cam] = {
                "status": "unrecoverable",
                "n_gaps": n_gaps,
                "coverage": round(coverage, 3),
                "reason": f"couverture après trim = {coverage*100:.0f}% < seuil",
            }
            all_ok = False
            continue

        # Extraire les frames du meilleur segment
        trimmed = frames[idx_start:idx_end + 1]

        # Renuméroter les index
        for i, fr in enumerate(trimmed):
            fr["index"] = i

        cam_reports[cam] = {
            "status":      "corrected" if not dry_run else "would_correct",
            "n_gaps":      n_gaps,
            "frames_before": len(frames),
            "frames_after":  len(trimmed),
            "coverage":    round(coverage, 3),
            "segment_t0":  trimmed[0].get("capture_time"),
            "segment_t1":  trimmed[-1].get("capture_time"),
        }

        if not dry_run:
            _write_jsonl(jsonl_path, trimmed)
            any_corrected = True

    if not dry_run and any_corrected:
        meta[GAPS_MARKER_KEY] = True
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    all_corrected = all(
        r.get("status") in ("corrected", "ok", "skip")
        for r in cam_reports.values()
    )

    return {
        "session":     name,
        "status":      ("corrected" if any_corrected
                        else ("dry-run" if dry_run
                              else ("ok" if all_corrected
                                    else "partial"))),
        "recoverable": all_ok,
        "cam_reports": cam_reports,
    }
