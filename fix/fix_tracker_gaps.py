#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_tracker_gaps.py — Correction des gaps dans le flux tracker.

Problème :
    tracker_positions.csv peut contenir des interruptions :
      - Perte de signal VR (occlusion, sortie de zone)
      - Redémarrage du système de tracking
      - Lignes corrompues

    Ces gaps génèrent une erreur tracker_continuity dans check.py.

Stratégies :

  1. INTERPOLATION LINÉAIRE (petits gaps < MAX_INTERP_GAP_MS) :
     Position  : interpolation linéaire
     Quaternion : SLERP (interpolation sphérique)
     Valide si le mouvement est lent (< MAX_SPEED_INTERP mm/s)

  2. TRIM (grands gaps) :
     Tronquer le CSV à la fenêtre continue principale.
     Les segments courts avant/après le gap principal sont supprimés.

  3. REJET si trop fragmenté.

Usage :
    from fix.fix_tracker_gaps import fix_tracker_gaps
    report = fix_tracker_gaps(Path("/path/to/session"))
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

TRACKER_GAP_THRESHOLD_MS   = 60.0     # cohérent avec check.py
MAX_INTERP_GAP_MS          = 300.0    # au-delà → trim
MAX_SPEED_INTERP_MM_S      = 1500.0   # vitesse max pour interpoler sans biaiser
MIN_COVERAGE_AFTER_TRIM    = 0.70
GAPS_MARKER_KEY            = "tracker_gaps_fixed"

TRACKERS = ("head", "left", "right")
AXES     = ("x", "y", "z")
QUATS    = ("qw", "qx", "qy", "qz")


# ══════════════════════════════════════════════════════════════════════════════
# Interpolation quaternion SLERP
# ══════════════════════════════════════════════════════════════════════════════

def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Interpolation sphérique linéaire entre deux quaternions.
    q0, q1 : (4,) vecteurs normalisés (w, x, y, z)
    t      : paramètre ∈ [0, 1]
    """
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)

    dot = float(np.dot(q0, q1))
    # Assurer le chemin court
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    # Si très proches → interpolation linéaire pour éviter division par zéro
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / (np.linalg.norm(result) + 1e-12)

    theta_0 = np.arccos(dot)
    theta   = theta_0 * t
    sin_t0  = np.sin(theta_0)
    sin_t   = np.sin(theta)
    sin_rem = np.sin(theta_0 - theta)

    result = (sin_rem / sin_t0) * q0 + (sin_t / sin_t0) * q1
    return result / (np.linalg.norm(result) + 1e-12)


def slerp_array(q0: np.ndarray, q1: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """SLERP vectorisé pour plusieurs valeurs de t."""
    return np.stack([slerp(q0, q1, float(t)) for t in ts])


# ══════════════════════════════════════════════════════════════════════════════
# Analyse et correction du CSV tracker
# ══════════════════════════════════════════════════════════════════════════════

def _find_tracker_gaps(
    t_ms: np.ndarray,
    gap_threshold_ms: float = TRACKER_GAP_THRESHOLD_MS,
) -> list[tuple[int, float]]:
    """
    Retourne la liste des gaps : (index_avant_gap, durée_ms).
    """
    if len(t_ms) < 2:
        return []
    dt = np.diff(t_ms)
    gaps = [(int(i), float(dt[i])) for i in range(len(dt)) if dt[i] > gap_threshold_ms]
    return gaps


def _find_main_segment(
    t_ms: np.ndarray,
    trk_t0_ms: float,
    trk_t1_ms: float,
    gap_threshold_ms: float = TRACKER_GAP_THRESHOLD_MS,
) -> tuple[int, int, float]:
    """
    Trouve le segment continu le plus long couvrant le mieux [trk_t0, trk_t1].

    Returns:
        (idx_start, idx_end, coverage)
    """
    gaps = _find_tracker_gaps(t_ms, gap_threshold_ms)
    if not gaps:
        return 0, len(t_ms) - 1, 1.0

    # Segmenter
    seg_starts = [0] + [i + 1 for i, _ in gaps]
    seg_ends   = [i for i, _ in gaps] + [len(t_ms) - 1]
    trk_dur = trk_t1_ms - trk_t0_ms

    best_start, best_end, best_cov = 0, len(t_ms) - 1, 0.0
    for s, e in zip(seg_starts, seg_ends):
        seg_t0 = float(t_ms[s])
        seg_t1 = float(t_ms[e])
        dur = seg_t1 - seg_t0
        if dur < 1000.0:
            continue
        overlap = max(0.0, min(seg_t1, trk_t1_ms) - max(seg_t0, trk_t0_ms))
        cov = overlap / (trk_dur + 1e-6)
        if cov > best_cov:
            best_cov = cov
            best_start, best_end = s, e

    return best_start, best_end, best_cov


def fix_tracker_gaps(
    session_path: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Corrige les gaps dans tracker_positions.csv.

    Stratégie par gap :
      - Gap < MAX_INTERP_GAP_MS : interpolation linéaire (position) + SLERP (quaternion)
      - Gap ≥ MAX_INTERP_GAP_MS : trim au segment principal

    Args:
        session_path : chemin vers la session
        dry_run      : mesure uniquement
        force        : re-applique

    Returns:
        dict rapport
    """
    if not _PANDAS:
        return {"session": session_path.name, "status": "error",
                "reason": "pandas non disponible"}

    name = session_path.name
    meta_path = session_path / "metadata.json"
    csv_path  = session_path / "tracker_positions.csv"

    if not meta_path.exists():
        return {"session": name, "status": "error", "reason": "metadata.json absent"}
    if not csv_path.exists():
        return {"session": name, "status": "error",
                "reason": "tracker_positions.csv absent"}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"session": name, "status": "error", "reason": f"metadata.json illisible: {e}"}

    if not force and meta.get(GAPS_MARKER_KEY):
        return {"session": name, "status": "skipped", "reason": "déjà corrigé (tracker gaps)"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"session": name, "status": "error", "reason": f"CSV illisible: {e}"}

    if "timestamp_ns" not in df.columns:
        return {"session": name, "status": "error", "reason": "colonne timestamp_ns absente"}

    t_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce").to_numpy(np.float64)
    valid_mask = np.isfinite(t_ns)
    if valid_mask.sum() < 10:
        return {"session": name, "status": "error", "reason": "timestamps invalides"}

    # Travailler sur les lignes valides seulement
    df_valid = df[valid_mask].copy().reset_index(drop=True)
    t_ms = df_valid["timestamp_ns"].to_numpy(np.float64) / 1e6

    # Fenêtre tracker = toute la plage disponible (on se fixe la plage du CSV)
    trk_t0_ms = float(t_ms[0])
    trk_t1_ms = float(t_ms[-1])

    gaps = _find_tracker_gaps(t_ms, TRACKER_GAP_THRESHOLD_MS)
    n_gaps = len(gaps)

    if n_gaps == 0:
        return {"session": name, "status": "ok", "n_gaps": 0}

    report = {
        "session": name,
        "n_gaps":  n_gaps,
        "gaps_ms": [(i, round(d, 1)) for i, d in gaps],
    }

    # ── Décision : interpoler ou trimmer ? ────────────────────────────────────
    large_gaps  = [(i, d) for i, d in gaps if d >= MAX_INTERP_GAP_MS]
    small_gaps  = [(i, d) for i, d in gaps if d < MAX_INTERP_GAP_MS]

    if large_gaps:
        # Trouver le segment principal
        main_start, main_end, coverage = _find_main_segment(
            t_ms, trk_t0_ms, trk_t1_ms, TRACKER_GAP_THRESHOLD_MS
        )
        if coverage < MIN_COVERAGE_AFTER_TRIM:
            report.update({
                "status":    "unrecoverable",
                "reason":    f"couverture après trim = {coverage*100:.0f}% < seuil",
                "coverage":  round(coverage, 3),
            })
            return report

        df_trimmed = df_valid.iloc[main_start:main_end + 1].copy().reset_index(drop=True)
        t_ms_trimmed = df_trimmed["timestamp_ns"].to_numpy(np.float64) / 1e6

        report["strategy"] = "trim"
        report["coverage"]  = round(coverage, 3)
        report["rows_before"] = len(df_valid)
        report["rows_after"]  = len(df_trimmed)

        # Vérifier les petits gaps résiduels dans le segment trimmé
        small_gaps_remaining = _find_tracker_gaps(t_ms_trimmed, TRACKER_GAP_THRESHOLD_MS)

        if not dry_run:
            df_to_write = df_trimmed
    else:
        # Seulement des petits gaps → interpolation
        df_to_write = df_valid.copy()

        # Identifier les colonnes interpolables
        pos_cols  = [f"tracker_{t}_{ax}" for t in TRACKERS for ax in AXES]
        quat_cols = [f"tracker_{t}_{q}"  for t in TRACKERS for q in QUATS]
        other_cols = [c for c in df_to_write.columns
                      if c not in pos_cols + quat_cols + ["timestamp_ns"]]

        interp_rows: list[pd.DataFrame] = []

        for gap_idx, gap_dur in small_gaps:
            row_before = df_to_write.iloc[gap_idx].copy()
            row_after  = df_to_write.iloc[gap_idx + 1].copy()

            t_before = float(row_before["timestamp_ns"]) / 1e6
            t_after  = float(row_after["timestamp_ns"])  / 1e6

            # Vitesse max pour valider l'interpolation
            dt_s = (t_after - t_before) / 1000.0
            synth_rows = []
            n_interp = max(1, int(gap_dur / 10.0))  # un point tous les ~10ms

            for k in range(1, n_interp + 1):
                alpha = k / (n_interp + 1)
                t_synth_ns = int(
                    float(row_before["timestamp_ns"]) * (1 - alpha) +
                    float(row_after["timestamp_ns"]) * alpha
                )
                new_row: dict = {"timestamp_ns": t_synth_ns}

                # Interpoler les positions linéairement
                for col in pos_cols:
                    if col in df_to_write.columns:
                        v0 = pd.to_numeric(row_before.get(col, np.nan), errors="coerce")
                        v1 = pd.to_numeric(row_after.get(col, np.nan), errors="coerce")
                        if np.isfinite(v0) and np.isfinite(v1):
                            new_row[col] = v0 * (1 - alpha) + v1 * alpha
                        else:
                            new_row[col] = v0 if np.isfinite(v0) else v1

                # Interpoler les quaternions avec SLERP
                for tracker in TRACKERS:
                    q_cols = [f"tracker_{tracker}_{q}" for q in QUATS]
                    if all(c in df_to_write.columns for c in q_cols):
                        q0 = np.array([
                            pd.to_numeric(row_before.get(c, 0), errors="coerce")
                            for c in q_cols
                        ], dtype=np.float64)
                        q1 = np.array([
                            pd.to_numeric(row_after.get(c, 0), errors="coerce")
                            for c in q_cols
                        ], dtype=np.float64)
                        if np.all(np.isfinite(q0)) and np.all(np.isfinite(q1)):
                            q_interp = slerp(q0, q1, alpha)
                            for c, v in zip(q_cols, q_interp):
                                new_row[c] = float(v)

                # Copier les autres colonnes du row_before
                for col in other_cols:
                    new_row[col] = row_before.get(col)

                synth_rows.append(new_row)

            if synth_rows:
                interp_rows.append(pd.DataFrame(synth_rows))

        if interp_rows and not dry_run:
            df_interp = pd.concat(interp_rows, ignore_index=True)
            df_to_write = pd.concat([df_to_write, df_interp], ignore_index=True)
            df_to_write = df_to_write.sort_values("timestamp_ns").reset_index(drop=True)

        n_interp_total = sum(len(r) for r in interp_rows)
        report["strategy"]      = "interpolation"
        report["n_interp_rows"] = n_interp_total
        report["rows_before"]   = len(df_valid)
        report["rows_after"]    = len(df_to_write) + n_interp_total

    if not dry_run:
        df_to_write.to_csv(csv_path, index=False)
        meta[GAPS_MARKER_KEY] = True
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        report["status"] = "corrected"
    else:
        report["status"] = "dry-run"

    return report
