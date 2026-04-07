#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_quaternions.py — Réparation des quaternions NaN/inf dans tracker_positions.csv.

Problème :
    Des lignes du CSV peuvent contenir des quaternions invalides (NaN, inf, zéro-vecteur).
    Ces valeurs corrompent :
      - L'identification géométrique head/left/right
      - Les features d'orientation pour l'IA
      - Les exports LeRobot

Algorithme :
  1. Localiser toutes les lignes avec quaternions invalides
  2. Pour chaque ligne invalide :
     a. Chercher la fenêtre de voisins valides [i-W, i+W]
     b. Interpoler par SLERP entre les deux voisins valides les plus proches
     c. Si aucun voisin trouvé dans la fenêtre → copie du dernier valide
  3. Valider la norme des quaternions corrigés (doit être ≈ 1.0)
  4. Si > QUAT_MAX_CORRUPT_FRAC invalides → non récupérable

Usage :
    from fix.fix_quaternions import fix_quaternions
    report = fix_quaternions(Path("/path/to/session"))
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

QUAT_MARKER_KEY       = "quaternions_repaired"
TRACKERS              = ("head", "left", "right")
QUATS                 = ("qw", "qx", "qy", "qz")

# Fraction max de quaternions invalides récupérables
QUAT_MAX_CORRUPT_FRAC = 0.20   # au-delà → non récupérable

# Fenêtre de recherche de voisins valides (lignes)
NEIGHBOR_WINDOW       = 50


# ══════════════════════════════════════════════════════════════════════════════
# SLERP (réutilisé depuis fix_tracker_gaps)
# ══════════════════════════════════════════════════════════════════════════════

def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / (np.linalg.norm(result) + 1e-12)
    theta_0 = np.arccos(dot)
    theta   = theta_0 * t
    sin_t0  = np.sin(theta_0)
    result  = (np.sin(theta_0 - theta) / sin_t0) * q0 + (np.sin(theta) / sin_t0) * q1
    return result / (np.linalg.norm(result) + 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Fix principal
# ══════════════════════════════════════════════════════════════════════════════

def fix_quaternions(
    session_path: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Répare les quaternions invalides dans tracker_positions.csv.

    Args:
        session_path : chemin vers la session
        dry_run      : mesure uniquement, sans modifier
        force        : re-applique même si déjà corrigé

    Returns:
        dict rapport avec n_repaired par tracker
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

    if not force and meta.get(QUAT_MARKER_KEY):
        return {"session": name, "status": "skipped", "reason": "déjà réparé"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"session": name, "status": "error", "reason": f"CSV illisible: {e}"}

    n_rows = len(df)
    tracker_reports: dict[str, dict] = {}
    total_repaired = 0

    for tracker in TRACKERS:
        q_cols = [f"tracker_{tracker}_{q}" for q in QUATS]
        if not all(c in df.columns for c in q_cols):
            tracker_reports[tracker] = {"status": "skip", "reason": "colonnes absentes"}
            continue

        # Récupérer les quaternions en tableau numpy
        Q = np.stack([
            pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
            for c in q_cols
        ], axis=1)  # (n, 4)

        # Identifier les lignes invalides
        invalid_mask = ~np.all(np.isfinite(Q), axis=1)
        # Aussi invalide : quaternion zéro (norme ≈ 0)
        norms = np.linalg.norm(Q, axis=1)
        invalid_mask |= (norms < 0.1)

        n_invalid = int(invalid_mask.sum())
        frac = n_invalid / (n_rows + 1e-6)

        if n_invalid == 0:
            tracker_reports[tracker] = {"status": "ok", "n_invalid": 0}
            continue

        if frac > QUAT_MAX_CORRUPT_FRAC:
            tracker_reports[tracker] = {
                "status": "unrecoverable",
                "n_invalid": n_invalid,
                "frac": round(frac, 4),
                "reason": f"{frac*100:.1f}% invalides > seuil {QUAT_MAX_CORRUPT_FRAC*100:.0f}%",
            }
            continue

        if dry_run:
            tracker_reports[tracker] = {
                "status": "would_repair",
                "n_invalid": n_invalid,
                "frac": round(frac, 4),
            }
            continue

        # ── Réparation ────────────────────────────────────────────────────────
        Q_fixed = Q.copy()
        invalid_indices = np.where(invalid_mask)[0]
        repaired = 0

        for idx in invalid_indices:
            # Chercher les voisins valides dans la fenêtre
            lo = max(0, idx - NEIGHBOR_WINDOW)
            hi = min(n_rows - 1, idx + NEIGHBOR_WINDOW)

            # Voisin gauche
            left_valid = None
            for j in range(idx - 1, lo - 1, -1):
                if not invalid_mask[j]:
                    left_valid = j
                    break

            # Voisin droit
            right_valid = None
            for j in range(idx + 1, hi + 1):
                if not invalid_mask[j]:
                    right_valid = j
                    break

            if left_valid is not None and right_valid is not None:
                # Interpoler par SLERP selon la position temporelle relative
                total_span = right_valid - left_valid
                alpha = (idx - left_valid) / (total_span + 1e-6)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                Q_fixed[idx] = slerp(Q[left_valid], Q[right_valid], alpha)
                repaired += 1
            elif left_valid is not None:
                Q_fixed[idx] = Q[left_valid] / (np.linalg.norm(Q[left_valid]) + 1e-12)
                repaired += 1
            elif right_valid is not None:
                Q_fixed[idx] = Q[right_valid] / (np.linalg.norm(Q[right_valid]) + 1e-12)
                repaired += 1
            # Si aucun voisin : laisser tel quel (cas rare, couvert par max_frac)

        # Mettre à jour le DataFrame
        for i, col in enumerate(q_cols):
            df[col] = Q_fixed[:, i]

        tracker_reports[tracker] = {
            "status":   "repaired",
            "n_invalid": n_invalid,
            "repaired": repaired,
            "frac":     round(frac, 4),
        }
        total_repaired += repaired

    # ── Écriture ──────────────────────────────────────────────────────────────
    if not dry_run and total_repaired > 0:
        df.to_csv(csv_path, index=False)
        meta[QUAT_MARKER_KEY] = True
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    any_unrecoverable = any(
        r.get("status") == "unrecoverable" for r in tracker_reports.values()
    )

    return {
        "session":          name,
        "status":           ("repaired" if total_repaired > 0
                             else ("dry-run" if dry_run
                                   else "ok")),
        "total_repaired":   total_repaired,
        "recoverable":      not any_unrecoverable,
        "tracker_reports":  tracker_reports,
    }
