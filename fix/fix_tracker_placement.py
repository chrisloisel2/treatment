#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_tracker_placement.py — Re-identification et correction du placement des trackers.

Problème :
    Les 3 trackers VR (head, left, right) peuvent être mal assignés dans le CSV.
    Les colonnes tracker_head_*, tracker_left_*, tracker_right_* peuvent correspondre
    à des trackers physiquement incorrects.

    Cela dégrade :
      - L'identification géométrique head/left/right
      - La synchronisation caméra-tracker (mauvaise paire)
      - L'export LeRobot (positions incohérentes)

Algorithme de re-identification (5 critères pondérés, adaptatifs) :

  1. HAUTEUR (axe Y) — poids fort (6 × adaptatif)
     Le head devrait être le plus haut.
     Critère le plus discriminant en contexte VR.

  2. CENTRALITÉ — poids moyen (3 × adaptatif)
     Le head est entre les deux mains (distance moyenne aux autres trackers).

  3. MOBILITÉ — poids faible (1)
     La tête bouge moins que les mains (médiane de la norme de déplacement).

  4. DISPERSION — poids faible (1)
     La tête est moins dispersée que les mains.

  5. AXE SECONDAIRE — poids faible (1)
     Meilleur séparateur parmi X et Z pour le head.

  Identification gauche/droite :
     Projection des mains sur l'axe latéral du head (rotation quaternion).
     Règle : wxyz, axis=0, sign=+1 (validée empiriquement dans trakeur.py).

CONFIANCE :
    Si la confiance de la prédiction < CONFIDENCE_THRESHOLD → ne pas corriger.
    Douter de tout : si la confiance est faible, mieux vaut ne pas changer.

CORRECTION :
    Si head prédit ≠ head actuel (ou left/right inversés) :
    → Renommer les colonnes dans le CSV

Usage :
    from fix.fix_tracker_placement import fix_tracker_placement
    report = fix_tracker_placement(Path("/path/to/session"))
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

PLACEMENT_MARKER_KEY   = "tracker_placement_verified"
TRACKERS               = ("head", "left", "right")
AXES                   = ("x", "y", "z")
QUATS                  = ("qw", "qx", "qy", "qz")

# Confiance minimale pour appliquer une correction
CONFIDENCE_THRESHOLD   = 0.60

# Fenêtre de lissage (lignes) pour réduire le bruit
SMOOTH_WINDOW          = 9


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires mathématiques
# ══════════════════════════════════════════════════════════════════════════════

def _moving_average(arr: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / window
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(arr[:, j], kernel, mode="same")
    return out


def _rank_points(values: np.ndarray, higher_better: bool = True) -> np.ndarray:
    """Attribue 0/1/2 pts aux 3 valeurs (pire/moyen/meilleur)."""
    values = np.asarray(values, dtype=float)
    order  = np.argsort(values)
    pts    = np.zeros(len(values), dtype=float)
    if higher_better:
        pts[order[0]] = 0.0
        pts[order[1]] = 1.0
        pts[order[2]] = 2.0
    else:
        pts[order[0]] = 2.0
        pts[order[1]] = 1.0
        pts[order[2]] = 0.0
    return pts


def _sep(vals: np.ndarray, higher: bool) -> float:
    """Séparation relative entre le leader et le 2ème (mesure de clarté du signal)."""
    sv = np.sort(vals)
    gap = (sv[-1] - sv[-2]) if higher else (sv[1] - sv[0])
    return float(gap / (np.ptp(vals) + 1e-9))


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    """Convertit quaternions (n, 4) wxyz en matrices de rotation (n, 3, 3)."""
    q = q.astype(float)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norms, 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


# ══════════════════════════════════════════════════════════════════════════════
# Chargement des données tracker
# ══════════════════════════════════════════════════════════════════════════════

def _load_tracker_blocks(
    df: pd.DataFrame,
) -> Optional[list[tuple[str, np.ndarray, np.ndarray]]]:
    """
    Charge les 3 blocs de données tracker depuis le DataFrame.

    Returns:
        Liste de (label, pos_smoothed, quat) ou None si incomplet.
    """
    blocks = []
    for label in TRACKERS:
        pos_cols  = [f"tracker_{label}_{ax}" for ax in AXES]
        quat_cols = [f"tracker_{label}_{q}"  for q in QUATS]

        if not all(c in df.columns for c in pos_cols + quat_cols):
            return None

        try:
            pos = np.stack([
                pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(np.float64)
                for c in pos_cols
            ], axis=1)
            quat = np.stack([
                pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(np.float64)
                for c in quat_cols
            ], axis=1)
        except Exception:
            return None

        # Filtrer les lignes invalides
        valid = np.all(np.isfinite(pos), axis=1) & np.all(np.isfinite(quat), axis=1)
        if valid.sum() < 20:
            return None

        pos_clean  = pos[valid]
        quat_clean = quat[valid]

        # Lisser les positions
        pos_smooth = _moving_average(pos_clean, SMOOTH_WINDOW)

        blocks.append((label, pos_smooth, quat_clean))

    return blocks if len(blocks) == 3 else None


# ══════════════════════════════════════════════════════════════════════════════
# Identification des trackers
# ══════════════════════════════════════════════════════════════════════════════

def detect_head(
    blocks: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[str, np.ndarray, float]:
    """
    Identifie quel tracker est la tête par analyse géométrique pondérée.

    Returns:
        (predicted_label, scores_array, confidence)
        - predicted_label : label prédit pour "head"
        - scores_array    : scores [0]=head, [1]=left, [2]=right
        - confidence      : séparation relative du score (0=ambigu, 1=clair)
    """
    labels = [b[0] for b in blocks]
    centers = [np.median(b[1], axis=0) for b in blocks]
    motions = [np.median(np.linalg.norm(np.diff(b[1], axis=0), axis=1)) for b in blocks]
    spreads = [np.median(np.linalg.norm(b[1] - c, axis=1))
               for b, c in zip(blocks, centers)]

    centers  = np.array(centers)
    motions  = np.array(motions)
    spreads  = np.array(spreads)

    # Centralité : distance médiane aux autres trackers
    pair_mean = np.zeros(3, dtype=float)
    for i in range(3):
        dists = [
            np.median(np.linalg.norm(blocks[i][1] - blocks[j][1], axis=1))
            for j in range(3) if i != j
        ]
        pair_mean[i] = np.mean(dists)

    # Axe Y = axe vertical physique dans le repère VR
    h_y = centers[:, 1]

    # Meilleur axe secondaire parmi X et Z
    best_sep_other = -np.inf
    best_h_other   = np.zeros(3, dtype=float)
    for axis in (0, 2):
        for sign in (-1.0, 1.0):
            h   = sign * centers[:, axis]
            hs  = np.sort(h)
            sep = (hs[-1] - hs[-2]) / (np.ptp(h) + 1e-9)
            if sep > best_sep_other:
                best_sep_other = sep
                best_h_other   = h

    # Poids adaptatifs
    sep_hy   = _sep(h_y,       higher=True)
    sep_pair = _sep(pair_mean, higher=False)
    w_hy     = 6.0 * (1.0 + 2.0 * sep_hy)
    w_pair   = 3.0 * (1.0 + 2.0 * sep_pair)

    scores = (
        1.0    * _rank_points(motions,      higher_better=False) +
        1.0    * _rank_points(spreads,      higher_better=False) +
        w_pair * _rank_points(pair_mean,    higher_better=False) +
        w_hy   * _rank_points(h_y,          higher_better=True)  +
        1.0    * _rank_points(best_h_other, higher_better=True)
    )

    best_idx   = int(np.argmax(scores))
    total      = scores.sum()
    confidence = float(scores[best_idx] / (total + 1e-6))

    return labels[best_idx], scores, confidence


def detect_hands(
    blocks: list[tuple[str, np.ndarray, np.ndarray]],
    head_label: str,
) -> tuple[str, str, float]:
    """
    Identifie left/right parmi les deux trackers non-head.
    Utilise la projection sur l'axe latéral du head (quaternion-based).

    Règle validée : wxyz, axis=0, sign=+1

    Returns:
        (left_label, right_label, confidence)
    """
    head_block  = next(b for b in blocks if b[0] == head_label)
    other_blocks = [b for b in blocks if b[0] != head_label]

    if len(other_blocks) != 2:
        return other_blocks[0][0], other_blocks[1][0], 0.0

    _, head_pos, head_quat = head_block
    label_a, pos_a, _ = other_blocks[0]
    label_b, pos_b, _ = other_blocks[1]

    # Rotation matrices du head
    R = _quat_to_rotmat_wxyz(head_quat)

    # Axe latéral (axis=0, sign=+1 dans le repère head)
    basis = R[:, :, 0]  # colonne 0 de la matrice de rotation

    # Projection des deux trackers sur l'axe latéral du head
    n = min(len(head_pos), len(pos_a), len(pos_b), len(basis))
    proj_a = np.sum((pos_a[:n] - head_pos[:n]) * basis[:n], axis=1)
    proj_b = np.sum((pos_b[:n] - head_pos[:n]) * basis[:n], axis=1)

    median_a = float(np.median(proj_a))
    median_b = float(np.median(proj_b))
    separation = abs(median_a - median_b)

    # left = projection la plus petite (côté négatif de l'axe latéral)
    if median_a <= median_b:
        left_label, right_label = label_a, label_b
    else:
        left_label, right_label = label_b, label_a

    # Confiance basée sur la séparation
    pos_range = float(np.ptp(np.concatenate([pos_a[:n, 0], pos_b[:n, 0]])))
    confidence = float(np.clip(separation / (pos_range + 1e-6), 0.0, 1.0))

    return left_label, right_label, confidence


# ══════════════════════════════════════════════════════════════════════════════
# Correction du CSV
# ══════════════════════════════════════════════════════════════════════════════

def _rename_tracker_columns(
    df: pd.DataFrame,
    old_assignment: dict[str, str],  # {current_label: predicted_role}
    new_assignment: dict[str, str],  # {role: current_label}
) -> pd.DataFrame:
    """
    Renomme les colonnes du DataFrame pour corriger le placement.

    old_assignment : {label_dans_csv: role_prédit}
      ex: {"head": "left", "left": "head", "right": "right"}
    new_assignment : {role_prédit: label_correct}
      ex: {"head": "left", "left": "head", "right": "right"}

    Si head prédit = "left" : les colonnes tracker_left_* → tracker_head_*
                               les colonnes tracker_head_* → tracker_left_*
    """
    # Construire le mapping de renommage
    # role_cible → label_source
    rename_map: dict[str, str] = {}
    for role, source_label in new_assignment.items():
        if role != source_label:
            for ax in ("x", "y", "z", "qw", "qx", "qy", "qz"):
                old_col = f"tracker_{source_label}_{ax}"
                new_col = f"tracker_{role}_{ax}"
                rename_map[old_col] = f"__tmp_{new_col}"

    if not rename_map:
        return df

    # Phase 1 : renommer vers des noms temporaires (évite les collisions)
    df = df.rename(columns=rename_map)

    # Phase 2 : renommer les temporaires vers les noms finaux
    final_map = {f"__tmp_{v}": v for v in rename_map.values()}
    df = df.rename(columns=final_map)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Fix principal
# ══════════════════════════════════════════════════════════════════════════════

def fix_tracker_placement(
    session_path: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Vérifie et corrige le placement des trackers head/left/right.

    La correction n'est appliquée que si :
      - La confiance de détection du head ≥ CONFIDENCE_THRESHOLD
      - Le placement prédit est différent du placement actuel

    Args:
        session_path : chemin vers la session
        dry_run      : mesure uniquement, sans modifier
        force        : re-applique

    Returns:
        dict rapport avec predicted_assignment, confidence, action
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

    if not force and meta.get(PLACEMENT_MARKER_KEY):
        return {"session": name, "status": "skipped", "reason": "placement déjà vérifié"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"session": name, "status": "error", "reason": f"CSV illisible: {e}"}

    # Charger les blocs de données
    blocks = _load_tracker_blocks(df)
    if blocks is None:
        return {"session": name, "status": "error",
                "reason": "Impossible de charger les blocs tracker (colonnes manquantes ?)"}

    # ── Détection du head ──────────────────────────────────────────────────────
    head_pred, head_scores, head_conf = detect_head(blocks)

    report = {
        "session":             name,
        "head_predicted":      head_pred,
        "head_confidence":     round(head_conf, 3),
        "head_scores":         {b[0]: round(float(s), 3)
                                for b, s in zip(blocks, head_scores)},
    }

    if head_conf < CONFIDENCE_THRESHOLD:
        report["status"] = "uncertain"
        report["reason"] = (
            f"Confiance trop faible ({head_conf:.2f} < {CONFIDENCE_THRESHOLD}) "
            f"pour corriger le placement"
        )
        # Marquer quand même pour ne pas répéter
        if not dry_run:
            meta[PLACEMENT_MARKER_KEY] = True
            meta["tracker_placement_confidence"] = round(head_conf, 3)
            meta["tracker_placement_predicted"]  = head_pred
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    # ── Détection left/right ────────────────────────────────────────────────────
    left_pred, right_pred, hands_conf = detect_hands(blocks, head_pred)

    report["left_predicted"]   = left_pred
    report["right_predicted"]  = right_pred
    report["hands_confidence"] = round(hands_conf, 3)

    # Assignement prédit : {rôle → label_dans_csv}
    predicted = {
        "head":  head_pred,
        "left":  left_pred,
        "right": right_pred,
    }

    # Assignement actuel (labels = head, left, right tel que dans le CSV)
    current = {"head": "head", "left": "left", "right": "right"}

    # Comparer
    needs_correction = any(predicted[role] != current[role] for role in TRACKERS)
    report["current_assignment"]   = current
    report["predicted_assignment"] = predicted
    report["needs_correction"]     = needs_correction

    if not needs_correction:
        report["status"] = "ok"
        report["reason"] = "placement correct"
        if not dry_run:
            meta[PLACEMENT_MARKER_KEY] = True
            meta["tracker_placement_confidence"] = round(head_conf, 3)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    if dry_run:
        report["status"] = "would_correct"
        report["correction"] = {
            role: f"{current[role]} → {predicted[role]}"
            for role in TRACKERS
            if predicted[role] != current[role]
        }
        return report

    # ── Appliquer la correction ───────────────────────────────────────────────
    # new_assignment = {role: label_source_dans_csv}
    # ex: {"head": "left", "left": "head", "right": "right"}
    df_fixed = _rename_tracker_columns(df, current, predicted)
    df_fixed.to_csv(csv_path, index=False)

    meta[PLACEMENT_MARKER_KEY]             = True
    meta["tracker_placement_confidence"]   = round(head_conf, 3)
    meta["tracker_placement_corrected"]    = True
    meta["tracker_placement_old"]          = current
    meta["tracker_placement_new"]          = predicted
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    report["status"]     = "corrected"
    report["correction"] = {
        role: f"{predicted[role]} renommé en {role}"
        for role in TRACKERS
        if predicted[role] != current[role]
    }
    return report
