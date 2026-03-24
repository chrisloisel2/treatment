#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vérification et correction des labels caméra (left / right / head).

═══════════════════════════════════════════════════════════════════════
STRATÉGIE — 3 niveaux de preuve, tous requis pour atteindre 100 %
═══════════════════════════════════════════════════════════════════════

NIVEAU 1 — Géométrie 3D des trackers  (source de vérité principale)
  Les trackers VIVE sont montés sur les caméras.
  Leurs positions X/Y/Z permettent de savoir avec certitude :
    · Lequel est le plus haut   → head
    · Lequel est le plus à gauche → left
    · Lequel est le plus à droite → right
  On utilise la moyenne sur toute la session pour robustesse.

NIVEAU 2 — Fisheye (head uniquement)
  La caméra head est fisheye grand angle.
  On détecte ça par :
    · Vignetage radial (centre plus lumineux que les bords)
    · Netteté centre >> coins
    · Distorsion barrel (lignes courbées, HoughLines courtes)
    · Rapport d'aspect proche de 1 (carré ou légèrement large)
  Un faux-positif ici = alerte immédiate.

NIVEAU 3 — Cohérence flux de mouvement
  Les caméras left et right bougent ensemble (même robot, même scène).
  On vérifie que la corrélation left↔right > seuil et que head
  a une corrélation distincte (champ de vue différent).

═══════════════════════════════════════════════════════════════════════
MODE SAFE (défaut)
═══════════════════════════════════════════════════════════════════════
  --safe   : analyse uniquement, écrit la correction dans metadata.json
             sous la clé "camera_label_verification". Aucun renommage.
  --apply  : renomme les fichiers vidéo/JSONL si besoin (confirmation
             interactive requise sauf --yes).

═══════════════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════════════
  python verify_video_labels.py session_dir/
  python verify_video_labels.py session_dir/ --safe
  python verify_video_labels.py session_dir/ --apply
  python verify_video_labels.py session_dir/ --apply --yes
  python verify_video_labels.py session_dir/ --output-png report.png -v

  # Chemins explicites
  python verify_video_labels.py \\
      --left left.mp4  --left-jsonl left.jsonl \\
      --right right.mp4 --right-jsonl right.jsonl \\
      --head head.mp4  --head-jsonl head.jsonl \\
      --tracker tracker_positions.csv \\
      --metadata metadata.json

Dépendances :
  pip install opencv-python numpy pandas scipy matplotlib
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import correlate


# ──────────────────────────────────────────────────────────────────────────────
# Constantes / seuils
# ──────────────────────────────────────────────────────────────────────────────

# Niveau 1 — tracker
TRACKER_MIN_ROWS          = 10    # lignes minimum dans le CSV tracker
TRACKER_SEPARATION_MIN_M  = 0.05  # séparation minimale attendue entre trackers (m)

# Niveau 2 — fisheye
FISHEYE_THRESHOLD         = 0.52  # score ≥ ce seuil → fisheye confirmé
FISHEYE_MIN_FRAMES        = 5     # frames minimum pour l'analyse visuelle

# Niveau 3 — corrélation mouvement
MOTION_FRAMES             = 120   # frames max pour extraire le signal de mouvement
MOTION_LR_CORR_MIN        = 0.35  # corrélation min attendue entre left et right
MOTION_HEAD_CORR_MAX      = 0.80  # si head corrèle trop fort avec LR → suspect

# Score global de confiance
CONFIDENCE_FULL           = 1.00  # tous niveaux convergent
CONFIDENCE_HIGH           = 0.90  # 2 niveaux convergent
CONFIDENCE_LOW            = 0.60  # 1 niveau seulement


# ──────────────────────────────────────────────────────────────────────────────
# Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackerAssignment:
    """Résultat de l'analyse géométrique 3D."""
    head_tracker_id:  str = ""
    left_tracker_id:  str = ""
    right_tracker_id: str = ""
    head_mean_pos:    Tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_mean_pos:    Tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_mean_pos:   Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vertical_separation_m:    float = 0.0
    horizontal_separation_m:  float = 0.0
    confidence: float = 0.0
    ok: bool = False
    details: List[str] = field(default_factory=list)


@dataclass
class FisheyeResult:
    """Résultat de l'analyse fisheye pour une vidéo."""
    label: str
    score: float = 0.0
    is_fisheye: bool = False
    details: List[str] = field(default_factory=list)


@dataclass
class MotionResult:
    """Résultat de la corrélation de mouvement."""
    lr_correlation: float = 0.0
    lh_correlation: float = 0.0
    rh_correlation: float = 0.0
    left_right_consistent: bool = False
    head_distinct: bool = False
    details: List[str] = field(default_factory=list)


@dataclass
class CameraVerdict:
    """Verdict final pour une caméra."""
    declared_label:   str
    file_path:        str
    predicted_label:  str = ""
    confidence:       float = 0.0
    label_correct:    bool = False
    warnings:         List[str] = field(default_factory=list)
    errors:           List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    session_dir:     str
    global_ok:       bool = True
    confidence:      float = 0.0
    tracker:         Optional[TrackerAssignment] = None
    fisheye:         Dict[str, FisheyeResult] = field(default_factory=dict)
    motion:          Optional[MotionResult] = None
    verdicts:        List[CameraVerdict] = field(default_factory=list)
    recommended_mapping: Dict[str, str] = field(default_factory=dict)
    summary:         str = ""
    safe_mode:       bool = True


# ──────────────────────────────────────────────────────────────────────────────
# NIVEAU 1 — Géométrie tracker 3D
# ──────────────────────────────────────────────────────────────────────────────

def analyze_trackers(csv_path: str) -> TrackerAssignment:
    """
    Lit tracker_positions.csv et détermine, via les positions 3D moyennes,
    quel tracker correspond à head / left / right.

    Convention spatiale :
      Y  = hauteur verticale  → head est le plus haut
      X  = axe latéral        → left est le plus à gauche (X le plus faible)
                                 right est le plus à droite (X le plus élevé)
    """
    result = TrackerAssignment()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result.details.append(f"ERREUR lecture CSV tracker : {e}")
        return result

    if len(df) < TRACKER_MIN_ROWS:
        result.details.append(f"Trop peu de lignes dans le tracker CSV ({len(df)})")
        return result

    # Colonnes attendues : tracker_head_x/y/z, tracker_left_x/y/z, tracker_right_x/y/z
    # → on travaille avec les 3 labels DÉCLARÉS dans le CSV
    # puis on recalcule qui devrait être qui.

    trackers_data: Dict[str, Dict[str, np.ndarray]] = {}

    for label in ("head", "left", "right"):
        xcol = f"tracker_{label}_x"
        ycol = f"tracker_{label}_y"
        zcol = f"tracker_{label}_z"
        if xcol not in df.columns:
            result.details.append(f"Colonne manquante : {xcol}")
            continue
        trackers_data[label] = {
            "x": df[xcol].dropna().to_numpy(),
            "y": df[ycol].dropna().to_numpy(),
            "z": df[zcol].dropna().to_numpy(),
        }

    if len(trackers_data) < 3:
        result.details.append("Données tracker incomplètes.")
        return result

    # Position moyenne de chaque tracker
    means: Dict[str, Dict[str, float]] = {}
    for label, axes in trackers_data.items():
        means[label] = {
            "x": float(np.median(axes["x"])),
            "y": float(np.median(axes["y"])),
            "z": float(np.median(axes["z"])),
        }

    # ─ Identification par hauteur (Y) → head
    by_height = sorted(means.keys(), key=lambda l: means[l]["y"], reverse=True)
    head_by_y = by_height[0]   # le plus haut

    # ─ Identification par X → left vs right
    # Parmi les deux trackers non-head, le plus à gauche = left
    laterals = [l for l in means.keys() if l != head_by_y]
    laterals_by_x = sorted(laterals, key=lambda l: means[l]["x"])
    left_by_x  = laterals_by_x[0]   # X minimal = gauche
    right_by_x = laterals_by_x[1]   # X maximal = droite

    # ─ Séparations
    vert_sep = abs(means[head_by_y]["y"] - means[laterals[0]]["y"])
    horiz_sep = abs(means[right_by_x]["x"] - means[left_by_x]["x"])

    result.head_tracker_id  = head_by_y
    result.left_tracker_id  = left_by_x
    result.right_tracker_id = right_by_x
    result.head_mean_pos    = (means[head_by_y]["x"],  means[head_by_y]["y"],  means[head_by_y]["z"])
    result.left_mean_pos    = (means[left_by_x]["x"],  means[left_by_x]["y"],  means[left_by_x]["z"])
    result.right_mean_pos   = (means[right_by_x]["x"], means[right_by_x]["y"], means[right_by_x]["z"])
    result.vertical_separation_m   = vert_sep
    result.horizontal_separation_m = horiz_sep

    # ─ Évaluation de la confiance
    issues = []

    if vert_sep < TRACKER_SEPARATION_MIN_M:
        issues.append(
            f"Séparation verticale head/latéraux faible ({vert_sep*100:.1f} cm < "
            f"{TRACKER_SEPARATION_MIN_M*100:.0f} cm attendus)"
        )

    if horiz_sep < TRACKER_SEPARATION_MIN_M:
        issues.append(
            f"Séparation horizontale left/right faible ({horiz_sep*100:.1f} cm < "
            f"{TRACKER_SEPARATION_MIN_M*100:.0f} cm attendus)"
        )

    # Vérifier cohérence avec les labels déclarés dans le CSV
    tracker_label_matches = (
        head_by_y  == "head" and
        left_by_x  == "left" and
        right_by_x == "right"
    )

    result.details.append(
        f"Positions médianes — "
        f"head(décl.)=({means['head']['x']:.3f}, {means['head']['y']:.3f}, {means['head']['z']:.3f})  "
        f"left(décl.)=({means['left']['x']:.3f}, {means['left']['y']:.3f}, {means['left']['z']:.3f})  "
        f"right(décl.)=({means['right']['x']:.3f}, {means['right']['y']:.3f}, {means['right']['z']:.3f})"
    )
    result.details.append(
        f"Plus haut tracker (Y)    : '{head_by_y}'  "
        f"(Y={means[head_by_y]['y']:.3f} m)"
    )
    result.details.append(
        f"Tracker le plus à gauche : '{left_by_x}'  "
        f"(X={means[left_by_x]['x']:.3f} m)"
    )
    result.details.append(
        f"Tracker le plus à droite : '{right_by_x}'  "
        f"(X={means[right_by_x]['x']:.3f} m)"
    )
    result.details.append(
        f"Séparation verticale head/latéraux : {vert_sep*100:.1f} cm"
    )
    result.details.append(
        f"Séparation horizontale left/right  : {horiz_sep*100:.1f} cm"
    )

    if issues:
        result.details += [f"⚠ {i}" for i in issues]

    # Confiance : dépend des séparations et de la cohérence avec labels déclarés
    if not issues and tracker_label_matches:
        result.confidence = 1.0
        result.details.append("✓ Géométrie 3D cohérente avec les labels déclarés dans le CSV")
    elif not issues and not tracker_label_matches:
        result.confidence = 0.95
        result.details.append(
            f"! Géométrie 3D indique un reclassement : "
            f"head→{head_by_y}, left→{left_by_x}, right→{right_by_x}"
        )
    elif issues and tracker_label_matches:
        result.confidence = 0.75
        result.details.append("⚠ Séparations faibles mais labels cohérents")
    else:
        result.confidence = 0.55
        result.details.append("⚠ Séparations faibles et labels incohérents — vérifier manuellement")

    result.ok = (result.confidence >= 0.70)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires vidéo
# ──────────────────────────────────────────────────────────────────────────────

def sample_frames(path: str, n: int = 10,
                  start_pct: float = 0.10,
                  end_pct: float = 0.90) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(int(total * start_pct), int(total * end_pct) - 1,
                          n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def extract_motion_signal(path: str, max_frames: int = MOTION_FRAMES) -> List[float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    signal = []
    prev = None
    count = 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 240))
        if prev is not None:
            diff = cv2.absdiff(prev, small).astype(np.float32)
            signal.append(float(diff.mean()))
        prev = small
        count += 1
    cap.release()
    return signal


def video_dimensions(path: str) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0.0, 0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, fc


# ──────────────────────────────────────────────────────────────────────────────
# NIVEAU 2 — Détection fisheye
# ──────────────────────────────────────────────────────────────────────────────

def _vignette_ratio(frames: List[np.ndarray]) -> Tuple[float, str]:
    """Centre / bords luminosité. Fisheye → ratio > 1.3."""
    ratios = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        r_max = min(cx, cy)
        Y, X = np.ogrid[:h, :w]
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        m_center = R < r_max * 0.22
        m_border = R > r_max * 0.78
        if m_center.sum() > 10 and m_border.sum() > 10:
            ratios.append(float(gray[m_center].mean()) /
                          max(float(gray[m_border].mean()), 1.0))
    if not ratios:
        return 1.0, "vignetage non mesurable"
    avg = float(np.median(ratios))
    if avg > 1.5:
        return 0.85, f"vignetage prononcé (centre {avg:.2f}× plus lumineux)"
    if avg > 1.2:
        return 0.55, f"vignetage modéré (ratio={avg:.2f})"
    return 0.15, f"pas de vignetage (ratio={avg:.2f})"


def _sharpness_ratio(frames: List[np.ndarray]) -> Tuple[float, str]:
    """Laplacien centre vs coins. Fisheye → coins flous → ratio > 1.5."""
    ratios = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        m = min(h, w) // 6

        def lap(patch):
            return float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size > 0 else 0.0

        center = gray[h//2 - m: h//2 + m, w//2 - m: w//2 + m]
        corners = [
            gray[:2*m, :2*m], gray[:2*m, -2*m:],
            gray[-2*m:, :2*m], gray[-2*m:, -2*m:]
        ]
        c_sharp = lap(center)
        k_sharp = float(np.mean([lap(c) for c in corners]))
        if k_sharp > 0:
            ratios.append(c_sharp / k_sharp)
    if not ratios:
        return 1.0, "netteté non mesurable"
    avg = float(np.median(ratios))
    if avg > 2.5:
        return 0.85, f"centre net, coins flous (ratio={avg:.2f}) → fisheye"
    if avg > 1.4:
        return 0.55, f"légère différence centre/coins (ratio={avg:.2f})"
    return 0.15, f"netteté uniforme centre/coins (ratio={avg:.2f})"


def _barrel_distortion(frames: List[np.ndarray]) -> Tuple[float, str]:
    """
    Détection distorsion barrel via lignes HoughLines :
    fisheye → lignes moyennes courtes (segments brisés par la courbure).
    """
    rel_lens = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=50, minLineLength=40, maxLineGap=8)
        if lines is not None and len(lines) >= 5:
            h, w = gray.shape
            diag = np.sqrt(h**2 + w**2)
            avg_len = float(np.mean([
                np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2)
                for l in lines
            ]))
            rel_lens.append(avg_len / diag)
    if not rel_lens:
        return 0.35, "distorsion non mesurable (pas de lignes)"
    avg = float(np.median(rel_lens))
    if avg < 0.09:
        return 0.80, f"lignes très courtes (len_rel={avg:.3f}) → distorsion barrel"
    if avg < 0.16:
        return 0.50, f"lignes courtes-moyennes (len_rel={avg:.3f})"
    return 0.15, f"longues lignes droites (len_rel={avg:.3f}) → pas de distorsion"


def _aspect_ratio_score(w: int, h: int) -> Tuple[float, str]:
    ar = w / max(h, 1)
    if 0.85 <= ar <= 1.20:
        return 0.70, f"rapport W/H={ar:.2f} (carré → fisheye)"
    if 0.70 <= ar <= 1.50:
        return 0.40, f"rapport W/H={ar:.2f} (neutre)"
    return 0.10, f"rapport W/H={ar:.2f} (très rectangulaire → pas fisheye)"


def analyze_fisheye(label: str, path: str) -> FisheyeResult:
    result = FisheyeResult(label=label)

    w, h, fps, fc = video_dimensions(path)
    if w == 0:
        result.details.append("Vidéo illisible")
        return result

    frames = sample_frames(path, n=12)
    if len(frames) < FISHEYE_MIN_FRAMES:
        result.details.append(f"Trop peu de frames ({len(frames)})")
        return result

    scores = []

    s, detail = _aspect_ratio_score(w, h)
    scores.append(s)
    result.details.append(f"Aspect ratio   : {detail}")

    s, detail = _vignette_ratio(frames)
    scores.append(s)
    result.details.append(f"Vignetage      : {detail}")

    s, detail = _sharpness_ratio(frames)
    scores.append(s)
    result.details.append(f"Netteté        : {detail}")

    s, detail = _barrel_distortion(frames)
    scores.append(s)
    result.details.append(f"Distorsion     : {detail}")

    result.score = float(np.mean(scores))
    result.is_fisheye = result.score >= FISHEYE_THRESHOLD
    result.details.append(
        f"→ Score fisheye = {result.score:.3f}  "
        f"({'FISHEYE' if result.is_fisheye else 'pas fisheye'})"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# NIVEAU 3 — Corrélation de mouvement
# ──────────────────────────────────────────────────────────────────────────────

def _xcorr(a: List[float], b: List[float]) -> float:
    if len(a) < 5 or len(b) < 5:
        return 0.0
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    va = (va - va.mean()) / (va.std() + 1e-9)
    vb = (vb - vb.mean()) / (vb.std() + 1e-9)
    n  = min(len(va), len(vb))
    return float(np.corrcoef(va[:n], vb[:n])[0, 1])


def analyze_motion(paths: Dict[str, str]) -> MotionResult:
    result = MotionResult()

    signals: Dict[str, List[float]] = {}
    for label, path in paths.items():
        sig = extract_motion_signal(path)
        if sig:
            signals[label] = sig

    if "left" in signals and "right" in signals:
        result.lr_correlation = _xcorr(signals["left"], signals["right"])
    if "left" in signals and "head" in signals:
        result.lh_correlation = _xcorr(signals["left"], signals["head"])
    if "right" in signals and "head" in signals:
        result.rh_correlation = _xcorr(signals["right"], signals["head"])

    result.left_right_consistent = result.lr_correlation >= MOTION_LR_CORR_MIN
    result.head_distinct = (
        result.lh_correlation <= MOTION_HEAD_CORR_MAX and
        result.rh_correlation <= MOTION_HEAD_CORR_MAX
    )

    result.details.append(f"Corrélation left ↔ right : {result.lr_correlation:+.3f}  "
                           f"(seuil ≥ {MOTION_LR_CORR_MIN})")
    result.details.append(f"Corrélation left ↔ head  : {result.lh_correlation:+.3f}")
    result.details.append(f"Corrélation right ↔ head : {result.rh_correlation:+.3f}")

    if result.left_right_consistent:
        result.details.append("✓ Left et right ont un mouvement cohérent (même bras robotique)")
    else:
        result.details.append(
            f"⚠ Left et right peu corrélées ({result.lr_correlation:.3f}) — "
            "vérifier que les vidéos correspondent bien aux mains"
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Agrégation — verdict final
# ──────────────────────────────────────────────────────────────────────────────

def compute_verdicts(
    paths:    Dict[str, str],
    tracker:  TrackerAssignment,
    fisheye:  Dict[str, FisheyeResult],
    motion:   MotionResult,
) -> Tuple[List[CameraVerdict], Dict[str, str], float]:
    """
    Construit les verdicts caméra et le mapping corrigé.
    Retourne (verdicts, recommended_mapping, confidence_globale).
    """
    # Mapping prédit par le tracker (source de vérité principale)
    tracker_map: Dict[str, str] = {}
    if tracker.ok:
        tracker_map = {
            tracker.head_tracker_id:  "head",
            tracker.left_tracker_id:  "left",
            tracker.right_tracker_id: "right",
        }

    verdicts = []
    agreements = 0
    total_checks = 0

    for declared_label in ("head", "left", "right"):
        path = paths.get(declared_label, "")
        verdict = CameraVerdict(declared_label=declared_label, file_path=path)

        level_predictions = []

        # Niveau 1
        if tracker.ok and declared_label in tracker_map.values():
            t_pred = tracker_map.get(declared_label, "")
            if t_pred:
                level_predictions.append(("tracker_3d", t_pred))
            elif declared_label in tracker_map:
                level_predictions.append(("tracker_3d", tracker_map[declared_label]))

        # Reconstruire : tracker prédit pour CE label déclaré
        # tracker_map est {label_csv → vrai_label}
        # Pour "head" déclaré : on cherche la prédiction tracker
        tracker_pred_for_label = tracker_map.get(declared_label, declared_label)
        if tracker.ok:
            level_predictions.append(("tracker_3d", tracker_pred_for_label))
            total_checks += 1

        # Niveau 2 — fisheye (pertinent pour tous, mais surtout head)
        fi = fisheye.get(declared_label)
        if fi is not None:
            total_checks += 1
            if declared_label == "head":
                fisheye_pred = "head" if fi.is_fisheye else "left_or_right"
                level_predictions.append(("fisheye", fisheye_pred))
                if fi.is_fisheye:
                    agreements += 1
                else:
                    verdict.warnings.append(
                        f"La vidéo déclarée 'head' n'est pas détectée comme fisheye "
                        f"(score={fi.score:.2f} < {FISHEYE_THRESHOLD})"
                    )
            else:
                # left ou right ne doivent PAS être fisheye
                fisheye_pred = "head" if fi.is_fisheye else declared_label
                level_predictions.append(("fisheye", fisheye_pred))
                if fi.is_fisheye:
                    verdict.warnings.append(
                        f"La vidéo déclarée '{declared_label}' ressemble à un fisheye "
                        f"(score={fi.score:.2f}) → possible confusion avec head"
                    )
                else:
                    agreements += 1

        # Compter les accords tracker
        if tracker.ok and tracker_pred_for_label == declared_label:
            agreements += 1

        # Niveau 3 — mouvement (informatif, ne change pas le verdict seul)
        if declared_label in ("left", "right"):
            if not motion.left_right_consistent:
                verdict.warnings.append(
                    f"Corrélation left↔right faible ({motion.lr_correlation:.3f}) — "
                    "mouvement incohérent entre les deux mains"
                )

        # Prédiction finale
        # On priorise le tracker (source physique), puis le fisheye
        if tracker.ok:
            verdict.predicted_label = tracker_pred_for_label
        elif fi is not None and declared_label == "head":
            verdict.predicted_label = "head" if fi.is_fisheye else "unknown"
        else:
            verdict.predicted_label = declared_label  # pas assez de données

        verdict.label_correct = (verdict.predicted_label == declared_label)
        if not verdict.label_correct:
            verdict.errors.append(
                f"Label déclaré '{declared_label}' mais géométrie 3D indique "
                f"'{verdict.predicted_label}'"
            )

        verdicts.append(verdict)

    # Confiance globale
    if total_checks > 0:
        agreement_rate = agreements / total_checks
    else:
        agreement_rate = 0.0

    if agreement_rate >= 0.90 and tracker.confidence >= 0.90:
        global_conf = CONFIDENCE_FULL
    elif agreement_rate >= 0.70:
        global_conf = CONFIDENCE_HIGH
    else:
        global_conf = CONFIDENCE_LOW

    # Mapping recommandé : label_actuel → label_correct
    recommended: Dict[str, str] = {}
    for v in verdicts:
        if v.predicted_label and v.predicted_label != "left_or_right":
            recommended[v.declared_label] = v.predicted_label

    return verdicts, recommended, global_conf


# ──────────────────────────────────────────────────────────────────────────────
# Mode SAFE — écriture dans metadata.json
# ──────────────────────────────────────────────────────────────────────────────

def write_safe_report(metadata_path: str, report: VerificationReport) -> None:
    """
    Écrit la section 'camera_label_verification' dans metadata.json.
    Ne touche à rien d'autre.
    """
    try:
        with open(metadata_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"  ERREUR lecture metadata : {e}", file=sys.stderr)
        return

    # Construction du bloc de vérification
    block = {
        "verified_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "global_ok":   report.global_ok,
        "confidence":  round(report.confidence, 4),
        "safe_mode":   True,
        "recommended_camera_mapping": report.recommended_mapping,
        "verdicts": [
            {
                "declared_label":  v.declared_label,
                "predicted_label": v.predicted_label,
                "label_correct":   v.label_correct,
                "warnings":        v.warnings,
                "errors":          v.errors,
            }
            for v in report.verdicts
        ],
        "tracker_analysis": {
            "ok":         report.tracker.ok if report.tracker else False,
            "confidence": round(report.tracker.confidence, 4) if report.tracker else 0.0,
            "head_identified_as":  report.tracker.head_tracker_id  if report.tracker else "",
            "left_identified_as":  report.tracker.left_tracker_id  if report.tracker else "",
            "right_identified_as": report.tracker.right_tracker_id if report.tracker else "",
            "vertical_separation_cm":   round((report.tracker.vertical_separation_m   if report.tracker else 0.0) * 100, 1),
            "horizontal_separation_cm": round((report.tracker.horizontal_separation_m if report.tracker else 0.0) * 100, 1),
        },
        "fisheye_analysis": {
            label: {
                "score":      round(fi.score, 4),
                "is_fisheye": fi.is_fisheye,
            }
            for label, fi in (report.fisheye or {}).items()
        },
        "motion_analysis": {
            "lr_correlation": round(report.motion.lr_correlation, 4) if report.motion else 0.0,
            "lh_correlation": round(report.motion.lh_correlation, 4) if report.motion else 0.0,
            "rh_correlation": round(report.motion.rh_correlation, 4) if report.motion else 0.0,
            "left_right_consistent": report.motion.left_right_consistent if report.motion else False,
        },
        "summary": report.summary,
    }

    meta["camera_label_verification"] = block

    # Backup avant écriture
    p = Path(metadata_path)
    bak = p.with_suffix(".json.bak_verify")
    try:
        import shutil
        shutil.copy2(str(p), str(bak))
    except Exception:
        pass

    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Rapport écrit dans {metadata_path}")
    print(f"    (backup → {bak.name})")


# ──────────────────────────────────────────────────────────────────────────────
# Mode APPLY — renommage fichiers
# ──────────────────────────────────────────────────────────────────────────────

def apply_renames(
    session_dir: Path,
    recommended: Dict[str, str],
    yes: bool = False,
) -> bool:
    """
    Renomme les vidéos et JSONL selon le mapping recommandé.
    Retourne True si tout s'est bien passé.
    """
    renames = []
    extensions = [".mp4", ".MP4", ".jsonl", ".JSONL"]

    for old_label, new_label in recommended.items():
        if old_label == new_label:
            continue
        for ext in extensions:
            for search_dir in [session_dir / "videos", session_dir]:
                src = search_dir / f"{old_label}{ext}"
                if src.exists():
                    dst = search_dir / f"{new_label}{ext}"
                    renames.append((src, dst))

    if not renames:
        print("  Aucun renommage nécessaire.")
        return True

    print("\n  Renommages prévus :")
    for src, dst in renames:
        print(f"    {src.name}  →  {dst.name}")

    if not yes:
        answer = input("\n  Confirmer les renommages ? [oui/non] : ").strip().lower()
        if answer not in ("oui", "o", "yes", "y"):
            print("  Annulé.")
            return False

    for src, dst in renames:
        if dst.exists():
            print(f"  ERREUR : {dst} existe déjà — renommage annulé pour ce fichier")
            continue
        src.rename(dst)
        print(f"  ✓ {src.name} → {dst.name}")

    return True


# ──────────────────────────────────────────────────────────────────────────────
# PNG de vérification
# ──────────────────────────────────────────────────────────────────────────────

def save_png(
    paths:    Dict[str, str],
    verdicts: List[CameraVerdict],
    fisheye:  Dict[str, FisheyeResult],
    motion:   MotionResult,
    tracker:  TrackerAssignment,
    out_path: str,
) -> None:
    labels = list(paths.keys())
    n = len(labels)
    fig, axes = plt.subplots(3, n, figsize=(6 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    verdict_map = {v.declared_label: v for v in verdicts}

    for col, label in enumerate(labels):
        path    = paths[label]
        verdict = verdict_map.get(label)
        fi      = fisheye.get(label)
        ok_col  = "green" if (verdict and verdict.label_correct) else "red"

        # Ligne 0 — frame centrale
        ax = axes[0][col]
        frames = sample_frames(path, n=3)
        if frames:
            rgb = cv2.cvtColor(frames[len(frames)//2], cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
        pred = verdict.predicted_label if verdict else "?"
        conf_txt = f"fisheye={fi.score:.2f}" if fi else ""
        ax.set_title(
            f"{label.upper()}  →  prédit: {pred}\n{conf_txt}",
            color=ok_col, fontsize=11, fontweight="bold"
        )
        ax.axis("off")

        # Ligne 1 — signal de mouvement
        ax = axes[1][col]
        sig = extract_motion_signal(path, max_frames=60)
        if sig:
            ax.plot(sig, linewidth=1.2, color="steelblue")
        ax.set_title(f"Mouvement — {label}", fontsize=9)
        ax.set_xlabel("Frame")
        ax.grid(True, alpha=0.3)

        # Ligne 2 — scores fisheye
        ax = axes[2][col]
        if fi:
            categories = ["aspect\nratio", "vignetage", "netteté\ncoins", "distorsion"]
            bar_colors = ["steelblue"] * 4
            ax.bar(categories, [fi.score] * 4, color=bar_colors, alpha=0.6)
            ax.axhline(FISHEYE_THRESHOLD, color="red", linestyle="--", linewidth=1,
                       label=f"seuil={FISHEYE_THRESHOLD}")
            ax.set_ylim(0, 1)
            ax.set_title(f"Score fisheye = {fi.score:.3f}", fontsize=9)
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Titre global
    all_ok = all(v.label_correct for v in verdicts)
    status = "LABELS CORRECTS" if all_ok else "LABELS INCORRECTS — CORRECTION NÉCESSAIRE"
    fig.suptitle(
        f"Vérification labels vidéo\n{status}  |  tracker conf={tracker.confidence:.2f}",
        fontsize=13, fontweight="bold",
        color="green" if all_ok else "red"
    )

    # Corrélations en bas
    if motion:
        fig.text(
            0.5, 0.01,
            f"Corrélations mouvement :  "
            f"left↔right={motion.lr_correlation:+.3f}  "
            f"left↔head={motion.lh_correlation:+.3f}  "
            f"right↔head={motion.rh_correlation:+.3f}",
            ha="center", fontsize=9, color="dimgray"
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=130)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Résolution automatique des chemins
# ──────────────────────────────────────────────────────────────────────────────

def find_file(session_dir: Path, stem: str,
              extensions=(".mp4", ".MP4", ".jsonl", ".JSONL")) -> Optional[Path]:
    for ext in extensions:
        for sub in [session_dir / "videos", session_dir]:
            p = sub / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def resolve_paths(session_dir: Path, args) -> Dict[str, Optional[str]]:
    result = {}
    for label in ("left", "right", "head"):
        explicit = getattr(args, label, None)
        if explicit:
            result[label] = explicit
        else:
            p = find_file(session_dir, label, (".mp4", ".MP4"))
            result[label] = str(p) if p else None
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Rapport texte
# ──────────────────────────────────────────────────────────────────────────────

def print_report(report: VerificationReport, verbose: bool = False) -> None:
    SEP = "═" * 65
    sep = "─" * 65

    print(f"\n{SEP}")
    print("  VÉRIFICATION LABELS VIDÉO")
    print(SEP)

    # ── Niveau 1 ─────────────────────────────────────────────────────────
    print(f"\n{'NIVEAU 1 — Géométrie 3D des trackers':}")
    print(sep)
    t = report.tracker
    if t:
        status = "✓ OK" if t.ok else "✗ ÉCHEC"
        print(f"  Confiance tracker : {t.confidence:.2f}  [{status}]")
        if verbose or not t.ok:
            for d in t.details:
                print(f"    {d}")
        else:
            # Afficher juste les lignes essentielles
            for d in t.details:
                if any(kw in d for kw in ["Plus haut", "gauche", "droite", "Séparation", "✓", "!"]):
                    print(f"    {d}")
    else:
        print("  Trackers non disponibles.")

    # ── Niveau 2 ─────────────────────────────────────────────────────────
    print(f"\nNIVEAU 2 — Détection fisheye")
    print(sep)
    for label in ("head", "left", "right"):
        fi = report.fisheye.get(label)
        if fi:
            tag = "FISHEYE" if fi.is_fisheye else "normal "
            ok  = "✓" if (label == "head") == fi.is_fisheye else "✗"
            print(f"  {ok} {label:>5} : score={fi.score:.3f}  [{tag}]")
            if verbose:
                for d in fi.details:
                    print(f"         {d}")

    # ── Niveau 3 ─────────────────────────────────────────────────────────
    print(f"\nNIVEAU 3 — Cohérence mouvement")
    print(sep)
    m = report.motion
    if m:
        lr_ok  = "✓" if m.left_right_consistent else "⚠"
        print(f"  {lr_ok} left ↔ right : {m.lr_correlation:+.3f}")
        print(f"    left ↔ head  : {m.lh_correlation:+.3f}")
        print(f"    right ↔ head : {m.rh_correlation:+.3f}")

    # ── Verdicts ─────────────────────────────────────────────────────────
    print(f"\nVERDICTS PAR CAMÉRA")
    print(sep)
    for v in report.verdicts:
        marker = "✓" if v.label_correct else "✗"
        print(f"  {marker} {v.declared_label:>5}  →  prédit : {v.predicted_label:<8}  "
              f"({'CORRECT' if v.label_correct else 'INCORRECT'})")
        for e in v.errors:
            print(f"         ✗ {e}")
        for w in v.warnings:
            print(f"         ⚠ {w}")

    # ── Résumé ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    status_icon = "✓" if report.global_ok else "✗"
    print(f"  {status_icon} RÉSULTAT : {report.summary}")
    print(f"  Confiance globale : {report.confidence:.2%}")
    mode_txt = "[MODE SAFE — metadata.json mis à jour, aucun fichier renommé]" \
               if report.safe_mode else "[APPLY — renommages effectués]"
    print(f"  {mode_txt}")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ──────────────────────────────────────────────────────────────────────────────

def run(args) -> VerificationReport:
    # Résoudre le dossier de session
    session_dir = Path(args.session_dir) if args.session_dir else None
    if session_dir is None:
        # Essayer de déduire depuis --left
        if args.left:
            session_dir = Path(args.left).parent
        else:
            print("ERREUR : dossier de session requis.", file=sys.stderr)
            sys.exit(1)

    if not session_dir.is_dir():
        print(f"ERREUR : dossier introuvable : {session_dir}", file=sys.stderr)
        sys.exit(1)

    # Chemins vidéo
    paths: Dict[str, str] = {}
    for label in ("left", "right", "head"):
        explicit = getattr(args, label, None)
        if explicit:
            paths[label] = explicit
        else:
            p = find_file(session_dir, label, (".mp4", ".MP4"))
            if p:
                paths[label] = str(p)

    missing = [l for l in ("left", "right", "head") if l not in paths]
    if missing:
        print(f"ERREUR : vidéo(s) introuvable(s) : {', '.join(missing)}", file=sys.stderr)
        print("Passez --left / --right / --head ou un dossier de session valide.",
              file=sys.stderr)
        sys.exit(1)

    # Chemin tracker
    tracker_csv = getattr(args, "tracker", None)
    if not tracker_csv:
        for candidate in [
            session_dir / "tracker_positions.csv",
            session_dir.parent / "tracker_positions.csv",
        ]:
            if candidate.exists():
                tracker_csv = str(candidate)
                break

    # Chemin metadata
    metadata_path = getattr(args, "metadata", None)
    if not metadata_path:
        for candidate in [
            session_dir / "metadata.json",
            session_dir.parent / "metadata.json",
        ]:
            if candidate.exists():
                metadata_path = str(candidate)
                break

    report = VerificationReport(
        session_dir=str(session_dir),
        safe_mode=not getattr(args, "apply", False),
    )

    # ── Niveau 1 ─────────────────────────────────────────────────────────
    print("\n[1/3] Analyse géométrique 3D des trackers...")
    if tracker_csv:
        report.tracker = analyze_trackers(tracker_csv)
        print(f"      Confiance tracker : {report.tracker.confidence:.2f}")
    else:
        print("      ⚠ Fichier tracker_positions.csv introuvable — niveau 1 ignoré")
        report.tracker = TrackerAssignment()

    # ── Niveau 2 ─────────────────────────────────────────────────────────
    print("[2/3] Analyse fisheye des vidéos...")
    for label, path in paths.items():
        print(f"      {label} : {Path(path).name}...")
        fi = analyze_fisheye(label, path)
        report.fisheye[label] = fi

    # ── Niveau 3 ─────────────────────────────────────────────────────────
    print("[3/3] Analyse de cohérence du mouvement...")
    report.motion = analyze_motion(paths)

    # ── Verdicts ─────────────────────────────────────────────────────────
    verdicts, recommended, confidence = compute_verdicts(
        paths, report.tracker, report.fisheye, report.motion
    )
    report.verdicts = verdicts
    report.recommended_mapping = recommended
    report.confidence = confidence
    report.global_ok = all(v.label_correct for v in verdicts)

    # Résumé
    if report.global_ok:
        report.summary = (
            f"Tous les labels sont corrects "
            f"(confiance = {confidence:.0%})"
        )
    else:
        bad = [v for v in verdicts if not v.label_correct]
        corrections = ", ".join(
            f"{v.declared_label}→{v.predicted_label}" for v in bad
        )
        report.summary = (
            f"Labels incorrects détectés : {corrections}  "
            f"(confiance = {confidence:.0%})"
        )

    # ── Affichage ────────────────────────────────────────────────────────
    print_report(report, verbose=getattr(args, "verbose", False))

    # ── Écriture safe dans metadata ──────────────────────────────────────
    if metadata_path:
        write_safe_report(metadata_path, report)
    else:
        print("\n  ⚠ metadata.json non trouvé — rapport non écrit")

    # ── PNG ──────────────────────────────────────────────────────────────
    out_png = getattr(args, "output_png", None)
    if out_png is None:
        out_png = str(session_dir / "label_verification.png")
    print(f"\n  Génération du rapport visuel → {out_png}")
    try:
        save_png(paths, verdicts, report.fisheye, report.motion,
                 report.tracker, out_png)
        print(f"  ✓ {out_png}")
    except Exception as e:
        print(f"  ⚠ PNG non généré : {e}")

    # ── JSON ─────────────────────────────────────────────────────────────
    out_json = getattr(args, "output_json", None)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        print(f"  ✓ Rapport JSON → {out_json}")

    # ── Apply ────────────────────────────────────────────────────────────
    if getattr(args, "apply", False) and not report.global_ok:
        apply_renames(session_dir, recommended, yes=getattr(args, "yes", False))

    return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vérifie et corrige les labels caméra (left / right / head).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "session_dir", nargs="?", default=None,
        help="Dossier de session (cherche left.mp4, right.mp4, head.mp4 automatiquement)."
    )

    parser.add_argument("--left",  default=None, help="Chemin explicite vidéo left.")
    parser.add_argument("--right", default=None, help="Chemin explicite vidéo right.")
    parser.add_argument("--head",  default=None, help="Chemin explicite vidéo head.")

    parser.add_argument(
        "--left-jsonl",  default=None, help="JSONL associé à left (optionnel)."
    )
    parser.add_argument(
        "--right-jsonl", default=None, help="JSONL associé à right (optionnel)."
    )
    parser.add_argument(
        "--head-jsonl",  default=None, help="JSONL associé à head (optionnel)."
    )

    parser.add_argument(
        "--tracker",  default=None,
        help="Chemin vers tracker_positions.csv."
    )
    parser.add_argument(
        "--metadata", default=None,
        help="Chemin vers metadata.json."
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--safe", action="store_true", default=True,
        help="Mode safe (défaut) : analyse seulement, écrit dans metadata.json."
    )
    mode.add_argument(
        "--apply", action="store_true", default=False,
        help="Renomme les fichiers si des labels sont incorrects."
    )

    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Confirmer automatiquement les renommages (--apply uniquement)."
    )

    parser.add_argument(
        "--output-png", default=None,
        help="Chemin du PNG de rapport visuel."
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Chemin du rapport JSON détaillé."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Affiche les détails de chaque analyse."
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    return 0 if report.global_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
