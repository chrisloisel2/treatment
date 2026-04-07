#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_camera_labels.py — Identification des caméras head/left/right.

Approche en deux étapes :

ÉTAPE 1 — POSITION DES TRACKERS (certitude physique)
    Utilise fix_tracker_labels pour déterminer quel tracker CSV est
    head / left / right à partir de la hauteur Y, centralité, mobilité
    et projection latérale basée sur la POSITION MOYENNE relative à la tête.

ÉTAPE 2 — CORRESPONDANCE CAMÉRA ↔ TRACKER (flux optique)
    Chaque caméra est physiquement fixée sur un gripper ou la tête.
    Son mouvement (flux optique) est donc directement lié au mouvement
    du tracker correspondant.

    a) HEAD : la caméra avec le flux optique moyen le plus faible.
       La tête bouge moins vite que les mains → signal fiable.

    b) LEFT / RIGHT : pour les deux caméras restantes, on calcule un score
       d'association avec chaque tracker de main :
       - Signal global : corrélation entre flux optique et vitesse tracker
         (avec recherche de décalage temporel ±15 frames pour robustesse)
       - Signal asymétrique : quand un tracker est CLAIREMENT plus actif
         que l'autre (ratio ≥ 2.5×), la caméra correspondante devrait
         montrer un flux plus élevé.
       Le score combiné donne un vote robuste même quand les deux mains
       bougent en même temps.

NOTE : les numéros de série des caméras ne sont PAS utilisés car les flux
vidéo ne correspondent pas toujours aux numéros de série enregistrés.

Usage :
    python -m fix.fix_camera_labels /chemin/session [--dry-run] [--force]
    python -m fix.fix_camera_labels /chemin/root --batch

    from fix.fix_camera_labels import fix_camera_labels
    report = fix_camera_labels(Path("/chemin/session"))
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import sys

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
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

from fix.fix_tracker_labels import (
    fix_tracker_labels,
    _load_blocks,
    _test_height,
    _test_centrality,
    _test_mobility,
    _test_lateral,
    _consensus,
)


# ── Constantes ────────────────────────────────────────────────────────────────

MARKER_KEY         = "camera_labels_verified"
CAMERAS            = ("head", "left", "right")
MAX_FLOW_FRAMES    = 800        # frames max pour le flux optique
FLOW_RESIZE        = (160, 90)  # taille de redimensionnement (speed vs precision)
LAG_MAX_FRAMES     = 15         # décalage max ±15 frames ≈ ±500ms à 30fps
ASYM_RATIO         = 2.5        # ratio vitesse pour définir une période asymétrique
ASYM_MIN_FRAMES    = 5          # frames asymétriques minimales pour le signal
CONFIDENCE_CERTAIN = 0.15       # séparation minimale (score_best - score_2nd) pour certitude


# ── Structures ────────────────────────────────────────────────────────────────

@dataclass
class CameraLabelReport:
    session: str
    status: str           # "ok"|"corrected"|"uncertain"|"error"|"skipped"
    reason: str = ""
    predicted: dict = field(default_factory=dict)   # {cam_file: role}
    current: dict = field(default_factory=dict)     # {cam_file: role} (metadata actuel)
    tracker_prediction: dict = field(default_factory=dict)  # {role: csv_label}
    flow_scores: dict = field(default_factory=dict) # {cam_file: {role: score}}
    corrected: bool = False
    dry_run: bool = False


# ── Flux optique ──────────────────────────────────────────────────────────────

def _optical_flow_signal(
    video_path: Path,
    max_frames: int = MAX_FLOW_FRAMES,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calcule la magnitude moyenne du flux optique frame-par-frame.

    Returns:
        (flow_mag, timestamps_ms) ou (None, None) si la vidéo est inaccessible.
    """
    if not _CV2:
        return None, None

    cap = cv2.VideoCapture(str(video_path))
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None, None

    prev_gray = cv2.resize(
        cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), FLOW_RESIZE
    )

    # Lire les timestamps depuis le .jsonl correspondant
    jsonl = video_path.with_suffix(".jsonl")
    times: list[float] = []
    if jsonl.exists():
        with open(jsonl, errors="ignore") as f:
            for line in f:
                try:
                    times.append(float(json.loads(line)["capture_time"]))
                except Exception:
                    pass

    mags: list[float] = []
    while len(mags) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), FLOW_RESIZE)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 10, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(float(np.mean(mag)))
        prev_gray = gray

    cap.release()
    if not mags:
        return None, None

    t = np.array(times, dtype=float)
    n = len(mags)
    if len(t) > n:
        t_out = t[1 : n + 1]
    else:
        # Fallback : timestamps synthétiques à 30fps
        t0 = float(t[0]) if len(t) > 0 else 0.0
        t_out = np.arange(n) * 33.0 + t0

    return np.array(mags, dtype=float), t_out[:n]


# ── Vitesse des trackers ───────────────────────────────────────────────────────

def _tracker_speed_at(
    df: pd.DataFrame,
    csv_label: str,
    times_ms: np.ndarray,
) -> np.ndarray:
    """
    Calcule la vitesse 3D (m/s) du tracker `csv_label` rééchantillonnée
    aux instants `times_ms` (millisecondes).
    """
    pos = df[
        [f"tracker_{csv_label}_x",
         f"tracker_{csv_label}_y",
         f"tracker_{csv_label}_z"]
    ].to_numpy(float)
    ts   = df["timestamp_ns"].to_numpy(float) / 1e6   # → ms
    dt   = np.diff(ts)
    speed = (
        np.linalg.norm(np.diff(pos, axis=0), axis=1)
        / np.maximum(dt / 1000.0, 1e-6)
    )
    t_mid = 0.5 * (ts[:-1] + ts[1:])
    return np.interp(times_ms, t_mid, speed)


# ── Score caméra ↔ tracker ───────────────────────────────────────────────────

def _xcorr_lagged(a: np.ndarray, b: np.ndarray, max_lag: int = LAG_MAX_FRAMES) -> float:
    """Corrélation normalisée maximale avec décalage ±max_lag."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    an = (a - a.mean()) / a.std()
    bn = (b - b.mean()) / b.std()
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            c = float(np.mean(an * bn))
        elif lag > 0:
            c = float(np.mean(an[lag:] * bn[:-lag]))
        else:
            c = float(np.mean(an[:lag] * bn[-lag:]))
        if c > best:
            best = c
    return best


def _asymmetric_score(
    cam_mag: np.ndarray,
    sp_a: np.ndarray,
    sp_b: np.ndarray,
    ratio: float = ASYM_RATIO,
) -> float:
    """
    Score asymétrique : combien de fois le mouvement de la caméra est-il
    plus élevé quand le tracker A est actif que quand le tracker B est actif ?

    Valeur positive → caméra suit le tracker A.
    Valeur négative → caméra suit le tracker B.
    """
    n = min(len(cam_mag), len(sp_a), len(sp_b))
    cam = cam_mag[:n]
    a   = sp_a[:n]
    b   = sp_b[:n]

    # Lissage pour réduire le bruit frame-par-frame
    k = 5
    cam = np.convolve(cam, np.ones(k) / k, "same")
    a   = np.convolve(a,   np.ones(k) / k, "same")
    b   = np.convolve(b,   np.ones(k) / k, "same")

    eps   = 1e-3
    mask_a = a > ratio * (b + eps)   # A clairement plus actif
    mask_b = b > ratio * (a + eps)   # B clairement plus actif

    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())

    if n_a < ASYM_MIN_FRAMES and n_b < ASYM_MIN_FRAMES:
        return 0.0   # pas assez de signal asymétrique

    cam_a = float(np.mean(cam[mask_a])) if n_a >= ASYM_MIN_FRAMES else float(np.median(cam))
    cam_b = float(np.mean(cam[mask_b])) if n_b >= ASYM_MIN_FRAMES else float(np.median(cam))

    return cam_a - cam_b


def _camera_tracker_score(
    cam_mag: np.ndarray,
    sp_a: np.ndarray,
    sp_b: np.ndarray,
) -> float:
    """
    Score combiné pour évaluer si `cam_mag` suit le tracker A plutôt que B.

    Combine :
    - corrélation globale (avec décalage ±15 frames)
    - score asymétrique (comportement pendant les mouvements unilatéraux)
    """
    corr  = _xcorr_lagged(cam_mag, sp_a)
    asym  = _asymmetric_score(cam_mag, sp_a, sp_b)

    # Normalisation de l'asymétrie sur la plage attendue (~0–2 pixel/frame)
    asym_norm = float(np.clip(asym / 0.5, -1.0, 1.0))

    return 0.5 * corr + 0.5 * asym_norm


# ── Attribution caméra → rôle ─────────────────────────────────────────────────

def _assign_cameras(
    cam_signals: dict[str, tuple[np.ndarray, np.ndarray]],
    df: pd.DataFrame,
    tracker_prediction: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict[str, float]], float]:
    """
    Attribue chaque flux caméra à un rôle (head/left/right).

    Returns:
        assignment  : {cam_file: role}
        score_matrix: {cam_file: {role: score}}
        confidence  : séparation best - 2nd pour la décision la plus ambiguë
    """
    # Étape 1 — HEAD : caméra avec flux optique moyen le plus faible
    mean_flows = {cf: float(np.mean(mag)) for cf, (mag, _) in cam_signals.items()}
    head_cam   = min(mean_flows, key=mean_flows.get)
    hand_cams  = [c for c in cam_signals if c != head_cam]

    assignment: dict[str, str] = {head_cam: "head"}

    if len(hand_cams) < 2:
        # Pas assez de caméras hand → assigner par défaut
        for cf in hand_cams:
            assignment[cf] = "left" if cf == "left" else "right"
        return assignment, {}, 0.0

    c0, c1    = hand_cams[0], hand_cams[1]
    mag0, t0  = cam_signals[c0]
    mag1, t1  = cam_signals[c1]

    # Roles des mains (exclure head)
    hand_roles = [r for r in tracker_prediction if r != "head"]
    r0, r1     = hand_roles[0], hand_roles[1]
    csv0, csv1 = tracker_prediction[r0], tracker_prediction[r1]

    # Étape 2 — MAIN GAUCHE / DROITE : score asymétrique + corrélation globale
    sp0_on_c0 = _tracker_speed_at(df, csv0, t0)
    sp1_on_c0 = _tracker_speed_at(df, csv1, t0)
    sp0_on_c1 = _tracker_speed_at(df, csv0, t1)
    sp1_on_c1 = _tracker_speed_at(df, csv1, t1)

    score_c0_r0 = _camera_tracker_score(mag0, sp0_on_c0, sp1_on_c0)
    score_c0_r1 = _camera_tracker_score(mag0, sp1_on_c0, sp0_on_c0)
    score_c1_r0 = _camera_tracker_score(mag1, sp0_on_c1, sp1_on_c1)
    score_c1_r1 = _camera_tracker_score(mag1, sp1_on_c1, sp0_on_c1)

    score_matrix: dict[str, dict[str, float]] = {
        c0: {r0: round(score_c0_r0, 3), r1: round(score_c0_r1, 3)},
        c1: {r0: round(score_c1_r0, 3), r1: round(score_c1_r1, 3)},
        head_cam: {"head": 1.0},
    }

    # Décision : c0→r0 ou c0→r1 ?
    # Vote de c0 ET de c1 (cohérence croisée)
    vote_c0_r0 = score_c0_r0 - score_c0_r1   # > 0 si c0 préfère r0
    vote_c1_r0 = score_c1_r1 - score_c1_r0   # > 0 si c1 préfère r1 (= c0 va vers r0)

    # Score global pour l'assignement c0→r0
    combined = vote_c0_r0 + vote_c1_r0
    confidence = abs(combined) / 2.0   # normalisation approximative [0, 1]

    if combined >= 0:
        assignment[c0] = r0
        assignment[c1] = r1
    else:
        assignment[c0] = r1
        assignment[c1] = r0

    return assignment, score_matrix, float(confidence)


# ── Lecture de l'état actuel ──────────────────────────────────────────────────

def _current_camera_assignment(session_path: Path) -> dict[str, str]:
    """
    Retourne l'assignement actuel des fichiers caméra depuis les noms de fichiers JSONL.
    Retourne {cam_file: role_label} où role_label est déduit du nom de fichier.
    Par convention : head.jsonl → "head", left.jsonl → "left", right.jsonl → "right".
    """
    current: dict[str, str] = {}
    videos_dir = session_path / "videos"
    for role in CAMERAS:
        if (videos_dir / f"{role}.jsonl").exists():
            current[role] = role
    return current


# ── Point d'entrée principal ──────────────────────────────────────────────────

def fix_camera_labels(
    session_path: Path,
    dry_run: bool = False,
    force: bool = False,
) -> CameraLabelReport:
    """
    Identifie et corrige (si nécessaire) les labels head/left/right des caméras.

    Algorithme :
    1. Corrige les labels de trackers (fix_tracker_labels).
    2. Calcule le flux optique de chaque vidéo (mp4 requis).
    3. Attribue chaque caméra au rôle dont le tracker a la vitesse la plus
       corrélée, en privilégiant les périodes de mouvement asymétrique.

    Retourne un CameraLabelReport avec status :
      "ok"        — labels déjà corrects
      "corrected" — labels corrigés
      "uncertain" — corrélation insuffisante / ambiguë
      "error"     — données manquantes ou incompatibles
      "skipped"   — déjà vérifié (MARKER_KEY présent)
    """
    if not _PANDAS:
        return CameraLabelReport(session=str(session_path.name),
                                  status="error", reason="pandas non disponible")
    if not _CV2:
        return CameraLabelReport(session=str(session_path.name),
                                  status="error", reason="cv2 non disponible (pip install opencv-python)")

    name      = session_path.name
    meta_path = session_path / "metadata.json"
    csv_path  = session_path / "tracker_positions.csv"
    videos_dir = session_path / "videos"

    if not meta_path.exists():
        return CameraLabelReport(session=name, status="error",
                                  reason="metadata.json absent")
    if not csv_path.exists():
        return CameraLabelReport(session=name, status="error",
                                  reason="tracker_positions.csv absent")
    if not videos_dir.exists():
        return CameraLabelReport(session=name, status="error",
                                  reason="répertoire videos/ absent")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return CameraLabelReport(session=name, status="error",
                                  reason=f"metadata.json illisible: {e}")

    if not force and meta.get(MARKER_KEY):
        return CameraLabelReport(session=name, status="skipped",
                                  reason="labels déjà vérifiés (MARKER_KEY présent)")

    # ── Étape 1 : correction des labels trackers ──────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return CameraLabelReport(session=name, status="error",
                                  reason=f"tracker_positions.csv illisible: {e}")

    blocks = _load_blocks(df)
    if blocks is None:
        return CameraLabelReport(session=name, status="error",
                                  reason="colonnes tracker manquantes dans le CSV")

    t1 = _test_height(blocks)
    t2 = _test_centrality(blocks)
    t3 = _test_mobility(blocks)
    t4 = _test_lateral(blocks, t1.head_vote)
    tracker_prediction, _, _ = _consensus([t1, t2, t3, t4])
    # tracker_prediction = {role: csv_label}  ex: {head:'head', left:'right', right:'left'}

    # ── Étape 2 : flux optique des caméras ────────────────────────────────────
    # Chercher les mp4 dans videos/
    cam_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing_mp4: list[str] = []

    for cam_file in CAMERAS:
        mp4 = videos_dir / f"{cam_file}.mp4"
        if not mp4.exists():
            missing_mp4.append(cam_file)
            continue
        mag, times = _optical_flow_signal(mp4)
        if mag is None or times is None:
            missing_mp4.append(cam_file)
            continue
        cam_signals[cam_file] = (mag, times)

    if len(cam_signals) < 3:
        return CameraLabelReport(
            session=name, status="error",
            reason=f"mp4 manquants ou illisibles: {missing_mp4}",
            tracker_prediction=tracker_prediction,
        )

    # ── Étape 3 : attribution caméra → rôle ──────────────────────────────────
    assignment, score_matrix, confidence = _assign_cameras(
        cam_signals, df, tracker_prediction
    )

    # État actuel : par convention les fichiers sont nommés par leur rôle actuel
    current = _current_camera_assignment(session_path)

    # Est-ce que les noms de fichiers doivent changer ?
    # assignment = {cam_file_actuel: rôle_prédit}
    # On cherche si le rôle prédit ≠ nom de fichier actuel
    needs_correction = any(
        assignment.get(cf) != cf for cf in CAMERAS if cf in assignment
    )

    report = CameraLabelReport(
        session=name,
        status="",
        predicted=assignment,
        current=current,
        tracker_prediction=tracker_prediction,
        flow_scores=score_matrix,
    )

    if confidence < CONFIDENCE_CERTAIN and needs_correction:
        report.status = "uncertain"
        report.reason = (
            f"Corrélation ambiguë (confiance={confidence:.3f} < {CONFIDENCE_CERTAIN}). "
            f"Les deux mains bougent trop similairement pour distinguer."
        )
        if not dry_run:
            meta[MARKER_KEY] = False
            meta["camera_labels_confidence"] = round(confidence, 3)
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return report

    if not needs_correction:
        report.status = "ok"
        report.reason = (
            f"Labels caméra corrects "
            f"(confiance={confidence:.3f}, tracker={tracker_prediction})"
        )
        if not dry_run:
            meta[MARKER_KEY] = True
            meta["camera_labels_confidence"] = round(confidence, 3)
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return report

    if dry_run:
        report.status  = "would_correct"
        report.dry_run = True
        report.reason  = (
            f"Correction nécessaire (confiance={confidence:.3f}). "
            f"Assignement prédit: {assignment}"
        )
        return report

    # ── Appliquer la correction : renommer les fichiers JSONL (et MP4) ─────────
    # assignment = {ancien_nom: nouveau_rôle}
    # On veut renommer : ancien_nom.jsonl → nouveau_rôle.jsonl (et même pour .mp4)
    old_to_new: dict[str, str] = {
        cf: assignment[cf]
        for cf in assignment
        if assignment[cf] != cf
    }

    for ext in (".jsonl", ".mp4"):
        # Phase 1 : vers noms temporaires
        for old_name, new_name in old_to_new.items():
            src = videos_dir / f"{old_name}{ext}"
            tmp = videos_dir / f"__tmp_{new_name}{ext}"
            if src.exists():
                shutil.move(str(src), str(tmp))
        # Phase 2 : noms temporaires → noms finaux
        for old_name, new_name in old_to_new.items():
            tmp = videos_dir / f"__tmp_{new_name}{ext}"
            dst = videos_dir / f"{new_name}{ext}"
            if tmp.exists():
                shutil.move(str(tmp), str(dst))

    # Mise à jour metadata.json
    meta[MARKER_KEY]               = True
    meta["camera_labels_confidence"] = round(confidence, 3)
    meta["camera_labels_old"]      = {cf: cf for cf in CAMERAS}
    meta["camera_labels_new"]      = assignment
    meta["camera_labels_method"]   = "optical_flow_tracker_correlation"
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report.status    = "corrected"
    report.corrected = True
    report.reason    = (
        f"Labels corrigés (confiance={confidence:.3f}). "
        f"Renommages: {old_to_new}"
    )
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_report(r: CameraLabelReport) -> None:
    icons = {
        "ok": "✓", "corrected": "↺", "uncertain": "⚠",
        "error": "✗", "skipped": "–", "would_correct": "~",
    }
    icon = icons.get(r.status, "?")
    print(f"\n{icon} [{r.status.upper()}] {r.session}")
    if r.reason:
        print(f"  Raison       : {r.reason}")
    if r.tracker_prediction:
        print(f"  Trackers     : {r.tracker_prediction}")
    if r.predicted:
        print(f"  Caméras préd : {r.predicted}")
    if r.flow_scores:
        for cam, scores in r.flow_scores.items():
            print(f"  [{cam:5s}] {scores}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vérifie et corrige le labeling head/left/right des caméras "
                    "par corrélation flux optique ↔ trackers."
    )
    parser.add_argument("sessions", nargs="+", type=Path,
                        help="Répertoires de session(s)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Analyse uniquement, sans modifier")
    parser.add_argument("--force",    action="store_true",
                        help="Ré-analyse même si MARKER_KEY présent")
    parser.add_argument("--batch",    action="store_true",
                        help="Traiter toutes les sessions sous chaque chemin")
    parser.add_argument("--json",     action="store_true",
                        help="Sortie JSON")
    args = parser.parse_args()

    sessions: list[Path] = []
    for p in args.sessions:
        p = p.resolve()
        if args.batch and p.is_dir():
            sessions.extend(
                m.parent
                for m in p.rglob("metadata.json")
                if (m.parent / "videos").exists()
            )
        else:
            sessions.append(p)

    results = []
    for s in sessions:
        if not s.is_dir():
            continue
        r = fix_camera_labels(s, dry_run=args.dry_run, force=args.force)
        results.append(r)
        if not args.json:
            _print_report(r)

    if args.json:
        import dataclasses
        print(json.dumps(
            [asdict(r) for r in results],
            indent=2, ensure_ascii=False, default=str,
        ))

    if not args.json and results:
        counts = Counter(r.status for r in results)
        print(f"\n{'─'*60}")
        print("Total %d : " % len(results) +
              "  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
