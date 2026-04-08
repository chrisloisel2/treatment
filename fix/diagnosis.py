#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/diagnosis.py — Convertit un SessionReport (check.py) en liste de problèmes diagnostiqués.

Le diagnostic est en plusieurs couches :
  1. Problèmes structurels évidents (gates failed dans le rapport check)
  2. Mesures approfondies pour affiner le diagnostic (drift, offset précis, placement)
  3. Estimation de la récupérabilité

Le diagnostiqueur DOUTE de tout :
  - Il ne fait pas confiance aux noms de colonnes pour le placement tracker
  - Il re-mesure l'offset caméra indépendamment
  - Il détecte le drift même si l'offset semble faible
  - Il vérifie le placement physique des trackers via géométrie
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Ajouter le parent au path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fix.problems import (
    DiagnosedProblem,
    ProblemCode,
    make_problem,
)
from fix.fix_gripper_video_sync import (
    _load_gripper_df   as _load_gripper_df_gsync,
    _load_jsonl_times  as _load_jsonl_times_gsync,
    _analyse_level1    as _gripper_level1,
    OFFSET_SIGNIFICANT_MS as GRIPPER_OFFSET_SIGNIFICANT_MS,
)

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    from scipy.stats import linregress as _linregress
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ── Seuils de diagnostic ──────────────────────────────────────────────────────

# Offset brut caméra/tracker au-delà duquel on diagnostique CAMERA_OFFSET
OFFSET_DIAG_MS          = 200.0

# Drift: si la pente linéaire (offset vs temps) dépasse ce seuil en ppm
# (microsecondes par seconde), on diagnostique CLOCK_DRIFT
DRIFT_PPM_THRESHOLD     = 500.0     # 500µs/s = 0.5ms/s = 30ms/min

# Drift: R² minimum pour que la régression linéaire soit significative
DRIFT_R2_MIN            = 0.50

# Seuil de détection de mauvais placement tracker
# (différence de confiance entre le label prédit et le label actuel)
TRACKER_CONFIDENCE_THRESHOLD = 0.60

# Score IA faible
LOW_IA_SCORE_THRESHOLD  = 0.45

# Gaps
TRACKER_GAP_MS          = 60.0
TRACKER_GAP_FAIL_N      = 3
CAMERA_GAP_MS           = 120.0
CAMERA_GAP_FAIL_N       = 5

# Quaternions corrompus
QUAT_CORRUPT_FRAC       = 0.05

# Couverture caméra minimale
MIN_COVERAGE_RATIO      = 0.65


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires de lecture
# ══════════════════════════════════════════════════════════════════════════════

def _load_jsonl_times(path: Path) -> Optional[np.ndarray]:
    """Charge les capture_time depuis un fichier JSONL caméra."""
    if not path.exists():
        return None
    times = []
    try:
        raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        for line in raw.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ct = obj.get("capture_time")
                if ct is not None:
                    times.append(float(ct))
            except Exception:
                continue
    except Exception:
        return None
    return np.array(times, dtype=np.float64) if times else None


def _load_tracker_times(session_path: Path) -> Optional[np.ndarray]:
    """Charge les timestamps tracker en ms depuis timestamp_ns."""
    if not _PANDAS:
        return None
    path = session_path / "tracker_positions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "timestamp_ns" in df.columns:
            t_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce").dropna().to_numpy(np.float64)
            return t_ns / 1e6  # → ms
        if "time_seconds" in df.columns:
            t_s = pd.to_numeric(df["time_seconds"], errors="coerce").dropna().to_numpy(np.float64)
            return t_s * 1000.0
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics approfondis
# ══════════════════════════════════════════════════════════════════════════════

def _diagnose_camera_offset(
    session_path: Path,
    tracker_t_ms: np.ndarray,
) -> List[DiagnosedProblem]:
    """
    Mesure l'offset de chaque caméra par rapport au tracker.
    Détecte également le drift linéaire si l'overlap est suffisant.

    Offset mesuré = médiane des (cam_t - tracker_t) sur le début du signal.
    Drift mesuré = pente de régression linéaire sur l'offset au cours du temps.
    """
    problems = []
    trk_t0_ms = float(tracker_t_ms[0])
    trk_t1_ms = float(tracker_t_ms[-1])
    trk_dur_ms = trk_t1_ms - trk_t0_ms

    cameras = ("head", "left", "right")
    max_offset_ms = 0.0
    max_drift_ppm = 0.0
    worst_coverage = 1.0
    cam_details: dict = {}

    for cam in cameras:
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        times = _load_jsonl_times(jsonl_path)
        if times is None or len(times) < 10:
            cam_details[cam] = {"error": "no_data"}
            continue

        cam_t0 = float(times[0])
        cam_t1 = float(times[-1])

        # ── Offset grossier ────────────────────────────────────────────────────
        # On prend la médiane des 3 premières frames pour plus de robustesse
        head_times = times[:min(5, len(times))]
        raw_offset_ms = float(np.median(head_times)) - trk_t0_ms

        # ── Couverture ─────────────────────────────────────────────────────────
        overlap_ms = max(0.0, min(cam_t1, trk_t1_ms) - max(cam_t0, trk_t0_ms))
        coverage = overlap_ms / (trk_dur_ms + 1e-6)
        worst_coverage = min(worst_coverage, coverage)

        # ── Drift linéaire (si chevauchement suffisant) ─────────────────────────
        drift_ppm = 0.0
        drift_r2 = 0.0
        if _SCIPY and trk_dur_ms > 2000.0:
            # Points de mesure d'offset distribués dans le temps
            # On échantillonne à 1Hz dans la zone commune
            overlap_start = max(cam_t0, trk_t0_ms)
            overlap_end   = min(cam_t1, trk_t1_ms)
            if overlap_end - overlap_start > 1000.0:
                # Points tracker dans la zone commune
                trk_in_overlap = tracker_t_ms[
                    (tracker_t_ms >= overlap_start) & (tracker_t_ms <= overlap_end)
                ]
                if len(trk_in_overlap) >= 10:
                    # Pour chaque point tracker, trouver le frame caméra le plus proche
                    cam_sorted = np.sort(times)
                    cam_offsets = []
                    t_points = []
                    step = max(1, len(trk_in_overlap) // 20)  # max 20 points
                    for t_trk in trk_in_overlap[::step]:
                        idx = np.searchsorted(cam_sorted, t_trk)
                        idx = np.clip(idx, 0, len(cam_sorted) - 1)
                        nearest_cam = cam_sorted[idx]
                        local_offset = nearest_cam - t_trk
                        cam_offsets.append(local_offset)
                        t_points.append(t_trk - trk_t0_ms)  # temps relatif

                    if len(cam_offsets) >= 5:
                        t_arr = np.array(t_points, dtype=np.float64)
                        o_arr = np.array(cam_offsets, dtype=np.float64)
                        slope, intercept, r, p, _ = _linregress(t_arr, o_arr)
                        drift_r2 = float(r ** 2)
                        # slope en ms/ms → ppm (µs/s)
                        drift_ppm = abs(float(slope)) * 1e6  # ms/ms × 1e6 = µs/s
                        max_drift_ppm = max(max_drift_ppm, drift_ppm)

        max_offset_ms = max(max_offset_ms, abs(raw_offset_ms))
        cam_details[cam] = {
            "offset_ms": round(raw_offset_ms, 1),
            "coverage": round(coverage, 3),
            "drift_ppm": round(drift_ppm, 1),
            "drift_r2": round(drift_r2, 3),
        }

    # Diagnostiquer CAMERA_OFFSET si nécessaire
    if max_offset_ms >= OFFSET_DIAG_MS:
        problems.append(make_problem(
            ProblemCode.CAMERA_OFFSET,
            f"Offset caméra max = {max_offset_ms:.0f} ms (seuil {OFFSET_DIAG_MS:.0f} ms)",
            cam_offsets=cam_details,
            max_offset_ms=max_offset_ms,
            trk_t0_ms=trk_t0_ms,
            trk_t1_ms=trk_t1_ms,
        ))

    # Diagnostiquer CLOCK_DRIFT si nécessaire
    if max_drift_ppm >= DRIFT_PPM_THRESHOLD:
        problems.append(make_problem(
            ProblemCode.CLOCK_DRIFT,
            f"Drift d'horloge détecté : {max_drift_ppm:.0f} µs/s (seuil {DRIFT_PPM_THRESHOLD:.0f})",
            cam_details=cam_details,
            max_drift_ppm=max_drift_ppm,
        ))

    # Diagnostiquer couverture insuffisante (si pas déjà CAMERA_OFFSET)
    if worst_coverage < MIN_COVERAGE_RATIO and max_offset_ms < OFFSET_DIAG_MS:
        # Couverture mauvaise sans offset visible → peut être camera_misplaced
        problems.append(make_problem(
            ProblemCode.CAMERA_OFFSET,
            f"Couverture caméra = {worst_coverage*100:.0f}% sans offset détectable — "
            f"possible mauvais assignement caméra",
            cam_offsets=cam_details,
            max_offset_ms=max_offset_ms,
            worst_coverage=worst_coverage,
        ))

    return problems


def _diagnose_tracker_placement(session_path: Path) -> List[DiagnosedProblem]:
    """
    Vérifie si le placement géométrique des trackers correspond aux labels dans le CSV.
    Utilise les mêmes règles que trakeur.py mais en mode diagnostic.
    """
    if not _PANDAS:
        return []

    path = session_path / "tracker_positions.csv"
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    # Identifier les colonnes de position pour chaque tracker
    # Format attendu : tracker_{label}_x, tracker_{label}_y, tracker_{label}_z
    label_positions: dict[str, np.ndarray] = {}
    for label in ("head", "left", "right"):
        cols = [f"tracker_{label}_{ax}" for ax in ("x", "y", "z")]
        if not all(c in df.columns for c in cols):
            return []  # Colonnes manquantes → pas de diagnostic possible
        try:
            xyz = np.stack([
                pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
                for c in cols
            ], axis=1)
            valid = np.all(np.isfinite(xyz), axis=1)
            if valid.sum() < 20:
                return []
            label_positions[label] = xyz[valid]
        except Exception:
            return []

    if len(label_positions) < 3:
        return []

    # Analyse géométrique des 3 trackers
    centers = {lbl: np.median(pos, axis=0) for lbl, pos in label_positions.items()}

    # Critère principal : le tracker "head" doit être le plus haut (axe Y en VR)
    # et doit être central horizontalement (médiane des deux autres)
    labels = ["head", "left", "right"]
    y_vals = np.array([centers[lbl][1] for lbl in labels])
    x_vals = np.array([centers[lbl][0] for lbl in labels])

    # Le head devrait avoir le Y le plus élevé
    head_by_y = labels[int(np.argmax(y_vals))]

    # Confiance : séparation relative entre le leader et le 2ème
    y_sorted = np.sort(y_vals)[::-1]
    y_range = np.ptp(y_vals) + 1e-9
    y_sep = (y_sorted[0] - y_sorted[1]) / y_range

    current_head = "head"
    confidence_ok = head_by_y == current_head

    problems = []
    if not confidence_ok and y_sep > 0.30:
        # La géométrie suggère que le "head" actuel n'est PAS le tracker le plus haut
        problems.append(make_problem(
            ProblemCode.TRACKER_MISPLACED,
            f"Tracker 'head' n'est pas le plus haut (Y={centers['head'][1]:.3f}) — "
            f"le tracker '{head_by_y}' devrait être 'head' (Y={centers[head_by_y][1]:.3f}, "
            f"séparation={y_sep:.2f})",
            predicted_head=head_by_y,
            y_separation=round(y_sep, 3),
            centers={lbl: centers[lbl].tolist() for lbl in labels},
        ))

    # Vérification gauche/droite :
    # Dans le repère VR, left_x < head_x < right_x (environ)
    # Mais c'est relatif à l'orientation du head, donc moins fiable sans quaternions
    # On vérifie juste une incohérence flagrante
    if "left" in centers and "right" in centers:
        left_x = centers["left"][0]
        right_x = centers["right"][0]
        x_sep = abs(right_x - left_x)
        if x_sep > 0.05 and right_x < left_x:
            # Right est à gauche de Left — incohérent
            problems.append(make_problem(
                ProblemCode.TRACKER_MISPLACED,
                f"Trackers left/right inversés : left_x={left_x:.3f} > right_x={right_x:.3f}",
                x_separation=round(x_sep, 3),
                centers={lbl: centers[lbl].tolist() for lbl in labels},
            ))

    return problems


def _diagnose_sync_lag(
    session_path: Path,
    tracker_t_ms: np.ndarray,
    ia_score: float,
) -> List[DiagnosedProblem]:
    """
    Détecte un lag résiduel sub-seconde entre caméra et tracker.
    Seulement diagnostiqué si toutes les autres portes sont passées
    mais le score IA reste faible.
    """
    problems = []

    if ia_score >= LOW_IA_SCORE_THRESHOLD:
        return []

    # Chercher le lag optimal par cross-corrélation sur chaque caméra
    best_lags = {}
    for cam in ("head", "left", "right"):
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        times = _load_jsonl_times(jsonl_path)
        if times is None or len(times) < 30:
            continue

        # Signal caméra : IFI centré (variation de densité de frames)
        ifi = np.diff(times, prepend=times[0])
        med_ifi = float(np.median(ifi))
        if med_ifi < 1e-3:
            continue
        cam_signal = np.abs(ifi - med_ifi)
        cam_signal = (cam_signal - np.mean(cam_signal)) / (np.std(cam_signal) + 1e-8)

        # Signal tracker : vitesse de déplacement
        if len(tracker_t_ms) < 20:
            continue

        # Interpoler sur grille commune
        trk_t0 = float(tracker_t_ms[0])
        cam_t0 = float(times[0])
        delta_ms = cam_t0 - trk_t0

        # Grille commune
        t_start = max(trk_t0, cam_t0 - delta_ms)
        t_end   = min(float(tracker_t_ms[-1]), float(times[-1]) - delta_ms)
        if t_end - t_start < 1000.0:
            continue

        grid = np.arange(t_start, t_end, 10.0)
        if len(grid) < 50:
            continue

        trk_interp = np.interp(grid, tracker_t_ms, np.ones(len(tracker_t_ms)))  # placeholder
        cam_t_rel = times - cam_t0 + trk_t0
        cam_interp = np.interp(grid, cam_t_rel, cam_signal)

        # Cross-corrélation
        # (simplifiée sans tracker signal — utiliser seulement si tracker signal dispo)
        best_lags[cam] = 0.0  # placeholder

    if ia_score < LOW_IA_SCORE_THRESHOLD:
        problems.append(make_problem(
            ProblemCode.SYNC_LAG,
            f"Score IA faible ({ia_score:.3f} < {LOW_IA_SCORE_THRESHOLD}) — "
            f"lag résiduel possible entre caméra et tracker",
            ia_score=ia_score,
            best_lags=best_lags,
        ))

    return problems


def _diagnose_gripper_sync(session_path: Path) -> List[DiagnosedProblem]:
    """
    Détecte un décalage temporel entre le flux capteur gripper et la vidéo.

    Utilise l'analyse niveau 1 (timestamps uniquement, rapide, sans MP4) :
    - Δt_start = t_camera[0] - t_gripper[0]
    - Si |Δt_start| > GRIPPER_OFFSET_SIGNIFICANT_MS → GRIPPER_SYNC

    La correction précise (convolution position fermée) est effectuée lors du fix.
    """
    if not _PANDAS:
        return []

    problems: List[DiagnosedProblem] = []
    max_delta_ms = 0.0
    details: dict = {}

    for side in ("left", "right"):
        gripper_path = session_path / f"gripper_{side}_data.csv"
        jsonl_path   = session_path / "videos" / f"{side}.jsonl"

        gripper_df = _load_gripper_df_gsync(gripper_path)
        cam_times  = _load_jsonl_times_gsync(jsonl_path)

        if gripper_df is None or cam_times is None:
            continue

        result = _gripper_level1(side, gripper_df, cam_times)
        abs_delta = abs(result.temporal_delta_start_ms)
        max_delta_ms = max(max_delta_ms, abs_delta)

        details[side] = {
            "delta_start_ms": result.temporal_delta_start_ms,
            "overlap_ms":     result.temporal_overlap_ms,
            "status":         result.status,
        }

    if not details:
        return []

    if max_delta_ms > GRIPPER_OFFSET_SIGNIFICANT_MS:
        problems.append(make_problem(
            ProblemCode.GRIPPER_SYNC,
            f"Décalage gripper/vidéo : {max_delta_ms:.0f}ms "
            f"(seuil {GRIPPER_OFFSET_SIGNIFICANT_MS:.0f}ms) — "
            f"correction par convolution disponible",
            max_delta_ms=round(max_delta_ms, 1),
            sides=details,
        ))

    return problems


# ══════════════════════════════════════════════════════════════════════════════
# Fonction principale de diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_session(
    session_path: Path,
    check_report,  # SessionReport from check.py
    deep: bool = True,
) -> List[DiagnosedProblem]:
    """
    Analyse un SessionReport et produit une liste de DiagnosedProblem triée par priorité.

    Args:
        session_path  : chemin vers la session
        check_report  : SessionReport retourné par check.check_session()
        deep          : si True, effectue des mesures approfondies au-delà des portes

    Returns:
        Liste de DiagnosedProblem triée par priorité (priorité basse = à corriger en premier)
    """
    problems: List[DiagnosedProblem] = []

    # ── 1. Convertir les gates du rapport check en problèmes ──────────────────
    failed_gates = {g.name for g in check_report.gates if not g.passed}
    passed_gates = {g.name for g in check_report.gates if g.passed}

    # Fichiers manquants / metadata
    if "structure" in failed_gates:
        gate = next(g for g in check_report.gates if g.name == "structure")
        problems.append(make_problem(
            ProblemCode.MISSING_FILES,
            gate.message or "Fichiers requis manquants",
        ))
        return problems  # Fatal : arrêt immédiat

    if "metadata" in failed_gates:
        gate = next(g for g in check_report.gates if g.name == "metadata")
        problems.append(make_problem(
            ProblemCode.METADATA_CORRUPT,
            gate.message or "metadata.json illisible",
        ))
        return problems  # Fatal

    # Quaternions
    if "quaternions" in failed_gates:
        gate = next(g for g in check_report.gates if g.name == "quaternions")
        problems.append(make_problem(
            ProblemCode.QUATERNION_CORRUPT,
            gate.message or "Quaternions corrompus",
            frac_invalid=gate.value,
        ))

    # Gaps tracker
    if "tracker_continuity" in failed_gates:
        gate = next(g for g in check_report.gates if g.name == "tracker_continuity")
        problems.append(make_problem(
            ProblemCode.TRACKER_GAPS,
            gate.message or "Trop de gaps tracker",
            n_gaps=gate.value,
        ))

    # Gaps caméra
    if "camera_continuity" in failed_gates:
        gate = next(g for g in check_report.gates if g.name == "camera_continuity")
        problems.append(make_problem(
            ProblemCode.CAMERA_GAPS,
            gate.message or "Trop de gaps caméra",
            n_gaps=gate.value,
        ))

    # Couverture caméra insuffisante → possible offset ou mauvais assignement
    if "camera_coverage" in failed_gates:
        gate = next(g for g in check_report.gates if g.name == "camera_coverage")
        problems.append(make_problem(
            ProblemCode.CAMERA_OFFSET,
            f"Couverture caméra insuffisante : {gate.message}",
            coverage=gate.value,
        ))

    # ── 2. Analyse approfondie (indépendante des gates) ───────────────────────
    if deep:
        tracker_t_ms = _load_tracker_times(session_path)

        # Offset et drift caméra (même si camera_coverage est OK)
        if tracker_t_ms is not None and len(tracker_t_ms) >= 10:
            offset_problems = _diagnose_camera_offset(session_path, tracker_t_ms)
            for p in offset_problems:
                # Ne pas dupliquer si CAMERA_OFFSET déjà détecté via gate
                if p.code not in {pp.code for pp in problems}:
                    problems.append(p)
                elif p.code == ProblemCode.CLOCK_DRIFT:
                    problems.append(p)

        # Placement tracker
        tracker_problems = _diagnose_tracker_placement(session_path)
        problems.extend(tracker_problems)

        # Lag résiduel si score IA faible et pas de gate critique
        ia_score = getattr(check_report, "ia_score", 0.0)
        if not failed_gates or failed_gates.issubset({"camera_coverage", "camera_continuity"}):
            lag_problems = _diagnose_sync_lag(session_path, tracker_t_ms or np.array([]), ia_score)
            problems.extend(lag_problems)

        # Synchronisation gripper ↔ vidéo (timestamps, sans MP4)
        gripper_problems = _diagnose_gripper_sync(session_path)
        for p in gripper_problems:
            if p.code not in {pp.code for pp in problems}:
                problems.append(p)

    # ── 3. Score IA global faible (dernier recours) ───────────────────────────
    ia_score = getattr(check_report, "ia_score", 0.0)
    score    = getattr(check_report, "score", 0.0)
    if score < 45.0 and not failed_gates and not problems:
        problems.append(make_problem(
            ProblemCode.LOW_IA_SCORE,
            f"Score IA global faible : {score:.1f}% — session potentiellement inutilisable",
            score=score,
            ia_score=ia_score,
        ))

    # ── 4. Trier par priorité ─────────────────────────────────────────────────
    problems.sort(key=lambda p: p.priority)

    return problems


def format_diagnosis(problems: List[DiagnosedProblem]) -> str:
    """Formatte un résumé textuel des problèmes diagnostiqués."""
    if not problems:
        return "  ✓ Aucun problème détecté"

    lines = []
    for p in problems:
        icon = "✓" if p.recoverable else "✗"
        lines.append(f"  {icon} [{p.code.value}] {p.message}")
    return "\n".join(lines)
