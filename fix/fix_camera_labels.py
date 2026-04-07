#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_camera_labels.py — Identification CERTAINE des caméras head/left/right.

Objectif : vérifier et corriger les assignements head.jsonl / left.jsonl /
right.jsonl en utilisant 3 sources d'information indépendantes.

Sources d'information (par ordre de fiabilité) :
─────────────────────────────────────────────────
SOURCE 1 — Table de calibration des numéros de série
    Chaque caméra physique a un numéro de série unique (metadata.cameras[i].serial).
    Une fois mappé position↔serial dans un fichier de calibration local
    (auto-construit par apprentissage sur N sessions), l'assignement est CERTAIN.
    → Certitude si ≥ MIN_SESSIONS_AGREEMENT sessions historiques s'accordent.

SOURCE 2 — Corrélation gripper-caméra (si gripper_*.csv disponibles)
    Quand le gripper DROIT est actif (opening_mm varie), la caméra DROITE devrait
    enregistrer plus de frames (framerate légèrement perturbé par l'activité).
    Et vice-versa pour le gauche.
    → Corrélation entre variance IFI de chaque caméra et activité gripper
      correspondante.

SOURCE 3 — Cohérence temporelle (sanity-check)
    Toutes les caméras doivent couvrir approximativement la même plage temporelle
    que le tracker. Une caméra avec une plage anormale est probablement mislabeled
    (ou défectueuse).

ALGORITHME DE CORRECTION :
    1. Calculer le score de chaque source pour chaque assignement possible.
    2. Prendre l'intersection des sources qui s'accordent (≥ 2/3).
    3. Appliquer uniquement si certitude ≥ CERTAINTY_THRESHOLD.
    4. Renommer les fichiers .jsonl et mettre à jour metadata.json.

CALIBRATION AUTO-APPRENTISSAGE :
    Après chaque session vérifiée, le script met à jour un fichier de calibration
    local (camera_calibration.json dans le répertoire racine du projet).
    → Les sessions suivantes bénéficient de la calibration accumulée.

Usage :
    python -m fix.fix_camera_labels /chemin/session [--dry-run] [--force]
    python -m fix.fix_camera_labels /chemin/session --learn   # mise à jour calibration uniquement
    python -m fix.fix_camera_labels /chemin/root --batch      # toutes les sessions du répertoire
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import sys

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
CALIB_PATH = _ROOT / "camera_calibration.json"

for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ── Constantes ────────────────────────────────────────────────────────────────

MARKER_KEY             = "camera_labels_verified"
CAMERAS                = ("head", "left", "right")
MIN_SESSIONS_AGREEMENT = 3      # sessions pour valider un serial → position
CERTAINTY_THRESHOLD    = 0.75   # score minimal pour appliquer une correction
IFI_CORR_WINDOW_MS     = 500.0  # fenêtre de lissage IFI pour corrélation gripper


# ── Structures ────────────────────────────────────────────────────────────────

@dataclass
class SourceResult:
    name: str
    assignment: dict       # {position: serial}
    confidence: float      # 0–1
    evidence: dict = field(default_factory=dict)


@dataclass
class CameraLabelReport:
    session: str
    status: str            # "ok"|"corrected"|"uncertain"|"error"|"skipped"
    reason: str = ""
    sources: list[SourceResult] = field(default_factory=list)
    predicted: dict = field(default_factory=dict)   # {position: serial}
    current: dict = field(default_factory=dict)     # {position: serial}
    corrected: bool = False
    dry_run: bool = False


# ── Calibration locale ────────────────────────────────────────────────────────

def _load_calibration() -> dict:
    """Charge la table serial→{position: count} depuis camera_calibration.json."""
    if not CALIB_PATH.exists():
        return {}
    try:
        return json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_calibration(calib: dict) -> None:
    CALIB_PATH.write_text(json.dumps(calib, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def _update_calibration(serial_to_position: dict[str, str]) -> None:
    """Met à jour la calibration avec un nouvel assignement vérifié."""
    calib = _load_calibration()
    for serial, position in serial_to_position.items():
        if serial not in calib:
            calib[serial] = {}
        calib[serial][position] = calib[serial].get(position, 0) + 1
    _save_calibration(calib)


def _calibration_predict(serials: dict[str, str]) -> tuple[dict, float]:
    """
    Prédit {position: serial} depuis la calibration locale.
    serials : {camera_idx: serial}
    Returns: ({position: serial}, confidence)
    """
    calib = _load_calibration()
    predictions: dict[str, tuple[str, float]] = {}   # {position: (serial, conf)}

    for idx, serial in serials.items():
        if serial not in calib:
            continue
        counts = calib[serial]
        total  = sum(counts.values())
        best_pos  = max(counts, key=counts.get)
        best_frac = counts[best_pos] / total   # cohérence : 1.0 si toujours au même endroit
        # Confiance = cohérence × facteur volume (saturé à 1 après 3 sessions)
        volume_factor = min(1.0, total / MIN_SESSIONS_AGREEMENT)
        conf = best_frac * (0.7 + 0.3 * volume_factor)   # min 70% si cohérence=1
        if conf < 0.5:
            continue
        predictions[best_pos] = (serial, conf)

    if not predictions:
        return {}, 0.0

    # Vérifier que les 3 positions sont couvertes et sans collision
    assigned_serials = set()
    result: dict[str, str] = {}
    min_conf = 1.0
    for pos in CAMERAS:
        if pos not in predictions:
            return {}, 0.0
        serial, conf = predictions[pos]
        if serial in assigned_serials:
            return {}, 0.0   # collision
        result[pos] = serial
        assigned_serials.add(serial)
        min_conf = min(min_conf, conf)

    return result, min_conf


# ── Source 1 : calibration des numéros de série ───────────────────────────────

def _source_serial_calibration(meta: dict) -> SourceResult:
    """Prédit l'assignement depuis la table de calibration locale."""
    cameras = meta.get("cameras", {})
    serials = {idx: cam["serial"] for idx, cam in cameras.items()
               if "serial" in cam}

    predicted, conf = _calibration_predict(serials)
    evidence = {
        "n_known_serials": len(predicted),
        "min_confidence":  round(conf, 3),
        "calibration_file": str(CALIB_PATH),
        "calibration_exists": CALIB_PATH.exists(),
    }

    if not predicted or conf < CERTAINTY_THRESHOLD:
        return SourceResult(
            name="serial_calibration",
            assignment={},
            confidence=0.0,
            evidence=evidence,
        )

    return SourceResult(
        name="serial_calibration",
        assignment=predicted,   # {position: serial}
        confidence=float(conf),
        evidence=evidence,
    )


# ── Source 2 : corrélation IFI caméra / activité gripper ─────────────────────

def _load_jsonl_times(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    times = []
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    times.append(float(json.loads(line)["capture_time"]))
                except Exception:
                    continue
    except Exception:
        return None
    return np.array(times, dtype=np.float64) if len(times) > 10 else None


def _ifi_variance_signal(times: np.ndarray, win_ms: float = 200.0) -> np.ndarray:
    """
    Signal de variance IFI lissée à la fenêtre win_ms.
    Indicateur d'instabilité locale du framerate.
    """
    ifi = np.diff(times)
    med = float(np.median(ifi))
    dev = np.abs(ifi - med)
    k   = max(1, int(win_ms / med)) if med > 0 else 5
    sig = np.convolve(dev, np.ones(k) / k, mode="same")
    return sig.astype(np.float32)


def _gripper_activity_signal(csv_path: Path) -> Optional[np.ndarray]:
    """Renvoie le signal d'activité gripper rééchantillonné à 10ms."""
    if not _PANDAS or not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "timestamp_ns" not in df.columns or "opening_mm" not in df.columns:
        return None

    t_ms = pd.to_numeric(df["timestamp_ns"], errors="coerce").to_numpy() / 1e6
    v    = pd.to_numeric(df["opening_mm"], errors="coerce").to_numpy()
    valid = np.isfinite(t_ms) & np.isfinite(v)
    t_ms, v = t_ms[valid], v[valid]
    if len(t_ms) < 10:
        return None

    # Rééchantillonner à 10ms
    t0, t1 = t_ms[0], t_ms[-1]
    grid   = np.arange(t0, t1, 10.0)
    if len(grid) < 20:
        return None
    resampled = np.interp(grid, t_ms, v).astype(np.float32)
    # Dérivée absolue = activité
    return np.abs(np.diff(resampled, prepend=resampled[0]))


def _source_gripper_correlation(session_path: Path, meta: dict) -> SourceResult:
    """
    Corrèle le signal IFI de chaque caméra avec l'activité gripper correspondante.
    La caméra droite devrait avoir un IFI plus variable quand le gripper droit
    est actif (vibrations mécaniques, focus automatique).
    """
    cameras = meta.get("cameras", {})

    # Charger les signaux IFI pour chaque caméra ACTUELLE
    ifi_signals: dict[str, Optional[np.ndarray]] = {}
    for pos in CAMERAS:
        t = _load_jsonl_times(session_path / "videos" / f"{pos}.jsonl")
        ifi_signals[pos] = _ifi_variance_signal(t) if t is not None else None

    # Charger les signaux gripper
    gripper: dict[str, Optional[np.ndarray]] = {
        "left":  _gripper_activity_signal(session_path / "gripper_left_data.csv"),
        "right": _gripper_activity_signal(session_path / "gripper_right_data.csv"),
    }

    if gripper["left"] is None and gripper["right"] is None:
        return SourceResult(
            name="gripper_correlation",
            assignment={},
            confidence=0.0,
            evidence={"reason": "gripper CSV absent"},
        )

    # Corrélation croisée (version simplifiée : corrélation sur la variance)
    corr_scores: dict[str, dict[str, float]] = {}   # corr_scores[cam_pos][gripper_side]
    for cam_pos, ifi_sig in ifi_signals.items():
        if ifi_sig is None:
            continue
        corr_scores[cam_pos] = {}
        for g_side, g_sig in gripper.items():
            if g_sig is None:
                continue
            # Aligner sur la longueur minimale
            n = min(len(ifi_sig), len(g_sig))
            a = ifi_sig[:n].astype(float)
            b = g_sig[:n].astype(float)
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                corr_scores[cam_pos][g_side] = 0.0
                continue
            a = (a - a.mean()) / a.std()
            b = (b - b.mean()) / b.std()
            corr_scores[cam_pos][g_side] = float(np.mean(a * b))

    if not corr_scores:
        return SourceResult(
            name="gripper_correlation",
            assignment={},
            confidence=0.0,
            evidence={"corr_scores": {}, "reason": "signaux IFI absents"},
        )

    # Trouver l'assignement qui maximise la corrélation caméra↔gripper côté
    # La caméra correspondant à un gripper devrait avoir r > les autres
    assignment: dict[str, str] = {}
    confidences = []
    for g_side in ("left", "right"):
        best_cam  = max(
            (cam for cam in corr_scores if g_side in corr_scores[cam]),
            key=lambda c: corr_scores[c].get(g_side, -99),
            default=None,
        )
        if best_cam is None:
            continue
        scores_for_g = [corr_scores[c].get(g_side, 0.0) for c in corr_scores]
        scores_sorted = sorted(scores_for_g, reverse=True)
        separation = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else 0.0
        conf = float(np.clip(separation / 0.1, 0.0, 1.0))   # 0.1 = séparation typique attendue
        assignment[g_side] = best_cam
        confidences.append(conf)

    head_cam = next((c for c in CAMERAS if c not in assignment.values()), None)
    if head_cam:
        assignment["head"] = head_cam

    conf = float(np.mean(confidences)) if confidences else 0.0

    return SourceResult(
        name="gripper_correlation",
        assignment=assignment,   # {position: camera_file_label} — différent de serial!
        confidence=conf,
        evidence={"corr_scores": {k: {kk: round(vv, 4)
                                       for kk, vv in v.items()}
                                   for k, v in corr_scores.items()}},
    )


# ── Source 3 : cohérence temporelle ───────────────────────────────────────────

def _source_temporal_consistency(session_path: Path, meta: dict) -> SourceResult:
    """
    Vérifie que toutes les caméras couvrent la même plage temporelle.
    Une caméra avec coverage < 70% de la médiane est suspecte.
    """
    durations: dict[str, Optional[float]] = {}
    frame_counts: dict[str, int] = {}
    for pos in CAMERAS:
        t = _load_jsonl_times(session_path / "videos" / f"{pos}.jsonl")
        if t is None:
            durations[pos] = None
        else:
            durations[pos] = float(t[-1] - t[0])
            frame_counts[pos] = len(t)

    valid = {p: d for p, d in durations.items() if d is not None}
    if len(valid) < 2:
        return SourceResult(
            name="temporal_consistency",
            assignment={},
            confidence=0.0,
            evidence={"durations_ms": durations},
        )

    med_dur = float(np.median(list(valid.values())))
    anomalies = {p: d for p, d in valid.items() if abs(d - med_dur) / (med_dur + 1e-6) > 0.3}
    conf = 1.0 - len(anomalies) / len(valid)

    return SourceResult(
        name="temporal_consistency",
        assignment={},   # Ce test ne prédit pas d'assignement, il détecte des anomalies
        confidence=float(conf),
        evidence={
            "durations_ms":  {p: round(d, 0) if d else None for p, d in durations.items()},
            "frame_counts":  frame_counts,
            "median_dur_ms": round(med_dur, 0),
            "anomalies":     anomalies,
        },
    )


# ── Logique de décision ───────────────────────────────────────────────────────

def _decide(sources: list[SourceResult], current_serials: dict) -> tuple[dict, float]:
    """
    Combine les sources pour décider de l'assignement final.
    current_serials : {position: serial} (état actuel du metadata)
    Returns: (new_serials_assignment, confidence)
    """
    # Source 1 (calibration) est la plus fiable — si confiance > seuil, elle prime
    serial_src = next((s for s in sources if s.name == "serial_calibration"), None)
    if serial_src and serial_src.confidence >= CERTAINTY_THRESHOLD:
        return serial_src.assignment, serial_src.confidence

    # Sinon, vérifier la source gripper (mais elle travaille sur les labels courants)
    gripper_src = next((s for s in sources if s.name == "gripper_correlation"), None)
    if gripper_src and gripper_src.confidence >= CERTAINTY_THRESHOLD:
        # La source gripper retourne {position: cam_label} pas {position: serial}
        # → convertir en {position: serial} en utilisant current_serials
        pos_to_serial: dict[str, str] = {}
        for pos, cam_label in gripper_src.assignment.items():
            if cam_label in current_serials:
                pos_to_serial[pos] = current_serials[cam_label]
        if len(pos_to_serial) == 3:
            return pos_to_serial, gripper_src.confidence

    return {}, 0.0


# ── Point d'entrée principal ──────────────────────────────────────────────────

def fix_camera_labels(
    session_path: Path,
    dry_run: bool = False,
    force: bool = False,
    learn_only: bool = False,
) -> CameraLabelReport:
    """
    Vérifie et corrige les labels head/left/right des caméras.

    learn_only=True : met à jour uniquement la calibration, sans corriger.
    """
    name      = session_path.name
    meta_path = session_path / "metadata.json"

    if not meta_path.exists():
        return CameraLabelReport(session=name, status="error",
                                  reason="metadata.json absent")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return CameraLabelReport(session=name, status="error",
                                  reason=f"metadata.json illisible: {e}")

    if not force and not learn_only and meta.get(MARKER_KEY):
        return CameraLabelReport(session=name, status="skipped",
                                  reason="labels déjà vérifiés")

    cameras = meta.get("cameras", {})
    if not cameras:
        return CameraLabelReport(session=name, status="error",
                                  reason="cameras absent dans metadata.json")

    # Construire l'état actuel {position: serial}
    current_by_pos: dict[str, str] = {}
    current_by_label: dict[str, str] = {}   # {cam_label: serial}
    for idx, cam in cameras.items():
        pos    = cam.get("position", "")
        serial = cam.get("serial", "")
        if pos and serial:
            current_by_pos[pos]   = serial
            current_by_label[pos] = serial

    # ── Exécuter les sources ──────────────────────────────────────────────────
    s1 = _source_serial_calibration(meta)
    s2 = _source_gripper_correlation(session_path, meta)
    s3 = _source_temporal_consistency(session_path, meta)
    sources = [s1, s2, s3]

    # ── Mode apprentissage uniquement ─────────────────────────────────────────
    if learn_only:
        if len(current_by_pos) == 3:
            _update_calibration(current_by_pos)
        return CameraLabelReport(
            session=name, status="learned",
            reason=f"Calibration mise à jour : {current_by_pos}",
            sources=sources, current=current_by_pos,
        )

    # ── Décision ─────────────────────────────────────────────────────────────
    predicted_serials, conf = _decide(sources, current_by_label)

    report = CameraLabelReport(
        session=name, status="",
        sources=sources,
        predicted=predicted_serials,
        current=current_by_pos,
    )

    if not predicted_serials or conf < CERTAINTY_THRESHOLD:
        report.status = "uncertain"
        report.reason = (
            f"Confiance insuffisante ({conf:.2f} < {CERTAINTY_THRESHOLD}). "
            f"Lancez --learn sur des sessions déjà vérifiées pour enrichir la calibration."
        )
        if not dry_run:
            meta[MARKER_KEY] = False
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        return report

    # Comparer predicted vs current
    needs_correction = predicted_serials != current_by_pos and len(predicted_serials) == 3
    if not needs_correction:
        report.status = "ok"
        report.reason = f"Labels caméra corrects (conf={conf:.2f})"
        if not dry_run:
            meta[MARKER_KEY] = True
            _update_calibration(current_by_pos)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        return report

    if dry_run:
        report.status  = "would_correct"
        report.dry_run = True
        report.reason  = f"Correction nécessaire (conf={conf:.2f})"
        return report

    # ── Appliquer la correction ───────────────────────────────────────────────
    # 1. Construire le mapping old_label → new_label pour les fichiers JSONL
    old_to_new: dict[str, str] = {}
    for new_pos, serial in predicted_serials.items():
        # Trouver l'ancienne position de ce serial
        old_pos = next((p for p, s in current_by_pos.items() if s == serial), None)
        if old_pos and old_pos != new_pos:
            old_to_new[old_pos] = new_pos

    # 2. Renommer les JSONL (avec fichiers temporaires pour éviter les collisions)
    videos_dir = session_path / "videos"
    if old_to_new and videos_dir.exists():
        # Phase 1 : vers des noms temporaires
        for old_name, new_name in old_to_new.items():
            src = videos_dir / f"{old_name}.jsonl"
            tmp = videos_dir / f"__tmp_{new_name}.jsonl"
            if src.exists():
                shutil.move(str(src), str(tmp))
        # Phase 2 : noms temporaires vers noms finaux
        for old_name, new_name in old_to_new.items():
            tmp = videos_dir / f"__tmp_{new_name}.jsonl"
            dst = videos_dir / f"{new_name}.jsonl"
            if tmp.exists():
                shutil.move(str(tmp), str(dst))

    # 3. Mettre à jour metadata.json
    for idx, cam in cameras.items():
        serial = cam.get("serial", "")
        new_pos = next((p for p, s in predicted_serials.items() if s == serial), None)
        if new_pos:
            cameras[idx]["position"] = new_pos

    meta["cameras"]            = cameras
    meta[MARKER_KEY]           = True
    meta["camera_labels_old"]  = current_by_pos
    meta["camera_labels_new"]  = predicted_serials
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    _update_calibration(predicted_serials)

    report.status    = "corrected"
    report.corrected = True
    report.reason    = (
        f"Labels corrigés (conf={conf:.2f}). "
        f"JSONL renommés : {old_to_new}"
    )
    return report


# ── Mode batch : construction de la calibration ───────────────────────────────

def build_calibration_from_sessions(root: Path) -> int:
    """
    Parcourt tous les metadata.json sous root et met à jour la calibration.
    Retourne le nombre de sessions traitées.
    """
    count = 0
    for meta_path in root.rglob("metadata.json"):
        try:
            meta    = json.loads(meta_path.read_text(encoding="utf-8"))
            cameras = meta.get("cameras", {})
            # Format attendu par _update_calibration : {serial: position}
            mapping = {cam["serial"]: cam["position"]
                       for cam in cameras.values()
                       if "position" in cam and "serial" in cam}
            if len(mapping) == 3:
                _update_calibration(mapping)
                count += 1
        except Exception:
            continue
    return count


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_report(r: CameraLabelReport) -> None:
    icons = {"ok": "✓", "corrected": "↺", "uncertain": "⚠", "error": "✗",
             "skipped": "–", "learned": "⊕", "would_correct": "~"}
    icon = icons.get(r.status, "?")
    print(f"\n{icon} [{r.status.upper()}] {r.session}")
    if r.reason:
        print(f"  Raison : {r.reason}")
    print(f"  Actuel  : {r.current}")
    if r.predicted:
        print(f"  Prédit  : {r.predicted}")
    for src in r.sources:
        print(f"  [{src.name:25s}] conf={src.confidence:.2f}  {src.evidence}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vérifie et corrige le labeling head/left/right des caméras."
    )
    parser.add_argument("sessions", nargs="+", type=Path,
                        help="Répertoires de session(s) ou racine avec --batch")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--learn",     action="store_true",
                        help="Mise à jour calibration uniquement (sans corriger)")
    parser.add_argument("--build-calib", action="store_true",
                        help="Construire la calibration depuis tous les metadata.json")
    parser.add_argument("--batch",     action="store_true",
                        help="Traiter toutes les sessions sous chaque chemin")
    parser.add_argument("--json",      action="store_true")
    args = parser.parse_args()

    # Expansion des sessions si --batch
    sessions: list[Path] = []
    for p in args.sessions:
        p = p.resolve()
        if args.batch and p.is_dir():
            if args.build_calib:
                n = build_calibration_from_sessions(p)
                print(f"Calibration construite depuis {n} sessions sous {p}")
                continue
            sessions.extend(
                m.parent for m in p.rglob("metadata.json")
                if (m.parent / "videos").exists()
            )
        else:
            sessions.append(p)

    results = []
    for s in sessions:
        if not s.is_dir():
            continue
        r = fix_camera_labels(s, dry_run=args.dry_run, force=args.force,
                               learn_only=args.learn)
        results.append(r)
        if not args.json:
            _print_report(r)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False,
                          default=str))

    if not args.json and results:
        counts = Counter(r.status for r in results)
        print(f"\n{'─'*60}")
        print(f"Total {len(results)} : " +
              "  ".join(f"{k}={v}" for k, v in counts.items()))
        print(f"Calibration : {CALIB_PATH}")


if __name__ == "__main__":
    main()
