#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/test_fix_camera_labels.py — Validation de fix_camera_labels.

Tests :
1. Données synthétiques : vérifie que _assign_cameras identifie correctement
   la caméra HEAD (flux faible) et discrimine LEFT/RIGHT par mouvement asymétrique.
2. Données réelles (selection 2/do) : pipeline complet sur sessions avec mp4.
   Mesure la précision all-3 (head+left+right corrects).

Usage :
    python -m fix.test_fix_camera_labels
    python fix/test_fix_camera_labels.py
"""
from __future__ import annotations

import sys
import glob
import json
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fix.fix_camera_labels import (
    _assign_cameras,
    _optical_flow_signal,
    _tracker_speed_at,
)
from fix.fix_tracker_labels import (
    _load_blocks, _test_height, _test_centrality,
    _test_mobility, _test_lateral, _consensus,
)

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ── TEST 1 : données synthétiques ─────────────────────────────────────────────

def test_synthetic():
    """
    Construit des signaux synthétiques pour 3 caméras (head, left, right)
    avec des vitesses de tracker connues et vérifie l'assignement.
    """
    print("=" * 60)
    print("TEST 1 : DONNÉES SYNTHÉTIQUES")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N = 500   # frames

    # Tracker prediction (labels corrects)
    tracker_prediction = {"head": "head", "left": "left", "right": "right"}

    failures = []

    test_scenarios = [
        ("Gauche rapide, droite lente, tête très lente",  3.0, 1.0, 0.2),
        ("Droite rapide, gauche lente, tête très lente",  1.0, 3.0, 0.2),
        ("Activité égale, tête lente",                    2.0, 2.0, 0.3),
        ("Pics alternés (asym forte)",                    None, None, 0.2),
        ("Tâche bimanale (moins asym)",                   2.5, 2.5, 0.4),
    ]

    for desc, amp_left, amp_right, amp_head in test_scenarios:
        if amp_left is None:
            # Alternating peaks
            t = np.arange(N)
            trk_left  = np.where((t // 50) % 2 == 0, 3.0, 0.3)
            trk_right = np.where((t // 50) % 2 == 1, 3.0, 0.3)
        else:
            trk_left  = amp_left  * np.abs(rng.normal(0, 1, N)) + 0.1
            trk_right = amp_right * np.abs(rng.normal(0, 1, N)) + 0.1

        trk_head = amp_head * np.abs(rng.normal(0, 1, N)) + 0.05

        # Camera signals: correlated with their tracker + noise
        cam_left_flow  = trk_left  * 0.8 + rng.normal(0, 0.2, N)
        cam_right_flow = trk_right * 0.8 + rng.normal(0, 0.2, N)
        cam_head_flow  = trk_head  * 0.8 + rng.normal(0, 0.05, N)

        cam_left_flow  = np.abs(cam_left_flow)
        cam_right_flow = np.abs(cam_right_flow)
        cam_head_flow  = np.abs(cam_head_flow)

        times = np.arange(N, dtype=float) * 33.0

        # Build cam_signals dict (mimic structure expected by _assign_cameras)
        cam_signals = {
            "left":  (cam_left_flow,  times),
            "right": (cam_right_flow, times),
            "head":  (cam_head_flow,  times),
        }

        # Build fake DataFrame for _assign_cameras
        # _assign_cameras calls _tracker_speed_at(df, csv_label, times_ms)
        # We need tracker_positions.csv columns + timestamp_ns
        ts_ns = (times * 1e6).astype(np.int64)
        df_data = {
            "timestamp_ns":   ts_ns,
            "tracker_left_x": np.cumsum(np.diff(np.r_[0, trk_left]) * 0.033),
            "tracker_left_y": np.zeros(N),
            "tracker_left_z": np.zeros(N),
            "tracker_right_x": np.cumsum(np.diff(np.r_[0, trk_right]) * 0.033),
            "tracker_right_y": np.zeros(N),
            "tracker_right_z": np.zeros(N),
            "tracker_head_x": np.zeros(N),
            "tracker_head_y": np.cumsum(np.diff(np.r_[0, trk_head]) * 0.033),
            "tracker_head_z": np.zeros(N),
        }
        df = pd.DataFrame(df_data)

        assignment, scores, confidence = _assign_cameras(
            cam_signals, df, tracker_prediction
        )

        ok = (
            assignment.get("left") == "left"
            and assignment.get("right") == "right"
            and assignment.get("head") == "head"
        )
        status = "✓" if ok else "✗"
        print(
            f"  {status}  {desc:<40s}  "
            f"assign={assignment}  conf={confidence:.3f}"
        )
        if not ok:
            failures.append(desc)

    n = len(test_scenarios)
    print()
    if not failures:
        print(f"✓ RÉSULTAT TEST 1 : {n}/{n} PASSÉS (100%)")
    else:
        print(f"✗ RÉSULTAT TEST 1 : {len(failures)}/{n} échoués")
    return len(failures) == 0


# ── TEST 2 : données réelles ──────────────────────────────────────────────────

def test_real_data():
    """
    Pipeline complet sur sessions avec mp4.
    Compare l'assignement prédit au nom de fichier actuel.

    NOTE : certains "échecs" peuvent être des corrections légitimes
    (sessions où les métadonnées étaient déjà incorrectes).
    """
    if not _PANDAS:
        print("\n[TEST 2 SKIPPED : pandas non disponible]")
        return True

    try:
        import cv2
    except ImportError:
        print("\n[TEST 2 SKIPPED : cv2 non disponible]")
        return True

    print()
    print("=" * 60)
    print("TEST 2 : DONNÉES RÉELLES (sessions avec mp4)")
    print("=" * 60)

    sel2 = Path("/Users/christopher/selection 2/do")
    if not sel2.exists():
        print("  [Aucune session trouvée — test ignoré]")
        return True

    sessions = sorted(sel2.glob("*/"))

    total = 0
    ok_head = 0
    ok_lr = 0
    ok_all = 0
    uncertain_count = 0

    for s in sessions:
        if not (s / "videos" / "left.mp4").exists():
            continue

        # Load tracker CSV
        csv_path = s / "tracker_positions.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        blocks = _load_blocks(df)
        if not blocks:
            continue

        # Correct tracker labels
        t1 = _test_height(blocks)
        t2 = _test_centrality(blocks)
        t3 = _test_mobility(blocks)
        t4 = _test_lateral(blocks, t1.head_vote)
        tracker_prediction, _, _ = _consensus([t1, t2, t3, t4])

        # Optical flow for all cameras
        cam_signals = {}
        for cf in ("left", "right", "head"):
            mp4 = s / "videos" / f"{cf}.mp4"
            if not mp4.exists():
                continue
            mag, times = _optical_flow_signal(mp4)
            if mag is not None:
                cam_signals[cf] = (mag, times)

        if len(cam_signals) < 3:
            continue

        assignment, scores, confidence = _assign_cameras(
            cam_signals, df, tracker_prediction
        )

        total += 1
        head_ok  = assignment.get("head") == "head"
        left_ok  = assignment.get("left") == "left"
        right_ok = assignment.get("right") == "right"

        if head_ok:
            ok_head += 1
        if left_ok and right_ok:
            ok_lr += 1
        if head_ok and left_ok and right_ok:
            ok_all += 1

        if not (head_ok and left_ok and right_ok):
            note = ""
            if tracker_prediction != {"head": "head", "left": "left", "right": "right"}:
                note = "  [tracker swap: %s]" % {k: v for k, v in tracker_prediction.items() if k != v}
            print(
                f"  DIFF {s.name[:32]:<32s}  "
                f"assign={assignment}  conf={confidence:.3f}{note}"
            )

    if total == 0:
        print("  [Aucune session testée]")
        return True

    acc_head = 100 * ok_head / total
    acc_lr   = 100 * ok_lr   / total
    acc_all  = 100 * ok_all  / total

    print()
    print(f"  Sessions testées   : {total}")
    print(f"  Head correct       : {ok_head}/{total} = {acc_head:.1f}%")
    print(f"  L/R both correct   : {ok_lr}/{total}   = {acc_lr:.1f}%")
    print(f"  All-3 correct      : {ok_all}/{total}  = {acc_all:.1f}%")
    print()
    print(
        "  NOTE : les sessions marquées DIFF avec [tracker swap] ont des "
        "trackers CSV mislabeled — le pipeline les détecte et les corrige."
    )
    print()

    ok = acc_all >= 80.0
    print(f"{'✓' if ok else '✗'} RÉSULTAT TEST 2 : all-3 = {acc_all:.1f}% (seuil ≥ 80%)")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = []
    results.append(("Synthétique 100%",  test_synthetic()))
    results.append(("Réel ≥ 80%",        test_real_data()))

    print()
    print("=" * 60)
    print("BILAN GLOBAL")
    print("=" * 60)
    all_ok = True
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'}  {name}")
        if not ok:
            all_ok = False
    print()
    print("Résultat final : %s" % ("✓ TOUS LES TESTS PASSÉS" if all_ok else "✗ DES TESTS ONT ÉCHOUÉ"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
