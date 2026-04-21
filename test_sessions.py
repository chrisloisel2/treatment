#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_sessions.py — Script unifié de test et tri des sessions.

Valide les algorithmes internes (trackers, caméras, gripper) puis
évalue chaque session d'un dossier d'entrée et les copie dans deux
dossiers de sortie selon leur grade :

  --out-ab   → sessions grade A ou B  (score ≥ 75)
  --out-rest → sessions grade C, D, F (score < 75)

Usage :
    python test_sessions.py --input /chemin/sessions \\
                            --out-ab /chemin/sortie_AB \\
                            --out-rest /chemin/sortie_autres

Options :
    --input   DIR   Dossier contenant les sessions à traiter
    --out-ab  DIR   Dossier de sortie pour les grades A et B
    --out-rest DIR  Dossier de sortie pour les grades C, D et F
    --no-unit-tests Ne pas exécuter les tests unitaires
    --no-copy       Afficher le tri sans copier les fichiers
    --json          Afficher le rapport en JSON
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import glob
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE / "fix")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import cv2  # noqa: F401
    _CV2 = True
except ImportError:
    _CV2 = False


# ══════════════════════════════════════════════════════════════════════════════
# Couleurs terminal
# ══════════════════════════════════════════════════════════════════════════════

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _c(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{_RESET}"
    return text


def _ok(msg: str)   -> str: return _c(f"✓  {msg}", _GREEN)
def _fail(msg: str) -> str: return _c(f"✗  {msg}", _RED)
def _warn(msg: str) -> str: return _c(f"●  {msg}", _YELLOW)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Tests unitaires : fix_tracker_labels
# ══════════════════════════════════════════════════════════════════════════════

def _quat_from_yaw(yaw_rad: float) -> np.ndarray:
    cy, sy = np.cos(yaw_rad / 2), np.sin(yaw_rad / 2)
    return np.array([cy, 0.0, sy, 0.0])


def _quat_rotmat(q: np.ndarray) -> np.ndarray:
    q = q.astype(float)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-z*w); R[:,0,2]=2*(x*z+y*w)
    R[:,1,0]=2*(x*y+z*w);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-x*w)
    R[:,2,0]=2*(x*z-y*w);   R[:,2,1]=2*(y*z+x*w);   R[:,2,2]=1-2*(x*x+y*y)
    return R


def _make_synthetic_tracker_blocks(yaw_rad: float, n: int = 300, noise: float = 0.005):
    q = _quat_from_yaw(yaw_rad)
    quat = np.tile(q, (n, 1)) + np.random.randn(n, 4) * 0.001
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = _quat_rotmat(quat)
    local_x = np.mean(R[:, :, 0], axis=0)
    head_pos  = np.array([0.0, 1.7, 0.0]) + np.random.randn(n, 3) * noise
    right_pos = head_pos +  local_x * 0.35 + np.array([0, -0.7, 0]) + np.random.randn(n, 3) * noise
    left_pos  = head_pos + -local_x * 0.35 + np.array([0, -0.7, 0]) + np.random.randn(n, 3) * noise
    return [
        ("head",  head_pos,  quat),
        ("right", right_pos, quat * 0 + np.array([1,0,0,0])),
        ("left",  left_pos,  quat * 0 + np.array([1,0,0,0])),
    ]


def test_tracker_synthetic() -> bool:
    """Test 1 : _test_lateral doit donner 100 % sur données synthétiques."""
    print("=" * 60)
    print("TEST TRACKER 1 : DONNÉES SYNTHÉTIQUES (orientations tête)")
    print("=" * 60)

    try:
        from fix.fix_tracker_labels import _test_lateral
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    yaws = np.radians(np.arange(0, 360, 15))
    failures = []

    for yaw in yaws:
        blocks = _make_synthetic_tracker_blocks(yaw)
        t4 = _test_lateral(blocks, "head")
        ok = (t4.right_vote == "right" and t4.left_vote == "left")
        deg = round(np.degrees(yaw))
        sym = "✓" if ok else "✗"
        print(f"  {sym}  yaw={deg:3d}°  right_vote={t4.right_vote}  left_vote={t4.left_vote}"
              f"  sep={t4.evidence.get('separation_m', 0.0):.3f}m")
        if not ok:
            failures.append(deg)

    n = len(yaws)
    if not failures:
        print(_ok(f"RÉSULTAT : {n}/{n} PASSÉS (100%)"))
    else:
        print(_fail(f"RÉSULTAT : {len(failures)}/{n} échoués aux yaw={failures}"))
    return len(failures) == 0


def test_tracker_consensus() -> bool:
    """Test 2 : le consensus utilise UNIQUEMENT test 4 pour left/right."""
    print()
    print("=" * 60)
    print("TEST TRACKER 2 : CONSENSUS LEFT/RIGHT ≡ TEST 4 UNIQUEMENT")
    print("=" * 60)

    try:
        from fix.fix_tracker_labels import (
            _test_height, _test_centrality, _test_mobility, _test_lateral, _consensus
        )
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    np.random.seed(42)
    blocks = _make_synthetic_tracker_blocks(0.0)

    t1 = _test_height(blocks)
    t2 = _test_centrality(blocks)
    t3 = _test_mobility(blocks)
    t4 = _test_lateral(blocks, t1.head_vote)
    predicted, agree_count, certain = _consensus([t1, t2, t3, t4])

    ok_right = predicted.get("right") == t4.right_vote
    ok_left  = predicted.get("left")  == t4.left_vote

    print(f"  t4 vote         : right={t4.right_vote}  left={t4.left_vote}")
    print(f"  consensus prédit: right={predicted.get('right')}  left={predicted.get('left')}")
    print(f"  right == t4.right_vote : {ok_right}")
    print(f"  left  == t4.left_vote  : {ok_left}")

    ok = ok_right and ok_left
    print()
    print(_ok("RÉSULTAT : consensus = test 4") if ok else _fail("RÉSULTAT : consensus ≠ test 4"))
    return ok


def test_tracker_real_data() -> bool:
    """Test 3 : pipeline complet sur données réelles (> 90 % accuracy)."""
    if not _PANDAS:
        print()
        print(_warn("TEST TRACKER 3 SKIPPED : pandas non disponible"))
        return True

    print()
    print("=" * 60)
    print("TEST TRACKER 3 : DONNÉES RÉELLES")
    print("=" * 60)

    try:
        from fix.fix_tracker_labels import (
            _load_blocks, _test_height, _test_centrality,
            _test_mobility, _test_lateral, _consensus
        )
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    session_dirs = [str(_HERE / "_training" / "desync")]
    sel2 = Path("/Users/christopher/selection 2/do")
    if sel2.exists():
        session_dirs.append(str(sel2))

    csvs = []
    for d in session_dirs:
        csvs += glob.glob(d + "/*/tracker_positions.csv")

    if not csvs:
        print("  [Aucune session trouvée — test ignoré]")
        return True

    ok_head = ok_right = total = 0
    for csv in csvs:
        df = pd.read_csv(csv)
        blocks = _load_blocks(df)
        if not blocks:
            continue
        t1 = _test_height(blocks)
        t2 = _test_centrality(blocks)
        t3 = _test_mobility(blocks)
        t4 = _test_lateral(blocks, t1.head_vote)
        predicted, _, _ = _consensus([t1, t2, t3, t4])
        total += 1
        if predicted.get("head")  == "head":  ok_head  += 1
        if predicted.get("right") == "right": ok_right += 1

    if total == 0:
        print("  [Aucune session traitée]")
        return True

    acc_head  = 100 * ok_head  / total
    acc_right = 100 * ok_right / total
    fails     = total - ok_right

    print(f"  Sessions testées : {total}")
    print(f"  Head accuracy    : {ok_head}/{total} = {acc_head:.1f}%")
    print(f"  Right accuracy   : {ok_right}/{total} = {acc_right:.1f}%")
    print(f"  NOTE : les {fails} sessions 'échouées' pour right sont probablement")
    print("  des sessions dont les étiquettes CSV sont déjà inversées.")

    ok = acc_right > 90.0
    print()
    print(_ok(f"RÉSULTAT : right accuracy = {acc_right:.1f}% (seuil > 90%)")
          if ok else _fail(f"RÉSULTAT : right accuracy = {acc_right:.1f}% (seuil > 90%)"))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Tests unitaires : fix_camera_labels
# ══════════════════════════════════════════════════════════════════════════════

def test_camera_synthetic() -> bool:
    """Test caméra 1 : assignement sur données synthétiques (100%)."""
    if not _PANDAS:
        print()
        print(_warn("TEST CAMÉRA 1 SKIPPED : pandas non disponible"))
        return True

    print()
    print("=" * 60)
    print("TEST CAMÉRA 1 : DONNÉES SYNTHÉTIQUES")
    print("=" * 60)

    try:
        from fix.fix_camera_labels import _assign_cameras
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    rng = np.random.default_rng(42)
    N   = 500
    tracker_prediction = {"head": "head", "left": "left", "right": "right"}
    failures = []

    scenarios = [
        ("Gauche rapide, droite lente, tête très lente",  3.0, 1.0, 0.2),
        ("Droite rapide, gauche lente, tête très lente",  1.0, 3.0, 0.2),
        ("Activité égale, tête lente",                    2.0, 2.0, 0.3),
        ("Pics alternés (asym forte)",                    None, None, 0.2),
        ("Tâche bimanale (moins asym)",                   2.5, 2.5, 0.4),
    ]

    for desc, amp_l, amp_r, amp_h in scenarios:
        if amp_l is None:
            t = np.arange(N)
            trk_left  = np.where((t // 50) % 2 == 0, 3.0, 0.3)
            trk_right = np.where((t // 50) % 2 == 1, 3.0, 0.3)
        else:
            trk_left  = amp_l * np.abs(rng.normal(0, 1, N)) + 0.1
            trk_right = amp_r * np.abs(rng.normal(0, 1, N)) + 0.1

        trk_head = amp_h * np.abs(rng.normal(0, 1, N)) + 0.05
        times    = np.arange(N, dtype=float) * 33.0
        ts_ns    = (times * 1e6).astype(np.int64)

        cam_signals = {
            "left":  (np.abs(trk_left  * 0.8 + rng.normal(0, 0.2, N)), times),
            "right": (np.abs(trk_right * 0.8 + rng.normal(0, 0.2, N)), times),
            "head":  (np.abs(trk_head  * 0.8 + rng.normal(0, 0.05, N)), times),
        }
        df = pd.DataFrame({
            "timestamp_ns":    ts_ns,
            "tracker_left_x":  np.cumsum(np.diff(np.r_[0, trk_left])  * 0.033),
            "tracker_left_y":  np.zeros(N),
            "tracker_left_z":  np.zeros(N),
            "tracker_right_x": np.cumsum(np.diff(np.r_[0, trk_right]) * 0.033),
            "tracker_right_y": np.zeros(N),
            "tracker_right_z": np.zeros(N),
            "tracker_head_x":  np.zeros(N),
            "tracker_head_y":  np.cumsum(np.diff(np.r_[0, trk_head]) * 0.033),
            "tracker_head_z":  np.zeros(N),
        })

        assignment, scores, confidence = _assign_cameras(cam_signals, df, tracker_prediction)
        ok = (
            assignment.get("left")  == "left"  and
            assignment.get("right") == "right" and
            assignment.get("head")  == "head"
        )
        sym = "✓" if ok else "✗"
        print(f"  {sym}  {desc:<40s}  assign={assignment}  conf={confidence:.3f}")
        if not ok:
            failures.append(desc)

    n = len(scenarios)
    if not failures:
        print(_ok(f"RÉSULTAT : {n}/{n} PASSÉS (100%)"))
    else:
        print(_fail(f"RÉSULTAT : {len(failures)}/{n} échoués"))
    return len(failures) == 0


def test_camera_real_data() -> bool:
    """Test caméra 2 : pipeline complet sur sessions réelles (≥ 80% all-3)."""
    if not _PANDAS or not _CV2:
        missing = [x for x, ok in [("pandas", _PANDAS), ("cv2", _CV2)] if not ok]
        print()
        print(_warn(f"TEST CAMÉRA 2 SKIPPED : {', '.join(missing)} non disponible"))
        return True

    print()
    print("=" * 60)
    print("TEST CAMÉRA 2 : DONNÉES RÉELLES (sessions avec mp4)")
    print("=" * 60)

    try:
        from fix.fix_camera_labels import _assign_cameras, _optical_flow_signal
        from fix.fix_tracker_labels import (
            _load_blocks, _test_height, _test_centrality,
            _test_mobility, _test_lateral, _consensus
        )
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    sel2 = Path("/Users/christopher/selection 2/do")
    if not sel2.exists():
        print("  [Aucune session trouvée — test ignoré]")
        return True

    total = ok_head = ok_lr = ok_all = 0

    for s in sorted(sel2.glob("*/")):
        if not (s / "videos" / "left.mp4").exists():
            continue
        csv_path = s / "tracker_positions.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        blocks = _load_blocks(df)
        if not blocks:
            continue

        t1 = _test_height(blocks)
        t2 = _test_centrality(blocks)
        t3 = _test_mobility(blocks)
        t4 = _test_lateral(blocks, t1.head_vote)
        tracker_pred, _, _ = _consensus([t1, t2, t3, t4])

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

        assignment, scores, confidence = _assign_cameras(cam_signals, df, tracker_pred)
        total += 1
        head_ok  = assignment.get("head")  == "head"
        left_ok  = assignment.get("left")  == "left"
        right_ok = assignment.get("right") == "right"

        if head_ok: ok_head += 1
        if left_ok and right_ok: ok_lr += 1
        if head_ok and left_ok and right_ok: ok_all += 1
        else:
            note = ""
            if tracker_pred != {"head": "head", "left": "left", "right": "right"}:
                note = f"  [tracker swap: {tracker_pred}]"
            print(f"  DIFF {s.name[:32]:<32s}  assign={assignment}  conf={confidence:.3f}{note}")

    if total == 0:
        print("  [Aucune session testée]")
        return True

    acc_head = 100 * ok_head / total
    acc_lr   = 100 * ok_lr   / total
    acc_all  = 100 * ok_all  / total
    print(f"\n  Sessions testées : {total}")
    print(f"  Head correct     : {ok_head}/{total} = {acc_head:.1f}%")
    print(f"  L/R both correct : {ok_lr}/{total}   = {acc_lr:.1f}%")
    print(f"  All-3 correct    : {ok_all}/{total}  = {acc_all:.1f}%")

    ok = acc_all >= 80.0
    print()
    print(_ok(f"RÉSULTAT : all-3 = {acc_all:.1f}% (seuil ≥ 80%)")
          if ok else _fail(f"RÉSULTAT : all-3 = {acc_all:.1f}% (seuil ≥ 80%)"))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Tests unitaires : fix_gripper_video_sync
# ══════════════════════════════════════════════════════════════════════════════

def _make_gripper_signals(
    n: int = 1500, dt_ms: float = 33.0, true_offset_ms: float = 0.0,
    close_period: int = 100, close_duration: int = 30,
    noise_mm: float = 0.5, noise_vis: float = 0.04,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(42)
    sensor_t_ms = np.arange(n, dtype=float) * dt_ms
    sensor_mm   = np.ones(n) * 45.0
    for start in range(0, n, close_period):
        sensor_mm[start : min(start + close_duration, n)] = 3.5
    sensor_mm += rng.normal(0, noise_mm, n)
    sensor_mm  = np.clip(sensor_mm, 0.5, 60.0)
    visual_t_ms = sensor_t_ms + true_offset_ms
    visual_frac = np.clip(
        (sensor_mm - 0.5) / (60.0 - 0.5) + rng.normal(0, noise_vis, n), 0.0, 1.0
    )
    return sensor_t_ms, sensor_mm, visual_t_ms, visual_frac


def test_gripper_offset_detection() -> bool:
    """Test gripper 1 : détection d'offset connu (±tolérance 2× resample)."""
    print()
    print("=" * 60)
    print("TEST GRIPPER 1 : DÉTECTION D'OFFSET — DONNÉES SYNTHÉTIQUES")
    print("=" * 60)

    try:
        from fix.fix_gripper_video_sync import _compute_closed_offset, CONV_RESAMPLE_MS
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    tolerance_ms = 2.0 * CONV_RESAMPLE_MS
    failures = []

    cases = [
        ("offset=   0ms",    0.0),
        ("offset= +150ms",  150.0),
        ("offset= -200ms", -200.0),
        ("offset= +500ms",  500.0),
        ("offset= -500ms", -500.0),
        ("offset=+1000ms", 1000.0),
        ("offset=-1000ms",-1000.0),
    ]

    for label, true_ms in cases:
        s_t, s_mm, v_t, v_f = _make_gripper_signals(
            true_offset_ms=true_ms, noise_mm=0.3, noise_vis=0.04,
            rng=np.random.default_rng(7),
        )
        offset_ms, peak_r, snr, n_s, n_v = _compute_closed_offset(s_t, s_mm, v_t, v_f)
        error_ms = abs(offset_ms - true_ms)
        ok = (error_ms <= tolerance_ms) and (snr >= 2.0)
        sym = "✓" if ok else "✗"
        print(f"  {sym}  {label}  détecté={offset_ms:+8.1f}ms  "
              f"erreur={error_ms:5.1f}ms  SNR={snr:5.1f}  r={peak_r:.3f}")
        if not ok:
            reasons = []
            if error_ms > tolerance_ms:
                reasons.append(f"erreur {error_ms:.0f}ms > tolérance {tolerance_ms:.0f}ms")
            if snr < 2.0:
                reasons.append(f"SNR={snr:.1f} < 2")
            failures.append(f"{label}: {', '.join(reasons)}")

    n = len(cases)
    if not failures:
        print(_ok(f"RÉSULTAT : {n}/{n} PASSÉS (100%)"))
    else:
        for f in failures:
            print(f"  ✗ {f}")
        print(_fail(f"RÉSULTAT : {len(failures)}/{n} ÉCHOUÉS"))
    return len(failures) == 0


def test_gripper_noise_robustness() -> bool:
    """Test gripper 2 : robustesse au bruit visuel jusqu'à 15 %."""
    print()
    print("=" * 60)
    print("TEST GRIPPER 2 : ROBUSTESSE AU BRUIT VISUEL")
    print("=" * 60)

    try:
        from fix.fix_gripper_video_sync import _compute_closed_offset, CONV_RESAMPLE_MS
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    true_ms      = 300.0
    tolerance_ms = 2.0 * CONV_RESAMPLE_MS
    failures     = []

    for noise in (0.02, 0.05, 0.10, 0.15, 0.20):
        s_t, s_mm, v_t, v_f = _make_gripper_signals(
            true_offset_ms=true_ms, noise_mm=0.3, noise_vis=noise,
            rng=np.random.default_rng(99),
        )
        offset_ms, peak_r, snr, _, _ = _compute_closed_offset(s_t, s_mm, v_t, v_f)
        error_ms = abs(offset_ms - true_ms)
        ok = (error_ms <= tolerance_ms) and (snr >= 2.0)
        sym = "✓" if ok else ("✗ (acceptable)" if noise > 0.15 else "✗")
        print(f"  {sym}  bruit={noise:.2f}  détecté={offset_ms:+8.1f}ms  "
              f"erreur={error_ms:5.1f}ms  SNR={snr:5.1f}")
        if not ok and noise <= 0.15:
            failures.append(f"bruit={noise:.2f}")

    if not failures:
        print(_ok("RÉSULTAT : robuste jusqu'à bruit=0.15"))
    else:
        print(_fail(f"RÉSULTAT : échec pour {failures}"))
    return len(failures) == 0


def test_gripper_edge_cases() -> bool:
    """Test gripper 3 : gestion des cas dégénérés."""
    print()
    print("=" * 60)
    print("TEST GRIPPER 3 : CAS LIMITES")
    print("=" * 60)

    try:
        from fix.fix_gripper_video_sync import _compute_closed_offset, CONV_MIN_CLOSED_FRAMES
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    failures = []
    rng = np.random.default_rng(1)
    n   = 600

    # Capteur toujours ouvert
    t  = np.arange(n, dtype=float) * 33.0
    mm = np.ones(n) * 45.0 + rng.normal(0, 0.3, n)
    vf = np.ones(n) * 0.9  + rng.normal(0, 0.01, n)
    r  = _compute_closed_offset(t, mm, t, vf)
    ok = r[3] < CONV_MIN_CLOSED_FRAMES
    print(f"  {'✓' if ok else '✗'}  Capteur toujours ouvert  → n_s={r[3]} "
          f"({'ignoré' if ok else 'NON IGNORÉ'})")
    if not ok: failures.append("capteur toujours ouvert non ignoré")

    # Signal visuel constant
    s_t, s_mm, v_t, _ = _make_gripper_signals(n=n, true_offset_ms=0.0, rng=rng)
    r = _compute_closed_offset(s_t, s_mm, v_t, np.ones(n) * 0.8)
    ok = r == (0.0, 0.0, 0.0, 0, 0)
    print(f"  {'✓' if ok else '✗'}  Visuel constant          → retour={r[:3]} "
          f"({'ignoré' if ok else 'NON IGNORÉ'})")
    if not ok: failures.append("visuel constant non ignoré")

    # Chevauchement temporel insuffisant
    s_t2 = np.arange(30, dtype=float) * 50.0
    v_t2 = s_t2 + 5000.0
    mm2  = np.ones(30) * 45.0; mm2[:10] = 3.0
    vf2  = np.ones(30) * 0.9;  vf2[:10] = 0.05
    r    = _compute_closed_offset(s_t2, mm2, v_t2, vf2)
    ok   = (r[0] == 0.0 and r[1] == 0.0)
    print(f"  {'✓' if ok else '✗'}  Chevauchement insuffisant → retour={r[:2]} "
          f"({'ignoré' if ok else 'NON IGNORÉ'})")
    if not ok: failures.append("overlap insuffisant non ignoré")

    if not failures:
        print(_ok("RÉSULTAT : tous les cas limites gérés"))
    else:
        print(_fail(f"RÉSULTAT : {len(failures)} cas non gérés"))
    return len(failures) == 0


def test_gripper_real_data() -> bool:
    """Test gripper 4 : fix dry-run sur sessions réelles."""
    if not _PANDAS:
        print()
        print(_warn("TEST GRIPPER 4 SKIPPED : pandas non disponible"))
        return True

    print()
    print("=" * 60)
    print("TEST GRIPPER 4 : FIX DRY-RUN — DONNÉES RÉELLES")
    print("=" * 60)

    try:
        from fix.fix_gripper_video_sync import fix_gripper_closed_offset
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    roots = [
        Path("/Users/christopher/selection 2/do"),
        _HERE,
    ]
    sessions: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for csv in sorted(root.rglob("gripper_left_data.csv"))[:8]:
            sessions.append(csv.parent)

    if not sessions:
        print("  [Aucune session avec CSV gripper — test ignoré]")
        return True

    failures = []
    n_ok = n_corrected = 0

    for s in sessions:
        result       = fix_gripper_closed_offset(s, dry_run=True)
        status       = result["status"]
        corrections  = result["corrections_ms"]
        details      = result.get("details", {})
        ok           = isinstance(status, str) and isinstance(corrections, dict)
        sym          = "✓" if ok else "✗"
        corr_str     = (
            "  ".join(f"{side}:{ms:+.0f}ms" for side, ms in corrections.items())
            if corrections else "aucune correction"
        )
        print(f"  {sym}  {s.name[:38]:<38s}  status={status:<12s}  {corr_str}")
        for side, d in details.items():
            print(f"       [{side}]  offset={d.get('offset_ms',0):+7.1f}ms  "
                  f"SNR={d.get('snr',0.0):4.1f}  "
                  f"n_s={d.get('n_closed_sensor',0):3d}  "
                  f"n_v={d.get('n_closed_visual',0):3d}  "
                  f"st={d.get('status','?')}")
        if ok: n_ok += 1
        else:  failures.append(s.name)
        if corrections: n_corrected += 1

    print(f"\n  Sessions analysées         : {len(sessions)}")
    print(f"  Retour valide              : {n_ok}/{len(sessions)}")
    print(f"  Corrections détectées (dry): {n_corrected}")

    if not failures:
        print(_ok("RÉSULTAT : retour fix_gripper_closed_offset correct"))
    else:
        print(_fail(f"RÉSULTAT : retour invalide pour {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Évaluation des sessions et tri A/B vs autres
# ══════════════════════════════════════════════════════════════════════════════

def _grade_label(score: float) -> str:
    if score >= 90.0: return "A"
    if score >= 75.0: return "B"
    if score >= 60.0: return "C"
    if score >= 45.0: return "D"
    return "F"


def _is_ab(grade: str) -> bool:
    return grade in ("A", "B")


def _grade_color(grade: str) -> str:
    return _GREEN if grade in ("A", "B") else (_YELLOW if grade in ("C", "D") else _RED)


def evaluate_sessions(
    input_dir: Path,
    out_ab: Path,
    out_rest: Path,
    dry_run: bool = False,
) -> list[dict]:
    """
    Évalue chaque session dans input_dir et copie/déplace vers out_ab ou out_rest.

    Une session est un sous-dossier contenant metadata.json OU tracker_positions.csv.
    Retourne la liste des résultats.
    """
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "session_check",
            _HERE / "verification" / "session_check.py"
        )
        mod = _ilu.module_from_spec(spec)
        sys.modules["session_check"] = mod
        spec.loader.exec_module(mod)
        check_fn = mod.check_session_full
    except Exception as e:
        print(_fail(f"Impossible de charger session_check.py : {e}"))
        sys.exit(1)

    sessions = sorted([
        p for p in input_dir.iterdir()
        if p.is_dir()
        and ((p / "metadata.json").exists() or (p / "tracker_positions.csv").exists())
    ])

    if not sessions:
        print(_warn(f"Aucune session trouvée dans {input_dir}"))
        return []

    if not dry_run:
        out_ab.mkdir(parents=True, exist_ok=True)
        out_rest.mkdir(parents=True, exist_ok=True)

    results = []

    print()
    print("=" * 70)
    print(f"ÉVALUATION DE {len(sessions)} SESSION(S)")
    print(f"  Entrée     : {input_dir}")
    print(f"  Sortie A/B : {out_ab}")
    print(f"  Sortie C/D/F: {out_rest}")
    print("=" * 70)
    print()

    for session_path in sessions:
        name = session_path.name
        print(f"  Analyse : {name} …", flush=True)

        try:
            result = check_fn(session_path)
        except Exception as e:
            print(_fail(f"    ERREUR lors de l'analyse : {e}"))
            results.append({
                "session": name,
                "path": str(session_path),
                "score": 0.0,
                "grade": "F",
                "verdict": "ERREUR",
                "error": str(e),
                "destination": str(out_rest / name),
                "bucket": "rest",
            })
            if not dry_run:
                _copy_session(session_path, out_rest / name)
            continue

        score   = result.get("score", 0.0)
        grade   = result.get("grade", _grade_label(score))
        verdict = result.get("verdict", "")
        blocked = result.get("blocked", False)
        perfect = result.get("perfect", False)

        is_ab   = _is_ab(grade)
        dest    = out_ab / name if is_ab else out_rest / name
        bucket  = "AB" if is_ab else "rest"

        gc = _grade_color(grade)
        tag = _c(f"grade {grade}", _BOLD + gc)
        print(f"    {tag}  score={score:.1f}%  verdict={verdict}  → {bucket}")

        if not dry_run:
            _copy_session(session_path, dest)

        results.append({
            "session": name,
            "path": str(session_path),
            "score": score,
            "grade": grade,
            "verdict": verdict,
            "blocked": blocked,
            "perfect": perfect,
            "destination": str(dest),
            "bucket": bucket,
        })

    return results


def _copy_session(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _print_summary(results: list[dict]) -> None:
    print()
    print("=" * 70)
    print("RÉSUMÉ DU TRI")
    print("=" * 70)

    ab    = [r for r in results if r["bucket"] == "AB"]
    rest  = [r for r in results if r["bucket"] == "rest"]
    total = len(results)

    print(f"\n  Total sessions : {total}")
    print(f"  Grade A ou B   : {len(ab)}")
    print(f"  Grade C/D/F    : {len(rest)}")

    if ab:
        print()
        print(_c("  ── Sessions A/B (utilisables) ──", _BOLD + _GREEN))
        for r in ab:
            gc = _grade_color(r["grade"])
            print(f"    {_c(r['grade'], gc)}  {r['score']:5.1f}%  {r['session']}")

    if rest:
        print()
        print(_c("  ── Sessions C/D/F (à corriger ou rejeter) ──", _BOLD + _YELLOW))
        for r in rest:
            gc = _grade_color(r["grade"])
            print(f"    {_c(r['grade'], gc)}  {r['score']:5.1f}%  {r['session']}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Bilan des tests unitaires
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests() -> bool:
    """Lance tous les tests unitaires et retourne True si tous passent."""
    np.random.seed(0)

    suite = [
        ("Tracker synthétique 100%",         test_tracker_synthetic),
        ("Tracker consensus = test 4",        test_tracker_consensus),
        ("Tracker données réelles > 90%",     test_tracker_real_data),
        ("Caméra synthétique 100%",           test_camera_synthetic),
        ("Caméra données réelles ≥ 80%",      test_camera_real_data),
        ("Gripper détection offset",          test_gripper_offset_detection),
        ("Gripper robustesse bruit",          test_gripper_noise_robustness),
        ("Gripper cas limites",               test_gripper_edge_cases),
        ("Gripper dry-run réel",              test_gripper_real_data),
    ]

    results_unit = []
    for name, fn in suite:
        try:
            ok = fn()
        except Exception as e:
            print(_fail(f"Exception dans {name} : {e}"))
            ok = False
        results_unit.append((name, ok))

    print()
    print("=" * 60)
    print("BILAN TESTS UNITAIRES")
    print("=" * 60)
    all_ok = True
    for name, ok in results_unit:
        print(f"  {'✓' if ok else '✗'}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print(_ok("TOUS LES TESTS UNITAIRES PASSÉS"))
    else:
        print(_fail("DES TESTS UNITAIRES ONT ÉCHOUÉ"))
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",   type=Path, help="Dossier contenant les sessions à évaluer")
    p.add_argument("--out-ab",  type=Path, default=Path("output_AB"),
                   help="Dossier de sortie pour grades A et B (défaut: output_AB)")
    p.add_argument("--out-rest", type=Path, default=Path("output_CDF"),
                   help="Dossier de sortie pour grades C, D et F (défaut: output_CDF)")
    p.add_argument("--no-unit-tests", action="store_true",
                   help="Ne pas exécuter les tests unitaires")
    p.add_argument("--no-copy", action="store_true",
                   help="Afficher le tri sans copier les fichiers")
    p.add_argument("--json", action="store_true",
                   help="Afficher le rapport des sessions en JSON")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Tests unitaires ───────────────────────────────────────────────────
    unit_ok = True
    if not args.no_unit_tests:
        print(_c("\n══ TESTS UNITAIRES ══════════════════════════════════════════\n", _BOLD))
        unit_ok = run_unit_tests()

    # ── Évaluation des sessions ───────────────────────────────────────────
    if args.input:
        input_dir = args.input.resolve()
        if not input_dir.exists():
            print(_fail(f"Dossier d'entrée introuvable : {input_dir}"))
            sys.exit(1)

        print(_c("\n══ ÉVALUATION DES SESSIONS ══════════════════════════════════\n", _BOLD))

        results = evaluate_sessions(
            input_dir=input_dir,
            out_ab=args.out_ab.resolve(),
            out_rest=args.out_rest.resolve(),
            dry_run=args.no_copy,
        )

        _print_summary(results)

        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))

        session_ok = all(r.get("grade") in ("A", "B") for r in results) if results else True
    else:
        print()
        if args.no_unit_tests:
            print(_warn("Aucun dossier --input fourni et tests unitaires désactivés. Rien à faire."))
        else:
            print(_warn("Aucun dossier --input fourni. Seuls les tests unitaires ont été exécutés."))
        print("  Exemple : python test_sessions.py --input /chemin/sessions "
              "--out-ab ./AB --out-rest ./CDF")
        session_ok = True

    sys.exit(0 if unit_ok else 1)


if __name__ == "__main__":
    main()
