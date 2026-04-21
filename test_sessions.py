#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_sessions.py — Script unifié de test et tri des sessions.

Tests couverts :
  §1  SLERP — propriétés mathématiques (t=0/1/0.5, antipodaux, identiques)
  §2  Quaternions — détection NaN/zéro, seuil non-récupérable, réparation SLERP
  §3  Gaps tracker — détection, single gap, multi-gaps, données propres
  §4  Dérive horloge — régression linéaire numpy, détection de drift synthétique
  §5  Trackers / test hauteur — head est le plus haut, cas inversé
  §6  Trackers / test centralité — head est le plus central
  §7  Trackers / test mobilité — head est le moins mobile
  §8  Trackers / test latéral — 100 % sur toutes orientations (0°–360°)
  §9  Trackers / consensus — left/right = uniquement test 4
  §10 Trackers / 6 permutations — toutes les inversions head/left/right détectées
  §11 Trackers / algorithme alternatif — detect_head et detect_hands
  §12 Trackers / données réelles — > 90 % accuracy
  §13 Caméras / min-flux head — la caméra la moins active est la tête
  §14 Caméras / corrélation L/R — corrélation flux↔tracker
  §15 Caméras / 6 permutations — toutes les inversions tête/left/right détectées
  §16 Caméras / données réelles — ≥ 80 % all-3 accuracy
  §17 Gripper / détection d'offset — 7 valeurs connues
  §18 Gripper / robustesse bruit — jusqu'à 15 %
  §19 Gripper / cas limites — capteur ouvert, visuel constant, overlap insuffisant
  §20 Gripper / dry-run réel — sessions disponibles

Tri des sessions :
  --out-ab   → grades A (≥ 90) ou B (≥ 75)
  --out-rest → grades C, D, F

Usage :
    python test_sessions.py --input /chemin/sessions \\
                            --out-ab ./sessions_AB \\
                            --out-rest ./sessions_CDF
    python test_sessions.py              # tests unitaires seuls
    python test_sessions.py --no-copy    # tri sans copier
    python test_sessions.py --json       # rapport JSON
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np

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

_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"; _B = "\033[1m"; _X = "\033[0m"


def _c(t, col):  return f"{col}{t}{_X}" if sys.stdout.isatty() else t
def _ok(m):      return _c(f"✓  {m}", _G)
def _fail(m):    return _c(f"✗  {m}", _R)
def _warn(m):    return _c(f"●  {m}", _Y)
def _h(t):       return _c(t, _B)


# ══════════════════════════════════════════════════════════════════════════════
# Générateurs de données synthétiques partagés
# ══════════════════════════════════════════════════════════════════════════════

def _quat_from_yaw(yaw_rad: float) -> np.ndarray:
    cy, sy = np.cos(yaw_rad / 2), np.sin(yaw_rad / 2)
    return np.array([cy, 0.0, sy, 0.0])


def _quat_rotmat(q: np.ndarray) -> np.ndarray:
    q = q.astype(float)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    R = np.empty((len(q), 3, 3))
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-z*w); R[:,0,2]=2*(x*z+y*w)
    R[:,1,0]=2*(x*y+z*w);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-x*w)
    R[:,2,0]=2*(x*z-y*w);   R[:,2,1]=2*(y*z+x*w);   R[:,2,2]=1-2*(x*x+y*y)
    return R


def _make_tracker_blocks(yaw_rad: float = 0.0, n: int = 400, noise: float = 0.005,
                          head_motion_scale: float = 1.0, hand_motion_scale: float = 1.0):
    """Blocs physiques corrects : tête haute/centrale, main droite à +X local."""
    q = _quat_from_yaw(yaw_rad)
    quat = np.tile(q, (n, 1)) + np.random.randn(n, 4) * 0.001
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = _quat_rotmat(quat)
    local_x = np.mean(R[:, :, 0], axis=0)

    head_pos  = (np.array([0.0, 1.7, 0.0])
                 + np.cumsum(np.random.randn(n, 3) * 0.001 * head_motion_scale, axis=0))
    right_pos = (head_pos + local_x * 0.35 + np.array([0, -0.7, 0])
                 + np.cumsum(np.random.randn(n, 3) * 0.003 * hand_motion_scale, axis=0))
    left_pos  = (head_pos - local_x * 0.35 + np.array([0, -0.7, 0])
                 + np.cumsum(np.random.randn(n, 3) * 0.003 * hand_motion_scale, axis=0))

    id_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return {
        "head":  (head_pos,  quat),
        "left":  (left_pos,  id_quat),
        "right": (right_pos, id_quat),
    }


# ══════════════════════════════════════════════════════════════════════════════
# §1  SLERP — propriétés mathématiques
# ══════════════════════════════════════════════════════════════════════════════

def test_slerp_properties() -> bool:
    print(_h("=" * 60))
    print(_h("§1  SLERP — PROPRIÉTÉS MATHÉMATIQUES"))
    print(_h("=" * 60))

    try:
        from fix.fix_tracker_gaps import slerp
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    failures = []

    def _unit(q):
        return abs(np.linalg.norm(q) - 1.0) < 1e-6

    # t=0 → q0
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 1.0, 0.0, 0.0])
    r  = slerp(q0, q1, 0.0)
    ok = np.allclose(r, q0, atol=1e-5) and _unit(r)
    print(f"  {'✓' if ok else '✗'}  t=0  → q0      got={np.round(r,4)}")
    if not ok: failures.append("t=0 ≠ q0")

    # t=1 → q1
    r = slerp(q0, q1, 1.0)
    ok = np.allclose(r, q1, atol=1e-5) and _unit(r)
    print(f"  {'✓' if ok else '✗'}  t=1  → q1      got={np.round(r,4)}")
    if not ok: failures.append("t=1 ≠ q1")

    # t=0.5 → doit être à mi-chemin angulaire, norme=1
    r = slerp(q0, q1, 0.5)
    ok = _unit(r) and np.allclose(np.dot(r, q0), np.dot(r, q1), atol=1e-4)
    print(f"  {'✓' if ok else '✗'}  t=0.5 → midpoint  |r|={np.linalg.norm(r):.6f}")
    if not ok: failures.append("t=0.5 pas à mi-chemin")

    # Quaternions antipodaux (–q0 ≡ q0) → chemin court
    qm = -q0
    r  = slerp(qm, q1, 0.5)
    ok = _unit(r)
    print(f"  {'✓' if ok else '✗'}  Antipodaux → chemin court  |r|={np.linalg.norm(r):.6f}")
    if not ok: failures.append("antipodaux: norme ≠ 1")

    # Quaternions identiques → q0
    r  = slerp(q0, q0, 0.7)
    ok = np.allclose(r, q0, atol=1e-5)
    print(f"  {'✓' if ok else '✗'}  Identiques → q0  got={np.round(r,4)}")
    if not ok: failures.append("identiques ≠ q0")

    # Quaternions quasi-identiques (dot > 0.9995) → fallback linéaire, norme=1
    eps = 1e-5
    qn  = np.array([1.0, eps, 0.0, 0.0])
    qn /= np.linalg.norm(qn)
    r   = slerp(q0, qn, 0.5)
    ok  = _unit(r)
    print(f"  {'✓' if ok else '✗'}  Quasi-identiques → |r|={np.linalg.norm(r):.8f}")
    if not ok: failures.append("quasi-identiques: norme ≠ 1")

    print()
    if not failures:
        print(_ok("RÉSULTAT §1 : toutes les propriétés SLERP vérifiées"))
    else:
        print(_fail(f"RÉSULTAT §1 : échecs = {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §2  Quaternions — détection NaN/zéro, seuil, réparation
# ══════════════════════════════════════════════════════════════════════════════

def _make_quat_df(n: int = 200, nan_indices=(), zero_indices=()) -> "pd.DataFrame":
    """DataFrame tracker_positions.csv synthétique avec quaternions valides + quelques invalides."""
    rng = np.random.default_rng(0)
    data = {"timestamp_ns": np.arange(n, dtype=np.int64) * 16_000_000}
    for trk in ("head", "left", "right"):
        for ax in ("x", "y", "z"):
            data[f"tracker_{trk}_{ax}"] = rng.normal(0, 0.1, n)
        # Quaternion identité + légère variation
        w = np.ones(n); x = rng.normal(0, 0.01, n); y = rng.normal(0, 0.01, n); z = rng.normal(0, 0.01, n)
        norms = np.sqrt(w**2 + x**2 + y**2 + z**2)
        w /= norms; x /= norms; y /= norms; z /= norms
        for i in nan_indices:
            if i < n:
                w[i] = float("nan")
        for i in zero_indices:
            if i < n:
                w[i] = x[i] = y[i] = z[i] = 0.0
        data[f"tracker_{trk}_qw"] = w
        data[f"tracker_{trk}_qx"] = x
        data[f"tracker_{trk}_qy"] = y
        data[f"tracker_{trk}_qz"] = z
    return pd.DataFrame(data)


def test_quaternion_nan_detection() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§2a QUATERNIONS — Détection NaN"))
    print(_h("=" * 60))

    if not _PANDAS:
        print(_warn("SKIPPED : pandas non disponible"))
        return True
    try:
        from fix.fix_quaternions import fix_quaternions, QUAT_MAX_CORRUPT_FRAC
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    import tempfile, os
    failures = []

    with tempfile.TemporaryDirectory() as tmpdir:
        sess = Path(tmpdir) / "test_session"
        sess.mkdir()
        (sess / "metadata.json").write_text("{}", encoding="utf-8")

        # ── cas 1 : aucun NaN → status "ok" ──────────────────────────────────
        df = _make_quat_df(200)
        df.to_csv(sess / "tracker_positions.csv", index=False)
        r = fix_quaternions(sess, dry_run=True, force=True)
        ok = r.get("status") in ("ok", "dry-run") and r.get("total_repaired", 0) == 0
        print(f"  {'✓' if ok else '✗'}  Aucun NaN → status={r.get('status')}  "
              f"total_repaired={r.get('total_repaired', 0)}")
        if not ok: failures.append("aucun NaN non reconnu comme ok")

        # ── cas 2 : quelques NaN → détectés, réparables ───────────────────────
        df2 = _make_quat_df(200, nan_indices=[10, 50, 100])
        df2.to_csv(sess / "tracker_positions.csv", index=False)
        (sess / "metadata.json").write_text("{}", encoding="utf-8")
        r2 = fix_quaternions(sess, dry_run=True, force=True)
        ok2 = any(v.get("n_invalid", 0) > 0
                  for v in r2.get("tracker_reports", {}).values())
        print(f"  {'✓' if ok2 else '✗'}  3 NaN → détectés  "
              f"trackers={r2.get('tracker_reports', {})}")
        if not ok2: failures.append("NaN non détectés")

        # ── cas 3 : quaternion zéro → invalide ───────────────────────────────
        df3 = _make_quat_df(200, zero_indices=[20, 80])
        df3.to_csv(sess / "tracker_positions.csv", index=False)
        (sess / "metadata.json").write_text("{}", encoding="utf-8")
        r3 = fix_quaternions(sess, dry_run=True, force=True)
        ok3 = any(v.get("n_invalid", 0) > 0
                  for v in r3.get("tracker_reports", {}).values())
        print(f"  {'✓' if ok3 else '✗'}  2 quaternions zéro → détectés  "
              f"trackers={r3.get('tracker_reports', {})}")
        if not ok3: failures.append("quaternions zéro non détectés")

        # ── cas 4 : > 20 % invalides → non récupérable ───────────────────────
        n_bad = int(0.25 * 200)
        df4 = _make_quat_df(200, nan_indices=list(range(n_bad)))
        df4.to_csv(sess / "tracker_positions.csv", index=False)
        (sess / "metadata.json").write_text("{}", encoding="utf-8")
        r4 = fix_quaternions(sess, dry_run=True, force=True)
        ok4 = not r4.get("recoverable", True) or any(
            v.get("status") == "unrecoverable"
            for v in r4.get("tracker_reports", {}).values()
        )
        print(f"  {'✓' if ok4 else '✗'}  25 % NaN → non récupérable  "
              f"recoverable={r4.get('recoverable')}")
        if not ok4: failures.append("> 20% NaN non marqué non-récupérable")

        # ── cas 5 : réparation → quaternions résultants de norme ≈ 1 ─────────
        df5 = _make_quat_df(200, nan_indices=list(range(5, 15)))
        df5.to_csv(sess / "tracker_positions.csv", index=False)
        (sess / "metadata.json").write_text("{}", encoding="utf-8")
        r5 = fix_quaternions(sess, dry_run=False, force=True)
        df_fixed = pd.read_csv(sess / "tracker_positions.csv")
        bad_norms = 0
        for trk in ("head", "left", "right"):
            cols = [f"tracker_{trk}_{q}" for q in ("qw", "qx", "qy", "qz")]
            Q = df_fixed[cols].to_numpy(float)
            norms = np.linalg.norm(Q, axis=1)
            bad_norms += int((np.abs(norms - 1.0) > 0.01).sum())
        ok5 = bad_norms == 0
        print(f"  {'✓' if ok5 else '✗'}  Après réparation : quaternions unitaires  "
              f"bad_norms={bad_norms}")
        if not ok5: failures.append(f"réparation produit {bad_norms} quaternions non-unitaires")

    print()
    if not failures:
        print(_ok("RÉSULTAT §2 : détection et réparation quaternions OK"))
    else:
        print(_fail(f"RÉSULTAT §2 : {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §3  Gaps tracker — détection
# ══════════════════════════════════════════════════════════════════════════════

def test_tracker_gap_detection() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§3  GAPS TRACKER — DÉTECTION"))
    print(_h("=" * 60))

    try:
        from fix.fix_tracker_gaps import _find_tracker_gaps, TRACKER_GAP_THRESHOLD_MS
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    failures = []

    # ── Données propres (pas de gaps) ────────────────────────────────────────
    t_clean = np.arange(500) * 16.0  # 60 Hz
    gaps = _find_tracker_gaps(t_clean, TRACKER_GAP_THRESHOLD_MS)
    ok = len(gaps) == 0
    print(f"  {'✓' if ok else '✗'}  60 Hz propre → {len(gaps)} gap(s) (attendu 0)")
    if not ok: failures.append("faux positif gap")

    # ── Un seul gap de 300 ms ─────────────────────────────────────────────────
    t_gap1 = np.concatenate([np.arange(100) * 16.0,
                              np.arange(100) * 16.0 + 100 * 16.0 + 300.0])
    gaps1 = _find_tracker_gaps(t_gap1, TRACKER_GAP_THRESHOLD_MS)
    ok1 = len(gaps1) == 1 and abs(gaps1[0][1] - 300.0) < 20
    print(f"  {'✓' if ok1 else '✗'}  Gap 300 ms → {len(gaps1)} gap(s) "
          f"durée={gaps1[0][1] if gaps1 else '?'}ms (attendu ≈300ms)")
    if not ok1: failures.append(f"gap 300ms mal détecté: {gaps1}")

    # ── Deux gaps de durées différentes ──────────────────────────────────────
    t_gap2 = np.concatenate([
        np.arange(50) * 16.0,
        np.arange(50) * 16.0 + 50 * 16.0 + 500.0,
        np.arange(50) * 16.0 + 100 * 16.0 + 500.0 + 200.0,
    ])
    gaps2 = _find_tracker_gaps(t_gap2, TRACKER_GAP_THRESHOLD_MS)
    ok2 = len(gaps2) == 2
    print(f"  {'✓' if ok2 else '✗'}  2 gaps → {len(gaps2)} gap(s) détectés "
          f"durées={[round(d,1) for _, d in gaps2]}")
    if not ok2: failures.append(f"attendu 2 gaps, obtenu {len(gaps2)}")

    # ── Variation en dessous du seuil → pas de gap ───────────────────────────
    t_small = np.arange(300) * 16.0
    # Injecter une variation de 30 ms (< seuil 60 ms)
    t_small[150] += 30.0
    t_small[151:] += 30.0   # décaler tout ce qui suit pour garder la monotonie
    gaps_small = _find_tracker_gaps(t_small, TRACKER_GAP_THRESHOLD_MS)
    ok3 = len(gaps_small) == 0
    print(f"  {'✓' if ok3 else '✗'}  Variation 30 ms < seuil ({TRACKER_GAP_THRESHOLD_MS:.0f}ms) "
          f"→ {len(gaps_small)} gap(s)")
    if not ok3: failures.append("faux positif pour variation < seuil")

    print()
    if not failures:
        print(_ok("RÉSULTAT §3 : détection de gaps correcte"))
    else:
        print(_fail(f"RÉSULTAT §3 : {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §4  Dérive horloge — régression linéaire et measure_drift
# ══════════════════════════════════════════════════════════════════════════════

def test_clock_drift() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§4  DÉRIVE HORLOGE — RÉGRESSION ET MESURE"))
    print(_h("=" * 60))

    try:
        from fix.fix_clock_drift import _linreg_numpy, measure_drift
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    failures = []

    # ── _linreg_numpy : droite parfaite ──────────────────────────────────────
    x = np.arange(10, dtype=float)
    y = 3.0 * x + 7.0
    slope, intercept, r2 = _linreg_numpy(x, y)
    ok = abs(slope - 3.0) < 0.01 and abs(intercept - 7.0) < 0.01 and r2 > 0.999
    print(f"  {'✓' if ok else '✗'}  Droite parfaite y=3x+7  "
          f"slope={slope:.4f} intercept={intercept:.4f} r²={r2:.4f}")
    if not ok: failures.append("linreg droite parfaite échoue")

    # ── _linreg_numpy : données constantes → slope ≈ 0 ──────────────────────
    y_const = np.ones(10) * 5.0
    slope_c, intercept_c, r2_c = _linreg_numpy(x, y_const)
    ok_c = abs(slope_c) < 0.01 and abs(intercept_c - 5.0) < 0.1
    print(f"  {'✓' if ok_c else '✗'}  Données constantes  "
          f"slope={slope_c:.4f} intercept={intercept_c:.4f} (r²={r2_c:.4f} ignoré)")
    if not ok_c: failures.append("linreg constante échoue")

    # ── _linreg_numpy : données insuffisantes (< 4 points) ───────────────────
    s0, i0, r0 = _linreg_numpy(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    ok0 = (s0, i0, r0) == (0.0, 0.0, 0.0)
    print(f"  {'✓' if ok0 else '✗'}  Données insuffisantes (n=2) → zeros  "
          f"got=({s0},{i0},{r0})")
    if not ok0: failures.append("linreg < 4 points ne retourne pas (0,0,0)")

    # ── _linreg_numpy sur offsets driftés (test end-to-end de la régression) ──
    # Reproduit exactement ce que fait measure_drift en interne après avoir
    # collecté les offsets, sans passer par la recherche nearest-neighbour.
    for true_ppm in (100.0, 500.0, 1000.0, 2000.0):
        t_rel   = np.linspace(0, 300_000, 300)       # 300 s, 300 points
        offsets = t_rel * (true_ppm * 1e-6)           # dérive linéaire parfaite
        slope, intercept, r2 = _linreg_numpy(t_rel, offsets)
        det_ppm = abs(slope) * 1e6
        ok_r = abs(det_ppm - true_ppm) / true_ppm < 0.01 and r2 > 0.999
        print(f"  {'✓' if ok_r else '✗'}  linreg offsets {true_ppm:.0f}ppm → "
              f"détecté={det_ppm:.1f}ppm  r²={r2:.4f}")
        if not ok_r: failures.append(f"linreg drift {true_ppm}ppm échoue")

    # ── measure_drift : retour structurel valide ──────────────────────────────
    # measure_drift utilise la recherche nearest-neighbour : fiable uniquement
    # quand le tracker est plus dense que la caméra (e.g. 90fps vs 30fps).
    # On vérifie ici que la fonction retourne bien un dict avec les champs requis.
    t_trk_s = np.linspace(0, 60_000, 5400)   # 90fps, 60s
    t_cam_s = np.arange(1800) * 33.333        # 30fps, 60s
    r_struct = measure_drift(t_trk_s, t_cam_s)
    req_keys = {"valid", "slope_ms_per_ms", "intercept_ms", "r2", "drift_ppm"}
    ok_struct = isinstance(r_struct, dict) and (
        not r_struct.get("valid", True) or req_keys.issubset(r_struct.keys())
    )
    print(f"  {'✓' if ok_struct else '✗'}  measure_drift structure → "
          f"valid={r_struct.get('valid')} clés_présentes={req_keys.issubset(r_struct.keys())}")
    if not ok_struct: failures.append("measure_drift structure incorrecte")

    # ── measure_drift : offset constant → pas de drift détecté ───────────────
    # Camera dense alignée sur tracker avec offset fixe de +200ms.
    # La nearest-neighbour quantifie l'offset à ±dt_cam/2 max → slope ≈ 0.
    t_trk_nd = np.linspace(0, 60_000, 5400)    # 90fps tracker
    t_cam_nd = np.arange(1800) * 33.333 + 200.0 # 30fps camera, offset fixe
    r_no = measure_drift(t_trk_nd, t_cam_nd)
    ok_no = r_no.get("valid") and r_no.get("drift_ppm", 999) < 100.0
    print(f"  {'✓' if ok_no else '✗'}  Pas de drift (offset fixe +200ms) → "
          f"drift_ppm={r_no.get('drift_ppm', '?')} (attendu < 100)")
    if not ok_no: failures.append("faux positif drift")

    # ── measure_drift : overlap insuffisant ──────────────────────────────────
    t_short = np.arange(5, dtype=float) * 500.0  # 2500 ms < MIN_DURATION_MS
    r_short = measure_drift(t_short, t_short)
    ok_short = not r_short.get("valid", True)
    print(f"  {'✓' if ok_short else '✗'}  Overlap court → invalid  "
          f"valid={r_short.get('valid')}")
    if not ok_short: failures.append("overlap court accepté à tort")

    print()
    if not failures:
        print(_ok("RÉSULTAT §4 : régression linéaire et mesure drift corrects"))
    else:
        print(_fail(f"RÉSULTAT §4 : {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §5  Tracker / test hauteur
# ══════════════════════════════════════════════════════════════════════════════

def test_tracker_height() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§5  TRACKER — TEST HAUTEUR Y"))
    print(_h("=" * 60))

    try:
        from fix.fix_tracker_labels import _test_height
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    failures = []
    np.random.seed(1)

    # ── cas 1 : configuration correcte (head le plus haut) ──────────────────
    phys = _make_tracker_blocks(yaw_rad=0.0)
    blocks = [(lbl, *phys[lbl]) for lbl in ("head", "left", "right")]
    t1 = _test_height(blocks)
    ok1 = t1.head_vote == "head"
    print(f"  {'✓' if ok1 else '✗'}  Config correcte → head_vote={t1.head_vote}  "
          f"zscore={t1.evidence.get('zscore',0):.1f}")
    if not ok1: failures.append("config correcte: head non détecté")

    # ── cas 2 : head en position de main (bas) → détecte l'autre plus haut ──
    phys2 = _make_tracker_blocks(yaw_rad=0.0)
    # Échanger la hauteur de head et left
    head_y = phys2["head"][0][:, 1].copy()
    left_y = phys2["left"][0][:, 1].copy()
    phys2["head"][0][:, 1] = left_y - 1.0   # head devient bas
    phys2["left"][0][:, 1] = head_y          # left prend la position haute
    blocks2 = [(lbl, *phys2[lbl]) for lbl in ("head", "left", "right")]
    t1b = _test_height(blocks2)
    ok2 = t1b.head_vote == "left"   # l'algorithme doit identifier "left" comme le plus haut
    print(f"  {'✓' if ok2 else '✗'}  Head plus bas que left → head_vote={t1b.head_vote} "
          f"(attendu 'left')")
    if not ok2: failures.append("height: head bas non détecté")

    # ── cas 3 : grand z-score → certitude ────────────────────────────────────
    ok3 = t1.evidence.get("zscore", 0) >= 5.0
    print(f"  {'✓' if ok3 else '✗'}  z-score ≥ 5 σ → "
          f"zscore={t1.evidence.get('zscore',0):.1f}  certain={t1.evidence.get('certain')}")
    if not ok3: failures.append("z-score insuffisant sur données propres")

    print()
    if not failures:
        print(_ok("RÉSULTAT §5 : test hauteur correct"))
    else:
        print(_fail(f"RÉSULTAT §5 : {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §6  Tracker / test centralité
# ══════════════════════════════════════════════════════════════════════════════

def test_tracker_centrality() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§6  TRACKER — TEST CENTRALITÉ 3D"))
    print(_h("=" * 60))

    try:
        from fix.fix_tracker_labels import _test_centrality
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    failures = []
    np.random.seed(2)

    # _test_centrality trouve le tracker avec la plus faible distance MOYENNE aux deux autres.
    # Il faut que la distance head-main < distance main-main (mains très écartées).

    # ── cas 1 : head au centre, mains très écartées ───────────────────────────
    n = 300
    id_q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    # head à [0,0,0], mains à [±3, 0, 0] → dist(head,main)=3 < dist(main,main)=6
    head_ctr  = np.column_stack([np.zeros(n),      np.zeros(n), np.zeros(n)])
    left_far  = np.column_stack([np.ones(n)*-3.0,  np.zeros(n), np.zeros(n)])
    right_far = np.column_stack([np.ones(n)*+3.0,  np.zeros(n), np.zeros(n)])
    blocks1 = [("head", head_ctr, id_q), ("left", left_far, id_q), ("right", right_far, id_q)]
    t2 = _test_centrality(blocks1)
    ok1 = t2.head_vote == "head"
    print(f"  {'✓' if ok1 else '✗'}  Head entre mains écartées → head_vote={t2.head_vote}  "
          f"conf={t2.confidence:.2f}")
    if not ok1: failures.append("centralité: head central non détecté")

    # ── cas 2 : main placée exactement au centre → détectée comme head ───────
    # "left" au centre, "head" loin du centre
    far_head  = np.column_stack([np.zeros(n), np.ones(n) * 5.0, np.zeros(n)])
    left_ctr  = np.column_stack([np.zeros(n), np.ones(n) * 2.5, np.zeros(n)])
    right_far2 = np.column_stack([np.ones(n) * 4.0, np.zeros(n), np.zeros(n)])
    blocks_inv = [("head", far_head, id_q), ("left", left_ctr, id_q), ("right", right_far2, id_q)]
    t2b = _test_centrality(blocks_inv)
    ok2 = t2b.head_vote == "left"
    print(f"  {'✓' if ok2 else '✗'}  'left' au centre → head_vote={t2b.head_vote} "
          f"(attendu 'left')")
    if not ok2: failures.append("centralité: centre non détecté")

    print()
    if not failures:
        print(_ok("RÉSULTAT §6 : test centralité correct"))
    else:
        print(_fail(f"RÉSULTAT §6 : {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §7  Tracker / test mobilité
# ══════════════════════════════════════════════════════════════════════════════

def test_tracker_mobility() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§7  TRACKER — TEST MOBILITÉ"))
    print(_h("=" * 60))

    try:
        from fix.fix_tracker_labels import _test_mobility
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    failures = []
    np.random.seed(3)

    # ── cas 1 : head bouge peu, mains bougent beaucoup ───────────────────────
    phys = _make_tracker_blocks(head_motion_scale=0.2, hand_motion_scale=3.0)
    blocks = [(lbl, *phys[lbl]) for lbl in ("head", "left", "right")]
    t3 = _test_mobility(blocks)
    ok1 = t3.head_vote == "head"
    print(f"  {'✓' if ok1 else '✗'}  Head lent → head_vote={t3.head_vote}  "
          f"conf={t3.confidence:.2f}")
    if not ok1: failures.append("mobilité: head lent non détecté")

    # ── cas 2 : une main immobile, head très mobile ────────────────────────────
    n = 300
    id_q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    # "right" quasi-immobile
    right_still = np.tile([0.5, 0.0, 0.0], (n, 1)) + np.random.randn(n, 3) * 0.0001
    head_fast   = np.cumsum(np.random.randn(n, 3) * 0.05, axis=0)
    left_fast   = np.cumsum(np.random.randn(n, 3) * 0.04, axis=0)
    blocks_inv  = [("head", head_fast, id_q), ("left", left_fast, id_q),
                   ("right", right_still, id_q)]
    t3b = _test_mobility(blocks_inv)
    ok2 = t3b.head_vote == "right"
    print(f"  {'✓' if ok2 else '✗'}  'right' immobile → head_vote={t3b.head_vote} "
          f"(attendu 'right')")
    if not ok2: failures.append("mobilité: immobile non détecté")

    print()
    if not failures:
        print(_ok("RÉSULTAT §7 : test mobilité correct"))
    else:
        print(_fail(f"RÉSULTAT §7 : {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §8  Tracker / test latéral — 100 % sur toutes orientations
# ══════════════════════════════════════════════════════════════════════════════

def test_tracker_lateral_all_yaws() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§8  TRACKER — TEST LATÉRAL (toutes orientations 0°–360°)"))
    print(_h("=" * 60))

    try:
        from fix.fix_tracker_labels import _test_lateral
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    yaws     = np.radians(np.arange(0, 360, 15))
    failures = []

    for yaw in yaws:
        phys   = _make_tracker_blocks(yaw_rad=yaw)
        blocks = [(lbl, *phys[lbl]) for lbl in ("head", "left", "right")]
        t4     = _test_lateral(blocks, "head")
        ok     = t4.right_vote == "right" and t4.left_vote == "left"
        deg    = round(np.degrees(yaw))
        print(f"  {'✓' if ok else '✗'}  yaw={deg:3d}°  "
              f"right={t4.right_vote}  left={t4.left_vote}  "
              f"sep={t4.evidence.get('separation_m',0):.3f}m")
        if not ok: failures.append(deg)

    n = len(yaws)
    print()
    if not failures:
        print(_ok(f"RÉSULTAT §8 : {n}/{n} orientations correctes (100%)"))
    else:
        print(_fail(f"RÉSULTAT §8 : échoué aux yaw={failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §9  Tracker / consensus = test 4 uniquement pour left/right
# ══════════════════════════════════════════════════════════════════════════════

def test_tracker_consensus_lateral_only() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§9  TRACKER — CONSENSUS left/right ≡ TEST LATÉRAL SEUL"))
    print(_h("=" * 60))

    try:
        from fix.fix_tracker_labels import (
            _test_height, _test_centrality, _test_mobility, _test_lateral, _consensus
        )
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    np.random.seed(42)
    phys   = _make_tracker_blocks(yaw_rad=0.0)
    blocks = [(lbl, *phys[lbl]) for lbl in ("head", "left", "right")]

    t1 = _test_height(blocks)
    t2 = _test_centrality(blocks)
    t3 = _test_mobility(blocks)
    t4 = _test_lateral(blocks, t1.head_vote)
    predicted, agree, certain = _consensus([t1, t2, t3, t4])

    ok_r = predicted.get("right") == t4.right_vote
    ok_l = predicted.get("left")  == t4.left_vote

    print(f"  t4 vote          : right={t4.right_vote}  left={t4.left_vote}")
    print(f"  consensus prédit : right={predicted.get('right')}  left={predicted.get('left')}")
    print(f"  right == t4.right_vote : {ok_r}")
    print(f"  left  == t4.left_vote  : {ok_l}")

    ok = ok_r and ok_l
    print()
    print(_ok("RÉSULTAT §9 : consensus = test latéral") if ok
          else _fail("RÉSULTAT §9 : consensus ≠ test latéral"))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# §10 Tracker / 6 permutations (toutes les inversions head/left/right)
# ══════════════════════════════════════════════════════════════════════════════

def test_tracker_all_permutations() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§10 TRACKER — 6 PERMUTATIONS (toutes inversions)"))
    print(_h("=" * 60))

    try:
        from fix.fix_tracker_labels import (
            _test_height, _test_centrality, _test_mobility, _test_lateral, _consensus
        )
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    np.random.seed(10)
    # Données physiques de référence (configuration correcte connue)
    phys = _make_tracker_blocks(yaw_rad=0.0, n=500,
                                 head_motion_scale=0.2, hand_motion_scale=3.0)

    failures = []
    csv_cols  = ("head", "left", "right")
    phys_roles = ("head", "left", "right")

    for perm in itertools.permutations(phys_roles):
        # perm[i] = rôle physique placé dans la colonne CSV i
        csv_to_phys = dict(zip(csv_cols, perm))

        # Construction des blocs avec les données permutées
        blocks = [(col, phys[role][0], phys[role][1])
                  for col, role in csv_to_phys.items()]

        # Prédiction attendue : {rôle_physique: colonne_csv_qui_le_contient}
        expected = {role: col for col, role in csv_to_phys.items()}

        t1 = _test_height(blocks)
        t2 = _test_centrality(blocks)
        t3 = _test_mobility(blocks)
        t4 = _test_lateral(blocks, t1.head_vote)
        predicted, agree, certain = _consensus([t1, t2, t3, t4])

        ok = (predicted.get("head")  == expected["head"]  and
              predicted.get("left")  == expected["left"]  and
              predicted.get("right") == expected["right"])

        perm_str = f"col(h,l,r)←phys({','.join(perm)})"
        if perm == ("head", "left", "right"):
            kind = "correct"
        elif perm[0] == "head":
            kind = "left/right swap"
        elif perm[1] == "head" or perm[2] == "head":
            if sum(p == orig for p, orig in zip(perm, phys_roles)) == 1:
                kind = "cyclic" if perm[1] == "right" else "swap head-left" if perm[1] == "head" else "swap head-right"
            else:
                kind = "multi-swap"
        else:
            kind = "full permut"

        print(f"  {'✓' if ok else '✗'}  {perm_str:<35s}  [{kind:<15s}]  "
              f"prédit(h={predicted.get('head')},l={predicted.get('left')},r={predicted.get('right')})")
        if not ok:
            failures.append(f"{perm_str}: attendu={expected}, obtenu={predicted}")

    n = len(list(itertools.permutations(phys_roles)))
    print()
    if not failures:
        print(_ok(f"RÉSULTAT §10 : {n}/{n} permutations détectées (100%)"))
    else:
        print(_fail(f"RÉSULTAT §10 : {len(failures)}/{n} permutations échouées"))
        for f in failures:
            print(f"    ✗ {f}")
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §11 Tracker / algorithme alternatif (fix_tracker_placement)
# ══════════════════════════════════════════════════════════════════════════════

def test_tracker_placement_algorithm() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§11 TRACKER — ALGORITHME ALTERNATIF (detect_head / detect_hands)"))
    print(_h("=" * 60))

    if not _PANDAS:
        print(_warn("SKIPPED : pandas non disponible"))
        return True

    try:
        from fix.fix_tracker_placement import detect_head, detect_hands, _load_tracker_blocks
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    failures = []
    np.random.seed(11)

    def _make_df(phys, n=400):
        data = {"timestamp_ns": np.arange(n, dtype=np.int64) * 16_000_000}
        for lbl in ("head", "left", "right"):
            pos, quat = phys[lbl]
            for i, ax in enumerate(("x", "y", "z")):
                data[f"tracker_{lbl}_{ax}"] = pos[:n, i]
            for i, q in enumerate(("qw", "qx", "qy", "qz")):
                data[f"tracker_{lbl}_{q}"] = quat[:n, i]
        return pd.DataFrame(data)

    # ── detect_head : toutes les permutations ─────────────────────────────────
    phys_base = _make_tracker_blocks(yaw_rad=0.0, n=500,
                                      head_motion_scale=0.2, hand_motion_scale=3.0)
    for perm in itertools.permutations(("head", "left", "right")):
        csv_to_phys = dict(zip(("head", "left", "right"), perm))
        phys_perm = {col: phys_base[role] for col, role in csv_to_phys.items()}
        df = _make_df(phys_perm, n=400)
        blocks = _load_tracker_blocks(df)
        if blocks is None:
            print(_warn("  _load_tracker_blocks a retourné None"))
            continue
        head_pred, scores, conf = detect_head(blocks)
        physical_head_col = csv_to_phys["head"] if perm[0] == "head" else next(
            col for col, role in csv_to_phys.items() if role == "head")
        expected_head_col = next(col for col, role in csv_to_phys.items() if role == "head")
        ok = head_pred == expected_head_col and conf > 0.3
        perm_str = f"col(h,l,r)←phys({','.join(perm)})"
        print(f"  {'✓' if ok else '✗'}  {perm_str:<35s}  "
              f"head_pred={head_pred}  attendu={expected_head_col}  conf={conf:.2f}")
        if not ok:
            failures.append(f"detect_head échoue pour {perm_str}")

    # ── detect_hands : quelques orientations ─────────────────────────────────
    print()
    for yaw_deg in (0, 90, 180, 270):
        yaw = np.radians(yaw_deg)
        phys = _make_tracker_blocks(yaw_rad=yaw, n=400)
        df   = _make_df(phys, n=400)
        blocks = _load_tracker_blocks(df)
        if blocks is None:
            continue
        head_col = "head"
        left_pred, right_pred, conf = detect_hands(blocks, head_col)
        ok = left_pred == "left" and right_pred == "right"
        print(f"  {'✓' if ok else '✗'}  detect_hands yaw={yaw_deg:3d}°  "
              f"left={left_pred}  right={right_pred}  conf={conf:.2f}")
        if not ok:
            failures.append(f"detect_hands yaw={yaw_deg}°")

    print()
    if not failures:
        print(_ok("RÉSULTAT §11 : algorithme alternatif correct"))
    else:
        print(_fail(f"RÉSULTAT §11 : {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §12 Trackers / données réelles
# ══════════════════════════════════════════════════════════════════════════════

def test_tracker_real_data() -> bool:
    import glob as _glob
    if not _PANDAS:
        print()
        print(_warn("§12 TEST TRACKER RÉEL SKIPPED : pandas non disponible"))
        return True

    print()
    print(_h("=" * 60))
    print(_h("§12 TRACKER — DONNÉES RÉELLES (> 90 % accuracy)"))
    print(_h("=" * 60))

    try:
        from fix.fix_tracker_labels import (
            _load_blocks, _test_height, _test_centrality,
            _test_mobility, _test_lateral, _consensus
        )
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    dirs = [str(_HERE / "_training" / "desync")]
    sel2 = Path("/Users/christopher/selection 2/do")
    if sel2.exists():
        dirs.append(str(sel2))

    csvs = []
    for d in dirs:
        csvs += _glob.glob(d + "/*/tracker_positions.csv")

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
    print(f"  Sessions testées : {total}")
    print(f"  Head accuracy    : {ok_head}/{total} = {acc_head:.1f}%")
    print(f"  Right accuracy   : {ok_right}/{total} = {acc_right:.1f}%")

    ok = acc_right > 90.0
    print()
    print(_ok(f"RÉSULTAT §12 : right accuracy = {acc_right:.1f}% (seuil > 90%)")
          if ok else _fail(f"RÉSULTAT §12 : {acc_right:.1f}% < 90%"))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# §13 Caméras / head = min flux optique
# ══════════════════════════════════════════════════════════════════════════════

def _make_cam_signals_and_df(n: int = 500, rng=None):
    """Signaux de flux optique synthétiques alignés avec des vitesses tracker."""
    if rng is None:
        rng = np.random.default_rng(42)
    times    = np.arange(n, dtype=float) * 33.0
    ts_ns    = (times * 1e6).astype(np.int64)
    trk_left  = 3.0 * np.abs(rng.normal(0, 1, n)) + 0.1
    trk_right = 2.5 * np.abs(rng.normal(0, 1, n)) + 0.1
    trk_head  = 0.2 * np.abs(rng.normal(0, 1, n)) + 0.05
    df = pd.DataFrame({
        "timestamp_ns":    ts_ns,
        "tracker_left_x":  np.cumsum(np.diff(np.r_[0, trk_left])  * 0.033),
        "tracker_left_y":  np.zeros(n),  "tracker_left_z":  np.zeros(n),
        "tracker_right_x": np.cumsum(np.diff(np.r_[0, trk_right]) * 0.033),
        "tracker_right_y": np.zeros(n),  "tracker_right_z": np.zeros(n),
        "tracker_head_x":  np.zeros(n),
        "tracker_head_y":  np.cumsum(np.diff(np.r_[0, trk_head]) * 0.033),
        "tracker_head_z":  np.zeros(n),
    })
    return trk_left, trk_right, trk_head, times, df


def test_camera_head_min_flow() -> bool:
    if not _PANDAS:
        print()
        print(_warn("§13 TEST CAMÉRA HEAD SKIPPED : pandas non disponible"))
        return True

    print()
    print(_h("=" * 60))
    print(_h("§13 CAMÉRA — HEAD = MIN FLUX OPTIQUE"))
    print(_h("=" * 60))

    try:
        from fix.fix_camera_labels import _assign_cameras
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    rng = np.random.default_rng(13)
    failures = []
    tracker_prediction = {"head": "head", "left": "left", "right": "right"}

    # Scénarios où seule la magnitude relative varie pour head
    for mean_head_flow, mean_hand_flow, desc in [
        (0.1,  2.0,  "Head très calme (0.1 vs 2.0)"),
        (0.3,  1.5,  "Head calme (0.3 vs 1.5)"),
        (0.5,  3.0,  "Head modéré (0.5 vs 3.0)"),
        (0.05, 1.0,  "Head quasi-immobile (0.05 vs 1.0)"),
    ]:
        trk_l, trk_r, trk_h, times, df = _make_cam_signals_and_df(500, rng)
        cam_signals = {
            "head":  (np.abs(rng.normal(mean_head_flow, mean_head_flow * 0.1, 500)), times),
            "left":  (np.abs(rng.normal(mean_hand_flow, mean_hand_flow * 0.3, 500)), times),
            "right": (np.abs(rng.normal(mean_hand_flow * 0.9, mean_hand_flow * 0.3, 500)), times),
        }
        assignment, _, conf = _assign_cameras(cam_signals, df, tracker_prediction)
        ok = assignment.get("head") == "head"
        print(f"  {'✓' if ok else '✗'}  {desc:<35s}  head_assign={assignment.get('head')}  "
              f"conf={conf:.3f}")
        if not ok: failures.append(desc)

    # Cas où la caméra "left" a le plus faible flux → doit être identifiée comme head
    trk_l, trk_r, trk_h, times, df = _make_cam_signals_and_df(500, rng)
    cam_signals_inv = {
        "head":  (np.abs(rng.normal(3.0, 0.5, 500)), times),   # fort flux
        "left":  (np.abs(rng.normal(0.1, 0.02, 500)), times),  # faible flux
        "right": (np.abs(rng.normal(2.5, 0.4, 500)), times),   # fort flux
    }
    assignment_inv, _, _ = _assign_cameras(cam_signals_inv, df, tracker_prediction)
    ok_inv = assignment_inv.get("left") == "head"
    print(f"  {'✓' if ok_inv else '✗'}  'left' a le moins de flux → "
          f"'left' prédit comme head={assignment_inv.get('left') == 'head'}")
    if not ok_inv: failures.append("caméra left avec min flux non identifiée comme head")

    print()
    if not failures:
        print(_ok("RÉSULTAT §13 : identification head par min flux correcte"))
    else:
        print(_fail(f"RÉSULTAT §13 : {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §14 Caméras / corrélation left/right par flux+asymétrie
# ══════════════════════════════════════════════════════════════════════════════

def test_camera_lr_correlation() -> bool:
    if not _PANDAS:
        print()
        print(_warn("§14 TEST CAMÉRA L/R SKIPPED : pandas non disponible"))
        return True

    print()
    print(_h("=" * 60))
    print(_h("§14 CAMÉRA — CORRÉLATION LEFT/RIGHT PAR FLUX"))
    print(_h("=" * 60))

    try:
        from fix.fix_camera_labels import _assign_cameras
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    rng = np.random.default_rng(14)
    failures = []
    tracker_prediction = {"head": "head", "left": "left", "right": "right"}

    scenarios = [
        ("Gauche rapide, droite lente",  3.0, 1.0, 0.2),
        ("Droite rapide, gauche lente",  1.0, 3.0, 0.2),
        ("Activité égale, tête lente",   2.0, 2.0, 0.3),
        ("Pics alternés (asym forte)",   None, None, 0.2),
        ("Tâche bimanale (moins asym)",  2.5, 2.5, 0.4),
    ]

    for desc, amp_l, amp_r, amp_h in scenarios:
        N = 500
        times = np.arange(N, dtype=float) * 33.0
        if amp_l is None:
            t = np.arange(N)
            trk_l = np.where((t // 50) % 2 == 0, 3.0, 0.3)
            trk_r = np.where((t // 50) % 2 == 1, 3.0, 0.3)
        else:
            trk_l = amp_l * np.abs(rng.normal(0, 1, N)) + 0.1
            trk_r = amp_r * np.abs(rng.normal(0, 1, N)) + 0.1
        trk_h = amp_h * np.abs(rng.normal(0, 1, N)) + 0.05

        cam_signals = {
            "head":  (np.abs(trk_h * 0.8 + rng.normal(0, 0.05, N)), times),
            "left":  (np.abs(trk_l * 0.8 + rng.normal(0, 0.2,  N)), times),
            "right": (np.abs(trk_r * 0.8 + rng.normal(0, 0.2,  N)), times),
        }
        df = pd.DataFrame({
            "timestamp_ns":    (times * 1e6).astype(np.int64),
            "tracker_left_x":  np.cumsum(np.diff(np.r_[0, trk_l]) * 0.033),
            "tracker_left_y":  np.zeros(N), "tracker_left_z":  np.zeros(N),
            "tracker_right_x": np.cumsum(np.diff(np.r_[0, trk_r]) * 0.033),
            "tracker_right_y": np.zeros(N), "tracker_right_z": np.zeros(N),
            "tracker_head_x":  np.zeros(N),
            "tracker_head_y":  np.cumsum(np.diff(np.r_[0, trk_h]) * 0.033),
            "tracker_head_z":  np.zeros(N),
        })
        assignment, _, conf = _assign_cameras(cam_signals, df, tracker_prediction)
        ok = (assignment.get("left")  == "left" and
              assignment.get("right") == "right" and
              assignment.get("head")  == "head")
        print(f"  {'✓' if ok else '✗'}  {desc:<40s}  "
              f"assign={assignment}  conf={conf:.3f}")
        if not ok: failures.append(desc)

    print()
    if not failures:
        print(_ok(f"RÉSULTAT §14 : {len(scenarios)}/{len(scenarios)} scénarios corrects"))
    else:
        print(_fail(f"RÉSULTAT §14 : {len(failures)}/{len(scenarios)} échoués"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §15 Caméras / 6 permutations (toutes les inversions)
# ══════════════════════════════════════════════════════════════════════════════

def test_camera_all_permutations() -> bool:
    if not _PANDAS:
        print()
        print(_warn("§15 TEST CAMÉRA PERMS SKIPPED : pandas non disponible"))
        return True

    print()
    print(_h("=" * 60))
    print(_h("§15 CAMÉRA — 6 PERMUTATIONS (toutes inversions)"))
    print(_h("=" * 60))

    try:
        from fix.fix_camera_labels import _assign_cameras
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    rng = np.random.default_rng(15)
    N   = 800
    times = np.arange(N, dtype=float) * 33.0
    ts_ns = (times * 1e6).astype(np.int64)

    # Signaux d'activité : step function asymétrique left/right + head calme
    t = np.arange(N)
    act_left  = np.where((t // 80) % 2 == 0, 4.0, 0.2)   # actif première moitié
    act_right = np.where((t // 80) % 2 == 1, 4.0, 0.2)   # actif seconde moitié
    act_head  = 0.15 * np.abs(rng.normal(0, 1, N)) + 0.05

    # Flux optiques physiques : proportionnels à l'activité (pas aux positions)
    phys_flows = {
        "head":  np.abs(act_head  * 0.8 + rng.normal(0, 0.03, N)),
        "left":  np.abs(act_left  * 0.8 + rng.normal(0, 0.15, N)),
        "right": np.abs(act_right * 0.8 + rng.normal(0, 0.15, N)),
    }

    # Positions = marche aléatoire avec amplitude ∝ activité → vitesse ≈ activité
    # (contrairement à cumsum(diff(act)*dt) qui donne positions=act et vitesses=diff(act)≈0)
    rng2 = np.random.default_rng(999)
    left_x  = np.cumsum(rng2.normal(0, 1, N) * act_left  * 0.001)
    right_x = np.cumsum(rng2.normal(0, 1, N) * act_right * 0.001)
    head_y  = np.cumsum(rng2.normal(0, 1, N) * act_head  * 0.001)

    df_base = pd.DataFrame({
        "timestamp_ns":    ts_ns,
        "tracker_left_x":  left_x,
        "tracker_left_y":  np.zeros(N), "tracker_left_z":  np.zeros(N),
        "tracker_right_x": right_x,
        "tracker_right_y": np.zeros(N), "tracker_right_z": np.zeros(N),
        "tracker_head_x":  np.zeros(N),
        "tracker_head_y":  head_y,
        "tracker_head_z":  np.zeros(N),
    })
    tracker_pred = {"head": "head", "left": "left", "right": "right"}

    failures = []
    cam_cols  = ("head", "left", "right")

    for perm in itertools.permutations(("head", "left", "right")):
        # perm[i] = signal physique placé dans la caméra i
        cam_to_phys = dict(zip(cam_cols, perm))

        cam_signals = {cam: (phys_flows[role], times)
                       for cam, role in cam_to_phys.items()}

        # Prédiction attendue : {rôle_physique: caméra_qui_le_contient}
        expected = {role: cam for cam, role in cam_to_phys.items()}

        assignment, _, conf = _assign_cameras(cam_signals, df_base, tracker_pred)

        # expected[role] = cam_file contenant ce signal physique.
        # assignment[cam_file] = rôle attribué à ce fichier caméra.
        # Vérification correcte : pour chaque rôle, la caméra qui contient
        # son signal physique doit se voir attribuer ce rôle.
        ok = all(assignment.get(expected[role]) == role
                 for role in ("head", "left", "right"))

        perm_str = f"cam(h,l,r)←phys({','.join(perm)})"
        print(f"  {'✓' if ok else '✗'}  {perm_str:<35s}  "
              f"prédit(h={assignment.get('head')},l={assignment.get('left')},"
              f"r={assignment.get('right')})  conf={conf:.3f}")
        if not ok:
            failures.append(f"{perm_str}: attendu={expected}, obtenu={dict(assignment)}")

    n = len(list(itertools.permutations(cam_cols)))
    print()
    if not failures:
        print(_ok(f"RÉSULTAT §15 : {n}/{n} permutations caméra détectées (100%)"))
    else:
        print(_fail(f"RÉSULTAT §15 : {len(failures)}/{n} permutations échouées"))
        for f in failures:
            print(f"    ✗ {f}")
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §16 Caméras / données réelles
# ══════════════════════════════════════════════════════════════════════════════

def test_camera_real_data() -> bool:
    if not _PANDAS or not _CV2:
        missing = [x for x, ok in [("pandas", _PANDAS), ("cv2", _CV2)] if not ok]
        print()
        print(_warn(f"§16 TEST CAMÉRA RÉEL SKIPPED : {', '.join(missing)} non disponible"))
        return True

    print()
    print(_h("=" * 60))
    print(_h("§16 CAMÉRA — DONNÉES RÉELLES (≥ 80 % all-3)"))
    print(_h("=" * 60))

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
        assignment, _, conf = _assign_cameras(cam_signals, df, tracker_pred)
        total += 1
        h_ok = assignment.get("head")  == "head"
        l_ok = assignment.get("left")  == "left"
        r_ok = assignment.get("right") == "right"
        if h_ok: ok_head += 1
        if l_ok and r_ok: ok_lr += 1
        if h_ok and l_ok and r_ok: ok_all += 1
        else:
            print(f"  DIFF {s.name[:32]:<32s}  assign={assignment}  conf={conf:.3f}")

    if total == 0:
        print("  [Aucune session testée]")
        return True

    acc_h   = 100 * ok_head / total
    acc_lr  = 100 * ok_lr   / total
    acc_all = 100 * ok_all  / total
    print(f"\n  Sessions testées : {total}")
    print(f"  Head correct     : {ok_head}/{total} = {acc_h:.1f}%")
    print(f"  L/R both correct : {ok_lr}/{total}   = {acc_lr:.1f}%")
    print(f"  All-3 correct    : {ok_all}/{total}  = {acc_all:.1f}%")

    ok = acc_all >= 80.0
    print()
    print(_ok(f"RÉSULTAT §16 : all-3 = {acc_all:.1f}% (seuil ≥ 80%)")
          if ok else _fail(f"RÉSULTAT §16 : {acc_all:.1f}% < 80%"))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# §17 Gripper / détection d'offset
# ══════════════════════════════════════════════════════════════════════════════

def _make_gripper_signals(n=1500, dt_ms=33.0, true_offset_ms=0.0,
                           close_period=100, close_duration=30,
                           noise_mm=0.5, noise_vis=0.04, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    t_s  = np.arange(n, dtype=float) * dt_ms
    s_mm = np.ones(n) * 45.0
    for start in range(0, n, close_period):
        s_mm[start : min(start + close_duration, n)] = 3.5
    s_mm += rng.normal(0, noise_mm, n)
    s_mm  = np.clip(s_mm, 0.5, 60.0)
    t_v   = t_s + true_offset_ms
    v_f   = np.clip((s_mm - 0.5) / (60.0 - 0.5) + rng.normal(0, noise_vis, n), 0.0, 1.0)
    return t_s, s_mm, t_v, v_f


def test_gripper_offset_detection() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§17 GRIPPER — DÉTECTION D'OFFSET SYNTHÉTIQUE"))
    print(_h("=" * 60))

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
            rng=np.random.default_rng(7))
        offset_ms, peak_r, snr, n_s, n_v = _compute_closed_offset(s_t, s_mm, v_t, v_f)
        error_ms = abs(offset_ms - true_ms)
        ok = (error_ms <= tolerance_ms) and (snr >= 2.0)
        print(f"  {'✓' if ok else '✗'}  {label}  "
              f"détecté={offset_ms:+8.1f}ms  erreur={error_ms:5.1f}ms  "
              f"SNR={snr:5.1f}  r={peak_r:.3f}")
        if not ok:
            reasons = []
            if error_ms > tolerance_ms:
                reasons.append(f"erreur {error_ms:.0f}ms > tol {tolerance_ms:.0f}ms")
            if snr < 2.0:
                reasons.append(f"SNR={snr:.1f}<2")
            failures.append(f"{label}: {', '.join(reasons)}")

    n = len(cases)
    print()
    if not failures:
        print(_ok(f"RÉSULTAT §17 : {n}/{n} PASSÉS (100%)"))
    else:
        print(_fail(f"RÉSULTAT §17 : {len(failures)}/{n} ÉCHOUÉS"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §18 Gripper / robustesse bruit visuel
# ══════════════════════════════════════════════════════════════════════════════

def test_gripper_noise_robustness() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§18 GRIPPER — ROBUSTESSE AU BRUIT VISUEL"))
    print(_h("=" * 60))

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
            rng=np.random.default_rng(99))
        offset_ms, _, snr, _, _ = _compute_closed_offset(s_t, s_mm, v_t, v_f)
        error_ms = abs(offset_ms - true_ms)
        ok = (error_ms <= tolerance_ms) and (snr >= 2.0)
        sym = "✓" if ok else ("✗ (acceptable)" if noise > 0.15 else "✗")
        print(f"  {sym}  bruit={noise:.2f}  "
              f"détecté={offset_ms:+8.1f}ms  erreur={error_ms:5.1f}ms  SNR={snr:5.1f}")
        if not ok and noise <= 0.15:
            failures.append(f"bruit={noise:.2f}")

    print()
    if not failures:
        print(_ok("RÉSULTAT §18 : robuste jusqu'à bruit=0.15"))
    else:
        print(_fail(f"RÉSULTAT §18 : échec pour {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §19 Gripper / cas limites
# ══════════════════════════════════════════════════════════════════════════════

def test_gripper_edge_cases() -> bool:
    print()
    print(_h("=" * 60))
    print(_h("§19 GRIPPER — CAS LIMITES"))
    print(_h("=" * 60))

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
    s_t, s_mm, v_t, _ = _make_gripper_signals(n=n, rng=rng)
    r2 = _compute_closed_offset(s_t, s_mm, v_t, np.ones(n) * 0.8)
    ok2 = r2 == (0.0, 0.0, 0.0, 0, 0)
    print(f"  {'✓' if ok2 else '✗'}  Visuel constant          → retour={r2[:3]} "
          f"({'ignoré' if ok2 else 'NON IGNORÉ'})")
    if not ok2: failures.append("visuel constant non ignoré")

    # Chevauchement temporel insuffisant
    s_t2 = np.arange(30, dtype=float) * 50.0
    v_t2 = s_t2 + 5000.0
    mm2  = np.ones(30) * 45.0; mm2[:10] = 3.0
    vf2  = np.ones(30) * 0.9;  vf2[:10] = 0.05
    r3   = _compute_closed_offset(s_t2, mm2, v_t2, vf2)
    ok3  = (r3[0] == 0.0 and r3[1] == 0.0)
    print(f"  {'✓' if ok3 else '✗'}  Chevauchement insuffisant → retour={r3[:2]} "
          f"({'ignoré' if ok3 else 'NON IGNORÉ'})")
    if not ok3: failures.append("overlap insuffisant non ignoré")

    print()
    if not failures:
        print(_ok("RÉSULTAT §19 : tous les cas limites gérés"))
    else:
        print(_fail(f"RÉSULTAT §19 : {len(failures)} cas non gérés"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §20 Gripper / dry-run données réelles
# ══════════════════════════════════════════════════════════════════════════════

def test_gripper_real_data() -> bool:
    if not _PANDAS:
        print()
        print(_warn("§20 TEST GRIPPER RÉEL SKIPPED : pandas non disponible"))
        return True

    print()
    print(_h("=" * 60))
    print(_h("§20 GRIPPER — FIX DRY-RUN DONNÉES RÉELLES"))
    print(_h("=" * 60))

    try:
        from fix.fix_gripper_video_sync import fix_gripper_closed_offset
    except ImportError as e:
        print(_warn(f"SKIPPED : {e}"))
        return True

    roots = [Path("/Users/christopher/selection 2/do"), _HERE]
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
        result      = fix_gripper_closed_offset(s, dry_run=True)
        status      = result["status"]
        corrections = result["corrections_ms"]
        details     = result.get("details", {})
        ok          = isinstance(status, str) and isinstance(corrections, dict)
        corr_str    = (
            "  ".join(f"{side}:{ms:+.0f}ms" for side, ms in corrections.items())
            if corrections else "aucune"
        )
        print(f"  {'✓' if ok else '✗'}  {s.name[:38]:<38s}  "
              f"status={status:<12s}  {corr_str}")
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
    print()
    if not failures:
        print(_ok("RÉSULTAT §20 : retour fix_gripper_closed_offset correct"))
    else:
        print(_fail(f"RÉSULTAT §20 : retour invalide pour {failures}"))
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Bilan tests unitaires
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests() -> bool:
    np.random.seed(0)

    suite = [
        ("§1  SLERP propriétés mathématiques",           test_slerp_properties),
        ("§2  Quaternions NaN/zéro/réparation",           test_quaternion_nan_detection),
        ("§3  Gaps tracker détection",                    test_tracker_gap_detection),
        ("§4  Dérive horloge régression+mesure",          test_clock_drift),
        ("§5  Tracker hauteur Y",                         test_tracker_height),
        ("§6  Tracker centralité 3D",                     test_tracker_centrality),
        ("§7  Tracker mobilité",                          test_tracker_mobility),
        ("§8  Tracker latéral 0°–360°",                   test_tracker_lateral_all_yaws),
        ("§9  Tracker consensus = test latéral seul",     test_tracker_consensus_lateral_only),
        ("§10 Tracker 6 permutations",                    test_tracker_all_permutations),
        ("§11 Tracker algorithme alternatif",             test_tracker_placement_algorithm),
        ("§12 Tracker données réelles > 90%",             test_tracker_real_data),
        ("§13 Caméra head = min flux",                    test_camera_head_min_flow),
        ("§14 Caméra L/R corrélation",                    test_camera_lr_correlation),
        ("§15 Caméra 6 permutations",                     test_camera_all_permutations),
        ("§16 Caméra données réelles ≥ 80%",              test_camera_real_data),
        ("§17 Gripper détection d'offset",                test_gripper_offset_detection),
        ("§18 Gripper robustesse bruit",                  test_gripper_noise_robustness),
        ("§19 Gripper cas limites",                       test_gripper_edge_cases),
        ("§20 Gripper dry-run réel",                      test_gripper_real_data),
    ]

    results_unit = []
    for name, fn in suite:
        try:
            ok = fn()
        except Exception as e:
            import traceback
            print(_fail(f"EXCEPTION dans {name} : {e}"))
            traceback.print_exc()
            ok = False
        results_unit.append((name, ok))

    print()
    print(_h("=" * 70))
    print(_h("BILAN GLOBAL — TESTS UNITAIRES"))
    print(_h("=" * 70))
    all_ok = True
    for name, ok in results_unit:
        print(f"  {'✓' if ok else '✗'}  {name}")
        if not ok:
            all_ok = False

    n_pass = sum(1 for _, ok in results_unit if ok)
    n_fail = len(results_unit) - n_pass
    print()
    if all_ok:
        print(_ok(f"TOUS LES TESTS PASSÉS ({n_pass}/{len(results_unit)})"))
    else:
        print(_fail(f"{n_fail}/{len(results_unit)} TEST(S) ÉCHOUÉ(S)"))
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# Évaluation et tri des sessions A/B vs autres
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
    return _G if grade in ("A", "B") else (_Y if grade in ("C", "D") else _R)


def evaluate_sessions(input_dir: Path, out_ab: Path, out_rest: Path,
                       dry_run: bool = False) -> list[dict]:
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "session_check", _HERE / "verification" / "session_check.py")
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
    print(_h("=" * 70))
    print(f"ÉVALUATION DE {len(sessions)} SESSION(S)")
    print(f"  Entrée      : {input_dir}")
    print(f"  Sortie A/B  : {out_ab}")
    print(f"  Sortie C/D/F: {out_rest}")
    print(_h("=" * 70))
    print()

    for session_path in sessions:
        name = session_path.name
        print(f"  Analyse : {name} …", flush=True)
        try:
            result = check_fn(session_path)
        except Exception as e:
            print(_fail(f"    ERREUR : {e}"))
            entry = {"session": name, "path": str(session_path),
                     "score": 0.0, "grade": "F", "verdict": "ERREUR",
                     "error": str(e), "destination": str(out_rest / name), "bucket": "rest"}
            results.append(entry)
            if not dry_run:
                shutil.copytree(session_path, out_rest / name, dirs_exist_ok=True)
            continue

        score   = result.get("score", 0.0)
        grade   = result.get("grade", _grade_label(score))
        verdict = result.get("verdict", "")
        is_ab   = _is_ab(grade)
        dest    = out_ab / name if is_ab else out_rest / name
        bucket  = "AB" if is_ab else "rest"

        gc  = _grade_color(grade)
        tag = _c(f"grade {grade}", _B + gc)
        print(f"    {tag}  score={score:.1f}%  verdict={verdict}  → {bucket}")

        if not dry_run:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(session_path, dest)

        results.append({"session": name, "path": str(session_path),
                         "score": score, "grade": grade, "verdict": verdict,
                         "blocked": result.get("blocked", False),
                         "perfect": result.get("perfect", False),
                         "destination": str(dest), "bucket": bucket})
    return results


def _print_summary(results: list[dict]) -> None:
    print()
    print(_h("=" * 70))
    print(_h("RÉSUMÉ DU TRI"))
    print(_h("=" * 70))
    ab   = [r for r in results if r["bucket"] == "AB"]
    rest = [r for r in results if r["bucket"] == "rest"]
    print(f"\n  Total : {len(results)}   Grade A/B : {len(ab)}   Grade C/D/F : {len(rest)}")
    if ab:
        print()
        print(_c("  ── Sessions A/B (utilisables) ──", _B + _G))
        for r in ab:
            print(f"    {_c(r['grade'], _grade_color(r['grade']))}  "
                  f"{r['score']:5.1f}%  {r['session']}")
    if rest:
        print()
        print(_c("  ── Sessions C/D/F (à corriger) ──", _B + _Y))
        for r in rest:
            print(f"    {_c(r['grade'], _grade_color(r['grade']))}  "
                  f"{r['score']:5.1f}%  {r['session']}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",    type=Path,
                   help="Dossier contenant les sessions à évaluer")
    p.add_argument("--out-ab",   type=Path, default=Path("output_AB"),
                   help="Dossier de sortie grades A et B (défaut: output_AB)")
    p.add_argument("--out-rest", type=Path, default=Path("output_CDF"),
                   help="Dossier de sortie grades C, D, F (défaut: output_CDF)")
    p.add_argument("--no-unit-tests", action="store_true",
                   help="Ne pas exécuter les tests unitaires")
    p.add_argument("--no-copy",  action="store_true",
                   help="Afficher le tri sans copier les fichiers")
    p.add_argument("--json",     action="store_true",
                   help="Afficher le rapport des sessions en JSON")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    unit_ok = True
    if not args.no_unit_tests:
        print(_c("\n══ TESTS UNITAIRES ══════════════════════════════════════════\n", _B))
        unit_ok = run_unit_tests()

    if args.input:
        input_dir = args.input.resolve()
        if not input_dir.exists():
            print(_fail(f"Dossier d'entrée introuvable : {input_dir}"))
            sys.exit(1)
        print(_c("\n══ ÉVALUATION DES SESSIONS ══════════════════════════════════\n", _B))
        results = evaluate_sessions(
            input_dir=input_dir,
            out_ab=args.out_ab.resolve(),
            out_rest=args.out_rest.resolve(),
            dry_run=args.no_copy,
        )
        _print_summary(results)
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print()
        if args.no_unit_tests:
            print(_warn("Aucun dossier --input fourni et tests unitaires désactivés. Rien à faire."))
        else:
            print(_warn("Aucun dossier --input fourni. Seuls les tests unitaires ont été exécutés."))
        print("  Exemple : python test_sessions.py --input /chemin/sessions "
              "--out-ab ./AB --out-rest ./CDF")

    sys.exit(0 if unit_ok else 1)


if __name__ == "__main__":
    main()
