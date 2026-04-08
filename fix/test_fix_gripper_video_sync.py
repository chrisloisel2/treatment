#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/test_fix_gripper_video_sync.py — Validation de la synchronisation gripper-vidéo
                                     par convolution sur la position fermée.

Tests :
1. Données synthétiques — vérifie que _compute_closed_offset détecte correctement
   un offset connu (plusieurs valeurs, avec/sans bruit).
2. Cas limites — vérifie la robustesse face aux dégénérescences (pas de fermetures,
   signal constant, chevauchement insuffisant).
3. Données réelles (dry-run) — appelle fix_gripper_closed_offset sur les sessions
   disponibles et affiche le retour complet.

Usage :
    python -m fix.test_fix_gripper_video_sync
    python fix/test_fix_gripper_video_sync.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fix.fix_gripper_video_sync import (
    _compute_closed_offset,
    fix_gripper_closed_offset,
    CONV_RESAMPLE_MS,
    CONV_MIN_CLOSED_FRAMES,
    OFFSET_SIGNIFICANT_MS,
    _load_gripper_df,
    _load_jsonl_times,
)

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ── Générateur de signal synthétique ─────────────────────────────────────────

def _make_signals(
    n: int = 1500,
    dt_ms: float = 33.0,
    true_offset_ms: float = 0.0,
    close_period: int = 100,
    close_duration: int = 30,
    noise_mm: float = 0.5,
    noise_vis: float = 0.04,
    rng: "np.random.Generator | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Retourne (sensor_t_ms, sensor_mm, visual_t_ms, visual_frac).

    Le signal visuel représente le même état physique que le capteur, mais
    horodaté avec un décalage de true_offset_ms :
        visual_t_ms = sensor_t_ms + true_offset_ms

    true_offset_ms > 0 → capteur en avance sur la vidéo.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sensor_t_ms = np.arange(n, dtype=float) * dt_ms
    sensor_mm   = np.ones(n) * 45.0
    for start in range(0, n, close_period):
        sensor_mm[start : min(start + close_duration, n)] = 3.5
    sensor_mm += rng.normal(0, noise_mm, n)
    sensor_mm  = np.clip(sensor_mm, 0.5, 60.0)

    visual_t_ms = sensor_t_ms + true_offset_ms
    visual_frac = (sensor_mm - 0.5) / (60.0 - 0.5)
    visual_frac = np.clip(visual_frac + rng.normal(0, noise_vis, n), 0.0, 1.0)

    return sensor_t_ms, sensor_mm, visual_t_ms, visual_frac


# ── TEST 1 : détection d'offset ───────────────────────────────────────────────

def test_synthetic_offset_detection() -> bool:
    """
    Vérifie que _compute_closed_offset retrouve des offsets connus
    avec une précision ≤ 2 × CONV_RESAMPLE_MS et un SNR ≥ 2.
    """
    print("=" * 60)
    print("TEST 1 : DÉTECTION D'OFFSET — DONNÉES SYNTHÉTIQUES")
    print("=" * 60)

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
        s_t, s_mm, v_t, v_f = _make_signals(
            n=1500, dt_ms=33.0, true_offset_ms=true_ms,
            close_period=100, close_duration=30,
            noise_mm=0.3, noise_vis=0.04,
            rng=np.random.default_rng(7),
        )
        offset_ms, peak_r, snr, n_s, n_v = _compute_closed_offset(s_t, s_mm, v_t, v_f)

        error_ms = abs(offset_ms - true_ms)
        ok = (error_ms <= tolerance_ms) and (snr >= 2.0)
        sym = "✓" if ok else "✗"

        print(
            f"  {sym}  {label}  "
            f"détecté={offset_ms:+8.1f}ms  "
            f"erreur={error_ms:5.1f}ms  "
            f"SNR={snr:5.1f}  r={peak_r:.3f}"
        )
        if not ok:
            reasons = []
            if error_ms > tolerance_ms:
                reasons.append(f"erreur {error_ms:.0f}ms > tolérance {tolerance_ms:.0f}ms")
            if snr < 2.0:
                reasons.append(f"SNR={snr:.1f} < 2")
            failures.append(f"{label}: {', '.join(reasons)}")

    n = len(cases)
    print()
    if not failures:
        print(f"✓ RÉSULTAT TEST 1 : {n}/{n} PASSÉS (100%)")
    else:
        for f in failures:
            print(f"  ✗ {f}")
        print(f"✗ RÉSULTAT TEST 1 : {len(failures)}/{n} ÉCHOUÉS")
    return len(failures) == 0


# ── TEST 2 : robustesse bruit visuel ─────────────────────────────────────────

def test_noise_robustness() -> bool:
    """
    Vérifie que la détection reste fiable jusqu'à 15 % de bruit visuel.
    """
    print()
    print("=" * 60)
    print("TEST 2 : ROBUSTESSE AU BRUIT VISUEL")
    print("=" * 60)

    true_ms      = 300.0
    tolerance_ms = 2.0 * CONV_RESAMPLE_MS
    failures     = []

    for noise in (0.02, 0.05, 0.10, 0.15, 0.20):
        s_t, s_mm, v_t, v_f = _make_signals(
            n=1500, true_offset_ms=true_ms,
            noise_mm=0.3, noise_vis=noise,
            rng=np.random.default_rng(99),
        )
        offset_ms, peak_r, snr, _, _ = _compute_closed_offset(s_t, s_mm, v_t, v_f)
        error_ms = abs(offset_ms - true_ms)
        ok = (error_ms <= tolerance_ms) and (snr >= 2.0)
        sym = "✓" if ok else ("✗ (acceptable)" if noise > 0.15 else "✗")
        print(
            f"  {sym}  bruit={noise:.2f}  "
            f"détecté={offset_ms:+8.1f}ms  "
            f"erreur={error_ms:5.1f}ms  "
            f"SNR={snr:5.1f}  r={peak_r:.3f}"
        )
        if not ok and noise <= 0.15:
            failures.append(f"bruit={noise:.2f}")

    print()
    if not failures:
        print("✓ RÉSULTAT TEST 2 : robuste jusqu'à bruit=0.15")
    else:
        print(f"✗ RÉSULTAT TEST 2 : échec pour {failures}")
    return len(failures) == 0


# ── TEST 3 : cas limites ──────────────────────────────────────────────────────

def test_edge_cases() -> bool:
    """
    Vérifie la gestion correcte des cas dégénérés (retour (0,0,0,0,0) attendu).
    """
    print()
    print("=" * 60)
    print("TEST 3 : CAS LIMITES")
    print("=" * 60)

    failures = []
    rng = np.random.default_rng(1)
    n   = 600

    # ── signal capteur toujours ouvert ───────────────────────────────────────
    t  = np.arange(n, dtype=float) * 33.0
    mm = np.ones(n) * 45.0 + rng.normal(0, 0.3, n)
    vf = np.ones(n) * 0.9  + rng.normal(0, 0.01, n)
    r  = _compute_closed_offset(t, mm, t, vf)
    ok = r[3] < CONV_MIN_CLOSED_FRAMES
    print(f"  {'✓' if ok else '✗'}  Capteur toujours ouvert  → n_s={r[3]} "
          f"({'ignoré' if ok else 'NON IGNORÉ'})")
    if not ok:
        failures.append("capteur toujours ouvert non ignoré")

    # ── signal visuel constant ────────────────────────────────────────────────
    s_t, s_mm, v_t, _ = _make_signals(n=n, true_offset_ms=0.0, rng=rng)
    r = _compute_closed_offset(s_t, s_mm, v_t, np.ones(n) * 0.8)
    ok = r == (0.0, 0.0, 0.0, 0, 0)
    print(f"  {'✓' if ok else '✗'}  Visuel constant          → retour={r[:3]} "
          f"({'ignoré' if ok else 'NON IGNORÉ'})")
    if not ok:
        failures.append("visuel constant non ignoré")

    # ── chevauchement temporel insuffisant ────────────────────────────────────
    s_t2 = np.arange(30, dtype=float) * 50.0
    v_t2 = s_t2 + 5000.0           # vidéo 5s plus loin → 0s d'overlap
    mm2  = np.ones(30) * 45.0; mm2[:10] = 3.0
    vf2  = np.ones(30) * 0.9;  vf2[:10] = 0.05
    r    = _compute_closed_offset(s_t2, mm2, v_t2, vf2)
    ok   = (r[0] == 0.0 and r[1] == 0.0)
    print(f"  {'✓' if ok else '✗'}  Chevauchement insuffisant → retour={r[:2]} "
          f"({'ignoré' if ok else 'NON IGNORÉ'})")
    if not ok:
        failures.append("overlap insuffisant non ignoré")

    print()
    if not failures:
        print("✓ RÉSULTAT TEST 3 : tous les cas limites gérés")
    else:
        print(f"✗ RÉSULTAT TEST 3 : {len(failures)} cas non gérés")
    return len(failures) == 0


# ── TEST 4 : fix dry-run sur données réelles ──────────────────────────────────

def test_real_data_fix() -> bool:
    """
    Appelle fix_gripper_closed_offset (dry_run=True) sur les sessions réelles
    et affiche le retour complet pour inspection.

    Sans MP4, l'analyse convolution retourne no_data (SNR=0) →
    fix non appliqué → status=ok (pas de correction nécessaire / indéterminable).
    La valeur retournée est affichée intégralement pour vérification.
    """
    if not _PANDAS:
        print("\n[TEST 4 SKIPPED : pandas non disponible]")
        return True

    print()
    print("=" * 60)
    print("TEST 4 : FIX DRY-RUN — DONNÉES RÉELLES")
    print("=" * 60)

    roots = [
        Path("/Users/christopher/selection 2/do"),
        Path("/Users/christopher/treatment"),
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

    failures   = []
    n_ok       = 0
    n_corrected= 0

    for s in sessions:
        result = fix_gripper_closed_offset(s, dry_run=True)

        status       = result["status"]
        corrections  = result["corrections_ms"]
        details      = result.get("details", {})

        # Résumé compact par session
        ok = isinstance(status, str) and isinstance(corrections, dict)

        sym = "✓" if ok else "✗"
        corr_str = (
            "  ".join(f"{side}:{ms:+.0f}ms" for side, ms in corrections.items())
            if corrections else "aucune correction"
        )
        print(f"  {sym}  {s.name[:38]:<38s}  status={status:<12s}  {corr_str}")

        # Détail par côté
        for side, d in details.items():
            snr = d.get("snr", 0.0)
            ofs = d.get("offset_ms", 0.0)
            st  = d.get("status", "?")
            issues = d.get("issues", [])
            print(
                f"       [{side}]  offset={ofs:+7.1f}ms  SNR={snr:4.1f}  "
                f"n_s={d.get('n_closed_sensor',0):3d}  "
                f"n_v={d.get('n_closed_visual',0):3d}  "
                f"st={st}"
            )
            for iss in issues:
                print(f"            ↳ {iss}")

        if ok:
            n_ok += 1
        else:
            failures.append(s.name)

        if corrections:
            n_corrected += 1

    print()
    print(f"  Sessions analysées  : {len(sessions)}")
    print(f"  Retour valide       : {n_ok}/{len(sessions)}")
    print(f"  Corrections détectées (dry-run) : {n_corrected}")
    print()
    if not failures:
        print("✓ RÉSULTAT TEST 4 : retour fix_gripper_closed_offset correct")
    else:
        print(f"✗ RÉSULTAT TEST 4 : retour invalide pour {failures}")
    return len(failures) == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    results: list[tuple[str, bool]] = []

    results.append(("Détection offset synthétique (100%)", test_synthetic_offset_detection()))
    results.append(("Robustesse bruit (≤ 15 %)",           test_noise_robustness()))
    results.append(("Cas limites",                          test_edge_cases()))
    results.append(("Fix dry-run données réelles",          test_real_data_fix()))

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
    print("Résultat final : %s" % (
        "✓ TOUS LES TESTS PASSÉS" if all_ok else "✗ DES TESTS ONT ÉCHOUÉ"
    ))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
