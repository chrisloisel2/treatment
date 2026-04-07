#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_tracker_labels.py — Identification CERTAINE des trackers head/left/right.

Objectif : attribuer avec certitude absolue les rôles head / left / right aux
3 trackers présents dans tracker_positions.csv.

Algorithme (4 tests indépendants — tous doivent s'accorder) :
─────────────────────────────────────────────────────────────
TEST 1 — HAUTEUR Y (critère physique absolu)
    En contexte VR debout, la tête EST au-dessus des mains.
    → Sépare le head avec un z-score statistique.
    → Certitude si z-score > ZSCORE_CERTAIN (≈ 5σ par défaut).

TEST 2 — CENTRALITÉ 3D
    Le head est géométriquement entre les deux mains
    (distance médiane aux autres trackers la plus faible).

TEST 3 — PROFIL DE MOBILITÉ
    La tête bouge moins vite que les mains
    (médiane de la norme de déplacement inter-frame).

TEST 4 — POSITION MOYENNE RELATIVE À LA TÊTE (left/right)
    Pour chaque main, la POSITION MOYENNE sur toute la session (relative à la
    tête) est projetée sur l'axe X MOYEN du head (calculé depuis la moyenne
    des matrices de rotation quaternion, stable aux outliers).
    Main droite → projection positive (axe X local du head).
    Main gauche → projection négative.
    NOTE : seul ce test détermine left/right dans le consensus — les tests
    1-3 n'ont pas d'information physique sur la latéralité.

CERTITUDE :
    Un rôle est CERTAIN si au moins 3/4 tests s'accordent ET
    que le test de hauteur (Test 1) est ≥ ZSCORE_CERTAIN.
    En cas de désaccord → statut "uncertain", aucune modification.

Usage :
    python -m fix.fix_tracker_labels /chemin/session [--dry-run] [--force]

    from fix.fix_tracker_labels import fix_tracker_labels
    report = fix_tracker_labels(Path("/chemin/session"))
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
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


# ── Constantes ────────────────────────────────────────────────────────────────

MARKER_KEY      = "tracker_labels_verified"
TRACKERS        = ("head", "left", "right")
AXES            = ("x", "y", "z")
QUATS           = ("qw", "qx", "qy", "qz")
SMOOTH_WIN      = 7          # lissage positions (réduction bruit)
ZSCORE_CERTAIN  = 5.0        # z-score pour certitude absolue (hauteur)
ZSCORE_LIKELY   = 2.0        # z-score pour "probable" (requis si AGREE_ALL)
AGREE_MIN       = 3          # tests qui doivent s'accorder (sur 4)
AGREE_ALL       = 4          # 4/4 → compense un z-score plus faible


# ── Structures ────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    head_vote: str          # lequel des 3 trackers est prédit comme head
    left_vote: str
    right_vote: str
    confidence: float       # 0–1
    evidence: dict = field(default_factory=dict)


@dataclass
class TrackerLabelReport:
    session: str
    status: str             # "ok" | "corrected" | "uncertain" | "error"
    reason: str = ""
    tests: list[TestResult] = field(default_factory=list)
    agreement_count: int = 0  # nb de tests qui s'accordent
    predicted: dict = field(default_factory=dict)  # {role: label_csv}
    old_assignment: dict = field(default_factory=dict)
    corrected: bool = False
    dry_run: bool = False


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    """Moyenne glissante causal sur chaque colonne."""
    if win <= 1 or len(arr) < win:
        return arr.copy()
    k = np.ones(win) / win
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(arr[:, j], k, mode="same")
    return out


def _quat_rotmat(q: np.ndarray) -> np.ndarray:
    """(n,4) wxyz → (n,3,3) matrices de rotation."""
    q = q.astype(float)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2*(y*y + z*z);  R[:, 0, 1] = 2*(x*y - z*w);  R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w);      R[:, 1, 1] = 1 - 2*(x*x + z*z);  R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w);      R[:, 2, 1] = 2*(y*z + x*w);  R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def _load_blocks(df: pd.DataFrame) -> Optional[list[tuple[str, np.ndarray, np.ndarray]]]:
    """Charge les 3 blocs (label, pos_smooth, quat) depuis le DataFrame."""
    blocks = []
    for label in TRACKERS:
        pc = [f"tracker_{label}_{a}" for a in AXES]
        qc = [f"tracker_{label}_{q}" for q in QUATS]
        if not all(c in df.columns for c in pc + qc):
            return None
        pos  = np.stack([pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float) for c in pc], axis=1)
        quat = np.stack([pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float) for c in qc], axis=1)
        valid = np.isfinite(pos).all(axis=1) & np.isfinite(quat).all(axis=1)
        if valid.sum() < 20:
            return None
        blocks.append((label, _smooth(pos[valid]), quat[valid]))
    return blocks if len(blocks) == 3 else None


# ── Tests indépendants ────────────────────────────────────────────────────────

def _test_height(blocks: list) -> TestResult:
    """
    Test 1 : axe Y vertical.
    Le head a le Y médian le plus élevé.
    Certitude = z-score de séparation entre le best et le second.
    """
    labels  = [b[0] for b in blocks]
    medians = np.array([np.median(b[1][:, 1]) for b in blocks])   # Y median
    stds    = np.array([np.std(b[1][:, 1])    for b in blocks])

    order   = np.argsort(medians)[::-1]   # décroissant
    head_i  = int(order[0])
    delta_y = float(medians[order[0]] - medians[order[1]])
    # z-score : séparation divisée par l'écart-type du vainqueur (pire cas)
    pooled_std = float(max(stds[order[0]], stds[order[1]], 1e-6))
    zscore = delta_y / pooled_std

    conf = float(np.clip(zscore / (ZSCORE_CERTAIN * 2), 0.0, 1.0))

    # Les deux non-head sont dans l'ordre qui sera affiné par le test latéral
    others = [labels[order[1]], labels[order[2]]]
    return TestResult(
        name="height_Y",
        head_vote=labels[head_i],
        left_vote=others[0],   # provisoire
        right_vote=others[1],  # provisoire
        confidence=conf,
        evidence={
            "medians_Y": {lb: round(float(m), 4) for lb, m in zip(labels, medians)},
            "delta_Y_m": round(delta_y, 4),
            "zscore":    round(zscore, 2),
            "certain":   zscore >= ZSCORE_CERTAIN,
        },
    )


def _test_centrality(blocks: list) -> TestResult:
    """
    Test 2 : centralité 3D.
    Le head a la distance médiane aux deux autres la plus faible
    (il est entre les deux mains).
    """
    labels = [b[0] for b in blocks]
    mean_dists = np.zeros(3)
    for i in range(3):
        dists = []
        for j in range(3):
            if i == j:
                continue
            n = min(len(blocks[i][1]), len(blocks[j][1]))
            d = np.linalg.norm(blocks[i][1][:n] - blocks[j][1][:n], axis=1)
            dists.append(float(np.median(d)))
        mean_dists[i] = np.mean(dists)

    # Le head est le MOINS éloigné des autres (centralité)
    head_i = int(np.argmin(mean_dists))
    others = [i for i in range(3) if i != head_i]
    # Séparation relative
    sorted_d = np.sort(mean_dists)
    sep = float((sorted_d[1] - sorted_d[0]) / (np.ptp(mean_dists) + 1e-9))
    conf = float(np.clip(sep, 0.0, 1.0))

    others_labels = [labels[others[0]], labels[others[1]]]
    return TestResult(
        name="centrality_3D",
        head_vote=labels[head_i],
        left_vote=others_labels[0],
        right_vote=others_labels[1],
        confidence=conf,
        evidence={
            "mean_dist_m": {lb: round(float(d), 4) for lb, d in zip(labels, mean_dists)},
            "separation":  round(sep, 3),
        },
    )


def _test_mobility(blocks: list) -> TestResult:
    """
    Test 3 : mobilité (norme déplacement inter-frame).
    La tête bouge moins vite que les mains.
    """
    labels = [b[0] for b in blocks]
    mobilities = np.array([
        float(np.median(np.linalg.norm(np.diff(b[1], axis=0), axis=1)))
        for b in blocks
    ])

    head_i = int(np.argmin(mobilities))  # moins mobile = head
    others = [i for i in range(3) if i != head_i]
    sep = float((np.sort(mobilities)[1] - np.sort(mobilities)[0]) / (np.ptp(mobilities) + 1e-9))
    conf = float(np.clip(sep, 0.0, 1.0))

    others_labels = [labels[others[0]], labels[others[1]]]
    return TestResult(
        name="mobility",
        head_vote=labels[head_i],
        left_vote=others_labels[0],
        right_vote=others_labels[1],
        confidence=conf,
        evidence={
            "median_speed_m_per_frame": {lb: round(float(m), 6)
                                         for lb, m in zip(labels, mobilities)},
            "separation": round(sep, 3),
        },
    )


def _test_lateral(blocks: list, head_label: str) -> TestResult:
    """
    Test 4 : position MOYENNE relative à la tête (left/right).

    Algorithme robuste :
    1. Pour chaque main, calcule la POSITION MOYENNE relative à la tête
       sur l'ensemble de la session (intégration temporelle complète).
    2. Projette ce vecteur moyen sur l'axe X MOYEN du head (calculé depuis
       la moyenne des matrices de rotation quaternion → stable aux outliers).
    3. Main droite → projection positive (axe X local = droite du head).
       Main gauche → projection négative.

    Avantage vs frame-par-frame avec médiane :
    - La moyenne des positions lisse les artefacts de tracking.
    - L'axe X moyen de la tête est insensible aux sauts de quaternion.
    - Résultat déterministe et robuste sur toute la session.
    """
    head_block   = next(b for b in blocks if b[0] == head_label)
    other_blocks = [b for b in blocks if b[0] != head_label]

    _, head_pos, head_quat = head_block
    R = _quat_rotmat(head_quat)          # (n,3,3)

    # Axe X MOYEN de la tête (normalisé) — robuste aux sauts de quaternion
    mean_x_axis = np.mean(R[:, :, 0], axis=0)
    mean_x_axis /= float(np.linalg.norm(mean_x_axis)) + 1e-12

    projs = {}
    for label, pos, _ in other_blocks:
        n = min(len(head_pos), len(pos))
        # Position MOYENNE relative à la tête sur toute la session
        mean_rel = np.mean(pos[:n] - head_pos[:n], axis=0)   # (3,)
        projs[label] = float(np.dot(mean_rel, mean_x_axis))

    labels_other = list(projs.keys())
    p0, p1 = projs[labels_other[0]], projs[labels_other[1]]
    separation = abs(p0 - p1)

    # right → projection plus grande (positive = à droite de la tête)
    if p0 >= p1:
        right_label, left_label = labels_other[0], labels_other[1]
    else:
        right_label, left_label = labels_other[1], labels_other[0]

    # Confiance basée sur séparation normalisée
    all_pos = np.concatenate([b[1][:, 0] for b in other_blocks])
    pos_range = float(np.ptp(all_pos))
    conf = float(np.clip(separation / (pos_range + 1e-6), 0.0, 1.0))

    return TestResult(
        name="lateral_mean_position",
        head_vote=head_label,
        left_vote=left_label,
        right_vote=right_label,
        confidence=conf,
        evidence={
            "mean_proj_m":  {lb: round(v, 4) for lb, v in projs.items()},
            "separation_m": round(separation, 4),
            "conf":         round(conf, 3),
        },
    )


# ── Algorithme de consensus ───────────────────────────────────────────────────

def _consensus(tests: list[TestResult]) -> tuple[dict, int, bool]:
    """
    Consensus head / left / right.

    Stratégie :
    - HEAD     : vote majoritaire de tous les tests (tests 1-4).
    - LEFT/RIGHT : UNIQUEMENT le test de position moyenne (test 4).
      Les tests 1-3 (hauteur, centralité, mobilité) n'ont aucune information
      physique sur gauche/droite — leurs votes pour left/right sont arbitraires
      et ne doivent PAS participer au consensus gauche/droite.

    Returns:
        predicted : {role: label_csv}
        agree_count : nombre de tests qui s'accordent sur le head
        certain : True si critère hauteur ≥ ZSCORE_CERTAIN et ≥ AGREE_MIN tests s'accordent
    """
    from collections import Counter
    head_votes = Counter(t.head_vote for t in tests)

    head_pred = head_votes.most_common(1)[0][0]

    # Left/right : uniquement depuis le test de position moyenne
    lateral = next(
        (t for t in tests if t.name == "lateral_mean_position"),
        None,
    )
    if lateral is not None:
        right_pred = lateral.right_vote
        left_pred  = lateral.left_vote
    else:
        # Fallback si le test latéral est absent (ne devrait pas arriver)
        right_votes = Counter(t.right_vote for t in tests)
        left_votes  = Counter(t.left_vote  for t in tests)
        right_pred  = right_votes.most_common(1)[0][0]
        left_pred   = left_votes.most_common(1)[0][0]

    # Nb de tests qui s'accordent sur head
    agree_count = int(head_votes[head_pred])

    # Certitude si hauteur OK ET AGREE_MIN tests accordés
    height_test   = next((t for t in tests if t.name == "height_Y"), None)
    zscore        = height_test.evidence.get("zscore", 0.0) if height_test else 0.0
    height_ok     = zscore >= ZSCORE_CERTAIN
    # Niveau "probable" : z-score moindre mais 4/4 tests s'accordent
    height_likely = zscore >= ZSCORE_LIKELY and agree_count >= AGREE_ALL
    certain       = (height_ok and agree_count >= AGREE_MIN) or height_likely

    # S'assurer que les 3 rôles sont distincts (fallback si vote incohérent)
    assigned = set()
    predicted: dict[str, str] = {}
    for role, pred in [("head", head_pred), ("right", right_pred), ("left", left_pred)]:
        if pred not in assigned:
            predicted[role] = pred
            assigned.add(pred)
        else:
            # Collision : prendre le tracker non encore assigné
            remaining = [lb for lb in TRACKERS if lb not in assigned]
            if remaining:
                predicted[role] = remaining[0]
                assigned.add(remaining[0])

    return predicted, agree_count, certain


# ── Correction CSV ────────────────────────────────────────────────────────────

def _swap_columns(df: pd.DataFrame, predicted: dict) -> pd.DataFrame:
    """
    Renomme les colonnes tracker_<src>_* → tracker_<role>_* selon le
    mapping prédit {role: src_label}.
    """
    rename: dict[str, str] = {}
    for role, src in predicted.items():
        if role == src:
            continue
        for ax in list(AXES) + list(QUATS):
            rename[f"tracker_{src}_{ax}"] = f"__TMP_tracker_{role}_{ax}"

    if not rename:
        return df
    df = df.rename(columns=rename)
    df = df.rename(columns={v: v.replace("__TMP_", "") for v in rename.values()})
    return df


# ── Point d'entrée principal ──────────────────────────────────────────────────

def fix_tracker_labels(
    session_path: Path,
    dry_run: bool = False,
    force: bool = False,
) -> TrackerLabelReport:
    """
    Identifie et corrige (si nécessaire) les labels head/left/right des trackers.

    Retourne un TrackerLabelReport avec status:
      "ok"        — labels déjà corrects, aucun changement
      "corrected" — labels corrigés dans le CSV et metadata.json
      "uncertain" — les tests ne s'accordent pas, aucune modification
      "error"     — problème de lecture/données insuffisantes
    """
    if not _PANDAS:
        return TrackerLabelReport(
            session=str(session_path.name),
            status="error", reason="pandas non disponible"
        )

    name      = session_path.name
    meta_path = session_path / "metadata.json"
    csv_path  = session_path / "tracker_positions.csv"

    if not meta_path.exists():
        return TrackerLabelReport(session=name, status="error",
                                   reason="metadata.json absent")
    if not csv_path.exists():
        return TrackerLabelReport(session=name, status="error",
                                   reason="tracker_positions.csv absent")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return TrackerLabelReport(session=name, status="error",
                                   reason=f"metadata.json illisible: {e}")

    if not force and meta.get(MARKER_KEY):
        return TrackerLabelReport(session=name, status="skipped",
                                   reason="labels déjà vérifiés (MARKER_KEY présent)")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return TrackerLabelReport(session=name, status="error",
                                   reason=f"CSV illisible: {e}")

    blocks = _load_blocks(df)
    if blocks is None:
        return TrackerLabelReport(session=name, status="error",
                                   reason="colonnes tracker manquantes dans le CSV")

    # ── Exécuter les 4 tests ──────────────────────────────────────────────────
    t1 = _test_height(blocks)
    t2 = _test_centrality(blocks)
    t3 = _test_mobility(blocks)
    t4 = _test_lateral(blocks, t1.head_vote)   # Test 4 utilise le head du Test 1

    tests = [t1, t2, t3, t4]

    # ── Consensus ────────────────────────────────────────────────────────────
    predicted, agree_count, certain = _consensus(tests)

    # Assignement actuel
    current = {"head": "head", "left": "left", "right": "right"}
    needs_correction = any(predicted[r] != current[r] for r in TRACKERS)

    report = TrackerLabelReport(
        session=name,
        status="",
        tests=tests,
        agreement_count=agree_count,
        predicted=predicted,
        old_assignment=current,
    )

    if not certain:
        report.status = "uncertain"
        reasons = []
        height_t = t1
        if not height_t.evidence.get("certain"):
            reasons.append(
                f"z-score hauteur insuffisant ({height_t.evidence.get('zscore', 0):.1f} < {ZSCORE_CERTAIN})"
            )
        if agree_count < AGREE_MIN:
            reasons.append(f"seulement {agree_count}/{len(tests)} tests s'accordent")
        report.reason = "; ".join(reasons) or "incertitude indéterminée"
        if not dry_run:
            meta[MARKER_KEY] = False
            meta["tracker_labels_agreement"] = agree_count
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        return report

    if not needs_correction:
        report.status = "ok"
        report.reason = f"labels corrects (accord {agree_count}/{len(tests)} tests)"
        if not dry_run:
            meta[MARKER_KEY] = True
            meta["tracker_labels_agreement"] = agree_count
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        return report

    # ── Appliquer la correction ──────────────────────────────────────────────
    if dry_run:
        report.status = "would_correct"
        report.reason = (
            f"Correction nécessaire ({agree_count}/{len(tests)} tests, "
            f"z={t1.evidence.get('zscore', 0):.1f}σ)"
        )
        report.dry_run = True
        return report

    df_fixed = _swap_columns(df, predicted)
    df_fixed.to_csv(csv_path, index=False)

    meta[MARKER_KEY]                   = True
    meta["tracker_labels_agreement"]   = agree_count
    meta["tracker_labels_corrected"]   = True
    meta["tracker_labels_old"]         = current
    meta["tracker_labels_new"]         = predicted
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    report.status    = "corrected"
    report.corrected = True
    report.reason    = (
        f"Labels corrigés ({agree_count}/{len(tests)} tests s'accordent, "
        f"z={t1.evidence.get('zscore', 0):.1f}σ)"
    )
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_report(r: TrackerLabelReport) -> None:
    icons = {"ok": "✓", "corrected": "↺", "uncertain": "⚠", "error": "✗",
             "skipped": "–", "would_correct": "~"}
    icon = icons.get(r.status, "?")
    print(f"\n{icon} [{r.status.upper()}] {r.session}")
    if r.reason:
        print(f"  Raison : {r.reason}")
    if r.predicted:
        print(f"  Prédit : head={r.predicted.get('head')} "
              f"left={r.predicted.get('left')} right={r.predicted.get('right')}")
    for t in r.tests:
        ok = "✓" if t.head_vote == r.predicted.get("head") else "✗"
        print(f"  [{ok}] {t.name:22s}  head={t.head_vote:8s}"
              f"  conf={t.confidence:.2f}  {t.evidence}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vérifie et corrige le labeling head/left/right des trackers."
    )
    parser.add_argument("sessions", nargs="+", type=Path,
                        help="Répertoires de session(s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse uniquement, sans modifier")
    parser.add_argument("--force", action="store_true",
                        help="Ré-analyse même si MARKER_KEY présent")
    parser.add_argument("--json", action="store_true",
                        help="Sortie JSON")
    args = parser.parse_args()

    results = []
    for s in args.sessions:
        s = s.resolve()
        if not s.is_dir():
            print(f"[SKIP] {s} n'est pas un répertoire")
            continue
        r = fix_tracker_labels(s, dry_run=args.dry_run, force=args.force)
        results.append(r)
        if not args.json:
            _print_report(r)

    if args.json:
        import dataclasses
        print(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False,
                         default=str))

    # Résumé
    if not args.json:
        n_ok  = sum(1 for r in results if r.status in ("ok", "skipped"))
        n_fix = sum(1 for r in results if r.status == "corrected")
        n_unc = sum(1 for r in results if r.status == "uncertain")
        n_err = sum(1 for r in results if r.status == "error")
        print(f"\n{'─'*60}")
        print(f"Total {len(results)} session(s) : "
              f"✓ {n_ok} ok  ↺ {n_fix} corrigé(s)  "
              f"⚠ {n_unc} incertain(s)  ✗ {n_err} erreur(s)")


if __name__ == "__main__":
    main()
