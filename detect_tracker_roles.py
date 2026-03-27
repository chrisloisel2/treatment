#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_tracker_roles.py — Détection des rôles trackers (head / left / right).

Analyse le fichier tracker_positions.csv d'une ou plusieurs sessions pour
déterminer quel tracker physique est la tête (en hauteur), lequel est à
gauche, et lequel est à droite.

Retourne un score de confiance entre 0 (problème détecté) et 1 (tout est bon).

Usage standalone :
    # Une session
    python detect_tracker_roles.py /path/to/session/

    # Toutes les sessions sous un dossier
    python detect_tracker_roles.py /path/to/sessions/ --all

    # Sortie JSON brute
    python detect_tracker_roles.py /path/to/session/ --json

    # Seuil de confiance personnalisé
    python detect_tracker_roles.py /path/to/session/ --min-confidence 0.75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ── Ajoute la racine du projet au chemin Python ──────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data_prep import (
    find_tracker_blocks,
    infer_head,
    infer_left_right,
    _build_mapping,
    _global_confidence,
    _is_already_correct,
)

import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Seuils
# ══════════════════════════════════════════════════════════════════════════════

MIN_CONFIDENCE_DEFAULT = 0.65   # en dessous → score 0.0 (problème)
MIN_ROWS               = 30     # lignes minimales dans tracker_positions.csv


# ══════════════════════════════════════════════════════════════════════════════
# Logique principale
# ══════════════════════════════════════════════════════════════════════════════

def analyze_session(session_path: Path, min_confidence: float = MIN_CONFIDENCE_DEFAULT) -> dict:
    """
    Analyse les rôles trackers d'une session.

    Retourne un dict avec :
      - score          : float [0, 1] — 1 = tout OK, 0 = problème
      - confidence     : float [0, 1] — confiance brute de l'inférence géométrique
      - head_col       : str — colonne préfixe du tracker identifié comme tête
      - left_col       : str — colonne préfixe du tracker gauche
      - right_col      : str — colonne préfixe du tracker droit
      - labels_match   : bool — True si les labels du CSV correspondent à l'inférence
      - swaps          : list — paires (label_actuel, rôle_inféré) en cas de désaccord
      - world_up_axis  : int — axe vertical du monde (0=X, 1=Y, 2=Z)
      - world_up_sign  : int — signe de l'axe vertical (+1 ou -1)
      - error          : str|None — message d'erreur si échec
    """
    result = {
        "session":       str(session_path),
        "score":         0.0,
        "confidence":    0.0,
        "head_col":      None,
        "left_col":      None,
        "right_col":     None,
        "labels_match":  False,
        "swaps":         [],
        "world_up_axis": -1,
        "world_up_sign": 0,
        "error":         None,
    }

    # ── Localiser le CSV ──────────────────────────────────────────────────────
    session_path = Path(session_path)
    if session_path.is_dir():
        csv_path = session_path / "tracker_positions.csv"
    elif session_path.suffix == ".csv":
        csv_path = session_path
    else:
        csv_path = session_path / "tracker_positions.csv"

    if not csv_path.exists():
        result["error"] = f"Fichier introuvable : {csv_path}"
        return result

    # ── Lecture du CSV ────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result["error"] = f"Impossible de lire {csv_path}: {e}"
        return result

    if len(df) < MIN_ROWS:
        result["error"] = f"Trop peu de lignes ({len(df)} < {MIN_ROWS}) pour une inférence fiable"
        return result

    # ── Inférence des rôles ───────────────────────────────────────────────────
    try:
        blocks    = find_tracker_blocks(df)
        head_info = infer_head(blocks)
        lr_info   = infer_left_right(blocks, head_info)
        mapping   = _build_mapping(head_info, lr_info)
        conf      = _global_confidence(head_info, lr_info)
        labels_ok = _is_already_correct(blocks, mapping)
    except Exception as e:
        result["error"] = f"Échec de l'inférence : {e}"
        return result

    # ── Mapping gid → colonne préfixe ─────────────────────────────────────────
    # Chaque bloc contient col_names : ["tracker_head_x", "tracker_head_y", ...]
    # On extrait le préfixe (ex: "tracker_head") pour chaque rôle.
    gid_to_prefix = {}
    for b in blocks:
        if b.col_names:
            # ex: "tracker_head_x" → "tracker_head"
            prefix = "_".join(b.col_names[0].split("_")[:-1])
        else:
            prefix = b.gid
        gid_to_prefix[b.gid] = prefix

    # Rôle → gid (inversion du mapping gid→rôle)
    role_to_gid = {v: k for k, v in mapping.items()}

    swaps = [
        (b.role_hint, mapping[b.gid])
        for b in blocks
        if b.role_hint is not None and mapping[b.gid] != b.role_hint
    ]

    result["confidence"]    = round(conf, 4)
    result["labels_match"]  = labels_ok
    result["swaps"]         = swaps
    result["world_up_axis"] = head_info.up_axis_index
    result["world_up_sign"] = head_info.up_axis_sign
    result["head_col"]      = gid_to_prefix.get(role_to_gid.get("head", ""), "?")
    result["left_col"]      = gid_to_prefix.get(role_to_gid.get("left", ""), "?")
    result["right_col"]     = gid_to_prefix.get(role_to_gid.get("right", ""), "?")

    # ── Score final ───────────────────────────────────────────────────────────
    # score = confidence si confiance suffisante ET labels cohérents (ou sans hint)
    # score = 0 si confiance trop faible ou swap détecté
    if conf < min_confidence:
        # Confiance géométrique trop faible → on ne peut pas se fier à l'inférence
        result["score"] = round(conf * 0.5, 4)   # score dégradé, pas zéro brutal
    elif swaps:
        # Labels dans le CSV ne correspondent pas à l'inférence → anomalie
        result["score"] = round(conf * 0.3, 4)   # fort malus
    else:
        # Tout concorde
        result["score"] = round(conf, 4)

    return result


def analyze_all_sessions(root: Path, min_confidence: float = MIN_CONFIDENCE_DEFAULT) -> list:
    """Analyse toutes les sessions (sous-dossiers contenant tracker_positions.csv)."""
    results = []
    for csv_path in sorted(root.rglob("tracker_positions.csv")):
        r = analyze_session(csv_path.parent, min_confidence)
        results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Affichage console
# ══════════════════════════════════════════════════════════════════════════════

_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}


def _print_result(r: dict, verbose: bool = True) -> None:
    score    = r["score"]
    conf     = r["confidence"]
    session  = Path(r["session"]).name

    if r["error"]:
        status = "ERROR"
        bar    = "──"
    elif score >= 0.80:
        status = "OK"
        bar    = "██"
    elif score >= 0.60:
        status = "WARN"
        bar    = "▓▓"
    else:
        status = "FAIL"
        bar    = "░░"

    print(f"[{status:4s}] {bar} {session}")
    if r["error"]:
        print(f"       Erreur : {r['error']}")
        return

    axis_name = _AXIS_NAMES.get(r["world_up_axis"], "?")
    sign_str  = "+" if r["world_up_sign"] > 0 else "-"

    print(f"       score={score:.3f}  conf={conf:.3f}  axe_vertical={sign_str}{axis_name}")
    print(f"       head  → {r['head_col']}")
    print(f"       left  → {r['left_col']}")
    print(f"       right → {r['right_col']}")

    if r["swaps"]:
        print(f"       SWAPS détectés :")
        for (cur, inferred) in r["swaps"]:
            print(f"         {cur} → {inferred}")
    elif r["labels_match"]:
        print(f"       Labels CSV cohérents avec l'inférence.")
    else:
        print(f"       (labels CSV absents ou non vérifiables)")


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Détecte les rôles trackers (head / left / right) dans une session."
    )
    parser.add_argument(
        "path",
        help="Chemin vers une session (dossier) ou un dossier contenant plusieurs sessions (avec --all).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Cherche récursivement tous les tracker_positions.csv sous PATH.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Sortie JSON brute (une ligne par session).",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=MIN_CONFIDENCE_DEFAULT,
        metavar="SEUIL",
        help=f"Seuil de confiance minimum (défaut: {MIN_CONFIDENCE_DEFAULT}).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Affichage minimal (score seulement).",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"[ERREUR] Chemin introuvable : {root}", file=sys.stderr)
        return 2

    if args.all:
        results = analyze_all_sessions(root, args.min_confidence)
    else:
        results = [analyze_session(root, args.min_confidence)]

    if not results:
        print("[ERREUR] Aucune session trouvée.", file=sys.stderr)
        return 2

    if args.json:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))
    else:
        for r in results:
            _print_result(r, verbose=not args.quiet)
            print()

    # Code de sortie : 0 si tout OK, 1 si au moins une session en échec
    all_ok = all(
        r["error"] is None and r["score"] >= args.min_confidence
        for r in results
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
