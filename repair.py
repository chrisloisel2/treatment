#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair.py — Marque les sessions parfaites et répare les sessions défectueuses.

Pour chaque session :
  - Score ≥ 70 et toutes les portes passées → marque "perfect" dans metadata.json
  - Porte camera_coverage ou camera_continuity échouée → tente fix_camera_offset
  - Autre échec (structure, quaternions, tracker) → marquée "unrecoverable"

Usage :
    python repair.py <session_path>
    python repair.py <root_dir> --batch
    python repair.py <root_dir> --batch --dry-run
    python repair.py <root_dir> --batch --json-out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ── Imports internes ──────────────────────────────────────────────────────────
# On importe check.py et fix_camera_offset.py directement
import check as chk
import fix_camera_offset as fix_cam

MARKER_PERFECT       = "repair_perfect"
MARKER_REPAIRED      = "repair_camera_fixed"
MARKER_UNRECOVERABLE = "repair_unrecoverable"

# Portes qu'un fix caméra peut résoudre
CAMERA_FIXABLE_GATES = {"camera_coverage", "camera_continuity"}


# ══════════════════════════════════════════════════════════════════════════════
# Core
# ══════════════════════════════════════════════════════════════════════════════

def _read_meta(session_path: Path) -> Optional[dict]:
    meta_path = session_path / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_meta(session_path: Path, meta: dict) -> None:
    meta_path = session_path / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def process_session(
    session_path: Path,
    model,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Analyse et répare (ou marque) une session.

    Retourne un dict rapport avec les clés :
      session, action, score, reason, fix_report
    """
    name = session_path.name

    meta = _read_meta(session_path)
    if meta is None:
        return {"session": name, "action": "skipped", "reason": "metadata.json absent"}

    # Déjà traité ?
    if not force:
        if meta.get(MARKER_PERFECT):
            return {"session": name, "action": "skipped", "reason": "déjà marquée parfaite"}
        if meta.get(MARKER_REPAIRED):
            return {"session": name, "action": "skipped", "reason": "déjà réparée (camera fix)"}
        if meta.get(MARKER_UNRECOVERABLE):
            return {"session": name, "action": "skipped", "reason": "déjà marquée non-récupérable"}

    # ── 1ère vérification ─────────────────────────────────────────────────────
    report = chk.check_session(session_path, model)
    score  = report.score
    failed = [g for g in report.gates if not g.passed]

    # Cas 1 : session parfaite
    if not failed and score >= 70.0:
        if not dry_run:
            meta[MARKER_PERFECT] = True
            meta["repair_score"] = score
            _write_meta(session_path, meta)
        return {
            "session": name,
            "action":  "marked_perfect",
            "score":   score,
            "reason":  f"score={score:.0f}% toutes portes OK",
        }

    # Cas 2 : toutes les portes échouées sont des portes caméra → tentative de réparation
    failed_names = {g.name for g in failed}
    if failed_names and failed_names.issubset(CAMERA_FIXABLE_GATES):
        fix_report = fix_cam.fix_session(session_path, dry_run=dry_run, force=force)
        fix_status = fix_report.get("status", "error")

        if fix_status in ("corrected", "ok", "dry-run"):
            # Revérifier après fix (sauf en dry-run où rien n'a changé sur disque)
            if not dry_run:
                report2 = chk.check_session(session_path, model)
                score2  = report2.score
                failed2 = [g for g in report2.gates if not g.passed]

                if not failed2 and score2 >= 70.0:
                    meta = _read_meta(session_path) or meta
                    meta[MARKER_REPAIRED] = True
                    meta["repair_score_before"] = score
                    meta["repair_score_after"]  = score2
                    _write_meta(session_path, meta)
                    return {
                        "session":    name,
                        "action":     "repaired",
                        "score_before": score,
                        "score_after":  score2,
                        "reason":     "fix caméra réussi, session maintenant parfaite",
                        "fix_report": fix_report,
                    }
                else:
                    # Fix appliqué mais session encore insuffisante
                    meta = _read_meta(session_path) or meta
                    meta[MARKER_UNRECOVERABLE] = True
                    meta["repair_score"] = score2
                    meta["repair_failure_reason"] = (
                        f"après fix caméra : score={score2:.0f}%  "
                        f"portes échouées={[g.name for g in failed2]}"
                    )
                    _write_meta(session_path, meta)
                    return {
                        "session":    name,
                        "action":     "unrecoverable",
                        "score_before": score,
                        "score_after":  score2,
                        "reason":     f"fix caméra appliqué mais score={score2:.0f}% encore insuffisant",
                        "gates_failed": [g.name for g in failed2],
                        "fix_report": fix_report,
                    }
            else:
                # dry-run : on simule
                return {
                    "session":    name,
                    "action":     "would_repair",
                    "score":      score,
                    "reason":     f"fix caméra applicable (gates: {sorted(failed_names)})",
                    "fix_report": fix_report,
                }
        else:
            # Fix a échoué (erreur)
            if not dry_run:
                meta[MARKER_UNRECOVERABLE] = True
                meta["repair_failure_reason"] = f"fix_camera échoué: {fix_report.get('reason','')}"
                _write_meta(session_path, meta)
            return {
                "session":  name,
                "action":   "unrecoverable",
                "score":    score,
                "reason":   f"fix caméra échoué: {fix_report.get('reason', '')}",
                "fix_report": fix_report,
            }

    # Cas 3 : portes bloquantes non-caméra (structure, quaternions, tracker…)
    if not dry_run:
        meta[MARKER_UNRECOVERABLE] = True
        meta["repair_score"] = score
        meta["repair_failure_reason"] = (
            f"portes non-récupérables: {sorted(failed_names)}"
        )
        _write_meta(session_path, meta)
    return {
        "session":     name,
        "action":      "unrecoverable",
        "score":       score,
        "reason":      f"portes non-récupérables: {sorted(failed_names)}",
        "gates_failed": sorted(failed_names),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Affichage
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


def print_result(r: dict) -> None:
    action  = r.get("action", "")
    session = r.get("session", "")

    if action == "skipped":
        print(f"  [SKIP]  {session} — {r.get('reason','')}")

    elif action == "marked_perfect":
        score = r.get("score", 0)
        print(f"  {_c('[PERFECT]', _GREEN + _BOLD)}  {session}  score={score:.0f}%")

    elif action == "repaired":
        sb = r.get("score_before", 0)
        sa = r.get("score_after", 0)
        print(f"  {_c('[REPAIRED]', _YELLOW + _BOLD)}  {session}  "
              f"{sb:.0f}% → {sa:.0f}%  — {r.get('reason','')}")

    elif action == "would_repair":
        print(f"  {_c('[DRY-RUN REPAIR]', _YELLOW)}  {session}  — {r.get('reason','')}")

    elif action == "unrecoverable":
        score = r.get("score", 0)
        gates = r.get("gates_failed", [])
        g_str = f"  portes={gates}" if gates else ""
        print(f"  {_c('[UNRECOVERABLE]', _RED)}  {session}  score={score:.0f}%{g_str} — {r.get('reason','')}")

    else:
        print(f"  [?] {session}  {r}")


def print_summary(results: list[dict]) -> None:
    n_perfect       = sum(1 for r in results if r["action"] == "marked_perfect")
    n_repaired      = sum(1 for r in results if r["action"] == "repaired")
    n_would_repair  = sum(1 for r in results if r["action"] == "would_repair")
    n_unrecoverable = sum(1 for r in results if r["action"] == "unrecoverable")
    n_skipped       = sum(1 for r in results if r["action"] == "skipped")

    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ  {len(results)} sessions")
    print(f"  {_c(f'Parfaites        : {n_perfect}', _GREEN)}")
    print(f"  {_c(f'Réparées         : {n_repaired}', _YELLOW)}")
    if n_would_repair:
        print(f"  {_c(f'À réparer (dry)  : {n_would_repair}', _YELLOW)}")
    print(f"  {_c(f'Non-récupérables : {n_unrecoverable}', _RED)}")
    print(f"  Ignorées         : {n_skipped}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Marque les sessions parfaites et répare les sessions défectueuses."
    )
    parser.add_argument("path", help="Session unique ou dossier racine")
    parser.add_argument("--batch",    action="store_true",
                        help="Traiter toutes les sessions du dossier")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Afficher les actions sans modifier les fichiers")
    parser.add_argument("--force",    action="store_true",
                        help="Réanalyser même les sessions déjà marquées")
    parser.add_argument("--json-out", metavar="FILE",
                        help="Sauvegarder les résultats en JSON")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"Chemin introuvable : {root}")
        sys.exit(1)

    # Charger le modèle une seule fois
    model = chk.load_model()
    if model is None:
        print("[repair] ⚠  Aucun modèle trouvé — scores heuristiques uniquement.\n"
              "           Lancez d'abord : python check.py train\n")

    if args.batch or (root.is_dir() and not (root / "metadata.json").exists()):
        sessions = sorted(
            p.parent for p in root.rglob("metadata.json")
            if (p.parent / "videos").exists()
        )
        if not sessions:
            print(f"Aucune session trouvée dans {root}")
            sys.exit(1)

        prefix = "[DRY-RUN] " if args.dry_run else ""
        print(f"{prefix}Traitement de {len(sessions)} session(s)...\n")

        results = []
        for sess in sessions:
            r = process_session(sess, model, dry_run=args.dry_run, force=args.force)
            print_result(r)
            results.append(r)

        print_summary(results)

    else:
        r = process_session(root, model, dry_run=args.dry_run, force=args.force)
        print_result(r)
        results = [r]

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[repair] Rapport JSON → {args.json_out}")

    if args.dry_run:
        print("Mode dry-run : aucun fichier modifié.")


if __name__ == "__main__":
    main()
