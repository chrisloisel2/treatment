#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/repair.py — Orchestrateur complet de réparation des sessions.

Remplace repair.py (racine). Architecture en 4 étapes :

  1. CHECK    : vérification complète via check.check_session()
  2. DIAGNOSE : conversion du rapport en liste de problèmes (fix/diagnosis.py)
  3. FIX      : application des fixes dans l'ordre de priorité
  4. RECHECK  : re-vérification pour valider l'amélioration

Le cycle FIX→RECHECK est répété jusqu'à :
  - Plus aucun problème récupérable
  - Score ≥ PERFECT_THRESHOLD
  - MAX_ITERATIONS atteint

Principe clé — DOUTER DE TOUT :
  Après chaque fix, on re-vérifie intégralement.
  Si le score a diminué après un fix → le fix est annulé (backup/restore).
  Si le score n'augmente pas après plusieurs fixes → session non récupérable.

Marqueurs dans metadata.json :
  repair_perfect        : True si session parfaite (score ≥ PERFECT_THRESHOLD)
  repair_fixed          : True si réparée avec succès
  repair_unrecoverable  : True si non récupérable
  repair_score_before   : score avant réparation
  repair_score_after    : score après réparation

Usage :
    python -m fix.repair <session_path>
    python -m fix.repair <root_dir> --batch
    python -m fix.repair <root_dir> --batch --dry-run
    python -m fix.repair <root_dir> --batch --json-out results.json

    # Mode diagnose seulement (aucune modification)
    python -m fix.repair <session_path> --diagnose
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check as chk
from fix.problems import DiagnosedProblem, ProblemCode
from fix.diagnosis import diagnose_session, format_diagnosis


# ── Configuration ─────────────────────────────────────────────────────────────

PERFECT_THRESHOLD       = 70.0   # score ≥ 70% → session parfaite
MAX_ITERATIONS          = 4      # max cycles fix→recheck par session

MARKER_PERFECT          = "repair_perfect"
MARKER_FIXED            = "repair_fixed"
MARKER_UNRECOVERABLE    = "repair_unrecoverable"

# Mappage ProblemCode → fonction fix
FIX_DISPATCH: dict[str, str] = {
    "quaternion_corrupt":   "fix.fix_quaternions.fix_quaternions",
    "tracker_misplaced":    "fix.fix_tracker_placement.fix_tracker_placement",
    "camera_misplaced":     "fix.fix_tracker_placement.fix_tracker_placement",
    "camera_offset":        "fix.fix_camera_offset.fix_camera_offset",
    "clock_drift":          "fix.fix_clock_drift.fix_clock_drift",
    "camera_gaps":          "fix.fix_camera_gaps.fix_camera_gaps",
    "tracker_gaps":         "fix.fix_tracker_gaps.fix_tracker_gaps",
    "sync_lag":             "fix.fix_sync_lag.fix_sync_lag",
    "gripper_sync":         "fix.fix_gripper_video_sync.fix_gripper_closed_offset",
}


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def _read_meta(session_path: Path) -> Optional[dict]:
    p = session_path / "metadata.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_meta(session_path: Path, meta: dict) -> None:
    p = session_path / "metadata.json"
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _call_fix(fix_func_path: str, session_path: Path,
              dry_run: bool, force: bool) -> dict:
    """
    Appelle dynamiquement une fonction de fix par son chemin module.func.

    ex: "fix.fix_camera_offset.fix_camera_offset"
    """
    parts     = fix_func_path.rsplit(".", 1)
    mod_path  = parts[0]
    func_name = parts[1]
    try:
        mod  = importlib.import_module(mod_path)
        func = getattr(mod, func_name)
        return func(session_path, dry_run=dry_run, force=force)
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def _backup_session(session_path: Path) -> Path:
    """Crée un backup complet de la session dans un dossier temporaire."""
    tmp = Path(tempfile.mkdtemp(prefix="repair_backup_"))
    shutil.copytree(str(session_path), str(tmp / session_path.name),
                    ignore=shutil.ignore_patterns("*.mp4"))
    return tmp / session_path.name


def _restore_session(backup_path: Path, session_path: Path) -> None:
    """Restaure la session depuis le backup (hors MP4)."""
    for src in backup_path.rglob("*"):
        if src.is_file():
            rel = src.relative_to(backup_path)
            dst = session_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))


# ══════════════════════════════════════════════════════════════════════════════
# Traitement d'une session
# ══════════════════════════════════════════════════════════════════════════════

def process_session(
    session_path: Path,
    model,
    dry_run:  bool = False,
    force:    bool = False,
    diagnose_only: bool = False,
    verbose:  bool = False,
) -> dict:
    """
    Analyse, diagnostique et répare une session.

    Returns:
        dict rapport avec :
          - session       : nom de la session
          - action        : marked_perfect | fixed | unrecoverable | skipped | diagnosed
          - score_before  : score initial
          - score_after   : score final
          - problems      : liste des problèmes diagnostiqués
          - fixes_applied : liste des fixes appliqués avec leur résultat
          - iterations    : nombre de cycles fix→recheck
    """
    name = session_path.name

    # ── Vérification préliminaire ─────────────────────────────────────────────
    meta = _read_meta(session_path)
    if meta is None:
        return {"session": name, "action": "skipped",
                "reason": "metadata.json absent ou illisible"}

    # Déjà traité ?
    if not force:
        if meta.get(MARKER_PERFECT):
            return {"session": name, "action": "skipped", "reason": "déjà parfaite"}
        if meta.get(MARKER_FIXED):
            return {"session": name, "action": "skipped", "reason": "déjà réparée"}
        if meta.get(MARKER_UNRECOVERABLE):
            return {"session": name, "action": "skipped", "reason": "déjà non-récupérable"}

    # ── Check initial ──────────────────────────────────────────────────────────
    report = chk.check_session(session_path, model)
    score_initial = report.score
    failed_gates  = [g for g in report.gates if not g.passed]

    if verbose:
        print(f"\n  ── {name}")
        print(f"     Score initial : {score_initial:.0f}%")
        if failed_gates:
            for g in failed_gates:
                print(f"     ✗ {g.name}: {g.message}")

    # Session déjà parfaite ?
    if not failed_gates and score_initial >= PERFECT_THRESHOLD:
        if not dry_run:
            meta[MARKER_PERFECT] = True
            meta["repair_score"] = score_initial
            _write_meta(session_path, meta)
        return {
            "session":     name,
            "action":      "marked_perfect",
            "score_before": score_initial,
            "score_after":  score_initial,
            "problems":    [],
        }

    # ── Diagnostic ────────────────────────────────────────────────────────────
    problems = diagnose_session(session_path, report, deep=True)

    if verbose:
        print(f"     Problèmes détectés : {len(problems)}")
        for p in problems:
            icon = "↑" if p.recoverable else "✗"
            print(f"       {icon} [{p.code.value}] {p.message}")

    if diagnose_only:
        return {
            "session":     name,
            "action":      "diagnosed",
            "score_before": score_initial,
            "problems":    [{"code": p.code.value, "message": p.message,
                             "recoverable": p.recoverable}
                            for p in problems],
        }

    # Problèmes fatals ?
    fatal = [p for p in problems
             if p.code in (ProblemCode.MISSING_FILES, ProblemCode.METADATA_CORRUPT)]
    if fatal:
        reason = fatal[0].message
        if not dry_run:
            meta[MARKER_UNRECOVERABLE] = True
            meta["repair_failure_reason"] = reason
            _write_meta(session_path, meta)
        return {
            "session":      name,
            "action":       "unrecoverable",
            "score_before": score_initial,
            "reason":       reason,
            "problems":     [{"code": p.code.value, "message": p.message}
                             for p in problems],
        }

    # Aucun problème récupérable ?
    recoverable = [p for p in problems if p.recoverable]
    if not recoverable:
        if not dry_run:
            meta[MARKER_UNRECOVERABLE] = True
            meta["repair_score"]       = score_initial
            meta["repair_failure_reason"] = (
                f"Portes échouées non récupérables: "
                f"{[g.name for g in failed_gates]}"
            )
            _write_meta(session_path, meta)
        return {
            "session":      name,
            "action":       "unrecoverable",
            "score_before": score_initial,
            "reason":       "aucun problème récupérable",
            "problems":     [{"code": p.code.value, "message": p.message,
                              "recoverable": p.recoverable}
                             for p in problems],
        }

    # ── Boucle fix→recheck ─────────────────────────────────────────────────────
    current_score   = score_initial
    fixes_applied:  list[dict[str, Any]] = []
    iteration = 0

    for iteration in range(MAX_ITERATIONS):
        if verbose:
            print(f"     Itération {iteration + 1}/{MAX_ITERATIONS} "
                  f"(score={current_score:.0f}%)")

        # Re-diagnostiquer à chaque itération (les problèmes évoluent)
        if iteration > 0:
            report = chk.check_session(session_path, model)
            current_score = report.score
            problems = diagnose_session(session_path, report, deep=True)
            recoverable = [p for p in problems if p.recoverable]

            if not recoverable:
                if verbose:
                    print(f"     → Plus de problèmes récupérables")
                break

        # Trier par priorité et appliquer le premier fix disponible
        to_fix = sorted(recoverable, key=lambda p: p.priority)

        fixed_something = False
        for problem in to_fix:
            func_path = FIX_DISPATCH.get(problem.code.value)
            if func_path is None:
                continue

            # Backup avant le fix (pour pouvoir annuler si la correction dégrade)
            backup_path = None
            if not dry_run:
                backup_path = _backup_session(session_path)

            if verbose:
                print(f"       → Applying [{problem.code.value}] via {func_path}")

            fix_result = _call_fix(func_path, session_path,
                                   dry_run=dry_run, force=False)
            fix_status = fix_result.get("status", "unknown")

            fix_record = {
                "iteration": iteration + 1,
                "problem":   problem.code.value,
                "func":      func_path,
                "status":    fix_status,
                "result":    fix_result,
            }

            if fix_status in ("error", "skipped") and not dry_run:
                # Fix sans effet → supprimer le backup et passer au suivant
                if backup_path:
                    shutil.rmtree(backup_path, ignore_errors=True)
                fixes_applied.append(fix_record)
                continue

            if dry_run:
                fixes_applied.append(fix_record)
                fixed_something = True
                continue

            # Vérifier que le fix a bien amélioré le score
            report_after = chk.check_session(session_path, model)
            score_after  = report_after.score

            if verbose:
                print(f"         Score : {current_score:.0f}% → {score_after:.0f}%")

            # Si le score a diminué → annuler le fix
            if score_after < current_score - 2.0:
                if verbose:
                    print(f"         ⚠ Score dégradé — annulation du fix")
                if backup_path:
                    _restore_session(backup_path, session_path)
                    shutil.rmtree(backup_path, ignore_errors=True)
                fix_record["status"]   = "reverted"
                fix_record["reason"]   = (
                    f"Score dégradé : {current_score:.0f}% → {score_after:.0f}%"
                )
                fixes_applied.append(fix_record)
                continue

            # Fix validé
            if backup_path:
                shutil.rmtree(backup_path, ignore_errors=True)

            fix_record["score_before"] = current_score
            fix_record["score_after"]  = score_after
            fixes_applied.append(fix_record)
            current_score = score_after
            fixed_something = True

            # Si score parfait → on arrête
            if current_score >= PERFECT_THRESHOLD and not report_after.is_blocked():
                break

        if not fixed_something:
            if verbose:
                print(f"     → Aucun fix applicable — arrêt")
            break

        if not dry_run and current_score >= PERFECT_THRESHOLD:
            report_final = chk.check_session(session_path, model)
            if not report_final.is_blocked():
                break

    # ── Résultat final ────────────────────────────────────────────────────────
    if not dry_run:
        report_final = chk.check_session(session_path, model)
        score_final  = report_final.score
        still_failed = [g for g in report_final.gates if not g.passed]
    else:
        score_final  = current_score
        still_failed = []

    meta = _read_meta(session_path) or {}

    if not dry_run and score_final >= PERFECT_THRESHOLD and not still_failed:
        meta[MARKER_FIXED]          = True
        meta["repair_perfect"]      = True
        meta["repair_score_before"] = score_initial
        meta["repair_score_after"]  = score_final
        _write_meta(session_path, meta)
        action = "fixed"
    elif not dry_run and fixes_applied and score_final > score_initial:
        # Améliorée mais pas parfaite
        meta[MARKER_FIXED]          = True
        meta["repair_score_before"] = score_initial
        meta["repair_score_after"]  = score_final
        _write_meta(session_path, meta)
        action = "improved"
    elif not dry_run:
        meta[MARKER_UNRECOVERABLE]    = True
        meta["repair_score"]          = score_final
        meta["repair_failure_reason"] = (
            f"Après {iteration + 1} itérations : score={score_final:.0f}%  "
            f"portes={[g.name for g in still_failed]}"
        )
        _write_meta(session_path, meta)
        action = "unrecoverable"
    else:
        action = "would_fix" if fixes_applied else "unrecoverable"

    return {
        "session":      name,
        "action":       action,
        "score_before": score_initial,
        "score_after":  score_final,
        "iterations":   iteration + 1,
        "problems":     [{"code": p.code.value, "message": p.message,
                          "recoverable": p.recoverable}
                         for p in problems],
        "fixes_applied": fixes_applied,
        "gates_failed":  [g.name for g in still_failed],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Affichage
# ══════════════════════════════════════════════════════════════════════════════

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BLUE   = "\033[94m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _c(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{_RESET}"
    return text


def print_result(r: dict) -> None:
    action  = r.get("action", "")
    session = r.get("session", "")
    sb      = r.get("score_before", 0.0)
    sa      = r.get("score_after",  sb)

    if action == "skipped":
        print(f"  [SKIP ]  {session}  — {r.get('reason', '')}")

    elif action == "marked_perfect":
        print(f"  {_c('[PERFECT]', _GREEN + _BOLD)}  {session}  score={sa:.0f}%")

    elif action == "fixed":
        print(f"  {_c('[FIXED  ]', _GREEN)}  {session}  "
              f"{sb:.0f}% → {sa:.0f}%  "
              f"({r.get('iterations', 1)} iter)")

    elif action == "improved":
        print(f"  {_c('[BETTER ]', _YELLOW)}  {session}  "
              f"{sb:.0f}% → {sa:.0f}%  "
              f"(non-parfaite, {r.get('iterations', 1)} iter)")

    elif action in ("would_fix", "would_repair"):
        print(f"  {_c('[DRY-RUN]', _BLUE)}  {session}  "
              f"({len(r.get('fixes_applied', []))} fix(es) possible(s))")

    elif action == "diagnosed":
        problems = r.get("problems", [])
        n_rec    = sum(1 for p in problems if p.get("recoverable"))
        print(f"  {_c('[DIAGNOS]', _BLUE)}  {session}  "
              f"score={sb:.0f}%  {len(problems)} problèmes ({n_rec} récupérables)")
        for p in problems:
            icon = "✓" if p.get("recoverable") else "✗"
            print(f"      {icon} [{p['code']}] {p['message']}")

    elif action == "unrecoverable":
        gates = r.get("gates_failed", [])
        g_str = f"  gates={gates}" if gates else ""
        print(f"  {_c('[UNREC  ]', _RED)}  {session}  score={sa:.0f}%{g_str}")

    else:
        print(f"  [?]  {session}  {r}")


def print_summary(results: list[dict]) -> None:
    n_perfect     = sum(1 for r in results if r["action"] == "marked_perfect")
    n_fixed       = sum(1 for r in results if r["action"] == "fixed")
    n_improved    = sum(1 for r in results if r["action"] == "improved")
    n_would_fix   = sum(1 for r in results if r["action"] in ("would_fix", "would_repair"))
    n_diagnosed   = sum(1 for r in results if r["action"] == "diagnosed")
    n_unrec       = sum(1 for r in results if r["action"] == "unrecoverable")
    n_skipped     = sum(1 for r in results if r["action"] == "skipped")

    scores_fixed = [r.get("score_after", 0) for r in results
                    if r["action"] in ("fixed", "improved", "marked_perfect")]

    print(f"\n{'='*64}")
    print(f"  RÉSUMÉ  {len(results)} session(s)")
    print(f"  {_c(f'Parfaites          : {n_perfect}', _GREEN + _BOLD)}")
    print(f"  {_c(f'Réparées (parfait)  : {n_fixed}', _GREEN)}")
    if n_improved:
        print(f"  {_c(f'Améliorées (partiel): {n_improved}', _YELLOW)}")
    if n_would_fix:
        print(f"  {_c(f'À réparer (dry-run) : {n_would_fix}', _BLUE)}")
    if n_diagnosed:
        print(f"  {_c(f'Diagnostiquées      : {n_diagnosed}', _BLUE)}")
    print(f"  {_c(f'Non-récupérables    : {n_unrec}', _RED)}")
    print(f"  Ignorées            : {n_skipped}")
    if scores_fixed:
        print(f"\n  Score moyen (réparées) : {np.mean(scores_fixed):.1f}%")
    print(f"{'='*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _collect_sessions(root: Path) -> list[Path]:
    return sorted(
        p.parent for p in root.rglob("metadata.json")
        if (p.parent / "videos").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrateur de réparation des sessions robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Réparer une session unique
  python -m fix.repair /path/to/session

  # Réparer toutes les sessions d'un dossier
  python -m fix.repair /path/to/root --batch

  # Voir les problèmes sans rien modifier
  python -m fix.repair /path/to/root --batch --diagnose

  # Simulation (mesure uniquement)
  python -m fix.repair /path/to/root --batch --dry-run

  # Forcer la re-analyse même des sessions déjà marquées
  python -m fix.repair /path/to/root --batch --force --verbose
        """,
    )
    parser.add_argument("path",          help="Session unique ou dossier racine")
    parser.add_argument("--batch",       action="store_true",
                        help="Traiter toutes les sessions du dossier")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Mesure et simule sans modifier les fichiers")
    parser.add_argument("--diagnose",    action="store_true",
                        help="Diagnostic uniquement (pas de fix)")
    parser.add_argument("--force",       action="store_true",
                        help="Réanalyser même les sessions déjà marquées")
    parser.add_argument("--verbose",     action="store_true",
                        help="Affichage détaillé")
    parser.add_argument("--json-out",    metavar="FILE",
                        help="Sauvegarder les résultats en JSON")
    parser.add_argument("--workers",     type=int, default=1,
                        help="Nombre de processus parallèles (défaut=1)")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"Chemin introuvable : {root}", file=sys.stderr)
        sys.exit(1)

    # Charger le modèle IA une seule fois
    model = chk.load_model()
    if model is None:
        print("[repair] ⚠  Aucun modèle IA trouvé — scores heuristiques uniquement.\n"
              "           Lancez d'abord : python check.py train\n")

    if args.batch or (root.is_dir() and not (root / "metadata.json").exists()):
        sessions = _collect_sessions(root)
        if not sessions:
            print(f"Aucune session trouvée dans {root}", file=sys.stderr)
            sys.exit(1)

        prefix = "[DRY-RUN] " if args.dry_run else ("[DIAGNOSE] " if args.diagnose else "")
        print(f"{prefix}Traitement de {len(sessions)} session(s)...\n")

        results = []
        for sess in sessions:
            r = process_session(
                sess, model,
                dry_run=args.dry_run,
                force=args.force,
                diagnose_only=args.diagnose,
                verbose=args.verbose,
            )
            print_result(r)
            results.append(r)

        print_summary(results)
    else:
        r = process_session(
            root, model,
            dry_run=args.dry_run,
            force=args.force,
            diagnose_only=args.diagnose,
            verbose=True,
        )
        print_result(r)
        results = [r]

    if args.json_out:
        def _serial(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(type(obj))

        Path(args.json_out).write_text(
            json.dumps(results, indent=2, default=_serial, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[repair] Rapport JSON → {args.json_out}")

    if args.dry_run:
        print("Mode dry-run : aucun fichier modifié.")
    if args.diagnose:
        print("Mode diagnose : aucun fix appliqué.")


if __name__ == "__main__":
    main()
