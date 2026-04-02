#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SyncML Studio — Point d'entrée unique.

Usage :
  python main.py server  [--host HOST] [--port PORT] [--reload]
  python main.py gui
  python main.py run     [--bronze-dir DIR] [--all | --session NAME] [--write] [--delete-after-store]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Racine du projet dans sys.path pour que utils/ et pipeline/ soient toujours trouvables
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def cmd_server(args: argparse.Namespace) -> None:
    import uvicorn
    # Ajoute server/ dans sys.path pour que "server:app" soit résolvable par uvicorn
    # (nécessaire en mode --reload où uvicorn réimporte le module par nom)
    server_dir = str(_ROOT / "server")
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_gui(args: argparse.Namespace) -> None:
    from server.app_gui import main as gui_main
    gui_main()


def cmd_run(args: argparse.Namespace) -> None:
    # Importer la pipeline depuis son emplacement canonique
    from pipeline.pipeline import run as pipeline_run  # type: ignore
    pipeline_run(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="SyncML Studio — lanceur unique",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── server ────────────────────────────────────────────────────────────────
    sp = sub.add_parser("server", help="Lance le serveur web FastAPI")
    sp.add_argument("--host",   default="127.0.0.1")
    sp.add_argument("--port",   type=int, default=8000)
    sp.add_argument("--reload", action="store_true", help="Rechargement auto (dev)")

    # ── gui ───────────────────────────────────────────────────────────────────
    sub.add_parser("gui", help="Lance l'interface graphique PyQt6")

    # ── run (pipeline CLI) ────────────────────────────────────────────────────
    rp = sub.add_parser("run", help="Lance la pipeline en ligne de commande")
    rp.add_argument("--bronze-dir",          default="/mnt/inbox")
    rp.add_argument("--silver-dir",          default=str(Path.home() / "silver"))
    rp.add_argument("--session",             default=None,  help="Traiter une session spécifique")
    rp.add_argument("--all",                 action="store_true", help="Traiter toutes les sessions non traitées")
    rp.add_argument("--write",               action="store_true", help="Copier vers silver après validation")
    rp.add_argument("--delete-after-store",  action="store_true", help="Supprimer de bronze après copie (requiert --write)")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "server":
        cmd_server(args)
    elif args.command == "gui":
        cmd_gui(args)
    elif args.command == "run":
        cmd_run(args)
