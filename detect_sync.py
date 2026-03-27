#!/usr/bin/env python3
"""
detect_sync.py — Détecte le décalage vidéo/capteurs dans une session.

Usage:
    python detect_sync.py <session_dir>
    python detect_sync.py <sessions_root_dir>   # scanne tous les sous-dossiers

Retour (stdout JSON):
    {
        "session": "...",
        "status": "DECALAGE" | "GOOD",
        "delay_ms": 2311.0,       # délai du premier frame caméra par rapport au trigger
        "threshold_ms": 500,
        "camera_delays_ms": {"head": 2311, "left": 2308, "right": 2311},
        "details": "..."
    }

Exit code: 0 = GOOD, 1 = DECALAGE détecté
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


# ─── Seuil de détection ─────────────────────────────────────────────────────
# GOOD:     caméras démarrent 40–120 ms après le trigger
# DECALAGE: caméras démarrent 2 000–2 500 ms après le trigger
# Threshold conservateur à 500 ms — bien au-dessus du bruit GOOD (120ms max)
# et bien en-dessous du plus petit décalage observé (2113ms)
THRESHOLD_MS = 500


# ─── Helpers ────────────────────────────────────────────────────────────────

def _read_jsonl_first(path: Path) -> Optional[int]:
    """Retourne le premier capture_time (ms) du fichier JSONL. Robuste CRLF."""
    try:
        with open(path, "rb") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    obj = json.loads(line)
                    return int(obj["capture_time"])
    except Exception:
        pass
    return None


def _read_trigger_ms(metadata_path: Path) -> Optional[int]:
    """Retourne trigger_time_ns converti en ms depuis metadata.json."""
    try:
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta["trigger_time_ns"]) // 1_000_000
    except Exception:
        pass
    return None


# ─── Détection pour une session ─────────────────────────────────────────────

def detect_session(session_dir: Path) -> dict:
    """
    Analyse une session et retourne un dict de résultat.

    La signature du décalage est simple et fiable :
      - GOOD:     1er frame caméra arrive < THRESHOLD_MS après trigger_time_ns
      - DECALAGE: 1er frame caméra arrive > THRESHOLD_MS après trigger_time_ns

    Logique: on prend le MAX du délai sur les 3 caméras disponibles.
    Si au moins une caméra est en retard, la session est décalée.
    """
    result = {
        "session": str(session_dir),
        "status": "UNKNOWN",
        "delay_ms": None,
        "threshold_ms": THRESHOLD_MS,
        "camera_delays_ms": {},
        "details": "",
    }

    # 1. Lire le trigger depuis metadata.json
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        result["status"] = "ERROR"
        result["details"] = "metadata.json introuvable"
        return result

    trigger_ms = _read_trigger_ms(metadata_path)
    if trigger_ms is None:
        result["status"] = "ERROR"
        result["details"] = "trigger_time_ns manquant dans metadata.json"
        return result

    # 2. Lire le premier frame de chaque camera JSONL
    videos_dir = session_dir / "videos"
    if not videos_dir.exists():
        result["status"] = "ERROR"
        result["details"] = "répertoire videos/ introuvable"
        return result

    camera_delays = {}
    for cam in ("head", "left", "right"):
        jsonl_path = videos_dir / f"{cam}.jsonl"
        if not jsonl_path.exists():
            continue
        first_ms = _read_jsonl_first(jsonl_path)
        if first_ms is not None:
            camera_delays[cam] = first_ms - trigger_ms

    if not camera_delays:
        result["status"] = "ERROR"
        result["details"] = "aucun fichier JSONL caméra lisible"
        return result

    # 3. Décision : décalage = MAX délai > seuil
    max_delay = max(camera_delays.values())
    result["camera_delays_ms"] = camera_delays
    result["delay_ms"] = max_delay

    if max_delay > THRESHOLD_MS:
        result["status"] = "DECALAGE"
        result["details"] = (
            f"Premier frame vidéo arrive {max_delay}ms après le trigger "
            f"(seuil={THRESHOLD_MS}ms). "
            f"Les vidéos sont décalées d'environ {max_delay/1000:.2f}s par rapport aux capteurs."
        )
    else:
        result["status"] = "GOOD"
        result["details"] = (
            f"Premier frame vidéo arrive {max_delay}ms après le trigger "
            f"(seuil={THRESHOLD_MS}ms). Synchronisation normale."
        )

    return result


# ─── Scan récursif ──────────────────────────────────────────────────────────

def find_sessions(root: Path) -> list[Path]:
    """
    Retourne toutes les sessions (dossiers contenant metadata.json + videos/).
    Cherche en profondeur jusqu'à 3 niveaux.
    """
    sessions = []
    for candidate in sorted(root.rglob("metadata.json")):
        session_dir = candidate.parent
        if (session_dir / "videos").exists():
            sessions.append(session_dir)
    return sessions


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    target = Path(sys.argv[1])

    if not target.exists():
        print(json.dumps({"error": f"Chemin introuvable: {target}"}))
        sys.exit(2)

    # Cas 1 : on pointe directement sur une session
    if (target / "metadata.json").exists() and (target / "videos").exists():
        sessions = [target]
    # Cas 2 : on pointe sur un répertoire racine → scan récursif
    elif target.is_dir():
        sessions = find_sessions(target)
        if not sessions:
            print(json.dumps({"error": "Aucune session trouvée dans ce répertoire"}))
            sys.exit(2)
    else:
        print(json.dumps({"error": "Le chemin doit pointer sur une session ou un répertoire de sessions"}))
        sys.exit(2)

    results = []
    any_decalage = False

    for session_dir in sessions:
        r = detect_session(session_dir)
        results.append(r)
        if r["status"] == "DECALAGE":
            any_decalage = True

    # Affichage
    if len(results) == 1:
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
        _print_summary(results[0])
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print("\n" + "="*60)
        print(f"{'SESSION':<50} {'STATUS':<12} {'DELAY':>10}")
        print("="*60)
        for r in results:
            session_name = Path(r["session"]).name
            flag = " ← DECALAGE !" if r["status"] == "DECALAGE" else ""
            delay_str = f"{r['delay_ms']}ms" if r["delay_ms"] is not None else "?"
            print(f"{session_name:<50} {r['status']:<12} {delay_str:>10}{flag}")
        print("="*60)
        n_decalage = sum(1 for r in results if r["status"] == "DECALAGE")
        n_good = sum(1 for r in results if r["status"] == "GOOD")
        print(f"Total: {len(results)} sessions — {n_good} GOOD, {n_decalage} DECALAGE")

    sys.exit(1 if any_decalage else 0)


def _print_summary(r: dict):
    print()
    status_icon = "DECALAGE" if r["status"] == "DECALAGE" else "GOOD"
    print(f"  Status : {status_icon}")
    print(f"  Délai  : {r['delay_ms']}ms")
    if r["camera_delays_ms"]:
        for cam, d in r["camera_delays_ms"].items():
            print(f"    {cam:>5}: +{d}ms")
    print(f"  Info   : {r['details']}")


if __name__ == "__main__":
    main()
