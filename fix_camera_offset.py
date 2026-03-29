#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_camera_offset.py — Recale et tronque les capture_time des caméras sur la fenêtre du tracker.

Problème dans les sessions désynchronisées :
    Les caméras démarrent 6 à 23 secondes AVANT le tracker.
    Les capture_time sont des timestamps epoch ms corrects, mais décalés.
    Résultat : la majorité des frames caméra n'ont pas de correspondance tracker
    et la session est inutilisable.

Deux opérations effectuées :

  1. RECALAGE (offset)
     offset_ms = cam[0].capture_time - tracker[0].timestamp_ns/1e6
     capture_time_corrigé = capture_time_original - round(offset_ms)
     => cam[0] s'aligne sur tracker[0]

  2. TRONCATURE
     On supprime du JSONL toutes les frames dont le capture_time corrigé
     est en dehors de [tracker_t0, tracker_t1].
     Ces frames n'ont pas de correspondance proprioceptive et corrompent
     la visualisation / l'export LeRobot.
     La vidéo mp4 n'est PAS modifiée : les frames supprimées du JSONL
     sont simplement ignorées par la pipeline (seek par index).

Marker anti-double-application :
    metadata.json reçoit "camera_tracker_sync_applied": true après correction.
    Utiliser --force pour ré-appliquer.

Usage :
    python3 fix_camera_offset.py <session_path>
    python3 fix_camera_offset.py <root_dir> --batch
    python3 fix_camera_offset.py <root_dir> --batch --dry-run
    python3 fix_camera_offset.py <session_path> --force
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

CAMERAS = ["head", "left", "right"]
MARKER_KEY = "camera_tracker_sync_applied"

# Sessions sync : écart cam/tracker typiquement 20-100 ms → pas de correction
# Sessions desync : écart 5 000 à 25 000 ms → correction nécessaire
OFFSET_THRESHOLD_MS = 500.0


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        raw = f.read()
    frames = []
    for part in raw.split(b"\r\n"):
        part = part.strip()
        if len(part) > 5:
            try:
                frames.append(json.loads(part))
            except json.JSONDecodeError:
                pass
    return frames


def write_jsonl(path: Path, frames: list[dict]) -> None:
    lines = [json.dumps(frame, separators=(",", ":")) + "\r\n" for frame in frames]
    with open(path, "wb") as f:
        f.write("".join(lines).encode("utf-8"))


def read_tracker_window_ms(session_path: Path) -> tuple[float, float] | tuple[None, None]:
    """
    Retourne (tracker_t0_ms, tracker_t1_ms) depuis tracker_positions.csv.
    Utilise timestamp_ns (epoch nanoseconds).
    """
    tracker_path = session_path / "tracker_positions.csv"
    if not tracker_path.exists():
        return None, None

    t_first = None
    t_last = None
    with open(tracker_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ns_str = row.get("timestamp_ns", "").strip()
            if not ns_str:
                continue
            try:
                t_ms = int(ns_str) / 1_000_000
            except ValueError:
                continue
            if t_first is None:
                t_first = t_ms
            t_last = t_ms

    return t_first, t_last


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def fix_session(session_path: Path, dry_run: bool = False, force: bool = False) -> dict:
    """
    Corrige une session :
      - recale les capture_time des caméras sur tracker[0]
      - tronque le JSONL à la fenêtre [tracker_t0, tracker_t1]

    Retourne un dict rapport.
    """
    session_name = session_path.name
    meta_path = session_path / "metadata.json"

    if not meta_path.exists():
        return {"session": session_name, "status": "skipped", "reason": "metadata.json absent"}

    with open(meta_path, "rb") as f:
        meta = json.loads(f.read())

    if not force and meta.get(MARKER_KEY):
        return {
            "session": session_name,
            "status": "skipped",
            "reason": f"déjà corrigée ({MARKER_KEY}=true)",
        }

    trk_t0, trk_t1 = read_tracker_window_ms(session_path)
    if trk_t0 is None:
        return {
            "session": session_name,
            "status": "error",
            "reason": "tracker_positions.csv introuvable ou sans timestamp_ns valide",
        }

    # Calcul des offsets par caméra
    offsets: dict[str, float] = {}
    cam_frames: dict[str, list[dict]] = {}

    for cam in CAMERAS:
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        if not jsonl_path.exists():
            continue
        frames = read_jsonl(jsonl_path)
        if not frames:
            continue
        cam_frames[cam] = frames
        offsets[cam] = frames[0]["capture_time"] - trk_t0

    if not offsets:
        return {"session": session_name, "status": "error", "reason": "aucun fichier .jsonl trouvé"}

    max_offset = max(abs(v) for v in offsets.values())
    if max_offset < OFFSET_THRESHOLD_MS and not force:
        return {
            "session": session_name,
            "status": "ok",
            "reason": (
                f"offset max={max_offset:.1f} ms < seuil {OFFSET_THRESHOLD_MS} ms — "
                f"caméras déjà alignées sur le tracker"
            ),
            "offsets_ms": offsets,
            "tracker_t0_ms": trk_t0,
            "tracker_t1_ms": trk_t1,
        }

    report = {
        "session": session_name,
        "status": "dry-run" if dry_run else "corrected",
        "tracker_t0_ms": trk_t0,
        "tracker_t1_ms": trk_t1,
        "tracker_duration_s": (trk_t1 - trk_t0) / 1000,
        "offsets_ms": offsets,
        "cameras_fixed": [],
    }

    for cam, offset_ms in offsets.items():
        jsonl_path = session_path / "videos" / f"{cam}.jsonl"
        frames = cam_frames[cam]
        offset_int = round(offset_ms)

        # 1. Recalage
        recaled = [
            {**frame, "capture_time": frame["capture_time"] - offset_int}
            for frame in frames
        ]

        # 2. Troncature : ne garder que les frames dans la fenêtre tracker
        truncated = [
            frame for frame in recaled
            if trk_t0 <= frame["capture_time"] <= trk_t1
        ]

        n_removed = len(recaled) - len(truncated)
        overlap_s = (truncated[-1]["capture_time"] - truncated[0]["capture_time"]) / 1000 if truncated else 0

        report["cameras_fixed"].append({
            "camera": cam,
            "offset_ms": offset_ms,
            "offset_applied_ms": offset_int,
            "frames_original": len(frames),
            "frames_kept": len(truncated),
            "frames_removed": n_removed,
            "overlap_s": round(overlap_s, 2),
            "first_original_ms": frames[0]["capture_time"],
            "first_corrected_ms": truncated[0]["capture_time"] if truncated else None,
            "last_corrected_ms": truncated[-1]["capture_time"] if truncated else None,
        })

        if not dry_run and truncated:
            bak_path = jsonl_path.with_suffix(".jsonl.bak")
            if not bak_path.exists():
                shutil.copy2(jsonl_path, bak_path)
            write_jsonl(jsonl_path, truncated)

    if not dry_run:
        meta[MARKER_KEY] = True
        meta["camera_tracker_sync_offsets_ms"] = offsets
        meta["camera_tracker_sync_tracker_t0_ms"] = trk_t0
        meta["camera_tracker_sync_tracker_t1_ms"] = trk_t1
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI output
# ──────────────────────────────────────────────────────────────────────────────

def print_report(report: dict) -> None:
    status = report["status"]
    session = report["session"]
    reason = report.get("reason", "")

    if status in ("skipped", "ok"):
        print(f"  [{status.upper()}] {session} — {reason}")
        return
    if status == "error":
        print(f"  [ERROR] {session} — {reason}")
        return

    trk_dur = report.get("tracker_duration_s", 0)
    print(f"  [{status.upper()}] {session}  (tracker window={trk_dur:.1f}s)")
    for c in report.get("cameras_fixed", []):
        cam     = c["camera"]
        off     = c["offset_ms"]
        kept    = c["frames_kept"]
        removed = c["frames_removed"]
        total   = c["frames_original"]
        overlap = c["overlap_s"]
        pct     = kept / total * 100 if total else 0
        print(
            f"    {cam}: offset={off:+.0f}ms  "
            f"kept={kept}/{total} ({pct:.0f}%)  "
            f"removed={removed}  overlap={overlap:.1f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recale et tronque les capture_time caméras sur la fenêtre du tracker."
    )
    parser.add_argument("path", help="Chemin vers la session ou le dossier racine")
    parser.add_argument("--batch", action="store_true", help="Traiter toutes les sessions du dossier")
    parser.add_argument("--dry-run", action="store_true", help="Afficher sans modifier les fichiers")
    parser.add_argument("--force", action="store_true", help="Forcer même si déjà corrigée")
    args = parser.parse_args()

    root = Path(args.path)

    if args.batch:
        sessions = sorted(
            p.parent for p in root.rglob("metadata.json")
            if (p.parent / "videos").exists()
        )
        if not sessions:
            print(f"Aucune session trouvée dans {root}")
            sys.exit(1)
        print(f"{'[DRY-RUN] ' if args.dry_run else ''}Traitement de {len(sessions)} session(s)...\n")
        for session_path in sessions:
            report = fix_session(session_path, dry_run=args.dry_run, force=args.force)
            print_report(report)
    else:
        if not root.exists():
            print(f"Chemin introuvable : {root}")
            sys.exit(1)
        report = fix_session(root, dry_run=args.dry_run, force=args.force)
        print_report(report)

    print()
    if args.dry_run:
        print("Mode dry-run : aucun fichier modifié.")


if __name__ == "__main__":
    main()
