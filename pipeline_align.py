#!/usr/bin/env python3
"""
pipeline_align.py - Aligne les trackers avec les timestamps JSONL des caméras

Usage:
  python3 pipeline_align.py <session_path>
  python3 pipeline_align.py <root_sessions> --batch
  python3 pipeline_align.py <root_sessions> --batch --validate-only
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CAMERAS = ["head", "left", "right"]

# Colonnes tracker par caméra (toutes sont dans tracker_positions.csv)
TRACKER_COLS = {
    "head": ["tracker_head_x", "tracker_head_y", "tracker_head_z",
             "tracker_head_qw", "tracker_head_qx", "tracker_head_qy", "tracker_head_qz"],
    "left": ["tracker_left_x", "tracker_left_y", "tracker_left_z",
             "tracker_left_qw", "tracker_left_qx", "tracker_left_qy", "tracker_left_qz"],
    "right": ["tracker_right_x", "tracker_right_y", "tracker_right_z",
              "tracker_right_qw", "tracker_right_qx", "tracker_right_qy", "tracker_right_qz"],
}


def load_jsonl(path):
    frames = []
    bad_lines = 0

    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                frames.append(obj)
            except json.JSONDecodeError:
                bad_lines += 1
                print(f"⚠️ Ligne JSON invalide ignorée dans {path.name} à la ligne {i}")

    if len(frames) == 0:
        raise Exception(f"Aucune ligne JSON valide dans {path}")

    if bad_lines > 0:
        print(f"⚠️ {bad_lines} ligne(s) invalide(s) ignorée(s) dans {path.name}")

    return frames


def find_jsonl(session_path, cam):
    """Cherche le JSONL dans root ou videos/"""
    for candidate in [
        session_path / f"{cam}.jsonl",
        session_path / "videos" / f"{cam}.jsonl",
    ]:
        if candidate.exists():
            return candidate
    return None


def align_session(session_path, validate_only=False):
    session_path = Path(session_path)

    tracker_csv = session_path / "tracker_positions.csv"
    metadata_file = session_path / "metadata.json"

    if not tracker_csv.exists():
        print(f"  [SKIP] tracker_positions.csv absent dans {session_path.name}")
        return {"status": "invalid", "reason": "tracker_positions.csv manquant"}

    if not metadata_file.exists():
        print(f"  [SKIP] metadata.json absent dans {session_path.name}")
        return {"status": "invalid", "reason": "metadata.json manquant"}

    # Charger metadata
    with open(metadata_file) as f:
        metadata = json.load(f)

    capture_unit = metadata.get("jsonl_capture_time_unit", "milliseconds")

    # Charger tracker CSV
    df_tracker = pd.read_csv(tracker_csv)
    if "timestamp_ns" in df_tracker.columns:
        tracker_times_ns = df_tracker["timestamp_ns"].values.astype(float)
    else:
        df_tracker["_ts"] = pd.to_datetime(df_tracker["timestamp"])
        tracker_times_ns = df_tracker["_ts"].astype(np.int64).values.astype(float)

    output_dir = session_path / "aligned_outputs"
    report = {"session": session_path.name, "cameras": {}, "status": "ok"}

    for cam in CAMERAS:
        jsonl_path = find_jsonl(session_path, cam)

        if not jsonl_path:
            print(f"  [SKIP] {cam}: JSONL introuvable")
            report["cameras"][cam] = {"status": "invalid", "reason": "jsonl manquant"}
            report["status"] = "needs_review"
            continue

        frames = load_jsonl(jsonl_path)
        if not frames:
            print(f"  [SKIP] {cam}: JSONL vide")
            report["cameras"][cam] = {"status": "invalid", "reason": "jsonl vide"}
            report["status"] = "needs_review"
            continue

        # Convertir capture_time en nanosecondes
        if capture_unit == "milliseconds":
            frame_times_ns = np.array([f["capture_time"] * 1_000_000 for f in frames], dtype=float)
        elif capture_unit == "microseconds":
            frame_times_ns = np.array([f["capture_time"] * 1_000 for f in frames], dtype=float)
        elif capture_unit == "nanoseconds":
            frame_times_ns = np.array([f["capture_time"] for f in frames], dtype=float)
        else:
            # Hypothèse par défaut : millisecondes
            frame_times_ns = np.array([f["capture_time"] * 1_000_000 for f in frames], dtype=float)

        frame_indices = [f["index"] for f in frames]

        # Vérifier le chevauchement temporel
        tracker_min = tracker_times_ns.min()
        tracker_max = tracker_times_ns.max()
        frame_min = frame_times_ns.min()
        frame_max = frame_times_ns.max()
        overlap_start = max(tracker_min, frame_min)
        overlap_end = min(tracker_max, frame_max)

        if overlap_start >= overlap_end:
            print(f"  [WARN] {cam}: pas de chevauchement temporel")
            print(f"         tracker: {tracker_min:.0f}–{tracker_max:.0f} ns")
            print(f"         frames:  {frame_min:.0f}–{frame_max:.0f} ns")
            report["cameras"][cam] = {"status": "needs_review", "reason": "pas de chevauchement temporel"}
            report["status"] = "needs_review"
            continue

        # Offset estimé (diagnostic)
        mean_offset_ms = (frame_times_ns.mean() - tracker_times_ns.mean()) / 1_000_000

        if validate_only:
            status = "ok" if abs(mean_offset_ms) < 50 else "needs_review"
            print(f"  [{cam}] offset estimé: {mean_offset_ms:+.1f} ms | "
                  f"frames: {len(frames)} | trackers: {len(df_tracker)} | status: {status}")
            report["cameras"][cam] = {
                "status": status,
                "estimated_offset_ms": float(mean_offset_ms),
                "n_frames": len(frames),
                "n_tracker_rows": len(df_tracker),
            }
            continue

        # Interpoler toutes les colonnes tracker pour toutes les frames (vectorisé)
        all_tracker_cols = [c for c in df_tracker.columns if c.startswith("tracker_")]
        result = {
            "frame_index": frame_indices,
            "capture_time_ns": frame_times_ns,
            "capture_time_s": (frame_times_ns - frame_times_ns[0]) / 1e9,
        }
        for col in all_tracker_cols:
            vals = df_tracker[col].values.astype(float)
            result[col] = np.interp(frame_times_ns, tracker_times_ns, vals)

        df_aligned = pd.DataFrame(result)

        output_dir.mkdir(exist_ok=True)
        out_path = output_dir / f"{cam}_aligned_trackers.csv"
        df_aligned.to_csv(out_path, index=False)
        print(f"  [OK] {cam}: {len(df_aligned)} frames alignées → {out_path.relative_to(session_path)}")

        align_status = "corrected" if abs(mean_offset_ms) > 10 else "ok"
        report["cameras"][cam] = {
            "status": align_status,
            "estimated_offset_ms": float(mean_offset_ms),
            "n_frames": len(df_aligned),
            "output": f"aligned_outputs/{cam}_aligned_trackers.csv",
        }

    # Statut global
    all_statuses = [v["status"] for v in report["cameras"].values()]
    if "invalid" in all_statuses or "needs_review" in all_statuses:
        report["status"] = "needs_review"
    elif "corrected" in all_statuses:
        report["status"] = "corrected"
    else:
        report["status"] = "ok"

    if not validate_only:
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / "alignment_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  [REPORT] status={report['status']} → {report_path.relative_to(session_path)}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Aligne les trackers avec les timestamps JSONL des caméras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python3 pipeline_align.py /chemin/session_20260322_133839
  python3 pipeline_align.py ~/Desktop --batch
  python3 pipeline_align.py ~/Desktop --batch --validate-only
        """,
    )
    parser.add_argument("path", help="Chemin de session ou dossier racine (avec --batch)")
    parser.add_argument("--batch", action="store_true",
                        help="Traiter toutes les sessions sous ce dossier")
    parser.add_argument("--validate-only", action="store_true",
                        help="Valider sans écrire de fichiers")
    args = parser.parse_args()

    root = Path(args.path).expanduser()

    if not root.exists():
        print(f"Chemin introuvable: {root}")
        sys.exit(1)

    if args.batch:
        # Trouver toutes les sessions (dossiers avec tracker_positions.csv)
        sessions = []
        for item in sorted(root.iterdir()):
            if item.is_dir() and (item / "tracker_positions.csv").exists():
                sessions.append(item)
            elif item.is_dir():
                # Un niveau plus profond
                for sub in sorted(item.iterdir()):
                    if sub.is_dir() and (sub / "tracker_positions.csv").exists():
                        sessions.append(sub)

        if not sessions:
            print(f"Aucune session trouvée dans {root}")
            sys.exit(1)

        mode = "VALIDATION SEULE" if args.validate_only else "ALIGNEMENT"
        print(f"\n=== BATCH {mode}: {len(sessions)} session(s) ===\n")

        results = {}
        for s in sessions:
            print(f"Session: {s.name}")
            results[s.name] = align_session(s, validate_only=args.validate_only)
            print()

        # Résumé
        print("=== RÉSUMÉ ===")
        for name, r in results.items():
            status = r.get("status", "?")
            symbol = "✓" if status == "ok" else ("~" if status == "corrected" else "!")
            print(f"  {symbol} {name}: {status}")

    else:
        print(f"\n=== Alignement: {root.name} ===\n")
        align_session(root, validate_only=args.validate_only)


if __name__ == "__main__":
    main()
