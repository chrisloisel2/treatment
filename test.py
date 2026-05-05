#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set


TARGET_DATE = "20260428"

# # CONFIG MAIL
# SMTP_SERVER = "smtp.example.com"
# SMTP_PORT = 587
# SMTP_USER = "user@example.com"
# SMTP_PASSWORD = "password"
# MAIL_FROM = "user@example.com"
# MAIL_TO = ["destinataire@example.com"]

ROOTS = {
    "inbox": [Path("/Volumes/T9/poste_20_28_04_2026")],
}

CHUNK_SIZE = 1000


def send_mail(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = MAIL_FROM
    msg["To"] = ", ".join(MAIL_TO)
    msg.set_content(body)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)


def is_target_session_dir(path: Path, target_date: str) -> bool:
    return path.is_dir() and path.name.startswith(f"session_{target_date}")


def iter_session_dirs(base: Path, target_date: str) -> Iterable[Path]:
    if not base.exists():
        return

    seen: Set[Path] = set()

    for meta in base.rglob("metadata.json"):
        session_dir = meta.parent
        if is_target_session_dir(session_dir, target_date):
            resolved = session_dir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield session_dir


def safe_load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def iter_files_in_chunks(path: Path, chunk_size: int = CHUNK_SIZE) -> Iterator[List[Path]]:
    chunk: List[Path] = []

    try:
        for root, dirs, files in os.walk(path, followlinks=False):
            for name in files:
                chunk.append(Path(root) / name)

                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    except (PermissionError, OSError):
        pass

    if chunk:
        yield chunk


def get_chunk_size_bytes(files: List[Path]) -> int:
    total = 0
    for p in files:
        try:
            total += p.stat().st_size
        except (FileNotFoundError, PermissionError, OSError):
            pass
    return total


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(bucket, current_session, sessions, files, size_bytes, duration):
    line = (
        f"\rBucket: {bucket:<10} | "
        f"Session: {current_session[:28]:<28} | "
        f"Sessions: {sessions:<6} | "
        f"Fichiers: {files:<10} | "
        f"Taille: {size_bytes / (1024 ** 3):.3f} GiB | "
        f"Duree: {format_duration(duration)}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def analyze_bucket(bucket_name, bases, target_date):
    session_dirs: Dict[Path, None] = {}

    for base in bases:
        for session_dir in iter_session_dirs(base, target_date):
            session_dirs[session_dir.resolve()] = None

    count = 0
    total_size_bytes = 0
    total_duration_seconds = 0.0
    total_files = 0

    for session_dir in sorted(session_dirs.keys()):
        metadata_path = session_dir / "metadata.json"
        metadata = safe_load_json(metadata_path)

        duration = metadata.get("duration_seconds", 0)
        if not isinstance(duration, (int, float)):
            duration = 0.0

        session_size_bytes = 0
        session_files = 0

        for files_chunk in iter_files_in_chunks(session_dir, CHUNK_SIZE):
            session_files += len(files_chunk)
            session_size_bytes += get_chunk_size_bytes(files_chunk)

            print_progress(
                bucket_name,
                session_dir.name,
                count,
                total_files + session_files,
                total_size_bytes + session_size_bytes,
                total_duration_seconds + float(duration),
            )

        count += 1
        total_files += session_files
        total_size_bytes += session_size_bytes
        total_duration_seconds += float(duration)

    print()

    return {
        "bucket": bucket_name,
        "sessions": count,
        "files": total_files,
        "size_gib": total_size_bytes / (1024 ** 3),
        "duration_seconds": total_duration_seconds,
    }


def build_report(rows: List[Dict[str, float]], target_date: str) -> str:
    lines = []
    lines.append(f"Recapitulatif des sessions du {target_date}\n")

    total_sessions = 0
    total_files = 0
    total_size_gib = 0.0
    total_duration_seconds = 0.0

    for row in rows:
        total_sessions += int(row["sessions"])
        total_files += int(row["files"])
        total_size_gib += float(row["size_gib"])
        total_duration_seconds += float(row["duration_seconds"])

        lines.append(
            f"{row['bucket']} | sessions={row['sessions']} | files={row['files']} | "
            f"size={row['size_gib']:.3f} GiB | duration={format_duration(row['duration_seconds'])}"
        )

    lines.append("\nTOTAL")
    lines.append(
        f"sessions={total_sessions} | files={total_files} | "
        f"size={total_size_gib:.3f} GiB | duration={format_duration(total_duration_seconds)}"
    )

    return "\n".join(lines)


def main() -> None:
    rows = []

    for bucket, bases in ROOTS.items():
        existing_bases = [p for p in bases if p.exists()]
        rows.append(analyze_bucket(bucket, existing_bases, TARGET_DATE))

    report = build_report(rows, TARGET_DATE)
    print(report)

    send_mail(
        subject=f"Rapport sessions {TARGET_DATE}",
        body=report
    )


if __name__ == "__main__":
    main()
