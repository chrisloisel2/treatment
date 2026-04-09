#!/usr/bin/env python3
# DÉSACTIVÉ — pipeline automatique suspendue jusqu'à nouvel ordre.
# Pour relancer, retirer le raise ci-dessous.
raise SystemExit("session_ingest_service désactivé — pipeline automatique suspendue.")

from __future__ import annotations

import contextlib
import fcntl
import importlib.util
import json
import logging
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pymongo import MongoClient


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INBOX_DIR = Path(os.getenv("INBOX_DIR", "/mnt/inbox")).resolve()
BRONZE_DIR = Path(os.getenv("BRONZE_DIR", "/mnt/storage/bronze")).resolve()

REORGANIZE_MODULE_PATH = Path(
    os.getenv("REORGANIZE_MODULE_PATH", "/opt/session_ingest/reorganize_sessions.py")
).resolve()

LOCK_DIR = Path(os.getenv("LOCK_DIR", "/run/session_ingest")).resolve()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
QUIET_PERIOD = int(os.getenv("QUIET_PERIOD", "15"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
VALIDATION_THREADS = int(os.getenv("VALIDATION_THREADS", "8"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "physical_data")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "cameras")

REJECT_DIR_NAME = "_REJECTED_MISSING_FILES"
REPORT_FILE_NAME = "_rejected_report.jsonl"
VALID_POSITIONS = {"left", "right", "head"}

STOP = False


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("session_ingest")


# -----------------------------------------------------------------------------
# Signals
# -----------------------------------------------------------------------------

def _handle_signal(signum, frame):
    global STOP
    STOP = True
    log.info("signal=%s stop_requested=1", signum)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# -----------------------------------------------------------------------------
# Load existing python module
# -----------------------------------------------------------------------------

def load_reorganize_module(module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("reorganize_sessions", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORG = load_reorganize_module(REORGANIZE_MODULE_PATH)

required_symbols = [
    "sanitize_name",
    "validate_session_files",
    "move_directory",
]
for symbol in required_symbols:
    if not hasattr(ORG, symbol):
        raise RuntimeError(f"missing symbol in module: {symbol}")


# -----------------------------------------------------------------------------
# Mongo
# -----------------------------------------------------------------------------

_mongo_client: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client


def mongo_position_by_serial(serial: str) -> Optional[str]:
    if not serial:
        return None

    coll = get_mongo_client()[MONGO_DB][MONGO_COLLECTION]
    doc = coll.find_one(
        {"serial_number": serial},
        {"_id": 0, "position": 1},
    )
    if not doc:
        return None

    position = str(doc.get("position", "")).strip()
    if position not in VALID_POSITIONS:
        return None
    return position


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: Dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def session_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out = []
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.startswith("session_"):
            out.append(entry)
    return sorted(out)


def file_is_stable(path: Path, quiet_period: int) -> bool:
    now = time.time()
    try:
        st = path.stat()
    except FileNotFoundError:
        return False
    return (now - st.st_mtime) >= quiet_period


def session_is_stable(session_dir: Path, quiet_period: int) -> Tuple[bool, str]:
    metadata = session_dir / "metadata.json"
    videos = session_dir / "videos"

    if not metadata.exists():
        return False, "missing_metadata"
    if not videos.exists() or not videos.is_dir():
        return False, "missing_videos_dir"

    part_files = list(videos.glob("*.part"))
    if part_files:
        return False, "partial_files_present"

    interesting = [metadata]
    interesting.extend(videos.glob("*.mp4"))
    interesting.extend(videos.glob("*.jsonl"))
    interesting.extend(session_dir.glob("*.csv"))

    if not interesting:
        return False, "empty_session"

    for p in interesting:
        if not file_is_stable(p, quiet_period):
            return False, f"not_quiet:{p.name}"

    return True, "stable"


@contextlib.contextmanager
def session_lock(session_name: str):
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / f"{session_name}.lock"
    with lock_path.open("w") as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("already_locked")
        yield


def safe_append_jsonl(path: Path, item: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def backup_file(path: Path) -> None:
    if path.exists():
        ts = int(time.time())
        shutil.copy2(path, path.with_name(f"{path.name}.bak.{ts}"))


# -----------------------------------------------------------------------------
# Camera normalization
# -----------------------------------------------------------------------------

def normalize_camera_positions(session_dir: Path) -> Tuple[Dict, bool]:
    metadata_path = session_dir / "metadata.json"
    videos_dir = session_dir / "videos"

    metadata = read_json(metadata_path)
    cameras = metadata.get("cameras", {})
    if not isinstance(cameras, dict) or not cameras:
        return metadata, False

    desired_by_cam_id: Dict[str, str] = {}
    current_by_cam_id: Dict[str, str] = {}

    for cam_id, cam_info in cameras.items():
        if not isinstance(cam_info, dict):
            continue

        serial = str(cam_info.get("serial", "")).strip()
        current = str(cam_info.get("position", "")).strip()

        if not serial or not current:
            continue

        desired = mongo_position_by_serial(serial)
        if not desired:
            continue

        current_by_cam_id[cam_id] = current
        desired_by_cam_id[cam_id] = desired

    if not desired_by_cam_id:
        return metadata, False

    used_destinations = {}
    for cam_id, dst in desired_by_cam_id.items():
        used_destinations.setdefault(dst, []).append(cam_id)

    collisions = {k: v for k, v in used_destinations.items() if len(v) > 1}
    if collisions:
        raise RuntimeError(f"duplicate_destination_positions={collisions}")

    changed = False
    tmp_prefix = f".rename_tmp_{os.getpid()}"

    for cam_id, src_pos in current_by_cam_id.items():
        dst_pos = desired_by_cam_id[cam_id]
        if src_pos == dst_pos:
            continue

        src_mp4 = videos_dir / f"{src_pos}.mp4"
        src_jsonl = videos_dir / f"{src_pos}.jsonl"
        tmp_mp4 = videos_dir / f"{tmp_prefix}_{cam_id}.mp4"
        tmp_jsonl = videos_dir / f"{tmp_prefix}_{cam_id}.jsonl"

        if src_mp4.exists():
            src_mp4.rename(tmp_mp4)
            changed = True
        if src_jsonl.exists():
            src_jsonl.rename(tmp_jsonl)
            changed = True

    for cam_id, src_pos in current_by_cam_id.items():
        dst_pos = desired_by_cam_id[cam_id]

        if src_pos != dst_pos:
            tmp_mp4 = videos_dir / f"{tmp_prefix}_{cam_id}.mp4"
            tmp_jsonl = videos_dir / f"{tmp_prefix}_{cam_id}.jsonl"
            dst_mp4 = videos_dir / f"{dst_pos}.mp4"
            dst_jsonl = videos_dir / f"{dst_pos}.jsonl"

            if tmp_mp4.exists():
                tmp_mp4.rename(dst_mp4)
            if tmp_jsonl.exists():
                tmp_jsonl.rename(dst_jsonl)

        metadata["cameras"][cam_id]["position"] = dst_pos

    if changed:
        backup_file(metadata_path)
        write_json_atomic(metadata_path, metadata)

    leftovers = list(videos_dir.glob("cam*"))
    if leftovers:
        raise RuntimeError(
            "unexpected_cam_files_remaining=" + ",".join(sorted(p.name for p in leftovers))
        )

    return metadata, changed


# -----------------------------------------------------------------------------
# Bronze organization
# -----------------------------------------------------------------------------

def reject_destination(session_name: str) -> Path:
    return BRONZE_DIR / REJECT_DIR_NAME / session_name


def bronze_flat_destination(session_name: str) -> Path:
    return BRONZE_DIR / session_name


def bronze_scenario_destination(metadata: Dict, session_name: str) -> Path:
    scenario = ORG.sanitize_name(metadata.get("scenario", "UNKNOWN_SCENARIO"))
    return BRONZE_DIR / scenario / session_name


def organize_session_in_bronze(session_path_in_bronze: Path) -> str:
    metadata_path = session_path_in_bronze / "metadata.json"
    metadata = read_json(metadata_path)
    dst = bronze_scenario_destination(metadata, session_path_in_bronze.name)
    return ORG.move_directory(session_path_in_bronze, dst, dry_run=False)


# -----------------------------------------------------------------------------
# Session processing
# -----------------------------------------------------------------------------

def process_one_session(session_dir: Path) -> Dict:
    result = {
        "session": session_dir.name,
        "source": str(session_dir),
        "status": None,
        "reason": None,
        "missing_files": [],
        "destination": None,
        "error": None,
    }

    try:
        with session_lock(session_dir.name):
            stable, stable_reason = session_is_stable(session_dir, QUIET_PERIOD)
            if not stable:
                result["status"] = "skipped"
                result["reason"] = stable_reason
                return result

            metadata, renamed = normalize_camera_positions(session_dir)
            if renamed:
                log.info("session=%s camera_positions_updated=1", session_dir.name)

            is_valid, missing_files = ORG.validate_session_files(
                session_dir=session_dir,
                metadata=metadata,
                max_threads=VALIDATION_THREADS,
            )

            if not is_valid:
                dst = reject_destination(session_dir.name)
                final_dst = ORG.move_directory(session_dir, dst, dry_run=False)

                result["status"] = "rejected"
                result["reason"] = "missing_files"
                result["missing_files"] = missing_files
                result["destination"] = final_dst

                safe_append_jsonl(
                    BRONZE_DIR / REJECT_DIR_NAME / REPORT_FILE_NAME,
                    result,
                )
                return result

            flat_dst = bronze_flat_destination(session_dir.name)
            flat_final_dst = ORG.move_directory(session_dir, flat_dst, dry_run=False)

            organized_dst = organize_session_in_bronze(Path(flat_final_dst))

            result["status"] = "moved"
            result["reason"] = "ok"
            result["destination"] = organized_dst
            return result

    except RuntimeError as e:
        if str(e) == "already_locked":
            result["status"] = "skipped"
            result["reason"] = "already_locked"
            return result

        result["status"] = "error"
        result["reason"] = "runtime_error"
        result["error"] = str(e)
        return result

    except Exception as e:
        result["status"] = "error"
        result["reason"] = "exception"
        result["error"] = f"{type(e).__name__}: {e}"
        return result


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def ensure_layout() -> None:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    (BRONZE_DIR / REJECT_DIR_NAME).mkdir(parents=True, exist_ok=True)


def log_result(res: Dict) -> None:
    if res["status"] == "moved":
        log.info(
            "session=%s status=moved destination=%s",
            res["session"],
            res["destination"],
        )
    elif res["status"] == "rejected":
        log.warning(
            "session=%s status=rejected missing=%s destination=%s",
            res["session"],
            res["missing_files"],
            res["destination"],
        )
    elif res["status"] == "skipped":
        log.debug(
            "session=%s status=skipped reason=%s",
            res["session"],
            res["reason"],
        )
    else:
        log.error(
            "session=%s status=error reason=%s error=%s",
            res["session"],
            res["reason"],
            res["error"],
        )


def main() -> int:
    ensure_layout()

    log.info(
        "start inbox=%s bronze=%s module=%s workers=%s interval=%s quiet=%s",
        INBOX_DIR,
        BRONZE_DIR,
        REORGANIZE_MODULE_PATH,
        MAX_WORKERS,
        POLL_INTERVAL,
        QUIET_PERIOD,
    )

    while not STOP:
        sessions = session_dirs(INBOX_DIR)

        if sessions:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(process_one_session, s): s for s in sessions}
                for fut in as_completed(futures):
                    res = fut.result()
                    log_result(res)

        time.sleep(POLL_INTERVAL)

    log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
