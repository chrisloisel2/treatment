"""
lerobot_convert.py — Conversion de sessions classiques → dataset LeRobot v3.0

Usage standalone :
    python lerobot_convert.py \\
        --sessions /path/sess1 /path/sess2 \\
        --output   /path/to/output \\
        [--dataset-name robot_dataset] \\
        [--robot-type so100] \\
        [--fps 30] \\
        [--chunks-size 1000] \\
        [--push-to-hub] [--hf-token TOKEN] [--repo-id owner/repo]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import traceback
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConvertOptions:
    dataset_name: str = "robot_dataset"
    robot_type:   str = "so100"
    fps:          int = 30
    chunks_size:  int = 1000
    batch_size:   int = 5
    push_to_hub:  bool = False
    hf_token:     str = ""
    repo_id:      str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str, level: str = "INFO") -> None:
    print(f"[{level}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Filtrage des fichiers de backup
# ─────────────────────────────────────────────────────────────────────────────

_BACKUP_SUFFIXES = {".bak", ".orig", ".old", ".tmp", ".swp"}
_BACKUP_PATTERNS = ("_backup", "_bak", "_old", "_orig", ".bak.", "_copy")


def _is_backup(path: Path) -> bool:
    """Retourne True si le fichier ressemble à un backup et doit être ignoré."""
    name = path.name.lower()
    if path.suffix.lower() in _BACKUP_SUFFIXES:
        return True
    return any(pat in name for pat in _BACKUP_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers vidéo / JSONL
# ─────────────────────────────────────────────────────────────────────────────

def _read_video_info(mp4_path: Path, fallback_fps: int = 30) -> dict:
    """Lit les métadonnées réelles d'une vidéo via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(mp4_path)],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(r.stdout)
        vs = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
        if vs is None:
            raise ValueError("no video stream")
        num, den = vs.get("r_frame_rate", "30/1").split("/")
        fps = int(round(int(num) / max(1, int(den))))
        pix_fmt = vs.get("pix_fmt", "yuv420p")
        if pix_fmt == "yuvj420p":
            pix_fmt = "yuv420p"
        channels = 1 if "gray" in pix_fmt else (4 if "rgba" in pix_fmt or "yuva" in pix_fmt else 3)
        has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])
        return {
            "video.fps":         fps,
            "video.codec":       vs.get("codec_name", "h264"),
            "video.pix_fmt":     pix_fmt,
            "video.height":      int(vs.get("height", 480)),
            "video.width":       int(vs.get("width", 640)),
            "video.channels":    channels,
            "video.is_depth_map": False,
            "has_audio":         has_audio,
        }
    except Exception:
        return {
            "video.fps":         fallback_fps,
            "video.codec":       "h264",
            "video.pix_fmt":     "yuv420p",
            "video.height":      480,
            "video.width":       640,
            "video.channels":    3,
            "video.is_depth_map": False,
            "has_audio":         False,
        }


def _read_jsonl(path: Path) -> list[dict]:
    """Lit un fichier JSONL caméra (format {index, capture_time}).
    Les lignes tronquées ou invalides sont ignorées silencieusement."""
    frames = []
    try:
        raw = path.read_bytes()
        for part in raw.split(b"\r\n"):
            part = part.strip()
            if len(part) > 5:
                try:
                    obj = json.loads(part)
                    if "capture_time" in obj and "index" in obj:
                        frames.append(obj)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Construction de la timeline et fusion des CSV
# ─────────────────────────────────────────────────────────────────────────────

# Colonnes structurelles qui ne doivent pas être préfixées
_NO_PREFIX = {
    "time_seconds", "timestamp", "timestamp_ns",
    "frame_index", "episode_index", "index", "task_index", "_ts_ms_key",
}

STANDARD_COLS = {
    "timestamp", "frame_index", "episode_index", "index", "task_index",
    "time_seconds", "capture_time_ms", "_ts_ms_key",
}


def _prefixed_df(df: "pd.DataFrame", stem: str) -> "pd.DataFrame":
    first_word = stem.split("_")[0]
    rename = {}
    for col in df.columns:
        if col in _NO_PREFIX:
            continue
        safe = col.replace("-", "_").replace(" ", "_")
        if safe.startswith(first_word):
            rename[col] = safe
        else:
            rename[col] = f"{stem}_{safe}"
    return df.rename(columns=rename)


def _build_camera_timeline(sess: Path, camera_keys: list[str]) -> "pd.DataFrame | None":
    """
    Construit la timeline de référence à partir du JSONL de la première caméra disponible.
    Retourne un DataFrame avec colonnes : capture_time_ms, timestamp (relatif en secondes).
    """
    import pandas as pd

    for cam_key in camera_keys:
        cam_name = cam_key.replace("observation.images.", "")
        jsonl_path = sess / "videos" / f"{cam_name}.jsonl"
        if not jsonl_path.exists():
            continue
        frames = _read_jsonl(jsonl_path)
        if not frames:
            continue
        capture_times = [float(f["capture_time"]) for f in frames]
        t0 = capture_times[0]
        _log(f"Timeline caméra : {jsonl_path.name} — {len(capture_times)} frames, t0={t0}")
        timestamps = [(t - t0) / 1000.0 for t in capture_times]
        return pd.DataFrame({
            "capture_time_ms": capture_times,
            "timestamp":       timestamps,
        })
    return None


def _merge_session_csvs(
    sess: Path,
    cam_timeline: "pd.DataFrame | None",
) -> "pd.DataFrame | None":
    """
    Lit tous les CSV de la session, préfixe leurs colonnes avec le nom de fichier,
    puis les fusionne sur la timeline caméra via merge_asof.
    """
    import numpy as np  # noqa: F401 — requis par pandas en interne
    import pandas as pd

    csv_files = [f for f in sorted(sess.glob("*.csv")) if not _is_backup(f)]
    if not csv_files:
        return cam_timeline

    dfs: list[tuple[str, "pd.DataFrame"]] = []
    for csv_f in csv_files:
        try:
            df_tmp = pd.read_csv(csv_f)
            if df_tmp.empty:
                continue
            if "timestamp_ns" in df_tmp.columns:
                df_tmp["_ts_ms_key"] = pd.to_numeric(df_tmp["timestamp_ns"], errors="coerce") / 1_000_000
            elif "time_seconds" in df_tmp.columns:
                df_tmp["_ts_ms_key"] = pd.to_numeric(df_tmp["time_seconds"], errors="coerce") * 1000
            if "_ts_ms_key" in df_tmp.columns:
                df_tmp = df_tmp.sort_values("_ts_ms_key").reset_index(drop=True)
            stem = csv_f.stem
            if stem.endswith("_data"):
                stem = stem[:-5]
            df_tmp = _prefixed_df(df_tmp, stem)
            dfs.append((stem, df_tmp))
        except Exception as e:
            _log(f"    CSV ignoré ({csv_f.name}): {e}", "WARN")

    if not dfs:
        return cam_timeline

    if cam_timeline is not None:
        base = cam_timeline.copy()
        for stem, other in dfs:
            if "_ts_ms_key" not in other.columns:
                continue
            new_cols = [
                c for c in other.columns
                if c not in base.columns and c != "_ts_ms_key"
                and pd.api.types.is_numeric_dtype(other[c])
            ]
            if not new_cols:
                continue
            other_sub = other[["_ts_ms_key"] + new_cols].copy()
            base = pd.merge_asof(
                base.sort_values("capture_time_ms"),
                other_sub.sort_values("_ts_ms_key"),
                left_on="capture_time_ms",
                right_on="_ts_ms_key",
                direction="nearest",
                tolerance=500.0,
            )
            base = base.drop(
                columns=[c for c in base.columns if c.startswith("_ts_ms_key")],
                errors="ignore",
            )
        base = base.drop(columns=["capture_time_ms"], errors="ignore")
        return base

    # Fallback : pas de timeline caméra, on fusionne les CSV entre eux
    dfs.sort(key=lambda t: len(t[1]), reverse=True)
    base = dfs[0][1].copy()
    for _, other in dfs[1:]:
        new_cols = [c for c in other.columns if c != "time_seconds" and c not in base.columns]
        if not new_cols:
            continue
        if "time_seconds" in base.columns and "time_seconds" in other.columns:
            other_sub = other[["time_seconds"] + new_cols].copy()
            base = pd.merge_asof(
                base, other_sub, on="time_seconds", direction="nearest", tolerance=0.5,
            )
        else:
            for col in new_cols:
                base[col] = other[col].reindex(base.index)
    base = base.drop(columns=["_ts_ms_key"], errors="ignore")
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Flush intermédiaire des épisodes
# ─────────────────────────────────────────────────────────────────────────────

def _flush_episode_parquet(meta_dir: Path, episode_rows: list[dict]) -> None:
    """Écrit (ou écrase) le parquet meta/episodes avec les lignes courantes."""
    try:
        import pandas as pd
        ep_parquet_path = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
        ep_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(episode_rows).to_parquet(ep_parquet_path, index=False)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale de conversion
# ─────────────────────────────────────────────────────────────────────────────

def build_lerobot_dataset(
    sessions: list[Path],
    opts: ConvertOptions,
    tmp_dir: Path,
    log_fn=None,
) -> Path:
    """
    Construit un dataset LeRobot v3.0 complet depuis une liste de sessions.

    Chaque session devient un épisode. Les épisodes sont répartis en chunks
    de opts.chunks_size fichiers chacun.

    Retourne le chemin du répertoire dataset créé.

    :param sessions:  Liste de Path vers chaque dossier de session.
    :param opts:      Options de conversion.
    :param tmp_dir:   Répertoire de travail (le dataset y sera construit).
    :param log_fn:    Fonction de log (str, level) → None. Défaut : print.
    """
    import pandas as pd

    log = log_fn or _log

    dataset_dir = tmp_dir / opts.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    chunks_size    = max(1, opts.chunks_size)
    total_episodes = len(sessions)

    # ── Découverte des caméras ────────────────────────────────────────────────
    camera_keys: list[str] = []
    for sess in sessions:
        vid_dir = sess / "videos"
        if vid_dir.is_dir():
            keys = [f"observation.images.{v.stem}" for v in sorted(vid_dir.glob("*.mp4")) if not _is_backup(v)]
            if keys:
                camera_keys = keys
                break
    if not camera_keys:
        for sess in sessions:
            vids = [v for v in sorted(sess.glob("*.mp4")) if not _is_backup(v)]
            if vids:
                camera_keys = [f"observation.images.{v.stem}" for v in vids]
                break

    # ── Métadonnées vidéo ─────────────────────────────────────────────────────
    video_infos: dict[str, dict] = {}
    for cam_key in camera_keys:
        cam_name = cam_key.replace("observation.images.", "")
        for sess in sessions:
            src = sess / "videos" / f"{cam_name}.mp4"
            if not src.exists():
                src = sess / f"{cam_name}.mp4"
            if src.exists():
                video_infos[cam_key] = _read_video_info(src, opts.fps)
                break
        if cam_key not in video_infos:
            video_infos[cam_key] = _read_video_info(Path("/dev/null"), opts.fps)

    # ── Schéma des features ───────────────────────────────────────────────────
    feature_columns: dict[str, dict] = {}
    for sess in sessions:
        for csv_f in sorted(sess.glob("*.csv")):
            if _is_backup(csv_f):
                continue
            try:
                df_sample = pd.read_csv(csv_f, nrows=2)
                stem = csv_f.stem
                if stem.endswith("_data"):
                    stem = stem[:-5]
                df_sample = _prefixed_df(df_sample, stem)
                for col in df_sample.columns:
                    if col in STANDARD_COLS:
                        continue
                    if pd.api.types.is_numeric_dtype(df_sample[col]):
                        feature_columns[col] = {"dtype": "float64", "shape": (1,), "names": None}
            except Exception:
                pass

    features: dict = {
        "timestamp":     {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index":   {"dtype": "int64",   "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64",   "shape": (1,), "names": None},
        "index":         {"dtype": "int64",   "shape": (1,), "names": None},
        "task_index":    {"dtype": "int64",   "shape": (1,), "names": None},
        **feature_columns,
    }
    for cam_key in camera_keys:
        vi = video_infos[cam_key]
        features[cam_key] = {
            "dtype": "video",
            "shape": (vi["video.height"], vi["video.width"], vi["video.channels"]),
            "names": ["height", "width", "channel"],
            "info":  vi,
        }

    log(f"Schema : {len(feature_columns)} colonnes de données + {len(camera_keys)} caméras", "INFO")

    # ── Itérer les sessions → épisodes ────────────────────────────────────────
    batch_size   = max(1, opts.batch_size)
    total_frames = 0
    episode_rows: list[dict] = []
    dataset_from = 0
    ep_errors:   list[str]  = []

    for ep_idx, sess in enumerate(sessions):
        chunk_idx = ep_idx // chunks_size
        file_idx  = ep_idx % chunks_size

        data_path = dataset_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"

        # Reprise après crash
        if data_path.exists():
            try:
                n_frames = len(pd.read_parquet(data_path, columns=["frame_index"]))
            except Exception:
                n_frames = 0
            ep_row: dict = {
                "episode_index":             ep_idx,
                "tasks":                     [0],
                "length":                    n_frames,
                "dataset_from_index":        dataset_from,
                "dataset_to_index":          dataset_from + n_frames,
                "data/chunk_index":          chunk_idx,
                "data/file_index":           file_idx,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index":  0,
            }
            for cam_key in camera_keys:
                ep_row[f"videos/{cam_key}/chunk_index"] = chunk_idx
                ep_row[f"videos/{cam_key}/file_index"]  = file_idx
            episode_rows.append(ep_row)
            dataset_from += n_frames
            total_frames += n_frames
            log(f"  épisode {ep_idx:04d} ({sess.name}) — déjà traité, skip", "INFO")
            continue

        try:
            cam_timeline = _build_camera_timeline(sess, camera_keys)
            ref_df       = _merge_session_csvs(sess, cam_timeline)
            n_frames     = len(ref_df) if ref_df is not None else 0

            rows: dict = {
                "frame_index":   list(range(n_frames)),
                "episode_index": [ep_idx] * n_frames,
                "index":         [total_frames + i for i in range(n_frames)],
                "task_index":    [0] * n_frames,
            }

            if ref_df is not None and "timestamp" in ref_df.columns:
                rows["timestamp"] = [float(v) for v in ref_df["timestamp"]]
            elif ref_df is not None and "time_seconds" in ref_df.columns:
                t0 = float(ref_df["time_seconds"].iloc[0])
                rows["timestamp"] = [float(v) - t0 for v in ref_df["time_seconds"]]
            else:
                rows["timestamp"] = [i / opts.fps for i in range(n_frames)]

            _SKIP_COLS = STANDARD_COLS | {"capture_time_ms", "_ts_ms_key"}
            if ref_df is not None:
                for col in ref_df.columns:
                    safe = col.replace("-", "_").replace(" ", "_")
                    if safe in rows or safe in _SKIP_COLS:
                        continue
                    col_data = ref_df[col]
                    if not pd.api.types.is_numeric_dtype(col_data):
                        continue
                    try:
                        rows[safe] = [float(v) if pd.notna(v) else 0.0 for v in col_data]
                    except (ValueError, TypeError):
                        pass

            ep_df = pd.DataFrame(rows)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            ep_df.to_parquet(data_path, index=False)

            parquet_duration_s = float(ep_df["timestamp"].iloc[-1]) if n_frames > 0 else 0.0
            del ep_df, rows, ref_df

            # Vidéos
            for cam_key in camera_keys:
                cam_name = cam_key.replace("observation.images.", "")
                src_mp4  = sess / "videos" / f"{cam_name}.mp4"
                if not src_mp4.exists():
                    src_mp4 = sess / f"{cam_name}.mp4"
                if not src_mp4.exists():
                    log(f"    vidéo manquante : {cam_name}.mp4 dans {sess.name}", "WARN")
                    continue

                vid_out = dataset_dir / "videos" / cam_key / f"chunk-{chunk_idx:03d}"
                vid_out.mkdir(parents=True, exist_ok=True)
                dst_mp4 = vid_out / f"file-{file_idx:03d}.mp4"

                try:
                    r_probe  = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(src_mp4)],
                        capture_output=True, text=True, timeout=10,
                    )
                    probe_d  = json.loads(r_probe.stdout)
                    probe_vs = next((s for s in probe_d["streams"] if s["codec_type"] == "video"), None)
                    src_duration = float(probe_vs["duration"]) if probe_vs else 0.0
                except Exception:
                    src_duration = 0.0

                needs_trim = src_duration > parquet_duration_s + 0.2 and parquet_duration_s > 0
                if needs_trim:
                    tmp_mp4 = tmp_dir / f"_trim_ep{ep_idx}_{cam_name}.mp4"
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(src_mp4),
                        "-t", f"{parquet_duration_s:.6f}",
                        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                        "-pix_fmt", "yuv420p",
                        str(tmp_mp4),
                    ]
                    r_ffmpeg = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if r_ffmpeg.returncode == 0:
                        shutil.move(str(tmp_mp4), str(dst_mp4))
                        log(f"    {cam_name}: trimmed {src_duration:.2f}s → {parquet_duration_s:.2f}s", "INFO")
                    else:
                        shutil.copy2(src_mp4, dst_mp4)
                        log(f"    {cam_name}: trim échoué, copie directe", "WARN")
                else:
                    shutil.copy2(src_mp4, dst_mp4)

            ep_row = {
                "episode_index":             ep_idx,
                "tasks":                     [0],
                "length":                    n_frames,
                "dataset_from_index":        dataset_from,
                "dataset_to_index":          dataset_from + n_frames,
                "data/chunk_index":          chunk_idx,
                "data/file_index":           file_idx,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index":  0,
            }
            for cam_key in camera_keys:
                ep_row[f"videos/{cam_key}/chunk_index"] = chunk_idx
                ep_row[f"videos/{cam_key}/file_index"]  = file_idx

            episode_rows.append(ep_row)
            dataset_from += n_frames
            total_frames += n_frames
            log(
                f"  épisode {ep_idx:04d} ({sess.name}) — {n_frames} frames"
                f" → chunk-{chunk_idx:03d}/file-{file_idx:03d}",
                "INFO",
            )

        except Exception as exc:
            ep_errors.append(f"ep {ep_idx:04d} ({sess.name}): {exc}")
            log(f"  ✗ épisode {ep_idx:04d} ({sess.name}): {exc}\n{traceback.format_exc()}", "ERROR")
            if data_path.exists():
                try:
                    data_path.unlink()
                except Exception:
                    pass

        if len(episode_rows) % batch_size == 0 and episode_rows:
            _flush_episode_parquet(meta_dir, episode_rows)

    # ── Métadonnées finales ───────────────────────────────────────────────────
    tasks_df = pd.DataFrame([{"task_index": 0, "task": "robot episode"}])
    (meta_dir / "tasks.parquet").write_bytes(tasks_df.to_parquet(index=False))
    (meta_dir / "stats.json").write_text("{}", encoding="utf-8")
    _flush_episode_parquet(meta_dir, episode_rows)

    total_chunks      = max(1, (total_episodes + chunks_size - 1) // chunks_size)
    successful_eps    = len(episode_rows)
    info = {
        "codebase_version":      "v3.0",
        "robot_type":            opts.robot_type,
        "total_episodes":        successful_eps,
        "total_frames":          total_frames,
        "total_tasks":           1,
        "total_videos":          successful_eps * len(camera_keys),
        "total_chunks":          total_chunks,
        "chunks_size":           chunks_size,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps":                   opts.fps,
        "splits":                {"train": f"0:{successful_eps}"},
        "data_path":             "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path":            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features":              features,
    }
    (meta_dir / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log(
        f"Dataset LeRobot v3.0 : {successful_eps} épisodes, {total_frames} frames,"
        f" {total_chunks} chunk(s)",
        "OK",
    )
    if ep_errors:
        log(f"⚠ {len(ep_errors)} épisode(s) ignoré(s) :\n" + "\n".join(ep_errors), "WARN")

    # ── Push HuggingFace Hub (optionnel) ──────────────────────────────────────
    if opts.push_to_hub and opts.hf_token and opts.repo_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=opts.hf_token)
            api.create_repo(repo_id=opts.repo_id, repo_type="dataset", exist_ok=True)
            api.upload_folder(folder_path=str(dataset_dir), repo_id=opts.repo_id, repo_type="dataset")
            log(f"✓ Poussé sur HF Hub : {opts.repo_id}", "OK")
        except Exception as e:
            log(f"✗ HF Hub push échoué : {e}", "WARN")

    return dataset_dir


# ─────────────────────────────────────────────────────────────────────────────
# Entrée standalone
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(
        description="Convertit des sessions classiques en dataset LeRobot v3.0"
    )
    parser.add_argument(
        "--sessions", nargs="+", required=True, metavar="PATH",
        help="Chemins vers les dossiers de sessions à convertir",
    )
    parser.add_argument(
        "--output", required=True, metavar="PATH",
        help="Répertoire de sortie (le dataset sera créé dedans)",
    )
    parser.add_argument("--dataset-name", default="robot_dataset")
    parser.add_argument("--robot-type",   default="so100")
    parser.add_argument("--fps",          type=int, default=30)
    parser.add_argument("--chunks-size",  type=int, default=1000)
    parser.add_argument("--batch-size",   type=int, default=5)
    parser.add_argument("--push-to-hub",  action="store_true")
    parser.add_argument("--hf-token",     default="")
    parser.add_argument("--repo-id",      default="")
    parser.add_argument(
        "--work-dir", default=None, metavar="PATH",
        help="Répertoire de travail temporaire (défaut : dossier temp système)",
    )
    args = parser.parse_args()

    sessions = [Path(p) for p in args.sessions]
    output   = Path(args.output)

    # Valider les sessions
    valid_sessions = []
    for sess in sessions:
        if not sess.exists():
            _log(f"SKIP {sess} — introuvable", "WARN")
            continue
        missing = [
            f"gripper_{side}_data.csv"
            for side in ("left", "right")
            if not (sess / f"gripper_{side}_data.csv").exists()
        ]
        if missing:
            _log(f"SKIP {sess.name} — manquants : {', '.join(missing)}", "WARN")
            continue
        valid_sessions.append(sess)

    if not valid_sessions:
        _log("Aucune session valide à convertir.", "ERROR")
        raise SystemExit(1)

    opts = ConvertOptions(
        dataset_name=args.dataset_name,
        robot_type=args.robot_type,
        fps=args.fps,
        chunks_size=args.chunks_size,
        batch_size=args.batch_size,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
        repo_id=args.repo_id,
    )

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = build_lerobot_dataset(valid_sessions, opts, work_dir)
        # Copier vers output
        dest = output / opts.dataset_name
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in dataset_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(dataset_dir)
                dst = dest / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
        _log(f"Dataset copié → {dest} ({n} fichiers)", "OK")
    else:
        # Construire directement dans output (pas de copie intermédiaire)
        output.mkdir(parents=True, exist_ok=True)
        dataset_dir = build_lerobot_dataset(valid_sessions, opts, output)
        _log(f"Dataset disponible → {dataset_dir}", "OK")


if __name__ == "__main__":
    main()
