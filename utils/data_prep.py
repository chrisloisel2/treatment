#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_prep.py — Préparation des données : rôles trackers, rotation vidéos, vérification labels.

Regroupe :
  - fix.py               : inférence des rôles {head, left, right} depuis tracker_positions.csv
  - rotate_videos.py     : rotation 180° des vidéos (FFmpeg hflip+vflip, libx264 crf=18)
  - verify_video_labels.py : vérification et correction des labels caméra (3 niveaux)

Usage standalone :
    # Corriger les rôles trackers
    python data_prep.py fix tracker_positions.csv
    python data_prep.py fix --all /path/to/sessions

    # Rotation 180°
    python data_prep.py rotate /path/to/session [--sides head left right] [--force]

    # Vérifier les labels vidéo
    python data_prep.py verify /path/to/session [--apply] [--yes]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import correlate


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — Inférence des rôles trackers (fix.py)
# ══════════════════════════════════════════════════════════════════════════════

EPS_FIX = 1e-9

TRACKER_SUFFIXES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
TRACKER_ROLES    = ["head", "left", "right"]


@dataclass
class TrackerBlock:
    gid: str
    role_hint: Optional[str]
    pos: np.ndarray       # (N, 3)
    quat: np.ndarray      # (N, 4)
    col_names: List[str]


@dataclass
class HeadInference:
    head_gid: str
    world_up: np.ndarray
    up_axis_index: int
    up_axis_sign: int
    score: float
    details: Dict


@dataclass
class LRInference:
    left_gid: str
    right_gid: str
    head_local_up_axis: int
    head_local_up_sign: int
    head_local_right_axis: int
    head_local_right_sign: int
    score: float
    details: Dict


# ── Helpers numériques ────────────────────────────────────────────────────────

def _robust_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size > 0 else float("nan")


def _robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))) + EPS_FIX)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS_FIX)


def _quat_to_rotmat(qwqxqyqz: np.ndarray) -> np.ndarray:
    q = np.asarray(qwqxqyqz, dtype=float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), EPS_FIX)
    w, x, y, z = [q[..., i] for i in range(4)]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1.0 - 2.0 * (y*y + z*z)
    R[..., 0, 1] = 2.0 * (x*y - w*z)
    R[..., 0, 2] = 2.0 * (x*z + w*y)
    R[..., 1, 0] = 2.0 * (x*y + w*z)
    R[..., 1, 1] = 1.0 - 2.0 * (x*x + z*z)
    R[..., 1, 2] = 2.0 * (y*z - w*x)
    R[..., 2, 0] = 2.0 * (x*z - w*y)
    R[..., 2, 1] = 2.0 * (y*z + w*x)
    R[..., 2, 2] = 1.0 - 2.0 * (x*x + y*y)
    return R


# ── Détection des blocs tracker ───────────────────────────────────────────────

def find_tracker_blocks(df: pd.DataFrame) -> List[TrackerBlock]:
    df_num = df.copy()
    for c in df_num.columns:
        if not pd.api.types.is_numeric_dtype(df_num[c]):
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    named = _try_named_blocks(df, df_num)
    if named:
        return named
    return _fallback_anonymous_blocks(df_num)


def _try_named_blocks(df: pd.DataFrame, df_num: pd.DataFrame) -> List[TrackerBlock]:
    blocks = []
    for i, role in enumerate(TRACKER_ROLES):
        found = None
        for prefix in (f"tracker_{role}_", f"{role}_"):
            cols = [f"{prefix}{s}" for s in TRACKER_SUFFIXES]
            if all(c in df.columns for c in cols):
                found = cols
                break
        if found is None:
            return []
        data = df_num[found].to_numpy(dtype=float)
        if data.shape[0] < 10:
            return []
        blocks.append(TrackerBlock(
            gid=f"G{i}", role_hint=role,
            pos=data[:, 0:3], quat=data[:, 3:7], col_names=found,
        ))
    return blocks


def _fallback_anonymous_blocks(df_num: pd.DataFrame) -> List[TrackerBlock]:
    numeric_cols = [c for c in df_num.columns if pd.api.types.is_numeric_dtype(df_num[c])]
    if len(numeric_cols) < 21:
        raise ValueError("Pas assez de colonnes numériques pour 3 trackers x 7 variables.")
    tail = numeric_cols[-21:]
    arr  = df_num[tail].to_numpy(dtype=float)
    if arr.shape[0] < 10:
        raise ValueError("Pas assez de lignes pour une inférence robuste.")
    blocks = []
    for i in range(3):
        s = i * 7
        blocks.append(TrackerBlock(
            gid=f"G{i}", role_hint=None,
            pos=arr[:, s:s+3], quat=arr[:, s+3:s+7], col_names=tail[s:s+7],
        ))
    return blocks


# ── Inférence tête ────────────────────────────────────────────────────────────

def _score_head_candidate(blocks: List[TrackerBlock], axis: int, sign: int) -> Tuple[str, float, Dict]:
    vals = np.stack([sign * b.pos[:, axis] for b in blocks], axis=1)
    winners = np.argmax(vals, axis=1)
    scores = []
    details_all = {}
    for k, b in enumerate(blocks):
        win_rate = float(np.mean(winners == k))
        v_self = vals[:, k]
        others_mean = np.mean(np.delete(vals, k, axis=1), axis=1)
        gap = v_self - others_mean
        med_gap = _robust_median(gap)
        mad_gap = _robust_mad(gap)
        if not np.isfinite(mad_gap) or mad_gap < EPS_FIX:
            normalized_gap = 20.0 if med_gap > 0 else -20.0
        else:
            normalized_gap = float(np.clip(med_gap / mad_gap, -50.0, 50.0))
        score = 2.5 * win_rate + 2.0 * float(np.mean(gap > 0)) + 1.5 * max(0.0, normalized_gap)
        scores.append(score)
        details_all[b.gid] = {"win_rate": win_rate, "median_gap": med_gap, "score": score}
    best_idx = int(np.argmax(scores))
    return blocks[best_idx].gid, float(scores[best_idx]), details_all[blocks[best_idx].gid]


def infer_head(blocks: List[TrackerBlock]) -> HeadInference:
    best = None
    for axis in range(3):
        for sign in (-1, +1):
            gid, score, details = _score_head_candidate(blocks, axis, sign)
            if best is None or score > best["score"]:
                best = {"gid": gid, "axis": axis, "sign": sign, "score": score, "details": details}
    assert best is not None
    world_up = np.zeros(3, dtype=float)
    world_up[best["axis"]] = float(best["sign"])
    return HeadInference(
        head_gid=best["gid"],
        world_up=world_up,
        up_axis_index=best["axis"],
        up_axis_sign=best["sign"],
        score=float(best["score"]),
        details=best["details"],
    )


# ── Inférence gauche / droite ─────────────────────────────────────────────────

def _get_block(blocks: List[TrackerBlock], gid: str) -> TrackerBlock:
    for b in blocks:
        if b.gid == gid:
            return b
    raise KeyError(gid)


def _average_alignment(vs: np.ndarray, ref: np.ndarray) -> float:
    dots = np.sum(_unit(vs) * _unit(ref.reshape(1, 3)), axis=1)
    return float(np.mean(np.abs(dots)))


def _infer_head_local_up(head_block: TrackerBlock, world_up: np.ndarray) -> Tuple[int, int, float]:
    R = _quat_to_rotmat(head_block.quat)
    best = None
    for local_axis in range(3):
        axis_world = R[:, :, local_axis]
        for sign in (-1, +1):
            score = _average_alignment(sign * axis_world, world_up)
            if best is None or score > best[2]:
                best = (local_axis, sign, score)
    assert best is not None
    return best


def infer_left_right(blocks: List[TrackerBlock], head_info: HeadInference) -> LRInference:
    head = _get_block(blocks, head_info.head_gid)
    hands = [b for b in blocks if b.gid != head_info.head_gid]
    A, B = hands
    R = _quat_to_rotmat(head.quat)
    up_local_axis, up_local_sign, up_score = _infer_head_local_up(head, head_info.world_up)
    remaining_axes = [ax for ax in (0, 1, 2) if ax != up_local_axis]

    rel_A_world = A.pos - head.pos
    rel_B_world = B.pos - head.pos
    rel_A_local = np.einsum("nij,ni->nj", R, rel_A_world)
    rel_B_local = np.einsum("nij,ni->nj", R, rel_B_world)

    best = None
    for right_local_axis in remaining_axes:
        proj_A = rel_A_local[:, right_local_axis]
        proj_B = rel_B_local[:, right_local_axis]
        diff   = proj_A - proj_B
        mad_diff = _robust_mad(diff)
        med_diff_abs = np.nanmedian(np.abs(diff))
        if not np.isfinite(mad_diff) or mad_diff < EPS_FIX:
            sep = 20.0 if med_diff_abs > 0 else 0.0
        else:
            sep = float(np.clip(_robust_median(np.abs(diff)) / mad_diff, 0.0, 50.0))
        consistent = float(np.mean(np.abs(diff) > (0.25 * med_diff_abs + EPS_FIX)))
        med_diff   = _robust_median(diff)
        right_gid, left_gid = (A.gid, B.gid) if med_diff >= 0 else (B.gid, A.gid)
        score = 2.0 * max(0.0, sep) + 1.0 * consistent + 0.5 * up_score
        right_axis_world = R[:, :, right_local_axis]
        vertical_alignment = _average_alignment(right_axis_world, head_info.world_up)
        score -= 2.0 * vertical_alignment
        item = {
            "left_gid": left_gid, "right_gid": right_gid,
            "right_local_axis": right_local_axis, "right_local_sign": +1,
            "score": float(score),
            "details": {
                "up_alignment_score":                    float(up_score),
                "right_hand_separation_score":           float(sep),
                "right_hand_consistency":                float(consistent),
                "right_axis_vertical_alignment_penalty": float(vertical_alignment),
                "median_projection_diff_A_minus_B":      float(med_diff),
            },
        }
        if best is None or item["score"] > best["score"]:
            best = item

    assert best is not None
    return LRInference(
        left_gid=best["left_gid"],
        right_gid=best["right_gid"],
        head_local_up_axis=up_local_axis,
        head_local_up_sign=up_local_sign,
        head_local_right_axis=best["right_local_axis"],
        head_local_right_sign=best["right_local_sign"],
        score=best["score"],
        details=best["details"],
    )


def _global_confidence(head_info: HeadInference, lr_info: LRInference) -> float:
    h_score  = float(np.clip(head_info.score,  -50.0, 50.0))
    lr_score = float(np.clip(lr_info.score,    -50.0, 50.0))
    h  = 1.0 / (1.0 + math.exp(-(h_score  - 4.0)))
    lr = 1.0 / (1.0 + math.exp(-(lr_score - 2.5)))
    return float(np.clip(0.55 * h + 0.45 * lr, 0.0, 1.0))


def _build_mapping(head_info: HeadInference, lr_info: LRInference) -> Dict[str, str]:
    return {
        head_info.head_gid: "head",
        lr_info.left_gid:   "left",
        lr_info.right_gid:  "right",
    }


def _is_already_correct(blocks: List[TrackerBlock], mapping: Dict[str, str]) -> bool:
    for b in blocks:
        if b.role_hint is None:
            return False
        if mapping[b.gid] != b.role_hint:
            return False
    return True


def infer_roles(csv_path) -> dict:
    """API publique : inférence des rôles tracker. Retourne un dict avec mapping, confidence, swaps, etc."""
    csv_path = Path(csv_path)
    result: dict = {
        "csv_path":      str(csv_path),
        "confidence":    0.0,
        "swapped":       False,
        "mapping":       {},
        "role_hints":    {},
        "swaps":         [],
        "world_up_axis": -1,
        "world_up_sign": 0,
        "error":         None,
    }
    try:
        df        = pd.read_csv(csv_path)
        blocks    = find_tracker_blocks(df)
        head_info = infer_head(blocks)
        lr_info   = infer_left_right(blocks, head_info)
        mapping   = _build_mapping(head_info, lr_info)
        conf      = _global_confidence(head_info, lr_info)
        already_ok = _is_already_correct(blocks, mapping)
        swaps = [(b.role_hint, mapping[b.gid]) for b in blocks
                 if b.role_hint is not None and mapping[b.gid] != b.role_hint]
        result["confidence"]    = conf
        result["swapped"]       = not already_ok
        result["mapping"]       = mapping
        result["role_hints"]    = {b.gid: (b.role_hint or "?") for b in blocks}
        result["swaps"]         = swaps
        result["world_up_axis"] = head_info.up_axis_index
        result["world_up_sign"] = head_info.up_axis_sign
    except Exception as e:
        result["error"] = str(e)
    return result


def rewrite_csv(df: pd.DataFrame, blocks: List[TrackerBlock], mapping: Dict[str, str],
                output_path: Path) -> None:
    """Réécrit le CSV avec les bons rôles. Crée un backup .bak_syncml avant écriture."""
    if output_path.exists():
        bak = output_path.with_suffix(output_path.suffix + ".bak_syncml")
        if not bak.exists():
            shutil.copy2(output_path, bak)

    out = df.copy()
    role_to_block = {mapping[b.gid]: b for b in blocks}
    prefix = "tracker_" if (blocks[0].col_names and blocks[0].col_names[0].startswith("tracker_")) else ""

    for target_role in TRACKER_ROLES:
        src_block = role_to_block[target_role]
        target_cols = [f"{prefix}{target_role}_{s}" for s in TRACKER_SUFFIXES]
        src_data = np.hstack([src_block.pos, src_block.quat])
        for j, col in enumerate(target_cols):
            if col in out.columns:
                out[col] = src_data[:, j]
            else:
                out.insert(list(out.columns).index(src_block.col_names[0]), col, src_data[:, j])

    out.to_csv(output_path, index=False)


def _process_fix_file(csv_path: Path, output_path: Optional[Path], dry_run: bool,
                      verbose: bool, min_confidence: float = 0.65) -> dict:
    label = str(csv_path)
    entry = {
        "path": label, "status": "error", "confidence": 0.0,
        "swapped": False, "swaps": [], "modified": False, "error": None,
    }
    try:
        df = pd.read_csv(csv_path)
        df.attrs["_source_path"] = str(csv_path)
    except Exception as e:
        print(f"[ERREUR] {label}: lecture impossible: {e}", file=sys.stderr)
        entry["error"] = str(e)
        return entry

    try:
        blocks    = find_tracker_blocks(df)
        head_info = infer_head(blocks)
        lr_info   = infer_left_right(blocks, head_info)
        mapping   = _build_mapping(head_info, lr_info)
        conf      = _global_confidence(head_info, lr_info)
    except Exception as e:
        print(f"[ERREUR] {label}: inférence échouée: {e}", file=sys.stderr)
        entry["error"] = str(e)
        return entry

    already_ok = _is_already_correct(blocks, mapping)
    swaps = [(b.role_hint, mapping[b.gid]) for b in blocks
             if b.role_hint is not None and mapping[b.gid] != b.role_hint]

    entry["confidence"] = conf
    entry["swapped"]    = not already_ok
    entry["swaps"]      = swaps

    conf_tag = "confiance élevée" if conf >= 0.80 else ("confiance moyenne" if conf >= 0.65 else "confiance FAIBLE")
    status_tag = "[OK]" if already_ok else "[SWAP]"
    print(f"{status_tag} {label}  conf={conf:.3f} ({conf_tag})")

    if verbose:
        for b in blocks:
            inferred = mapping[b.gid]
            current  = b.role_hint or "?"
            arrow    = "✓" if inferred == current else f"{current} → {inferred}"
            print(f"       {b.gid}: {arrow}")

    if conf < min_confidence:
        print(f"  [AVERTISSEMENT] Confiance {conf:.3f} < {min_confidence} — résultat ignoré.", file=sys.stderr)
        entry["status"] = "skipped_low_confidence"
        return entry

    entry["status"] = "ok"
    if already_ok:
        return entry
    if dry_run:
        print("  [DRY-RUN] Pas de modification.")
        return entry

    dest = output_path if output_path else csv_path
    try:
        rewrite_csv(df, blocks, mapping, dest)
        print(f"  Sauvegardé: {dest}")
        entry["modified"] = True
    except Exception as e:
        print(f"[ERREUR] {label}: écriture impossible: {e}", file=sys.stderr)
        entry["error"] = str(e)
        entry["status"] = "error"
    return entry


def find_all_tracker_csvs(root: Path) -> List[Path]:
    found = []
    seen  = set()
    def _walk(path: Path, depth: int):
        if depth > 3:
            return
        csv = path / "tracker_positions.csv"
        if csv.exists() and csv not in seen:
            found.append(csv)
            seen.add(csv)
        try:
            for sub in sorted(path.iterdir()):
                if sub.is_dir() and not sub.name.startswith("."):
                    _walk(sub, depth + 1)
        except PermissionError:
            pass
    _walk(root, 0)
    return found


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — Rotation 180° des vidéos (rotate_videos.py)
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_SIDES    = ("head", "left", "right")
ROTATE_MARKER  = ".rotate_done"


def _ffmpeg_available() -> Optional[str]:
    for candidate in ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        # shutil.which fonctionne pour les noms simples ; pour les chemins absolus on vérifie l'existence
        if os.path.isabs(candidate):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        elif shutil.which(candidate):
            return candidate
    return None


def _probe_rotation(ffmpeg: str, mp4: Path) -> int:
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if not shutil.which(ffprobe):
        ffprobe = "ffprobe"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams", str(mp4)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(r.stdout)
        for stream in data.get("streams", []):
            tags = stream.get("tags", {})
            rot  = tags.get("rotate", tags.get("Rotate", "0"))
            try:
                return int(rot)
            except (ValueError, TypeError):
                pass
            for sd in stream.get("side_data_list", []):
                if sd.get("side_data_type") == "Display Matrix":
                    try:
                        return int(sd.get("rotation", 0))
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return 0


def rotate_video_180(ffmpeg: str, src: Path, dst: Path, log=None) -> bool:
    """Retourne src à 180° et écrit le résultat dans dst. Retourne True si succès."""
    def _log(msg, level="INFO"):
        if log:
            log(msg, level)
        else:
            print(f"[{level}] {msg}")

    tmp = dst.with_suffix(".tmp_rotate.mp4")
    cmd = [
        ffmpeg, "-y",
        "-i", str(src),
        "-vf", "hflip,vflip",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-metadata:s:v:0", "rotate=0",
        "-movflags", "+faststart",
        str(tmp),
    ]
    _log(f"Rotation 180° : {src.name} …")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        _log(f"Timeout rotation {src.name}", "ERROR")
        tmp.unlink(missing_ok=True)
        return False
    except Exception as e:
        _log(f"Erreur FFmpeg {src.name}: {e}", "ERROR")
        tmp.unlink(missing_ok=True)
        return False

    if result.returncode != 0:
        err = result.stderr.strip()[-600:]
        _log(f"FFmpeg erreur ({src.name}): {err}", "ERROR")
        tmp.unlink(missing_ok=True)
        return False

    if not tmp.exists() or tmp.stat().st_size < 1024:
        _log(f"Fichier de sortie invalide : {tmp}", "ERROR")
        tmp.unlink(missing_ok=True)
        return False

    tmp.rename(dst)
    _log(f"Rotation OK : {dst.name} ({time.time()-t0:.1f}s)", "OK")
    return True


def rotate_session_videos(
    session_dir: Path,
    sides: List[str] = list(VIDEO_SIDES),
    force: bool = False,
    log=None,
) -> dict:
    """
    Applique la rotation 180° à toutes les vidéos d'une session.
    Sauvegarde les originaux en <side>.mp4.bak_rotate.
    Écrit .rotate_done après succès total.
    """
    def _log(msg, level="INFO"):
        if log:
            log(msg, level)
        else:
            print(f"[{level}] {msg}")

    vid_dir = session_dir / "videos"
    marker  = vid_dir / ROTATE_MARKER

    if marker.exists() and not force:
        _log("Rotation déjà effectuée (.rotate_done présent) — skip", "INFO")
        return {"rotated": [], "skipped": list(sides), "errors": [], "already_done": True}

    ffmpeg = _ffmpeg_available()
    if ffmpeg is None:
        raise RuntimeError("FFmpeg introuvable. Installez-le avec : brew install ffmpeg")
    if not vid_dir.exists():
        raise FileNotFoundError(f"Dossier vidéo absent : {vid_dir}")

    rotated = []
    skipped = []
    errors  = []

    # Préparer les backups d'abord (séquentiel, rapide)
    sides_to_rotate = []
    for side in sides:
        mp4 = vid_dir / f"{side}.mp4"
        bak = vid_dir / f"{side}.mp4.bak_rotate"
        if not mp4.exists():
            _log(f"{side}.mp4 absent — skip", "WARN")
            skipped.append(side)
            continue
        if not bak.exists():
            _log(f"Backup : {mp4.name} → {bak.name}")
            shutil.copyfile(mp4, bak)
            try:
                os.chmod(bak, 0o644)
            except OSError:
                pass
        else:
            _log(f"Backup existant conservé : {bak.name}", "INFO")
            try:
                os.chmod(bak, 0o644)
            except OSError:
                pass
        sides_to_rotate.append(side)

    # Rotation en parallèle (une vidéo par thread)
    def _rotate_one(side):
        mp4 = vid_dir / f"{side}.mp4"
        bak = vid_dir / f"{side}.mp4.bak_rotate"
        ok = rotate_video_180(ffmpeg, bak, mp4, log=log)
        if not ok:
            shutil.copyfile(bak, mp4)
            _log(f"Restauration backup {side}.mp4 après échec", "WARN")
        return side, ok

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=len(sides_to_rotate) or 1) as pool:
        futures = {pool.submit(_rotate_one, s): s for s in sides_to_rotate}
        for fut in as_completed(futures):
            side, ok = fut.result()
            if ok:
                rotated.append(side)
            else:
                errors.append(side)

    if not errors:
        marker.write_text(
            json.dumps({
                "rotated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sides": rotated,
                "ffmpeg": ffmpeg,
            }, indent=2),
            encoding="utf-8",
        )
        _log(f"Marqueur .rotate_done écrit ({len(rotated)} vidéo(s))", "OK")
    else:
        _log(f"Erreurs sur {errors} — marqueur NON écrit", "WARN")

    return {"rotated": rotated, "skipped": skipped, "errors": errors, "already_done": False}


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 3 — Vérification labels vidéo (verify_video_labels.py)
# ══════════════════════════════════════════════════════════════════════════════

TRACKER_MIN_ROWS          = 10
TRACKER_SEPARATION_MIN_M  = 0.05
FISHEYE_THRESHOLD         = 0.52
FISHEYE_MIN_FRAMES        = 5
MOTION_FRAMES             = 120
MOTION_LR_CORR_MIN        = 0.35
MOTION_HEAD_CORR_MAX      = 0.80
CONFIDENCE_FULL           = 1.00
CONFIDENCE_HIGH           = 0.90
CONFIDENCE_LOW            = 0.60


@dataclass
class TrackerAssignment:
    head_tracker_id:  str = ""
    left_tracker_id:  str = ""
    right_tracker_id: str = ""
    head_mean_pos:    Tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_mean_pos:    Tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_mean_pos:   Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vertical_separation_m:    float = 0.0
    horizontal_separation_m:  float = 0.0
    confidence: float = 0.0
    ok: bool = False
    details: List[str] = field(default_factory=list)


@dataclass
class FisheyeResult:
    label: str
    score: float = 0.0
    is_fisheye: bool = False
    details: List[str] = field(default_factory=list)


@dataclass
class MotionResult:
    lr_correlation: float = 0.0
    lh_correlation: float = 0.0
    rh_correlation: float = 0.0
    left_right_consistent: bool = False
    head_distinct: bool = False
    details: List[str] = field(default_factory=list)


@dataclass
class CameraVerdict:
    declared_label:  str
    file_path:       str
    predicted_label: str = ""
    confidence:      float = 0.0
    label_correct:   bool = False
    warnings:        List[str] = field(default_factory=list)
    errors:          List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    session_dir:     str
    global_ok:       bool = True
    confidence:      float = 0.0
    tracker:         Optional[TrackerAssignment] = None
    fisheye:         Dict[str, FisheyeResult] = field(default_factory=dict)
    motion:          Optional[MotionResult] = None
    verdicts:        List[CameraVerdict] = field(default_factory=list)
    recommended_mapping: Dict[str, str] = field(default_factory=dict)
    summary:         str = ""
    safe_mode:       bool = True


# ── Niveau 1 — Géométrie tracker 3D ──────────────────────────────────────────

def analyze_trackers(csv_path: str) -> TrackerAssignment:
    """
    Identifie les rôles head/left/right des trackers depuis tracker_positions.csv.

    Utilise l'algorithme robuste de fix_tracker_labels :
    - Head : tracker avec hauteur Y la plus élevée (z-score statistique)
    - Left/Right : projection de la POSITION MOYENNE relative à la tête sur
      l'axe X MOYEN du head (quaternion) → robuste à toutes les orientations.

    Cette approche remplace l'ancienne (world X axis) qui échouait à ~50%
    quand le joueur ne fait pas face à une direction fixe.
    """
    result = TrackerAssignment()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result.details.append(f"ERREUR lecture CSV tracker : {e}")
        return result

    if len(df) < TRACKER_MIN_ROWS:
        result.details.append(f"Trop peu de lignes dans le tracker CSV ({len(df)})")
        return result

    # Déléguer à fix_tracker_labels (algorithme validé à 93%+)
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _root = _Path(csv_path).resolve().parent.parent
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from fix.fix_tracker_labels import (
            _load_blocks, _test_height, _test_centrality,
            _test_mobility, _test_lateral, _consensus,
        )
        blocks = _load_blocks(df)
        if blocks is None:
            result.details.append("Colonnes tracker manquantes dans le CSV")
            return result

        t1 = _test_height(blocks)
        t2 = _test_centrality(blocks)
        t3 = _test_mobility(blocks)
        t4 = _test_lateral(blocks, t1.head_vote)
        predicted, agree_count, certain = _consensus([t1, t2, t3, t4])
        # predicted = {role: csv_label}  ex: {head:'head', left:'right', right:'left'}

    except Exception as e:
        result.details.append(f"Erreur fix_tracker_labels : {e}")
        return result

    # Remplir TrackerAssignment depuis le résultat
    head_csv  = predicted.get("head",  "head")
    left_csv  = predicted.get("left",  "left")
    right_csv = predicted.get("right", "right")

    result.head_tracker_id  = head_csv
    result.left_tracker_id  = left_csv
    result.right_tracker_id = right_csv

    # Positions moyennes
    for role, csv_label, attr in [
        ("head",  head_csv,  "head_mean_pos"),
        ("left",  left_csv,  "left_mean_pos"),
        ("right", right_csv, "right_mean_pos"),
    ]:
        try:
            x = float(np.median(df[f"tracker_{csv_label}_x"].dropna()))
            y = float(np.median(df[f"tracker_{csv_label}_y"].dropna()))
            z = float(np.median(df[f"tracker_{csv_label}_z"].dropna()))
            setattr(result, attr, (x, y, z))
        except Exception:
            pass

    # Séparations
    try:
        head_y  = result.head_mean_pos[1]
        left_y  = result.left_mean_pos[1]
        right_y = result.right_mean_pos[1]
        vert_sep  = abs(head_y - max(left_y, right_y))
        # Séparation latérale sur l'axe X local du head (projection moyenne)
        block_h = next(b for b in blocks if b[0] == head_csv)
        block_l = next(b for b in blocks if b[0] == left_csv)
        block_r = next(b for b in blocks if b[0] == right_csv)
        _, hp, hq = block_h
        _, lp, _ = block_l
        _, rp, _ = block_r

        def _qrot(q):
            q = q.astype(float)
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
            R = np.empty((len(q),3,3))
            R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-z*w); R[:,0,2]=2*(x*z+y*w)
            R[:,1,0]=2*(x*y+z*w);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-x*w)
            R[:,2,0]=2*(x*z-y*w);   R[:,2,1]=2*(y*z+x*w);   R[:,2,2]=1-2*(x*x+y*y)
            return R

        R = _qrot(hq)
        mean_x = np.mean(R[:,:,0], axis=0); mean_x /= np.linalg.norm(mean_x)+1e-12
        n = min(len(hp), len(lp), len(rp))
        proj_l = float(np.dot(np.mean(lp[:n]-hp[:n], axis=0), mean_x))
        proj_r = float(np.dot(np.mean(rp[:n]-hp[:n], axis=0), mean_x))
        horiz_sep = abs(proj_r - proj_l)

        result.vertical_separation_m   = vert_sep
        result.horizontal_separation_m = horiz_sep
    except Exception:
        vert_sep  = 0.0
        horiz_sep = 0.0

    issues = []
    if result.vertical_separation_m   < TRACKER_SEPARATION_MIN_M:
        issues.append(f"Séparation verticale head/latéraux faible ({result.vertical_separation_m*100:.1f} cm)")
    if result.horizontal_separation_m < TRACKER_SEPARATION_MIN_M:
        issues.append(f"Séparation latérale left/right faible ({result.horizontal_separation_m*100:.1f} cm)")

    tracker_label_matches = (head_csv == "head" and left_csv == "left" and right_csv == "right")

    result.details.append(
        f"Trackers identifiés — head←{head_csv}  left←{left_csv}  right←{right_csv}  "
        f"(accord {agree_count}/4 tests, certain={certain})"
    )
    result.details.append(
        f"Positions — "
        f"head=({result.head_mean_pos[0]:.3f},{result.head_mean_pos[1]:.3f},{result.head_mean_pos[2]:.3f})  "
        f"left=({result.left_mean_pos[0]:.3f},{result.left_mean_pos[1]:.3f},{result.left_mean_pos[2]:.3f})  "
        f"right=({result.right_mean_pos[0]:.3f},{result.right_mean_pos[1]:.3f},{result.right_mean_pos[2]:.3f})"
    )
    result.details.append(f"Séparation verticale : {result.vertical_separation_m*100:.1f} cm")
    result.details.append(f"Séparation latérale  : {result.horizontal_separation_m*100:.1f} cm")
    if issues:
        result.details += [f"⚠ {i}" for i in issues]

    if certain and not issues:
        result.confidence = 1.0
        result.details.append("✓ Labels trackers certains (4 tests concordants)")
    elif certain and issues:
        result.confidence = 0.85
        result.details.append("⚠ Labels certains mais séparations faibles")
    elif agree_count >= 3:
        result.confidence = 0.80
        result.details.append(f"~ Labels probables ({agree_count}/4 tests concordants)")
    else:
        result.confidence = 0.55
        result.details.append(f"⚠ Confiance faible ({agree_count}/4 tests)")

    if not tracker_label_matches:
        result.details.append(
            f"! Labels CSV incorrects : head→{head_csv}, left→{left_csv}, right→{right_csv}"
        )
        # La confiance reste celle calculée ci-dessus — c'est un résultat valide

    result.ok = (result.confidence >= 0.70)
    return result


# ── Utilitaires vidéo ─────────────────────────────────────────────────────────

def sample_frames(path: str, n: int = 10,
                  start_pct: float = 0.10, end_pct: float = 0.90) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(int(total * start_pct), int(total * end_pct) - 1, n, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def extract_motion_signal(path: str, max_frames: int = MOTION_FRAMES) -> List[float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    signal = []
    prev   = None
    count  = 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 240))
        if prev is not None:
            diff = cv2.absdiff(prev, small).astype(np.float32)
            signal.append(float(diff.mean()))
        prev  = small
        count += 1
    cap.release()
    return signal


def video_dimensions(path: str) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0.0, 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    fc   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, fc


# ── Niveau 2 — Fisheye ────────────────────────────────────────────────────────

def _vignette_ratio(frames: List[np.ndarray]) -> Tuple[float, str]:
    ratios = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        r_max = min(cx, cy)
        Y, X = np.ogrid[:h, :w]
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        m_center = R < r_max * 0.22
        m_border = R > r_max * 0.78
        if m_center.sum() > 10 and m_border.sum() > 10:
            ratios.append(float(gray[m_center].mean()) / max(float(gray[m_border].mean()), 1.0))
    if not ratios:
        return 1.0, "vignetage non mesurable"
    avg = float(np.median(ratios))
    if avg > 1.5:
        return 0.85, f"vignetage prononcé (centre {avg:.2f}× plus lumineux)"
    if avg > 1.2:
        return 0.55, f"vignetage modéré (ratio={avg:.2f})"
    return 0.15, f"pas de vignetage (ratio={avg:.2f})"


def _sharpness_ratio(frames: List[np.ndarray]) -> Tuple[float, str]:
    ratios = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        m = min(h, w) // 6

        def lap(patch):
            return float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size > 0 else 0.0

        center  = gray[h//2 - m: h//2 + m, w//2 - m: w//2 + m]
        corners = [gray[:2*m, :2*m], gray[:2*m, -2*m:], gray[-2*m:, :2*m], gray[-2*m:, -2*m:]]
        c_sharp = lap(center)
        k_sharp = float(np.mean([lap(c) for c in corners]))
        if k_sharp > 0:
            ratios.append(c_sharp / k_sharp)
    if not ratios:
        return 1.0, "netteté non mesurable"
    avg = float(np.median(ratios))
    if avg > 2.5:
        return 0.85, f"centre net, coins flous (ratio={avg:.2f}) → fisheye"
    if avg > 1.4:
        return 0.55, f"légère différence centre/coins (ratio={avg:.2f})"
    return 0.15, f"netteté uniforme centre/coins (ratio={avg:.2f})"


def _barrel_distortion(frames: List[np.ndarray]) -> Tuple[float, str]:
    rel_lens = []
    for frame in frames:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=50, minLineLength=40, maxLineGap=8)
        if lines is not None and len(lines) >= 5:
            h, w = gray.shape
            diag = np.sqrt(h**2 + w**2)
            avg_len = float(np.mean([
                np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2)
                for l in lines
            ]))
            rel_lens.append(avg_len / diag)
    if not rel_lens:
        return 0.35, "distorsion non mesurable (pas de lignes)"
    avg = float(np.median(rel_lens))
    if avg < 0.09:
        return 0.80, f"lignes très courtes (len_rel={avg:.3f}) → distorsion barrel"
    if avg < 0.16:
        return 0.50, f"lignes courtes-moyennes (len_rel={avg:.3f})"
    return 0.15, f"longues lignes droites (len_rel={avg:.3f}) → pas de distorsion"


def _aspect_ratio_score(w: int, h: int) -> Tuple[float, str]:
    ar = w / max(h, 1)
    if 0.85 <= ar <= 1.20:
        return 0.70, f"rapport W/H={ar:.2f} (carré → fisheye)"
    if 0.70 <= ar <= 1.50:
        return 0.40, f"rapport W/H={ar:.2f} (neutre)"
    return 0.10, f"rapport W/H={ar:.2f} (très rectangulaire → pas fisheye)"


def analyze_fisheye(label: str, path: str) -> FisheyeResult:
    result = FisheyeResult(label=label)
    w, h, fps, fc = video_dimensions(path)
    if w == 0:
        result.details.append("Vidéo illisible")
        return result
    frames = sample_frames(path, n=12)
    if len(frames) < FISHEYE_MIN_FRAMES:
        result.details.append(f"Trop peu de frames ({len(frames)})")
        return result

    scores = []
    s, d = _aspect_ratio_score(w, h);  scores.append(s); result.details.append(f"Aspect ratio   : {d}")
    s, d = _vignette_ratio(frames);    scores.append(s); result.details.append(f"Vignetage      : {d}")
    s, d = _sharpness_ratio(frames);   scores.append(s); result.details.append(f"Netteté        : {d}")
    s, d = _barrel_distortion(frames); scores.append(s); result.details.append(f"Distorsion     : {d}")

    result.score = float(np.mean(scores))
    result.is_fisheye = result.score >= FISHEYE_THRESHOLD
    result.details.append(
        f"→ Score fisheye = {result.score:.3f}  ({'FISHEYE' if result.is_fisheye else 'pas fisheye'})"
    )
    return result


# ── Niveau 3 — Corrélation de mouvement ──────────────────────────────────────

def _xcorr(a: List[float], b: List[float]) -> float:
    if len(a) < 5 or len(b) < 5:
        return 0.0
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    va = (va - va.mean()) / (va.std() + 1e-9)
    vb = (vb - vb.mean()) / (vb.std() + 1e-9)
    n  = min(len(va), len(vb))
    return float(np.corrcoef(va[:n], vb[:n])[0, 1])


def analyze_motion(paths: Dict[str, str]) -> MotionResult:
    result  = MotionResult()
    signals: Dict[str, List[float]] = {}
    for label, path in paths.items():
        sig = extract_motion_signal(path)
        if sig:
            signals[label] = sig

    if "left" in signals and "right" in signals:
        result.lr_correlation = _xcorr(signals["left"], signals["right"])
    if "left" in signals and "head" in signals:
        result.lh_correlation = _xcorr(signals["left"], signals["head"])
    if "right" in signals and "head" in signals:
        result.rh_correlation = _xcorr(signals["right"], signals["head"])

    result.left_right_consistent = result.lr_correlation >= MOTION_LR_CORR_MIN
    result.head_distinct = (
        result.lh_correlation <= MOTION_HEAD_CORR_MAX and
        result.rh_correlation <= MOTION_HEAD_CORR_MAX
    )
    result.details.append(f"Corrélation left ↔ right : {result.lr_correlation:+.3f}  (seuil ≥ {MOTION_LR_CORR_MIN})")
    result.details.append(f"Corrélation left ↔ head  : {result.lh_correlation:+.3f}")
    result.details.append(f"Corrélation right ↔ head : {result.rh_correlation:+.3f}")
    if result.left_right_consistent:
        result.details.append("✓ Left et right ont un mouvement cohérent")
    else:
        result.details.append(f"⚠ Left et right peu corrélées ({result.lr_correlation:.3f})")
    return result


# ── Verdicts ──────────────────────────────────────────────────────────────────

def compute_verdicts(
    paths:    Dict[str, str],
    tracker:  TrackerAssignment,
    fisheye:  Dict[str, FisheyeResult],
    motion:   MotionResult,
) -> Tuple[List[CameraVerdict], Dict[str, str], float]:
    tracker_map: Dict[str, str] = {}
    if tracker.ok:
        tracker_map = {
            tracker.head_tracker_id:  "head",
            tracker.left_tracker_id:  "left",
            tracker.right_tracker_id: "right",
        }

    verdicts   = []
    agreements = 0
    total_checks = 0

    for declared_label in ("head", "left", "right"):
        path    = paths.get(declared_label, "")
        verdict = CameraVerdict(declared_label=declared_label, file_path=path)
        tracker_pred_for_label = tracker_map.get(declared_label, declared_label)

        if tracker.ok:
            total_checks += 1

        fi = fisheye.get(declared_label)
        if fi is not None:
            total_checks += 1
            if declared_label == "head":
                if fi.is_fisheye:
                    agreements += 1
                else:
                    verdict.warnings.append(
                        f"La vidéo déclarée 'head' n'est pas détectée comme fisheye (score={fi.score:.2f})"
                    )
            else:
                if fi.is_fisheye:
                    verdict.warnings.append(
                        f"La vidéo déclarée '{declared_label}' ressemble à un fisheye (score={fi.score:.2f})"
                    )
                else:
                    agreements += 1

        if tracker.ok and tracker_pred_for_label == declared_label:
            agreements += 1

        if declared_label in ("left", "right") and not motion.left_right_consistent:
            verdict.warnings.append(
                f"Corrélation left↔right faible ({motion.lr_correlation:.3f})"
            )

        if tracker.ok:
            verdict.predicted_label = tracker_pred_for_label
        elif fi is not None and declared_label == "head":
            verdict.predicted_label = "head" if fi.is_fisheye else "unknown"
        else:
            verdict.predicted_label = declared_label

        verdict.label_correct = (verdict.predicted_label == declared_label)
        if not verdict.label_correct:
            verdict.errors.append(
                f"Label déclaré '{declared_label}' mais géométrie 3D indique '{verdict.predicted_label}'"
            )
        verdicts.append(verdict)

    agreement_rate = agreements / total_checks if total_checks > 0 else 0.0
    if agreement_rate >= 0.90 and tracker.confidence >= 0.90:
        global_conf = CONFIDENCE_FULL
    elif agreement_rate >= 0.70:
        global_conf = CONFIDENCE_HIGH
    else:
        global_conf = CONFIDENCE_LOW

    recommended: Dict[str, str] = {
        v.declared_label: v.predicted_label
        for v in verdicts
        if v.predicted_label and v.predicted_label != "left_or_right"
    }
    return verdicts, recommended, global_conf


# ── Metadata / rapport ────────────────────────────────────────────────────────

def write_safe_report(metadata_path: str, report: VerificationReport) -> None:
    try:
        with open(metadata_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"  ERREUR lecture metadata : {e}", file=sys.stderr)
        return

    block = {
        "verified_at":               pd.Timestamp.now(tz="UTC").isoformat(),
        "global_ok":                 report.global_ok,
        "confidence":                round(report.confidence, 4),
        "safe_mode":                 True,
        "recommended_camera_mapping": report.recommended_mapping,
        "verdicts": [
            {
                "declared_label":  v.declared_label,
                "predicted_label": v.predicted_label,
                "label_correct":   v.label_correct,
                "warnings":        v.warnings,
                "errors":          v.errors,
            }
            for v in report.verdicts
        ],
        "tracker_analysis": {
            "ok":         report.tracker.ok if report.tracker else False,
            "confidence": round(report.tracker.confidence, 4) if report.tracker else 0.0,
            "head_identified_as":  report.tracker.head_tracker_id  if report.tracker else "",
            "left_identified_as":  report.tracker.left_tracker_id  if report.tracker else "",
            "right_identified_as": report.tracker.right_tracker_id if report.tracker else "",
            "vertical_separation_cm":   round((report.tracker.vertical_separation_m   if report.tracker else 0.0) * 100, 1),
            "horizontal_separation_cm": round((report.tracker.horizontal_separation_m if report.tracker else 0.0) * 100, 1),
        },
        "fisheye_analysis": {
            label: {"score": round(fi.score, 4), "is_fisheye": fi.is_fisheye}
            for label, fi in (report.fisheye or {}).items()
        },
        "motion_analysis": {
            "lr_correlation":      round(report.motion.lr_correlation, 4) if report.motion else 0.0,
            "lh_correlation":      round(report.motion.lh_correlation, 4) if report.motion else 0.0,
            "rh_correlation":      round(report.motion.rh_correlation, 4) if report.motion else 0.0,
            "left_right_consistent": report.motion.left_right_consistent if report.motion else False,
        },
        "summary": report.summary,
    }

    meta["camera_label_verification"] = block

    p   = Path(metadata_path)
    bak = p.with_suffix(".json.bak_verify")
    try:
        shutil.copy2(str(p), str(bak))
    except Exception:
        pass

    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Rapport écrit dans {metadata_path}")
    print(f"    (backup → {bak.name})")


def apply_renames(session_dir: Path, recommended: Dict[str, str], yes: bool = False) -> bool:
    renames    = []
    extensions = [".mp4", ".MP4", ".jsonl", ".JSONL"]
    for old_label, new_label in recommended.items():
        if old_label == new_label:
            continue
        for ext in extensions:
            for search_dir in [session_dir / "videos", session_dir]:
                src = search_dir / f"{old_label}{ext}"
                if src.exists():
                    renames.append((src, search_dir / f"{new_label}{ext}"))

    if not renames:
        print("  Aucun renommage nécessaire.")
        return True

    print("\n  Renommages prévus :")
    for src, dst in renames:
        print(f"    {src.name}  →  {dst.name}")

    if not yes:
        answer = input("\n  Confirmer les renommages ? [oui/non] : ").strip().lower()
        if answer not in ("oui", "o", "yes", "y"):
            print("  Annulé.")
            return False

    for src, dst in renames:
        if dst.exists():
            print(f"  ERREUR : {dst} existe déjà — renommage annulé pour ce fichier")
            continue
        src.rename(dst)
        print(f"  ✓ {src.name} → {dst.name}")

    return True


def save_png(
    paths:    Dict[str, str],
    verdicts: List[CameraVerdict],
    fisheye:  Dict[str, FisheyeResult],
    motion:   MotionResult,
    tracker:  TrackerAssignment,
    out_path: str,
) -> None:
    labels = list(paths.keys())
    n = len(labels)
    fig, axes = plt.subplots(3, n, figsize=(6 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    verdict_map = {v.declared_label: v for v in verdicts}
    for col, label in enumerate(labels):
        path    = paths[label]
        verdict = verdict_map.get(label)
        fi      = fisheye.get(label)
        ok_col  = "green" if (verdict and verdict.label_correct) else "red"

        ax = axes[0][col]
        frames = sample_frames(path, n=3)
        if frames:
            rgb = cv2.cvtColor(frames[len(frames)//2], cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
        pred = verdict.predicted_label if verdict else "?"
        conf_txt = f"fisheye={fi.score:.2f}" if fi else ""
        ax.set_title(f"{label.upper()}  →  prédit: {pred}\n{conf_txt}",
                     color=ok_col, fontsize=11, fontweight="bold")
        ax.axis("off")

        ax = axes[1][col]
        sig = extract_motion_signal(path, max_frames=60)
        if sig:
            ax.plot(sig, linewidth=1.2, color="steelblue")
        ax.set_title(f"Mouvement — {label}", fontsize=9)
        ax.set_xlabel("Frame")
        ax.grid(True, alpha=0.3)

        ax = axes[2][col]
        if fi:
            categories = ["aspect\nratio", "vignetage", "netteté\ncoins", "distorsion"]
            ax.bar(categories, [fi.score] * 4, color="steelblue", alpha=0.6)
            ax.axhline(FISHEYE_THRESHOLD, color="red", linestyle="--", linewidth=1,
                       label=f"seuil={FISHEYE_THRESHOLD}")
            ax.set_ylim(0, 1)
            ax.set_title(f"Score fisheye = {fi.score:.3f}", fontsize=9)
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    all_ok = all(v.label_correct for v in verdicts)
    status = "LABELS CORRECTS" if all_ok else "LABELS INCORRECTS — CORRECTION NÉCESSAIRE"
    fig.suptitle(
        f"Vérification labels vidéo\n{status}  |  tracker conf={tracker.confidence:.2f}",
        fontsize=13, fontweight="bold",
        color="green" if all_ok else "red",
    )
    if motion:
        fig.text(
            0.5, 0.01,
            f"Corrélations mouvement :  "
            f"left↔right={motion.lr_correlation:+.3f}  "
            f"left↔head={motion.lh_correlation:+.3f}  "
            f"right↔head={motion.rh_correlation:+.3f}",
            ha="center", fontsize=9, color="dimgray",
        )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=130)
    plt.close()


def _find_file(session_dir: Path, stem: str,
               extensions=(".mp4", ".MP4", ".jsonl", ".JSONL")) -> Optional[Path]:
    for ext in extensions:
        for sub in [session_dir / "videos", session_dir]:
            p = sub / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def run_verify(args) -> VerificationReport:
    session_dir = Path(args.session_dir) if args.session_dir else None
    if session_dir is None:
        if args.left:
            session_dir = Path(args.left).parent
        else:
            print("ERREUR : dossier de session requis.", file=sys.stderr)
            sys.exit(1)
    if not session_dir.is_dir():
        print(f"ERREUR : dossier introuvable : {session_dir}", file=sys.stderr)
        sys.exit(1)

    paths: Dict[str, str] = {}
    for label in ("left", "right", "head"):
        explicit = getattr(args, label, None)
        if explicit:
            paths[label] = explicit
        else:
            p = _find_file(session_dir, label, (".mp4", ".MP4"))
            if p:
                paths[label] = str(p)

    missing = [l for l in ("left", "right", "head") if l not in paths]
    if missing:
        print(f"ERREUR : vidéo(s) introuvable(s) : {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    tracker_csv = getattr(args, "tracker", None)
    if not tracker_csv:
        for candidate in [session_dir / "tracker_positions.csv",
                          session_dir.parent / "tracker_positions.csv"]:
            if candidate.exists():
                tracker_csv = str(candidate)
                break

    metadata_path = getattr(args, "metadata", None)
    if not metadata_path:
        for candidate in [session_dir / "metadata.json",
                          session_dir.parent / "metadata.json"]:
            if candidate.exists():
                metadata_path = str(candidate)
                break

    report = VerificationReport(
        session_dir=str(session_dir),
        safe_mode=not getattr(args, "apply", False),
    )

    print("\n[1/3] Analyse géométrique 3D des trackers...")
    if tracker_csv:
        report.tracker = analyze_trackers(tracker_csv)
        print(f"      Confiance tracker : {report.tracker.confidence:.2f}")
    else:
        print("      ⚠ tracker_positions.csv introuvable — niveau 1 ignoré")
        report.tracker = TrackerAssignment()

    print("[2/3] Analyse fisheye des vidéos...")
    for label, path in paths.items():
        print(f"      {label} : {Path(path).name}...")
        report.fisheye[label] = analyze_fisheye(label, path)

    print("[3/3] Analyse de cohérence du mouvement...")
    report.motion = analyze_motion(paths)

    verdicts, recommended, confidence = compute_verdicts(
        paths, report.tracker, report.fisheye, report.motion
    )
    report.verdicts  = verdicts
    report.recommended_mapping = recommended
    report.confidence = confidence
    report.global_ok  = all(v.label_correct for v in verdicts)

    if report.global_ok:
        report.summary = f"Tous les labels sont corrects (confiance = {confidence:.0%})"
    else:
        bad = [v for v in verdicts if not v.label_correct]
        corrections = ", ".join(f"{v.declared_label}→{v.predicted_label}" for v in bad)
        report.summary = f"Labels incorrects détectés : {corrections}  (confiance = {confidence:.0%})"

    _print_verify_report(report, verbose=getattr(args, "verbose", False))

    if metadata_path:
        write_safe_report(metadata_path, report)
    else:
        print("\n  ⚠ metadata.json non trouvé — rapport non écrit")

    out_png = getattr(args, "output_png", None) or str(session_dir / "label_verification.png")
    print(f"\n  Génération du rapport visuel → {out_png}")
    try:
        save_png(paths, verdicts, report.fisheye, report.motion, report.tracker, out_png)
        print(f"  ✓ {out_png}")
    except Exception as e:
        print(f"  ⚠ PNG non généré : {e}")

    out_json = getattr(args, "output_json", None)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        print(f"  ✓ Rapport JSON → {out_json}")

    if getattr(args, "apply", False) and not report.global_ok:
        apply_renames(session_dir, recommended, yes=getattr(args, "yes", False))

    return report


def _print_verify_report(report: VerificationReport, verbose: bool = False) -> None:
    SEP = "═" * 65
    sep = "─" * 65
    print(f"\n{SEP}\n  VÉRIFICATION LABELS VIDÉO\n{SEP}")

    print(f"\n{'NIVEAU 1 — Géométrie 3D des trackers':}")
    print(sep)
    t = report.tracker
    if t:
        status = "✓ OK" if t.ok else "✗ ÉCHEC"
        print(f"  Confiance tracker : {t.confidence:.2f}  [{status}]")
        for d in t.details:
            if verbose or not t.ok or any(kw in d for kw in ["Plus haut", "gauche", "droite", "Séparation", "✓", "!"]):
                print(f"    {d}")

    print(f"\nNIVEAU 2 — Détection fisheye\n{sep}")
    for label in ("head", "left", "right"):
        fi = report.fisheye.get(label)
        if fi:
            tag = "FISHEYE" if fi.is_fisheye else "normal "
            ok  = "✓" if (label == "head") == fi.is_fisheye else "✗"
            print(f"  {ok} {label:>5} : score={fi.score:.3f}  [{tag}]")
            if verbose:
                for d in fi.details:
                    print(f"         {d}")

    print(f"\nNIVEAU 3 — Cohérence mouvement\n{sep}")
    m = report.motion
    if m:
        lr_ok = "✓" if m.left_right_consistent else "⚠"
        print(f"  {lr_ok} left ↔ right : {m.lr_correlation:+.3f}")
        print(f"    left ↔ head  : {m.lh_correlation:+.3f}")
        print(f"    right ↔ head : {m.rh_correlation:+.3f}")

    print(f"\nVERDICTS PAR CAMÉRA\n{sep}")
    for v in report.verdicts:
        marker = "✓" if v.label_correct else "✗"
        print(f"  {marker} {v.declared_label:>5}  →  prédit : {v.predicted_label:<8}  "
              f"({'CORRECT' if v.label_correct else 'INCORRECT'})")
        for e in v.errors:
            print(f"         ✗ {e}")
        for w in v.warnings:
            print(f"         ⚠ {w}")

    print(f"\n{SEP}")
    status_icon = "✓" if report.global_ok else "✗"
    print(f"  {status_icon} RÉSULTAT : {report.summary}")
    print(f"  Confiance globale : {report.confidence:.2%}")
    mode_txt = "[MODE SAFE — metadata.json mis à jour]" if report.safe_mode else "[APPLY — renommages effectués]"
    print(f"  {mode_txt}")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_fix_parser(sub):
    p = sub.add_parser("fix", help="Corriger les rôles trackers head/left/right.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("csv_path", nargs="?", help="Fichier CSV unique")
    group.add_argument("--all", metavar="DIR", dest="all_dir",
                       help="Traite tous les tracker_positions.csv sous DIR")
    p.add_argument("--output",         help="Fichier de sortie (mode fichier unique)")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--report",         metavar="FILE")
    p.add_argument("--min-confidence", type=float, default=0.65)
    p.add_argument("-v", "--verbose",  action="store_true")
    return p


def _build_rotate_parser(sub):
    p = sub.add_parser("rotate", help="Rotation 180° des vidéos d'une session.")
    p.add_argument("session", type=str, help="Chemin de la session")
    p.add_argument("--sides", nargs="+", default=list(VIDEO_SIDES), choices=list(VIDEO_SIDES))
    p.add_argument("--force", action="store_true")
    return p


def _build_verify_parser(sub):
    p = sub.add_parser("verify", help="Vérifier et corriger les labels caméra.")
    p.add_argument("session_dir", nargs="?", default=None)
    p.add_argument("--left",  default=None)
    p.add_argument("--right", default=None)
    p.add_argument("--head",  default=None)
    p.add_argument("--tracker",  default=None)
    p.add_argument("--metadata", default=None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--safe",  action="store_true", default=True)
    mode.add_argument("--apply", action="store_true", default=False)
    p.add_argument("--yes", "-y",        action="store_true")
    p.add_argument("--output-png",       default=None)
    p.add_argument("--output-json",      default=None)
    p.add_argument("--verbose", "-v",    action="store_true")
    return p


def _cmd_fix(args) -> int:
    report_entries = []
    if args.all_dir:
        root = Path(args.all_dir)
        if not root.is_dir():
            print(f"[ERREUR] {root} n'est pas un dossier.", file=sys.stderr)
            return 2
        csv_files = find_all_tracker_csvs(root)
        if not csv_files:
            print(f"[ERREUR] Aucun tracker_positions.csv trouvé sous {root}.", file=sys.stderr)
            return 2
        print(f"Trouvé {len(csv_files)} fichier(s) à analyser.\n")
        modified = 0
        for p in csv_files:
            entry = _process_fix_file(p, None, dry_run=args.dry_run,
                                      verbose=args.verbose, min_confidence=args.min_confidence)
            report_entries.append(entry)
            if entry.get("modified"):
                modified += 1
        n_swap = sum(1 for e in report_entries if e["swapped"])
        n_skip = sum(1 for e in report_entries if e["status"] == "skipped_low_confidence")
        n_err  = sum(1 for e in report_entries if e["status"] == "error")
        print(f"\n{modified}/{len(csv_files)} fichier(s) modifié(s).  "
              f"SWAP détectés: {n_swap}  faible conf: {n_skip}  erreurs: {n_err}")
    else:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"[ERREUR] Fichier introuvable: {csv_path}", file=sys.stderr)
            return 2
        output = Path(args.output) if args.output else None
        entry  = _process_fix_file(csv_path, output, dry_run=args.dry_run,
                                   verbose=args.verbose, min_confidence=args.min_confidence)
        report_entries.append(entry)

    if args.report:
        report_path = Path(args.report)
        report_path.write_text(json.dumps(report_entries, indent=2, ensure_ascii=False))
        print(f"\nRapport JSON → {report_path}")
    return 0


def _cmd_rotate(args) -> int:
    sess = Path(args.session)
    if not sess.exists():
        print(f"ERREUR : session introuvable : {sess}", file=sys.stderr)
        return 1
    result = rotate_session_videos(sess, sides=args.sides, force=args.force)
    print(f"\nRésultat :")
    print(f"  Rotées  : {result['rotated']}")
    print(f"  Sautées : {result['skipped']}")
    print(f"  Erreurs : {result['errors']}")
    return 0 if not result["errors"] else 1


def _cmd_verify(args) -> int:
    report = run_verify(args)
    return 0 if report.global_ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="data_prep.py — préparation données : trackers, rotation, labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _build_fix_parser(sub)
    _build_rotate_parser(sub)
    _build_verify_parser(sub)

    args = parser.parse_args()
    if args.cmd == "fix":
        return _cmd_fix(args)
    elif args.cmd == "rotate":
        return _cmd_rotate(args)
    elif args.cmd == "verify":
        return _cmd_verify(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
