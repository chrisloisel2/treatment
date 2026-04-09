#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
identify_side.py — Identification visuelle gauche/droite des caméras

Principe :
  Cherche dans les flux capteur un moment où une pince est clairement fermée
  et l'autre clairement ouverte. Pour chaque candidat, génère une image
  composite avec :
    - Strip de 5 frames (contexte temporel ±2s) pour chaque vidéo
    - Graphe des signaux capteur sur la durée de la session avec le moment marqué

Usage :
  python identify_side.py --session /path/to/session
  python identify_side.py --session /path/to/session --output_dir /tmp/out
  python identify_side.py --session /path/to/session --n_candidates 5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io


# ─────────────────────────────────────────────────────────────────────────────
# Chargement données
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_timestamps(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse videos/{side}.jsonl → (frame_positions_0based, timestamps_ns)
    Les indices JSONL sont absolus → rebasés à 0.
    capture_time en ms → × 1_000_000 = ns
    """
    raw = open(path, "rb").read()
    indices, ts_ns = [], []

    for line in raw.split(b"\r\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
            indices.append(int(obj["index"]))
            ts_ns.append(int(obj["capture_time"]) * 1_000_000)
        except Exception:
            continue

    if not indices:
        raise RuntimeError(f"Aucune entrée valide dans {path}")

    indices = np.array(indices, dtype=np.int32)
    ts_ns   = np.array(ts_ns,   dtype=np.int64)
    order   = np.argsort(indices)
    indices = indices[order]
    ts_ns   = ts_ns[order]

    # Rebase : index absolu → position 0-based dans la vidéo
    indices = indices - indices[0]
    return indices, ts_ns


def load_sensor(path: str) -> pd.DataFrame:
    """
    Parse gripper_{side}_data.csv → DataFrame (timestamp_ns, opening_mm)
    """
    df = pd.read_csv(path)
    for col in ("timestamp_ns", "opening_mm"):
        if col not in df.columns:
            raise RuntimeError(f"Colonne manquante '{col}' dans {path}")
    df = df[["timestamp_ns", "opening_mm"]].copy()
    df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce")
    df["opening_mm"]   = pd.to_numeric(df["opening_mm"],   errors="coerce")
    df = df.dropna().sort_values("timestamp_ns").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"Données capteur vides : {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Recherche du moment le plus asymétrique
# ─────────────────────────────────────────────────────────────────────────────

def find_asymmetric_moments(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    n_candidates: int = 5,
    closed_threshold_mm: float = 6.0,
    open_threshold_mm: float = 14.0,
) -> list:
    """
    Retourne les n meilleurs moments où un gripper est fermé et l'autre ouvert.
    """
    ts_min = max(left_df["timestamp_ns"].iloc[0],  right_df["timestamp_ns"].iloc[0])
    ts_max = min(left_df["timestamp_ns"].iloc[-1], right_df["timestamp_ns"].iloc[-1])

    if ts_max <= ts_min:
        raise RuntimeError("Pas de recouvrement temporel entre les deux capteurs.")

    step_ns = 10_000_000  # 10ms
    grid = np.arange(ts_min, ts_max, step_ns, dtype=np.int64)

    op_left  = np.interp(grid.astype(float),
                         left_df["timestamp_ns"].values.astype(float),
                         left_df["opening_mm"].values)
    op_right = np.interp(grid.astype(float),
                         right_df["timestamp_ns"].values.astype(float),
                         right_df["opening_mm"].values)

    diff = np.abs(op_left - op_right)

    one_closed = (op_left < closed_threshold_mm) | (op_right < closed_threshold_mm)
    one_open   = (op_left > open_threshold_mm)   | (op_right > open_threshold_mm)
    valid_mask = one_closed & one_open

    if valid_mask.sum() == 0:
        print("[WARN] Aucun moment avec seuils stricts — élargissement des critères.")
        valid_mask = np.ones(len(grid), dtype=bool)

    valid_diff = np.where(valid_mask, diff, -1.0)

    min_gap = int(1_000_000_000 / step_ns)  # 1 seconde minimum entre candidats
    candidates = []

    for _ in range(n_candidates * 10):
        if len(candidates) >= n_candidates:
            break
        idx = int(np.argmax(valid_diff))
        if valid_diff[idx] < 0:
            break
        candidates.append({
            "timestamp_ns":  int(grid[idx]),
            "opening_left":  float(op_left[idx]),
            "opening_right": float(op_right[idx]),
            "score":         float(valid_diff[idx]),
        })
        lo = max(0, idx - min_gap)
        hi = min(len(grid), idx + min_gap)
        valid_diff[lo:hi] = -1.0

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Temps relatif depuis début session
    t0 = min(left_df["timestamp_ns"].iloc[0], right_df["timestamp_ns"].iloc[0])
    for c in candidates:
        c["time_rel_s"] = (c["timestamp_ns"] - t0) / 1e9

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Extraction frames vidéo
# ─────────────────────────────────────────────────────────────────────────────

def ts_ns_to_frame_index(ts_ns: int,
                          jsonl_indices: np.ndarray,
                          jsonl_ts_ns: np.ndarray) -> int:
    idx = int(np.argmin(np.abs(jsonl_ts_ns - ts_ns)))
    return int(jsonl_indices[idx])


def extract_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def extract_frame_strip(
    video_path: str,
    center_frame: int,
    jsonl_indices: np.ndarray,
    jsonl_ts_ns: np.ndarray,
    n_frames: int = 5,
    spread_s: float = 2.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Extrait n_frames réparties sur ±spread_s autour de center_frame.
    Retourne liste de (frame_bgr, offset_s_from_center).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Timestamp du centre
    center_in_jsonl = np.searchsorted(jsonl_indices, center_frame)
    center_in_jsonl = min(center_in_jsonl, len(jsonl_indices) - 1)
    ts_center = jsonl_ts_ns[center_in_jsonl]

    offsets_s = np.linspace(-spread_s, spread_s, n_frames)
    results = []

    for off_s in offsets_s:
        ts_target = ts_center + int(off_s * 1e9)
        fi = ts_ns_to_frame_index(ts_target, jsonl_indices, jsonl_ts_ns)
        fi = max(0, min(fi, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        results.append((frame if ok else None, float(off_s)))

    cap.release()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Graphe signal capteur (matplotlib → numpy BGR)
# ─────────────────────────────────────────────────────────────────────────────

def make_signal_graph(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    t0_ns: int,
    candidate_ts_ns: int,
    width_px: int = 1400,
    height_px: int = 220,
) -> np.ndarray:
    """
    Graphe des deux signaux capteur sur toute la session,
    avec ligne verticale au moment candidat.
    Retourne une image BGR numpy.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    t_left  = (left_df["timestamp_ns"].values  - t0_ns) / 1e9
    t_right = (right_df["timestamp_ns"].values - t0_ns) / 1e9
    t_cand  = (candidate_ts_ns - t0_ns) / 1e9

    ax.plot(t_left,  left_df["opening_mm"].values,  color="#00d4ff", lw=1.2,
            alpha=0.8, label="left gripper")
    ax.plot(t_right, right_df["opening_mm"].values, color="#ff6b6b", lw=1.2,
            alpha=0.8, label="right gripper")

    ax.axvline(x=t_cand, color="#ffd700", lw=2.0, linestyle="--", label=f"t={t_cand:.1f}s")

    ax.set_xlabel("Temps (s)", color="white", fontsize=8)
    ax.set_ylabel("Ouverture (mm)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e",
              labelcolor="white", framealpha=0.8)
    ax.set_xlim(t_left[0], max(t_left[-1], t_right[-1]))

    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    buf.close()
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Construction du strip de frames annoté
# ─────────────────────────────────────────────────────────────────────────────

def build_frame_strip(
    frames_offsets: List[Tuple[Optional[np.ndarray], float]],
    opening_mm: float,
    label: str,
    strip_h: int = 270,
    is_center_list: List[bool] = None,
) -> np.ndarray:
    """
    Construit un strip horizontal de frames avec annotations.
    La frame centrale (offset ≈ 0) est plus grande et mise en évidence.
    """
    n = len(frames_offsets)
    if is_center_list is None:
        is_center_list = [abs(off) < 0.05 for _, off in frames_offsets]

    # Largeur par frame : centre plus large
    center_w = int(strip_h * 16 / 9)
    side_w   = int(center_w * 0.65)

    cells = []
    for i, (frame, off_s) in enumerate(frames_offsets):
        is_center = is_center_list[i]
        cell_w = center_w if is_center else side_w
        cell_h = strip_h if is_center else int(strip_h * 0.80)

        if frame is None:
            cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(cell, "NO FRAME", (10, cell_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        else:
            # Resize en gardant ratio
            h0, w0 = frame.shape[:2]
            scale = min(cell_w / w0, cell_h / h0)
            rw, rh = int(w0 * scale), int(h0 * scale)
            cell = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
            # Padding
            pad_top  = (cell_h - rh) // 2
            pad_bot  = cell_h - rh - pad_top
            pad_left = (cell_w - rw) // 2
            pad_right = cell_w - rw - pad_left
            cell = cv2.copyMakeBorder(cell, pad_top, pad_bot, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=(20, 20, 20))

        # Overlay timestamp
        off_text = f"{'NOW' if is_center else f'{off_s:+.1f}s'}"
        cv2.putText(cell, off_text, (6, cell_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Bordure colorée sur la frame centrale
        if is_center:
            color = (0, 200, 0) if opening_mm > 12 else (0, 0, 220)
            cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), color, 4)

        # Padding vertical pour aligner au bas
        if cell.shape[0] < strip_h:
            pad = np.full((strip_h - cell.shape[0], cell.shape[1], 3), 30, dtype=np.uint8)
            cell = np.vstack([pad, cell])

        cells.append(cell)

    strip = np.hstack(cells)

    # Étiquette latérale gauche
    label_w = 90
    label_img = np.full((strip_h, label_w, 3), 25, dtype=np.uint8)
    state    = "OUVERT" if opening_mm > 12 else "FERME"
    color    = (0, 200, 0) if opening_mm > 12 else (0, 0, 220)
    cv2.putText(label_img, label,   (4, strip_h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.putText(label_img, state,   (4, strip_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  color, 2)
    cv2.putText(label_img, f"{opening_mm:.1f}mm", (4, strip_h // 2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    return np.hstack([label_img, strip])


# ─────────────────────────────────────────────────────────────────────────────
# Assemblage image finale
# ─────────────────────────────────────────────────────────────────────────────

def make_composite_image(
    strip_left: np.ndarray,
    strip_right: np.ndarray,
    graph: np.ndarray,
    cand: dict,
    candidate_rank: int,
) -> np.ndarray:
    """
    Assemble : titre + strip left + séparateur + strip right + graphe signal.
    """
    total_w = max(strip_left.shape[1], strip_right.shape[1], graph.shape[1])

    def pad_w(img, w):
        if img.shape[1] < w:
            p = np.full((img.shape[0], w - img.shape[1], 3), 15, dtype=np.uint8)
            return np.hstack([img, p])
        return img

    strip_left  = pad_w(strip_left,  total_w)
    strip_right = pad_w(strip_right, total_w)
    graph       = pad_w(graph,       total_w)

    # Titre
    title_h = 50
    title   = np.full((title_h, total_w, 3), 15, dtype=np.uint8)
    t_rel   = cand.get("time_rel_s", 0.0)
    txt = (f"  Candidat #{candidate_rank}  |  t={t_rel:.1f}s  |  "
           f"left.mp4: {cand['opening_left']:.1f}mm  |  "
           f"right.mp4: {cand['opening_right']:.1f}mm  |  "
           f"diff={cand['score']:.1f}mm")
    cv2.putText(title, txt, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 80), 2)

    # Séparateur
    sep = np.full((6, total_w, 3), 60, dtype=np.uint8)

    return np.vstack([title, strip_left, sep, strip_right, sep, graph])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Identification visuelle gauche/droite des caméras")
    p.add_argument("--session",      required=True, help="Chemin du dossier session")
    p.add_argument("--output_dir",   default=None,  help="Dossier de sortie")
    p.add_argument("--n_candidates", type=int,   default=5,    help="Nombre de candidats")
    p.add_argument("--closed_mm",    type=float, default=6.0,  help="Seuil fermeture (mm)")
    p.add_argument("--open_mm",      type=float, default=14.0, help="Seuil ouverture (mm)")
    p.add_argument("--strip_frames", type=int,   default=5,    help="Frames par strip (impair)")
    p.add_argument("--spread_s",     type=float, default=2.0,  help="Étendue temporelle strip (±s)")
    args = p.parse_args()

    session    = Path(args.session)
    output_dir = Path(args.output_dir) if args.output_dir else session / "identify_side"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_left    = session / "gripper_left_data.csv"
    csv_right   = session / "gripper_right_data.csv"
    jsonl_left  = session / "videos" / "left.jsonl"
    jsonl_right = session / "videos" / "right.jsonl"
    video_left  = session / "videos" / "left.mp4"
    video_right = session / "videos" / "right.mp4"

    for f in [csv_left, csv_right, jsonl_left, jsonl_right, video_left, video_right]:
        if not f.exists():
            print(f"[ERREUR] Fichier manquant : {f}", file=sys.stderr)
            sys.exit(1)

    print(f"Session : {session}")

    print("Chargement CSV capteurs...")
    left_df  = load_sensor(str(csv_left))
    right_df = load_sensor(str(csv_right))
    print(f"  left  : {len(left_df)} éch.  [{left_df['opening_mm'].min():.1f}–{left_df['opening_mm'].max():.1f}] mm")
    print(f"  right : {len(right_df)} éch.  [{right_df['opening_mm'].min():.1f}–{right_df['opening_mm'].max():.1f}] mm")

    print("Chargement timestamps JSONL...")
    idx_left,  ts_left  = load_jsonl_timestamps(str(jsonl_left))
    idx_right, ts_right = load_jsonl_timestamps(str(jsonl_right))

    t0_ns = min(left_df["timestamp_ns"].iloc[0], right_df["timestamp_ns"].iloc[0])

    print(f"Recherche moments asymétriques (fermé<{args.closed_mm}mm / ouvert>{args.open_mm}mm)...")
    candidates = find_asymmetric_moments(
        left_df, right_df,
        n_candidates=args.n_candidates,
        closed_threshold_mm=args.closed_mm,
        open_threshold_mm=args.open_mm,
    )

    if not candidates:
        print("[ERREUR] Aucun moment asymétrique trouvé.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(candidates)} candidats trouvés.")

    # Strip : la frame centrale est celle avec offset ≈ 0
    n = args.strip_frames
    if n % 2 == 0:
        n += 1
    center_i = n // 2
    is_center = [i == center_i for i in range(n)]

    report_lines = [
        "=== IDENTIFY SIDE REPORT ===",
        f"Session: {session}",
        f"Seuil fermé: <{args.closed_mm}mm  |  Seuil ouvert: >{args.open_mm}mm",
        "",
        "Candidats (trié par asymétrie décroissante):",
        "",
    ]

    for i, cand in enumerate(candidates):
        ts = cand["timestamp_ns"]

        fi_left  = ts_ns_to_frame_index(ts, idx_left,  ts_left)
        fi_right = ts_ns_to_frame_index(ts, idx_right, ts_right)

        # Strips de frames
        frames_l = extract_frame_strip(str(video_left),  fi_left,
                                       idx_left,  ts_left,
                                       n_frames=n, spread_s=args.spread_s)
        frames_r = extract_frame_strip(str(video_right), fi_right,
                                       idx_right, ts_right,
                                       n_frames=n, spread_s=args.spread_s)

        # Vérifier que la frame centrale est bien disponible
        center_frame_l = frames_l[center_i][0]
        center_frame_r = frames_r[center_i][0]
        if center_frame_l is None or center_frame_r is None:
            print(f"  [SKIP] Candidat {i+1} — frame centrale non lisible")
            continue

        strip_l = build_frame_strip(frames_l, cand["opening_left"],
                                    "left.mp4",  is_center_list=is_center)
        strip_r = build_frame_strip(frames_r, cand["opening_right"],
                                    "right.mp4", is_center_list=is_center)

        # Graphe signal
        graph = make_signal_graph(left_df, right_df, t0_ns, ts,
                                  width_px=strip_l.shape[1],
                                  height_px=200)

        composite = make_composite_image(strip_l, strip_r, graph, cand, i + 1)

        out_path = output_dir / f"candidate_{i+1:02d}.png"
        cv2.imwrite(str(out_path), composite)

        ts_rel_s = cand.get("time_rel_s", 0.0)
        line = (f"  [{i+1}] t={ts_rel_s:.1f}s  "
                f"left={cand['opening_left']:.1f}mm  "
                f"right={cand['opening_right']:.1f}mm  "
                f"diff={cand['score']:.1f}mm  → {out_path.name}")
        print(line)
        report_lines.append(line)

    report_lines += [
        "",
        "COMMENT LIRE LES IMAGES :",
        "  Chaque image = strip de 5 frames (contexte ±2s) pour left.mp4 puis right.mp4,",
        "  + graphe des signaux capteur sur toute la session (trait jaune = moment candidat).",
        "",
        "  Bordure VERTE  = gripper ouvert selon capteur",
        "  Bordure ROUGE  = gripper fermé selon capteur",
        "",
        "  Si la vidéo LEFT a une bordure ROUGE mais montre visuellement un gripper OUVERT,",
        "  alors les étiquettes left/right sont inversées.",
    ]

    report_path = output_dir / "identify_side_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nRapport : {report_path}")
    print(f"Images  : {output_dir}/candidate_XX.png")


if __name__ == "__main__":
    main()
