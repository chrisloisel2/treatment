#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère un flux 1D de mouvement à partir d'une vidéo.

Le flux généré peut servir à :
- comparer la vidéo aux trackeurs
- faire de la corrélation temporelle
- trouver un offset
- visualiser l'énergie de mouvement dans le temps

Sorties :
- CSV avec un signal temporel par frame
- PNG avec la courbe du flux
- npy optionnel

Flux calculés :
1. motion_mean
   moyenne de la magnitude de l'optical flow

2. motion_median
   médiane de la magnitude de l'optical flow

3. motion_p90
   percentile 90 de la magnitude

4. diff_mean
   différence moyenne absolue entre frames

5. diff_median
   médiane de la différence absolue

Ancrage temporel absolu :
  Si --jsonl est fourni, chaque frame est ancrée sur le capture_time du JSONL
  (ms depuis epoch). Le CSV de sortie contient alors une colonne timestamp_abs_ms
  utilisable directement par sync_analyzer.py.

Usage :
    python video.py video.mp4
    python video.py video.mp4 --jsonl head.jsonl --output-csv head_flux.csv
    python video.py video.mp4 --output-csv video_flux.csv --show
    python video.py video.mp4 --resize-width 640 --smooth-window 9
    python video.py video.mp4 --start-sec 10 --end-sec 40
    python video.py video.mp4 --roi 200 100 800 600

Dépendances :
    pip install opencv-python numpy pandas matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Génère des flux temporels de mouvement à partir d'une vidéo."
    )

    parser.add_argument("video_path", type=str, help="Chemin de la vidéo.")

    parser.add_argument(
        "--output-csv",
        type=str,
        default="video_flux.csv",
        help="CSV de sortie."
    )

    parser.add_argument(
        "--output-plot",
        type=str,
        default="video_flux.png",
        help="PNG de sortie."
    )

    parser.add_argument(
        "--output-npy",
        type=str,
        default=None,
        help="Fichier NPY optionnel."
    )

    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="Largeur cible. 0 = taille originale."
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Fenêtre de lissage. 1 = aucun lissage."
    )

    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Temps de début en secondes."
    )

    parser.add_argument(
        "--end-sec",
        type=float,
        default=-1.0,
        help="Temps de fin en secondes. -1 = fin vidéo."
    )

    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Zone d'intérêt dans l'image."
    )

    parser.add_argument(
        "--farneback-pyr-scale",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--farneback-levels",
        type=int,
        default=3
    )
    parser.add_argument(
        "--farneback-winsize",
        type=int,
        default=15
    )
    parser.add_argument(
        "--farneback-iterations",
        type=int,
        default=3
    )
    parser.add_argument(
        "--farneback-poly-n",
        type=int,
        default=5
    )
    parser.add_argument(
        "--farneback-poly-sigma",
        type=float,
        default=1.2
    )

    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help=(
            "Fichier JSONL associé à la vidéo (ex: head.jsonl). "
            "Chaque ligne doit contenir {\"index\": N, \"capture_time\": T_ms}. "
            "Permet d'ancrer chaque frame sur un timestamp absolu (ms)."
        )
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Affiche la figure."
    )

    return parser.parse_args()


def load_jsonl_timestamps(jsonl_path: str) -> Dict[int, float]:
    """
    Charge un fichier JSONL et retourne un dict {frame_index: capture_time_ms}.
    """
    result: Dict[int, float] = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                result[int(rec["index"])] = float(rec["capture_time"])
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return result


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def preprocess_frame(frame: np.ndarray, resize_width: int, roi):
    h, w = frame.shape[:2]

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(x1, w - 1))
        x2 = max(1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(1, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI invalide.")
        frame = frame[y1:y2, x1:x2]

    if resize_width and resize_width > 0:
        h, w = frame.shape[:2]
        new_h = int(round(h * (resize_width / float(w))))
        frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def main():
    args = parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"ERREUR : vidéo introuvable : {video_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERREUR : impossible d'ouvrir la vidéo.", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("ERREUR : FPS invalide.", file=sys.stderr)
        return 1

    duration_sec = frame_count / fps

    start_sec = max(0.0, args.start_sec)
    end_sec = duration_sec if args.end_sec < 0 else min(args.end_sec, duration_sec)

    if end_sec <= start_sec:
        print("ERREUR : intervalle temporel invalide.", file=sys.stderr)
        return 1

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Chargement optionnel du JSONL pour l'ancrage temporel absolu
    jsonl_timestamps: Optional[Dict[int, float]] = None
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            print(f"ERREUR : JSONL introuvable : {jsonl_path}", file=sys.stderr)
            return 1
        jsonl_timestamps = load_jsonl_timestamps(str(jsonl_path))
        print(f"JSONL chargé : {len(jsonl_timestamps)} entrées")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame = cap.read()
    if not ok:
        print("ERREUR : impossible de lire la première frame.", file=sys.stderr)
        return 1

    prev_gray = preprocess_frame(frame, args.resize_width, args.roi)

    rows = []

    current_frame_idx = start_frame

    while True:
        if current_frame_idx + 1 >= end_frame:
            break

        ok, frame = cap.read()
        if not ok:
            break

        current_frame_idx += 1
        time_sec = current_frame_idx / fps

        gray = preprocess_frame(frame, args.resize_width, args.roi)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            args.farneback_pyr_scale,
            args.farneback_levels,
            args.farneback_winsize,
            args.farneback_iterations,
            args.farneback_poly_n,
            args.farneback_poly_sigma,
            0
        )

        mag = np.linalg.norm(flow, axis=2)

        abs_diff = cv2.absdiff(prev_gray, gray).astype(np.float32)

        row = {
            "frame_index": current_frame_idx,
            "time_seconds": time_sec,
            "motion_mean": float(np.mean(mag)),
            "motion_median": float(np.median(mag)),
            "motion_p90": float(np.percentile(mag, 90)),
            "motion_max": float(np.max(mag)),
            "diff_mean": float(np.mean(abs_diff)),
            "diff_median": float(np.median(abs_diff)),
            "diff_p90": float(np.percentile(abs_diff, 90)),
            "diff_max": float(np.max(abs_diff)),
        }

        # Ancrage absolu depuis le JSONL
        if jsonl_timestamps is not None:
            t_abs = jsonl_timestamps.get(current_frame_idx)
            row["timestamp_abs_ms"] = float(t_abs) if t_abs is not None else float("nan")

        rows.append(row)
        prev_gray = gray

    cap.release()

    if not rows:
        print("ERREUR : aucune donnée générée.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)

    numeric_cols = [
        "motion_mean",
        "motion_median",
        "motion_p90",
        "motion_max",
        "diff_mean",
        "diff_median",
        "diff_p90",
        "diff_max",
    ]

    for col in numeric_cols:
        df[col + "_smooth"] = moving_average(df[col].to_numpy(), args.smooth_window)

    df.to_csv(args.output_csv, index=False)

    if args.output_npy:
        np.save(
            args.output_npy,
            {
                "time_seconds": df["time_seconds"].to_numpy(),
                "motion_mean": df["motion_mean_smooth"].to_numpy(),
                "motion_median": df["motion_median_smooth"].to_numpy(),
                "motion_p90": df["motion_p90_smooth"].to_numpy(),
                "diff_mean": df["diff_mean_smooth"].to_numpy(),
                "diff_median": df["diff_median_smooth"].to_numpy(),
            },
            allow_pickle=True
        )

    plt.figure(figsize=(16, 8))
    plt.plot(df["time_seconds"], df["motion_mean_smooth"], label="motion_mean")
    plt.plot(df["time_seconds"], df["motion_median_smooth"], label="motion_median")
    plt.plot(df["time_seconds"], df["motion_p90_smooth"], label="motion_p90")
    plt.plot(df["time_seconds"], df["diff_mean_smooth"], label="diff_mean")
    plt.plot(df["time_seconds"], df["diff_median_smooth"], label="diff_median")

    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.title("Flux de mouvement extraits de la vidéo")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=160)

    if args.show:
        plt.show()
    else:
        plt.close()

    print(f"Vidéo           : {video_path}")
    print(f"FPS             : {fps:.6f}")
    print(f"Frames totales  : {frame_count}")
    print(f"Début           : {start_sec:.3f} s")
    print(f"Fin             : {end_sec:.3f} s")
    print(f"CSV             : {args.output_csv}")
    print(f"PNG             : {args.output_plot}")
    if args.output_npy:
        print(f"NPY             : {args.output_npy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
