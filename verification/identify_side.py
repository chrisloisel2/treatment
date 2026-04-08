#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
identify_side.py — Identification visuelle gauche/droite des caméras

Principe :
  Cherche dans les flux capteur un moment où une pince est clairement fermée
  et l'autre clairement ouverte. Extrait les frames correspondantes des deux
  vidéos (actuellement étiquetées left/right) et génère une image côte à côte
  pour vérification visuelle.

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
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Chargement données
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_timestamps(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse videos/{side}.jsonl → (frame_positions, timestamps_ns)

    Les indices JSONL sont absolus (ex: 128→710). On les rebase à 0
    pour correspondre aux positions réelles dans le fichier vidéo.
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
    Chaque résultat : dict avec timestamp_ns, opening_left, opening_right, score.

    Stratégie : on interpole les deux signaux sur une grille commune,
    puis on cherche les pics d'asymétrie |left - right| avec contraintes
    ouvert/fermé.
    """
    # Grille commune : tous les timestamps des deux capteurs
    ts_min = max(left_df["timestamp_ns"].iloc[0],  right_df["timestamp_ns"].iloc[0])
    ts_max = min(left_df["timestamp_ns"].iloc[-1], right_df["timestamp_ns"].iloc[-1])

    if ts_max <= ts_min:
        raise RuntimeError("Pas de recouvrement temporel entre les deux capteurs.")

    # Résolution ~10ms
    step_ns = 10_000_000
    grid = np.arange(ts_min, ts_max, step_ns, dtype=np.int64)

    op_left  = np.interp(grid.astype(float),
                         left_df["timestamp_ns"].values.astype(float),
                         left_df["opening_mm"].values)
    op_right = np.interp(grid.astype(float),
                         right_df["timestamp_ns"].values.astype(float),
                         right_df["opening_mm"].values)

    # Score d'asymétrie : différence absolue, avec bonus si vraiment ouvert/fermé
    diff = np.abs(op_left - op_right)

    # Filtre : on veut que l'un soit clairement fermé ET l'autre clairement ouvert
    one_closed = (op_left < closed_threshold_mm) | (op_right < closed_threshold_mm)
    one_open   = (op_left > open_threshold_mm)   | (op_right > open_threshold_mm)
    valid_mask = one_closed & one_open

    if valid_mask.sum() == 0:
        print("[WARN] Aucun moment avec seuils stricts — élargissement des critères.")
        # Fallback : top différences sans contrainte
        valid_mask = np.ones(len(grid), dtype=bool)

    valid_diff = np.where(valid_mask, diff, -1.0)

    # Sous-échantillonnage pour éviter les candidats trop proches (≥ 1s d'écart)
    min_gap = int(1_000_000_000 / step_ns)  # 1 seconde
    candidates = []
    used = np.zeros(len(grid), dtype=bool)

    for _ in range(n_candidates * 10):
        if len(candidates) >= n_candidates:
            break
        idx = int(np.argmax(valid_diff))
        if valid_diff[idx] < 0:
            break
        candidates.append({
            "timestamp_ns":   int(grid[idx]),
            "opening_left":   float(op_left[idx]),
            "opening_right":  float(op_right[idx]),
            "score":          float(valid_diff[idx]),
        })
        # Masquer le voisinage
        lo = max(0, idx - min_gap)
        hi = min(len(grid), idx + min_gap)
        valid_diff[lo:hi] = -1.0

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Ajouter temps relatif (depuis le début de la session)
    if candidates:
        t0 = min(left_df["timestamp_ns"].iloc[0], right_df["timestamp_ns"].iloc[0])
        for c in candidates:
            c["time_rel_s"] = (c["timestamp_ns"] - t0) / 1e9

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Extraction frame vidéo
# ─────────────────────────────────────────────────────────────────────────────

def ts_ns_to_frame_index(ts_ns: int,
                          jsonl_indices: np.ndarray,
                          jsonl_ts_ns: np.ndarray) -> int:
    """
    Retourne l'index de frame le plus proche du timestamp donné.
    """
    idx = int(np.argmin(np.abs(jsonl_ts_ns - ts_ns)))
    return int(jsonl_indices[idx])


def extract_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Extrait la frame numéro frame_idx d'une vidéo MP4.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ─────────────────────────────────────────────────────────────────────────────
# Génération image de comparaison
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_image(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    info: dict,
    label_left: str = "left.mp4",
    label_right: str = "right.mp4",
) -> np.ndarray:
    """
    Assemble deux frames côte à côte avec annotations.
    """
    h = max(frame_left.shape[0], frame_right.shape[0])
    target_h = 480
    scale = target_h / h

    def resize(img):
        w = int(img.shape[1] * scale)
        h2 = int(img.shape[0] * scale)
        return cv2.resize(img, (w, h2), interpolation=cv2.INTER_AREA)

    fl = resize(frame_left)
    fr = resize(frame_right)

    # Padding vertical si besoin
    max_h = max(fl.shape[0], fr.shape[0])
    def pad_h(img, target):
        if img.shape[0] < target:
            pad = np.zeros((target - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, pad])
        return img

    fl = pad_h(fl, max_h)
    fr = pad_h(fr, max_h)

    # Bande d'info en haut (60px)
    bar_h = 60
    total_w = fl.shape[1] + fr.shape[1] + 10  # 10px séparation
    bar = np.zeros((bar_h, total_w, 3), dtype=np.uint8)

    ts_rel_s = info.get("time_rel_s", info["timestamp_ns"] / 1e9)
    text = (f"t={ts_rel_s:.1f}s  |  "
            f"{label_left}: {info['opening_left']:.1f}mm  |  "
            f"{label_right}: {info['opening_right']:.1f}mm  |  "
            f"diff={info['score']:.1f}mm")

    cv2.putText(bar, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Étiquettes sur les frames
    def label_frame(img, label, opening_mm):
        img = img.copy()
        color = (0, 255, 0) if opening_mm > 12 else (0, 0, 255)
        state = "OUVERT" if opening_mm > 12 else "FERME"
        cv2.putText(img, f"{label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"{state}  {opening_mm:.1f}mm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # Bordure colorée
        cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), color, 4)
        return img

    fl = label_frame(fl, label_left,  info["opening_left"])
    fr = label_frame(fr, label_right, info["opening_right"])

    # Assemblage
    sep = np.zeros((max_h, 10, 3), dtype=np.uint8)
    row = np.hstack([fl, sep, fr])

    # Pad bar to row width
    if bar.shape[1] < row.shape[1]:
        pad = np.zeros((bar_h, row.shape[1] - bar.shape[1], 3), dtype=np.uint8)
        bar = np.hstack([bar, pad])

    return np.vstack([bar, row])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Identification visuelle gauche/droite des caméras")
    p.add_argument("--session",     required=True, help="Chemin du dossier session")
    p.add_argument("--output_dir",  default=None,  help="Dossier de sortie (défaut: session/identify_side/)")
    p.add_argument("--n_candidates", type=int, default=5, help="Nombre de moments candidats à exporter")
    p.add_argument("--closed_mm",   type=float, default=6.0,  help="Seuil fermeture (mm)")
    p.add_argument("--open_mm",     type=float, default=14.0, help="Seuil ouverture (mm)")
    args = p.parse_args()

    session = Path(args.session)
    output_dir = Path(args.output_dir) if args.output_dir else session / "identify_side"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Fichiers requis ---
    csv_left  = session / "gripper_left_data.csv"
    csv_right = session / "gripper_right_data.csv"
    jsonl_left  = session / "videos" / "left.jsonl"
    jsonl_right = session / "videos" / "right.jsonl"
    video_left  = session / "videos" / "left.mp4"
    video_right = session / "videos" / "right.mp4"

    for f in [csv_left, csv_right, jsonl_left, jsonl_right, video_left, video_right]:
        if not f.exists():
            print(f"[ERREUR] Fichier manquant : {f}", file=sys.stderr)
            sys.exit(1)

    print(f"Session : {session}")

    # --- Chargement ---
    print("Chargement CSV capteurs...")
    left_df  = load_sensor(str(csv_left))
    right_df = load_sensor(str(csv_right))
    print(f"  left  : {len(left_df)} échantillons, "
          f"ouverture [{left_df['opening_mm'].min():.1f}–{left_df['opening_mm'].max():.1f}] mm")
    print(f"  right : {len(right_df)} échantillons, "
          f"ouverture [{right_df['opening_mm'].min():.1f}–{right_df['opening_mm'].max():.1f}] mm")

    print("Chargement timestamps JSONL...")
    idx_left,  ts_left  = load_jsonl_timestamps(str(jsonl_left))
    idx_right, ts_right = load_jsonl_timestamps(str(jsonl_right))

    # --- Recherche moments asymétriques ---
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

    # --- Extraction frames et génération images ---
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

        # Frame index le plus proche dans chaque vidéo
        fi_left  = ts_ns_to_frame_index(ts, idx_left,  ts_left)
        fi_right = ts_ns_to_frame_index(ts, idx_right, ts_right)

        # Extraction frames
        frame_l = extract_frame(str(video_left),  fi_left)
        frame_r = extract_frame(str(video_right), fi_right)

        if frame_l is None or frame_r is None:
            print(f"  [SKIP] Candidat {i+1} — frame non lisible")
            continue

        # Image de comparaison
        comp = make_comparison_image(frame_l, frame_r, cand)
        out_path = output_dir / f"candidate_{i+1:02d}.png"
        cv2.imwrite(str(out_path), comp)

        ts_rel_s = cand.get("time_rel_s", 0.0)
        line = (f"  [{i+1}] t={ts_rel_s:.1f}s  "
                f"left.mp4={cand['opening_left']:.1f}mm  "
                f"right.mp4={cand['opening_right']:.1f}mm  "
                f"diff={cand['score']:.1f}mm  "
                f"→ {out_path.name}")
        print(line)
        report_lines.append(line)

    report_lines += [
        "",
        "COMMENT LIRE LES IMAGES :",
        "  Chaque image montre côte à côte la frame de left.mp4 (gauche) et right.mp4 (droite).",
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
