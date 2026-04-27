#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assign_cameras.py — Re-identification des cameras head / left / right.

Pour chaque session, teste toutes les 6 permutations possibles d'assignation
des fichiers videos (*.mp4 + *.jsonl) vers les roles (head, left, right) et
choisit celle qui maximise la correlation gripper video/capteur.

Signal : span horizontal des blobs sombres dans le ROI de chaque frame
         (correlé avec l'ouverture du gripper en mm).

Usage :
    python assign_cameras.py --sessions_dir /path/to/sessions
    python assign_cameras.py --sessions_dir /path/to/sessions --fix
    python assign_cameras.py --session /path/to/single/session --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))

SIDES = ("head", "left", "right")


# ══════════════════════════════════════════════════════════════════════════════
# Chargement données
# ══════════════════════════════════════════════════════════════════════════════

def _load_jsonl(jsonl_path: Path) -> Tuple[np.ndarray, float]:
    """
    Retourne (t_rel_s, epoch0_s).
    t_rel_s[i] = temps relatif de la frame i depuis la première frame (secondes).
    epoch0_s   = timestamp absolu de la première frame (secondes).
    """
    entries = []
    try:
        for line in jsonl_path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "capture_time" in obj:
                    entries.append(float(obj["capture_time"]))
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    if not entries:
        return np.array([], dtype=np.float64), 0.0
    arr = np.array(entries, dtype=np.float64)
    epoch0_s = arr[0] / 1000.0          # capture_time en ms → epoch0 en secondes
    t_rel_s  = (arr - arr[0]) / 1000.0  # relatif en secondes
    return t_rel_s, epoch0_s


def _load_sensor(sensor_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retourne (ts_s, opening_mm) — timestamps en secondes absolus.
    Gère les colonnes timestamp_ns, t_ms, time_seconds automatiquement.
    """
    df = pd.read_csv(sensor_path)
    TIME_PREF = ["timestamp_ns", "t_ms_corrected_ns", "t_ms", "time_seconds"]
    time_col = next(
        (c for c in TIME_PREF if c in df.columns and pd.api.types.is_numeric_dtype(df[c])),
        next((c for c in df.columns
              if pd.api.types.is_numeric_dtype(df[c]) and
              ("time" in c.lower() or "ts" in c.lower())), None),
    )
    open_col = next(
        (c for c in df.columns if c == "opening_mm"),
        next((c for c in df.columns
              if "open" in c.lower() and "mm" in c.lower()), None),
    )
    if time_col is None or open_col is None:
        raise ValueError(f"Colonnes introuvables dans {sensor_path}: {list(df.columns)}")
    ts  = df[time_col].to_numpy(dtype=np.float64)
    val = df[open_col].to_numpy(dtype=np.float64)
    # Conversion vers secondes
    if ts.max() > 1e15:
        ts = ts / 1e9      # nanoseconds → secondes
    elif ts.max() > 1e9:
        ts = ts / 1e3      # millisecondes → secondes
    # Tri temporel
    order = np.argsort(ts)
    return ts[order], val[order]


# ══════════════════════════════════════════════════════════════════════════════
# Extraction signal gripper depuis la vidéo
# ══════════════════════════════════════════════════════════════════════════════

def _frame_gripper_feature(gray: np.ndarray, fw: int, roi_h: int,
                            thr: int, kernel: np.ndarray) -> float:
    """
    Feature = écart-centre entre les 2 plus grands blobs dans le ROI.

    Pour un gripper : les 2 doigts = 2 gros blobs sombres. Quand le gripper
    ouvre, la distance entre leurs centres augmente. C'est un signal bien plus
    discriminant que le span total (qui sature si des blobs de fond sont présents).
    Retourne 0 si < 2 blobs détectés.
    """
    roi = gray[:roi_h, :]
    _, binary = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = roi_h * fw * 0.005
    big = sorted([c for c in contours if cv2.contourArea(c) > min_area],
                 key=cv2.contourArea, reverse=True)

    if len(big) < 2:
        return 0.0

    # Centroïdes des 2 plus grands blobs
    def cx(c):
        M = cv2.moments(c)
        return (M["m10"] / M["m00"]) / fw if M["m00"] > 0 else 0.5

    c0, c1 = cx(big[0]), cx(big[1])
    return abs(c0 - c1)   # distance normalisée [0, 1]


def _gripper_signal_from_video(
    video_path: Path,
    t_rel_s:    np.ndarray,
    roi_frac:   float = 0.65,
    thr_dark:   int   = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrait un signal d'ouverture gripper frame par frame.

    Signal = distance entre les centroïdes des 2 plus grands blobs sombres
             dans le ROI supérieur (les doigts du gripper s'écartent à l'ouverture).
    Retourne (t_s, signal_norm) normalisé [0,1].
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.array([]), np.array([])

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    roi_h  = max(1, int(fh * roi_frac))
    kernel = np.ones((3, 3), np.uint8)

    signals: List[float] = []
    times:   List[float] = []
    fi = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feat = _frame_gripper_feature(gray, fw, roi_h, thr_dark, kernel)
        signals.append(feat)
        t = float(t_rel_s[fi]) if fi < len(t_rel_s) else fi / fps
        times.append(t)
        fi += 1

    cap.release()

    if not signals:
        return np.array([]), np.array([])

    t_arr   = np.array(times,   dtype=np.float64)
    sig_arr = np.array(signals, dtype=np.float64)

    # Normalisation percentile robuste
    lo, hi = np.percentile(sig_arr, 2), np.percentile(sig_arr, 98)
    if hi - lo > 1e-6:
        sig_arr = np.clip((sig_arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        sig_arr = np.zeros_like(sig_arr)

    return t_arr, sig_arr


# ══════════════════════════════════════════════════════════════════════════════
# Corrélation vidéo / capteur
# ══════════════════════════════════════════════════════════════════════════════

def _corr_score(
    t_vid:    np.ndarray,
    sig_vid:  np.ndarray,
    ts_sen:   np.ndarray,
    val_sen:  np.ndarray,
    epoch0_s: float,
) -> float:
    """
    Corrélation (Pearson |r|) entre signal vidéo et signal capteur.
    epoch0_s sert à aligner les timestamps absolus du capteur sur les
    timestamps relatifs de la vidéo.
    """
    if len(t_vid) < 15 or len(ts_sen) < 15:
        return 0.0

    # Capteur aligné sur l'epoch de la vidéo
    ts_rel = ts_sen - epoch0_s

    # Chevauchement temporel minimum 2 secondes
    t_lo = max(float(t_vid[0]),    float(ts_rel[0]))
    t_hi = min(float(t_vid[-1]),   float(ts_rel[-1]))
    if t_hi - t_lo < 2.0:
        return 0.0

    # Normalisation capteur
    s_lo, s_hi = val_sen.min(), val_sen.max()
    if s_hi - s_lo < 0.5:   # capteur quasi-immobile : pas exploitable
        return 0.0
    sen_norm = (val_sen - s_lo) / (s_hi - s_lo)

    # Interpolation capteur sur les timestamps vidéo
    f_sen  = interp1d(ts_rel, sen_norm, kind="linear",
                      bounds_error=False, fill_value=np.nan)
    sen_at = f_sen(t_vid)
    valid  = np.isfinite(sen_at)
    if valid.sum() < 15:
        return 0.0

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = pearsonr(sig_vid[valid], sen_at[valid])
        # On prend la valeur absolue : le signal peut être inversé selon
        # l'éclairage (blob sombre = ouvert ou fermé selon la caméra).
        return abs(float(r)) if math.isfinite(r) else 0.0
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Logique d'assignation principale
# ══════════════════════════════════════════════════════════════════════════════

def assign_session(session_path: Path, verbose: bool = False) -> Dict:
    """
    Détermine la bonne assignation head/left/right pour une session.

    Retourne un dict :
      'assignment'      : {nom_actuel → role_correct}  ex: {'left':'right', ...}
      'current_correct' : bool
      'best_score'      : float  (somme corr_left + corr_right)
      'corr_matrix'     : {nom_video: {sensor_side: corr}}
      'all_perms'       : [(score, (head,left,right)), ...]  trié décroissant
      'error'           : str si problème (absent si OK)
    """
    videos_dir = session_path / "videos"

    video_files  = {s: videos_dir / f"{s}.mp4"                 for s in SIDES}
    jsonl_files  = {s: videos_dir / f"{s}.jsonl"               for s in SIDES}
    sensor_files = {s: session_path / f"gripper_{s}_data.csv"  for s in ("left", "right")}

    # ── Vérification des fichiers requis ──────────────────────────────────────
    missing = [str(p) for p in [*video_files.values(), *sensor_files.values()]
               if not p.exists()]
    if missing:
        return {"error": f"Fichiers absents : {[Path(p).name for p in missing]}",
                "session": str(session_path)}

    # ── Chargement des timestamps JSONL ───────────────────────────────────────
    jsonl_data: Dict[str, Tuple[np.ndarray, float]] = {}
    for s in SIDES:
        if jsonl_files[s].exists():
            jsonl_data[s] = _load_jsonl(jsonl_files[s])
        else:
            jsonl_data[s] = (np.array([]), 0.0)

    # ── Extraction des signaux vidéo ──────────────────────────────────────────
    if verbose:
        print("    Extraction signaux vidéo ...", flush=True)

    vid_signals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for s in SIDES:
        t_rel, epoch0 = jsonl_data[s]
        t_v, sig_v    = _gripper_signal_from_video(video_files[s], t_rel)
        vid_signals[s] = (t_v, sig_v, epoch0)  # type: ignore[assignment]

    # ── Chargement des capteurs ───────────────────────────────────────────────
    sen_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for s in ("left", "right"):
        sen_data[s] = _load_sensor(sensor_files[s])

    # ── Matrice de corrélation : video_side × sensor_side ────────────────────
    # Clé : (video_name, sensor_side) → score [0,1]
    corr: Dict[str, Dict[str, float]] = {v: {} for v in SIDES}
    for v_side in SIDES:
        t_v, sig_v, epoch0 = vid_signals[v_side]  # type: ignore[misc]
        for s_side in ("left", "right"):
            ts_s, val_s = sen_data[s_side]
            corr[v_side][s_side] = _corr_score(t_v, sig_v, ts_s, val_s, epoch0)

    if verbose:
        print(f"    {'':8} {'sensor_left':>12} {'sensor_right':>12}")
        for v in SIDES:
            print(f"    {v:8} {corr[v]['left']:12.4f} {corr[v]['right']:12.4f}")

    # ── Test des 6 permutations ───────────────────────────────────────────────
    # Score = corr(video_left, sensor_left) + corr(video_right, sensor_right)
    # On cherche l'assignation qui maximise ce score.
    best_score = -1.0
    best_perm  = None
    all_perms  = []

    for perm in permutations(SIDES):
        head_v, left_v, right_v = perm
        score = corr[left_v]["left"] + corr[right_v]["right"]
        all_perms.append((score, perm))
        if score > best_score:
            best_score = score
            best_perm  = perm

    all_perms.sort(key=lambda x: x[0], reverse=True)

    # best_perm = (nom_fichier_pour_head, nom_fichier_pour_left, nom_fichier_pour_right)
    if best_perm is None:
        best_perm = tuple(SIDES)

    assignment = {
        best_perm[0]: "head",
        best_perm[1]: "left",
        best_perm[2]: "right",
    }
    current_correct = all(assignment[s] == s for s in SIDES)

    # ── Confiance : marge entre le 1er et le 2ème meilleur score (toutes permutations)
    current_score  = corr["left"]["left"] + corr["right"]["right"]
    second_score   = all_perms[1][0] if len(all_perms) > 1 else 0.0
    margin         = best_score - second_score   # clarté du gagnant

    # Confiance : marge > 0.06 → HIGH ; 0.03-0.06 → LOW ; < 0.03 → UNCERTAIN
    if margin < 0.03 or best_score < 0.20:
        confidence = "UNCERTAIN"
    elif margin < 0.06:
        confidence = "LOW"
    else:
        confidence = "HIGH"

    return {
        "session":         str(session_path),
        "session_name":    session_path.name,
        "assignment":      assignment,
        "current_correct": current_correct,
        "current_score":   round(current_score, 4),
        "best_score":      round(best_score, 4),
        "margin":          round(margin, 4),
        "confidence":      confidence,
        "corr_matrix":     {v: {s: round(corr[v][s], 4) for s in ("left", "right")}
                            for v in SIDES},
        "all_perms":       [(round(sc, 4), list(p)) for sc, p in all_perms],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Application des corrections
# ══════════════════════════════════════════════════════════════════════════════

def apply_fix(session_path: Path, assignment: Dict[str, str]) -> List[str]:
    """
    Renomme les fichiers .mp4 et .jsonl selon l'assignation correcte.
    Utilise des noms temporaires pour éviter les conflits de renommage cyclique.
    Retourne la liste des opérations effectuées.
    """
    videos_dir = session_path / "videos"
    ops: List[str] = []

    # Paires (current_name → correct_role) à renommer
    to_rename = [(cur, role) for cur, role in assignment.items() if cur != role]
    if not to_rename:
        return ["  Assignation déjà correcte — rien à renommer."]

    # Étape 1 : renommer vers noms temporaires (évite les conflits A↔B)
    for cur, _role in to_rename:
        for ext in (".mp4", ".jsonl"):
            src = videos_dir / f"{cur}{ext}"
            if src.exists():
                tmp = videos_dir / f"_tmp_{cur}{ext}"
                src.rename(tmp)
                ops.append(f"    {src.name} → {tmp.name}  (temp)")

    # Étape 2 : renommer des noms temporaires vers les noms définitifs
    for cur, role in to_rename:
        for ext in (".mp4", ".jsonl"):
            tmp = videos_dir / f"_tmp_{cur}{ext}"
            if tmp.exists():
                dst = videos_dir / f"{role}{ext}"
                tmp.rename(dst)
                ops.append(f"    {tmp.name} → {dst.name}")

    return ops


# ══════════════════════════════════════════════════════════════════════════════
# Collecte des sessions
# ══════════════════════════════════════════════════════════════════════════════

def collect_sessions(root: Path) -> List[Path]:
    return sorted(p.parent for p in root.rglob("metadata.json") if p.parent != root)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="assign_cameras.py — Re-identification caméras head/left/right",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sessions_dir", type=Path,
                     help="Dossier racine contenant les sessions")
    grp.add_argument("--session", type=Path,
                     help="Chemin vers une session unique")
    parser.add_argument("--fix", action="store_true",
                        help="Renommer les fichiers mal assignés (défaut : lecture seule)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Afficher la matrice de corrélation et toutes les permutations")
    parser.add_argument("--json_out", metavar="FILE",
                        help="Sauvegarder les résultats en JSON")
    args = parser.parse_args()

    sessions = [args.session] if args.session else collect_sessions(args.sessions_dir)
    if not sessions:
        print("Aucune session trouvée.")
        sys.exit(1)

    mode = "FIX" if args.fix else "TEST"
    print(f"\n{'='*68}")
    print(f"  ASSIGN_CAMERAS — mode {mode}")
    print(f"  {len(sessions)} session(s)")
    print(f"{'='*68}\n")

    all_results = []
    n_ok = n_wrong = n_err = 0

    for sess in sessions:
        print(f"[{sess.name}]")
        result = assign_session(sess, verbose=args.verbose)
        all_results.append(result)

        if "error" in result:
            print(f"  ✗ ERREUR : {result['error']}")
            n_err += 1
            continue

        assignment      = result["assignment"]
        current_correct = result["current_correct"]
        best_score      = result["best_score"]

        confidence  = result["confidence"]
        current_score = result["current_score"]
        margin      = result["margin"]
        conf_icon   = {"HIGH": "★", "LOW": "△", "UNCERTAIN": "?"}.get(confidence, "?")

        if current_correct:
            print(f"  ✓ Correcte  score={best_score:.4f}  confiance={confidence} {conf_icon}")
            n_ok += 1
        else:
            print(f"  ✗ Incorrecte  best={best_score:.4f}  actuel={current_score:.4f}"
                  f"  marge={margin:.4f}  confiance={confidence} {conf_icon}")
            for cur, role in sorted(assignment.items()):
                if cur != role:
                    print(f"      {cur}.mp4/.jsonl  →  {role}.mp4/.jsonl")
            n_wrong += 1

            if args.fix:
                if confidence == "UNCERTAIN":
                    print("  ⚠ FIX annulé : confiance UNCERTAIN (marge trop faible).")
                    print("    Vérifiez manuellement ou relancez sur une session plus longue.")
                else:
                    ops = apply_fix(sess, assignment)
                    print("  Renommages :")
                    for op in ops:
                        print(op)

        if args.verbose:
            mx = result["corr_matrix"]
            print(f"\n    Matrice de corrélation :")
            print(f"    {'':8} {'sensor_left':>12} {'sensor_right':>12}")
            for v in SIDES:
                print(f"    {v:8} {mx[v]['left']:12.4f} {mx[v]['right']:12.4f}")
            print(f"\n    Toutes les permutations (head→left→right) :")
            for sc, perm in result["all_perms"]:
                marker = " ← CHOISI" if tuple(perm) == (
                    [k for k, v in assignment.items() if v == "head"][0],
                    [k for k, v in assignment.items() if v == "left"][0],
                    [k for k, v in assignment.items() if v == "right"][0],
                ) else ""
                print(f"      {sc:.4f}  {perm[0]}→head  {perm[1]}→left  {perm[2]}→right{marker}")
            print()

    print(f"\n{'='*68}")
    print(f"  RÉSUMÉ : {len(sessions)} sessions")
    print(f"  ✓ Correctes   : {n_ok}")
    print(f"  ✗ Incorrectes : {n_wrong}")
    print(f"  ! Erreurs     : {n_err}")
    print(f"{'='*68}\n")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(all_results, indent=2, default=str), encoding="utf-8"
        )
        print(f"  Rapport JSON → {args.json_out}")


if __name__ == "__main__":
    main()
