#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gripper_aruco_tracker.py — Tracking 3D des pinces via ArUco dans la vidéo head.

PRINCIPE
--------
La caméra "head" observe la scène. Chaque pince porte un marqueur ArUco
imprimé de taille connue. Pour chaque frame on calcule la pose 3D du
marqueur (X, Y, Z en mm dans le repère caméra) via solvePnP si une
calibration intrinsèque est disponible, ou via le modèle sténopé simple
(distance = taille_réelle * f_px / taille_pixels) sinon.

SORTIES
-------
  {session}/gripper_aruco_positions.csv — une ligne par (frame, marqueur) :
    frame_idx, timestamp_ns, marker_id, side,
    x_mm, y_mm, z_mm, distance_mm, cx_px, cy_px, pixel_size_px

  {session}/videos/head_aruco_debug.mp4 — vidéo annotée (si --debug-video)

USAGE
-----
  # mode session unique
  python gripper_aruco_tracker.py --session /path/to/session_xxx \
      --marker-size-mm 30 \
      --left-id 1 --right-id 2

  # mode batch (toutes les sessions d'un répertoire)
  python gripper_aruco_tracker.py --sessions-dir /path/to/sessions \
      --marker-size-mm 30 --left-id 1 --right-id 2 \
      --calib camera_calibration.npz --debug-video

  # sans calibration — avec focale connue
  python gripper_aruco_tracker.py --session /path/to/session_xxx \
      --marker-size-mm 30 --left-id 1 --right-id 2 --focal-px 920

  # sans calibration — avec distance connue pour auto-calibrer focal
  python gripper_aruco_tracker.py --session /path/to/session_xxx \
      --marker-size-mm 30 --left-id 1 --right-id 2 \
      --known-distance-mm 400

Installation :
  pip install opencv-contrib-python numpy pandas
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pandas as pd
    PD_OK = True
except ImportError:
    pd = None  # type: ignore
    PD_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Dictionnaires ArUco supportés
# ─────────────────────────────────────────────────────────────────────────────

DICT_NAMES: Dict[str, int] = {
    "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":  cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50":   cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":  cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250":  cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50":   cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100":  cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250":  cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
}


# ─────────────────────────────────────────────────────────────────────────────
# Structures de résultat
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarkerDetection:
    frame_idx:      int
    timestamp_ns:   int       # horloge Unix en ns (0 si JSONL absent)
    marker_id:      int
    side:           str       # "left" | "right" | "unknown"
    x_mm:           float     # repère caméra : droite
    y_mm:           float     # repère caméra : bas
    z_mm:           float     # repère caméra : profondeur
    distance_mm:    float
    cx_px:          float     # centre pixel X
    cy_px:          float     # centre pixel Y
    pixel_size_px:  float     # taille apparente du marqueur (px, moyenne des 4 côtés)


@dataclass
class SessionResult:
    session_name:   str
    success:        bool
    error:          str = ""
    n_frames:       int = 0
    n_detections:   int = 0
    csv_path:       str = ""
    debug_video_path: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Calibration caméra
# ─────────────────────────────────────────────────────────────────────────────

def load_calibration(path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Charge camera_matrix et dist_coeffs depuis un fichier .npz."""
    if not path:
        return None, None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration introuvable : {path}")

    data = np.load(path)
    keys = set(data.files)

    cam_key  = next((k for k in ("camera_matrix", "K", "mtx")     if k in keys), None)
    dist_key = next((k for k in ("dist_coeffs",   "dist", "D")    if k in keys), None)

    if cam_key is None or dist_key is None:
        raise ValueError(
            f"{path} doit contenir camera_matrix/K/mtx et dist_coeffs/dist/D. "
            f"Clés trouvées : {list(keys)}"
        )
    return data[cam_key].astype(np.float64), data[dist_key].astype(np.float64)


def make_pinhole_camera_matrix(fw: int, fh: int, focal_px: float) -> np.ndarray:
    """Matrice intrinsèque minimale (pas de distorsion) pour le mode simple."""
    return np.array([
        [focal_px, 0.0,      fw / 2.0],
        [0.0,      focal_px, fh / 2.0],
        [0.0,      0.0,      1.0     ],
    ], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Détecteur ArUco
# ─────────────────────────────────────────────────────────────────────────────

def make_detector(dict_name: str):
    if dict_name not in DICT_NAMES:
        raise ValueError(f"Dictionnaire inconnu : {dict_name}. Valides : {sorted(DICT_NAMES)}")

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_NAMES[dict_name])
    try:
        params   = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return detector, dictionary, params
    except Exception:
        params = cv2.aruco.DetectorParameters_create()  # type: ignore
        return None, dictionary, params


def detect_markers(gray: np.ndarray, detector, dictionary, params):
    if detector is not None:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


# ─────────────────────────────────────────────────────────────────────────────
# Géométrie des marqueurs
# ─────────────────────────────────────────────────────────────────────────────

def marker_pixel_size(corners: np.ndarray) -> float:
    """Taille apparente moyenne (moy. des 4 côtés) en pixels."""
    c = corners.reshape(4, 2).astype(np.float64)
    sides = [
        np.linalg.norm(c[1] - c[0]),
        np.linalg.norm(c[2] - c[1]),
        np.linalg.norm(c[3] - c[2]),
        np.linalg.norm(c[0] - c[3]),
    ]
    return float(np.mean(sides))


def marker_center(corners: np.ndarray) -> Tuple[float, float]:
    c = corners.reshape(4, 2).astype(np.float64)
    m = c.mean(axis=0)
    return float(m[0]), float(m[1])


def solve_pose(
    corners: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Retourne (rvec, tvec, distance_m).
    tvec[:,0] = [X_cam, Y_cam, Z_cam] en mètres (repère caméra : Z = profondeur).
    """
    half = marker_size_m / 2.0
    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)

    img_pts = corners.reshape(4, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None, None, float("nan")

    x, y, z   = float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0])
    distance_m = math.sqrt(x * x + y * y + z * z)
    return rvec, tvec, distance_m


# ─────────────────────────────────────────────────────────────────────────────
# JSONL timestamps
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_timestamps(jsonl_path: str) -> List[int]:
    """
    Retourne une liste de timestamps en nanosecondes (horloge Unix).
    capture_time dans le JSONL est en millisecondes → × 1_000_000.
    """
    timestamps: List[int] = []
    for line in Path(jsonl_path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            ts_ms = entry.get("capture_time")
            if ts_ms is not None:
                timestamps.append(int(ts_ms) * 1_000_000)
        except (json.JSONDecodeError, ValueError):
            pass
    return timestamps


# ─────────────────────────────────────────────────────────────────────────────
# Annotation vidéo
# ─────────────────────────────────────────────────────────────────────────────

def _draw_text(frame: np.ndarray, text: str, x: int, y: int,
               color=(0, 255, 0), scale: float = 0.55) -> None:
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


SIDE_COLORS = {
    "left":    (0, 220, 0),    # vert
    "right":   (0, 120, 255),  # orange
    "unknown": (180, 180, 180),
}


# ─────────────────────────────────────────────────────────────────────────────
# Traitement d'une session
# ─────────────────────────────────────────────────────────────────────────────

def process_session(
    session_path:    str,
    marker_size_mm:  float,
    id_to_side:      Dict[int, str],
    camera_matrix:   Optional[np.ndarray],
    dist_coeffs:     Optional[np.ndarray],
    focal_px:        Optional[float],
    dict_name:       str      = "DICT_4X4_50",
    debug_video:     bool     = False,
    output_dir:      Optional[str] = None,
    progress_fn=None,   # callable(frame_idx: int, total_frames: int) | None
) -> SessionResult:
    """
    Traite la vidéo head.mp4 d'une session et produit le CSV de positions.

    Paramètres
    ----------
    camera_matrix / dist_coeffs : calibration complète (mode précis solvePnP)
    focal_px : focale en pixels pour le mode simple (ignoré si calibration fournie)
    id_to_side : {marker_id: "left"|"right"}  — IDs attendus
    """
    sname  = Path(session_path).name
    result = SessionResult(session_name=sname, success=False)

    videos_dir  = Path(session_path) / "cameras"
    video_path  = videos_dir / "head.mp4"
    jsonl_path  = videos_dir / "head.jsonl"

    if not video_path.exists():
        result.error = f"head.mp4 absent : {video_path}"
        return result

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result.error = f"Impossible d'ouvrir : {video_path}"
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Timestamps JSONL
    ts_list: List[int] = []
    if jsonl_path.exists():
        ts_list = load_jsonl_timestamps(str(jsonl_path))

    # Calibration effective
    calib_mode = camera_matrix is not None and dist_coeffs is not None
    focal_used = focal_px  # peut être None (auto-calibré sur 1re détection)

    marker_size_m  = marker_size_mm / 1000.0
    detector, dictionary, params = make_detector(dict_name)

    # Sortie CSV
    out_base = Path(output_dir) if output_dir else Path(session_path)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / "gripper_aruco_positions.csv"

    # Vidéo debug
    writer = None
    debug_path = ""
    if debug_video:
        debug_path = str(out_base / "head_aruco_debug.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(debug_path, fourcc, fps, (fw, fh))
        result.debug_video_path = debug_path

    detections: List[MarkerDetection] = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if progress_fn is not None:
            progress_fn(frame_idx, total_frames)

        ts_ns = ts_list[frame_idx] if frame_idx < len(ts_list) else 0
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_list, ids, _ = detect_markers(gray, detector, dictionary, params)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

            for corners, mid in zip(corners_list, ids.flatten()):
                mid     = int(mid)
                px_size = marker_pixel_size(corners)
                cx, cy  = marker_center(corners)
                side    = id_to_side.get(mid, "unknown")

                x_mm = y_mm = z_mm = dist_mm = float("nan")
                rvec_draw = tvec_draw = None

                if calib_mode:
                    rvec_draw, tvec_draw, dist_m = solve_pose(
                        corners, marker_size_m, camera_matrix, dist_coeffs  # type: ignore
                    )
                    if tvec_draw is not None:
                        x_mm  = float(tvec_draw[0, 0]) * 1000.0
                        y_mm  = float(tvec_draw[1, 0]) * 1000.0
                        z_mm  = float(tvec_draw[2, 0]) * 1000.0
                        dist_mm = dist_m * 1000.0
                else:
                    # Auto-calibration focale sur la 1re détection avec distance connue
                    # (si --known-distance-mm, la focale sera passée déjà calculée)
                    if focal_used is not None and px_size > 1:
                        dist_mm = (marker_size_mm * focal_used) / px_size
                        # Estimation X,Y par déprojection pinhole simple
                        # (Z = dist, X/Y depuis centre optique estimé à fw/2, fh/2)
                        x_mm = (cx - fw / 2.0) * dist_mm / focal_used
                        y_mm = (cy - fh / 2.0) * dist_mm / focal_used
                        z_mm = dist_mm

                det = MarkerDetection(
                    frame_idx     = frame_idx,
                    timestamp_ns  = ts_ns,
                    marker_id     = mid,
                    side          = side,
                    x_mm          = x_mm,
                    y_mm          = y_mm,
                    z_mm          = z_mm,
                    distance_mm   = dist_mm,
                    cx_px         = cx,
                    cy_px         = cy,
                    pixel_size_px = px_size,
                )
                detections.append(det)

                # Annotation frame
                color = SIDE_COLORS.get(side, (200, 200, 200))
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                if calib_mode and rvec_draw is not None:
                    cv2.drawFrameAxes(
                        frame, camera_matrix, dist_coeffs,  # type: ignore
                        rvec_draw, tvec_draw, marker_size_m * 0.6,
                    )
                    _draw_text(frame,
                               f"ID{mid}({side})  d={dist_mm:.0f}mm",
                               int(cx) + 10, int(cy) - 14, color)
                    _draw_text(frame,
                               f"X={x_mm:.0f} Y={y_mm:.0f} Z={z_mm:.0f}",
                               int(cx) + 10, int(cy) + 10, color)
                elif focal_used is not None and math.isfinite(dist_mm):
                    _draw_text(frame,
                               f"ID{mid}({side})  d~{dist_mm:.0f}mm",
                               int(cx) + 10, int(cy) - 14, color)
                    _draw_text(frame,
                               f"X={x_mm:.0f} Y={y_mm:.0f} Z={z_mm:.0f}",
                               int(cx) + 10, int(cy) + 10, color)
                else:
                    _draw_text(frame,
                               f"ID{mid}({side})  px={px_size:.1f}",
                               int(cx) + 10, int(cy) - 14, color)
        else:
            _draw_text(frame, "Aucun ArUco détecté", 20, 36, (0, 0, 255))

        # Info overlay (bas de frame)
        mode_label = "solvePnP+calib" if calib_mode else (
            f"pinhole f={focal_used:.0f}px" if focal_used else "focal inconnue"
        )
        _draw_text(
            frame,
            f"frame={frame_idx}  {mode_label}  marker={marker_size_mm:.0f}mm  "
            f"dict={dict_name}",
            10, fh - 12, (220, 220, 220), scale=0.45,
        )

        if writer is not None:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # ── Écriture CSV ──────────────────────────────────────────────────────────
    _write_csv(csv_path, detections)
    result.csv_path     = str(csv_path)
    result.n_frames     = frame_idx
    result.n_detections = len(detections)
    result.success      = True
    return result


def _write_csv(path: Path, detections: List[MarkerDetection]) -> None:
    columns = [
        "frame_idx", "timestamp_ns", "marker_id", "side",
        "x_mm", "y_mm", "z_mm", "distance_mm",
        "cx_px", "cy_px", "pixel_size_px",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for d in detections:
            writer.writerow([
                d.frame_idx, d.timestamp_ns, d.marker_id, d.side,
                _fmt(d.x_mm), _fmt(d.y_mm), _fmt(d.z_mm), _fmt(d.distance_mm),
                f"{d.cx_px:.2f}", f"{d.cy_px:.2f}", f"{d.pixel_size_px:.2f}",
            ])


def _fmt(v: float) -> str:
    return f"{v:.2f}" if math.isfinite(v) else ""


# ─────────────────────────────────────────────────────────────────────────────
# Collecte des sessions
# ─────────────────────────────────────────────────────────────────────────────

def collect_sessions(sessions_dir: str, pattern: str = "session_") -> List[str]:
    return sorted([
        os.path.join(sessions_dir, name)
        for name in os.listdir(sessions_dir)
        if name.startswith(pattern)
           and os.path.isdir(os.path.join(sessions_dir, name))
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tracking 3D des pinces via ArUco dans la vidéo head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source vidéo
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--session",      help="chemin d'une session unique")
    src.add_argument("--sessions-dir", help="répertoire contenant plusieurs sessions")

    # Marqueurs
    parser.add_argument("--marker-size-mm", type=float, required=True,
                        help="taille réelle du côté du marqueur ArUco (mm)")
    parser.add_argument("--left-id",  type=int, default=None,
                        help="ID ArUco du marqueur de la pince gauche")
    parser.add_argument("--right-id", type=int, default=None,
                        help="ID ArUco du marqueur de la pince droite")
    parser.add_argument("--dictionary", default="DICT_4X4_50",
                        choices=sorted(DICT_NAMES.keys()))

    # Calibration
    calib_grp = parser.add_mutually_exclusive_group()
    calib_grp.add_argument("--calib",             default=None,
                           help="fichier .npz de calibration caméra (mode précis)")
    calib_grp.add_argument("--focal-px",          type=float, default=None,
                           help="focale en pixels (mode simple, sans calibration)")
    calib_grp.add_argument("--known-distance-mm", type=float, default=None,
                           help="distance connue pour auto-calculer la focale "
                                "(1re détection utilisée)")

    # Sorties
    parser.add_argument("--output-dir",  default=None,
                        help="répertoire de sortie (défaut : dans chaque session)")
    parser.add_argument("--debug-video", action="store_true",
                        help="générer une vidéo annotée head_aruco_debug.mp4")

    args = parser.parse_args()

    # ── Calibration ──────────────────────────────────────────────────────────
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs:   Optional[np.ndarray] = None
    focal_px:      Optional[float]      = args.focal_px

    if args.calib:
        try:
            camera_matrix, dist_coeffs = load_calibration(args.calib)
            print(f"Calibration chargée : {args.calib}")
        except Exception as e:
            print(f"ERREUR calibration : {e}", file=sys.stderr)
            return 1

    if camera_matrix is None and focal_px is None and args.known_distance_mm is None:
        print(
            "ATTENTION : ni --calib, ni --focal-px, ni --known-distance-mm "
            "fourni. Le script détectera les marqueurs mais ne calculera pas "
            "la distance/position 3D.",
            file=sys.stderr,
        )

    # Auto-calibration focale : sera faite à la 1re détection dans process_session
    # si known_distance_mm est fourni — on passe la focale à None et on la calcule.
    known_distance_mm = args.known_distance_mm

    # ── Mapping ID → côté ────────────────────────────────────────────────────
    id_to_side: Dict[int, str] = {}
    if args.left_id  is not None:
        id_to_side[args.left_id]  = "left"
    if args.right_id is not None:
        id_to_side[args.right_id] = "right"

    if not id_to_side:
        print(
            "ATTENTION : --left-id / --right-id non spécifiés. "
            "Tous les marqueurs détectés seront étiquetés 'unknown'.",
            file=sys.stderr,
        )

    # ── Sessions ─────────────────────────────────────────────────────────────
    if args.session:
        sessions = [args.session]
    else:
        sessions = collect_sessions(args.sessions_dir)
        if not sessions:
            print(f"Aucune session trouvée dans {args.sessions_dir}", file=sys.stderr)
            return 1
        print(f"{len(sessions)} session(s) trouvée(s)")

    n_ok = n_fail = 0

    for session_path in sessions:
        sname = Path(session_path).name
        print(f"  {sname} ...", end=" ", flush=True)

        # Auto-calibration focale par --known-distance-mm :
        # On fait un pré-scan des premières frames pour estimer focal_px
        eff_focal = focal_px
        if camera_matrix is None and focal_px is None and known_distance_mm is not None:
            eff_focal = _auto_focal(
                session_path, args.marker_size_mm,
                known_distance_mm, args.dictionary,
            )
            if eff_focal is not None:
                print(f"focal auto={eff_focal:.1f}px  ", end="", flush=True)

        out_dir = args.output_dir or session_path

        res = process_session(
            session_path   = session_path,
            marker_size_mm = args.marker_size_mm,
            id_to_side     = id_to_side,
            camera_matrix  = camera_matrix,
            dist_coeffs    = dist_coeffs,
            focal_px       = eff_focal,
            dict_name      = args.dictionary,
            debug_video    = args.debug_video,
            output_dir     = out_dir,
        )

        if res.success:
            n_ok += 1
            print(f"OK  frames={res.n_frames}  détections={res.n_detections}  → {res.csv_path}")
        else:
            n_fail += 1
            print(f"FAIL  {res.error}")

    print(f"\nOK={n_ok}  FAIL={n_fail}  sur {len(sessions)} sessions")
    return 0 if n_fail == 0 else 1


def _auto_focal(
    session_path:       str,
    marker_size_mm:     float,
    known_distance_mm:  float,
    dict_name:          str,
    max_frames:         int = 60,
) -> Optional[float]:
    """
    Lit les premières frames de head.mp4 et estime focal_px à partir de
    la taille apparente du marqueur à distance connue.
    """
    video_path = Path(session_path) / "videos" / "head.mp4"
    if not video_path.exists():
        return None

    detector, dictionary, params = make_detector(dict_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    focals: List[float] = []
    frame_count = 0
    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = detect_markers(gray, detector, dictionary, params)
        if ids is not None and len(ids) > 0:
            for corners in corners_list:
                px = marker_pixel_size(corners)
                if px > 2:
                    focals.append(px * known_distance_mm / marker_size_mm)
        frame_count += 1

    cap.release()
    if not focals:
        return None
    return float(np.median(focals))


if __name__ == "__main__":
    sys.exit(main())
