#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify.py — Script unique de vérification du pipeline de capture.

Regroupe en un seul fichier :
  • Logique trakeur      : identification géométrique head / left / right
  • Logique session_pinces : alignement horloge vidéo / capteur + correction gaps
  • Orchestrateur verify : exécute tous les vérificateurs et produit le rapport

─────────────────────────────────────────────────────────────────────
MODES
─────────────────────────────────────────────────────────────────────
  --test   Lecture seule. Mesure tout, écrit un rapport complet.
           Code de sortie : 0 = OK, 1 = warnings, 2 = erreurs.

  --fix    Applique les corrections (camera_offset) puis re-vérifie.
           Code de sortie : 0 = tout corrigé, 2 = erreurs résiduelles.

─────────────────────────────────────────────────────────────────────
VÉRIFICATEURS (dans l'ordre d'exécution)
─────────────────────────────────────────────────────────────────────
  [1] check.py          — Score de synchronisation IA (0–100%)
  [2] fix_camera_offset — Décalage caméra / tracker
  [3] trakeur           — Précision identification head / left / right
  [4] pinces            — Alignement horloge vidéo / capteur

─────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────
  python verify.py --sessions_dir /path/to/sessions --test
  python verify.py --sessions_dir /path/to/sessions --fix
  python verify.py --sessions_dir /path/to/sessions --test --verbose
  python verify.py --sessions_dir /path/to/sessions --test --no_check
"""

from __future__ import annotations

# ═════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═════════════════════════════════════════════════════════════════════════════

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import correlate, find_peaks, savgol_filter
from scipy.stats import linregress, pearsonr

# ─── Racine du projet dans sys.path ──────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
for _p in [str(_ROOT), str(_ROOT / "verification")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRAKEUR
# Identification géométrique des trackers VR : head / left / right
# ═════════════════════════════════════════════════════════════════════════════

# ─── Utilitaires mathématiques ───────────────────────────────────────────────

def _trk_moving_average(arr: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(arr[:, j], kernel, mode="same")
    return out


def _trk_rank_points(values: np.ndarray, higher_better: bool = True) -> np.ndarray:
    """Attribue 0 / 1 / 2 points aux 3 valeurs (pire / moyen / meilleur)."""
    values = np.asarray(values, dtype=float)
    order  = np.argsort(values)
    pts    = np.zeros(len(values), dtype=float)
    if higher_better:
        pts[order[0]] = 0.0
        pts[order[1]] = 1.0
        pts[order[2]] = 2.0
    else:
        pts[order[0]] = 2.0
        pts[order[1]] = 1.0
        pts[order[2]] = 0.0
    return pts


def _trk_quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    q = q.astype(float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z);  R[:, 0, 1] = 2*(x*y - z*w);  R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w);      R[:, 1, 1] = 1-2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w);      R[:, 2, 1] = 2*(y*z + x*w);  R[:, 2, 2] = 1-2*(x*x + y*y)
    return R


def _trk_quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    q = q.astype(float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z);  R[:, 0, 1] = 2*(x*y - z*w);  R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w);      R[:, 1, 1] = 1-2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w);      R[:, 2, 1] = 2*(y*z + x*w);  R[:, 2, 2] = 1-2*(x*x + y*y)
    return R


# ─── Chargement données tracker ──────────────────────────────────────────────

def trk_split_blocks(
    df: pd.DataFrame,
    meta_cols: int = 3,
    block_size: int = 7,
    smooth_window: int = 9,
) -> list:
    data    = df.iloc[:, meta_cols:].to_numpy(dtype=float)
    n_blocks = data.shape[1] // block_size
    if n_blocks != 3:
        raise ValueError(f"Expected 3 tracker blocks, found {n_blocks}")
    blocks = []
    for i in range(n_blocks):
        block = data[:, i * block_size:(i + 1) * block_size]
        pos   = _trk_moving_average(block[:, :3], smooth_window)
        quat  = block[:, 3:7]
        blocks.append((i, pos, quat))
    return blocks


def trk_parse_truth_from_headers(
    df: pd.DataFrame,
    meta_cols: int = 3,
    block_size: int = 7,
) -> Dict[str, int]:
    cols    = list(df.columns)[meta_cols:]
    n_blocks = len(cols) // block_size
    if n_blocks != 3:
        raise ValueError(f"Expected 3 tracker blocks in headers, found {n_blocks}")
    truth = {}
    for i in range(n_blocks):
        block_cols = cols[i * block_size:(i + 1) * block_size]
        joined = " ".join(str(c).lower() for c in block_cols)
        found  = [lbl for lbl in ("head", "left", "right") if lbl in joined]
        if len(found) != 1:
            raise ValueError(f"Cannot infer unique label for block {i}: {block_cols}")
        truth[found[0]] = i
    if set(truth.keys()) != {"head", "left", "right"}:
        raise ValueError(f"Incomplete truth mapping: {truth}")
    return truth


# ─── Détection du head ───────────────────────────────────────────────────────

def trk_detect_head(blocks: list) -> Tuple[int, np.ndarray]:
    """
    Score pondéré sur 5 critères, avec poids adaptatifs selon la qualité du signal.

    Critères :
      - motion    : le head bouge le moins (poids fixe 1)
      - spread    : le head est le moins dispersé (poids fixe 1)
      - pair_mean : le head est le plus central entre les deux mains
                    (poids adaptatif : 3 × (1 + 2 × sep_pair))
      - h_y       : le head est le plus haut sur l'axe Y (vertical physique VR)
                    (poids adaptatif : 6 × (1 + 2 × sep_hy))
      - h_other   : meilleur axe secondaire X ou Z (poids fixe 1)

    Les poids adaptatifs amplifient le signal quand le leader est clairement
    isolé (grande séparation relative), et le réduisent quand les valeurs sont
    proches (signal ambigu). Cela résout les cas où une main levée compétitionne
    avec la tête sur l'axe Y, et les cas où la centralité pair_mean est un
    signal fort pour le head.
    """
    centers, motions, spreads = [], [], []
    for _, pos, _ in blocks:
        center = np.median(pos, axis=0)
        centers.append(center)
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        motions.append(np.median(step))
        radial = np.linalg.norm(pos - center, axis=1)
        spreads.append(np.median(radial))

    centers = np.asarray(centers)
    motions = np.asarray(motions)
    spreads = np.asarray(spreads)

    # Centralité dynamique : distance médiane aux deux autres trackers
    pair_mean = np.zeros(3, dtype=float)
    for i in range(3):
        d = [np.median(np.linalg.norm(blocks[i][1] - blocks[j][1], axis=1))
             for j in range(3) if i != j]
        pair_mean[i] = np.mean(d)

    # Axe Y = axe vertical physique dans le repère VR (jamais inversé)
    h_y = centers[:, 1]

    # Meilleur axe secondaire parmi X et Z (les deux signes autorisés)
    best_sep_other = -np.inf
    best_h_other   = np.zeros(3, dtype=float)
    for axis in (0, 2):
        for sign in (-1, 1):
            h   = sign * centers[:, axis]
            hs  = np.sort(h)
            sep = (hs[-1] - hs[-2]) / (np.ptp(h) + 1e-9)
            if sep > best_sep_other:
                best_sep_other = sep
                best_h_other   = h

    # Séparation relative : (leader - 2ème) / range — mesure la clarté du signal
    def _sep(vals: np.ndarray, higher: bool) -> float:
        sv = np.sort(vals)
        gap = (sv[-1] - sv[-2]) if higher else (sv[1] - sv[0])
        return float(gap / (np.ptp(vals) + 1e-9))

    sep_hy   = _sep(h_y,       higher=True)
    sep_pair = _sep(pair_mean, higher=False)

    # Poids adaptatifs : base × (1 + boost × séparation)
    w_hy   = 6.0 * (1.0 + 2.0 * sep_hy)
    w_pair = 3.0 * (1.0 + 2.0 * sep_pair)

    score = (
        1.0    * _trk_rank_points(motions,      higher_better=False) +
        1.0    * _trk_rank_points(spreads,      higher_better=False) +
        w_pair * _trk_rank_points(pair_mean,    higher_better=False) +
        w_hy   * _trk_rank_points(h_y,          higher_better=True)  +
        1.0    * _trk_rank_points(best_h_other, higher_better=True)
    )

    return int(np.argmax(score)), score


# ─── Prédiction des mains ────────────────────────────────────────────────────

def trk_predict_hands_with_rule(
    blocks: list,
    head_idx: int,
    quat_mode: str,
    axis: int,
    sign: int,
) -> Tuple[int, int]:
    """
    Règle globale figée (best global rule = wxyz, axis=0, sign=+1).
    Projette les deux trackers non-head sur l'axe "gauche/droite"
    de la tête pour les distinguer.
    """
    head   = next(b for b in blocks if b[0] == head_idx)
    others = [b for b in blocks if b[0] != head_idx]

    _, head_pos, head_quat = head
    idx_a, pos_a, _ = others[0]
    idx_b, pos_b, _ = others[1]

    if quat_mode == "xyzw":
        R = _trk_quat_to_rotmat_xyzw(head_quat)
    elif quat_mode == "wxyz":
        R = _trk_quat_to_rotmat_wxyz(head_quat)
    else:
        raise ValueError(f"Unknown quat_mode: {quat_mode}")

    basis  = sign * R[:, :, axis]
    proj_a = np.sum((pos_a - head_pos) * basis, axis=1)
    proj_b = np.sum((pos_b - head_pos) * basis, axis=1)

    if float(np.median(proj_a)) <= float(np.median(proj_b)):
        return idx_a, idx_b   # left, right
    return idx_b, idx_a


# ─── Évaluation leave-one-out ────────────────────────────────────────────────

def trk_collect_sessions(root_path: str) -> list:
    sessions = []
    for name in sorted(os.listdir(root_path)):
        session_path = os.path.join(root_path, name)
        if not os.path.isdir(session_path):
            continue
        csv_path = os.path.join(session_path, "tracker_positions.csv")
        if not os.path.exists(csv_path):
            continue
        df     = pd.read_csv(csv_path)
        blocks = trk_split_blocks(df)
        truth  = trk_parse_truth_from_headers(df)
        pred_head, head_score = trk_detect_head(blocks)
        sessions.append({
            "name": name, "csv_path": csv_path,
            "blocks": blocks, "truth": truth,
            "pred_head": pred_head, "head_score": head_score,
        })
    return sessions


def _trk_candidate_rules() -> list:
    return [
        (qm, ax, sg)
        for qm in ("xyzw", "wxyz")
        for ax in (0, 1, 2)
        for sg in (-1, 1)
    ]


def trk_choose_best_global_rule(
    train_sessions: list,
) -> Tuple[Tuple[str, int, int], float, int, int]:
    best_rule, best_acc, best_ok, best_total = ("xyzw", 0, -1), -1.0, 0, 0

    for quat_mode, axis, sign in _trk_candidate_rules():
        ok = total = 0
        for s in train_sessions:
            if s["pred_head"] != s["truth"]["head"]:
                continue
            pred_left, pred_right = trk_predict_hands_with_rule(
                s["blocks"], s["pred_head"], quat_mode, axis, sign
            )
            ok    += int(pred_left  == s["truth"]["left"])
            ok    += int(pred_right == s["truth"]["right"])
            total += 2
        acc = ok / total if total > 0 else -1.0
        if acc > best_acc:
            best_acc   = acc
            best_rule  = (quat_mode, axis, sign)
            best_ok    = ok
            best_total = total

    return best_rule, best_acc, best_ok, best_total


def trk_evaluate_leave_one_out(sessions: list) -> list:
    results = []
    for i, test_session in enumerate(sessions):
        train_sessions = [s for j, s in enumerate(sessions) if j != i]
        rule, rule_acc, _, _ = trk_choose_best_global_rule(train_sessions)

        pred_head               = test_session["pred_head"]
        pred_left, pred_right   = trk_predict_hands_with_rule(
            test_session["blocks"], pred_head, rule[0], rule[1], rule[2]
        )

        pred  = {"head": pred_head, "left": pred_left, "right": pred_right}
        truth = test_session["truth"]

        correct = {k: pred[k] == truth[k] for k in ("head", "left", "right")}
        results.append({
            "name": test_session["name"],
            "pred": pred, "truth": truth, "correct": correct,
            "exact": all(correct.values()),
            "rule": rule, "rule_acc_train": rule_acc,
            "head_score": test_session["head_score"],
        })
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SESSION PINCES
# Alignement horloge vidéo / capteur + correction des gaps
# ═════════════════════════════════════════════════════════════════════════════

# ─── Constantes ──────────────────────────────────────────────────────────────

_NOMINAL_FPS      = 30.0
_NOMINAL_PERIOD_S = 1.0 / _NOMINAL_FPS
_DROP_THRESHOLD_S = 2.5 * _NOMINAL_PERIOD_S   # > 83ms = frame drop


# ─── Seuils configurables ────────────────────────────────────────────────────

@dataclass
class Thresholds:
    offset_ms:      float = 200.0   # |offset démarrage vidéo / capteur| max (ms)
    latency_max_ms: float = 25.0    # latence max frame → capteur (ms)
    jitter_std_ms:  float = 15.0    # jitter MAD timestamps vidéo (ms)
    max_drops:      int   = 5       # nb de frame drops autorisés
    max_vel_mm_s:   float = 2000.0  # vitesse ouverture physique max (mm/s)
    min_overlap_s:  float = 3.0     # recouvrement vidéo / capteur minimum (s)


# ─── Chargement JSONL (timestamps vidéo) ─────────────────────────────────────

def pnc_load_jsonl_timestamps(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse videos/{side}.jsonl.
    Retourne (indices int32, ts_ns int64).
    capture_time est en millisecondes → × 1 000 000 = ns.
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
    return indices[order], ts_ns[order]


# ─── Chargement signal capteur ────────────────────────────────────────────────

def pnc_load_sensor(path: str) -> pd.DataFrame:
    """
    Parse gripper_{side}_data.csv.
    Retourne un DataFrame trié par timestamp_ns avec :
      timestamp_ns (int64), opening_mm (float), dt_ms (float)
    Le tri corrige le bug systémique : les ~12 premières lignes arrivent
    hors ordre (buffer de démarrage vidé après le reste du flux).
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
        raise RuntimeError(f"Capteur vide après nettoyage : {path}")
    dt = np.diff(df["timestamp_ns"].values, prepend=df["timestamp_ns"].values[0]) / 1e6
    df["dt_ms"] = dt
    return df


# ─── Correction capteur : comblement de gaps + extrapolation arrière ──────────

def pnc_fill_sensor_gaps(
    ts_ns: np.ndarray,
    opening_mm: np.ndarray,
    dt_nominal_ms: float = 17.0,
    max_opening_change_mm: float = 2.0,
    max_gap_ms: float = 1500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Comble les gaps dans le signal capteur par interpolation linéaire.

    Règle :
    - Gap ≤ 4 × dt_nominal     → jitter normal, ignoré
    - Gap > 4 × dt et ≤ max_gap_ms et |Δopening| ≤ max_opening_change_mm
      → interpolation linéaire (≈ constante si pince immobile)
    - Gap > max_gap_ms ou Δopening > max_opening_change_mm
      → conservé tel quel (signal dynamique ou trop long)
    """
    if len(ts_ns) < 2:
        return ts_ns.copy(), opening_mm.copy()

    gap_thresh_ns = 4.0 * dt_nominal_ms * 1e6
    max_gap_ns    = max_gap_ms * 1e6
    filled_ts: list = list(ts_ns)
    filled_op: list = list(opening_mm.astype(float))

    for i, gap_ns in enumerate(np.diff(ts_ns)):
        if gap_ns <= gap_thresh_ns:
            continue
        gap_ms   = gap_ns / 1e6
        delta_op = abs(float(opening_mm[i + 1]) - float(opening_mm[i]))
        if gap_ms > max_gap_ms or delta_op > max_opening_change_mm:
            continue
        n = max(1, int(round(gap_ms / dt_nominal_ms)) - 1)
        synth_ts = np.linspace(
            ts_ns[i] + dt_nominal_ms * 1e6,
            ts_ns[i + 1] - dt_nominal_ms * 1e6,
            n, dtype=np.int64,
        )
        synth_op = np.linspace(float(opening_mm[i]), float(opening_mm[i + 1]), n)
        filled_ts.extend(synth_ts.tolist())
        filled_op.extend(synth_op.tolist())

    order = np.argsort(filled_ts)
    return (
        np.array(filled_ts, dtype=np.int64)[order],
        np.array(filled_op, dtype=np.float64)[order],
    )


def pnc_extend_sensor_backward(
    ts_ns: np.ndarray,
    opening_mm: np.ndarray,
    vid_start_ns: int,
    dt_nominal_ms: float = 17.0,
    max_extrap_ms: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Étend le capteur vers l'arrière si la vidéo commence avant le capteur.
    Extrapole à valeur constante = première valeur connue du capteur.
    Limite : max_extrap_ms (au-delà, état inconnu de la pince).
    """
    gap_ms = (ts_ns[0] - vid_start_ns) / 1e6
    if gap_ms <= 0 or gap_ms > max_extrap_ms:
        return ts_ns, opening_mm
    dt_ns    = int(dt_nominal_ms * 1e6)
    synth_ts = np.arange(vid_start_ns, ts_ns[0], dt_ns, dtype=np.int64)
    if len(synth_ts) == 0:
        return ts_ns, opening_mm
    synth_op = np.full(len(synth_ts), float(opening_mm[0]))
    return (
        np.concatenate([synth_ts, ts_ns]),
        np.concatenate([synth_op, opening_mm.astype(float)]),
    )


def pnc_apply_sensor_fixes(
    sensor_df: pd.DataFrame,
    vid_start_ns: int,
    dt_nominal_ms: float = 17.0,
) -> pd.DataFrame:
    """
    Applique au DataFrame capteur :
    1. Comblement des gaps statiques  (pnc_fill_sensor_gaps)
    2. Extrapolation arrière          (pnc_extend_sensor_backward)
    """
    ts = sensor_df["timestamp_ns"].values.astype(np.int64)
    op = sensor_df["opening_mm"].values.astype(np.float64)
    ts, op = pnc_fill_sensor_gaps(ts, op, dt_nominal_ms=dt_nominal_ms)
    ts, op = pnc_extend_sensor_backward(ts, op, vid_start_ns, dt_nominal_ms=dt_nominal_ms)
    dt = np.diff(ts, prepend=ts[0]).astype(float) / 1e6
    return pd.DataFrame({"timestamp_ns": ts, "opening_mm": op, "dt_ms": dt})


# ─── Métriques timestamps vidéo ──────────────────────────────────────────────

@dataclass
class VideoTimestampMetrics:
    n_frames:        int
    duration_s:      float
    dt_mean_ms:      float
    dt_std_ms:       float
    dt_min_ms:       float
    dt_max_ms:       float
    frame_drops:     int
    missing_indices: int
    jitter_std_ms:   float
    ts_ns:           np.ndarray = field(repr=False)
    indices:         np.ndarray = field(repr=False)


def pnc_analyze_video_timestamps(
    indices: np.ndarray, ts_ns: np.ndarray
) -> VideoTimestampMetrics:
    dt_ms    = np.diff(ts_ns) / 1e6
    drops    = int((dt_ms > _DROP_THRESHOLD_S * 1000).sum())
    expected = np.arange(indices[0], indices[-1] + 1)
    missing  = int(len(expected) - len(indices))
    mad      = np.median(np.abs(dt_ms - np.median(dt_ms)))
    return VideoTimestampMetrics(
        n_frames        = len(ts_ns),
        duration_s      = float((ts_ns[-1] - ts_ns[0]) / 1e9),
        dt_mean_ms      = float(dt_ms.mean()),
        dt_std_ms       = float(dt_ms.std()),
        dt_min_ms       = float(dt_ms.min()),
        dt_max_ms       = float(dt_ms.max()),
        frame_drops     = drops,
        missing_indices = missing,
        jitter_std_ms   = float(mad * 1.4826),
        ts_ns           = ts_ns,
        indices         = indices,
    )


# ─── Métriques capteur ───────────────────────────────────────────────────────

@dataclass
class SensorMetrics:
    n_samples:      int
    duration_s:     float
    dt_mean_ms:     float
    dt_std_ms:      float
    neg_dt_count:   int
    neg_dt_details: List[Dict]
    vel_max_mm_s:   float
    vel_p99_mm_s:   float
    vel_anomalies:  int
    opening_range:  Tuple[float, float]


def pnc_analyze_sensor(df: pd.DataFrame, max_vel: float) -> SensorMetrics:
    ts  = df["timestamp_ns"].values
    op  = df["opening_mm"].values
    dt  = df["dt_ms"].values[1:]
    neg_idx = np.where(dt < 0)[0]
    neg_details = [
        {"idx": int(i), "dt_ms": float(dt[i]),
         "opening_before": float(op[i]), "opening_after": float(op[i + 1])}
        for i in neg_idx
    ]
    safe_dt_s = np.maximum(dt, 1.0) / 1000.0
    vel  = np.abs(np.diff(op)) / safe_dt_s
    return SensorMetrics(
        n_samples      = len(df),
        duration_s     = float((ts[-1] - ts[0]) / 1e9),
        dt_mean_ms     = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.0,
        dt_std_ms      = float(dt[dt > 0].std()) if (dt > 0).sum() > 1 else 0.0,
        neg_dt_count   = len(neg_idx),
        neg_dt_details = neg_details,
        vel_max_mm_s   = float(vel.max()) if len(vel) else 0.0,
        vel_p99_mm_s   = float(np.percentile(vel, 99)) if len(vel) else 0.0,
        vel_anomalies  = int((vel > max_vel).sum()),
        opening_range  = (float(op.min()), float(op.max())),
    )


# ─── Métriques alignement ────────────────────────────────────────────────────

@dataclass
class AlignmentMetrics:
    dur_vid_s:              float
    dur_sensor_s:           float
    overlap_s:              float
    offset_start_ms:        float
    latency_mean_ms:        float
    latency_std_ms:         float
    latency_max_abs_ms:     float
    latency_p95_abs_ms:     float
    frames_no_sensor:       int
    sensor_gap_max_ms:      float
    sensor_gap_count:       int
    linfit_slope:           float
    linfit_r2:              float
    linfit_residual_std_ms: float
    linfit_residual_max_ms: float
    opening_at_frames:      np.ndarray = field(repr=False)
    frame_ts_ns:            np.ndarray = field(repr=False)

    @property
    def drift_ms(self) -> float:
        return float(self.linfit_slope * (self.dur_vid_s * _NOMINAL_FPS) - self.linfit_slope)

    @property
    def drift_rate_ms_s(self) -> float:
        return float((self.linfit_slope - _NOMINAL_PERIOD_S * 1e9) / 1e6)


def pnc_compute_alignment(
    vid: VideoTimestampMetrics,
    sensor_df: pd.DataFrame,
) -> AlignmentMetrics:
    tv      = vid.ts_ns.astype(np.float64)
    ts      = sensor_df["timestamp_ns"].values.astype(np.float64)
    opening = sensor_df["opening_mm"].values

    dur_vid_s    = float((tv[-1] - tv[0]) / 1e9)
    dur_sensor_s = float((ts[-1] - ts[0]) / 1e9)
    t_ov0 = max(tv[0], ts[0]);  t_ov1 = min(tv[-1], ts[-1])
    overlap_s = float((t_ov1 - t_ov0) / 1e9) if t_ov1 > t_ov0 else 0.0
    offset_start_ms = float((tv[0] - ts[0]) / 1e6)

    # Latence frame → capteur le plus proche
    nearest_hi = np.clip(np.searchsorted(ts, tv), 0, len(ts) - 1)
    nearest_lo = np.clip(nearest_hi - 1, 0, len(ts) - 1)
    nearest_ts = np.where(
        np.abs(tv - ts[nearest_hi]) <= np.abs(tv - ts[nearest_lo]),
        ts[nearest_hi], ts[nearest_lo]
    )
    latency_ms = (tv - nearest_ts) / 1e6
    in_range   = (tv >= ts[0]) & (tv <= ts[-1])
    n_no_sensor = int((~in_range).sum())
    lat_valid   = latency_ms[in_range]
    if len(lat_valid) > 0:
        lat_mean = float(lat_valid.mean());  lat_std = float(lat_valid.std())
        lat_max  = float(np.abs(lat_valid).max())
        lat_p95  = float(np.percentile(np.abs(lat_valid), 95))
    else:
        lat_mean = lat_std = lat_max = lat_p95 = np.nan

    # Continuité capteur dans la fenêtre vidéo
    ts_in_vid    = ts[(ts >= tv[0]) & (ts <= tv[-1])]
    dt_nominal_ms = float(np.median(np.diff(ts)) / 1e6)
    if len(ts_in_vid) > 1:
        gaps_ms          = np.diff(ts_in_vid) / 1e6
        sensor_gap_max_ms = float(gaps_ms.max())
        sensor_gap_count  = int((gaps_ms > 4.0 * dt_nominal_ms).sum())
    else:
        sensor_gap_max_ms = float("inf");  sensor_gap_count = 0

    # Régression linéaire sur timestamps vidéo
    if len(tv) > 10:
        idx = np.arange(len(tv), dtype=np.float64)
        slope_ns, intercept_ns, r, _, _ = linregress(idx, tv)
        r2          = r ** 2
        fitted      = slope_ns * idx + intercept_ns
        residuals_ms = (tv - fitted) / 1e6
        lf_std = float(residuals_ms.std());  lf_max = float(np.abs(residuals_ms).max())
    else:
        slope_ns = (tv[-1] - tv[0]) / max(len(tv) - 1, 1)
        r2 = lf_std = lf_max = np.nan

    # Interpolation capteur aux frames
    f_opening = interp1d(ts, opening, kind="linear", bounds_error=False, fill_value=np.nan)
    opening_at_frames = f_opening(tv)

    return AlignmentMetrics(
        dur_vid_s=dur_vid_s, dur_sensor_s=dur_sensor_s, overlap_s=overlap_s,
        offset_start_ms=offset_start_ms,
        latency_mean_ms=lat_mean, latency_std_ms=lat_std,
        latency_max_abs_ms=lat_max, latency_p95_abs_ms=lat_p95,
        frames_no_sensor=n_no_sensor,
        sensor_gap_max_ms=sensor_gap_max_ms, sensor_gap_count=sensor_gap_count,
        linfit_slope=float(slope_ns), linfit_r2=float(r2),
        linfit_residual_std_ms=lf_std, linfit_residual_max_ms=lf_max,
        opening_at_frames=opening_at_frames, frame_ts_ns=vid.ts_ns,
    )


# ─── Cohérence physique ───────────────────────────────────────────────────────

@dataclass
class PhysicalCoherenceMetrics:
    n_frames_with_sensor: int
    n_frames_no_sensor:   int
    opening_mean_mm:      float
    opening_std_mm:       float
    opening_range:        Tuple[float, float]
    d_opening_max_mm_s:   float
    d_opening_p99_mm_s:   float
    impossible_jumps:     int


def pnc_compute_physical_coherence(
    alignment: AlignmentMetrics, max_vel: float
) -> PhysicalCoherenceMetrics:
    op    = alignment.opening_at_frames
    ts_ns = alignment.frame_ts_ns.astype(np.float64)
    valid = np.isfinite(op)
    n_ok  = int(valid.sum());  n_miss = int((~valid).sum())
    if n_ok < 2:
        return PhysicalCoherenceMetrics(
            n_ok, n_miss, np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, 0
        )
    op_valid = op[valid];  ts_valid = ts_ns[valid]
    dt_s = np.diff(ts_valid) / 1e9
    vel  = np.abs(np.diff(op_valid)) / np.maximum(dt_s, 1e-6)
    return PhysicalCoherenceMetrics(
        n_frames_with_sensor = n_ok,
        n_frames_no_sensor   = n_miss,
        opening_mean_mm      = float(op_valid.mean()),
        opening_std_mm       = float(op_valid.std()),
        opening_range        = (float(op_valid.min()), float(op_valid.max())),
        d_opening_max_mm_s   = float(vel.max()),
        d_opening_p99_mm_s   = float(np.percentile(vel, 99)),
        impossible_jumps     = int((vel > max_vel).sum()),
    )


# ─── Alertes ─────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    code:      str
    level:     str   # "ERROR" | "WARNING"
    message:   str
    value:     float
    threshold: float


def pnc_generate_alerts(
    vid: VideoTimestampMetrics,
    sen: SensorMetrics,
    aln: AlignmentMetrics,
    phy: PhysicalCoherenceMetrics,
    thr: Thresholds,
) -> List[Alert]:
    alerts: List[Alert] = []

    def add(code, level, msg, value, threshold):
        alerts.append(Alert(code=code, level=level, message=msg,
                            value=value, threshold=threshold))

    ONE_FRAME_MS = 1000.0 / _NOMINAL_FPS
    abs_offset   = abs(aln.offset_start_ms)

    if aln.offset_start_ms < -ONE_FRAME_MS:
        add("OFFSET_START", "ERROR",
            f"Vidéo démarre {aln.offset_start_ms:+.1f}ms avant le capteur — capteur manquant au début",
            abs_offset, ONE_FRAME_MS)
    elif abs_offset > thr.offset_ms:
        add("OFFSET_START", "ERROR",
            f"Offset démarrage {aln.offset_start_ms:+.1f}ms > seuil {thr.offset_ms:.0f}ms  "
            f"[corrigeable par fix_camera_offset]",
            abs_offset, thr.offset_ms)

    if math.isfinite(aln.latency_max_abs_ms) and aln.latency_max_abs_ms > thr.latency_max_ms:
        has_gap = aln.sensor_gap_count > 0
        level   = "WARNING" if has_gap else (
            "ERROR" if aln.latency_max_abs_ms > thr.latency_max_ms * 2 else "WARNING"
        )
        note = f"  (causée par gap capteur {aln.sensor_gap_max_ms:.0f}ms)" if has_gap else ""
        add("LATENCY_MAX", level,
            f"Latence max {aln.latency_max_abs_ms:.1f}ms > seuil {thr.latency_max_ms:.0f}ms "
            f"(P95={aln.latency_p95_abs_ms:.1f}ms){note}",
            aln.latency_max_abs_ms, thr.latency_max_ms)

    if aln.frames_no_sensor > 1:
        frac  = aln.frames_no_sensor / max(vid.n_frames, 1)
        level = "ERROR" if frac > 0.05 else "WARNING"
        add("FRAMES_NO_SENSOR", level,
            f"{aln.frames_no_sensor}/{vid.n_frames} frames ({frac*100:.1f}%) hors plage capteur",
            aln.frames_no_sensor, 0.0)

    if aln.sensor_gap_count > 0:
        level = "ERROR" if aln.sensor_gap_max_ms > 100.0 else "WARNING"
        add("SENSOR_GAP", level,
            f"{aln.sensor_gap_count} trou(s) capteur — gap max={aln.sensor_gap_max_ms:.1f}ms",
            aln.sensor_gap_max_ms, 0.0)

    if aln.overlap_s < thr.min_overlap_s:
        add("OVERLAP_SHORT", "ERROR",
            f"Recouvrement {aln.overlap_s:.1f}s < minimum {thr.min_overlap_s:.0f}s",
            aln.overlap_s, thr.min_overlap_s)

    if vid.jitter_std_ms > thr.jitter_std_ms:
        add("JITTER", "WARNING",
            f"Jitter timestamps vidéo {vid.jitter_std_ms:.2f}ms > seuil {thr.jitter_std_ms:.0f}ms",
            vid.jitter_std_ms, thr.jitter_std_ms)

    if vid.frame_drops > thr.max_drops:
        level = "ERROR" if vid.frame_drops > thr.max_drops * 3 else "WARNING"
        add("FRAME_DROPS", level,
            f"{vid.frame_drops} frame drops > seuil {thr.max_drops}",
            vid.frame_drops, thr.max_drops)

    if vid.missing_indices > 0:
        add("MISSING_FRAMES", "WARNING",
            f"{vid.missing_indices} indices manquants dans le flux JSONL",
            vid.missing_indices, 0.0)

    if sen.neg_dt_count > 0:
        level = "ERROR" if sen.neg_dt_count > 2 else "WARNING"
        add("SENSOR_NEG_DT", level,
            f"{sen.neg_dt_count} saut(s) temporel(s) négatif(s) dans le capteur",
            sen.neg_dt_count, 0.0)

    if sen.vel_anomalies > 0:
        add("SENSOR_VEL_ANOMALY", "WARNING",
            f"{sen.vel_anomalies} saut(s) impossible(s) (vel > {thr.max_vel_mm_s:.0f}mm/s)",
            sen.vel_max_mm_s, thr.max_vel_mm_s)

    if phy.impossible_jumps > 0:
        add("INTERP_VEL_ANOMALY", "WARNING",
            f"{phy.impossible_jumps} saut(s) impossible(s) dans le capteur interpolé",
            phy.impossible_jumps, 0.0)

    return alerts


# ─── Résultat complet d'un côté ───────────────────────────────────────────────

@dataclass
class SideResult:
    session_name: str
    side:         str
    success:      bool
    error:        str                            = ""
    video:        Optional[VideoTimestampMetrics]    = None
    sensor:       Optional[SensorMetrics]            = None
    alignment:    Optional[AlignmentMetrics]         = None
    physical:     Optional[PhysicalCoherenceMetrics] = None
    alerts:       List[Alert]                        = field(default_factory=list)
    has_errors:   bool                               = False
    sensor_df:    Optional[pd.DataFrame]             = field(default=None, repr=False)

    @property
    def status(self) -> str:
        if not self.success:    return "FAILED"
        if self.has_errors:     return "ERROR"
        if self.alerts:         return "WARNING"
        return "OK"

    @property
    def n_errors(self)   -> int: return sum(1 for a in self.alerts if a.level == "ERROR")
    @property
    def n_warnings(self) -> int: return sum(1 for a in self.alerts if a.level == "WARNING")


# ─── Traitement d'un côté ─────────────────────────────────────────────────────

def pnc_process_side(session_path: str, side: str, thr: Thresholds) -> SideResult:
    session_name = os.path.basename(session_path)
    jsonl_path   = os.path.join(session_path, "videos", f"{side}.jsonl")
    sensor_path  = os.path.join(session_path, f"gripper_{side}_data.csv")

    missing = [p for p in [jsonl_path, sensor_path] if not os.path.exists(p)]
    if missing:
        return SideResult(
            session_name=session_name, side=side, success=False,
            error=f"Fichiers absents : {[os.path.basename(p) for p in missing]}",
        )

    try:
        indices, ts_ns = pnc_load_jsonl_timestamps(jsonl_path)
        sensor_df      = pnc_load_sensor(sensor_path)

        dt_nominal_ms = float(np.median(np.diff(sensor_df["timestamp_ns"].values)) / 1e6)
        sensor_df     = pnc_apply_sensor_fixes(sensor_df, int(ts_ns[0]), dt_nominal_ms)

        vid = pnc_analyze_video_timestamps(indices, ts_ns)
        sen = pnc_analyze_sensor(sensor_df, thr.max_vel_mm_s)
        aln = pnc_compute_alignment(vid, sensor_df)
        phy = pnc_compute_physical_coherence(aln, thr.max_vel_mm_s)
        als = pnc_generate_alerts(vid, sen, aln, phy, thr)

        return SideResult(
            session_name=session_name, side=side, success=True,
            video=vid, sensor=sen, alignment=aln, physical=phy,
            alerts=als, has_errors=any(a.level == "ERROR" for a in als),
            sensor_df=sensor_df,
        )
    except Exception as exc:
        return SideResult(
            session_name=session_name, side=side, success=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


# ─── Rapports texte ───────────────────────────────────────────────────────────

def _fmt(v, fmt=".3f", unit=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:{fmt}}{unit}"


def pnc_write_side_report(result: SideResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        w = lambda line="": f.write(line + "\n")
        w("=" * 70);  w("RAPPORT ALIGNEMENT PINCE")
        w(f"Session : {result.session_name}");  w(f"Côté    : {result.side.upper()}")
        w(f"Statut  : {result.status}");        w("=" * 70);  w()
        if not result.success:
            w(f"ECHEC : {result.error}");  return

        if result.alerts:
            w("─" * 70);  w(f"ALERTES ({result.n_errors} erreur(s), {result.n_warnings} avertissement(s))")
            w("─" * 70)
            for a in result.alerts:
                w(f"  [{a.level:7s}] [{a.code}]");  w(f"           {a.message}")
            w()

        vid = result.video
        w("─" * 70);  w("TIMESTAMPS VIDÉO");  w("─" * 70)
        w(f"  Frames analysées       : {vid.n_frames}")
        w(f"  Durée couverte         : {vid.duration_s:.3f} s")
        w(f"  Intervalle inter-frames: moy={vid.dt_mean_ms:.2f}ms  std={vid.dt_std_ms:.2f}ms  [{vid.dt_min_ms:.1f}ms – {vid.dt_max_ms:.1f}ms]")
        w(f"  Jitter robuste (MAD)   : {vid.jitter_std_ms:.3f} ms")
        w(f"  Frame drops (>83ms)    : {vid.frame_drops}")
        w(f"  Indices manquants      : {vid.missing_indices}");  w()

        sen = result.sensor
        w("─" * 70);  w("SIGNAL CAPTEUR");  w("─" * 70)
        w(f"  Échantillons           : {sen.n_samples}")
        w(f"  Durée couverte         : {sen.duration_s:.3f} s")
        w(f"  Fréquence médiane      : {1000/max(sen.dt_mean_ms,1e-6):.1f} Hz  (dt médian = {sen.dt_mean_ms:.2f} ms)")
        w(f"  Sauts dt négatifs      : {sen.neg_dt_count}")
        w(f"  Ouverture plage        : [{sen.opening_range[0]:.1f}, {sen.opening_range[1]:.1f}] mm")
        w(f"  Vitesse max            : {sen.vel_max_mm_s:.0f} mm/s  (P99 = {sen.vel_p99_mm_s:.0f} mm/s)")
        w(f"  Anomalies vitesse      : {sen.vel_anomalies}");  w()

        aln = result.alignment
        w("─" * 70);  w("ALIGNEMENT HORLOGE");  w("─" * 70)
        w(f"  Durée vidéo            : {aln.dur_vid_s:.3f} s")
        w(f"  Durée capteur          : {aln.dur_sensor_s:.3f} s")
        w(f"  Recouvrement           : {aln.overlap_s:.3f} s");  w()
        w(f"  Offset démarrage       : {aln.offset_start_ms:+.3f} ms")
        w(f"  Latence frame→capteur  :")
        w(f"    moy                  : {_fmt(aln.latency_mean_ms, '+.3f', ' ms')}")
        w(f"    std                  : {_fmt(aln.latency_std_ms, '.3f', ' ms')}")
        w(f"    max |latence|        : {_fmt(aln.latency_max_abs_ms, '.3f', ' ms')}")
        w(f"    P95 |latence|        : {_fmt(aln.latency_p95_abs_ms, '.3f', ' ms')}")
        w(f"    frames hors capteur  : {aln.frames_no_sensor}");  w()
        w(f"  Continuité capteur :")
        w(f"    gap max              : {_fmt(aln.sensor_gap_max_ms, '.1f', ' ms')}")
        w(f"    nb gaps >4×dt_nom    : {aln.sensor_gap_count}");  w()
        w(f"  Modèle linéaire vidéo :")
        w(f"    slope                : {aln.linfit_slope:.3f} ns/frame")
        w(f"    R²                   : {_fmt(aln.linfit_r2, '.6f')}")
        w(f"    résidus std          : {_fmt(aln.linfit_residual_std_ms, '.3f', ' ms')}")
        w(f"    résidus max          : {_fmt(aln.linfit_residual_max_ms, '.3f', ' ms')}");  w()

        phy = result.physical
        w("─" * 70);  w("COHÉRENCE PHYSIQUE");  w("─" * 70)
        w(f"  Frames avec capteur    : {phy.n_frames_with_sensor}")
        w(f"  Frames sans capteur    : {phy.n_frames_no_sensor}")
        w(f"  Ouverture moy/std      : {_fmt(phy.opening_mean_mm, '.2f')} ± {_fmt(phy.opening_std_mm, '.2f')} mm")
        w(f"  Vitesse max inter-fr.  : {_fmt(phy.d_opening_max_mm_s, '.0f', ' mm/s')}")
        w(f"  Sauts impossibles      : {phy.impossible_jumps}")


def pnc_write_global_report(
    all_results: List[SideResult], path: str, thr: Thresholds
) -> None:
    ok_list   = [r for r in all_results if r.success and r.status == "OK"]
    warn_list = [r for r in all_results if r.success and r.status == "WARNING"]
    err_list  = [r for r in all_results if r.success and r.status == "ERROR"]
    fail_list = [r for r in all_results if not r.success]

    with open(path, "w", encoding="utf-8") as f:
        w = lambda line="": f.write(line + "\n")
        w("=" * 72);  w("RAPPORT GLOBAL — VÉRIFICATION ALIGNEMENT PINCES");  w("=" * 72)
        w(f"  Total analysés  : {len(all_results)}")
        w(f"  OK              : {len(ok_list)}")
        w(f"  WARNINGS        : {len(warn_list)}")
        w(f"  ERRORS          : {len(err_list)}")
        w(f"  FAILED          : {len(fail_list)}");  w()

        success_list = [r for r in all_results if r.success and r.alignment]
        if success_list:
            offsets   = [r.alignment.offset_start_ms    for r in success_list]
            latencies = [r.alignment.latency_max_abs_ms for r in success_list
                         if math.isfinite(r.alignment.latency_max_abs_ms)]
            gap_maxs  = [r.alignment.sensor_gap_max_ms  for r in success_list
                         if math.isfinite(r.alignment.sensor_gap_max_ms)]
            w("─" * 72);  w("STATISTIQUES GLOBALES");  w("─" * 72)

            def stat_row(label, values, fmt=".2f", unit="ms"):
                if not values:
                    w(f"  {label:<32} (aucune donnée)"); return
                arr = np.array(values)
                p95 = np.percentile(np.abs(arr), 95)
                w(f"  {label:<32} moy={arr.mean():{fmt}}{unit}  "
                  f"med={np.median(arr):{fmt}}{unit}  "
                  f"P95|.|={p95:{fmt}}{unit}  "
                  f"[{arr.min():{fmt}}, {arr.max():{fmt}}]{unit}")

            stat_row("Offset démarrage",          offsets)
            stat_row("Latence max |frame→capteur|", latencies)
            stat_row("Gap capteur max",             gap_maxs)
            w()

        w("─" * 72);  w("DÉTAIL PAR SESSION");  w("─" * 72)
        sessions_map: Dict[str, List[SideResult]] = {}
        for r in all_results:
            sessions_map.setdefault(r.session_name, []).append(r)
        for sname, res_list in sorted(sessions_map.items()):
            w(f"\n{sname}")
            for r in res_list:
                if not r.success:
                    w(f"  [{r.side:5s}] FAILED  {r.error[:80]}"); continue
                aln = r.alignment
                lat_str = f"{aln.latency_max_abs_ms:.1f}" if math.isfinite(aln.latency_max_abs_ms) else "N/A"
                gap_str = f"{aln.sensor_gap_max_ms:.1f}"  if math.isfinite(aln.sensor_gap_max_ms)  else "N/A"
                alerts_str = (
                    " ".join(f"[{a.code}]" for a in r.alerts[:3]) +
                    ("..." if len(r.alerts) > 3 else "")
                ) if r.alerts else "—"
                w(f"  [{r.side:5s}] {r.status:<8} "
                  f"off={aln.offset_start_ms:+7.1f}ms  "
                  f"lat_max={lat_str:>6}ms  "
                  f"gap_max={gap_str:>6}ms  "
                  f"jit={r.video.jitter_std_ms:5.2f}ms  "
                  f"drops={r.video.frame_drops:2d}  "
                  f"alerts={alerts_str}")


def pnc_save_plot(result: SideResult, path: str) -> None:
    if not result.success or result.alignment is None:
        return
    aln = result.alignment;  vid = result.video;  sensor_df = result.sensor_df
    t_ref_ns  = float(aln.frame_ts_ns[0])
    t_video_s = (aln.frame_ts_ns.astype(float) - t_ref_ns) / 1e9
    op_frames = aln.opening_at_frames

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
    fig.suptitle(
        f"{result.session_name}  |  côté {result.side.upper()}  |  {result.status}\n"
        f"offset={aln.offset_start_ms:+.1f}ms  "
        f"jitter={vid.jitter_std_ms:.2f}ms  drops={vid.frame_drops}",
        fontsize=11,
    )
    ax = axes[0]
    dt_ms = np.diff(aln.frame_ts_ns) / 1e6
    t_mid = (t_video_s[:-1] + t_video_s[1:]) / 2
    ax.plot(t_mid, dt_ms, linewidth=0.8, color="steelblue", label="Δt frames (ms)")
    ax.axhline(1000 / _NOMINAL_FPS, color="green", linestyle="--", linewidth=1)
    ax.axhline(_DROP_THRESHOLD_S * 1000, color="red", linestyle=":", linewidth=1)
    ax.set_ylabel("Δt (ms)");  ax.set_xlabel("Temps vidéo (s)");  ax.grid(True, alpha=0.3)

    ax = axes[1]
    if sensor_df is not None:
        t_sensor_s = (sensor_df["timestamp_ns"].values.astype(float) - t_ref_ns) / 1e9
        ax.plot(t_sensor_s, sensor_df["opening_mm"].values,
                color="orange", linewidth=0.8, alpha=0.7, label="Capteur brut")
    valid = np.isfinite(op_frames)
    ax.scatter(t_video_s[valid], op_frames[valid], s=3, color="royalblue", alpha=0.5, zorder=3)
    ax.set_ylabel("Ouverture (mm)");  ax.set_xlabel("Temps (s)");  ax.grid(True, alpha=0.3)

    ax = axes[2]
    t_ideal     = t_video_s[0] + np.arange(len(t_video_s)) * _NOMINAL_PERIOD_S
    residual_ms = (t_video_s - t_ideal) * 1000
    ax.plot(t_video_s, residual_ms, linewidth=1.0, color="purple")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Dérive cumulée (ms)");  ax.set_xlabel("Temps vidéo (s)");  ax.grid(True, alpha=0.3)

    plt.tight_layout();  plt.savefig(path, dpi=120);  plt.close(fig)


def pnc_collect_session_paths(sessions_dir: str, pattern: str = "session_") -> List[str]:
    return [
        os.path.join(sessions_dir, name)
        for name in sorted(os.listdir(sessions_dir))
        if name.startswith(pattern) and os.path.isdir(os.path.join(sessions_dir, name))
    ]



# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GRIPPER VISION
# Module dédié : verification/gripper_vision.py
# Pipeline géométrique avancé : 6 features de contours, auto-calibration du
# seuil, fusion adaptative, NCC avec sélection du pic positif dominant.
# ═════════════════════════════════════════════════════════════════════════════

# ─── Import du module dédié ───────────────────────────────────────────────────
try:
    from verification.gripper_vision import (
        GripperVisionConfig,
        GripperVisionResult,
        EventMatch        as GripperEventMatch,
        process_side      as gv_process_side,
        save_plot         as gv_save_plot,
        write_report      as gv_write_report,
        collect_session_paths as gv_collect_session_paths,
        CV2_OK            as _CV2_AVAILABLE,
    )
    _GV_MODULE_OK = True
except ImportError as _gv_import_err:
    _GV_MODULE_OK    = False
    _CV2_AVAILABLE   = False

    # Stubs minimalistes pour que le reste de verify.py compile
    @dataclass
    class GripperVisionConfig:   # type: ignore[no-redef]
        pass

    @dataclass
    class GripperVisionResult:   # type: ignore[no-redef]
        session_name: str = ""; side: str = ""; video_path: str = ""
        sensor_path: str = ""; success: bool = False; error: str = ""
        n_frames: int = 0; fps: float = 0.0; threshold_used: int = 35
        times_sec: List[float] = field(default_factory=list)
        vision_signal: List[float] = field(default_factory=list)
        sensor_signal: List[float] = field(default_factory=list)
        feature_weights_used: Dict[str, float] = field(default_factory=dict)
        pearson_r: float = float("nan"); pearson_p: float = float("nan")
        ncc_peak: float = float("nan"); ncc_lag_ms: float = float("nan")
        events_vision: List[float] = field(default_factory=list)
        events_sensor: List[float] = field(default_factory=list)
        matched_events: list = field(default_factory=list)
        n_unmatched_vision: int = 0; n_unmatched_sensor: int = 0
        signal_quality: float = float("nan")
        confidence: float = float("nan"); composite_score: float = float("nan")
        status: str = "UNKNOWN"

    GripperEventMatch = object

    def gv_process_side(session_path, side, cfg=None, debug_video=False, output_dir=None):  # type: ignore[no-redef]
        r = GripperVisionResult(session_name=Path(session_path).name, side=side,
                                video_path="", sensor_path="", success=False)
        r.error = f"verification.gripper_vision non disponible : {_gv_import_err}"
        return r

    def gv_save_plot(result, path):  # type: ignore[no-redef]
        pass

    def gv_write_report(results, path):  # type: ignore[no-redef]
        pass

    def gv_collect_session_paths(sessions_dir, pattern="session_"):  # type: ignore[no-redef]
        return []



# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ORCHESTRATEUR VERIFY
# Lance tous les vérificateurs et produit le rapport global
# ═════════════════════════════════════════════════════════════════════════════

# ─── Structures de résultat ──────────────────────────────────────────────────

@dataclass
class CheckerResult:
    name:       str
    status:     str       # "OK" | "WARNING" | "ERROR" | "FAILED" | "SKIPPED"
    duration_s: float     = 0.0
    summary:    str       = ""
    details:    List[str] = field(default_factory=list)
    error:      str       = ""


@dataclass
class VerifyReport:
    mode:         str
    sessions_dir: str
    output_dir:   str
    checkers:     List[CheckerResult] = field(default_factory=list)
    total_s:      float               = 0.0

    @property
    def global_status(self) -> str:
        statuses = [c.status for c in self.checkers]
        if "ERROR" in statuses or "FAILED" in statuses: return "ERROR"
        if "WARNING" in statuses:                        return "WARNING"
        return "OK"

    @property
    def exit_code(self) -> int:
        return {"OK": 0, "WARNING": 1}.get(self.global_status, 2)


# ─── [1] check.py ────────────────────────────────────────────────────────────

def run_check(sessions_dir: Path, output_dir: Path, mode: str) -> CheckerResult:
    t0   = time.time()
    name = "check.py (score sync IA)"
    try:
        import check as chk
    except ImportError as e:
        return CheckerResult(name=name, status="FAILED",
                             error=f"Import check.py impossible : {e}")

    sessions = sorted(
        p.parent for p in sessions_dir.rglob("metadata.json")
        if p.parent != sessions_dir
    )
    if not sessions:
        return CheckerResult(name=name, status="SKIPPED", summary="Aucune session trouvée")

    model   = chk.load_model()
    reports = [];  details = []
    for sess in sessions:
        try:
            r       = chk.check_session(sess, model)
            reports.append(r)
            verdict = r.verdict
            sym     = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(verdict, "?")
            msg     = f"  {sym} {sess.name:<40}  score={r.score:5.1f}%"
            if r.blocking_reason:
                msg += f"  [{r.blocking_reason[:60]}]"
            elif r.ia_score > 0:
                msg += f"  ia={r.ia_score:.3f}"
            details.append(msg)
        except Exception as exc:
            details.append(f"  ✗ {sess.name}  ERREUR: {exc}")

    n_pass = sum(1 for r in reports if r.verdict == "PASS")
    n_warn = sum(1 for r in reports if r.verdict == "WARN")
    n_fail = sum(1 for r in reports if r.verdict == "FAIL")
    scores = [r.score for r in reports]
    mean_s = sum(scores) / len(scores) if scores else 0.0
    summary = (f"{len(sessions)} sessions — PASS:{n_pass}  WARN:{n_warn}  FAIL:{n_fail}  "
               f"moy={mean_s:.1f}%")
    status  = "ERROR" if n_fail > 0 else ("WARNING" if n_warn > 0 else "OK")

    if mode == "test":
        import dataclasses
        output_dir.mkdir(parents=True, exist_ok=True)
        out = [{**dataclasses.asdict(r), "verdict": r.verdict} for r in reports]
        (output_dir / "check_report.json").write_text(
            json.dumps(out, indent=2, default=str), encoding="utf-8"
        )
    return CheckerResult(name=name, status=status, duration_s=time.time()-t0,
                         summary=summary, details=details)


# ─── [2] fix_camera_offset ───────────────────────────────────────────────────

def run_camera_offset(sessions_dir: Path, output_dir: Path, mode: str) -> CheckerResult:
    t0   = time.time()
    name = "fix_camera_offset (décalage caméra/tracker)"
    try:
        import fix_camera_offset as fco
    except ImportError as e:
        return CheckerResult(name=name, status="FAILED",
                             error=f"Import fix_camera_offset.py impossible : {e}")

    dry_run  = (mode == "test")
    sessions = sorted(
        p.parent for p in sessions_dir.rglob("metadata.json")
        if (p.parent / "videos").exists()
    )
    if not sessions:
        return CheckerResult(name=name, status="SKIPPED", summary="Aucune session trouvée")

    reports = [];  details = [];  n_corrected = n_skipped = n_errors = 0
    for sess in sessions:
        try:
            r      = fco.fix_session(sess, dry_run=dry_run, force=(mode == "fix"))
            reports.append(r)
            status = r.get("status", "?")
            reason = r.get("reason", "")
            if status == "error":
                n_errors += 1;  details.append(f"  ✗ {sess.name}  [{status.upper()}] {reason}")
            elif status in ("corrected", "dry-run"):
                n_corrected += 1
                cams = r.get("cameras_fixed", [])
                offsets_str = ", ".join(f"{c['camera']}:{c['offset_ms']:+.0f}ms" for c in cams)
                pfx = "[DRY-RUN] " if dry_run else ""
                details.append(f"  ⚡ {sess.name}  {pfx}offsets={offsets_str}")
            else:
                n_skipped += 1;  details.append(f"  ✓ {sess.name}  {reason[:70]}")
        except Exception as exc:
            n_errors += 1;  details.append(f"  ✗ {sess.name}  ERREUR: {exc}")

    action  = "détectées" if dry_run else "appliquées"
    summary = (f"{len(sessions)} sessions — corrections {action}:{n_corrected}  "
               f"déjà OK:{n_skipped}  erreurs:{n_errors}")
    status  = ("ERROR" if n_errors > 0
               else ("WARNING" if (dry_run and n_corrected > 0) else "OK"))

    if mode == "test":
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "camera_offset_report.json").write_text(
            json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return CheckerResult(name=name, status=status, duration_s=time.time()-t0,
                         summary=summary, details=details)


# ─── [3] trakeur ─────────────────────────────────────────────────────────────

def run_trakeur(sessions_dir: Path, output_dir: Path, mode: str) -> CheckerResult:
    t0   = time.time()
    name = "trakeur.py (identification trackers head/left/right)"

    try:
        sessions = trk_collect_sessions(str(sessions_dir))
    except Exception as e:
        return CheckerResult(name=name, status="FAILED",
                             error=f"collect_sessions : {e}")

    if not sessions:
        return CheckerResult(name=name, status="SKIPPED",
                             summary="Aucune session avec tracker_positions.csv")

    try:
        results = trk_evaluate_leave_one_out(sessions)
    except Exception as e:
        return CheckerResult(name=name, status="FAILED",
                             error=f"evaluate_leave_one_out : {e}")

    total    = len(results)
    head_ok  = sum(1 for r in results if r["correct"]["head"])
    left_ok  = sum(1 for r in results if r["correct"]["left"])
    right_ok = sum(1 for r in results if r["correct"]["right"])
    exact_ok = sum(1 for r in results if r["exact"])

    def acc(n): return n / total if total else 0.0
    exact_acc = acc(exact_ok);  label_acc = acc(head_ok + left_ok + right_ok) / 3
    head_acc  = acc(head_ok);   left_acc  = acc(left_ok);  right_acc = acc(right_ok)

    details = []
    for r in results:
        sym    = "✓" if r["exact"] else "✗"
        ok_str = (f"head={'✓' if r['correct']['head'] else '✗'} "
                  f"left={'✓' if r['correct']['left'] else '✗'} "
                  f"right={'✓' if r['correct']['right'] else '✗'}")
        details.append(f"  {sym} {r['name']:<40}  {ok_str}")

    try:
        best_rule, best_acc_val, _, _ = trk_choose_best_global_rule(sessions)
        qm, ax, sg = best_rule
        details.append(f"\n  Meilleure règle globale : quat={qm} axis={ax} sign={sg} acc={best_acc_val:.4f}")
    except Exception:
        pass

    summary = (f"{total} sessions — exact={exact_acc:.1%}  labels={label_acc:.1%}  "
               f"(head={head_acc:.1%} left={left_acc:.1%} right={right_acc:.1%})")
    # Seuils : plafond empirique ~90% (contraintes géométriques des sessions ambiguës)
    # WARNING si < 88% (dégradation notable), ERROR si < 65% (algo défaillant)
    status  = "ERROR" if exact_acc < 0.65 else ("WARNING" if exact_acc < 0.88 else "OK")

    if mode == "test":
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "trakeur_report.json").write_text(
            json.dumps({
                "total": total, "exact_acc": round(exact_acc, 4),
                "label_acc": round(label_acc, 4), "head_acc": round(head_acc, 4),
                "left_acc": round(left_acc, 4), "right_acc": round(right_acc, 4),
                "results": results,
            }, indent=2, default=str),
            encoding="utf-8",
        )
    return CheckerResult(name=name, status=status, duration_s=time.time()-t0,
                         summary=summary, details=details)


# ─── [4] gripper_vision ──────────────────────────────────────────────────────

def run_gripper_vision(
    sessions_dir: Path,
    output_dir:   Path,
    mode:         str,
    sides:        Tuple[str, ...] = ("left", "right"),
    cfg:          Optional[GripperVisionConfig] = None,
    debug_video:  bool = False,
    plots:        bool = False,
) -> CheckerResult:
    """
    Vérificateur [4] : corrélation signal visuel d'ouverture vs CSV capteur.
    Nécessite opencv-python. Passe en SKIPPED si absent.
    """
    t0   = time.time()
    name = "gripper_vision (alignement visuel/capteur)"

    if not _CV2_AVAILABLE:
        return CheckerResult(
            name=name, status="SKIPPED",
            summary="opencv-python non installé — pip install opencv-python",
        )

    if cfg is None:
        cfg = GripperVisionConfig()

    session_paths = gv_collect_session_paths(str(sessions_dir))
    if not session_paths:
        return CheckerResult(name=name, status="SKIPPED",
                             summary="Aucune session trouvée")

    gv_dir = output_dir / "gripper_vision"
    gv_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[GripperVisionResult] = []
    details: List[str] = []

    for spath in session_paths:
        sname = Path(spath).name
        for side in sides:
            r = gv_process_side(
                spath, side, cfg,
                debug_video=debug_video,
                output_dir=str(gv_dir) if debug_video else None,
            )
            all_results.append(r)

            if r.success:
                sym = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(r.status, "?")
                details.append(
                    f"  {sym} {sname}/{side:<5}  "
                    f"score={r.composite_score:.1f}/100  "
                    f"r={r.pearson_r:+.3f}  "
                    f"lag={r.ncc_lag_ms:+.1f}ms  "
                    f"ev_vis={len(r.events_vision)} ev_sen={len(r.events_sensor)}"
                )
                if plots:
                    try:
                        gv_save_plot(r, str(gv_dir / f"{sname}_{side}_plot.png"))
                    except Exception:
                        pass
            else:
                details.append(f"  ✗ {sname}/{side}  FAILED: {r.error[:70]}")

    n_ok   = sum(1 for r in all_results if r.success and r.status == "OK")
    n_warn = sum(1 for r in all_results if r.success and r.status == "WARNING")
    n_err  = sum(1 for r in all_results if r.success and r.status == "ERROR")
    n_fail = sum(1 for r in all_results if not r.success)
    scores = [r.composite_score for r in all_results
              if r.success and math.isfinite(r.composite_score)]
    mean_s = float(np.mean(scores)) if scores else 0.0

    summary = (f"{len(all_results)} côtés — "
               f"OK:{n_ok}  WARN:{n_warn}  ERR:{n_err}  FAIL:{n_fail}  "
               f"score_moy={mean_s:.1f}/100")
    status  = ("ERROR"   if n_err  > 0 or n_fail > 0
               else ("WARNING" if n_warn > 0 else "OK"))

    if mode == "test":
        gv_write_report(all_results, str(gv_dir / "gripper_vision_report.txt"))
        import dataclasses as _dc
        (gv_dir / "gripper_vision_report.json").write_text(
            json.dumps(
                [{k: v for k, v in _dc.asdict(r).items()
                  if k not in ("vision_signal", "sensor_signal", "times_sec")}
                 for r in all_results],
                indent=2, default=str,
            ),
            encoding="utf-8",
        )

    return CheckerResult(name=name, status=status, duration_s=time.time()-t0,
                         summary=summary, details=details)


# ─── [5] pinces ──────────────────────────────────────────────────────────────

_pinces_before_results: list = []


def _run_pinces_internal(
    sessions_dir: Path, output_dir: Optional[Path], plots: bool = False
) -> Tuple[CheckerResult, list]:
    thr           = Thresholds()
    session_paths = pnc_collect_session_paths(str(sessions_dir))
    name          = "session_pinces.py (alignement horloge vidéo/capteur)"

    if not session_paths:
        return CheckerResult(name=name, status="SKIPPED",
                             summary="Aucune session trouvée"), []

    all_results = [];  details = []
    for spath in session_paths:
        sname = Path(spath).name
        for side in ("left", "right"):
            r = pnc_process_side(spath, side, thr)
            all_results.append(r)
            if r.success:
                aln    = r.alignment
                sym    = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(r.status, "?")
                lat_str = f"{aln.latency_max_abs_ms:.0f}" if math.isfinite(aln.latency_max_abs_ms) else "N/A"
                codes  = " ".join(f"[{a.code}]" for a in r.alerts[:3])
                details.append(f"  {sym} {sname} / {side:<5}  "
                                f"off={aln.offset_start_ms:+.0f}ms  "
                                f"lat_max={lat_str}ms  {codes}")
            else:
                details.append(f"  ✗ {sname} / {side}  FAILED: {r.error[:60]}")

    ok_l   = [r for r in all_results if r.success and r.status == "OK"]
    warn_l = [r for r in all_results if r.success and r.status == "WARNING"]
    err_l  = [r for r in all_results if r.success and r.status == "ERROR"]
    fail_l = [r for r in all_results if not r.success]

    # Distinguer erreurs corrigeables (OFFSET_START seul) des erreurs hardware
    def _only_offset_error(r: SideResult) -> bool:
        return (r.success and r.status == "ERROR"
                and all(a.code == "OFFSET_START" for a in r.alerts if a.level == "ERROR"))

    n_fixable_offset = sum(1 for r in err_l if _only_offset_error(r))
    n_hard_errors    = len(err_l) - n_fixable_offset

    summary = (f"{len(all_results)} côtés — "
               f"OK:{len(ok_l)}  WARNING:{len(warn_l)}  "
               f"ERROR:{len(err_l)}  FAILED:{len(fail_l)}")
    if n_fixable_offset > 0:
        summary += f"  ({n_fixable_offset} offset→corrigeable via --fix)"

    # En mode test, les OFFSET_START seuls ne sont pas des vrais blocants (corrigeables par --fix)
    # On les classe WARNING pour ne pas masquer les vraies erreurs hardware
    has_hard = n_hard_errors > 0 or bool(fail_l)
    status   = "ERROR" if has_hard else ("WARNING" if (warn_l or err_l) else "OK")

    if output_dir is not None:
        pinces_dir = output_dir / "pinces"
        pinces_dir.mkdir(parents=True, exist_ok=True)
        pnc_write_global_report(all_results, str(pinces_dir / "gripper_alignment_report.txt"), thr)
        for r in all_results:
            prefix = str(pinces_dir / f"{r.session_name}_{r.side}")
            pnc_write_side_report(r, prefix + "_report.txt")
            if plots and r.success and r.alignment is not None:
                try:
                    pnc_save_plot(r, prefix + "_plot.png")
                except Exception:
                    pass

    return CheckerResult(name=name, status=status, summary=summary, details=details), all_results


def run_pinces(sessions_dir: Path, output_dir: Path, mode: str, plots: bool = False) -> CheckerResult:
    t0 = time.time()
    try:
        out = output_dir if mode == "test" else None
        result, _ = _run_pinces_internal(sessions_dir, out, plots=plots)
        result.duration_s = time.time() - t0
        return result
    except Exception as e:
        return CheckerResult(name="session_pinces.py", status="FAILED",
                             duration_s=time.time()-t0,
                             error=f"{type(e).__name__}: {e}")


def run_pinces_snapshot(
    sessions_dir: Path, output_dir: Path, label: str, plots: bool = False
) -> CheckerResult:
    global _pinces_before_results
    t0 = time.time()
    try:
        result, all_results = _run_pinces_internal(sessions_dir, output_dir, plots=plots)
        _pinces_before_results = all_results
        result.name       = f"session_pinces.py ({label})"
        result.duration_s = time.time() - t0
        return result
    except Exception as e:
        return CheckerResult(name=f"session_pinces.py ({label})", status="FAILED",
                             duration_s=time.time()-t0,
                             error=f"{type(e).__name__}: {e}")


def run_pinces_after_fix(
    sessions_dir: Path, output_dir: Path, plots: bool = False
) -> CheckerResult:
    global _pinces_before_results
    t0 = time.time()
    name = "session_pinces.py (APRÈS FIX — delta avant/après)"
    try:
        result_after, after_results = _run_pinces_internal(sessions_dir, output_dir, plots=plots)
    except Exception as e:
        return CheckerResult(name=name, status="FAILED",
                             duration_s=time.time()-t0,
                             error=f"{type(e).__name__}: {e}")

    before_map = {(r.session_name, r.side): r for r in _pinces_before_results}
    details = [];  changed_count = remaining_errors = 0

    for r_after in after_results:
        key      = (r_after.session_name, r_after.side)
        r_before = before_map.get(key)
        if r_before is None or not r_after.success:
            sym = "✗" if not r_after.success else "?"
            details.append(f"  {sym} {r_after.session_name}/{r_after.side}  (pas de données avant)")
            if r_after.success and r_after.status == "ERROR":
                remaining_errors += 1
            continue
        if r_before.success and r_after.success:
            a_b = r_before.alignment;  a_a = r_after.alignment
            off_delta = a_a.offset_start_ms - a_b.offset_start_ms
            lat_b = f"{a_b.latency_max_abs_ms:.0f}" if math.isfinite(a_b.latency_max_abs_ms) else "N/A"
            lat_a = f"{a_a.latency_max_abs_ms:.0f}" if math.isfinite(a_a.latency_max_abs_ms) else "N/A"
            sb = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(r_before.status, "?")
            sa = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(r_after.status, "?")
            if abs(off_delta) > 1.0:
                changed_count += 1
                details.append(
                    f"  {sb}→{sa} {r_after.session_name}/{r_after.side}  "
                    f"off: {a_b.offset_start_ms:+.0f}ms → {a_a.offset_start_ms:+.0f}ms "
                    f"(Δ={off_delta:+.0f}ms)  lat_max: {lat_b}ms→{lat_a}ms"
                )
            else:
                details.append(f"  {sa}  {r_after.session_name}/{r_after.side}  "
                                f"off={a_a.offset_start_ms:+.0f}ms  lat_max={lat_a}ms")
        else:
            sym = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(r_after.status, "?")
            details.append(f"  {sym} {r_after.session_name}/{r_after.side}")
        if r_after.success and r_after.status == "ERROR":
            remaining_errors += 1

    result_after.name       = name
    result_after.duration_s = time.time() - t0
    result_after.details    = details
    if changed_count > 0:
        result_after.summary += f"  |  {changed_count} offset(s) corrigé(s)"
    if remaining_errors > 0:
        result_after.summary += f"  |  {remaining_errors} erreur(s) résiduelle(s)"
    return result_after


# ─── Rapport console et fichier ──────────────────────────────────────────────

_STATUS_ICON  = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗", "FAILED": "✗", "SKIPPED": "—"}
_STATUS_COLOR = {
    "OK":      "\033[92m",
    "WARNING": "\033[93m",
    "ERROR":   "\033[91m",
    "FAILED":  "\033[91m",
    "SKIPPED": "\033[90m",
}
_RESET = "\033[0m";  _BOLD = "\033[1m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}" if sys.stdout.isatty() else text


def print_report(report: VerifyReport, verbose: bool) -> None:
    print();  print(_c("=" * 68, _BOLD))
    print(_c(f"  VERIFY — mode {report.mode.upper()}  —  {report.global_status}", _BOLD))
    print(_c("=" * 68, _BOLD))
    print(f"  Sessions : {report.sessions_dir}")
    print(f"  Durée    : {report.total_s:.1f}s");  print()
    for c in report.checkers:
        icon  = _STATUS_ICON.get(c.status, "?")
        color = _STATUS_COLOR.get(c.status, "")
        print(_c(f"  [{icon}] {c.name}", color))
        print(f"      {c.summary}  ({c.duration_s:.1f}s)")
        if c.error:
            print(_c(f"      ERREUR : {c.error}", _STATUS_COLOR["ERROR"]))
        if verbose and c.details:
            for line in c.details:
                print(line)
        print()
    gcolor = _STATUS_COLOR.get(report.global_status, "")
    print(_c(f"  Statut global : {report.global_status}", gcolor + _BOLD))
    print(_c("=" * 68, _BOLD))


def write_report(report: VerifyReport, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        w = lambda line="": f.write(line + "\n")
        w("=" * 68)
        w(f"VERIFY — mode {report.mode.upper()}")
        w(f"Sessions  : {report.sessions_dir}")
        w(f"Sortie    : {report.output_dir}")
        w(f"Durée     : {report.total_s:.1f}s")
        w(f"Statut    : {report.global_status}")
        w("=" * 68);  w()
        for c in report.checkers:
            icon = _STATUS_ICON.get(c.status, "?")
            w(f"[{icon}] {c.name}")
            w(f"    Statut  : {c.status}")
            w(f"    Durée   : {c.duration_s:.1f}s")
            w(f"    Résumé  : {c.summary}")
            if c.error:
                w(f"    ERREUR  : {c.error}")
            if c.details:
                w("    Détails :")
                for line in c.details:
                    w(line)
            w()


# ─── Orchestration principale ────────────────────────────────────────────────

def run_all(
    sessions_dir:        Path,
    output_dir:          Path,
    mode:                str,
    skip_check:          bool = False,
    skip_trakeur:        bool = False,
    skip_pinces:         bool = False,
    skip_gripper_vision: bool = False,
    verbose:             bool = False,
    plots:               bool = False,
    gripper_vision_cfg:  Optional[GripperVisionConfig] = None,
    gripper_debug_video: bool = False,
) -> VerifyReport:
    t_global = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = VerifyReport(
        mode=mode, sessions_dir=str(sessions_dir), output_dir=str(output_dir)
    )

    steps = []
    if not skip_check:
        steps.append(("check.py",
                       lambda: run_check(sessions_dir, output_dir, mode)))

    if not skip_pinces and mode == "fix":
        steps.append(("pinces_avant_fix",
                       lambda: run_pinces_snapshot(sessions_dir, output_dir / "before_fix",
                                                   label="AVANT FIX", plots=plots)))

    steps.append(("camera_offset",
                   lambda: run_camera_offset(sessions_dir, output_dir, mode)))

    if not skip_trakeur:
        steps.append(("trakeur",
                       lambda: run_trakeur(sessions_dir, output_dir, mode)))

    if not skip_pinces:
        if mode == "fix":
            steps.append(("pinces_après_fix",
                           lambda: run_pinces_after_fix(sessions_dir, output_dir, plots=plots)))
        else:
            steps.append(("pinces",
                           lambda: run_pinces(sessions_dir, output_dir, mode, plots=plots)))

    if not skip_gripper_vision:
        _cfg = gripper_vision_cfg  # capture pour la closure
        _dbv = gripper_debug_video
        steps.append(("gripper_vision",
                       lambda: run_gripper_vision(
                           sessions_dir, output_dir, mode,
                           cfg=_cfg, debug_video=_dbv, plots=plots,
                       )))

    for step_name, fn in steps:
        print(f"\n{'─'*68}")
        print(f"  Lancement : {step_name} ...")
        try:
            result = fn()
        except Exception as exc:
            result = CheckerResult(name=step_name, status="FAILED",
                                   error=f"Exception : {exc}\n{traceback.format_exc()}")
        report.checkers.append(result)

        icon  = _STATUS_ICON.get(result.status, "?")
        color = _STATUS_COLOR.get(result.status, "")
        print(_c(f"  [{icon}] {result.name}  —  {result.status}", color))
        print(f"      {result.summary}")
        if result.error:
            print(_c(f"      {result.error}", _STATUS_COLOR["ERROR"]))
        if verbose and result.details:
            for line in result.details:
                print(line)

    report.total_s = time.time() - t_global
    print_report(report, verbose)
    report_path = output_dir / "verify_report.txt"
    write_report(report, report_path)
    print(f"\n  Rapport écrit : {report_path}")
    return report


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="verify.py — Vérification complète du pipeline de capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sessions_dir", required=True,
                   help="Répertoire contenant les dossiers session_*")
    p.add_argument("--output_dir", default="verify_results",
                   help="Répertoire de sortie (défaut : verify_results/)")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true",
                      help="Mode lecture seule : vérifications + rapport")
    mode.add_argument("--fix",  action="store_true",
                      help="Mode correction : applique les fixes puis re-vérifie")

    p.add_argument("--verbose", "-v", action="store_true",
                   help="Afficher le détail de chaque session")
    p.add_argument("--plots", action="store_true",
                   help="Générer les graphiques PNG (désactivé par défaut)")
    p.add_argument("--no_check",          action="store_true", help="Sauter check.py")
    p.add_argument("--no_trakeur",        action="store_true", help="Sauter trakeur")
    p.add_argument("--no_pinces",         action="store_true", help="Sauter session_pinces")
    p.add_argument("--no_gripper_vision", action="store_true",
                   help="Sauter la vérification visuelle gripper (nécessite opencv)")
    p.add_argument("--gripper_debug_video", action="store_true",
                   help="Écrire les vidéos debug gripper (lent, optionnel)")
    # Paramètres ROI ajustables en CLI
    p.add_argument("--gv_roi",  nargs=4, type=int, metavar=("X1","Y1","X2","Y2"),
                   help="ROI gripper : x1 y1 x2 y2 (pixels, défaut: 540 120 1180 560)")
    p.add_argument("--gv_rows", nargs="+", type=int,
                   help="Rangées y à analyser dans la ROI (défaut: 120 145 170 195 220 245)")
    p.add_argument("--gv_threshold", type=int, default=None,
                   help="Seuil de luminosité [0-255] (défaut: 95 ; ignoré si --gv_otsu)")
    p.add_argument("--gv_otsu", action="store_true",
                   help="Utiliser Otsu par frame à la place du seuil fixe")
    p.add_argument("--session_pattern", default="session_",
                   help="Préfixe des dossiers session (défaut : session_)")
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    sessions_dir = Path(args.sessions_dir).resolve()
    output_dir   = Path(args.output_dir).resolve()
    mode         = "test" if args.test else "fix"

    if not sessions_dir.exists():
        print(f"[ERREUR] sessions_dir introuvable : {sessions_dir}", file=sys.stderr)
        sys.exit(2)

    # Construire GripperVisionConfig depuis les args CLI
    gv_cfg = GripperVisionConfig()
    if args.gv_roi:
        gv_cfg.roi_x1, gv_cfg.roi_y1, gv_cfg.roi_x2, gv_cfg.roi_y2 = args.gv_roi
    if args.gv_rows:
        gv_cfg.rows_to_scan = tuple(args.gv_rows)
    if args.gv_threshold is not None:
        gv_cfg.bright_threshold = args.gv_threshold
    if args.gv_otsu:
        gv_cfg.use_otsu = True

    print(_c(f"\n  verify.py — mode {mode.upper()}", _BOLD))
    print(f"  sessions : {sessions_dir}")
    print(f"  sortie   : {output_dir}")

    report = run_all(
        sessions_dir        = sessions_dir,
        output_dir          = output_dir,
        mode                = mode,
        skip_check          = args.no_check,
        skip_trakeur        = args.no_trakeur,
        skip_pinces         = args.no_pinces,
        skip_gripper_vision = args.no_gripper_vision,
        verbose             = args.verbose,
        plots               = args.plots,
        gripper_vision_cfg  = gv_cfg,
        gripper_debug_video = args.gripper_debug_video,
    )
    sys.exit(report.exit_code)


if __name__ == "__main__":
    main()
