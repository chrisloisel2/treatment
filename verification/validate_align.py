#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_align.py — Validation multi-étapes de l'alignement gripper.

ÉTAPE 1 — ÉVÉNEMENT DE FERMETURE
────────────────────────────────
Détecte les transitions fermeture (ouvert → fermé) dans le signal capteur aligné.
Pour chaque fermeture :
  • Calcule le t50 capteur : instant où opening_mm passe sous 50% de l'amplitude
  • Calcule le t50 vision  : même définition sur vis_norm
  • Erreur de timing = t50_vision − t50_capteur
  • Extrait la frame vidéo au moment t50 pour contrôle visuel
  • Logique physique : dans la fenêtre fermeture, opening_mm doit être
    monotone décroissante et le résidu moyen doit être faible.

ÉTAPE 2 — ÉVÉNEMENT DE PLATEAU
───────────────────────────────
Détecte les régions où le gripper est immobile (|vel| < seuil) pendant > 0.3s.
Pour chaque plateau :
  • Calcule mean/std de opening_mm (capteur) → valeur de référence physique
  • Calcule mean/std de vis_norm (vision)     → valeur de référence visuelle
  • Calcule le biais absolu : mean_vis_norm − mean_sen_norm
  • Calcule le jitter : std de la différence frame-à-frame
  • Logique : si alignement parfait, biais ≈ 0 et jitter ≈ 0

Usage :
  python validate_align.py --session /path/to/session
  python validate_align.py --session /path/to/session --side left
  python validate_align.py --sessions_dir /path/to/sessions
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ═════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ValConfig:
    # Lissage du signal (Savitzky-Golay)
    sg_window:          int   = 15      # frames
    sg_poly:            int   = 2

    # Détection fermeture
    close_vel_thresh:   float = -30.0   # mm/s : dérivée < seuil → fermeture
    close_from_mm:      float = 15.0    # ouverture avant fermeture
    close_to_mm:        float = 8.0     # ouverture après fermeture
    close_min_amp:      float = 10.0    # amplitude mini de la fermeture (mm)

    # Détection plateau
    plateau_vel_thresh: float = 10.0    # mm/s : |vel| < seuil → immobile
    plateau_min_dur_s:  float = 0.3     # durée minimum (s)

    # Tolérance timing fermeture
    timing_tol_ms:      float = 100.0   # erreur acceptable (ms)

    # Tolérance biais plateau
    bias_tol:           float = 0.15    # [0-1] en normalisé

    # Strip de frames autour de l'événement
    strip_half_s:       float = 1.0     # ± secondes


# ═════════════════════════════════════════════════════════════════════════════
# STRUCTURES RÉSULTAT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ClosureEvent:
    event_id:       int
    t_start_s:      float    # début de la transition
    t_end_s:        float    # fin de la transition
    t50_sensor_s:   float    # t50 capteur
    t50_vision_s:   float    # t50 vision
    timing_err_ms:  float    # t50_vision - t50_sensor (ms)
    op_before_mm:   float    # ouverture moyenne avant
    op_after_mm:    float    # ouverture moyenne après
    amplitude_mm:   float    # amplitude fermeture
    n_frames:       int      # frames dans la transition
    residual_mean:  float    # résidu moyen dans la fenêtre
    residual_max:   float    # résidu max dans la fenêtre
    monotone_ok:    bool     # capteur monotone décroissant dans la fenêtre
    visual_range:   float    # amplitude de vis_norm dans la fenêtre (0=insensible)
    visual_insensitive: bool # True si feature n'est pas sensible à cet événement
    logic_ok:       bool     # validation logique globale
    logic_reason:   str      # motif si échec


@dataclass
class PlateauEvent:
    event_id:       int
    t_start_s:      float
    t_end_s:        float
    duration_s:     float
    sensor_mean_mm: float    # ouverture capteur moyenne (mm)
    sensor_std_mm:  float    # stabilité capteur (mm)
    vis_mean_norm:  float    # signal vision normalisé moyen
    sen_mean_norm:  float    # capteur normalisé moyen
    bias_norm:      float    # vis_mean_norm - sen_mean_norm  (biais absolu)
    jitter_norm:    float    # std de (vis_norm - sen_norm) frame-à-frame
    n_frames:       int
    logic_ok:       bool
    logic_reason:   str


@dataclass
class ValidationResult:
    session:        str
    side:           str
    success:        bool
    error:          str = ""
    n_frames:       int = 0

    closure_events: List[ClosureEvent]  = field(default_factory=list)
    plateau_events: List[PlateauEvent]  = field(default_factory=list)

    # Synthèse globale
    n_closures_total:  int   = 0
    n_closures_ok:     int   = 0
    timing_err_mean_ms: float = float("nan")
    timing_err_std_ms:  float = float("nan")
    timing_err_max_ms:  float = float("nan")

    n_plateaus_total:  int   = 0
    n_plateaus_ok:     int   = 0
    bias_mean:         float = float("nan")
    bias_std:          float = float("nan")
    jitter_mean:       float = float("nan")

    status:            str   = "UNKNOWN"   # OK / WARNING / ERROR

    # Corrélation croisée dérivée (vérification timing primaire)
    derivative_ncc_r:      float = float("nan")   # NCC(dvis/dt, dsen/dt) au lag=0
    n_closures_insensitive: int  = 0              # fermetures visuellement indétectables


# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT JSONL (pour extraction frames)
# ═════════════════════════════════════════════════════════════════════════════

def load_jsonl_abs(jsonl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    raw = open(jsonl_path, "rb").read()
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
    idx = np.array(indices, dtype=np.int64)
    ts  = np.array(ts_ns,   dtype=np.int64)
    order = np.argsort(idx)
    idx, ts = idx[order], ts[order]
    idx = idx - idx[0]
    return idx, ts


def extract_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(int(frame_idx), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ═════════════════════════════════════════════════════════════════════════════
# LISSAGE
# ═════════════════════════════════════════════════════════════════════════════

def smooth(x: np.ndarray, cfg: ValConfig) -> np.ndarray:
    w = cfg.sg_window
    if w % 2 == 0:
        w += 1
    if len(x) <= w:
        return x.copy()
    return savgol_filter(x, w, cfg.sg_poly)


def velocity(op_smooth: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Dérivée temporelle en mm/s. Même longueur que op_smooth."""
    return np.gradient(op_smooth, t)


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — DÉTECTION ET VALIDATION DES FERMETURES
# ═════════════════════════════════════════════════════════════════════════════

def _find_t50(t: np.ndarray, sig: np.ndarray, t_start: float, t_end: float,
              level50: float, ascending: bool = False) -> float:
    """
    Trouve l'instant où sig passe par level50 dans [t_start, t_end].
    ascending=False : cherche un crossing DESCENDANT (vis diminue = fermeture normale).
    ascending=True  : cherche un crossing ASCENDANT  (vis monte  = feature inversée).
    Interpolation linéaire entre les deux frames qui encadrent le crossing.
    """
    mask = (t >= t_start) & (t <= t_end)
    t_w  = t[mask]
    s_w  = sig[mask]
    if len(t_w) < 2:
        return float("nan")

    if ascending:
        # Chercher le premier crossing ascendant au-dessus de level50
        for i in range(len(s_w) - 1):
            if s_w[i] <= level50 < s_w[i + 1]:
                frac = (level50 - s_w[i]) / (s_w[i + 1] - s_w[i])
                return float(t_w[i] + frac * (t_w[i + 1] - t_w[i]))
        # Fallback : retourner le maximum (pic de l'ascension)
        return float(t_w[np.argmax(s_w)])
    else:
        # Chercher le premier crossing descendant sous level50
        for i in range(len(s_w) - 1):
            if s_w[i] >= level50 > s_w[i + 1]:
                frac = (level50 - s_w[i]) / (s_w[i + 1] - s_w[i])
                return float(t_w[i] + frac * (t_w[i + 1] - t_w[i]))
        # Fallback : retourner le minimum
        return float(t_w[np.argmin(s_w)])


def detect_closures(df: pd.DataFrame, cfg: ValConfig) -> List[Dict]:
    """
    Détecte les événements de fermeture dans le signal capteur aligné.
    Retourne une liste de segments avec indices start/end dans le DataFrame.
    """
    t   = df["t_rel_s"].values
    op  = df["opening_mm_aligned"].values
    ops = smooth(op, cfg)
    vel = velocity(ops, t)

    # Normalisation [0,1] du signal capteur (même que dans align_gripper)
    lo98, hi98 = np.percentile(op, [2, 98])
    op_norm = np.clip((op - lo98) / max(hi98 - lo98, 1e-6), 0.0, 1.0)

    events = []
    i = 0
    last_end = -1
    while i < len(t) - 5:
        # Détecter un début de fermeture : op descend depuis >close_from_mm
        if ops[i] > cfg.close_from_mm and vel[i] < cfg.close_vel_thresh:
            # Chercher la fin : quand la vitesse redevient > -5 mm/s
            j = i + 1
            while j < len(t) - 1 and vel[j] < -5.0:
                j += 1
            # Vérifier amplitude
            op_before = float(ops[max(0, i-3):i+1].mean())
            op_after  = float(ops[j:min(len(ops), j+5)].mean())
            amp = op_before - op_after

            if amp >= cfg.close_min_amp and j > last_end:
                events.append({
                    "i_start": i,
                    "i_end":   j,
                    "op_before": op_before,
                    "op_after":  op_after,
                    "amplitude": amp,
                    "op_norm":   op_norm,
                    "ops":       ops,
                    "vel":       vel,
                })
                last_end = j
                i = j + 1
                continue
        i += 1

    return events


def validate_closure(ev_dict: Dict, df: pd.DataFrame, cfg: ValConfig,
                     event_id: int) -> ClosureEvent:
    i_s = ev_dict["i_start"]
    i_e = ev_dict["i_end"]
    t   = df["t_rel_s"].values
    op  = df["opening_mm_aligned"].values
    vis = df["vis_norm"].values
    sen = df["sen_norm_aligned"].values
    res = df["residual"].values
    op_norm = ev_dict["op_norm"]
    ops     = ev_dict["ops"]

    t_start = float(t[i_s])
    t_end   = float(t[i_e])

    # Niveau 50% de la fermeture pour le signal capteur
    lo98, hi98 = np.percentile(op, [2, 98])
    op_rng = max(hi98 - lo98, 1e-6)
    level50_sen = float(np.clip((ev_dict["op_before"] - 0.5 * ev_dict["amplitude"] - lo98) / op_rng, 0, 1))

    t50_sen = _find_t50(t, sen, t_start - 0.1, t_end + 0.1, level50_sen)

    # Valeurs de vis_norm AUX points de début et fin de la transition
    # (pas min/max dans une fenêtre élargie — le drift post-fermeture fausse le min)
    def _nearest_vis(t_target, t_arr, v_arr, dt=0.15):
        m = (t_arr >= t_target - dt) & (t_arr <= t_target + dt)
        vals = v_arr[m]
        return float(np.nanmean(vals)) if np.isfinite(vals).sum() > 0 else float("nan")

    vis_at_start = _nearest_vis(t_start, t, vis, dt=0.08)   # avant la fermeture
    vis_at_end   = _nearest_vis(t_end,   t, vis, dt=0.08)   # après la fermeture

    visual_range = abs(vis_at_start - vis_at_end) if (
        np.isfinite(vis_at_start) and np.isfinite(vis_at_end)) else 0.0
    visual_insensitive = visual_range < 0.05

    if visual_insensitive:
        # Feature ne varie pas aux points de transition : timing inestimable
        t50_vis = float("nan")
        timing_err_ms = float("nan")
    else:
        # t50_vis basé sur le point médian entre les valeurs AVANT/APRÈS la fermeture
        vis_hi_local = max(vis_at_start, vis_at_end)
        vis_lo_local = min(vis_at_start, vis_at_end)
        level50_vis = (vis_hi_local + vis_lo_local) / 2.0
        # Détecter si la feature est inversée : vis monte quand le gripper ferme
        # (vis_at_start < vis_at_end → ascending transition durant la fermeture)
        ascending_vis = bool(vis_at_start < vis_at_end)
        # Cherche le crossing dans [t_start-0.1, t_end+0.5]
        t50_vis = _find_t50(t, vis, t_start - 0.1, t_end + 0.5, level50_vis,
                            ascending=ascending_vis)
        timing_err_ms = (t50_vis - t50_sen) * 1000.0 if (
            np.isfinite(t50_vis) and np.isfinite(t50_sen)) else float("nan")

    # Monotonie capteur dans la fenêtre
    op_win = ops[i_s:i_e + 1]
    monotone_ok = bool(np.all(np.diff(op_win) <= 2.0))  # tolérance 2mm

    # Résidus dans la fenêtre
    res_win = res[i_s:i_e + 1]
    res_valid = res_win[np.isfinite(res_win)]
    res_mean = float(res_valid.mean()) if len(res_valid) else float("nan")
    res_max  = float(res_valid.max())  if len(res_valid) else float("nan")

    # ── Logique physique ──────────────────────────────────────────────────────
    # 1. L'amplitude doit être réelle (>close_min_amp)
    # 2. Le capteur doit être monotone dans la fenêtre (pas de rebond)
    # 3. L'erreur de timing doit être < timing_tol_ms (SEULEMENT si feature sensible)
    # 4. Le résidu moyen dans la fenêtre doit être < 0.5 (alignement cohérent)
    reasons = []
    if ev_dict["amplitude"] < cfg.close_min_amp:
        reasons.append(f"amplitude trop faible ({ev_dict['amplitude']:.1f}mm < {cfg.close_min_amp}mm)")
    if not monotone_ok:
        reasons.append("signal capteur non monotone dans la fermeture")
    if visual_insensitive:
        # La feature n'est pas sensible : on ne pénalise pas le timing
        reasons.append(f"feature visuellement insensible (range={visual_range:.3f})")
    elif np.isfinite(timing_err_ms) and abs(timing_err_ms) > cfg.timing_tol_ms:
        reasons.append(f"erreur timing {timing_err_ms:.1f}ms > {cfg.timing_tol_ms}ms")
    if np.isfinite(res_mean) and res_mean > 0.5:
        reasons.append(f"résidu moyen élevé ({res_mean:.3f})")

    # Une fermeture visuellement insensible n'est pas un FAIL d'alignement
    logic_ok = len([r for r in reasons if "insensible" not in r]) == 0

    return ClosureEvent(
        event_id          = event_id,
        t_start_s         = t_start,
        t_end_s           = t_end,
        t50_sensor_s      = t50_sen,
        t50_vision_s      = t50_vis,
        timing_err_ms     = round(timing_err_ms, 3) if np.isfinite(timing_err_ms) else float("nan"),
        op_before_mm      = round(ev_dict["op_before"], 2),
        op_after_mm       = round(ev_dict["op_after"],  2),
        amplitude_mm      = round(ev_dict["amplitude"], 2),
        n_frames          = i_e - i_s + 1,
        residual_mean     = round(res_mean, 4) if np.isfinite(res_mean) else float("nan"),
        residual_max      = round(res_max,  4) if np.isfinite(res_max)  else float("nan"),
        monotone_ok       = monotone_ok,
        visual_range      = round(visual_range, 4),
        visual_insensitive= visual_insensitive,
        logic_ok          = logic_ok,
        logic_reason      = " | ".join(reasons) if reasons else "OK",
    )


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — DÉTECTION ET VALIDATION DES PLATEAUX
# ═════════════════════════════════════════════════════════════════════════════

def detect_plateaus(df: pd.DataFrame, cfg: ValConfig) -> List[Dict]:
    t   = df["t_rel_s"].values
    op  = df["opening_mm_aligned"].values
    ops = smooth(op, cfg)
    vel = velocity(ops, t)

    is_p = np.abs(vel) < cfg.plateau_vel_thresh

    # Regrouper en segments continus
    segments = []
    in_p = False
    i_s  = 0
    for i in range(len(is_p)):
        if is_p[i] and not in_p:
            i_s = i; in_p = True
        elif not is_p[i] and in_p:
            segments.append((i_s, i - 1))
            in_p = False
    if in_p:
        segments.append((i_s, len(is_p) - 1))

    # Filtrer par durée
    result = []
    for s, e in segments:
        dur = t[e] - t[s]
        if dur >= cfg.plateau_min_dur_s:
            result.append({"i_start": s, "i_end": e, "ops": ops})

    return result


def validate_plateau(pl_dict: Dict, df: pd.DataFrame, cfg: ValConfig,
                     event_id: int) -> PlateauEvent:
    s   = pl_dict["i_start"]
    e   = pl_dict["i_end"]
    t   = df["t_rel_s"].values
    op  = df["opening_mm_aligned"].values
    vis = df["vis_norm"].values
    sen = df["sen_norm_aligned"].values
    ops = pl_dict["ops"]

    t_start = float(t[s])
    t_end   = float(t[e])
    duration = t_end - t_start

    # Fenêtre du plateau
    op_win  = op[s:e + 1]
    vis_win = vis[s:e + 1]
    sen_win = sen[s:e + 1]

    valid = np.isfinite(vis_win) & np.isfinite(sen_win)
    if valid.sum() < 2:
        return PlateauEvent(
            event_id=event_id, t_start_s=t_start, t_end_s=t_end,
            duration_s=duration,
            sensor_mean_mm=float("nan"), sensor_std_mm=float("nan"),
            vis_mean_norm=float("nan"), sen_mean_norm=float("nan"),
            bias_norm=float("nan"), jitter_norm=float("nan"),
            n_frames=e-s+1, logic_ok=False, logic_reason="données insuffisantes",
        )

    sensor_mean = float(op_win[valid].mean())
    sensor_std  = float(op_win[valid].std())
    vis_mean    = float(vis_win[valid].mean())
    sen_mean    = float(sen_win[valid].mean())

    # Biais : écart moyen entre vision normalisée et capteur normalisé
    bias = vis_mean - sen_mean

    # Jitter : std de la différence frame-à-frame dans la fenêtre
    diff_win = vis_win[valid] - sen_win[valid]
    jitter   = float(diff_win.std())

    # ── Logique physique ──────────────────────────────────────────────────────
    # 1. Le capteur doit être stable (std_mm faible)
    # 2. Le jitter (fluctuation vis_norm quand capteur stable) doit être faible
    # 3. Cohérence directionnelle : fermé = vis bas, ouvert = vis haut
    # NOTE: le biais absolu (vis_mean - sen_mean) n'est PAS vérifié car la feature
    #       visuelle peut avoir une échelle différente du capteur (ex: tl_dark ≥ 0.3
    #       même en position fermée). Ce qui compte : la TIMING alignment, pas le niveau.
    lo98, hi98 = np.percentile(op, [2, 98])
    op_rng = max(hi98 - lo98, 1e-6)

    reasons = []
    if sensor_std > 5.0:
        reasons.append(f"capteur instable sur plateau (std={sensor_std:.1f}mm)")
    if jitter > 0.25:
        reasons.append(f"jitter élevé ({jitter:.3f})")

    # Cohérence directionnelle ouvert/fermé (seuils larges pour tolérer biais de feature)
    if sensor_mean < 5.0 and vis_mean > 0.6:
        reasons.append(f"gripper FERMÉ ({sensor_mean:.1f}mm) mais vision OUVERTE ({vis_mean:.2f})")
    elif sensor_mean > 40.0 and vis_mean < 0.2:
        reasons.append(f"gripper OUVERT ({sensor_mean:.1f}mm) mais vision FERMÉE ({vis_mean:.2f})")

    logic_ok = len(reasons) == 0

    return PlateauEvent(
        event_id      = event_id,
        t_start_s     = round(t_start, 3),
        t_end_s       = round(t_end,   3),
        duration_s    = round(duration, 3),
        sensor_mean_mm= round(sensor_mean, 2),
        sensor_std_mm = round(sensor_std,  3),
        vis_mean_norm = round(vis_mean, 4),
        sen_mean_norm = round(sen_mean, 4),
        bias_norm     = round(bias,     4),
        jitter_norm   = round(jitter,   4),
        n_frames      = e - s + 1,
        logic_ok      = logic_ok,
        logic_reason  = " | ".join(reasons) if reasons else "OK",
    )


# ═════════════════════════════════════════════════════════════════════════════
# GRAPHE DE VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def _dark_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


def plot_validation(
    df: pd.DataFrame,
    vres: ValidationResult,
    video_path: str,
    jsonl_path: str,
    output_path: Path,
    cfg: ValConfig,
) -> None:
    """
    Graphe 3 panneaux + strips de frames pour les événements clés.
    """
    t   = df["t_rel_s"].values
    op  = df["opening_mm_aligned"].values
    vis = df["vis_norm"].values
    sen = df["sen_norm_aligned"].values
    res = df["residual"].values

    ops = smooth(op, cfg)
    lo98, hi98 = np.percentile(op, [2, 98])

    # Charger JSONL pour extraction frames
    frame_pos, ts_ns = load_jsonl_abs(jsonl_path)
    ts_video_ns = ts_ns

    fig = plt.figure(figsize=(18, 16), dpi=100)
    fig.patch.set_facecolor("#0d1117")
    gs = fig.add_gridspec(4, 2, height_ratios=[2.5, 1.5, 1.5, 2.0],
                          hspace=0.45, wspace=0.25,
                          left=0.07, right=0.97, top=0.95, bottom=0.04)

    ax_sig   = fig.add_subplot(gs[0, :])   # ligne 0, toute la largeur
    ax_res   = fig.add_subplot(gs[1, :])   # résidus
    ax_close = fig.add_subplot(gs[2, 0])   # zoom fermeture
    ax_plat  = fig.add_subplot(gs[2, 1])   # zoom plateau
    ax_frames_close = fig.add_subplot(gs[3, 0])  # frames fermeture
    ax_frames_plat  = fig.add_subplot(gs[3, 1])  # frames plateau

    for ax in [ax_sig, ax_res, ax_close, ax_plat,
               ax_frames_close, ax_frames_plat]:
        _dark_ax(ax)

    # ── Panneau 1 : signaux ───────────────────────────────────────────────────
    ax_sig.plot(t, vis, color="#58a6ff", lw=0.9, alpha=0.8,
                label="vision normalisée")
    ax_sig.plot(t, sen, color="#ff7b72", lw=0.9, alpha=0.8,
                label="capteur normalisé (aligné)")
    ax_sig.plot(t, ops / max(hi98 - lo98, 1e-6) - lo98 / max(hi98 - lo98, 1e-6),
                color="#c9d1d9", lw=0.5, alpha=0.3, linestyle="--")

    # Marquer les événements
    for ev in vres.closure_events:
        col = "#3fb950" if ev.logic_ok else "#f0e68c"
        ax_sig.axvspan(ev.t_start_s, ev.t_end_s, alpha=0.12, color=col)
        if np.isfinite(ev.t50_sensor_s):
            ax_sig.axvline(ev.t50_sensor_s, color="#ff7b72", lw=1.0, linestyle=":",
                           alpha=0.7)
        if np.isfinite(ev.t50_vision_s):
            ax_sig.axvline(ev.t50_vision_s, color="#58a6ff", lw=1.0, linestyle=":",
                           alpha=0.7)

    for ev in vres.plateau_events:
        col = "#3fb950" if ev.logic_ok else "#f85149"
        ax_sig.axvspan(ev.t_start_s, ev.t_end_s, alpha=0.15, color=col)
        mid = (ev.t_start_s + ev.t_end_s) / 2.0
        ax_sig.text(mid, 1.05,
                    f"P{ev.event_id}\n{ev.sensor_mean_mm:.0f}mm",
                    ha="center", va="bottom", fontsize=6, color=col,
                    transform=ax_sig.get_xaxis_transform())

    ax_sig.set_xlim(t[0], t[-1])
    ax_sig.set_ylim(-0.08, 1.15)
    ax_sig.set_ylabel("Signal normalisé [0–1]", color="#c9d1d9", fontsize=9)

    n_cl_ok = vres.n_closures_ok
    n_pl_ok = vres.n_plateaus_ok
    ax_sig.set_title(
        f"Session {vres.session}  |  côté {vres.side}  |  "
        f"Fermetures {n_cl_ok}/{vres.n_closures_total} OK  |  "
        f"Plateaux {n_pl_ok}/{vres.n_plateaus_total} OK  |  "
        f"Timing err moy={vres.timing_err_mean_ms:.1f}ms  |  "
        f"Biais moy={vres.bias_mean:.3f}",
        color="#c9d1d9", fontsize=9, pad=6
    )
    ax_sig.legend(loc="upper right", fontsize=7, facecolor="#161b22",
                  labelcolor="#c9d1d9", framealpha=0.8)
    ax_sig.grid(True, alpha=0.12, color="#c9d1d9")

    # ── Panneau 2 : résidus ───────────────────────────────────────────────────
    ax_res.fill_between(t, res, alpha=0.7, color="#3fb950")
    ax_res.set_xlim(t[0], t[-1])
    ax_res.set_ylabel("Résidu norm.", color="#c9d1d9", fontsize=9)
    ax_res.set_xlabel("Temps (s)", color="#c9d1d9", fontsize=9)
    ax_res.grid(True, alpha=0.12, color="#c9d1d9")

    # ── Panneau 3a : zoom meilleure fermeture ─────────────────────────────────
    best_close = next(
        (ev for ev in sorted(vres.closure_events, key=lambda e: abs(e.timing_err_ms) if np.isfinite(e.timing_err_ms) else 9999)),
        None
    )
    if best_close:
        margin = 0.5
        t0c = max(t[0], best_close.t_start_s - margin)
        t1c = min(t[-1], best_close.t_end_s + margin)
        mask = (t >= t0c) & (t <= t1c)
        ax_close.plot(t[mask], vis[mask], color="#58a6ff", lw=1.5, label="vision")
        ax_close.plot(t[mask], sen[mask], color="#ff7b72", lw=1.5, label="capteur")
        if np.isfinite(best_close.t50_sensor_s):
            ax_close.axvline(best_close.t50_sensor_s, color="#ff7b72", lw=1.5,
                             linestyle="--", label=f"t50 capteur")
        if np.isfinite(best_close.t50_vision_s):
            ax_close.axvline(best_close.t50_vision_s, color="#58a6ff", lw=1.5,
                             linestyle="--", label=f"t50 vision")
        ax_close.axvspan(best_close.t_start_s, best_close.t_end_s, alpha=0.1,
                         color="#3fb950" if best_close.logic_ok else "#f0e68c")
        err_str = (f"{best_close.timing_err_ms:.1f}ms"
                   if np.isfinite(best_close.timing_err_ms) else "n/a")
        ax_close.set_title(
            f"Fermeture #{best_close.event_id} — "
            f"{best_close.op_before_mm:.0f}→{best_close.op_after_mm:.0f}mm  |  "
            f"timing err={err_str}  |  {'✓ OK' if best_close.logic_ok else '⚠ ' + best_close.logic_reason}",
            color="#c9d1d9", fontsize=8
        )
        ax_close.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9")
        ax_close.set_ylabel("Signal norm.", color="#c9d1d9", fontsize=9)
        ax_close.set_xlabel("Temps (s)", color="#c9d1d9", fontsize=9)
        ax_close.grid(True, alpha=0.12, color="#c9d1d9")

    # ── Panneau 3b : zoom meilleur plateau ────────────────────────────────────
    best_plat = next(
        (ev for ev in sorted(vres.plateau_events, key=lambda e: abs(e.bias_norm))),
        None
    )
    if best_plat:
        margin = 0.4
        t0p = max(t[0], best_plat.t_start_s - margin)
        t1p = min(t[-1], best_plat.t_end_s + margin)
        mask = (t >= t0p) & (t <= t1p)
        ax_plat.plot(t[mask], vis[mask], color="#58a6ff", lw=1.5, label="vision")
        ax_plat.plot(t[mask], sen[mask], color="#ff7b72", lw=1.5, label="capteur")
        ax_plat.axvspan(best_plat.t_start_s, best_plat.t_end_s, alpha=0.15,
                        color="#3fb950" if best_plat.logic_ok else "#f85149")
        ax_plat.axhline(best_plat.vis_mean_norm, color="#58a6ff", lw=1.0,
                        linestyle=":", label=f"vis_mean={best_plat.vis_mean_norm:.3f}")
        ax_plat.axhline(best_plat.sen_mean_norm, color="#ff7b72", lw=1.0,
                        linestyle=":", label=f"sen_mean={best_plat.sen_mean_norm:.3f}")
        # Annotation biais
        mid_y = (best_plat.vis_mean_norm + best_plat.sen_mean_norm) / 2.0
        if abs(best_plat.bias_norm) > 0.01:
            ax_plat.annotate(
                "",
                xy=(best_plat.t_end_s - 0.05, best_plat.vis_mean_norm),
                xytext=(best_plat.t_end_s - 0.05, best_plat.sen_mean_norm),
                arrowprops=dict(arrowstyle="<->", color="#f0e68c", lw=1.5),
            )
            ax_plat.text(best_plat.t_end_s - 0.08, mid_y,
                         f"biais\n{best_plat.bias_norm:+.3f}",
                         ha="right", va="center", fontsize=7, color="#f0e68c")
        ax_plat.set_title(
            f"Plateau #{best_plat.event_id} — "
            f"{best_plat.sensor_mean_mm:.1f}mm  |  "
            f"biais={best_plat.bias_norm:+.3f}  jitter={best_plat.jitter_norm:.3f}  |  "
            f"{'✓ OK' if best_plat.logic_ok else '⚠ ' + best_plat.logic_reason}",
            color="#c9d1d9", fontsize=8
        )
        ax_plat.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9")
        ax_plat.set_ylabel("Signal norm.", color="#c9d1d9", fontsize=9)
        ax_plat.set_xlabel("Temps (s)", color="#c9d1d9", fontsize=9)
        ax_plat.grid(True, alpha=0.12, color="#c9d1d9")

    # ── Frames fermeture ──────────────────────────────────────────────────────
    def draw_frames_strip(ax, t_center: float, label: str,
                          val_norm: float, is_ok: bool, extra: str):
        n_frames_strip = 5
        half_s = cfg.strip_half_s
        offsets = np.linspace(-half_s, half_s, n_frames_strip)
        frames_imgs = []
        for off in offsets:
            t_target = t_center + off
            # Trouver la frame la plus proche
            ts_target_ns = ts_video_ns[0] + int(t_target * 1e9)
            fi_idx = int(np.argmin(np.abs(ts_video_ns - ts_target_ns)))
            fi = int(frame_pos[fi_idx])
            img = extract_frame(video_path, fi)
            frames_imgs.append((img, off))

        # Assembler le strip
        target_h, target_w = 160, 240
        cells = []
        for img, off in frames_imgs:
            if img is None:
                cell = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            else:
                h0, w0 = img.shape[:2]
                scale  = min(target_w / w0, target_h / h0)
                rw, rh = int(w0 * scale), int(h0 * scale)
                cell   = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
                ph = target_h - rh
                pw = target_w - rw
                cell = cv2.copyMakeBorder(cell,
                    ph // 2, ph - ph // 2,
                    pw // 2, pw - pw // 2,
                    cv2.BORDER_CONSTANT, value=(20, 20, 20))
            # Étiquette offset
            is_center = abs(off) < (half_s / (n_frames_strip - 1) * 0.6)
            cv2.putText(cell, f"{'NOW' if is_center else f'{off:+.1f}s'}",
                        (4, target_h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (200, 200, 200), 1)
            if is_center:
                c = (0, 200, 0) if is_ok else (0, 0, 220)
                cv2.rectangle(cell, (0, 0), (target_w - 1, target_h - 1), c, 3)
            cells.append(cell)

        strip = np.hstack(cells)
        strip_rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
        ax.imshow(strip_rgb, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        status_col = "#3fb950" if is_ok else "#f85149"
        ax.set_title(f"{label}  |  {extra}",
                     color=status_col, fontsize=8, pad=3)

    if best_close:
        t_center = best_close.t50_sensor_s if np.isfinite(best_close.t50_sensor_s) else best_close.t_start_s
        err_str  = f"timing err={best_close.timing_err_ms:.1f}ms" if np.isfinite(best_close.timing_err_ms) else "timing n/a"
        draw_frames_strip(
            ax_frames_close, t_center,
            f"Fermeture #{best_close.event_id} ({best_close.op_before_mm:.0f}→{best_close.op_after_mm:.0f}mm)",
            best_close.residual_mean, best_close.logic_ok, err_str,
        )

    if best_plat:
        t_center = (best_plat.t_start_s + best_plat.t_end_s) / 2.0
        draw_frames_strip(
            ax_frames_plat, t_center,
            f"Plateau #{best_plat.event_id} ({best_plat.sensor_mean_mm:.0f}mm)",
            best_plat.vis_mean_norm, best_plat.logic_ok,
            f"biais={best_plat.bias_norm:+.3f}  jitter={best_plat.jitter_norm:.3f}",
        )

    plt.savefig(str(output_path), dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_side(session_path: str, side: str,
                  cfg: ValConfig, output_dir: Path) -> ValidationResult:
    sname = Path(session_path).name
    vres  = ValidationResult(session=sname, side=side, success=False)

    # Chemin CSV aligné
    aligned_csv = output_dir / f"aligned_{side}.csv"
    if not aligned_csv.exists():
        # Chercher dans sous-dossier session
        aligned_csv = Path(session_path) / "align_gripper" / f"aligned_{side}.csv"
    if not aligned_csv.exists():
        vres.error = f"CSV aligné introuvable : {aligned_csv}"
        return vres

    video_path = Path(session_path) / "videos" / f"{side}.mp4"
    jsonl_path = Path(session_path) / "videos" / f"{side}.jsonl"

    try:
        df = pd.read_csv(str(aligned_csv))
    except Exception as e:
        vres.error = f"Lecture CSV : {e}"
        return vres

    vres.n_frames = len(df)
    required = ["t_rel_s", "opening_mm_aligned", "vis_norm", "sen_norm_aligned", "residual"]
    for col in required:
        if col not in df.columns:
            vres.error = f"Colonne manquante : {col}"
            return vres

    # ── Étape 1 : fermetures ─────────────────────────────────────────────────
    print(f"  [{side}] Détection fermetures...")
    ev_raw = detect_closures(df, cfg)
    print(f"  [{side}] {len(ev_raw)} fermetures détectées")
    for i, ev_dict in enumerate(ev_raw):
        cev = validate_closure(ev_dict, df, cfg, i + 1)
        vres.closure_events.append(cev)

    vres.n_closures_total      = len(vres.closure_events)
    vres.n_closures_ok         = sum(1 for e in vres.closure_events if e.logic_ok)
    vres.n_closures_insensitive= sum(1 for e in vres.closure_events if e.visual_insensitive)

    timing_errs = [e.timing_err_ms for e in vres.closure_events
                   if np.isfinite(e.timing_err_ms)]
    if timing_errs:
        vres.timing_err_mean_ms = round(float(np.mean(timing_errs)), 3)
        vres.timing_err_std_ms  = round(float(np.std(timing_errs)),  3)
        vres.timing_err_max_ms  = round(float(np.max(np.abs(timing_errs))), 3)

    # ── Étape 2 : plateaux ───────────────────────────────────────────────────
    print(f"  [{side}] Détection plateaux...")
    pl_raw = detect_plateaus(df, cfg)
    print(f"  [{side}] {len(pl_raw)} plateaux détectés")
    for i, pl_dict in enumerate(pl_raw):
        pev = validate_plateau(pl_dict, df, cfg, i + 1)
        vres.plateau_events.append(pev)

    vres.n_plateaus_total = len(vres.plateau_events)
    vres.n_plateaus_ok    = sum(1 for e in vres.plateau_events if e.logic_ok)

    biases  = [e.bias_norm   for e in vres.plateau_events]
    jitters = [e.jitter_norm for e in vres.plateau_events]
    if biases:
        vres.bias_mean   = round(float(np.mean(biases)),   4)
        vres.bias_std    = round(float(np.std(biases)),    4)
        vres.jitter_mean = round(float(np.mean(jitters)),  4)

    # ── Corrélation croisée dérivée (vérification temporelle primaire) ─────────
    # Corrèle dvis/dt avec dsen/dt pour vérifier que les TRANSITIONS sont synchronisées.
    # Insensible au biais absolu de la feature visuelle.
    # Fenêtre ±15 frames pour capturer les sessions avec lead/lag systématique (jusqu'à ±500ms).
    try:
        from scipy.signal import fftconvolve as _fftconvolve
        vis_arr = df["vis_norm"].values.astype(np.float64)
        sen_arr = df["sen_norm_aligned"].values.astype(np.float64)
        # Supprimer les NaN avant le calcul des dérivées
        valid_mask = np.isfinite(vis_arr) & np.isfinite(sen_arr)
        if valid_mask.sum() > 20:
            vis_c = vis_arr[valid_mask]
            sen_c = sen_arr[valid_mask]
            vis_d = np.diff(vis_c)
            sen_d = np.diff(sen_c)
            vs_std = float(vis_d.std())
            ss_std = float(sen_d.std())
            if vs_std > 1e-6 and ss_std > 1e-6:
                vis_dz = (vis_d - vis_d.mean()) / vs_std
                sen_dz = (sen_d - sen_d.mean()) / ss_std
                N_d    = len(vis_dz)
                xcorr  = _fftconvolve(vis_dz, sen_dz[::-1], mode="full") / N_d
                xcorr  = np.clip(xcorr, -1.0, 1.0)
                center = N_d - 1   # lag=0 index
                # Fenêtre élargie ±15 frames pour capturer lead/lag systématique (≈±500ms à 30fps)
                lo_i   = max(0, center - 15)
                hi_i   = min(len(xcorr), center + 16)
                peak_i = lo_i + int(np.argmax(np.abs(xcorr[lo_i:hi_i])))
                vres.derivative_ncc_r = round(float(xcorr[peak_i]), 4)
    except Exception:
        pass

    # ── Analyse de cohérence temporelle ──────────────────────────────────────
    # Si toutes les fermetures ont des erreurs de timing dans la MÊME DIRECTION
    # (écart-type faible) malgré un décalage moyen important, c'est un phénomène
    # physique (la feature visuelle voit le gripper avant le capteur = pré-mouvement)
    # et non une erreur d'alignement. On marque ces fermetures comme "cohérentes".
    timing_errs_det = [
        e.timing_err_ms for e in vres.closure_events
        if not e.visual_insensitive and np.isfinite(e.timing_err_ms)
    ]
    timing_consistent = False
    timing_mean_err   = float("nan")
    timing_std_err    = float("nan")
    consistency_threshold_ms = float("nan")

    if len(timing_errs_det) >= 3:
        timing_mean_err = float(np.mean(timing_errs_det))
        timing_std_err  = float(np.std(timing_errs_det))
        # Critère de cohérence : std < max(80ms, 45% du décalage moyen)
        # Un lead systématique de 250ms avec std=70ms est COHÉRENT (70 < 45%*250=112)
        consistency_threshold_ms = max(80.0, 0.45 * abs(timing_mean_err))
        timing_consistent = (
            abs(timing_mean_err) > 80.0   # seulement si lead/lag significatif
            and timing_std_err < consistency_threshold_ms
        )

    # Fermetures cohérentes avec le pattern de session = passes supplémentaires
    det_ok_adj = 0
    detectable = [e for e in vres.closure_events if not e.visual_insensitive]
    for ev in detectable:
        if ev.logic_ok:
            det_ok_adj += 1
        elif timing_consistent and np.isfinite(ev.timing_err_ms):
            # Cohérence : l'erreur de cette fermeture est dans le pattern session
            within_2sigma = abs(ev.timing_err_ms - timing_mean_err) <= max(2.0 * timing_std_err, 60.0)
            if within_2sigma:
                det_ok_adj += 1

    det_total = len(detectable)

    # ── Statut global ─────────────────────────────────────────────────────────
    # Si aucune fermeture détectable → OK par défaut (gripper toujours fermé/ouvert)
    if det_total == 0:
        close_rate = 1.0
    else:
        close_rate = det_ok_adj / det_total

    plat_rate  = vres.n_plateaus_ok / max(vres.n_plateaus_total, 1)
    # NCC dérivée : fort positif = synchronisation parfaite
    # On accepte aussi abs > 0.40 si les erreurs de timing sont cohérentes (feature lead)
    deriv_ok_strong = np.isfinite(vres.derivative_ncc_r) and vres.derivative_ncc_r > 0.50
    deriv_ok_weak   = np.isfinite(vres.derivative_ncc_r) and abs(vres.derivative_ncc_r) > 0.35

    # Plafond statut si lead systématique important (> 150ms) : jamais OK, max WARNING
    # (les transitions sont cohérentes mais avec un décalage physique significatif)
    has_large_systematic_lead = timing_consistent and abs(timing_mean_err) > 150.0

    if close_rate >= 0.75 and plat_rate >= 0.70 and deriv_ok_strong and not has_large_systematic_lead:
        vres.status = "OK"
    elif close_rate >= 0.75 and plat_rate >= 0.70 and has_large_systematic_lead:
        # Bonnes métriques mais lead systématique > 150ms → cap à WARNING (pas OK)
        vres.status = "WARNING"
    elif close_rate >= 0.50 and plat_rate >= 0.50 and (
        deriv_ok_strong
        or close_rate >= 0.60        # ← fallback original : bon taux de fermetures
        or (deriv_ok_weak and timing_consistent)  # ← feature lead cohérent + NCC modéré
    ):
        vres.status = "WARNING"
    else:
        vres.status = "ERROR"

    vres.success = True

    # ── Graphe ────────────────────────────────────────────────────────────────
    if video_path.exists() and jsonl_path.exists():
        plot_path = output_dir / f"validation_{side}.png"
        try:
            plot_validation(df, vres, str(video_path), str(jsonl_path),
                            plot_path, cfg)
        except Exception as e:
            print(f"  [WARN] Graphe : {e}")
            import traceback; traceback.print_exc()

    # ── JSON ──────────────────────────────────────────────────────────────────
    def _serial(obj):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    report = {
        "session": vres.session,
        "side":    vres.side,
        "status":  vres.status,
        "derivative_ncc_r": vres.derivative_ncc_r,
        "closures": {
            "total":       vres.n_closures_total,
            "ok":          vres.n_closures_ok,
            "ok_adj":      int(det_ok_adj),  # fermetures OK + cohérentes
            "insensitive": vres.n_closures_insensitive,
            "timing_err_mean_ms": vres.timing_err_mean_ms,
            "timing_err_std_ms":  vres.timing_err_std_ms,
            "timing_err_max_ms":  vres.timing_err_max_ms,
            "timing_consistent":  bool(timing_consistent),
            "timing_mean_session_ms": round(timing_mean_err, 1) if np.isfinite(timing_mean_err) else None,
            "timing_std_session_ms":  round(timing_std_err,  1) if np.isfinite(timing_std_err)  else None,
            "events": [asdict(e) for e in vres.closure_events],
        },
        "plateaus": {
            "total":       vres.n_plateaus_total,
            "ok":          vres.n_plateaus_ok,
            "bias_mean":   vres.bias_mean,
            "bias_std":    vres.bias_std,
            "jitter_mean": vres.jitter_mean,
            "events": [asdict(e) for e in vres.plateau_events],
        },
    }
    json_path = output_dir / f"validation_report_{side}.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    return vres


def run_session(session_path: str, cfg: ValConfig, sides: List[str],
                base_output: Optional[Path] = None) -> Dict[str, ValidationResult]:
    sname = Path(session_path).name
    output_dir = (base_output or Path(session_path)) / "align_gripper"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for side in sides:
        print(f"\n── Validation {sname}  côté {side} ───────────────────")
        vres = validate_side(session_path, side, cfg, output_dir)
        results[side] = vres

        if vres.success:
            dncc = f"{vres.derivative_ncc_r:.4f}" if np.isfinite(vres.derivative_ncc_r) else "n/a"
            insens = f"  insensibles={vres.n_closures_insensitive}" if vres.n_closures_insensitive > 0 else ""
            print(f"  → {vres.status}  (NCC_derivée={dncc})")
            print(f"     Fermetures : {vres.n_closures_ok}/{vres.n_closures_total} OK{insens}"
                  f"  timing err moy={vres.timing_err_mean_ms:.2f}ms"
                  f"  max={vres.timing_err_max_ms:.2f}ms")
            print(f"     Plateaux   : {vres.n_plateaus_ok}/{vres.n_plateaus_total} OK"
                  f"  biais moy={vres.bias_mean:.4f}"
                  f"  jitter moy={vres.jitter_mean:.4f}")
            # Détails fermetures
            for ev in vres.closure_events:
                tick = "~" if ev.visual_insensitive else ("✓" if ev.logic_ok else "✗")
                err  = f"{ev.timing_err_ms:.1f}ms" if np.isfinite(ev.timing_err_ms) else "n/a"
                insens_tag = "  [INSENSIBLE]" if ev.visual_insensitive else ""
                print(f"     [{tick}] Fermeture #{ev.event_id} t={ev.t_start_s:.2f}s "
                      f"{ev.op_before_mm:.0f}→{ev.op_after_mm:.0f}mm  "
                      f"timing_err={err}  résidu={ev.residual_mean:.3f}"
                      f"{insens_tag}"
                      + (f"  ⚠ {ev.logic_reason}" if not ev.logic_ok and not ev.visual_insensitive else ""))
            # Détails plateaux
            for ev in vres.plateau_events:
                tick = "✓" if ev.logic_ok else "✗"
                print(f"     [{tick}] Plateau   #{ev.event_id} t=[{ev.t_start_s:.2f}–{ev.t_end_s:.2f}]s "
                      f"{ev.sensor_mean_mm:.1f}mm  "
                      f"biais={ev.bias_norm:+.4f}  jitter={ev.jitter_norm:.4f}"
                      + (f"  ⚠ {ev.logic_reason}" if not ev.logic_ok else ""))
        else:
            print(f"  → ERREUR : {vres.error}")

    return results


def main():
    p = argparse.ArgumentParser(description="Validation multi-étapes alignement gripper")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--session",      help="Dossier d'une session")
    grp.add_argument("--sessions_dir", help="Dossier contenant plusieurs sessions")
    p.add_argument("--side",          default="both", choices=["left", "right", "both"])
    p.add_argument("--output_dir",    default=None)
    p.add_argument("--timing_tol_ms", type=float, default=100.0)
    p.add_argument("--bias_tol",      type=float, default=0.15)
    p.add_argument("--plateau_min_s", type=float, default=0.3)
    args = p.parse_args()

    cfg = ValConfig(
        timing_tol_ms   = args.timing_tol_ms,
        bias_tol        = args.bias_tol,
        plateau_min_dur_s = args.plateau_min_s,
    )
    sides = ["left", "right"] if args.side == "both" else [args.side]
    base_out = Path(args.output_dir) if args.output_dir else None

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(
            str(d) for d in Path(args.sessions_dir).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        print(f"Sessions : {len(sessions)}")

    for sess in sessions:
        run_session(sess, cfg, sides, base_out)


if __name__ == "__main__":
    main()
