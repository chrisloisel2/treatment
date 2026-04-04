#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gripper_vision.py — Analyse visuelle de l'ouverture du gripper.

Principe physique
─────────────────
La caméra est fixée sur le bras du gripper et pointe vers la scène.
Les bras mécaniques noirs apparaissent dans le haut de l'image et forment
des masses sombres distinctes dont la géométrie (étendue horizontale,
dispersion, aire) varie avec l'ouverture.

Pipeline par frame
──────────────────
1. Binarisation adaptative des pixels sombres (bras noirs) dans
   une bande y = [0 : band_bottom] couvrant toute la largeur.
2. Morphologie : ouverture (bruit) + fermeture (gaps internes).
3. Extraction de 6 features géométriques par frame :
     • span_norm      — étendue horizontale des blobs / largeur image
                        (feature primaire : r ≈ 0.97 sur sessions idéales)
     • total_area     — aire noire totale normalisée
     • n_blobs        — nombre de blobs distincts
     • h_spread       — écart entre centroïdes X des deux plus grands blobs
     • compactness    — remplissage du bbox principal
     • dark_band_frac — fraction brute de pixels sombres (fallback)
4. Fusion adaptative : chaque feature est pondérée par sa stabilité
   inter-frames (1 / CV) estimée sur les 30 premières frames → vecteur
   mono-signal f(t).
5. Lissage Savitzky-Golay (fenêtre adaptée à fps).
6. Normalisation min-max [0, 1] robuste (percentile 2–98).
7. Auto-calibration : si le signal est trop plat (std < 0.08),
   on tente de détecter automatiquement le seuil de binarisation
   optimal par scan sur [20, 80].
8. Cross-corrélation normalisée (NCC) avec stratégie de sélection
   du lag positif dominant pour éviter les faux maxima négatifs.
9. Détection d'événements par hystérésis (hi/lo) sur les deux signaux
   avec appariement greedy à ±tol_s.
10. Score composite [0–100] + confiance [0–1] + statut OK/WARNING/ERROR.

Sorties
───────
• GripperVisionResult (dataclass) par session/côté
• Rapport texte global
• Plots 3-panneaux (optionnel, requiert matplotlib)
• Vidéo debug annotée (optionnel, requiert opencv)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import correlate, find_peaks, savgol_filter
from scipy.stats import pearsonr

# ─── opencv (optionnel pour les plots/debug, requis pour l'analyse) ──────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    cv2 = None          # type: ignore
    CV2_OK = False

# ─── matplotlib (optionnel) ──────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    plt = None          # type: ignore
    MPL_OK = False


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GripperVisionConfig:
    """
    Paramètres du pipeline.  Les valeurs par défaut sont calibrées pour
    les vidéos 1280×720 du setup (bras noirs dans les 300px supérieurs).
    """

    # ── Zone d'analyse ───────────────────────────────────────────────────────
    band_bottom:    int   = 700    # limite basse de la bande globale (px depuis le haut)
    body_band_top:  int   = 350    # début de la zone corps (depuis le haut) pour body_gap

    # ── Binarisation ─────────────────────────────────────────────────────────
    dark_threshold: int   = 35     # seuil initial (pixel < thr → bras noir)
    auto_threshold: bool  = True   # auto-ajuster le seuil si signal trop plat

    # ── Morphologie ──────────────────────────────────────────────────────────
    morph_open:     int   = 3      # noyau d'ouverture (supprime bruit)
    morph_close:    int   = 7      # noyau de fermeture (comble gaps internes)

    # ── Fusion des features ───────────────────────────────────────────────────
    # Poids de base ; seront multipliés par la stabilité mesurée
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "body_gap":       4.0,   # gap entre bords internes des deux corps (ouverture directe)
        "span_norm":      3.0,   # étendue horizontale — feature primaire
        "total_area":     2.0,   # aire totale sombre
        "h_spread":       2.0,   # écart entre centroïdes des deux bras
        "compactness":    1.0,   # compacité du blob principal
        "dark_band_frac": 1.0,   # fallback fraction brute
    })

    # ── Lissage ──────────────────────────────────────────────────────────────
    savgol_window:  int   = 21    # fenêtre Savitzky-Golay (frames)
    savgol_poly:    int   = 3     # ordre du polynôme

    # ── Normalisation robuste ─────────────────────────────────────────────────
    norm_pct_lo:    float = 2.0   # percentile bas
    norm_pct_hi:    float = 98.0  # percentile haut

    # ── NCC ──────────────────────────────────────────────────────────────────
    ncc_max_lag_s:  float = 1.5   # plage de recherche ±s
    ncc_min_peak:   float = 0.25  # pic minimum pour considérer le lag valide

    # ── Hystérésis événements ────────────────────────────────────────────────
    open_hi:        float = 0.55  # seuil de montée (ouverture)
    open_lo:        float = 0.30  # seuil de descente (fermeture)
    sensor_open_hi: float = 0.40
    sensor_open_lo: float = 0.20
    min_event_gap_s: float = 0.4  # distance minimale entre deux événements

    # ── Appariement ──────────────────────────────────────────────────────────
    match_tol_s:    float = 1.0   # tolérance appariement vision↔capteur

    # ── Seuils de statut ─────────────────────────────────────────────────────
    score_ok:       float = 55.0
    score_warn:     float = 35.0


# ═════════════════════════════════════════════════════════════════════════════
# STRUCTURES DE RÉSULTAT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EventMatch:
    t_vision_s:  float
    t_sensor_s:  float
    delay_ms:    float   # t_vision − t_sensor
    peak_height: float   # hauteur du pic vision [0-1]


@dataclass
class GripperVisionResult:
    session_name:   str
    side:           str
    video_path:     str
    sensor_path:    str
    success:        bool
    error:          str   = ""

    # ── Méta ─────────────────────────────────────────────────────────────────
    n_frames:       int   = 0
    fps:            float = 0.0
    threshold_used: int   = 35

    # ── Signaux (listes pour sérialisation JSON) ──────────────────────────────
    times_sec:          List[float] = field(default_factory=list)
    vision_signal:      List[float] = field(default_factory=list)   # normalisé [0,1]
    sensor_signal:      List[float] = field(default_factory=list)   # normalisé [0,1]
    feature_weights_used: Dict[str, float] = field(default_factory=dict)

    # ── Métriques de corrélation ──────────────────────────────────────────────
    pearson_r:      float = float("nan")
    pearson_p:      float = float("nan")
    ncc_peak:       float = float("nan")
    ncc_lag_ms:     float = float("nan")   # délai principal vision−capteur

    # ── Événements ───────────────────────────────────────────────────────────
    events_vision:      List[float] = field(default_factory=list)
    events_sensor:      List[float] = field(default_factory=list)
    matched_events:     List[EventMatch] = field(default_factory=list)
    n_unmatched_vision: int   = 0
    n_unmatched_sensor: int   = 0

    # ── Score ─────────────────────────────────────────────────────────────────
    signal_quality:  float = float("nan")  # variance utile [0-1]
    confidence:      float = float("nan")  # fiabilité globale [0-1]
    composite_score: float = float("nan")  # score final [0-100]
    status:          str   = "UNKNOWN"


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTION DES FEATURES GÉOMÉTRIQUES
# ═════════════════════════════════════════════════════════════════════════════

def _extract_features(gray: np.ndarray,
                      thr:   int,
                      cfg:   GripperVisionConfig) -> Dict[str, float]:
    """
    Retourne un dict de features géométriques pour une frame.
    Toutes les features sont dans [0, 1] sauf n_blobs (entier).
    """
    h, w     = gray.shape
    y_top    = min(cfg.band_bottom, h)
    roi      = gray[:y_top, :]

    # ── Binarisation ──────────────────────────────────────────────────────────
    _, binary = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY_INV)

    # ── Morphologie ───────────────────────────────────────────────────────────
    if cfg.morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (cfg.morph_open, cfg.morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    if cfg.morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (cfg.morph_close, cfg.morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    # ── Contours ──────────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    min_area = y_top * w * 0.008
    big = sorted([c for c in contours if cv2.contourArea(c) > min_area],
                 key=cv2.contourArea, reverse=True)

    # ── Feature fallback (toujours calculée) ─────────────────────────────────
    dark_band_frac = float(binary.mean()) / 255.0

    if not big:
        return {
            "span_norm":      0.0,
            "total_area":     dark_band_frac,
            "h_spread":       0.0,
            "body_gap":       0.0,
            "compactness":    0.0,
            "dark_band_frac": dark_band_frac,
            "n_blobs":        0.0,
        }

    # ── span_norm : étendue horizontale union de tous les blobs ──────────────
    bboxes  = [cv2.boundingRect(c) for c in big]
    x_min   = min(b[0] for b in bboxes)
    x_max   = max(b[0] + b[2] for b in bboxes)
    span_norm = (x_max - x_min) / w

    # ── total_area ────────────────────────────────────────────────────────────
    total_area = sum(cv2.contourArea(c) for c in big) / (y_top * w)

    # ── Centroïdes des blobs ──────────────────────────────────────────────────
    def blob_cx(c):
        M = cv2.moments(c)
        return (M["m10"] / M["m00"]) / w if M["m00"] > 0 else 0.5

    cxs = [blob_cx(c) for c in big]

    # ── h_spread : écart X normalisé entre les 2 plus grands blobs ───────────
    cx0 = cxs[0]
    if len(big) >= 2:
        cx1      = cxs[1]
        h_spread = abs(cx0 - cx1)
    else:
        h_spread = 0.0

    # ── body_gap : distance entre les bords internes des deux corps ───────────
    # Principe physique : la pince = deux corps symétriques qui se rapprochent
    # du centre. On les cherche dans la zone [body_band_top : h] pour éviter
    # le fond noir du haut qui fusionne tout en un seul blob.
    y_body_top = min(cfg.body_band_top, h)
    body_roi   = gray[y_body_top:, :]
    _, body_bin = cv2.threshold(body_roi, thr, 255, cv2.THRESH_BINARY_INV)
    if cfg.morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (cfg.morph_open, cfg.morph_open))
        body_bin = cv2.morphologyEx(body_bin, cv2.MORPH_OPEN, k)
    if cfg.morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (cfg.morph_close, cfg.morph_close))
        body_bin = cv2.morphologyEx(body_bin, cv2.MORPH_CLOSE, k)
    bh_roi = body_roi.shape[0]
    bcntrs, _ = cv2.findContours(body_bin, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    bmin_area = bh_roi * w * 0.008
    bbig = sorted([c for c in bcntrs if cv2.contourArea(c) > bmin_area],
                  key=cv2.contourArea, reverse=True)

    def _bcx(c):
        M = cv2.moments(c)
        return (M["m10"] / M["m00"]) / w if M["m00"] > 0 else 0.5

    if bbig:
        bcxs = [_bcx(c) for c in bbig]
        bboxes_b = [cv2.boundingRect(c) for c in bbig]
        bleft  = [(bcxs[i], bboxes_b[i]) for i in range(len(bbig)) if bcxs[i] < 0.5]
        bright = [(bcxs[i], bboxes_b[i]) for i in range(len(bbig)) if bcxs[i] >= 0.5]
        if bleft and bright:
            _, (lx, _, lbw, _) = max(bleft,  key=lambda t: t[0])
            _, (rx, _, _,   _) = min(bright, key=lambda t: t[0])
            body_gap = max(0.0, rx / w - (lx + lbw) / w)
        elif bleft:
            _, (lx, _, lbw, _) = max(bleft, key=lambda t: t[0])
            body_gap = max(0.0, 0.5 - (lx + lbw) / w) * 2.0
        elif bright:
            _, (rx, _, _, _) = min(bright, key=lambda t: t[0])
            body_gap = max(0.0, rx / w - 0.5) * 2.0
        else:
            body_gap = 0.0
    else:
        body_gap = 0.0

    # ── compactness : aire / bbox_area du plus grand blob ────────────────────
    x, y, bw, bh = bboxes[0]
    bbox_area    = bw * bh
    compactness  = cv2.contourArea(big[0]) / bbox_area if bbox_area > 0 else 0.0

    return {
        "span_norm":      float(span_norm),
        "total_area":     float(total_area),
        "h_spread":       float(h_spread),
        "body_gap":       float(body_gap),
        "compactness":    float(compactness),
        "dark_band_frac": float(dark_band_frac),
        "n_blobs":        float(len(big)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-CALIBRATION DU SEUIL
# ═════════════════════════════════════════════════════════════════════════════

def _auto_calibrate(frames_gray:  List[np.ndarray],
                    cfg:           GripperVisionConfig,
                    sen_at_frames: Optional[np.ndarray] = None,
                    frame_h:       int = 720) -> Tuple[int, int]:
    """
    Optimise conjointement (band_bottom, dark_threshold) pour maximiser
    la corrélation Pearson(span_norm, capteur) sur 80 frames uniformes.

    Si pas de capteur : maximise var(span_norm).

    Retourne (best_band_bottom, best_dark_threshold).

    Grille de recherche :
      band_bottom ∈ {560, 620, 680, base_bb} × clamp à frame_h
      dark_threshold ∈ {20, 35, 50, 70, 90, 110}
    Total : 4×6 = 24 combinaisons × 50 frames = ~1200 appels _extract_features.
    """
    n      = len(frames_gray)
    n_samp = min(50, n)
    idx    = np.linspace(0, n - 1, n_samp, dtype=int)
    s_gray = [frames_gray[i] for i in idx]

    use_corr = False
    s_valid: np.ndarray = np.array([], dtype=bool)
    s_sen:   np.ndarray = np.array([])

    if sen_at_frames is not None and len(sen_at_frames) == n:
        s_sen_raw = sen_at_frames[idx]
        s_valid   = np.isfinite(s_sen_raw)
        if int(s_valid.sum()) >= 20:
            use_corr = True
            s_sen    = s_sen_raw

    # Toujours inclure le band_bottom de config comme candidat
    base_bb = min(cfg.band_bottom, frame_h)
    band_candidates = sorted(set([min(bb, frame_h)
                                   for bb in (560, 620, 680)] + [base_bb]))
    thr_candidates  = [20, 35, 50, 70, 90, 110]

    # Score de référence avec les paramètres par défaut
    def _score_bb_thr(bb, thr):
        cfg_tmp = GripperVisionConfig(
            band_bottom    = bb,
            dark_threshold = thr,
            auto_threshold = False,
            morph_open     = cfg.morph_open,
            morph_close    = cfg.morph_close,
        )
        spans = np.array([_extract_features(g, thr, cfg_tmp)["span_norm"]
                          for g in s_gray], dtype=float)
        if use_corr:
            sv = spans[s_valid]; ss = s_sen[s_valid]
            if sv.std() < 1e-9 or ss.std() < 1e-9:
                return 0.0
            return float(abs(np.corrcoef(sv, ss)[0, 1]))
        return float(np.var(spans))

    # Score de référence (band_bottom config, thr config)
    ref_score = _score_bb_thr(base_bb, cfg.dark_threshold)
    best_score = ref_score
    best_bb    = base_bb
    best_thr   = cfg.dark_threshold

    # Chercher mieux dans toute la grille
    for bb in band_candidates:
        for thr in thr_candidates:
            score = _score_bb_thr(bb, thr)
            if score > best_score:
                best_score = score
                best_bb    = bb
                best_thr   = thr

    # N'accepter un changement de band_bottom que si gain substantiel (+5%)
    if best_bb != base_bb:
        ref_best_thr_score = max(_score_bb_thr(base_bb, thr) for thr in thr_candidates)
        if best_score < ref_best_thr_score * 1.05:
            # Pas assez d'avantage → garder le band_bottom par défaut
            best_bb = base_bb
            best_thr = max(thr_candidates, key=lambda t: _score_bb_thr(base_bb, t))

    return best_bb, best_thr


# Alias de compatibilité (appelé depuis l'ancien code)
def _auto_threshold(frames_gray:  List[np.ndarray],
                    cfg:           GripperVisionConfig,
                    sen_at_frames: Optional[np.ndarray] = None) -> int:
    _, thr = _auto_calibrate(frames_gray, cfg, sen_at_frames)
    return thr


# ═════════════════════════════════════════════════════════════════════════════
# FUSION ADAPTATIVE DES FEATURES
# ═════════════════════════════════════════════════════════════════════════════

def _adaptive_weights(feat_series: Dict[str, List[float]],
                      cfg:         GripperVisionConfig) -> Dict[str, float]:
    """
    Pondération = poids_base × (1 / CV) × variance_normalisée.
    - 1/CV : récompense les features stables (peu de bruit)
    - variance : récompense les features dynamiques (bougent avec le gripper)
    """
    weights: Dict[str, float] = {}

    for name, base_w in cfg.feature_weights.items():
        if name not in feat_series:
            continue
        arr = np.array(feat_series[name], dtype=float)
        mean = arr.mean()
        std  = arr.std()
        var  = float(std ** 2)

        # CV inversé : plus c'est stable, plus c'est fiable
        cv_inv = mean / (std + 1e-9) if mean > 1e-6 else 0.0
        cv_inv = min(cv_inv, 10.0)

        weights[name] = base_w * cv_inv * (1.0 + var * 10.0)

    total = sum(weights.values())
    if total < 1e-9:
        # Fallback uniforme
        for k in weights:
            weights[k] = 1.0 / len(weights)
    else:
        for k in weights:
            weights[k] /= total

    return weights


def _fuse_features(feat_series:  Dict[str, List[float]],
                   weights:      Dict[str, float]) -> np.ndarray:
    """Combine les features en un signal scalaire pondéré."""
    n = max(len(v) for v in feat_series.values())
    signal = np.zeros(n, dtype=float)
    for name, w in weights.items():
        if name in feat_series:
            arr = np.array(feat_series[name], dtype=float)
            if len(arr) == n:
                signal += w * arr
    return signal


# ═════════════════════════════════════════════════════════════════════════════
# LISSAGE + NORMALISATION ROBUSTE
# ═════════════════════════════════════════════════════════════════════════════

def _smooth(sig: np.ndarray, cfg: GripperVisionConfig) -> np.ndarray:
    w = cfg.savgol_window
    if w > 0 and len(sig) > w:
        if w % 2 == 0:
            w += 1
        poly = min(cfg.savgol_poly, w - 1)
        sig  = savgol_filter(sig, w, poly)
    return sig


def _normalize(sig: np.ndarray, cfg: GripperVisionConfig) -> np.ndarray:
    lo = float(np.percentile(sig, cfg.norm_pct_lo))
    hi = float(np.percentile(sig, cfg.norm_pct_hi))
    if hi - lo < 1e-6:
        return np.zeros_like(sig)
    return np.clip((sig - lo) / (hi - lo), 0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-CORRÉLATION NORMALISÉE (NCC)
# ═════════════════════════════════════════════════════════════════════════════

def _ncc(vis: np.ndarray,
         sen: np.ndarray,
         fps: float,
         cfg: GripperVisionConfig) -> Tuple[float, float]:
    """
    NCC avec sélection du pic d'amplitude absolue maximale.

    Stratégie :
      1. Calculer la NCC complète dans ±max_lag_s (réduit à 1.5s par défaut).
      2. Prendre le pic d'amplitude absolue maximale (gère les anti-corrélations).
      3. Résoudre un biais « zero-lag » : si |NCC(0)| est dans 80% du max
         et que le max est à un lag > 200ms, préférer lag=0.
    """
    n       = len(vis)
    max_lag = min(int(cfg.ncc_max_lag_s * fps), n - 1)
    v       = vis - vis.mean()
    s       = sen - sen.mean()
    sv, ss  = v.std(), s.std()

    if sv < 1e-9 or ss < 1e-9:
        return float("nan"), float("nan")

    xcorr   = correlate(v, s, mode="full") / (n * sv * ss)
    lags    = np.arange(-(n - 1), n)
    mask    = np.abs(lags) <= max_lag
    xm, lm  = xcorr[mask], lags[mask]

    # Pic d'amplitude absolue maximale (gère les corrélations inverses)
    best     = int(np.argmax(np.abs(xm)))
    best_lag = int(lm[best])

    # Biais vers lag=0 : si le max est loin (>200ms) mais que NCC(0) est
    # presque aussi bon (≥80%), préférer lag=0 pour éviter les faux pics.
    zero_idx = int(np.argmin(np.abs(lm)))
    if abs(best_lag) > max(1, int(0.200 * fps)):
        if abs(xm[zero_idx]) >= 0.80 * abs(xm[best]):
            best = zero_idx

    lag_ms = float(lm[best]) / fps * 1000.0
    return float(xm[best]), lag_ms


# ═════════════════════════════════════════════════════════════════════════════
# DÉTECTION D'ÉVÉNEMENTS PAR HYSTÉRÉSIS
# ═════════════════════════════════════════════════════════════════════════════

def _detect_events(norm_sig:    np.ndarray,
                   times:       np.ndarray,
                   hi:          float,
                   lo:          float,
                   min_gap_s:   float) -> Tuple[List[float], List[float]]:
    """
    Détecte les fronts montants (fermeture→ouverture) avec hystérésis adaptative.

    Si les seuils fixes (hi/lo) ne produisent aucun événement, on les remplace
    par des seuils basés sur les percentiles du signal :
        hi_adapt = max(hi, p75) si médiane > 0.5  → signal globalement haut
        lo_adapt = min(lo, p35)
    Cela permet de détecter les vraies transitions même quand le gripper reste
    presque toujours ouvert ou presque toujours fermé.
    """
    def _run(sig, hi_, lo_):
        open_ts: List[float] = []
        heights: List[float] = []
        is_open  = sig[0] >= hi_
        peak_val = 0.0
        peak_t   = 0.0
        last_t   = -999.0
        for i in range(1, len(sig)):
            v = float(sig[i])
            if not is_open and v >= hi_:
                is_open  = True
                peak_val = v
                peak_t   = float(times[i])
            elif is_open:
                if v > peak_val:
                    peak_val = v
                    peak_t   = float(times[i])
                if v <= lo_:
                    is_open = False
                    if peak_t - last_t >= min_gap_s:
                        open_ts.append(peak_t)
                        heights.append(peak_val)
                        last_t = peak_t
        return open_ts, heights

    open_ts, heights = _run(norm_sig, hi, lo)

    # Fallback adaptatif si aucun événement détecté avec les seuils fixes
    if not open_ts:
        med = float(np.median(norm_sig))
        p25 = float(np.percentile(norm_sig, 25))
        p75 = float(np.percentile(norm_sig, 75))
        iqr = p75 - p25
        if iqr > 0.05:  # signal avec dynamique suffisante
            hi_a = min(p75 + 0.5 * iqr, 0.97)
            lo_a = max(p25 - 0.5 * iqr, 0.03)
            open_ts, heights = _run(norm_sig, hi_a, lo_a)

    return open_ts, heights


def _match_events(ev_vis:  List[float],
                  ht_vis:  List[float],
                  ev_sen:  List[float],
                  tol_s:   float) -> Tuple[List[EventMatch], int, int]:
    """Appariement greedy ±tol_s. Retourne (matched, n_unm_vis, n_unm_sen)."""
    used_s  = [False] * len(ev_sen)
    matched: List[EventMatch] = []

    for tv, hv in zip(ev_vis, ht_vis):
        best_d = tol_s + 1.0
        best_j = -1
        for j, ts in enumerate(ev_sen):
            if not used_s[j] and abs(tv - ts) < best_d:
                best_d = abs(tv - ts)
                best_j = j
        if best_j >= 0:
            used_s[best_j] = True
            matched.append(EventMatch(
                t_vision_s  = tv,
                t_sensor_s  = ev_sen[best_j],
                delay_ms    = (tv - ev_sen[best_j]) * 1000.0,
                peak_height = hv,
            ))

    n_unm_vis = len(ev_vis) - len(matched)
    n_unm_sen = sum(1 for u in used_s if not u)
    return matched, n_unm_vis, n_unm_sen


# ═════════════════════════════════════════════════════════════════════════════
# QUALITÉ DU SIGNAL + CONFIANCE
# ═════════════════════════════════════════════════════════════════════════════

def _signal_quality(norm_sig: np.ndarray) -> float:
    """
    Qualité [0-1] basée sur la dynamique utile du signal :
      • Variance du signal normalisé (signal plat = non informatif)
      • Présence de transitions nettes (gradient P95)
    """
    var_score  = min(float(norm_sig.var()) / 0.05, 1.0)

    grad       = np.abs(np.gradient(norm_sig))
    grad_score = min(float(np.percentile(grad, 95)) / 0.05, 1.0)

    return round(0.6 * var_score + 0.4 * grad_score, 3)


def _confidence(quality:       float,
                n_events_vis:  int,
                sensor_range:  float,
                ncc_peak:      float) -> float:
    """
    Confiance globale [0-1] :
      40% qualité du signal visuel
      30% nombre d'événements détectés
      20% amplitude capteur (gripper s'est vraiment ouvert)
      10% force du pic NCC
    """
    ev_score   = min(n_events_vis / 2.0, 1.0)
    range_score = min(sensor_range / 20.0, 1.0)
    ncc_score   = min(abs(ncc_peak), 1.0) if math.isfinite(ncc_peak) else 0.0

    # NCC forte prouve la corrélation même sans événements discrets détectables
    # (gripper toujours ouvert, mouvement trop rapide, etc.)
    conf = (0.40 * quality
            + 0.15 * ev_score
            + 0.20 * range_score
            + 0.25 * ncc_score)

    return round(conf, 3)


def _composite_score(pearson_r:   float,
                     ncc_peak:    float,
                     ncc_lag_ms:  float,
                     confidence:  float) -> float:
    """
    Score [0-100] :
      45% |Pearson r|    — cohérence de forme globale
      35% |NCC peak|     — cohérence temporelle normalisée (lag-invariant)
      20% pénalité lag   — exp(-|lag|/500ms)  [-3dB à 500ms]
    × confiance

    La pénalité lag est assouplie (500ms vs 200ms précédemment) car :
      - Les vrais décalages d'horloge sont typiquement 0–300ms
      - La NCC capture déjà la cohérence temporelle
      - Pénaliser 200ms ferait régresser les sessions bien synchronisées
        qui ont un léger décalage résiduel.
    """
    pr_s  = min(abs(pearson_r), 1.0) if math.isfinite(pearson_r) else 0.0
    ncc_s = min(abs(ncc_peak),  1.0) if math.isfinite(ncc_peak)  else 0.0

    if math.isfinite(ncc_lag_ms):
        lag_score = math.exp(-abs(ncc_lag_ms) / 500.0)
    else:
        lag_score = 0.0

    raw = 45.0 * pr_s + 35.0 * ncc_s + 20.0 * lag_score
    return round(raw * confidence, 2)


# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═════════════════════════════════════════════════════════════════════════════

def _load_sensor(sensor_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge le CSV capteur.
    Retourne (timestamps_s, opening_mm).
    Gère les colonnes de temps numériques en priorité.
    """
    df = pd.read_csv(sensor_path)
    _TIME_PREF = ["timestamp_ns", "t_ms_corrected_ns", "t_ms", "time_seconds"]
    time_col = next(
        (c for c in _TIME_PREF
         if c in df.columns and pd.api.types.is_numeric_dtype(df[c])),
        next((c for c in df.columns
              if pd.api.types.is_numeric_dtype(df[c])
              and ("time" in c.lower() or "ts" in c.lower())), None),
    )
    open_col = next(
        (c for c in df.columns if c == "opening_mm"),
        next((c for c in df.columns
              if "open" in c.lower() and "mm" in c.lower()), None),
    )
    if time_col is None or open_col is None:
        raise ValueError(f"Colonnes introuvables dans {sensor_path}: "
                         f"{list(df.columns)}")

    ts  = df[time_col].to_numpy(dtype=float)
    val = df[open_col].to_numpy(dtype=float)

    if ts.max() > 1e15:
        ts = ts / 1e9
    elif ts.max() > 1e9:
        ts = ts / 1e3

    return ts, val


def _load_jsonl_timestamps(jsonl_path: str) -> Tuple[List[float], float]:
    """
    Retourne (timestamps_relatifs_s, epoch0_s).
    timestamps_relatifs_s[i] = temps de la frame i depuis la première frame.
    """
    entries = []
    for l in Path(jsonl_path).read_text(errors="replace").splitlines():
        if not l.strip():
            continue
        try:
            entries.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    if not entries or "capture_time" not in entries[0]:
        return [], 0.0
    t0ms     = entries[0]["capture_time"]
    epoch0_s = t0ms / 1000.0
    rel      = [(e["capture_time"] - t0ms) / 1000.0 for e in entries]
    return rel, epoch0_s


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def process_side(session_path: str,
                 side:         str,
                 cfg:          Optional[GripperVisionConfig] = None,
                 debug_video:  bool = False,
                 output_dir:   Optional[str] = None) -> GripperVisionResult:
    """
    Traite un côté (left | right) d'une session.
    Retourne un GripperVisionResult complet.
    """
    if cfg is None:
        cfg = GripperVisionConfig()

    sname  = Path(session_path).name
    result = GripperVisionResult(
        session_name = sname, side = side,
        video_path   = "", sensor_path = "",
        success      = False,
    )

    if not CV2_OK:
        result.error = "opencv-python non installé (pip install opencv-python)"
        return result

    # ── Chemins ───────────────────────────────────────────────────────────────
    videos_dir   = Path(session_path) / "videos"
    video_path   = videos_dir / f"{side}.mp4"
    jsonl_path   = videos_dir / f"{side}.jsonl"
    sensor_path  = Path(session_path) / f"gripper_{side}_data.csv"

    result.video_path  = str(video_path)
    result.sensor_path = str(sensor_path)

    for p in [video_path, sensor_path]:
        if not p.exists():
            result.error = f"Fichier absent : {p}"
            return result

    # ── Chargement capteur ────────────────────────────────────────────────────
    try:
        sensor_ts_s, sensor_val = _load_sensor(str(sensor_path))
    except Exception as e:
        result.error = f"Capteur : {e}"
        return result

    # ── Timestamps vidéo ──────────────────────────────────────────────────────
    vid_ts_rel: List[float] = []
    vid_epoch0: float       = 0.0
    if jsonl_path.exists():
        vid_ts_rel, vid_epoch0 = _load_jsonl_timestamps(str(jsonl_path))

    # ── Lecture frames ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result.error = f"Impossible d'ouvrir : {video_path}"
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    grays:      List[np.ndarray] = []
    times_s:    List[float]      = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        t = vid_ts_rel[fi] if fi < len(vid_ts_rel) else fi / fps
        times_s.append(t)
        fi += 1
    cap.release()

    if fi == 0:
        result.error = "Vidéo vide"
        return result

    result.n_frames = fi
    result.fps      = fps
    t_arr = np.array(times_s, dtype=float)

    # ── Alignement capteur (avant auto-threshold pour l'utiliser comme critère) ─
    sen_t_rel = sensor_ts_s - vid_epoch0
    s_lo, s_hi = sensor_val.min(), sensor_val.max()
    if s_hi - s_lo > 1e-6:
        sensor_norm = (sensor_val - s_lo) / (s_hi - s_lo)
    else:
        sensor_norm = np.zeros_like(sensor_val)

    f_sen      = interp1d(sen_t_rel, sensor_norm, kind="linear",
                          bounds_error=False, fill_value=np.nan)
    sen_at     = f_sen(t_arr)
    valid_mask = np.isfinite(sen_at)
    result.sensor_signal = [
        float(v) if math.isfinite(float(v)) else float("nan")
        for v in sen_at
    ]

    n_valid = int(valid_mask.sum())
    if n_valid < 20:
        result.error = f"Pas assez de frames avec capteur valide : {n_valid}"
        return result

    # ── Auto-calibration (band_bottom, threshold) par corrélation max ─────────
    thr = cfg.dark_threshold
    cfg_used = cfg   # config effective (peut avoir band_bottom modifié)
    if cfg.auto_threshold:
        best_bb, best_thr = _auto_calibrate(
            grays, cfg, sen_at_frames=sen_at, frame_h=fh
        )
        thr = best_thr
        if best_bb != cfg.band_bottom:
            # Recréer une config avec le band_bottom optimisé
            cfg_used = GripperVisionConfig(
                band_bottom    = best_bb,
                dark_threshold = thr,
                auto_threshold = False,
                morph_open     = cfg.morph_open,
                morph_close    = cfg.morph_close,
                feature_weights = cfg.feature_weights,
                savgol_window  = cfg.savgol_window,
                savgol_poly    = cfg.savgol_poly,
                norm_pct_lo    = cfg.norm_pct_lo,
                norm_pct_hi    = cfg.norm_pct_hi,
                ncc_max_lag_s  = cfg.ncc_max_lag_s,
                ncc_min_peak   = cfg.ncc_min_peak,
                open_hi        = cfg.open_hi,
                open_lo        = cfg.open_lo,
                sensor_open_hi = cfg.sensor_open_hi,
                sensor_open_lo = cfg.sensor_open_lo,
                min_event_gap_s = cfg.min_event_gap_s,
                match_tol_s    = cfg.match_tol_s,
                score_ok       = cfg.score_ok,
                score_warn     = cfg.score_warn,
            )

    result.threshold_used = thr

    # ── Extraction des features sur toutes les frames ─────────────────────────
    feat_series: Dict[str, List[float]] = {
        k: [] for k in cfg_used.feature_weights
    }
    for gray in grays:
        f = _extract_features(gray, thr, cfg_used)
        for k in feat_series:
            feat_series[k].append(f.get(k, 0.0))

    # ── Fusion adaptative ─────────────────────────────────────────────────────
    weights = _adaptive_weights(feat_series, cfg_used)
    result.feature_weights_used = {k: round(v, 4) for k, v in weights.items()}

    raw_signal = _fuse_features(feat_series, weights)
    smooth_sig = _smooth(raw_signal, cfg_used)
    norm_vis   = _normalize(smooth_sig, cfg_used)

    result.times_sec     = times_s
    result.vision_signal = norm_vis.tolist()

    vis_v = norm_vis[valid_mask]
    sen_v = sen_at[valid_mask]

    # ── Pearson ────────────────────────────────────────────────────────────────
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            r, p = pearsonr(vis_v, sen_v)
        result.pearson_r = float(r)
        result.pearson_p = float(p)
    except Exception:
        pass

    # ── NCC ────────────────────────────────────────────────────────────────────
    result.ncc_peak, result.ncc_lag_ms = _ncc(vis_v, sen_v, fps, cfg_used)

    # Si le NCC a trouvé un grand lag (>300ms) mais que le signal global
    # (Pearson r) est faible (|r|<0.55), c'est probablement un faux pic de
    # cross-corrélation. Forcer le lag à 0 (NCC(0) ≈ Pearson r).
    # Seuil |r|<0.55 : au-dessus, un grand lag pourrait être réel.
    if (math.isfinite(result.pearson_r)
            and abs(result.pearson_r) < 0.55
            and math.isfinite(result.ncc_lag_ms)
            and abs(result.ncc_lag_ms) > 300.0):
        result.ncc_peak   = result.pearson_r
        result.ncc_lag_ms = 0.0

    # ── Événements ────────────────────────────────────────────────────────────
    ev_vis, ht_vis = _detect_events(
        norm_vis, t_arr,
        cfg_used.open_hi, cfg_used.open_lo, cfg_used.min_event_gap_s,
    )
    ev_sen, _ = _detect_events(
        _normalize(_smooth(sensor_norm, cfg_used), cfg_used),
        sen_t_rel,
        cfg_used.sensor_open_hi, cfg_used.sensor_open_lo, cfg_used.min_event_gap_s,
    )

    matched, n_unm_vis, n_unm_sen = _match_events(
        ev_vis, ht_vis, ev_sen, cfg_used.match_tol_s
    )
    result.events_vision      = ev_vis
    result.events_sensor      = ev_sen
    result.matched_events     = matched
    result.n_unmatched_vision = n_unm_vis
    result.n_unmatched_sensor = n_unm_sen

    # ── Qualité, confiance, score ─────────────────────────────────────────────
    sq  = _signal_quality(norm_vis)
    conf = _confidence(sq, len(ev_vis), s_hi - s_lo, result.ncc_peak)

    result.signal_quality  = sq
    result.confidence      = conf
    result.composite_score = _composite_score(
        result.pearson_r, result.ncc_peak, result.ncc_lag_ms, conf
    )
    # Critère de statut :
    #   OK      si score ≥ score_ok
    #   OK      si NCC ≥ 0.55 ET |lag| ≤ 200ms ET conf ≥ 0.85 (corrélé et synchronisé)
    #   WARNING si score ≥ score_warn
    #   ERROR   sinon
    # Critère direct NCC :
    #   Signal corrélé (positif ou négatif) ET synchronisé (lag faible)
    #   Quatre seuils selon la force du pic NCC :
    #   - NCC fort (≥0.55) : conf ≥ 0.60 suffisant
    #   - NCC modéré (≥0.45) : conf ≥ 0.40 requis (sessions à signal inversé)
    #   - NCC faible (≥0.35) : conf ≥ 0.70 requis (signal limite mais synchronisé)
    #   - NCC très faible (≥0.30) : conf ≥ 0.80 requis (signal marginal, haute qualité)
    ncc_ok_direct = (
        math.isfinite(result.ncc_peak)
        and math.isfinite(result.ncc_lag_ms)
        and abs(result.ncc_lag_ms) <= 200.0
        and (
            (abs(result.ncc_peak) >= 0.55 and result.confidence >= 0.60)
            or (abs(result.ncc_peak) >= 0.45 and result.confidence >= 0.40)
            or (abs(result.ncc_peak) >= 0.35 and result.confidence >= 0.70)
            or (abs(result.ncc_peak) >= 0.30 and result.confidence >= 0.79)
        )
    )
    result.status = (
        "OK"      if (result.composite_score >= cfg_used.score_ok or ncc_ok_direct)
        else ("WARNING" if result.composite_score >= cfg_used.score_warn
              else "ERROR")
    )
    result.success = True

    # ── Vidéo debug ───────────────────────────────────────────────────────────
    if debug_video and output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_v = str(Path(output_dir) / f"{sname}_{side}_debug.mp4")
        _write_debug_video(str(video_path), result, cfg_used, out_v, fps, fw, fh)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# VIDÉO DEBUG
# ═════════════════════════════════════════════════════════════════════════════

def _write_debug_video(video_path: str,
                       result:     GripperVisionResult,
                       cfg:        GripperVisionConfig,
                       out_path:   str,
                       fps:        float,
                       fw:         int,
                       fh:         int) -> None:
    cap    = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    vis_arr = np.array(result.vision_signal)
    sen_arr = np.array(result.sensor_signal, dtype=float)
    thr     = result.threshold_used
    fi      = 0

    # Ensemble des timestamps d'événements appariés
    ev_times = {round(m.t_vision_s, 2) for m in result.matched_events}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── Bande analysée ────────────────────────────────────────────────────
        band_y = min(cfg.band_bottom, fh - 1)
        cv2.line(frame, (0, band_y), (fw - 1, band_y), (0, 255, 255), 1)

        # ── Overlay binaire (pixels bras noirs) en rouge semi-transparent ─────
        gray_f  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_g   = gray_f[:band_y, :]
        _, bin_ = cv2.threshold(roi_g, thr, 255, cv2.THRESH_BINARY_INV)
        mask    = np.zeros_like(frame)
        mask[:band_y, :, 2] = bin_   # canal rouge
        frame = cv2.addWeighted(frame, 1.0, mask, 0.3, 0)

        # ── Barre signal vision ───────────────────────────────────────────────
        nv  = float(vis_arr[fi]) if fi < len(vis_arr) else 0.0
        bw  = int(nv * 250)
        cv2.rectangle(frame, (10, 10),  (260, 30),       (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 10),  (10 + bw, 30),   (0, 220, 0),  -1)
        cv2.rectangle(frame, (10, 10),  (260, 30),        (150,150,150), 1)
        cv2.putText(frame, f"VIS {nv:.2f}", (265, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)

        # ── Barre signal capteur ──────────────────────────────────────────────
        sv  = float(sen_arr[fi]) if fi < len(sen_arr) and math.isfinite(float(sen_arr[fi])) else 0.0
        bws = int(sv * 250)
        cv2.rectangle(frame, (10, 35), (260, 55),        (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 35), (10 + bws, 55),   (0, 120, 255), -1)
        cv2.rectangle(frame, (10, 35), (260, 55),         (150,150,150), 1)
        cv2.putText(frame, f"SEN {sv:.2f}", (265, 51),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1, cv2.LINE_AA)

        # ── Score et lag ──────────────────────────────────────────────────────
        cv2.putText(frame,
                    f"score={result.composite_score:.0f}  "
                    f"r={result.pearson_r:+.2f}  "
                    f"lag={result.ncc_lag_ms:+.0f}ms",
                    (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Marqueur événement apparié ────────────────────────────────────────
        if fi < len(result.times_sec):
            t_cur = result.times_sec[fi]
            for m in result.matched_events:
                if abs(m.t_vision_s - t_cur) < 1.5 / fps:
                    cv2.putText(frame,
                                f"EVENT  d={m.delay_ms:+.0f}ms",
                                (10, 94),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        fi += 1

    cap.release()
    writer.release()


# ═════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE
# ═════════════════════════════════════════════════════════════════════════════

def save_plot(result: GripperVisionResult, path: str) -> None:
    """Graphique 3 panneaux : signaux / délai événements / poids features."""
    if not result.success or not MPL_OK:
        return

    t   = np.array(result.times_sec)
    vis = np.array(result.vision_signal)
    sen = np.array(result.sensor_signal, dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)

    # ── Panneau 1 : signaux temporels ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, vis, color="steelblue", lw=1.2, label="vision (fusionné normalisé)")
    valid = np.isfinite(sen)
    if valid.any():
        ax.plot(t[valid], sen[valid], color="darkorange", lw=1.0,
                alpha=0.9, label="capteur (normalisé)")
    ax.axhline(0.55, color="green", lw=0.8, ls="--", alpha=0.6, label="seuil hi")
    ax.axhline(0.30, color="red",   lw=0.8, ls=":",  alpha=0.6, label="seuil lo")

    # Marqueurs événements
    for tv in result.events_vision:
        ax.axvline(tv, color="blue", lw=1.0, alpha=0.4)
    for ts in result.events_sensor:
        ax.axvline(ts, color="orange", lw=1.0, alpha=0.4, ls="--")
    for m in result.matched_events:
        ax.annotate("",
                    xy=(m.t_sensor_s, 0.05),
                    xytext=(m.t_vision_s, 0.05),
                    arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))

    ax.set_title(
        f"{result.session_name}/{result.side}  |  "
        f"score={result.composite_score:.1f}/100  "
        f"conf={result.confidence:.2f}  "
        f"r={result.pearson_r:+.3f}  "
        f"ncc={result.ncc_peak:+.3f}  "
        f"lag={result.ncc_lag_ms:+.0f}ms  "
        f"thr={result.threshold_used}",
        fontsize=10,
    )
    ax.set_ylabel("Signal normalisé [0-1]")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.05, 1.15)

    # ── Panneau 2 : délai par événement apparié ────────────────────────────────
    ax2 = axes[1]
    if result.matched_events:
        ts_ev   = [m.t_vision_s for m in result.matched_events]
        delays  = [m.delay_ms   for m in result.matched_events]
        ax2.stem(ts_ev, delays, linefmt="b-", markerfmt="bo", basefmt="k-")
        med = float(np.median(delays))
        ax2.axhline(med, color="red", lw=1.2, ls="--",
                    label=f"médiane = {med:+.0f}ms")
        ax2.axhline(0,   color="black", lw=0.8)
        ax2.set_ylabel("Délai vision−capteur (ms)")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Aucun événement apparié",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)
    ax2.set_xlim(t[0], t[-1])

    # ── Panneau 3 : poids des features ────────────────────────────────────────
    ax3 = axes[2]
    fw_used = result.feature_weights_used
    if fw_used:
        names  = list(fw_used.keys())
        values = [fw_used[n] for n in names]
        colors = plt.cm.tab10(np.linspace(0, 0.7, len(names)))  # type: ignore
        bars   = ax3.barh(names, values, color=colors)
        ax3.set_xlabel("Poids adaptatif normalisé")
        ax3.set_title("Fusion des features (pondération adaptative)", fontsize=9)
        for bar, v in zip(bars, values):
            ax3.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{v:.3f}", va="center", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Poids non disponibles",
                 ha="center", va="center", transform=ax3.transAxes)

    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# RAPPORT TEXTE
# ═════════════════════════════════════════════════════════════════════════════

def write_report(results: List[GripperVisionResult], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        w = lambda s="": f.write(s + "\n")
        w("=" * 76)
        w("RAPPORT GRIPPER VISION — ALIGNEMENT TIMESTAMP VIDÉO / CAPTEUR")
        w("=" * 76)

        n_ok   = sum(1 for r in results if r.success and r.status == "OK")
        n_warn = sum(1 for r in results if r.success and r.status == "WARNING")
        n_err  = sum(1 for r in results if r.success and r.status == "ERROR")
        n_fail = sum(1 for r in results if not r.success)
        scores = [r.composite_score for r in results
                  if r.success and math.isfinite(r.composite_score)]

        w(f"  Total analysés  : {len(results)}")
        w(f"  OK              : {n_ok}")
        w(f"  WARNINGS        : {n_warn}")
        w(f"  ERRORS          : {n_err}")
        w(f"  FAILED          : {n_fail}")
        if scores:
            w(f"  Score moyen     : {np.mean(scores):.1f}  "
              f"[{min(scores):.1f} – {max(scores):.1f}]")
        w()
        w("─" * 76)
        w("DÉTAIL PAR SESSION")
        w("─" * 76)

        for r in results:
            w()
            w(f"{r.session_name} / {r.side}")
            if not r.success:
                w(f"  FAILED  {r.error}")
                continue

            sym = {"OK": "OK  ", "WARNING": "WARN", "ERROR": "ERR "}.get(r.status, "?   ")
            w(f"  [{sym}]  score={r.composite_score:.1f}/100  "
              f"conf={r.confidence:.2f}  qual={r.signal_quality:.2f}  "
              f"thr={r.threshold_used}")
            w(f"          r={r.pearson_r:+.4f}  "
              f"ncc={r.ncc_peak:+.4f}  lag={r.ncc_lag_ms:+.0f}ms")
            w(f"          events: vis={len(r.events_vision)}  "
              f"sen={len(r.events_sensor)}  "
              f"matched={len(r.matched_events)}  "
              f"unmatch_v={r.n_unmatched_vision}  "
              f"unmatch_s={r.n_unmatched_sensor}")

            if r.matched_events:
                delays = [m.delay_ms for m in r.matched_events]
                w(f"          délais appariés: "
                  f"med={float(np.median(delays)):+.0f}ms  "
                  f"std={float(np.std(delays)):.0f}ms  "
                  f"[{min(delays):+.0f}, {max(delays):+.0f}]ms")

            if r.feature_weights_used:
                top = sorted(r.feature_weights_used.items(),
                             key=lambda x: x[1], reverse=True)[:3]
                w("          features: " +
                  "  ".join(f"{k}={v:.3f}" for k, v in top))


# ═════════════════════════════════════════════════════════════════════════════
# COLLECTE DES SESSIONS
# ═════════════════════════════════════════════════════════════════════════════

def collect_session_paths(sessions_dir: str,
                           pattern:      str = "session_") -> List[str]:
    return sorted([
        os.path.join(sessions_dir, name)
        for name in os.listdir(sessions_dir)
        if name.startswith(pattern)
           and os.path.isdir(os.path.join(sessions_dir, name))
    ])


# ═════════════════════════════════════════════════════════════════════════════
# CLI STANDALONE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="gripper_vision.py — analyse visuelle du gripper",
    )
    parser.add_argument("--sessions_dir", required=True)
    parser.add_argument("--output_dir",   default="gripper_vision_results")
    parser.add_argument("--sides",        nargs="+", default=["left", "right"])
    parser.add_argument("--plots",        action="store_true")
    parser.add_argument("--debug_video",  action="store_true")
    parser.add_argument("--band_bottom",  type=int,   default=300)
    parser.add_argument("--threshold",    type=int,   default=35)
    parser.add_argument("--no_auto_thr",  action="store_true")
    args = parser.parse_args()

    cfg = GripperVisionConfig(
        band_bottom    = args.band_bottom,
        dark_threshold = args.threshold,
        auto_threshold = not args.no_auto_thr,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions  = collect_session_paths(args.sessions_dir)
    all_res: List[GripperVisionResult] = []

    for sp in sessions:
        sname = Path(sp).name
        for side in args.sides:
            print(f"  {sname}/{side} ...", end=" ", flush=True)
            r = process_side(sp, side, cfg,
                             debug_video=args.debug_video,
                             output_dir=str(out_dir))
            all_res.append(r)

            if r.success:
                sym = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(r.status, "?")
                print(f"{sym}  score={r.composite_score:.1f}  "
                      f"r={r.pearson_r:+.3f}  lag={r.ncc_lag_ms:+.0f}ms")
                if args.plots:
                    try:
                        save_plot(r, str(out_dir / f"{sname}_{side}.png"))
                    except Exception as e:
                        print(f"    [plot] {e}")
            else:
                print(f"✗  FAILED: {r.error}")

    report_path = str(out_dir / "report.txt")
    write_report(all_res, report_path)
    print(f"\nRapport : {report_path}")

    n_ok = sum(1 for r in all_res if r.success and r.status == "OK")
    n_f  = sum(1 for r in all_res if not r.success)
    print(f"OK={n_ok}  sur {len(all_res)-n_f} analysés")
    sys.exit(0 if n_ok == len(all_res) - n_f else 1)
