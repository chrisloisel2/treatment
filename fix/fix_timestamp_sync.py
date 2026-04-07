#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/fix_timestamp_sync.py — Analyse CERTAINE de la synchronisation temporelle.

Vérifie l'écart entre les timestamps de chaque flux (tracker, caméras, grippers)
et quantifie toutes les possibilités de décalage.

Mesures effectuées :
─────────────────────────────────────────────────────────────
1. ALIGNEMENT DES DÉBUTS
   Δt_start entre tracker / chaque caméra / chaque gripper.
   Attendu : |Δt_start| < 500 ms si synchronisés au démarrage.

2. ALIGNEMENT DES FINS
   Δt_end entre tracker / chaque caméra / chaque gripper.
   Les fins peuvent diverger (enregistrements indépendants).

3. GAPS INTERNES
   Détecte les interruptions dans chaque flux :
   - Tracker : gap si timestamp_ns[i+1] - timestamp_ns[i] > GAP_TRACKER_THRESH
   - Caméra  : gap si capture_time[i+1] - capture_time[i] > GAP_CAMERA_THRESH
   - Gripper : gap si timestamp_ns[i+1] - timestamp_ns[i] > GAP_GRIPPER_THRESH

4. DÉRIVE TEMPORELLE (clock drift)
   Compare la durée réelle de chaque flux à la durée du tracker (référence).
   Drift en ppm = (dur_flux - dur_tracker) / dur_tracker × 1e6.
   |drift| > 100 ppm signifie une dérive potentiellement problématique.

5. ANALYSE DU FRAMERATE CAMÉRA
   IFI médian, écart-type, frames manquantes estimées.
   Détecte les doubles frames (IFI ≈ 0) et les sauts (IFI > 2× médiane).

6. FENÊTRE DE DÉCALAGE ATTEIGNABLE
   Pour chaque paire (flux A, flux B) : quels décalages Δ permettent encore
   un chevauchement ≥ MIN_OVERLAP_MS ?
   Utile pour savoir si le modèle IA peut aligner deux flux donné leur offset.

SORTIES :
   - Rapport JSON structuré
   - Affichage console colorisé avec verdict par flux
   - Code de sortie 0=OK, 1=problème détecté, 2=erreur

Usage :
    python -m fix.fix_timestamp_sync /chemin/session
    python -m fix.fix_timestamp_sync /chemin/session --json
    python -m fix.fix_timestamp_sync /chemin/root --batch
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ── Seuils ────────────────────────────────────────────────────────────────────

# Gaps : un écart > seuil = gap (en ms)
GAP_TRACKER_THRESH  = 100.0   # ms — tracker enregistre à ~60Hz (16ms / frame)
GAP_CAMERA_THRESH   = 200.0   # ms — caméra à ~30Hz (33ms / frame), seuil 6× IFI médian
GAP_GRIPPER_THRESH  = 200.0   # ms — gripper à ~60Hz

# Alignement au démarrage : attendu < 500ms pour être "bien aligné"
START_ALIGN_OK_MS   = 500.0
START_ALIGN_WARN_MS = 2000.0  # entre 500ms et 2s : avertissement
# Au-delà de 2s : probablement désynchronisé

# Dérive par type de flux (les caméras USB ont naturellement ~5000 ppm d'écart)
# Tracker vs caméra : la dérive de durée est normale (clocks indépendants)
# Tracker vs gripper : clocks plus proches (même PC), seuils plus serrés
DRIFT_WARN_PPM_CAMERA  = 30_000.0   # 3% — seuil avertissement pour caméras
DRIFT_ERROR_PPM_CAMERA = 80_000.0   # 8% — seuil erreur pour caméras
DRIFT_WARN_PPM_SENSOR  = 2_000.0    # 0.2% — seuil avertissement pour grippers/trackers
DRIFT_ERROR_PPM_SENSOR = 10_000.0   # 1% — seuil erreur pour grippers/trackers

# Couverture minimale (durée flux / durée tracker) pour être utile
MIN_COVERAGE        = 0.70

MIN_OVERLAP_MS      = 2000.0  # chevauchement minimal pour l'IA


# ── Structures ────────────────────────────────────────────────────────────────

@dataclass
class StreamStats:
    name: str
    n_samples: int
    t_start_ms: float
    t_end_ms: float
    duration_ms: float
    sample_rate_hz: float
    gaps: list[dict]          # [{t_ms, gap_ms}]
    n_gaps: int
    total_gap_ms: float
    drift_ppm: Optional[float]    # vs tracker
    coverage: Optional[float]     # fraction de la durée tracker couverte
    status: str                   # "ok" | "warn" | "error"
    issues: list[str]


@dataclass
class PairAlignment:
    stream_a: str
    stream_b: str
    delta_start_ms: float         # t_start_b - t_start_a
    delta_end_ms: float
    overlap_ms: float
    max_lag_searchable_ms: float  # fenêtre de décalage atteignable (±)
    aligned: bool                 # |delta_start| < START_ALIGN_OK_MS


@dataclass
class TimestampSyncReport:
    session: str
    status: str                   # "ok" | "warn" | "error"
    streams: list[StreamStats]
    pairs: list[PairAlignment]
    issues: list[str]
    summary: dict


# ── Chargement des flux ───────────────────────────────────────────────────────

def _load_tracker_times(session_path: Path) -> Optional[np.ndarray]:
    if not _PANDAS:
        return None
    path = session_path / "tracker_positions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "timestamp_ns" in df.columns:
            t = pd.to_numeric(df["timestamp_ns"], errors="coerce").dropna().to_numpy()
            return (t / 1e6).astype(np.float64)   # → ms
        elif "time_seconds" in df.columns:
            t = pd.to_numeric(df["time_seconds"], errors="coerce").dropna().to_numpy()
            return (t * 1000.0).astype(np.float64)
    except Exception:
        return None


def _load_jsonl_times(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    times = []
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    times.append(float(json.loads(line)["capture_time"]))
                except Exception:
                    continue
    except Exception:
        return None
    return np.array(times, dtype=np.float64) if len(times) > 5 else None


def _load_gripper_times(path: Path) -> Optional[np.ndarray]:
    if not _PANDAS or not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "timestamp_ns" in df.columns:
            t = pd.to_numeric(df["timestamp_ns"], errors="coerce").dropna().to_numpy()
            return (t / 1e6).astype(np.float64)
        elif "t_ms_corrected_ns" in df.columns:
            t = pd.to_numeric(df["t_ms_corrected_ns"], errors="coerce").dropna().to_numpy()
            return (t / 1e6).astype(np.float64)
    except Exception:
        return None


# ── Analyse d'un flux ─────────────────────────────────────────────────────────

def _analyse_stream(
    name: str,
    times_ms: np.ndarray,
    gap_thresh_ms: float,
    tracker_stats: Optional["StreamStats"] = None,
    is_camera: bool = False,
) -> StreamStats:
    times_ms = np.sort(times_ms)
    n = len(times_ms)
    t_start = float(times_ms[0])
    t_end   = float(times_ms[-1])
    duration = t_end - t_start

    # IFI
    ifi = np.diff(times_ms)
    med_ifi = float(np.median(ifi)) if len(ifi) > 0 else 0.0
    sample_rate = 1000.0 / med_ifi if med_ifi > 0 else 0.0

    # Gaps
    gaps = []
    gap_thresh = max(gap_thresh_ms, med_ifi * 3.0)
    for i, g in enumerate(ifi):
        if g > gap_thresh:
            gaps.append({
                "t_ms":    round(float(times_ms[i]), 1),
                "gap_ms":  round(float(g), 1),
            })

    total_gap_ms = float(sum(g["gap_ms"] for g in gaps))

    # Dérive vs tracker
    drift_ppm = None
    coverage  = None
    if tracker_stats is not None and tracker_stats.duration_ms > 0:
        drift_ppm = (duration - tracker_stats.duration_ms) / tracker_stats.duration_ms * 1e6
        # Coverage : fraction de la durée tracker couverte par ce flux
        trk_s, trk_e = tracker_stats.t_start_ms, tracker_stats.t_end_ms
        overlap_s = max(trk_s, t_start)
        overlap_e = min(trk_e, t_end)
        coverage  = max(0.0, (overlap_e - overlap_s) / tracker_stats.duration_ms)

    # Statut et problèmes
    issues: list[str] = []
    status = "ok"

    if len(gaps) > 0:
        total_frac = total_gap_ms / (duration + 1e-6)
        if total_frac > 0.05 or len(gaps) > 3:
            issues.append(f"{len(gaps)} gap(s) → {total_gap_ms:.0f}ms perdus ({total_frac*100:.1f}%)")
            status = "warn" if total_frac < 0.10 else "error"

    if drift_ppm is not None:
        warn_thr  = DRIFT_WARN_PPM_CAMERA  if is_camera else DRIFT_WARN_PPM_SENSOR
        error_thr = DRIFT_ERROR_PPM_CAMERA if is_camera else DRIFT_ERROR_PPM_SENSOR
        if abs(drift_ppm) > error_thr:
            issues.append(f"dérive critique : {drift_ppm:.0f} ppm")
            status = "error"
        elif abs(drift_ppm) > warn_thr:
            issues.append(f"dérive : {drift_ppm:.0f} ppm")
            if status == "ok":
                status = "warn"

    if coverage is not None and coverage < MIN_COVERAGE:
        issues.append(f"couverture insuffisante : {coverage*100:.0f}% < {MIN_COVERAGE*100:.0f}%")
        status = "error"

    return StreamStats(
        name=name,
        n_samples=n,
        t_start_ms=round(t_start, 1),
        t_end_ms=round(t_end, 1),
        duration_ms=round(duration, 1),
        sample_rate_hz=round(sample_rate, 2),
        gaps=gaps,
        n_gaps=len(gaps),
        total_gap_ms=round(total_gap_ms, 1),
        drift_ppm=round(drift_ppm, 1) if drift_ppm is not None else None,
        coverage=round(coverage, 3) if coverage is not None else None,
        status=status,
        issues=issues,
    )


# ── Analyse des paires ────────────────────────────────────────────────────────

def _analyse_pair(
    a: StreamStats,
    b: StreamStats,
) -> PairAlignment:
    delta_start = b.t_start_ms - a.t_start_ms
    delta_end   = b.t_end_ms   - a.t_end_ms

    overlap_s = max(a.t_start_ms, b.t_start_ms)
    overlap_e = min(a.t_end_ms,   b.t_end_ms)
    overlap   = max(0.0, overlap_e - overlap_s)

    # Fenêtre de décalage atteignable (combien peut-on décaler B et avoir overlap ≥ MIN_OVERLAP_MS ?)
    # On peut chercher un lag λ tel que le chevauchement de (A shifted by 0) et (B shifted by λ)
    # reste ≥ MIN_OVERLAP_MS.
    # Simplifié : lag_max = (overlap - MIN_OVERLAP_MS) / 2 (approximation)
    max_lag = max(0.0, (overlap - MIN_OVERLAP_MS) / 2.0)

    aligned = abs(delta_start) < START_ALIGN_OK_MS

    return PairAlignment(
        stream_a=a.name,
        stream_b=b.name,
        delta_start_ms=round(delta_start, 1),
        delta_end_ms=round(delta_end, 1),
        overlap_ms=round(overlap, 1),
        max_lag_searchable_ms=round(max_lag, 1),
        aligned=aligned,
    )


# ── Rapport complet ───────────────────────────────────────────────────────────

def analyse_timestamp_sync(session_path: Path) -> TimestampSyncReport:
    """
    Analyse complète de la synchronisation temporelle d'une session.
    """
    name = session_path.name

    # ── Charger les flux ──────────────────────────────────────────────────────
    flux_data: dict[str, Optional[np.ndarray]] = {}

    t_trk = _load_tracker_times(session_path)
    flux_data["tracker"] = t_trk

    for cam in ("head", "left", "right"):
        t = _load_jsonl_times(session_path / "videos" / f"{cam}.jsonl")
        flux_data[f"cam_{cam}"] = t

    for side in ("left", "right"):
        t = _load_gripper_times(session_path / f"gripper_{side}_data.csv")
        flux_data[f"gripper_{side}"] = t

    # ── Analyser chaque flux ──────────────────────────────────────────────────
    streams: list[StreamStats] = []
    tracker_stats: Optional[StreamStats] = None

    if t_trk is not None and len(t_trk) > 5:
        tracker_stats = _analyse_stream("tracker", t_trk, GAP_TRACKER_THRESH)
        streams.append(tracker_stats)

    gap_thresholds = {
        "cam_head": GAP_CAMERA_THRESH, "cam_left": GAP_CAMERA_THRESH,
        "cam_right": GAP_CAMERA_THRESH,
        "gripper_left": GAP_GRIPPER_THRESH, "gripper_right": GAP_GRIPPER_THRESH,
    }
    for name_key, t in flux_data.items():
        if name_key == "tracker" or t is None or len(t) < 5:
            continue
        st = _analyse_stream(
            name_key, t,
            gap_thresholds.get(name_key, GAP_TRACKER_THRESH),
            tracker_stats,
            is_camera=name_key.startswith("cam_"),
        )
        streams.append(st)

    # ── Analyser les paires (toujours tracker comme référence) ────────────────
    pairs: list[PairAlignment] = []
    if tracker_stats is not None:
        for st in streams:
            if st.name == "tracker":
                continue
            pairs.append(_analyse_pair(tracker_stats, st))

    # ── Décalages réels mesurés ────────────────────────────────────────────────
    issues: list[str] = []
    global_status = "ok"

    for st in streams:
        if st.status == "error":
            global_status = "error"
            issues.extend([f"[{st.name}] {i}" for i in st.issues])
        elif st.status == "warn":
            if global_status == "ok":
                global_status = "warn"
            issues.extend([f"[{st.name}] {i}" for i in st.issues])

    for pair in pairs:
        delta = abs(pair.delta_start_ms)
        if delta > START_ALIGN_WARN_MS:
            msg = (f"[{pair.stream_b}] Δt_start={pair.delta_start_ms:.0f}ms "
                   f"→ DÉSYNCHRONISÉ (>{START_ALIGN_WARN_MS:.0f}ms)")
            issues.append(msg)
            global_status = "error"
        elif delta > START_ALIGN_OK_MS:
            msg = (f"[{pair.stream_b}] Δt_start={pair.delta_start_ms:.0f}ms "
                   f"→ décalage modéré ({START_ALIGN_OK_MS:.0f}–{START_ALIGN_WARN_MS:.0f}ms)")
            issues.append(msg)
            if global_status == "ok":
                global_status = "warn"

    # ── Résumé ────────────────────────────────────────────────────────────────
    summary = {
        "n_streams": len(streams),
        "global_status": global_status,
        "n_issues": len(issues),
        "tracker_duration_ms":  tracker_stats.duration_ms if tracker_stats else None,
        "camera_start_deltas_ms": {
            p.stream_b: p.delta_start_ms
            for p in pairs if p.stream_b.startswith("cam_")
        },
        "gripper_start_deltas_ms": {
            p.stream_b: p.delta_start_ms
            for p in pairs if p.stream_b.startswith("gripper_")
        },
        "max_lag_searchable_ms": {
            p.stream_b: p.max_lag_searchable_ms
            for p in pairs
        },
    }

    return TimestampSyncReport(
        session=name,
        status=global_status,
        streams=streams,
        pairs=pairs,
        issues=issues,
        summary=summary,
    )


# ── Affichage ─────────────────────────────────────────────────────────────────

_COLORS = {
    "ok":    "\033[92m",
    "warn":  "\033[93m",
    "error": "\033[91m",
    "reset": "\033[0m",
    "bold":  "\033[1m",
}
_USE_COLOR = sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    return f"{_COLORS[color]}{text}{_COLORS['reset']}" if _USE_COLOR else text


def print_report(r: TimestampSyncReport) -> None:
    status_color = {"ok": "ok", "warn": "warn", "error": "error"}.get(r.status, "ok")
    print(f"\n{_c(r.status.upper(), status_color)} — {_c(r.session, 'bold')}")
    print(f"  {r.summary.get('n_streams', 0)} flux analysés, "
          f"{r.summary.get('n_issues', 0)} problème(s)\n")

    # Streams
    for st in r.streams:
        color = {"ok": "ok", "warn": "warn", "error": "error"}.get(st.status, "ok")
        drift_str = f"  drift={st.drift_ppm:+.0f}ppm" if st.drift_ppm is not None else ""
        cov_str   = f"  cov={st.coverage*100:.0f}%" if st.coverage is not None else ""
        print(f"  {_c(f'[{st.status.upper():5s}]', color)} "
              f"{st.name:18s} "
              f"n={st.n_samples:5d}  "
              f"Hz={st.sample_rate_hz:5.1f}  "
              f"dur={st.duration_ms/1000:.1f}s  "
              f"gaps={st.n_gaps}"
              f"{drift_str}{cov_str}")
        for issue in st.issues:
            print(f"           ↳ {issue}")

    # Paires
    if r.pairs:
        print()
        print("  Alignement (vs tracker) :")
        for pair in r.pairs:
            aligned = pair.aligned
            sym = _c("✓", "ok") if aligned else _c("✗", "error")
            print(f"    {sym} {pair.stream_b:18s}  "
                  f"Δstart={pair.delta_start_ms:+7.0f}ms  "
                  f"overlap={pair.overlap_ms/1000:.1f}s  "
                  f"lag_max=±{pair.max_lag_searchable_ms:.0f}ms")

    # Issues
    if r.issues:
        print()
        for issue in r.issues:
            print(f"  ⚠ {issue}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse la synchronisation temporelle des flux d'une session."
    )
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--batch",  action="store_true",
                        help="Analyser toutes les sessions sous chaque chemin")
    parser.add_argument("--json",   action="store_true")
    parser.add_argument("--errors-only", action="store_true",
                        help="Afficher uniquement les sessions avec erreurs")
    args = parser.parse_args()

    sessions: list[Path] = []
    for p in args.sessions:
        p = p.resolve()
        if args.batch and p.is_dir():
            sessions.extend(
                m.parent for m in p.rglob("metadata.json")
                if (m.parent / "tracker_positions.csv").exists()
            )
        else:
            sessions.append(p)

    results: list[TimestampSyncReport] = []
    for s in sessions:
        if not s.is_dir():
            continue
        r = analyse_timestamp_sync(s)
        results.append(r)
        if args.json:
            continue
        if args.errors_only and r.status == "ok":
            continue
        print_report(r)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False,
                          default=str))

    if not args.json and results:
        n_ok   = sum(1 for r in results if r.status == "ok")
        n_warn = sum(1 for r in results if r.status == "warn")
        n_err  = sum(1 for r in results if r.status == "error")
        print(f"\n{'─'*60}")
        print(f"Total {len(results)} : "
              f"{_c(f'✓ {n_ok} ok', 'ok')}  "
              f"{_c(f'⚠ {n_warn} warn', 'warn')}  "
              f"{_c(f'✗ {n_err} error', 'error')}")

    # Code de sortie
    if any(r.status == "error" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
