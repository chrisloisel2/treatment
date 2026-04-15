#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

TRACKERS = ("head", "left", "right")
AXES = ("x", "y", "z")


@dataclass
class ReferenceModel:
    head_left_median: np.ndarray
    head_left_mad: np.ndarray
    head_right_median: np.ndarray
    head_right_mad: np.ndarray
    left_right_median: float
    left_right_mad: float
    head_speed_p95: float
    left_speed_p95: float
    right_speed_p95: float

    def to_json(self) -> Dict[str, object]:
        return {
            "head_left_median": self.head_left_median.tolist(),
            "head_left_mad": self.head_left_mad.tolist(),
            "head_right_median": self.head_right_median.tolist(),
            "head_right_mad": self.head_right_mad.tolist(),
            "left_right_median": float(self.left_right_median),
            "left_right_mad": float(self.left_right_mad),
            "head_speed_p95": float(self.head_speed_p95),
            "left_speed_p95": float(self.left_speed_p95),
            "right_speed_p95": float(self.right_speed_p95),
        }

    @staticmethod
    def from_json(payload: Dict[str, object]) -> "ReferenceModel":
        return ReferenceModel(
            head_left_median=np.asarray(payload["head_left_median"], dtype=float),
            head_left_mad=np.asarray(payload["head_left_mad"], dtype=float),
            head_right_median=np.asarray(payload["head_right_median"], dtype=float),
            head_right_mad=np.asarray(payload["head_right_mad"], dtype=float),
            left_right_median=float(payload["left_right_median"]),
            left_right_mad=float(payload["left_right_mad"]),
            head_speed_p95=float(payload["head_speed_p95"]),
            left_speed_p95=float(payload["left_speed_p95"]),
            right_speed_p95=float(payload["right_speed_p95"]),
        )


def tracker_columns(prefix: str) -> List[str]:
    return [f"tracker_{prefix}_{axis}" for axis in AXES]


def validate_columns(df: pd.DataFrame) -> None:
    needed = ["time_seconds"]
    for t in TRACKERS:
        needed.extend(tracker_columns(t))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df)
    return df.copy()


def positions(df: pd.DataFrame, tracker: str) -> np.ndarray:
    return df[tracker_columns(tracker)].to_numpy(dtype=float)


def safe_dt(df: pd.DataFrame) -> np.ndarray:
    t = df["time_seconds"].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = np.median(dt[1:])
    dt = np.where(dt <= 1e-9, np.nan, dt)
    return dt


def speeds(df: pd.DataFrame, tracker: str) -> np.ndarray:
    pos = positions(df, tracker)
    delta = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[[0]]), axis=1)
    dt = safe_dt(df)
    out = np.divide(delta, dt, out=np.zeros_like(delta), where=np.isfinite(dt))
    return out


def robust_median_mad(values: np.ndarray, axis=None) -> Tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=axis)
    mad = np.nanmedian(np.abs(values - median), axis=axis)
    mad = np.maximum(mad, 1e-4)
    return median, mad


def build_reference(csv_paths: Iterable[str | Path], stable_speed_quantile: float = 0.6) -> ReferenceModel:
    head_left_rows = []
    head_right_rows = []
    left_right_rows = []
    speed_rows = {t: [] for t in TRACKERS}

    for path in csv_paths:
        df = load_csv(path)
        pos_h = positions(df, "head")
        pos_l = positions(df, "left")
        pos_r = positions(df, "right")

        sp_h = speeds(df, "head")
        sp_l = speeds(df, "left")
        sp_r = speeds(df, "right")
        total_speed = sp_h + sp_l + sp_r
        threshold = np.nanquantile(total_speed, stable_speed_quantile)
        stable = total_speed <= threshold
        if stable.sum() < max(20, len(df) // 20):
            stable = np.ones(len(df), dtype=bool)

        head_left_rows.append((pos_l - pos_h)[stable])
        head_right_rows.append((pos_r - pos_h)[stable])
        left_right_rows.append(np.linalg.norm(pos_r - pos_l, axis=1)[stable])

        speed_rows["head"].append(sp_h)
        speed_rows["left"].append(sp_l)
        speed_rows["right"].append(sp_r)

    hl = np.vstack(head_left_rows)
    hr = np.vstack(head_right_rows)
    lr = np.concatenate(left_right_rows)

    hl_med, hl_mad = robust_median_mad(hl, axis=0)
    hr_med, hr_mad = robust_median_mad(hr, axis=0)
    lr_med, lr_mad = robust_median_mad(lr, axis=0)

    p95 = {}
    for t in TRACKERS:
        s = np.concatenate(speed_rows[t])
        p95[t] = float(np.nanquantile(s, 0.95))

    return ReferenceModel(
        head_left_median=hl_med,
        head_left_mad=hl_mad,
        head_right_median=hr_med,
        head_right_mad=hr_mad,
        left_right_median=float(lr_med),
        left_right_mad=float(lr_mad),
        head_speed_p95=p95["head"],
        left_speed_p95=p95["left"],
        right_speed_p95=p95["right"],
    )


def detect_bad_frames(
    df: pd.DataFrame,
    ref: ReferenceModel,
    mad_multiplier: float = 8.0,
    speed_multiplier: float = 3.0,
) -> pd.DataFrame:
    pos_h = positions(df, "head")
    pos_l = positions(df, "left")
    pos_r = positions(df, "right")

    vec_hl = pos_l - pos_h
    vec_hr = pos_r - pos_h
    dist_lr = np.linalg.norm(pos_r - pos_l, axis=1)

    hl_err = np.abs(vec_hl - ref.head_left_median) / ref.head_left_mad
    hr_err = np.abs(vec_hr - ref.head_right_median) / ref.head_right_mad
    lr_err = np.abs(dist_lr - ref.left_right_median) / ref.left_right_mad

    sp_h = speeds(df, "head")
    sp_l = speeds(df, "left")
    sp_r = speeds(df, "right")

    bad_head = np.any(hl_err > mad_multiplier, axis=1) & (sp_h > ref.head_speed_p95 * speed_multiplier)
    bad_left = np.any(hl_err > mad_multiplier, axis=1) & (sp_l > ref.left_speed_p95 * speed_multiplier)
    bad_right = np.any(hr_err > mad_multiplier, axis=1) & (sp_r > ref.right_speed_p95 * speed_multiplier)
    bad_pair = lr_err > mad_multiplier

    bad_any = bad_head | bad_left | bad_right | bad_pair

    out = df[["time_seconds"]].copy()
    out["head_speed"] = sp_h
    out["left_speed"] = sp_l
    out["right_speed"] = sp_r
    out["hl_err_max"] = np.max(hl_err, axis=1)
    out["hr_err_max"] = np.max(hr_err, axis=1)
    out["lr_err"] = lr_err
    out["bad_head"] = bad_head
    out["bad_left"] = bad_left
    out["bad_right"] = bad_right
    out["bad_pair"] = bad_pair
    out["bad_any"] = bad_any
    return out


def bad_intervals(flag_df: pd.DataFrame, min_duration_s: float = 0.05) -> List[Dict[str, float | str]]:
    t = flag_df["time_seconds"].to_numpy(dtype=float)
    bad = flag_df["bad_any"].to_numpy(dtype=bool)
    if len(t) == 0:
        return []

    intervals: List[Dict[str, float | str]] = []
    start = None
    for i, is_bad in enumerate(bad):
        if is_bad and start is None:
            start = i
        elif not is_bad and start is not None:
            end = i - 1
            duration = float(t[end] - t[start])
            if duration >= min_duration_s:
                reasons = []
                chunk = flag_df.iloc[start : end + 1]
                for key, label in [
                    ("bad_head", "head"),
                    ("bad_left", "left"),
                    ("bad_right", "right"),
                    ("bad_pair", "pair_distance"),
                ]:
                    if bool(chunk[key].any()):
                        reasons.append(label)
                intervals.append(
                    {
                        "start_s": float(t[start]),
                        "end_s": float(t[end]),
                        "duration_s": duration,
                        "reasons": ",".join(reasons),
                    }
                )
            start = None

    if start is not None:
        end = len(t) - 1
        duration = float(t[end] - t[start])
        if duration >= min_duration_s:
            reasons = []
            chunk = flag_df.iloc[start : end + 1]
            for key, label in [
                ("bad_head", "head"),
                ("bad_left", "left"),
                ("bad_right", "right"),
                ("bad_pair", "pair_distance"),
            ]:
                if bool(chunk[key].any()):
                    reasons.append(label)
            intervals.append(
                {
                    "start_s": float(t[start]),
                    "end_s": float(t[end]),
                    "duration_s": duration,
                    "reasons": ",".join(reasons),
                }
            )
    return intervals


def save_reference(ref: ReferenceModel, path: str | Path) -> None:
    Path(path).write_text(json.dumps(ref.to_json(), indent=2), encoding="utf-8")


def load_reference(path: str | Path) -> ReferenceModel:
    return ReferenceModel.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def print_summary(flag_df: pd.DataFrame, intervals: List[Dict[str, float | str]]) -> None:
    total = len(flag_df)
    bad = int(flag_df["bad_any"].sum())
    pct = 100.0 * bad / total if total else 0.0
    print(f"Frames total: {total}")
    print(f"Frames suspectes: {bad} ({pct:.2f}%)")
    print(f"Intervalles suspects: {len(intervals)}")
    for idx, row in enumerate(intervals[:20], start=1):
        print(
            f"  {idx:02d}. {row['start_s']:.3f}s -> {row['end_s']:.3f}s | "
            f"durée={row['duration_s']:.3f}s | causes={row['reasons']}"
        )
    if len(intervals) > 20:
        print(f"  ... {len(intervals) - 20} intervalles en plus")


def main() -> None:
    parser = argparse.ArgumentParser(description="Détecte les trackers mal placés à partir de CSV de référence.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Construit un modèle de référence à partir de CSV corrects.")
    train.add_argument("reference_csv", nargs="+", help="CSV de référence considérés comme corrects")
    train.add_argument("--out", default="tracker_reference.json", help="Fichier JSON de sortie")

    check = sub.add_parser("check", help="Vérifie un CSV contre un modèle de référence.")
    check.add_argument("csv", help="CSV à vérifier")
    check.add_argument("--reference", required=True, help="JSON produit par la commande train")
    check.add_argument("--mad-multiplier", type=float, default=8.0, help="Tolérance sur l'écart relatif")
    check.add_argument("--speed-multiplier", type=float, default=3.0, help="Tolérance sur la vitesse")
    check.add_argument("--out", default="tracker_check_report.csv", help="CSV détaillé de sortie")

    args = parser.parse_args()

    if args.cmd == "train":
        ref = build_reference(args.reference_csv)
        save_reference(ref, args.out)
        print(f"Référence sauvegardée dans: {args.out}")
        return

    if args.cmd == "check":
        ref = load_reference(args.reference)
        df = load_csv(args.csv)
        flag_df = detect_bad_frames(
            df,
            ref,
            mad_multiplier=args.mad_multiplier,
            speed_multiplier=args.speed_multiplier,
        )
        intervals = bad_intervals(flag_df)
        flag_df.to_csv(args.out, index=False)
        print_summary(flag_df, intervals)
        print(f"Rapport détaillé: {args.out}")
        return


if __name__ == "__main__":
    main()
