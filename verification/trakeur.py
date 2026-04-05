#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def moving_average(arr, window=9):
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(arr[:, j], kernel, mode="same")
    return out


def rank_points(values, higher_better=True):
    """
    Sur 3 trackers:
    - meilleur = 2 points
    - moyen    = 1 point
    - pire     = 0 point
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)  # croissant
    pts = np.zeros(len(values), dtype=float)

    if higher_better:
        pts[order[0]] = 0.0
        pts[order[1]] = 1.0
        pts[order[2]] = 2.0
    else:
        pts[order[0]] = 2.0
        pts[order[1]] = 1.0
        pts[order[2]] = 0.0

    return pts


def quat_to_rotmat_wxyz(q):
    q = q.astype(float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = np.empty((q.shape[0], 3, 3), dtype=float)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def quat_to_rotmat_xyzw(q):
    q = q.astype(float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)

    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = np.empty((q.shape[0], 3, 3), dtype=float)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def split_blocks(df, meta_cols=3, block_size=7, smooth_window=9):
    data = df.iloc[:, meta_cols:].to_numpy(dtype=float)
    n_blocks = data.shape[1] // block_size

    if n_blocks != 3:
        raise ValueError(f"Expected 3 tracker blocks, found {n_blocks}")

    blocks = []
    for i in range(n_blocks):
        block = data[:, i * block_size:(i + 1) * block_size]
        pos = moving_average(block[:, :3], smooth_window)
        quat = block[:, 3:7]
        blocks.append((i, pos, quat))
    return blocks


def parse_truth_from_headers(df, meta_cols=3, block_size=7):
    cols = list(df.columns)[meta_cols:]
    n_blocks = len(cols) // block_size

    if n_blocks != 3:
        raise ValueError(f"Expected 3 tracker blocks in headers, found {n_blocks}")

    truth = {}

    for i in range(n_blocks):
        block_cols = cols[i * block_size:(i + 1) * block_size]
        joined = " ".join(str(c).lower() for c in block_cols)

        found = []
        if "head" in joined:
            found.append("head")
        if "left" in joined:
            found.append("left")
        if "right" in joined:
            found.append("right")

        if len(found) != 1:
            raise ValueError(f"Cannot infer unique label for block {i}: {block_cols}")

        truth[found[0]] = i

    required = {"head", "left", "right"}
    if set(truth.keys()) != required:
        raise ValueError(f"Incomplete truth mapping: {truth}")

    return truth


# ---------------------------------------------------------------------
# HEAD PREDICTION
# ---------------------------------------------------------------------

def detect_head(blocks):
    """
    Score stable par rangs.
    Le head est en général :
    - moins mobile
    - moins dispersé
    - plus haut sur l'axe Y (axis=1, hauteur physique dans le repère tracker)
    - éventuellement extrême sur un autre axe secondaire
    """

    centers = []
    motions = []
    spreads = []

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

    # centralité dynamique: distance médiane aux deux autres, frame par frame
    pair_mean = np.zeros(3, dtype=float)
    for i in range(3):
        d = []
        for j in range(3):
            if i == j:
                continue
            dij = np.linalg.norm(blocks[i][1] - blocks[j][1], axis=1)
            d.append(np.median(dij))
        pair_mean[i] = np.mean(d)

    # Axe Y = axe vertical physique dans le repère VR (jamais inversé)
    h_y = centers[:, 1]

    # Meilleur axe secondaire parmi X et Z (les deux signes autorisés)
    best_sep_other = -np.inf
    best_h_other = np.zeros(3, dtype=float)
    for axis in (0, 2):
        for sign in (-1, 1):
            h = sign * centers[:, axis]
            hs = np.sort(h)
            sep = (hs[-1] - hs[-2]) / (np.ptp(h) + 1e-9)
            if sep > best_sep_other:
                best_sep_other = sep
                best_h_other = h

    # Séparation relative : (leader - 2ème) / range
    def _sep(vals, higher):
        sv = np.sort(vals)
        gap = (sv[-1] - sv[-2]) if higher else (sv[1] - sv[0])
        return float(gap / (np.ptp(vals) + 1e-9))

    # Poids adaptatifs
    w_hy   = 6.0 * (1.0 + 2.0 * _sep(h_y,       higher=True))
    w_pair = 3.0 * (1.0 + 2.0 * _sep(pair_mean,  higher=False))

    score = (
        1.0    * rank_points(motions,      higher_better=False) +
        1.0    * rank_points(spreads,      higher_better=False) +
        w_pair * rank_points(pair_mean,    higher_better=False) +
        w_hy   * rank_points(h_y,          higher_better=True)  +
        1.0    * rank_points(best_h_other, higher_better=True)
    )

    head_idx = int(np.argmax(score))
    return head_idx, score


# ---------------------------------------------------------------------
# HAND PREDICTION WITH A FIXED GLOBAL RULE
# ---------------------------------------------------------------------

def predict_hands_with_rule(blocks, head_idx, quat_mode, axis, sign):
    """
    Règle figée:
    - quat_mode in {"xyzw", "wxyz"}
    - axis in {0,1,2}
    - sign in {-1,+1}

    Pas d'optimisation par session.
    """

    head = [b for b in blocks if b[0] == head_idx][0]
    others = [b for b in blocks if b[0] != head_idx]

    _, head_pos, head_quat = head
    idx_a, pos_a, _ = others[0]
    idx_b, pos_b, _ = others[1]

    if quat_mode == "xyzw":
        R = quat_to_rotmat_xyzw(head_quat)
    elif quat_mode == "wxyz":
        R = quat_to_rotmat_wxyz(head_quat)
    else:
        raise ValueError(f"Unknown quat_mode: {quat_mode}")

    basis = sign * R[:, :, axis]

    proj_a = np.sum((pos_a - head_pos) * basis, axis=1)
    proj_b = np.sum((pos_b - head_pos) * basis, axis=1)

    med_a = float(np.median(proj_a))
    med_b = float(np.median(proj_b))

    if med_a <= med_b:
        left, right = idx_a, idx_b
    else:
        left, right = idx_b, idx_a

    return left, right


# ---------------------------------------------------------------------
# SESSION COLLECTION
# ---------------------------------------------------------------------

def collect_sessions(root_path):
    sessions = []

    for name in sorted(os.listdir(root_path)):
        session_path = os.path.join(root_path, name)
        if not os.path.isdir(session_path):
            continue

        csv_path = os.path.join(session_path, "tracker_positions.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        blocks = split_blocks(df)
        truth = parse_truth_from_headers(df)
        pred_head, head_score = detect_head(blocks)

        sessions.append({
            "name": name,
            "csv_path": csv_path,
            "blocks": blocks,
            "truth": truth,
            "pred_head": pred_head,
            "head_score": head_score,
        })

    return sessions


# ---------------------------------------------------------------------
# GLOBAL RULE SEARCH
# ---------------------------------------------------------------------

def candidate_rules():
    # ordre volontaire: xyzw d'abord, car ton log montre qu'il domine
    rules = []
    for quat_mode in ("xyzw", "wxyz"):
        for axis in (0, 1, 2):
            for sign in (-1, 1):
                rules.append((quat_mode, axis, sign))
    return rules


def choose_best_global_rule(train_sessions):
    best_rule = None
    best_acc = -1.0
    best_ok = -1
    best_total = 0

    for quat_mode, axis, sign in candidate_rules():
        ok = 0
        total = 0

        for s in train_sessions:
            # on calibre la règle de mains uniquement sur les sessions
            # où le head a déjà été correctement trouvé
            if s["pred_head"] != s["truth"]["head"]:
                continue

            pred_left, pred_right = predict_hands_with_rule(
                s["blocks"],
                s["pred_head"],
                quat_mode,
                axis,
                sign,
            )

            ok += int(pred_left == s["truth"]["left"])
            ok += int(pred_right == s["truth"]["right"])
            total += 2

        acc = ok / total if total > 0 else -1.0

        if acc > best_acc:
            best_acc = acc
            best_rule = (quat_mode, axis, sign)
            best_ok = ok
            best_total = total

    if best_rule is None:
        # fallback minimal
        best_rule = ("xyzw", 0, -1)
        best_ok = 0
        best_total = 0
        best_acc = 0.0

    return best_rule, best_acc, best_ok, best_total


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def evaluate_leave_one_out(sessions):
    results = []

    for i, test_session in enumerate(sessions):
        train_sessions = [s for j, s in enumerate(sessions) if j != i]
        rule, rule_acc, rule_ok, rule_total = choose_best_global_rule(train_sessions)

        pred_head = test_session["pred_head"]
        pred_left, pred_right = predict_hands_with_rule(
            test_session["blocks"],
            pred_head,
            rule[0],
            rule[1],
            rule[2],
        )

        pred = {
            "head": pred_head,
            "left": pred_left,
            "right": pred_right,
        }

        truth = test_session["truth"]

        correct = {
            "head": pred["head"] == truth["head"],
            "left": pred["left"] == truth["left"],
            "right": pred["right"] == truth["right"],
        }

        results.append({
            "name": test_session["name"],
            "pred": pred,
            "truth": truth,
            "correct": correct,
            "exact": all(correct.values()),
            "rule": rule,
            "rule_acc_train": rule_acc,
            "head_score": test_session["head_score"],
        })

    return results


def print_results(results):
    total = len(results)
    head_ok = 0
    left_ok = 0
    right_ok = 0
    exact_ok = 0

    print("SCAN: .")
    print("=" * 88)

    for r in results:
        head_ok += int(r["correct"]["head"])
        left_ok += int(r["correct"]["left"])
        right_ok += int(r["correct"]["right"])
        exact_ok += int(r["exact"])

        hs = np.round(r["head_score"], 3).tolist()
        quat_mode, axis, sign = r["rule"]

        print(r["name"])
        print(f"  PRED  : head={r['pred']['head']} left={r['pred']['left']} right={r['pred']['right']}")
        print(f"  TRUE  : head={r['truth']['head']} left={r['truth']['left']} right={r['truth']['right']}")
        print(f"  OK    : head={r['correct']['head']} left={r['correct']['left']} right={r['correct']['right']}")
        print(f"  EXACT : {r['exact']}")
        print(f"  RULE  : quat={quat_mode} axis={axis} sign={sign}")
        print(f"  TRAIN : hand_acc={r['rule_acc_train']:.4f}")
        print(f"  HSCORE: {hs}")
        print("-" * 88)

    print()
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"Sessions évaluées      : {total}")

    if total == 0:
        return

    head_acc = head_ok / total
    left_acc = left_ok / total
    right_acc = right_ok / total
    label_acc = (head_ok + left_ok + right_ok) / (3 * total)
    exact_acc = exact_ok / total

    print(f"Accuracy head          : {head_acc:.4f} ({head_ok}/{total})")
    print(f"Accuracy left          : {left_acc:.4f} ({left_ok}/{total})")
    print(f"Accuracy right         : {right_acc:.4f} ({right_ok}/{total})")
    print(f"Accuracy labels globale: {label_acc:.4f} ({head_ok + left_ok + right_ok}/{3 * total})")
    print(f"Accuracy session exacte: {exact_acc:.4f} ({exact_ok}/{total})")


def print_best_rule_full_dataset(sessions):
    rule, acc, ok, total = choose_best_global_rule(sessions)
    quat_mode, axis, sign = rule

    print()
    print("=" * 88)
    print("BEST GLOBAL RULE ON ALL LABELED SESSIONS")
    print("=" * 88)
    print(f"quat_mode              : {quat_mode}")
    print(f"axis                   : {axis}")
    print(f"sign                   : {sign}")
    print(f"train hand accuracy    : {acc:.4f} ({ok}/{total})")


# ---------------------------------------------------------------------
# SINGLE-SESSION CHECK (utilisé par le serveur)
# ---------------------------------------------------------------------

def check_single_session(session_path, root_path=None):
    """Vérifie que les trackers d'une session unique sont bien positionnés.

    Règle fixe calibrée sur 21 sessions : xyzw, axis=0, sign=-1 → 100% accuracy.
    Pas besoin de corpus d'entraînement.

    Retourne un dict :
      {
        "ok": bool,               # True si head+left+right tous corrects
        "pred":  {"head":int, "left":int, "right":int},
        "truth": {"head":int, "left":int, "right":int},
        "correct": {"head":bool, "left":bool, "right":bool},
        "rule": (quat_mode, axis, sign),
        "error": str | None,
      }
    """
    # Règle fixe calibrée — ne pas modifier sans re-calibrer sur le corpus complet
    RULE = ("xyzw", 0, -1)

    session_path = str(session_path)
    csv_path = os.path.join(session_path, "tracker_positions.csv")

    if not os.path.exists(csv_path):
        return {"ok": False, "error": "tracker_positions.csv introuvable", "pred": {}, "truth": {}, "correct": {}, "rule": None}

    try:
        df = pd.read_csv(csv_path)
        blocks = split_blocks(df)
        truth = parse_truth_from_headers(df)
        pred_head, _ = detect_head(blocks)
        pred_left, pred_right = predict_hands_with_rule(blocks, pred_head, RULE[0], RULE[1], RULE[2])

        pred    = {"head": pred_head, "left": pred_left, "right": pred_right}
        correct = {k: pred[k] == truth[k] for k in ("head", "left", "right")}

        return {
            "ok":      all(correct.values()),
            "pred":    pred,
            "truth":   truth,
            "correct": correct,
            "rule":    RULE,
            "error":   None,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "pred": {}, "truth": {}, "correct": {}, "rule": None}


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python trakeur.py /path/to/root")
        sys.exit(1)

    root_path = sys.argv[1]
    sessions = collect_sessions(root_path)
    results = evaluate_leave_one_out(sessions)
    print_results(results)
    print_best_rule_full_dataset(sessions)


if __name__ == "__main__":
    main()
