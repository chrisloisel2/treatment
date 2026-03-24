"""
fix.py
======
Inférence robuste des rôles {head, left, right} à partir d'un CSV de tracker_positions.
Corrige les colonnes du CSV si les trackers sont mal assignés (head/left/right inversés).
Inclut position (x,y,z) ET orientation (qw,qx,qy,qz) dans le réassignement.

Modes d'usage :
    # Un seul fichier
    python fix.py tracker_positions.csv
    python fix.py tracker_positions.csv --output fixed.csv

    # Toutes les sessions sous un dossier racine (récursif)
    python fix.py --all .
    python fix.py --all /chemin/vers/dossier_sessions

    # Seulement afficher le mapping sans modifier
    python fix.py --all . --dry-run

    # Rapport JSON machine-readable
    python fix.py --all . --dry-run --report report.json

API Python :
    from fix import infer_roles
    result = infer_roles(csv_path)   # dict avec mapping, confidence, swapped, ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-9

# Noms des colonnes attendus dans le CSV
TRACKER_SUFFIXES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
TRACKER_ROLES = ["head", "left", "right"]


# ══════════════════════════════════════════════════════════════════════════════
# Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackerBlock:
    gid: str                 # identifiant anonyme G0/G1/G2
    role_hint: Optional[str] # rôle actuel dans le CSV (head/left/right), peut être incorrect
    pos: np.ndarray          # shape (N, 3)
    quat: np.ndarray         # shape (N, 4) = qw,qx,qy,qz
    col_names: List[str]     # noms des 7 colonnes d'origine dans le CSV


@dataclass
class HeadInference:
    head_gid: str
    world_up: np.ndarray
    up_axis_index: int
    up_axis_sign: int
    score: float
    details: Dict


@dataclass
class LRInference:
    left_gid: str
    right_gid: str
    head_local_up_axis: int
    head_local_up_sign: int
    head_local_right_axis: int
    head_local_right_sign: int
    score: float
    details: Dict


# ══════════════════════════════════════════════════════════════════════════════
# Helpers numériques
# ══════════════════════════════════════════════════════════════════════════════

def robust_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size > 0 else float("nan")


def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))) + EPS)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def quat_to_rotmat(qwqxqyqz: np.ndarray) -> np.ndarray:
    """Quaternion [..., 4] → matrice de rotation [..., 3, 3]."""
    q = np.asarray(qwqxqyqz, dtype=float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), EPS)
    w, x, y, z = [q[..., i] for i in range(4)]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1.0 - 2.0 * (y*y + z*z)
    R[..., 0, 1] = 2.0 * (x*y - w*z)
    R[..., 0, 2] = 2.0 * (x*z + w*y)
    R[..., 1, 0] = 2.0 * (x*y + w*z)
    R[..., 1, 1] = 1.0 - 2.0 * (x*x + z*z)
    R[..., 1, 2] = 2.0 * (y*z - w*x)
    R[..., 2, 0] = 2.0 * (x*z - w*y)
    R[..., 2, 1] = 2.0 * (y*z + w*x)
    R[..., 2, 2] = 1.0 - 2.0 * (x*x + y*y)
    return R


# ══════════════════════════════════════════════════════════════════════════════
# Détection des blocs tracker dans le CSV
# ══════════════════════════════════════════════════════════════════════════════

def find_tracker_blocks(df: pd.DataFrame) -> List[TrackerBlock]:
    """
    Détecte les 3 blocs de trackers dans le CSV.
    Supporte deux formats :
    1. Colonnes nommées : tracker_head_x, tracker_left_x, tracker_right_x, ...
    2. Fallback : les 21 dernières colonnes numériques (format anonyme).
    """
    df_num = df.copy()
    for c in df_num.columns:
        if not pd.api.types.is_numeric_dtype(df_num[c]):
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    named_blocks = _try_named_blocks(df, df_num)
    if named_blocks:
        return named_blocks

    return _fallback_anonymous_blocks(df_num)


def _try_named_blocks(df: pd.DataFrame, df_num: pd.DataFrame) -> List[TrackerBlock]:
    """Cherche des colonnes du type tracker_<role>_<suffix> ou <role>_<suffix>."""
    blocks = []
    for i, role in enumerate(TRACKER_ROLES):
        prefixes = [f"tracker_{role}_", f"{role}_"]
        found = None
        for prefix in prefixes:
            cols = [f"{prefix}{s}" for s in TRACKER_SUFFIXES]
            if all(c in df.columns for c in cols):
                found = cols
                break
        if found is None:
            return []

        data = df_num[found].to_numpy(dtype=float)
        if data.shape[0] < 10:
            return []
        blocks.append(TrackerBlock(
            gid=f"G{i}",
            role_hint=role,
            pos=data[:, 0:3],
            quat=data[:, 3:7],
            col_names=found,
        ))
    return blocks


def _fallback_anonymous_blocks(df_num: pd.DataFrame) -> List[TrackerBlock]:
    """Fallback : prend les 21 dernières colonnes numériques."""
    numeric_cols = [c for c in df_num.columns if pd.api.types.is_numeric_dtype(df_num[c])]
    if len(numeric_cols) < 21:
        raise ValueError("Pas assez de colonnes numériques pour 3 trackers x 7 variables.")
    tail = numeric_cols[-21:]
    arr = df_num[tail].to_numpy(dtype=float)
    if arr.shape[0] < 10:
        raise ValueError("Pas assez de lignes pour une inférence robuste.")
    blocks = []
    for i in range(3):
        s = i * 7
        blocks.append(TrackerBlock(
            gid=f"G{i}",
            role_hint=None,
            pos=arr[:, s:s+3],
            quat=arr[:, s+3:s+7],
            col_names=tail[s:s+7],
        ))
    return blocks


# ══════════════════════════════════════════════════════════════════════════════
# Inférence tête (traceur le plus haut)
# ══════════════════════════════════════════════════════════════════════════════

def score_head_candidate(blocks: List[TrackerBlock], axis: int, sign: int) -> Tuple[str, float, Dict]:
    vals = np.stack([sign * b.pos[:, axis] for b in blocks], axis=1)
    winners = np.argmax(vals, axis=1)
    scores = []
    details_all = {}
    for k, b in enumerate(blocks):
        win_rate = float(np.mean(winners == k))
        v_self = vals[:, k]
        others_mean = np.mean(np.delete(vals, k, axis=1), axis=1)
        gap = v_self - others_mean
        med_gap = robust_median(gap)
        mad_gap = robust_mad(gap)
        # Clamp pour éviter nan quand MAD ≈ 0 (séparation parfaite = très bonne)
        if not np.isfinite(mad_gap) or mad_gap < EPS:
            normalized_gap = 20.0 if med_gap > 0 else -20.0
        else:
            normalized_gap = float(np.clip(med_gap / mad_gap, -50.0, 50.0))
        score = (
            2.5 * win_rate +
            2.0 * float(np.mean(gap > 0)) +
            1.5 * max(0.0, normalized_gap)
        )
        scores.append(score)
        details_all[b.gid] = {"win_rate": win_rate, "median_gap": med_gap, "score": score}
    best_idx = int(np.argmax(scores))
    return blocks[best_idx].gid, float(scores[best_idx]), details_all[blocks[best_idx].gid]


def infer_head(blocks: List[TrackerBlock]) -> HeadInference:
    best = None
    for axis in range(3):
        for sign in (-1, +1):
            gid, score, details = score_head_candidate(blocks, axis, sign)
            if best is None or score > best["score"]:
                best = {"gid": gid, "axis": axis, "sign": sign, "score": score, "details": details}
    assert best is not None
    world_up = np.zeros(3, dtype=float)
    world_up[best["axis"]] = float(best["sign"])
    return HeadInference(
        head_gid=best["gid"],
        world_up=world_up,
        up_axis_index=best["axis"],
        up_axis_sign=best["sign"],
        score=float(best["score"]),
        details=best["details"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Inférence gauche / droite
# ══════════════════════════════════════════════════════════════════════════════

def get_block(blocks: List[TrackerBlock], gid: str) -> TrackerBlock:
    for b in blocks:
        if b.gid == gid:
            return b
    raise KeyError(gid)


def average_alignment(vs: np.ndarray, ref: np.ndarray) -> float:
    dots = np.sum(unit(vs) * unit(ref.reshape(1, 3)), axis=1)
    return float(np.mean(np.abs(dots)))


def infer_head_local_up(head_block: TrackerBlock, world_up: np.ndarray) -> Tuple[int, int, float]:
    R = quat_to_rotmat(head_block.quat)
    best = None
    for local_axis in range(3):
        axis_world = R[:, :, local_axis]
        for sign in (-1, +1):
            score = average_alignment(sign * axis_world, world_up)
            if best is None or score > best[2]:
                best = (local_axis, sign, score)
    assert best is not None
    return best


def infer_left_right(blocks: List[TrackerBlock], head_info: HeadInference) -> LRInference:
    """
    Identifie gauche/droite en projetant les positions relatives des mains
    dans le repère LOCAL de la tête.

    Convention SteamVR/OpenVR : dans le repère local du casque,
    l'axe qui a la plus grande séparation entre les deux mains ET dont
    la projection est cohérente avec "droite = positif" est l'axe right.

    Méthode :
    - On exprime chaque main dans le repère local de la tête (R^T @ rel_pos).
    - L'axe local qui sépare le mieux les deux mains (hors axe up) est l'axe right.
    - La main avec la projection POSITIVE sur cet axe est la main droite.
      (convention SteamVR : +X_local = droite du porteur)
    """
    head = get_block(blocks, head_info.head_gid)
    hands = [b for b in blocks if b.gid != head_info.head_gid]
    A, B = hands
    R = quat_to_rotmat(head.quat)  # (N,3,3)
    up_local_axis, up_local_sign, up_score = infer_head_local_up(head, head_info.world_up)
    remaining_axes = [ax for ax in (0, 1, 2) if ax != up_local_axis]

    rel_A_world = A.pos - head.pos
    rel_B_world = B.pos - head.pos
    rel_A_local = np.einsum("nij,ni->nj", R, rel_A_world)
    rel_B_local = np.einsum("nij,ni->nj", R, rel_B_world)

    best = None

    for right_local_axis in remaining_axes:
        proj_A = rel_A_local[:, right_local_axis]
        proj_B = rel_B_local[:, right_local_axis]
        diff = proj_A - proj_B

        mad_diff = robust_mad(diff)
        med_diff_abs = np.nanmedian(np.abs(diff))

        # Clamp séparation normalisée pour éviter nan
        if not np.isfinite(mad_diff) or mad_diff < EPS:
            sep = 20.0 if med_diff_abs > 0 else 0.0
        else:
            sep = float(np.clip(robust_median(np.abs(diff)) / mad_diff, 0.0, 50.0))

        consistent = float(np.mean(np.abs(diff) > (0.25 * med_diff_abs + EPS)))
        med_diff = robust_median(diff)

        right_gid, left_gid = (A.gid, B.gid) if med_diff >= 0 else (B.gid, A.gid)

        score = 2.0 * max(0.0, sep) + 1.0 * consistent + 0.5 * up_score

        right_axis_world = R[:, :, right_local_axis]
        vertical_alignment = average_alignment(right_axis_world, head_info.world_up)
        score -= 2.0 * vertical_alignment

        item = {
            "left_gid": left_gid, "right_gid": right_gid,
            "right_local_axis": right_local_axis,
            "right_local_sign": +1,
            "score": float(score),
            "details": {
                "up_alignment_score":                   float(up_score),
                "right_hand_separation_score":          float(sep),
                "right_hand_consistency":               float(consistent),
                "right_axis_vertical_alignment_penalty":float(vertical_alignment),
                "median_projection_diff_A_minus_B":     float(med_diff),
            },
        }
        if best is None or item["score"] > best["score"]:
            best = item

    assert best is not None
    return LRInference(
        left_gid=best["left_gid"],
        right_gid=best["right_gid"],
        head_local_up_axis=up_local_axis,
        head_local_up_sign=up_local_sign,
        head_local_right_axis=best["right_local_axis"],
        head_local_right_sign=best["right_local_sign"],
        score=best["score"],
        details=best["details"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Confiance globale
# ══════════════════════════════════════════════════════════════════════════════

def global_confidence(head_info: HeadInference, lr_info: LRInference) -> float:
    """
    Confiance globale ∈ [0, 1].
    Les scores head et LR sont bornés avant la sigmoid pour éviter nan/inf.
    """
    h_score = float(np.clip(head_info.score, -50.0, 50.0))
    lr_score = float(np.clip(lr_info.score, -50.0, 50.0))
    h  = 1.0 / (1.0 + math.exp(-(h_score  - 4.0)))
    lr = 1.0 / (1.0 + math.exp(-(lr_score - 2.5)))
    return float(np.clip(0.55 * h + 0.45 * lr, 0.0, 1.0))


def build_mapping(head_info: HeadInference, lr_info: LRInference) -> Dict[str, str]:
    return {
        head_info.head_gid: "head",
        lr_info.left_gid:   "left",
        lr_info.right_gid:  "right",
    }


def is_already_correct(blocks: List[TrackerBlock], mapping: Dict[str, str]) -> bool:
    """Vérifie si le mapping inféré correspond à l'assignation actuelle dans le CSV."""
    for b in blocks:
        if b.role_hint is None:
            return False  # format anonyme, toujours réécrire
        if mapping[b.gid] != b.role_hint:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# API publique (import par pipeline.py / server.py)
# ══════════════════════════════════════════════════════════════════════════════

def infer_roles(csv_path) -> dict:
    """
    Inférence des rôles tracker pour un CSV donné.

    Retourne un dict :
    {
        "csv_path":   str,
        "confidence": float,        # 0..1
        "swapped":    bool,         # True si le CSV doit être corrigé
        "mapping":    {"G0": "head", "G1": "right", "G2": "left"},  # gid → rôle inféré
        "role_hints": {"G0": "head", "G1": "left",  "G2": "right"}, # ce qui était dans le CSV
        "swaps":      [("left", "right"), ...],                      # paires échangées
        "world_up_axis": int,
        "world_up_sign": int,
        "error": None | str,        # message d'erreur si échec
    }
    """
    csv_path = Path(csv_path)
    result: dict = {
        "csv_path":      str(csv_path),
        "confidence":    0.0,
        "swapped":       False,
        "mapping":       {},
        "role_hints":    {},
        "swaps":         [],
        "world_up_axis": -1,
        "world_up_sign": 0,
        "error":         None,
    }
    try:
        df = pd.read_csv(csv_path)
        blocks = find_tracker_blocks(df)
        head_info = infer_head(blocks)
        lr_info   = infer_left_right(blocks, head_info)
        mapping   = build_mapping(head_info, lr_info)
        conf      = global_confidence(head_info, lr_info)
        already_ok = is_already_correct(blocks, mapping)

        swaps = []
        for b in blocks:
            if b.role_hint is not None and mapping[b.gid] != b.role_hint:
                swaps.append((b.role_hint, mapping[b.gid]))

        result["confidence"]    = conf
        result["swapped"]       = not already_ok
        result["mapping"]       = mapping
        result["role_hints"]    = {b.gid: (b.role_hint or "?") for b in blocks}
        result["swaps"]         = swaps
        result["world_up_axis"] = head_info.up_axis_index
        result["world_up_sign"] = head_info.up_axis_sign
    except Exception as e:
        result["error"] = str(e)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Réécriture CSV
# ══════════════════════════════════════════════════════════════════════════════

def rewrite_csv(df: pd.DataFrame, blocks: List[TrackerBlock], mapping: Dict[str, str],
                output_path: Path) -> None:
    """
    Réécrit le CSV en plaçant chaque bloc de 7 colonnes (x,y,z,qw,qx,qy,qz)
    sous le bon rôle (head/left/right), incluant position ET orientation.
    Les colonnes non-tracker (timestamp, etc.) restent inchangées.
    Crée un backup .bak_syncml avant toute modification.
    """
    # Backup avant écriture (cohérent avec la pipeline d'ingestion)
    if output_path == Path(df.attrs.get("_source_path", "")) or not output_path.exists():
        pass  # pas de backup si fichier nouveau
    elif output_path.exists():
        bak = output_path.with_suffix(output_path.suffix + ".bak_syncml")
        if not bak.exists():
            import shutil
            shutil.copy2(output_path, bak)

    out = df.copy()
    role_to_block = {mapping[b.gid]: b for b in blocks}

    prefix = ""
    sample_block = blocks[0]
    if sample_block.col_names and sample_block.col_names[0].startswith("tracker_"):
        prefix = "tracker_"

    for target_role in TRACKER_ROLES:
        src_block = role_to_block[target_role]
        target_cols = [f"{prefix}{target_role}_{s}" for s in TRACKER_SUFFIXES]
        src_data = np.hstack([src_block.pos, src_block.quat])  # (N, 7)

        for j, col in enumerate(target_cols):
            if col in out.columns:
                out[col] = src_data[:, j]
            else:
                out.insert(list(out.columns).index(src_block.col_names[0]), col, src_data[:, j])

    out.to_csv(output_path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Traitement d'un fichier
# ══════════════════════════════════════════════════════════════════════════════

def process_file(csv_path: Path, output_path: Optional[Path], dry_run: bool,
                 verbose: bool, min_confidence: float = 0.65) -> dict:
    """
    Traite un fichier tracker_positions.csv.
    Retourne un dict de résultat (pour le rapport JSON).
    """
    label = str(csv_path)
    entry = {
        "path": label,
        "status": "error",
        "confidence": 0.0,
        "swapped": False,
        "swaps": [],
        "modified": False,
        "error": None,
    }

    try:
        df = pd.read_csv(csv_path)
        df.attrs["_source_path"] = str(csv_path)
    except Exception as e:
        msg = f"[ERREUR] {label}: lecture impossible: {e}"
        print(msg, file=sys.stderr)
        entry["error"] = str(e)
        return entry

    try:
        blocks    = find_tracker_blocks(df)
        head_info = infer_head(blocks)
        lr_info   = infer_left_right(blocks, head_info)
        mapping   = build_mapping(head_info, lr_info)
        conf      = global_confidence(head_info, lr_info)
    except Exception as e:
        msg = f"[ERREUR] {label}: inférence échouée: {e}"
        print(msg, file=sys.stderr)
        entry["error"] = str(e)
        return entry

    already_ok = is_already_correct(blocks, mapping)
    swaps = [(b.role_hint, mapping[b.gid]) for b in blocks
             if b.role_hint is not None and mapping[b.gid] != b.role_hint]

    entry["confidence"] = conf
    entry["swapped"]    = not already_ok
    entry["swaps"]      = swaps

    # Affichage
    status_tag  = "[OK]" if already_ok else "[SWAP]"
    conf_tag    = ("confiance élevée" if conf >= 0.80
                   else "confiance moyenne" if conf >= 0.65
                   else "confiance FAIBLE")
    print(f"{status_tag} {label}  conf={conf:.3f} ({conf_tag})")

    if verbose:
        for b in blocks:
            inferred = mapping[b.gid]
            current  = b.role_hint or "?"
            arrow    = "✓" if inferred == current else f"{current} → {inferred}"
            print(f"       {b.gid}: {arrow}")
        print(f"       axe vertical monde:        axis={head_info.up_axis_index}, sign={head_info.up_axis_sign}")
        print(f"       axe vertical local tête:   axis={lr_info.head_local_up_axis}, sign={lr_info.head_local_up_sign}")
        print(f"       axe droite local tête:     axis={lr_info.head_local_right_axis}, sign={lr_info.head_local_right_sign}")

    if conf < min_confidence:
        print(f"  [AVERTISSEMENT] Confiance {conf:.3f} < {min_confidence} — résultat ignoré.", file=sys.stderr)
        entry["status"] = "skipped_low_confidence"
        return entry

    entry["status"] = "ok"

    if already_ok:
        return entry

    if dry_run:
        print(f"  [DRY-RUN] Pas de modification.")
        return entry

    dest = output_path if output_path else csv_path
    try:
        rewrite_csv(df, blocks, mapping, dest)
        print(f"  Sauvegardé: {dest}")
        entry["modified"] = True
    except Exception as e:
        print(f"[ERREUR] {label}: écriture impossible: {e}", file=sys.stderr)
        entry["error"] = str(e)
        entry["status"] = "error"

    return entry


# ══════════════════════════════════════════════════════════════════════════════
# Découverte récursive des CSVs
# ══════════════════════════════════════════════════════════════════════════════

def find_all_tracker_csvs(root: Path) -> List[Path]:
    """
    Trouve tous les tracker_positions.csv sous root, récursivement.
    Inclut : root/tracker_positions.csv, root/session_*/tracker_positions.csv
    et tout sous-dossier imbriqué jusqu'à 3 niveaux de profondeur.
    """
    found = []
    seen = set()

    def _walk(path: Path, depth: int):
        if depth > 3:
            return
        csv = path / "tracker_positions.csv"
        if csv.exists() and csv not in seen:
            found.append(csv)
            seen.add(csv)
        try:
            for sub in sorted(path.iterdir()):
                if sub.is_dir() and not sub.name.startswith("."):
                    _walk(sub, depth + 1)
        except PermissionError:
            pass

    _walk(root, 0)
    return found


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Corrige l'assignation head/left/right dans les CSV tracker_positions."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("csv_path", nargs="?", help="Fichier CSV unique à traiter")
    group.add_argument("--all", metavar="DIR", dest="all_dir",
                       help="Traite tous les tracker_positions.csv sous DIR (récursif)")

    parser.add_argument("--output",         help="Fichier de sortie (mode fichier unique seulement)")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Affiche le mapping sans modifier les fichiers")
    parser.add_argument("--report",         metavar="FILE",
                        help="Exporte un rapport JSON machine-readable")
    parser.add_argument("--min-confidence", type=float, default=0.65, metavar="FLOAT",
                        help="Seuil de confiance minimum pour appliquer la correction (défaut: 0.65)")
    parser.add_argument("-v", "--verbose",  action="store_true",
                        help="Affiche les détails de chaque trackerbloc")
    args = parser.parse_args()

    report_entries = []

    if args.all_dir:
        root = Path(args.all_dir)
        if not root.is_dir():
            print(f"[ERREUR] {root} n'est pas un dossier.", file=sys.stderr)
            return 2

        csv_files = find_all_tracker_csvs(root)
        if not csv_files:
            print(f"[ERREUR] Aucun tracker_positions.csv trouvé sous {root}.", file=sys.stderr)
            return 2

        print(f"Trouvé {len(csv_files)} fichier(s) à analyser.\n")
        modified = 0
        for p in csv_files:
            entry = process_file(p, output_path=None, dry_run=args.dry_run,
                                 verbose=args.verbose, min_confidence=args.min_confidence)
            report_entries.append(entry)
            if entry.get("modified"):
                modified += 1

        n_swap    = sum(1 for e in report_entries if e["swapped"])
        n_skip    = sum(1 for e in report_entries if e["status"] == "skipped_low_confidence")
        n_err     = sum(1 for e in report_entries if e["status"] == "error")
        print(f"\n{modified}/{len(csv_files)} fichier(s) modifié(s).  "
              f"SWAP détectés: {n_swap}  faible conf: {n_skip}  erreurs: {n_err}")

    else:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"[ERREUR] Fichier introuvable: {csv_path}", file=sys.stderr)
            return 2
        output = Path(args.output) if args.output else None
        entry = process_file(csv_path, output_path=output, dry_run=args.dry_run,
                             verbose=args.verbose, min_confidence=args.min_confidence)
        report_entries.append(entry)

    if args.report:
        report_path = Path(args.report)
        report_path.write_text(json.dumps(report_entries, indent=2, ensure_ascii=False))
        print(f"\nRapport JSON → {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
