#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_from_t9.py — Entraînement du modèle check.py sur les données T9.

Stratégie :
  sync/   → toutes les sessions T9 + gold SFTP (bien alignées, offset < 200ms)
  desync/ → copies synthétiques avec offset caméra aléatoire +2000 à +20000ms
            (simule les cas réels de caméra démarrée trop tôt)

Améliorations par rapport au modèle actuel :
  - 1680 sessions sync (vs 27 avant)
  - Desync synthétiques avec VRAIS grands décalages (vs pseudo-labels heuristiques)
  - MAX_LAG_MS = 3000ms (vs 500ms) pour capturer les décalages réels
  - WINDOW_MS = 2500ms (vs 2000ms)
  - EMB = 256 (vs 128)
  - 36 epochs

Usage :
    python train_from_t9.py
    python train_from_t9.py --n-desync 600 --epochs 36
    python train_from_t9.py --skip-setup  # si _training/ existe déjà
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
T9          = Path("/Volumes/T9")
SFTP_CACHE  = ROOT / "_sftp_cache" / "gold"
TRAIN_DIR   = ROOT / "_training"
SYNC_DIR    = TRAIN_DIR / "sync"
DESYNC_DIR  = TRAIN_DIR / "desync"
MODEL_DIR   = ROOT / "_check_model"

# ── Paramètres d'entraînement ─────────────────────────────────────────────────
DEFAULT_N_DESYNC = 700     # sessions desync synthétiques
DEFAULT_EPOCHS   = 36
DEFAULT_BATCH    = 128
DEFAULT_EMB      = 256

# Plage des offsets synthétiques desync (ms)
DESYNC_OFFSET_MIN_MS  = 2_000.0
DESYNC_OFFSET_MAX_MS  = 20_000.0

CAMERAS = ("head", "left", "right")


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def _read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    frames = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if len(line) > 5:
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return frames


def _write_jsonl(path: Path, frames: list[dict]) -> None:
    lines = [json.dumps(f, separators=(",", ":")) + "\r\n" for f in frames]
    path.write_bytes("".join(lines).encode("utf-8"))


def collect_t9_sessions() -> list[Path]:
    """Collecte toutes les sessions T9 avec les fichiers requis."""
    required = [
        "tracker_positions.csv",
        "videos/head.jsonl",
        "videos/left.jsonl",
        "videos/right.jsonl",
    ]
    sessions = []
    for p in T9.rglob("metadata.json"):
        s = p.parent
        if all((s / r).exists() for r in required):
            sessions.append(s)
    return sorted(sessions)


def create_desync_session(
    source_session: Path,
    dest_dir: Path,
    offset_ms: float,
) -> bool:
    """
    Crée une session desync synthétique en décalant les timestamps caméra.

    Seuls les fichiers légers sont copiés (JSONL + CSV + metadata).
    Les MP4 ne sont PAS copiés (non nécessaires pour l'entraînement).

    L'offset est ajouté (pas soustrait) : la caméra démarre APRÈS le tracker
    d'une durée offset_ms. Cela simule le cas où la caméra a été démarrée
    trop tard et ses timestamps sont en avance sur le tracker.

    Args:
        source_session : session T9 d'origine
        dest_dir       : dossier parent de destination
        offset_ms      : décalage à appliquer (ms)

    Returns:
        True si succès
    """
    sid = f"{source_session.name}_desynced_{int(offset_ms)}"
    dst = dest_dir / sid

    if dst.exists():
        return True  # Déjà créé

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "videos").mkdir(exist_ok=True)

    # Copier metadata.json
    meta_src = source_session / "metadata.json"
    if meta_src.exists():
        try:
            meta = json.loads(meta_src.read_text(encoding="utf-8"))
            meta["_synthetic_desync"] = True
            meta["_desync_offset_ms"] = offset_ms
            (dst / "metadata.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            shutil.copy2(str(meta_src), str(dst / "metadata.json"))

    # Copier tracker_positions.csv (inchangé)
    csv_src = source_session / "tracker_positions.csv"
    if csv_src.exists():
        shutil.copy2(str(csv_src), str(dst / "tracker_positions.csv"))

    # Décaler les timestamps JSONL
    for cam in CAMERAS:
        jsonl_src = source_session / "videos" / f"{cam}.jsonl"
        if not jsonl_src.exists():
            continue
        try:
            frames = _read_jsonl(jsonl_src)
            shifted = []
            for fr in frames:
                ct = fr.get("capture_time")
                if ct is not None:
                    fr_new = dict(fr)
                    # Ajouter l'offset : la caméra apparaît comme démarrant PLUS TÔT
                    # (elle couvre des timestamps avant le début du tracker)
                    fr_new["capture_time"] = round(float(ct) - offset_ms, 3)
                    shifted.append(fr_new)
            _write_jsonl(dst / "videos" / f"{cam}.jsonl", shifted)
        except Exception as e:
            print(f"  [WARN] {sid}/{cam}.jsonl: {e}")
            continue

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Setup des répertoires d'entraînement
# ══════════════════════════════════════════════════════════════════════════════

def setup_training_dirs(
    n_desync: int = DEFAULT_N_DESYNC,
    force: bool = False,
) -> tuple[int, int]:
    """
    Prépare _training/sync/ (symlinks) et _training/desync/ (copies modifiées).

    Returns:
        (n_sync, n_desync) effectifs
    """
    print(f"\n[setup] Collecte des sessions T9...")
    all_sessions = collect_t9_sessions()
    print(f"[setup] {len(all_sessions)} sessions T9 trouvées")

    # ── Sync : toutes les sessions T9 + sessions sync/ existantes ────────────
    if force and SYNC_DIR.exists():
        shutil.rmtree(SYNC_DIR)
    SYNC_DIR.mkdir(parents=True, exist_ok=True)

    n_sync = 0
    # Symlinks vers T9
    for sess in all_sessions:
        link = SYNC_DIR / sess.name
        if not link.exists():
            try:
                link.symlink_to(sess)
                n_sync += 1
            except Exception as e:
                print(f"  [WARN] symlink {sess.name}: {e}")
        else:
            n_sync += 1

    # Inclure les sessions sync/ existantes
    existing_sync = ROOT / "sync"
    if existing_sync.exists():
        for p in existing_sync.iterdir():
            if p.is_dir() and (p / "metadata.json").exists():
                link = SYNC_DIR / p.name
                if not link.exists():
                    try:
                        link.symlink_to(p.resolve())
                        n_sync += 1
                    except Exception:
                        pass

    print(f"[setup] sync/: {n_sync} sessions")

    # ── Desync : copies synthétiques ─────────────────────────────────────────
    if force and DESYNC_DIR.exists():
        shutil.rmtree(DESYNC_DIR)
    DESYNC_DIR.mkdir(parents=True, exist_ok=True)

    # Vérifier combien sont déjà créées
    existing_desync = list(DESYNC_DIR.iterdir())
    already_done = len(existing_desync)

    if already_done >= n_desync:
        print(f"[setup] desync/: {already_done} sessions (déjà créées)")
        return n_sync, already_done

    remaining = n_desync - already_done
    print(f"[setup] Création de {remaining} sessions desync synthétiques...")

    rng = random.Random(42)
    candidates = rng.sample(all_sessions, min(n_desync * 3, len(all_sessions)))

    created = already_done
    for i, sess in enumerate(candidates):
        if created >= n_desync:
            break

        # Offset aléatoire entre DESYNC_OFFSET_MIN_MS et DESYNC_OFFSET_MAX_MS
        offset_ms = rng.uniform(DESYNC_OFFSET_MIN_MS, DESYNC_OFFSET_MAX_MS)

        success = create_desync_session(sess, DESYNC_DIR, offset_ms)
        if success:
            created += 1

        if (i + 1) % 100 == 0:
            print(f"  {created}/{n_desync} sessions desync créées...", end="\r")

    print(f"\n[setup] desync/: {created} sessions synthétiques créées")
    return n_sync, created


# ══════════════════════════════════════════════════════════════════════════════
# Entraînement
# ══════════════════════════════════════════════════════════════════════════════

def _collect_sessions_followlinks(root: Path) -> list[Path]:
    """
    Version de _collect_sessions qui suit les symlinks.
    Nécessaire car _training/sync/ contient des symlinks vers T9.
    """
    import os
    sessions = []
    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=True):
        if "metadata.json" in filenames and "videos" in dirnames:
            p = Path(dirpath)
            if p != root:
                sessions.append(p)
    return sorted(sessions)


def patch_and_train(
    epochs:     int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    emb:        int = DEFAULT_EMB,
    max_lag_ms: float = 3000.0,
    window_ms:  float = 2500.0,
    n_workers:  int = None,
):
    """
    Patche les hyperparamètres de check.py et lance l'entraînement.

    Les paramètres critiques patchés :
      EMB      : taille de l'embedding (256 vs 128)
      MAX_LAG  : plage de recherche (3000ms vs 500ms)
      WINDOW   : fenêtre temporelle (2500ms vs 2000ms)
    """
    import check as chk

    # Patcher les hyperparamètres (l'architecture est déjà correcte dans check.py)
    chk.EMB          = emb
    chk.MAX_LAG_MS   = max_lag_ms
    chk.WINDOW_MS    = window_ms
    chk.TRAIN_EPOCHS = epochs
    chk.BATCH_SIZE   = batch_size

    print(f"\n[train] Configuration :")
    print(f"  EMB={emb}  MAX_LAG_MS={max_lag_ms}  WINDOW_MS={window_ms}")
    print(f"  EPOCHS={epochs}  BATCH={batch_size}")
    print(f"  sync_dir  = {SYNC_DIR}")
    print(f"  desync_dir = {DESYNC_DIR}")
    print(f"  model_dir  = {MODEL_DIR}")
    print()

    # Patcher _collect_sessions pour suivre les symlinks
    chk._collect_sessions = _collect_sessions_followlinks

    # Lancer l'entraînement
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)

    chk.train(
        sync_dir=SYNC_DIR,
        desync_dir=DESYNC_DIR,
        epochs=epochs,
        batch_size=batch_size,
        model_dir=MODEL_DIR,
        n_workers=n_workers,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Validation post-entraînement
# ══════════════════════════════════════════════════════════════════════════════

def validate_model(n_samples: int = 30) -> None:
    """
    Valide le modèle entraîné sur un échantillon T9 et sur des sessions desync.
    """
    import check as chk
    print("\n[validate] Chargement du nouveau modèle...")
    model = chk.load_model()
    if model is None:
        print("[validate] Modèle introuvable !")
        return

    # Sessions T9 (synced)
    t9_sessions = list(p.parent for p in T9.rglob("metadata.json") if (p.parent/"videos").exists())
    rng = random.Random(123)
    sample_sync   = rng.sample(t9_sessions, min(n_samples, len(t9_sessions)))
    sample_desync = rng.sample(list(DESYNC_DIR.iterdir()), min(n_samples, len(list(DESYNC_DIR.iterdir()))))

    print(f"\n[validate] {n_samples} sessions SYNC (T9) :")
    sync_scores = []
    for s in sample_sync:
        r = chk.check_session(s, model)
        sync_scores.append(r.score)

    arr = np.array(sync_scores)
    print(f"  mean={arr.mean():.1f}%  min={arr.min():.1f}%  max={arr.max():.1f}%")
    print(f"  ≥70%: {(arr>=70).sum()}/{len(arr)}")

    print(f"\n[validate] {n_samples} sessions DESYNC (synthétiques) :")
    desync_scores = []
    for s in sample_desync[:n_samples]:
        if (s / "metadata.json").exists():
            r = chk.check_session(s, model)
            desync_scores.append(r.score)

    if desync_scores:
        arr2 = np.array(desync_scores)
        print(f"  mean={arr2.mean():.1f}%  min={arr2.min():.1f}%  max={arr2.max():.1f}%")
        print(f"  <40%: {(arr2<40).sum()}/{len(arr2)}")

    if sync_scores and desync_scores:
        sep = np.mean(sync_scores) - np.mean(desync_scores)
        print(f"\n  Séparation sync-desync : {sep:.1f}% (objectif > 20%)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Entraîne check.py sur les données T9."
    )
    parser.add_argument("--n-desync",   type=int, default=DEFAULT_N_DESYNC)
    parser.add_argument("--epochs",     type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",      type=int, default=DEFAULT_BATCH)
    parser.add_argument("--emb",        type=int, default=DEFAULT_EMB)
    parser.add_argument("--max-lag",    type=float, default=3000.0)
    parser.add_argument("--window",     type=float, default=2500.0)
    parser.add_argument("--workers",    type=int, default=None)
    parser.add_argument("--skip-setup", action="store_true",
                        help="Ne pas recréer les répertoires d'entraînement")
    parser.add_argument("--force",      action="store_true",
                        help="Recréer même si les répertoires existent")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate_model()
        return

    if not args.skip_setup:
        n_sync, n_desync = setup_training_dirs(
            n_desync=args.n_desync,
            force=args.force,
        )
        print(f"\n[setup] Prêt : sync={n_sync}  desync={n_desync}")
    else:
        n_sync   = sum(1 for _ in SYNC_DIR.iterdir()) if SYNC_DIR.exists() else 0
        n_desync = sum(1 for _ in DESYNC_DIR.iterdir()) if DESYNC_DIR.exists() else 0
        print(f"[setup] Répertoires existants : sync={n_sync}  desync={n_desync}")

    if n_sync == 0 or n_desync == 0:
        print("[ERROR] Répertoires d'entraînement vides !", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    patch_and_train(
        epochs=args.epochs,
        batch_size=args.batch,
        emb=args.emb,
        max_lag_ms=args.max_lag,
        window_ms=args.window,
        n_workers=args.workers,
    )
    print(f"\n[done] Entraînement terminé en {(time.time()-t0)/60:.1f} min")

    validate_model()


if __name__ == "__main__":
    main()
