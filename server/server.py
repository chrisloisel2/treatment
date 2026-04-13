#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SyncML Studio — Serveur web FastAPI.

Architecture 3 chemins :
  /mnt/storage/silver/  → source brute, lecture seule
  /mnt/storage/silver//    → espace de travail (copie de travail)
 /home/ia/silver    → sortie finale validée (seulement si write_mode=True)

Intégration dans une pipeline big data :
  - API REST JSON pour automatisation / orchestrateurs (Airflow, Prefect, etc.)
  - WebSocket pour streaming temps-réel des logs et progress
  - Endpoints de statut machine-readable pour health checks
  - Tous les jobs tournent dans des threads séparés, non-bloquants

Lancement :
  python server.py [--host 0.0.0.0] [--port 8000]
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import io
import json
import multiprocessing
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ajoute la racine du projet au chemin pour trouver utils/ et pipeline/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# État global des jobs
# ──────────────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    ERROR     = "error"


@dataclass
class Job:
    id:         str
    kind:       str          # "train" | "infer" | "scan"
    status:     JobStatus    = JobStatus.PENDING
    created_at: str          = field(default_factory=lambda: _now())
    started_at: Optional[str] = None
    ended_at:   Optional[str] = None
    progress:   float        = 0.0    # 0–100
    result:     Any          = None
    error:      Optional[str] = None
    logs:       List[dict]   = field(default_factory=list)
    # Données de training
    losses:     List[float]  = field(default_factory=list)
    epochs_done: int         = 0
    total_epochs: int        = 0
    # Pseudo-labels
    pseudo_total: int = 0
    pseudo_pos:   int = 0
    pseudo_neg:   int = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_jsonl(path) -> list:
    """Lit un fichier JSONL en ignorant les lignes invalides (fichier tronqué, etc.)."""
    rows = []
    try:
        for line in Path(path).read_bytes().decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # ligne tronquée ou corrompue — ignorée
    except Exception:
        pass
    return rows


# Chemins hardcodés (importés depuis pipeline)
try:
    from pipeline.pipeline import INGEST_DIR, SILVER_DIR, MODEL_DIR
except ImportError:
    INGEST_DIR = Path("/mnt/storage/silver/")
    SILVER_DIR = Path("/home/ia/silver")
    MODEL_DIR  = INGEST_DIR / "_sync_ml_model"


# Répertoire de persistance des jobs sur disque
JOBS_DIR = INGEST_DIR.parent / "_server_jobs"

def _session_writable(session_path: str) -> bool:
    """Teste si le dossier session lui-même est accessible en écriture."""
    probe = Path(session_path) / ".write_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
        return True
    except Exception:
        return False

# Registre en mémoire des jobs
_jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()

# WebSocket : liste des clients connectés
_ws_clients: List[WebSocket] = []
_ws_lock = asyncio.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# Persistance des jobs sur disque
# ──────────────────────────────────────────────────────────────────────────────

def _job_file(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _persist_job(job: Job):
    """Sauvegarde le job sur disque (non-bloquant, erreurs silencieuses)."""
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        d = {
            "id":           job.id,
            "kind":         job.kind,
            "status":       job.status,
            "created_at":   job.created_at,
            "started_at":   job.started_at,
            "ended_at":     job.ended_at,
            "progress":     job.progress,
            "result":       job.result,
            "error":        job.error,
            "losses":       job.losses[-200:],
            "epochs_done":  job.epochs_done,
            "total_epochs": job.total_epochs,
            "pseudo_total": job.pseudo_total,
            "pseudo_pos":   job.pseudo_pos,
            "pseudo_neg":   job.pseudo_neg,
            "logs":         job.logs[-500:],   # garder les 500 dernières lignes
        }
        tmp = _job_file(job.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(_sanitize_json(d), ensure_ascii=False), encoding="utf-8")
        tmp.replace(_job_file(job.id))
    except Exception:
        pass


def _load_jobs_from_disk():
    """Recharge tous les jobs persistés au démarrage du serveur."""
    if not JOBS_DIR.exists():
        return
    for p in sorted(JOBS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            job = Job(
                id           = d["id"],
                kind         = d["kind"],
                status       = JobStatus(d["status"]),
                created_at   = d.get("created_at", _now()),
                started_at   = d.get("started_at"),
                ended_at     = d.get("ended_at"),
                progress     = d.get("progress", 0.0),
                result       = d.get("result"),
                error        = d.get("error"),
                losses       = d.get("losses", []),
                epochs_done  = d.get("epochs_done", 0),
                total_epochs = d.get("total_epochs", 0),
                pseudo_total = d.get("pseudo_total", 0),
                pseudo_pos   = d.get("pseudo_pos", 0),
                pseudo_neg   = d.get("pseudo_neg", 0),
                logs         = d.get("logs", []),
            )
            # Jobs interrompus par un crash → marqués ERROR
            if job.status in (JobStatus.RUNNING, JobStatus.PENDING):
                job.status  = JobStatus.ERROR
                job.error   = (job.error or "") + " [interrompu par redémarrage serveur]"
                job.ended_at = _now()
                _persist_job(job)
            _jobs[job.id] = job
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _new_job(kind: str) -> Job:
    j = Job(id=str(uuid.uuid4())[:8], kind=kind)
    with _jobs_lock:
        _jobs[j.id] = j
    _persist_job(j)
    return j


_LOG_MEM_LIMIT = 2000   # entrées max gardées en mémoire par job
_LOG_PERSIST_EVERY = 20  # on persiste sur disque toutes les N entrées (anti-thrash)

def _log_job(job: Job, msg: str, level: str = "INFO"):
    entry = {"ts": _now(), "msg": msg, "level": level}
    job.logs.append(entry)
    # Tronquer en mémoire pour éviter l'OOM sur les gros jobs
    if len(job.logs) > _LOG_MEM_LIMIT:
        job.logs = job.logs[-_LOG_MEM_LIMIT:]
    # Persister seulement toutes les N entrées (ou sur les niveaux importants)
    if level in ("ERROR", "WARN", "OK") or len(job.logs) % _LOG_PERSIST_EVERY == 0:
        _persist_job(job)
    asyncio.run_coroutine_threadsafe(
        _broadcast({"type": "log", "job_id": job.id, "entry": entry}),
        _loop,
    )


def _update_job(job: Job, **kwargs):
    for k, v in kwargs.items():
        setattr(job, k, v)
    _persist_job(job)
    asyncio.run_coroutine_threadsafe(
        _broadcast({"type": "job_update", "job": _job_to_dict(job)}),
        _loop,
    )


def _sanitize_json(obj):
    """Remplace récursivement nan/inf par None pour conformité JSON."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):  # nan or inf
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


def _job_to_dict(job: Job) -> dict:
    return _sanitize_json({
        "id":           job.id,
        "kind":         job.kind,
        "status":       job.status,
        "created_at":   job.created_at,
        "started_at":   job.started_at,
        "ended_at":     job.ended_at,
        "progress":     job.progress,
        "error":        job.error,
        "losses":       job.losses[-100:],   # dernières 100 valeurs max
        "epochs_done":  job.epochs_done,
        "total_epochs": job.total_epochs,
        "pseudo_total": job.pseudo_total,
        "pseudo_pos":   job.pseudo_pos,
        "pseudo_neg":   job.pseudo_neg,
        "result":       job.result,
        "log_count":    len(job.logs),
    })


async def _broadcast(payload: dict):
    async with _ws_lock:
        dead = []
        for ws in _ws_clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)


# ──────────────────────────────────────────────────────────────────────────────
# Workers (thread)
# ──────────────────────────────────────────────────────────────────────────────

def _worker_scan(job: Job, limit: int = 500, offset: int = 0, root: str = "", date_filter: str = ""):
    try:
        scan_dir = Path(root) if root else INGEST_DIR
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=10)
        _cap = f" (limite {limit}, offset {offset})" if limit or offset else ""
        _date_info = f", date={date_filter}" if date_filter else ""
        _log_job(job, f"Scan de {scan_dir}…{_cap}{_date_info}")

        import utils.sync as ia

        # ── Phase 1 : Découverte parallèle superficielle ──────────────────────
        # On explore max 3 niveaux de dossiers avec os.scandir (jamais rglob).
        # Chaque niveau est parallélisé via ThreadPoolExecutor (I/O-bound sur NFS).
        # Dès que `limit` sessions sont trouvées on arrête immédiatement.

        SCAN_WORKERS = min(32, (os.cpu_count() or 4) * 4)  # threads I/O

        def _skip_dir(name: str) -> bool:
            return name.startswith(("_", ".")) or "__FAILED" in name

        def _list_subdirs(p: Path):
            try:
                return [Path(e.path) for e in os.scandir(p)
                        if e.is_dir(follow_symlinks=False) and not _skip_dir(e.name)]
            except Exception:
                return []

        def _is_session(p: Path) -> bool:
            """Un seul scandir pour voir si metadata.json ou videos/ est présent."""
            try:
                names = {e.name for e in os.scandir(p)}
                return "metadata.json" in names or "videos" in names
            except Exception:
                return False

        found_lock = threading.Lock()
        sessions: list = []
        stop_flag = threading.Event()

        def _process_candidate(p: Path):
            if stop_flag.is_set():
                return
            if date_filter and date_filter not in p.name:
                return
            if _is_session(p):
                with found_lock:
                    if not stop_flag.is_set():
                        sessions.append(p)
                        if limit and len(sessions) >= limit + offset:
                            stop_flag.set()

        def _gather_subdirs_parallel(parents: list, ex) -> list:
            """Récupère en parallèle les sous-dossiers de chaque dossier parent."""
            if not parents:
                return []
            futs = {ex.submit(_list_subdirs, p): p for p in parents}
            result = []
            for fut in concurrent.futures.as_completed(futs):
                result.extend(fut.result())
            return result

        if not scan_dir.exists():
            _log_job(job, f"Répertoire introuvable : {scan_dir}", "ERROR")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
                lvl1 = _list_subdirs(scan_dir)
                _log_job(job, f"Niveau 1 : {len(lvl1)} sous-dossier(s) dans {scan_dir}", "INFO")
                futs1 = [ex.submit(_process_candidate, p) for p in lvl1]
                concurrent.futures.wait(futs1)
                _log_job(job, f"Après niveau 1 : {len(sessions)} session(s) trouvée(s)", "INFO")

                lvl2 = []
                if not stop_flag.is_set():
                    sess_set = set(sessions)
                    non_sess1 = [p for p in lvl1 if p not in sess_set]
                    lvl2 = _gather_subdirs_parallel(non_sess1, ex)
                    _log_job(job, f"Niveau 2 : {len(lvl2)} sous-dossier(s)", "INFO")
                    futs2 = [ex.submit(_process_candidate, p) for p in lvl2]
                    concurrent.futures.wait(futs2)
                    _log_job(job, f"Après niveau 2 : {len(sessions)} session(s) trouvée(s)", "INFO")

                if not stop_flag.is_set():
                    sess_set = set(sessions)
                    non_sess2 = [p for p in lvl2 if p not in sess_set]
                    lvl3 = _gather_subdirs_parallel(non_sess2, ex)
                    _log_job(job, f"Niveau 3 : {len(lvl3)} sous-dossier(s)", "INFO")
                    futs3 = [ex.submit(_process_candidate, p) for p in lvl3]
                    concurrent.futures.wait(futs3)
                    _log_job(job, f"Après niveau 3 : {len(sessions)} session(s) trouvée(s)", "INFO")

        model_exists = (MODEL_DIR / "model.pt").exists()
        total_found = len(sessions)
        # Trier, paginer
        all_sorted = sorted(sessions, key=lambda p: p.name, reverse=True)
        if offset:
            all_sorted = all_sorted[offset:]
        if limit:
            all_sorted = all_sorted[:limit]

        # ── Phase 2 : Enrichissement parallèle ───────────────────────────────
        # Pour chaque session retenue, lire metadata.json + flags en parallèle.
        RESULTS_JSON = getattr(ia, "RESULTS_JSON", None) if True else None
        try:
            import utils.sync as _ia2
            RESULTS_JSON = getattr(_ia2, "RESULTS_JSON", None)
        except Exception:
            pass

        def _enrich(s: Path) -> dict:
            meta = {}
            meta_path = s / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    pass

            vid_dir = s / "videos"
            vid_exists = vid_dir.is_dir()
            vid_names: set = set()
            if vid_exists:
                try:
                    vid_names = {e.name for e in os.scandir(vid_dir)}
                except Exception:
                    pass

            has_flux_csv  = any(n.endswith("_flux.csv") for n in vid_names)
            has_jsonl     = any(n.endswith(".jsonl") for n in vid_names)
            rotate_marker = vid_dir / ".rotate_done"
            video_rotated = ".rotate_done" in vid_names
            rotate_info   = None
            if video_rotated:
                try:
                    rotate_info = json.loads(rotate_marker.read_text(encoding="utf-8"))
                except Exception:
                    rotate_info = {}

            try:
                root_names = {e.name for e in os.scandir(s)}
            except Exception:
                root_names = set()

            has_tracker  = "tracker_positions.csv" in root_names
            has_gripper  = ("gripper_left_data.csv" in root_names
                            or "gripper_right_data.csv" in root_names)
            has_ux       = "ux_data.csv" in root_names
            has_subtitle = "episode_subtitle.json" in root_names

            pipeline_steps: dict = {}
            pipeline_done = False
            if "pipeline_state.json" in root_names:
                try:
                    ps = json.loads((s / "pipeline_state.json").read_text())
                    pipeline_steps = ps.get("steps", {})
                    pipeline_done  = ps.get("finished", False) and ps.get("success", False)
                except Exception:
                    pass

            last_result = None
            if RESULTS_JSON and RESULTS_JSON in root_names:
                try:
                    last_result = json.loads((s / RESULTS_JSON).read_text())
                except Exception:
                    pass

            return {
                "name":           s.name,
                "path":           str(s),
                "action":         s.parent.name,
                "has_tracker":    has_tracker,
                "has_gripper":    has_gripper,
                "has_ux":         has_ux,
                "has_flux_csv":   has_flux_csv,
                "has_jsonl":      has_jsonl,
                "has_subtitle":   has_subtitle,
                "video_rotated":  video_rotated,
                "rotate_info":    rotate_info,
                "meta":           meta,
                "last_result":    last_result,
                "pipeline_done":  pipeline_done,
                "pipeline_steps": pipeline_steps,
            }

        _update_job(job, progress=50)
        result = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
            futs = {ex.submit(_enrich, s): s for s in all_sorted}
            done_count = 0
            for fut in concurrent.futures.as_completed(futs):
                try:
                    result.append(fut.result())
                except Exception:
                    pass
                done_count += 1
                _update_job(job, progress=50 + int(48 * done_count / max(len(all_sorted), 1)))
        # Rétablir l'ordre (as_completed ne le garantit pas) — plus récent en premier
        result.sort(key=lambda r: r["name"], reverse=True)

        has_more = bool(limit and (offset + len(result)) < total_found)
        _log_job(job, f"{len(result)} sessions retournées (total trouvé: {total_found}).", "OK")
        if has_more:
            _log_job(job, f"  → {total_found - offset - len(result)} session(s) supplémentaire(s) disponibles (offset={offset+len(result)})", "INFO")
        _update_job(
            job,
            status    = JobStatus.DONE,
            ended_at  = _now(),
            progress  = 100,
            result    = {
                "sessions":    result,
                "ingest_dir":  str(scan_dir),
                "silver_dir":  str(SILVER_DIR),
                "model_exists": model_exists,
                "total_found": total_found,
                "offset":      offset,
                "limit":       limit,
                "has_more":    has_more,
            },
        )
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


def _worker_train(job: Job, sessions: List[str], params: dict):
    try:
        import utils.sync as ia
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=2,
                    total_epochs=params["epochs"])
        _log_job(job, f"Chargement de {len(sessions)} session(s)…")

        ia.set_seed()
        sess = [Path(s) for s in sessions]

        examples = ia.build_training_examples(
            sessions      = sess,
            resample_ms   = params["resample_ms"],
            max_lag_ms    = params["max_lag_ms"],
            window_ms     = params["window_ms"],
            signal_config = params.get("signal_config"),
        )

        if len(examples) < 50:
            raise RuntimeError(f"Seulement {len(examples)} exemples — données insuffisantes.")

        pos = sum(y for _, _, y, _ in examples)
        neg = len(examples) - pos
        _update_job(job, pseudo_total=len(examples), pseudo_pos=pos, pseudo_neg=neg, progress=8)
        _log_job(job, f"Pseudo-labels: total={len(examples)} pos={pos} neg={neg}", "OK")

        ds = ia.PairWindowDataset(examples)
        dl = DataLoader(ds, batch_size=params["batch_size"], shuffle=True, drop_last=False)

        model = ia.CrossModalAligner().to(ia.DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=ia.WEIGHT_DECAY)

        epochs = params["epochs"]
        losses = []
        t0 = time.time()

        for epoch in range(epochs):
            model.train()
            ep_losses = []
            for xa, xb, y in dl:
                xa, xb, y = xa.to(ia.DEVICE), xb.to(ia.DEVICE), y.to(ia.DEVICE)
                opt.zero_grad()
                logit, ea, eb = model(xa, xb)
                bce  = F.binary_cross_entropy_with_logits(logit, y)
                ctr  = ia.contrastive_margin(ea, eb, y)
                loss = 0.75 * bce + 0.25 * ctr
                loss.backward()
                opt.step()
                ep_losses.append(float(loss.item()))

            mean_loss = float(np.mean(ep_losses))
            losses.append(mean_loss)

            elapsed = time.time() - t0
            per_ep  = elapsed / (epoch + 1)
            remain  = per_ep * (epochs - epoch - 1)
            progress = 8 + int(90 * (epoch + 1) / epochs)

            _update_job(
                job,
                epochs_done = epoch + 1,
                losses      = losses,
                progress    = progress,
            )
            _log_job(job, f"Époque {epoch+1:02d}/{epochs}  loss={mean_loss:.4f}  ETA={remain:.0f}s", "TRAIN")

        model_dir = MODEL_DIR
        ia.save_model(model, model_dir)
        _log_job(job, f"Modèle sauvegardé → {model_dir}", "OK")
        _update_job(
            job,
            status   = JobStatus.DONE,
            ended_at = _now(),
            progress = 100,
            result   = {
                "model_dir":    str(model_dir),
                "final_loss":   float(losses[-1]) if losses else None,
                "n_examples":   len(examples),
                "device":       ia.DEVICE,
            },
        )
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


def _worker_infer(job: Job, session: str, params: dict, apply: bool, dry_run: bool):
    try:
        import utils.sync as ia

        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=5)
        _log_job(job, f"Inférence sur {Path(session).name}…")

        sess_p    = Path(session)
        model_dir = MODEL_DIR

        if not (model_dir / "model.pt").exists():
            raise RuntimeError("Aucun modèle trouvé. Entraînez d'abord le modèle.")

        model  = ia.load_model(model_dir)
        fluxes = ia.load_all_fluxes(sess_p, signal_config=params.get("signal_config"))

        estimates = []
        n_pairs   = len(ia.PAIRS)

        for i, (ref_name, tgt_name) in enumerate(ia.PAIRS):
            if ref_name not in fluxes or tgt_name not in fluxes:
                _log_job(job, f"Paire {ref_name}↔{tgt_name} ignorée (flux manquant)", "WARN")
                continue

            est = ia.estimate_pair_offset(
                model       = model,
                ref         = fluxes[ref_name],
                tgt         = fluxes[tgt_name],
                resample_ms = params["resample_ms"],
                max_lag_ms  = params["max_lag_ms"],
                window_ms   = params["window_ms"],
            )

            d = {
                "ref_name":          est.ref_name,
                "tgt_name":          est.tgt_name,
                "delta_start_ms":    float(est.delta_start_ms),
                "residual_ms":       float(est.residual_ms),
                "total_offset_ms":   float(est.total_offset_ms),
                "shift_to_apply_ms": float(est.shift_to_apply_ms),
                "confidence":        float(est.confidence),
                "peak_margin":       float(est.peak_margin),
                "best_score":        float(est.best_score),
                "second_score":      float(est.second_score),
                "is_reliable":       bool(est.is_reliable),
                "method":            est.method,
                "lags_ms":           est.lags_ms.tolist(),
                "scores":            est.scores.tolist(),
            }
            estimates.append(d)
            _update_job(job, progress=5 + int(90 * (i + 1) / n_pairs))
            _log_job(
                job,
                f"{ref_name} ↔ {tgt_name}  shift={est.shift_to_apply_ms:+.1f}ms  "
                f"conf={est.confidence:.3f}  reliable={est.is_reliable}",
                "OK" if est.is_reliable else "WARN",
            )

            # Broadcast résultat partiel immédiatement
            asyncio.run_coroutine_threadsafe(
                _broadcast({"type": "pair_result", "job_id": job.id, "estimate": d}),
                _loop,
            )

        if apply and not dry_run:
            applied = set()
            for ed in estimates:
                if not ed["is_reliable"] or ed["tgt_name"] in applied:
                    continue
                ia.apply_shift_to_target(sess_p, ed["tgt_name"], ed["shift_to_apply_ms"], dry_run=False)
                applied.add(ed["tgt_name"])
                _log_job(job, f"Offset appliqué: {ed['tgt_name']}  {ed['shift_to_apply_ms']:+.1f}ms", "OK")

        n_reliable = sum(1 for e in estimates if e["is_reliable"])
        mean_conf  = float(np.mean([e["confidence"] for e in estimates])) if estimates else 0.0

        _update_job(
            job,
            status   = JobStatus.DONE,
            ended_at = _now(),
            progress = 100,
            result   = {
                "session":    session,
                "estimates":  estimates,
                "n_reliable": n_reliable,
                "n_total":    len(estimates),
                "mean_conf":  mean_conf,
                "applied":    apply and not dry_run,
            },
        )
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "SyncML Studio API",
    description = "Alignement inter-flux deep learning — pipeline big data",
    version     = "3.0.0",
)


# Servir le frontend statique
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── Event loop référence (pour run_coroutine_threadsafe depuis threads) ──
_loop: asyncio.AbstractEventLoop = None

@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_running_loop()
    _load_jobs_from_disk()


# ──────────────────────────────────────────────────────────────────────────────
# Worker pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _poll_pipeline_state(job: "Job", session_path: str,  # type: ignore[name-defined]
                          stop_event: threading.Event) -> None:
    """Thread de polling : lit pipeline_state.json toutes les 2s et broadcaste les mises à jour."""
    from pipeline.pipeline import StepStatus, SessionPipelineState
    STEP_NAMES = ["detect", "check_sync", "verify", "rotate", "tracker", "video",
                  "verify_labels", "flux_csv", "ia_sync", "validate", "store"]
    sess = Path(session_path)
    last_current = None
    while not stop_event.is_set():
        try:
            state = SessionPipelineState.load(sess)
            if state is None:
                time.sleep(2)
                continue
            done = sum(
                1 for n in STEP_NAMES
                if state.steps.get(n) and state.steps[n].status
                   in (StepStatus.DONE, StepStatus.SKIPPED)
            )
            progress = int(done / len(STEP_NAMES) * 100)
            steps_serial = {
                n: {
                    "status":     str(state.steps[n].status) if n in state.steps else "pending",
                    "duration_s": state.steps[n].duration_s if n in state.steps else 0,
                    "message":    state.steps[n].message    if n in state.steps else "",
                }
                for n in STEP_NAMES
            }
            if state.current_step != last_current:
                last_current = state.current_step
                _update_job(job, progress=progress, result={
                    "session": state.session_name,
                    "steps":   steps_serial,
                    "current": state.current_step,
                })
                if _loop:
                    asyncio.run_coroutine_threadsafe(
                        _broadcast({"type": "pipeline_step", "job_id": job.id,
                                    "steps": steps_serial, "current": state.current_step,
                                    "finished": state.finished,
                                    "success": getattr(state, "success", False)}),
                        _loop,
                    )
        except Exception:
            pass
        time.sleep(2)


def _run_pipeline_in_process(source_path: str, params: dict,
                              write_mode: bool, force_flux: bool,
                              delete_after_store: bool,
                              steps_whitelist: Optional[List[str]]) -> dict:
    """Fonction top-level picklable — exécutée dans un processus séparé (ProcessPoolExecutor).
    Retourne un dict sérialisable avec logs, succès et erreur."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from pipeline.pipeline import PipelineRunner

    collected_logs: list = []

    def _log_cb(msg: str, level: str = "INFO") -> None:
        collected_logs.append({"msg": msg, "level": level, "ts": _now()})

    runner = PipelineRunner(
        source_path        = source_path,
        params             = params,
        write_mode         = write_mode,
        delete_after_store = delete_after_store,
        log_callback       = _log_cb,
        step_callback      = None,
        force_flux         = force_flux,
        resume             = True,
        steps_whitelist    = steps_whitelist,
    )
    state = runner.run()
    return {
        "success":  state.success,
        "error":    state.error,
        "logs":     collected_logs,
        "steps":    {n: s.status for n, s in state.steps.items()},
    }


# Pool de processus partagé pour le batch (max = nb CPU - 1, min = 2)
_PROCESS_POOL: Optional[concurrent.futures.ProcessPoolExecutor] = None
_PROCESS_POOL_LOCK = threading.Lock()
# Futures actifs : job_id → Future (pour annulation)
_POOL_FUTURES: Dict[str, concurrent.futures.Future] = {}
_POOL_FUTURES_LOCK = threading.Lock()


def _get_process_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _PROCESS_POOL
    with _PROCESS_POOL_LOCK:
        if _PROCESS_POOL is None:
            n_workers = max(2, (os.cpu_count() or 4) - 1)
            _PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
    return _PROCESS_POOL


def _worker_pipeline(job: "Job", source_path: str, params: dict,  # type: ignore[name-defined]
                     write_mode: bool, force_flux: bool,
                     delete_after_store: bool = False,
                     steps_whitelist: Optional[List[str]] = None):
    try:
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=2)
        if not _session_writable(source_path):
            raise PermissionError(
                f"Le dossier session n'est pas accessible en écriture : {source_path}\n"
                "Vérifiez les permissions (chmod 777) ou le montage NFS."
            )

        # Démarrer le polling WebSocket pendant que le process tourne
        stop_poll = threading.Event()
        threading.Thread(
            target=_poll_pipeline_state,
            args=(job, source_path, stop_poll),
            daemon=True,
        ).start()

        # Déléguer au process pool pour libérer le GIL sur le CPU-bound
        pool = _get_process_pool()
        fut  = pool.submit(
            _run_pipeline_in_process,
            source_path, params, write_mode, force_flux,
            delete_after_store, steps_whitelist,
        )
        with _POOL_FUTURES_LOCK:
            _POOL_FUTURES[job.id] = fut
        try:
            result = fut.result()  # bloque ce thread (pas le GIL global)
        finally:
            stop_poll.set()
            with _POOL_FUTURES_LOCK:
                _POOL_FUTURES.pop(job.id, None)

        # Rejouer les logs collectés dans le process
        for entry in result.get("logs", []):
            _log_job(job, entry["msg"], entry.get("level", "INFO"))

        final_status = JobStatus.DONE if result["success"] else JobStatus.ERROR
        _update_job(
            job,
            status   = final_status,
            ended_at = _now(),
            progress = 100 if result["success"] else job.progress,
            error    = result.get("error"),
        )

    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


# ──────────────────────────────────────────────────────────────────────────────
# Modèles Pydantic
# ──────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    sessions:      List[str]
    epochs:        int   = 18
    batch_size:    int   = 64
    lr:            float = 1e-3
    resample_ms:   float = 5.0
    max_lag_ms:    float = 400.0
    window_ms:     float = 2200.0
    signal_config: Optional[Dict[str, List[str]]] = None

class InferRequest(BaseModel):
    session:       str        # chemin dans /mnt/storage/silver//
    apply:         bool  = False
    dry_run:       bool  = True
    resample_ms:   float = 5.0
    max_lag_ms:    float = 400.0
    window_ms:     float = 2200.0
    signal_config: Optional[Dict[str, List[str]]] = None

class PipelineRunRequest(BaseModel):
    session:              str          # nom ou chemin de session dans /mnt/storage/silver//
    write_mode:           bool  = False
    delete_after_store:   bool  = False
    force_flux:           bool  = False
    resample_ms:          float = 5.0
    max_lag_ms:           float = 400.0
    window_ms:            float = 2200.0
    steps:                Optional[List[str]] = None  # None = toutes les étapes

class PipelineStateRequest(BaseModel):
    session_path: str


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(_static_dir / "index.html"))


def _worker_stats(job: Job):
    """Scanne bronze/silver/gold/rejected et agrège les stats par scénario."""
    BASE   = Path("/mnt/storage")
    TIERS  = ["bronze", "silver", "gold", "rejected"]
    try:
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=5)
        result: dict = {}
        for i, tier in enumerate(TIERS):
            tier_dir   = BASE / tier
            tier_data: dict = {}
            total_count   = 0
            total_seconds = 0.0
            if tier_dir.exists() and tier_dir.is_dir():
                try:
                    entries = [
                        d for d in tier_dir.iterdir()
                        if d.is_dir()
                        and not d.name.startswith(".")
                        and not d.name.startswith("_")
                    ]
                except PermissionError:
                    entries = []
                _log_job(job, f"[{tier}] {len(entries)} dossiers trouvés")
                for sess in sorted(entries, key=lambda p: p.name):
                    meta: dict = {}
                    meta_path = sess / "metadata.json"
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text())
                        except Exception:
                            pass
                    scenario = meta.get("scenario") or "(sans scénario)"
                    duration = float(meta.get("duration_seconds") or 0)
                    if scenario not in tier_data:
                        tier_data[scenario] = {"count": 0, "seconds": 0.0}
                    tier_data[scenario]["count"]   += 1
                    tier_data[scenario]["seconds"] += duration
                    total_count   += 1
                    total_seconds += duration
            result[tier] = {
                "total_count":   total_count,
                "total_seconds": round(total_seconds, 2),
                "by_scenario":   {
                    sc: {"count": v["count"], "seconds": round(v["seconds"], 2)}
                    for sc, v in sorted(tier_data.items())
                },
            }
            _update_job(job, progress=5 + int(90 * (i + 1) / len(TIERS)))
        _update_job(job, status=JobStatus.DONE, ended_at=_now(), progress=100, result=result)
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


@app.post("/api/stats")
async def stats_overview():
    """Lance un scan asynchrone de bronze/silver/gold/rejected. Retourne job_id."""
    job = _new_job("stats")
    threading.Thread(target=_worker_stats, args=(job,), daemon=True).start()
    return {"job_id": job.id}


@app.get("/api/health")
async def health():
    """Health check pour orchestrateurs (Kubernetes, Airflow, etc.)."""
    try:
        import utils.sync as ia
        import torch
        ia_ok  = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as e:
        ia_ok  = False
        device = "unavailable"
    return {
        "status":    "ok",
        "ts":        _now(),
        "ia_loaded": ia_ok,
        "device":    device,
        "jobs":      len(_jobs),
    }


@app.post("/api/scan")
async def scan(req: dict = None):
    """Lance un scan asynchrone.
    req.root         = répertoire racine à scanner (défaut : INGEST_DIR)
    req.input_format = "custom" (défaut) | "lerobot"
    req.lerobot_path = chemin vers le dataset LeRobot (si input_format == "lerobot")
    req.limit  = nombre max de sessions à retourner (défaut 500, 0 = illimité)
    req.offset = index de départ pour la pagination (défaut 0)
    """
    if req is None:
        req = {}
    input_format = req.get("input_format", "custom")
    limit       = int(req.get("limit",  500))
    offset      = int(req.get("offset", 0))
    root        = req.get("root", "") or str(INGEST_DIR)
    date_filter = req.get("date_filter", "").strip()  # "YYYYMMDD" ou ""
    job = _new_job("scan")
    if input_format == "lerobot":
        lerobot_path = req.get("lerobot_path", "") or root
        threading.Thread(target=_worker_scan_lerobot, args=(job, lerobot_path, limit, offset), daemon=True).start()
    else:
        threading.Thread(target=_worker_scan, args=(job, limit, offset, root, date_filter), daemon=True).start()
    return {"job_id": job.id}


def _worker_scan_lerobot(job: Job, dataset_path: str, limit: int = 500, offset: int = 0):
    """Scan d'un dataset au format LeRobot v3.

    Structure attendue :
      dataset_path/
        meta/
          info.json          ← informations globales du dataset
          episodes/chunk-000/file-000.parquet  ← métadonnées par épisode
        data/
          chunk-000/file-000.parquet  ← données par épisode (un fichier = un épisode)
        videos/
          observation.images.head/chunk-000/file-000.mp4
          ...

    Chaque fichier data/chunk-*/file-*.parquet correspond à un épisode,
    retourné comme une "session" pour le reste de l'UI.
    """
    try:
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=5)

        root = Path(dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")

        # Lire meta/info.json
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"meta/info.json introuvable dans {dataset_path} — pas un dataset LeRobot valide")

        info = json.loads(info_path.read_text(encoding="utf-8"))
        _log_job(job, f"Dataset LeRobot : {info.get('dataset_type','?')} v{info.get('codebase_version','?')}", "INFO")
        _log_job(job, f"  total_episodes={info.get('total_episodes','?')}  fps={info.get('fps','?')}", "INFO")

        video_keys = info.get("video_keys", []) or list(info.get("videos", {}).keys())
        features   = info.get("features", {})
        robot_type = info.get("robot_type", "—")
        fps        = info.get("fps", 0)

        # Lister tous les fichiers de données (un parquet = un épisode)
        data_dir = root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Dossier data/ introuvable dans {dataset_path}")

        data_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"Aucun fichier data/chunk-*/file-*.parquet dans {dataset_path}")

        total_found = len(data_files)
        _log_job(job, f"{total_found} épisode(s) trouvé(s)", "OK")
        if offset:
            data_files = data_files[offset:]
        if limit:
            data_files = data_files[:limit]
        _log_job(job, f"Chargement de {len(data_files)} épisode(s) (offset={offset}, limite={limit})", "INFO")
        _update_job(job, progress=20)

        # Essayer de charger les métadonnées d'épisodes depuis meta/episodes/
        ep_meta: dict[int, dict] = {}
        ep_meta_dir = root / "meta" / "episodes"
        if ep_meta_dir.exists():
            try:
                import pandas as pd
                for ep_file in sorted(ep_meta_dir.glob("chunk-*/file-*.parquet")):
                    df = pd.read_parquet(ep_file)
                    for _, row in df.iterrows():
                        idx = int(row.get("episode_index", -1))
                        if idx >= 0:
                            ep_meta[idx] = row.to_dict()
            except Exception as e:
                _log_job(job, f"meta/episodes non chargé : {e}", "WARN")

        result = []
        for i, data_file in enumerate(data_files):
            # Extraire chunk_idx et file_idx depuis le chemin
            parts = data_file.parts
            chunk_part = parts[-2]   # "chunk-000"
            file_part  = parts[-1]   # "file-000.parquet"
            chunk_idx  = int(chunk_part.split("-")[1])
            file_idx   = int(file_part.split("-")[1].split(".")[0])
            ep_idx     = chunk_idx * len(data_files) + file_idx  # approximation si chunks_size inconnu

            # Trouver l'index réel depuis ep_meta si disponible
            # (on cherche l'épisode dont dataset_from_index correspond à ce fichier)
            ep_info = ep_meta.get(file_idx + chunk_idx * info.get("chunks_size", 1000), {})
            if not ep_info:
                ep_info = ep_meta.get(file_idx, {})

            # Lire quelques stats depuis le parquet (léger : juste shape)
            n_frames = 0
            duration_s = 0.0
            try:
                import pandas as pd
                df = pd.read_parquet(data_file, columns=["timestamp"] if "timestamp" in features else None)
                n_frames = len(df)
                if "timestamp" in df.columns:
                    ts = df["timestamp"].to_numpy(dtype=float)
                    duration_s = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            except Exception:
                pass

            # Vidéos disponibles pour cet épisode
            available_videos = {}
            for vk in video_keys:
                vid = root / "videos" / vk / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
                available_videos[vk] = vid.exists()

            session_name = f"episode_{file_idx:04d}" if not ep_info.get("task_index") else f"episode_{file_idx:04d}"
            tasks_list   = ep_info.get("tasks", [])

            result.append({
                "name":            session_name,
                "path":            str(data_file),          # chemin vers le fichier parquet
                "lerobot":         True,                    # marqueur format
                "dataset_root":    str(root),
                "chunk_idx":       chunk_idx,
                "file_idx":        file_idx,
                "n_frames":        n_frames,
                "duration_s":      round(duration_s, 2),
                "fps":             fps,
                "robot_type":      robot_type,
                "video_keys":      video_keys,
                "available_videos": available_videos,
                "tasks":           tasks_list,
                "meta": {
                    "scenario":          f"chunk-{chunk_idx:03d}/file-{file_idx:03d}",
                    "duration_seconds":  round(duration_s, 2),
                    "fps":               fps,
                    "robot_type":        robot_type,
                    "n_frames":          n_frames,
                },
                # Champs attendus par l'UI existante
                "has_tracker":    False,
                "has_gripper":    False,
                "has_ux":         False,
                "has_flux_csv":   False,
                "has_jsonl":      False,
                "has_subtitle":   False,
                "pipeline_done":  False,
                "pipeline_steps": {},
                "last_result":    None,
            })

            progress = 20 + int(75 * (i + 1) / max(len(data_files), 1))
            _update_job(job, progress=progress)

        has_more = bool(limit and (offset + len(result)) < total_found)
        _log_job(job, f"{len(result)} épisodes chargés depuis {root}", "OK")
        _update_job(
            job,
            status    = JobStatus.DONE,
            ended_at  = _now(),
            progress  = 100,
            result    = {
                "sessions":      result,
                "ingest_dir":    str(root),
                "silver_dir":    str(SILVER_DIR),
                "model_exists":  (MODEL_DIR / "model.pt").exists(),
                "input_format":  "lerobot",
                "lerobot_info":  info,
                "total_found":   total_found,
                "offset":        offset,
                "limit":         limit,
                "has_more":      has_more,
            },
        )
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


@app.get("/api/paths")
async def get_paths():
    """Retourne les chemins hardcodés de la pipeline."""
    return {
        "ingest_dir":  str(INGEST_DIR),
        "silver_dir":  str(SILVER_DIR),
        "model_dir":   str(MODEL_DIR),
    }


@app.post("/api/train")
async def train(req: TrainRequest):
    """Lance un entraînement asynchrone sur les sessions de /mnt/storage/silver//."""
    params = {
        "epochs":        req.epochs,
        "batch_size":    req.batch_size,
        "lr":            req.lr,
        "resample_ms":   req.resample_ms,
        "max_lag_ms":    req.max_lag_ms,
        "window_ms":     req.window_ms,
        "signal_config": req.signal_config,
    }
    job = _new_job("train")
    threading.Thread(
        target=_worker_train,
        args=(job, req.sessions, params),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@app.post("/api/infer")
async def infer(req: InferRequest):
    """Lance une inférence asynchrone sur une session de /mnt/storage/silver//."""
    params = {
        "resample_ms":   req.resample_ms,
        "max_lag_ms":    req.max_lag_ms,
        "window_ms":     req.window_ms,
        "signal_config": req.signal_config,
    }
    job = _new_job("infer")
    threading.Thread(
        target=_worker_infer,
        args=(job, req.session, params, req.apply, req.dry_run),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@app.get("/api/jobs")
async def list_jobs():
    """Liste tous les jobs avec leur statut."""
    with _jobs_lock:
        return [_job_to_dict(j) for j in reversed(list(_jobs.values()))]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} introuvable")
    return _job_to_dict(job)


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, offset: int = 0):
    """Récupère les logs d'un job (pagination)."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} introuvable")
    return {"logs": job.logs[offset:], "total": len(job.logs)}


@app.post("/api/pipeline/run")
async def pipeline_run(req: PipelineRunRequest):
    """Lance la pipeline complète (9 étapes) sur une session de /mnt/storage/silver//."""
    params = {
        "resample_ms": req.resample_ms,
        "max_lag_ms":  req.max_lag_ms,
        "window_ms":   req.window_ms,
    }
    # req.session peut être un nom ou un chemin complet dans /mnt/storage/silver//
    source_path = req.session if Path(req.session).is_absolute() else str(INGEST_DIR / req.session)

    job = _new_job("pipeline")
    threading.Thread(
        target = _worker_pipeline,
        args   = (job, source_path, params, req.write_mode, req.force_flux,
                  req.delete_after_store, req.steps),
        daemon = True,
    ).start()
    return {"job_id": job.id}


@app.post("/api/pipeline/run_batch")
async def pipeline_run_batch(req: dict):
    """Lance la pipeline sur plusieurs sessions de /mnt/storage/silver// en parallèle."""
    sessions           = req.get("sessions", [])
    write_mode         = req.get("write_mode", False)
    delete_after_store = req.get("delete_after_store", False)
    force_flux         = req.get("force_flux", False)
    params = {
        "resample_ms": req.get("resample_ms", 5.0),
        "max_lag_ms":  req.get("max_lag_ms", 400.0),
        "window_ms":   req.get("window_ms", 2200.0),
    }

    job_ids = []
    for sess in sessions:
        source_path = sess if Path(sess).is_absolute() else str(INGEST_DIR / sess)
        job = _new_job("pipeline")
        threading.Thread(
            target = _worker_pipeline,
            args   = (job, source_path, params, write_mode, force_flux, delete_after_store),
            daemon = True,
        ).start()
        job_ids.append(job.id)

    return {"job_ids": job_ids, "count": len(job_ids)}


@app.get("/api/pipeline/state")
async def pipeline_state(session_path: str):
    """Retourne l'état pipeline persisté d'une session."""
    from pipeline.pipeline import SessionPipelineState
    state = SessionPipelineState.load(Path(session_path))
    if state is None:
        raise HTTPException(404, "Aucun état pipeline trouvé pour cette session")
    return state.to_dict()


@app.post("/api/pipeline/rollback")
async def pipeline_rollback(req: PipelineStateRequest):
    """Force un rollback sur une session (restaure les .bak_syncml)."""
    from pipeline.pipeline import _restore_bak_files, _restore_backup, SessionPipelineState, PIPELINE_LOCK_FILE
    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")

    # Supprimer le lock file s'il existe (pipeline interrompue)
    (sess / PIPELINE_LOCK_FILE).unlink(missing_ok=True)

    # Essayer d'abord le backup complet
    ok = _restore_backup(sess, "ia_sync")
    if not ok:
        # Fallback sur les .bak_syncml
        _restore_bak_files(sess)

    # Mettre à jour l'état
    state = SessionPipelineState.load(sess)
    if state:
        from pipeline.pipeline import StepStatus
        for step_name in ["ia_sync", "validate", "store"]:
            if step_name in state.steps:
                state.steps[step_name].status  = StepStatus.ROLLED_BACK
                state.steps[step_name].message = "Rollback manuel"
        state.finished = False
        state.success  = False
        state.save()

    return {"rolled_back": True, "session": str(sess.name)}


@app.post("/api/pipeline/unlock")
async def pipeline_unlock(req: PipelineStateRequest):
    """Supprime le fichier de lock d'une session sans rollback."""
    from pipeline.pipeline import PIPELINE_LOCK_FILE
    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")
    lock = sess / PIPELINE_LOCK_FILE
    existed = lock.exists()
    lock.unlink(missing_ok=True)
    return {"unlocked": existed, "session": str(sess.name)}


@app.post("/api/pipeline/check_sync")
async def pipeline_check_sync(req: PipelineStateRequest):
    """
    Lance uniquement l'étape check_sync sur une session (léger, pas de lock).
    Retourne immédiatement le résultat sans passer par la queue de jobs.
    """
    from pipeline.pipeline import step_check_sync, SessionPipelineState, PipelineLogger
    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")
    state = SessionPipelineState.load(sess)
    if state is None:
        state = SessionPipelineState(
            session_name = sess.name,
            session_path = str(sess),
        )
    logs: list = []
    log = PipelineLogger(sess.name, lambda msg, level: logs.append({"msg": msg, "level": level}))
    try:
        result = step_check_sync(state, log)
        return {"status": "ok", "result": result, "logs": logs}
    except ValueError as e:
        return {"status": "decalage", "error": str(e), "logs": logs}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/pipeline/verify")
async def pipeline_verify(req: PipelineStateRequest):
    """
    Lance l'étape verify (trackers + désynchronisation) sur une session.
    Écrit les résultats dans metadata.json["verification"] et retourne
    immédiatement le bilan sans passer par la queue de jobs.
    """
    from pipeline.pipeline import step_verify, SessionPipelineState, PipelineLogger
    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")
    state = SessionPipelineState.load(sess)
    if state is None:
        state = SessionPipelineState(
            session_name = sess.name,
            session_path = str(sess),
        )
    logs: list = []
    log = PipelineLogger(sess.name, lambda msg, level: logs.append({"msg": msg, "level": level}))
    try:
        result = step_verify(state, log)
        status = "issues" if result.get("has_issues") else "ok"
        return {"status": status, "result": result, "logs": logs}
    except ValueError as e:
        return {"status": "critical", "error": str(e), "logs": logs}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/pipeline/check")
async def pipeline_check(req: PipelineStateRequest):
    """Lance check_session sur une session (diagnostic seul, sans correction)."""
    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")
    job = _new_job("check")
    threading.Thread(target=_worker_check_only, args=(job, sess), daemon=True).start()
    return {"job_id": job.id}


@app.post("/api/pipeline/check_score")
async def pipeline_check_score(req: PipelineStateRequest):
    """Vérification complète d'une session (sans job, résultat immédiat).

    Orchestre 3 checks indépendants via verification/session_check.py :
      1. Synchronisation vidéo/tracker (IA — check.py)
      2. Synchronisation gripper CSV ↔ timestamps caméra (cross-corrélation)
      3. Placement trackers head/left/right (trakeur.py)

    Écrit les résultats dans metadata.json et retourne le rapport complet.
    """
    import asyncio
    import importlib.util as _ilu

    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")

    loop = asyncio.get_event_loop()

    def _run():
        import sys as _sys
        spec = _ilu.spec_from_file_location(
            "session_check", _ROOT / "verification" / "session_check.py"
        )
        mod = _ilu.module_from_spec(spec)
        _sys.modules["session_check"] = mod   # requis pour que les dataclasses résolvent leur module
        spec.loader.exec_module(mod)
        return mod.check_session_full(sess)

    result = await loop.run_in_executor(None, _run)

    tracker_auto_fixed = False
    tracker_fix_mapping: dict = {}

    score      = result["score"]
    grade      = result.get("grade", "?")
    verdict    = result.get("verdict", "")
    is_perfect = result["perfect"]
    blocked    = result["blocked"]
    components = result["components"]
    confidence = result.get("confidence", 0.0)
    repairs    = result.get("repairs", [])

    # ── Extraire les sous-composantes pour la rétro-compatibilité ──────────
    vt = components.get("video_tracker_sync", {})
    tp_details = components.get("tracker_placement", {}).get("details", {})
    gt = components.get("gripper_timestamp_sync", {})
    gt_sides = gt.get("details", {}).get("sides", {})

    # ── Persister dans metadata.json ─────────────────────────────────────
    meta_path = sess / "metadata.json"
    meta_updates = {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta["check_score"]      = score
        meta["check_grade"]      = grade
        meta["check_verdict"]    = verdict
        meta["check_confidence"] = confidence
        meta["check_blocking"]   = result.get("blocking_reason", "")
        meta["check_repairs"]    = repairs
        meta["check_components"] = {
            k: {"score": v.get("score"), "grade": v.get("grade"),
                "summary": v.get("summary"), "blocking": v.get("blocking")}
            for k, v in components.items()
        }
        meta["repair_perfect"]       = is_perfect
        meta["repair_unrecoverable"] = not is_perfect
        meta["repair_score"]         = score
        if not is_perfect:
            meta["repair_failure_reason"] = result.get("blocking_reason") or f"score={score:.0f}% grade={grade}"
        meta_updates = {
            "repair_perfect":       is_perfect,
            "repair_unrecoverable": not is_perfect,
            "repair_score":         score,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return {
        "status":           "ok" if is_perfect else ("issues" if score >= 40 else "fail"),
        "score":            score,
        "grade":            grade,
        "verdict":          verdict,
        "confidence":       confidence,
        "ia_score":         vt.get("details", {}).get("ia_score") or result.get("ia_score", 0.0),
        "blocking_reason":  result.get("blocking_reason", ""),
        "failed_gates":     vt.get("details", {}).get("failed_gates", []),
        "repair_perfect":   is_perfect,
        "tracker_ok":       tp_details.get("ok"),
        "tracker_result":   tp_details,
        "gripper_sync": {
            "left":  gt_sides.get("left", {}),
            "right": gt_sides.get("right", {}),
        },
        "repairs":              repairs,
        "components":           components,
        "tracker_auto_fixed":   tracker_auto_fixed,
        "tracker_fix_mapping":  tracker_fix_mapping,
        **meta_updates,
    }


def _worker_check_only(job: "Job", sess: Path):
    """Worker thread : check_session seul, sans appliquer de correctif."""
    try:
        from check import check_session
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=10)
        _log_job(job, f"Diagnostic {sess.name}…", "INFO")

        report = check_session(sess, model=None)

        failed_gates = [g for g in report.gates if not g.passed]
        for g in failed_gates:
            _log_job(job, f"  ✗ {g.name}: {g.message}", "ERROR")

        score_pct = report.score
        log_level = "OK" if score_pct >= 70 else ("WARN" if score_pct >= 40 else "ERROR")
        _log_job(job, f"Score : {score_pct:.0f}%  (IA={report.ia_score:.3f})", log_level)
        _update_job(
            job,
            status=JobStatus.DONE,
            ended_at=_now(),
            progress=100,
            result={
                "session":  str(sess),
                "score":    score_pct,
                "ia_score": report.ia_score,
                "verdict":  report.verdict,
                "gates":    [{"name": g.name, "passed": g.passed, "message": g.message} for g in report.gates],
                "blocking_reason": report.blocking_reason,
            },
        )
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


@app.post("/api/pipeline/align_pro")
async def pipeline_align_pro(req: PipelineStateRequest):
    """
    Lance pipeline_align_pro sur une session en tâche de fond.
    Alignement multi-résolution professionnel (gross→fine→subpixel + consensus).
    Retourne un job_id pour suivre la progression.
    """
    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")

    force = getattr(req, "force", False)
    job = _new_job("align_pro")
    threading.Thread(
        target=_worker_align_pro,
        args=(job, sess, force),
        daemon=True,
    ).start()
    return {"job_id": job.id}


def _worker_align_pro(job: "Job", sess: Path, force: bool = False):
    """Worker thread : check.py → diagnostic + fix_camera_offset si nécessaire."""
    try:
        from check import check_session
        import importlib.util

        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=5)
        _log_job(job, f"Check démarré sur {sess.name}", "INFO")

        # ── Étape 1 : diagnostic via check_session ────────────────────────
        report = check_session(sess, model=None)

        failed_gates = [g for g in report.gates if not g.passed]
        for g in failed_gates:
            _log_job(job, f"  ✗ {g.name}: {g.message}", "ERROR")

        _log_job(job, f"Score check : {report.score:.0f}%  (IA={report.ia_score:.3f})",
                 "OK" if report.score >= 70 else ("WARN" if report.score >= 40 else "ERROR"))
        _update_job(job, progress=40)

        # ── Étape 2 : corriger l'offset caméra si nécessaire ─────────────
        # Détecte un offset via la pénalité : si score faible mais portes OK
        # → l'offset est peut-être le problème. On tente fix_camera_offset.
        # On s'appuie aussi sur force.
        has_offset_issue = (report.score < 70 and not report.is_blocked())
        offset_errors = []  # compat variable name

        fix_result = None
        if has_offset_issue or offset_errors or force:
            fix_script = _ROOT / "fix_camera_offset.py"
            if fix_script.exists():
                spec = importlib.util.spec_from_file_location("fix_camera_offset", fix_script)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                _log_job(job, "Application fix_camera_offset…", "INFO")
                fix_result = mod.fix_session(sess, dry_run=False, force=force)
                status = fix_result.get("status", "?")
                if status == "corrected":
                    cams = fix_result.get("cameras_fixed", [])
                    for c in cams:
                        _log_job(job, f"  ✓ {c['camera']}: {c['offset_ms']:.0f} ms corrigé ({c['frames']} frames)", "OK")
                    _log_job(job, f"Offset caméra corrigé sur {len(cams)} caméra(s)", "OK")
                else:
                    reason = fix_result.get("reason", "")
                    _log_job(job, f"fix_camera_offset [{status}]: {reason}", "WARN")
            else:
                _log_job(job, "fix_camera_offset.py introuvable", "WARN")
        else:
            _log_job(job, "Aucun offset à corriger", "INFO")

        _update_job(job, progress=80)

        # ── Étape 3 : re-check post-correction ───────────────────────────
        if fix_result and fix_result.get("status") == "corrected":
            report2 = check_session(sess, model=None)
            _log_job(job, f"Score post-correction : {report2.score:.0f}%",
                     "OK" if report2.score >= 70 else ("WARN" if report2.score >= 40 else "ERROR"))
            final_score = report2.score
            final_ia    = report2.ia_score
            final_verdict = report2.verdict
            final_gates = [{"name": g.name, "passed": g.passed, "message": g.message} for g in report2.gates]
            final_blocking = report2.blocking_reason
        else:
            final_score = report.score
            final_ia    = report.ia_score
            final_verdict = report.verdict
            final_gates = [{"name": g.name, "passed": g.passed, "message": g.message} for g in report.gates]
            final_blocking = report.blocking_reason

        _update_job(
            job,
            status=JobStatus.DONE,
            ended_at=_now(),
            progress=100,
            result={
                "session":        str(sess),
                "score":          final_score,
                "ia_score":       final_ia,
                "verdict":        final_verdict,
                "gates":          final_gates,
                "blocking_reason": final_blocking,
                "fix_applied":    fix_result is not None and fix_result.get("status") == "corrected",
                "fix_result":     fix_result,
            },
        )
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


@app.delete("/api/jobs")
async def clear_jobs():
    """Vide les jobs terminés (mémoire + disque)."""
    with _jobs_lock:
        done = [k for k, v in _jobs.items() if v.status in (JobStatus.DONE, JobStatus.ERROR)]
        for k in done:
            del _jobs[k]
            _job_file(k).unlink(missing_ok=True)
    return {"cleared": len(done)}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Supprime un job individuel (tous statuts)."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} introuvable")
        del _jobs[job_id]
        _job_file(job_id).unlink(missing_ok=True)
    await _broadcast({"type": "job_deleted", "job_id": job_id})
    return {"deleted": job_id}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Annule un job en cours (marque ERROR + annule le future si possible)."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} introuvable")
        if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            raise HTTPException(400, f"Job {job_id} déjà terminé (status={job.status})")
        job.status   = JobStatus.ERROR
        job.ended_at = _now()
        job.error    = "Annulé par l'utilisateur"
        _persist_job(job)
    await _broadcast({"type": "job_update", "job": _job_to_dict(job)})
    # Annuler le future dans le process pool si possible
    fut = _POOL_FUTURES.get(job_id)
    if fut:
        fut.cancel()
        _POOL_FUTURES.pop(job_id, None)
    return {"cancelled": job_id}


@app.get("/api/pool/status")
async def pool_status():
    """Retourne l'état du pool de processus et des threads actifs."""
    pool = _PROCESS_POOL
    pool_info = {}
    if pool is not None:
        pool_info = {
            "max_workers":  pool._max_workers,
            "processes":    len(pool._processes),
            "pending_work": pool._work_queue.qsize() if hasattr(pool, "_work_queue") else None,
        }
    active_threads = [
        {"name": t.name, "daemon": t.daemon, "alive": t.is_alive()}
        for t in threading.enumerate()
        if t.name not in ("MainThread",) and not t.name.startswith("asyncio")
    ]
    with _jobs_lock:
        running = [_job_to_dict(j) for j in _jobs.values() if j.status == JobStatus.RUNNING]
        pending = [_job_to_dict(j) for j in _jobs.values() if j.status == JobStatus.PENDING]
    return {
        "pool":           pool_info,
        "active_threads": active_threads,
        "n_threads":      len(active_threads),
        "running_jobs":   running,
        "pending_jobs":   pending,
    }


@app.post("/api/pool/reset")
async def pool_reset():
    """Arrête le pool de processus existant et en crée un nouveau."""
    global _PROCESS_POOL
    _POOL_FUTURES.clear()
    with _PROCESS_POOL_LOCK:
        old = _PROCESS_POOL
        _PROCESS_POOL = None
    if old is not None:
        try:
            old.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    # Marquer les jobs RUNNING comme annulés
    with _jobs_lock:
        for job in _jobs.values():
            if job.status == JobStatus.RUNNING:
                job.status   = JobStatus.ERROR
                job.ended_at = _now()
                job.error    = "Pool réinitialisé par l'utilisateur"
                _persist_job(job)
    await _broadcast({"type": "pool_reset"})
    return {"reset": True}


class PoolConfigRequest(BaseModel):
    workers: int
    threads: int


@app.post("/api/pool/configure")
async def pool_configure(req: PoolConfigRequest):
    """Recrée le pool avec le nombre de workers (processus) et threads demandé."""
    global _PROCESS_POOL
    n_workers = max(1, min(req.workers, os.cpu_count() or 1))
    n_threads = max(1, min(req.threads, 256))
    _POOL_FUTURES.clear()
    with _PROCESS_POOL_LOCK:
        old = _PROCESS_POOL
        _PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
    if old is not None:
        try:
            old.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    # Marquer les jobs RUNNING comme annulés (pool changé sous leurs pieds)
    with _jobs_lock:
        for job in _jobs.values():
            if job.status == JobStatus.RUNNING:
                job.status   = JobStatus.ERROR
                job.ended_at = _now()
                job.error    = "Pool reconfiguré par l'utilisateur"
                _persist_job(job)
    await _broadcast({"type": "pool_reset"})
    return {"workers": n_workers, "threads": n_threads}


# ──────────────────────────────────────────────────────────────────────────────
# Signal config — colonnes CSV disponibles par flux
# ──────────────────────────────────────────────────────────────────────────────

def _get_csv_columns(session_path: str) -> dict:
    """
    Retourne les colonnes numériques disponibles dans chaque CSV de la session,
    organisées par flux : tracker, gripper_left, gripper_right, cam_head, cam_left, cam_right.
    Exclut les colonnes de timestamp et les colonnes non-numériques.
    """
    import pandas as pd

    sess = Path(session_path)
    result = {}

    TIMESTAMP_COLS = {"timestamp_ns", "t_ms_corrected_ns", "time_seconds",
                      "timestamp_abs_ms", "t_ms", "time_ms", "time", "t",
                      "capture_time", "index"}

    def _numeric_cols(df: "pd.DataFrame") -> List[str]:
        cols = []
        for c in df.columns:
            if c.lower() in TIMESTAMP_COLS:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        return cols

    # tracker_positions.csv → flux tracker_head / tracker_left / tracker_right
    trk_path = sess / "tracker_positions.csv"
    if trk_path.exists():
        df = pd.read_csv(trk_path, nrows=5)
        cols = _numeric_cols(df)
        for pos in ("head", "left", "right"):
            result[f"tracker_{pos}"] = cols

    # gripper_left_data.csv / gripper_right_data.csv
    for side in ("left", "right"):
        grip_path = sess / f"gripper_{side}_data.csv"
        if grip_path.exists():
            df = pd.read_csv(grip_path, nrows=5)
            result[f"gripper_{side}"] = _numeric_cols(df)

    # videos/*_flux.csv
    vid_dir = sess / "videos"
    if vid_dir.exists():
        for cam in ("head", "left", "right"):
            flux_path = vid_dir / f"{cam}_flux.csv"
            if flux_path.exists():
                df = pd.read_csv(flux_path, nrows=5)
                result[f"cam_{cam}"] = _numeric_cols(df)

    return result


@app.get("/api/session/csv_columns")
async def session_csv_columns(session_path: str):
    """
    Retourne les colonnes numériques de chaque CSV de la session,
    utilisables comme signal d'alignement.
    """
    sess = Path(session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")
    try:
        return JSONResponse(content=_get_csv_columns(session_path))
    except Exception as e:
        raise HTTPException(500, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation — données temporelles alignées
# ──────────────────────────────────────────────────────────────────────────────

def _compute_session_start_ns(session_path: str) -> int:
    """
    Calcule le vrai t=0 de la session : minimum des premiers timestamps de
    tous les flux disponibles (tracker, grippers, vidéos JSONL).
    Ne se base PAS sur start_time_ns du metadata.
    Retourne un timestamp en nanosecondes epoch.
    """
    import pandas as _pd
    sess = Path(session_path)
    candidates: list = []

    trk_path = sess / "tracker_positions.csv"
    if trk_path.exists():
        try:
            df = _pd.read_csv(trk_path, usecols=["timestamp_ns"], nrows=1)
            if "timestamp_ns" in df.columns:
                candidates.append(float(df["timestamp_ns"].iloc[0]))
        except Exception:
            pass

    for side in ("left", "right"):
        grip_path = sess / f"gripper_{side}_data.csv"
        if grip_path.exists():
            try:
                df = _pd.read_csv(grip_path, nrows=1)
                if "timestamp_ns" in df.columns:
                    candidates.append(float(df["timestamp_ns"].iloc[0]))
                elif "t_ms_corrected_ns" in df.columns:
                    candidates.append(float(df["t_ms_corrected_ns"].iloc[0]))
            except Exception:
                pass

    vid_dir = sess / "videos"
    if vid_dir.exists():
        for side in ("head", "left", "right"):
            jl = vid_dir / f"{side}.jsonl"
            if jl.exists():
                rows = _parse_jsonl(jl)
                if rows:
                    candidates.append(float(rows[0]["capture_time"]) * 1_000_000)

    if candidates:
        return int(min(candidates))

    # Fallback : start_time_ns du metadata
    meta_path = sess / "metadata.json"
    if meta_path.exists():
        try:
            return int(json.loads(meta_path.read_text()).get("start_time_ns", 0))
        except Exception:
            pass
    return 0


def _get_timestamp_cols(session_path: str) -> dict:
    """
    Retourne les colonnes temporelles disponibles dans chaque CSV de la session.
    Utilisé pour permettre à l'UI de choisir la colonne de référence temporelle.
    """
    import pandas as _pd

    # Noms de colonnes reconnus comme temporels (numériques uniquement)
    TIMESTAMP_COLS = {"timestamp_ns", "t_ms_corrected_ns", "time_seconds",
                      "timestamp_abs_ms", "t_ms", "time_ms", "time", "t",
                      "capture_time", "index"}

    def _numeric_time_cols(df) -> list:
        """Colonnes dont le nom est temporel ET dont les valeurs sont numériques."""
        cols = []
        for c in df.columns:
            if c.lower() not in TIMESTAMP_COLS:
                continue
            try:
                _pd.to_numeric(df[c], errors="raise")
                cols.append(c)
            except (ValueError, TypeError):
                pass  # colonne ISO string ou texte → ignorée
        return cols

    sess = Path(session_path)
    result = {}

    trk_path = sess / "tracker_positions.csv"
    if trk_path.exists():
        df = _pd.read_csv(trk_path, nrows=5)
        cols = _numeric_time_cols(df)
        if cols:
            result["tracker"] = cols

    for side in ("left", "right"):
        grip_path = sess / f"gripper_{side}_data.csv"
        if grip_path.exists():
            df = _pd.read_csv(grip_path, nrows=5)
            cols = _numeric_time_cols(df)
            if cols:
                result[f"gripper_{side}"] = cols

    return result


@app.get("/api/session/timestamp_cols")
async def session_timestamp_cols(session_path: str):
    """
    Retourne les colonnes temporelles disponibles dans chaque CSV de la session.
    """
    sess = Path(session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")
    try:
        return JSONResponse(content=_get_timestamp_cols(session_path))
    except Exception as e:
        raise HTTPException(500, str(e))


def _load_session_timeseries(session_path: str, time_cols: dict = None) -> dict:
    """
    Charge toutes les séries temporelles d'une session et les aligne sur
    une timeline commune en millisecondes depuis le début de session.

    Retourne un dict JSON-sérialisable avec :
      - meta           : metadata.json
      - tracker        : {t_ms, head_x/y/z, left_x/y/z, right_x/y/z,
                          head_qw/qx/qy/qz, left_qw..., right_qw...}
      - gripper_left   : {t_ms, angle_deg}
      - gripper_right  : {t_ms, angle_deg}
      - videos         : {head: {t_ms, frame_idx}, left: ..., right: ...}
      - flux           : {head: {t_ms, motion}, left: ..., right: ...}
      - start_ns       : epoch ns du début de session
      - duration_ms    : durée totale en ms

    time_cols : dict optionnel, ex. {"tracker": "timestamp_ns", "gripper_left": "t_ms_corrected_ns"}
                Permet de forcer la colonne temporelle utilisée pour chaque flux.
    """
    import pandas as pd

    if time_cols is None:
        time_cols = {}

    sess = Path(session_path)
    result: dict = {
        "meta": {}, "tracker": {}, "gripper_left": {}, "gripper_right": {},
        "videos": {}, "flux": {}, "start_ns": 0, "duration_ms": 0,
    }

    # ── Metadata ──
    meta_path = sess / "metadata.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    result["meta"] = meta

    # t=0 = minimum réel de tous les flux (indépendant de start_time_ns metadata)
    start_ns = _compute_session_start_ns(session_path)
    result["start_ns"] = start_ns

    # ── Charger les données brutes ──
    trk_path = sess / "tracker_positions.csv"
    trk_df = None
    if trk_path.exists():
        trk_df = pd.read_csv(trk_path)

    grip_dfs: dict = {}
    for side in ("left", "right"):
        grip_path = sess / f"gripper_{side}_data.csv"
        if grip_path.exists():
            grip_dfs[side] = pd.read_csv(grip_path)

    vid_dir = sess / "videos"
    jsonl_rows: dict = {}
    if vid_dir.exists():
        for side in ("head", "left", "right"):
            jl = vid_dir / f"{side}.jsonl"
            if jl.exists():
                rows = _parse_jsonl(jl)
                if rows:
                    jsonl_rows[side] = rows

    def _ns_to_ms(ns_arr):
        """Convertit des timestamps ns en ms depuis start_ns."""
        return ((np.asarray(ns_arr, dtype=np.float64) - start_ns) / 1_000_000).tolist()

    def _epoch_ms_to_ms(epoch_ms_arr):
        """Convertit des timestamps epoch-ms en ms depuis start_ns."""
        start_ms = start_ns / 1_000_000
        return (np.asarray(epoch_ms_arr, dtype=np.float64) - start_ms).tolist()

    def _auto_to_ms(arr) -> list:
        """
        Détecte automatiquement le format d'une colonne temporelle et la convertit
        en ms depuis start_ns.  Formats supportés (détection par magnitude) :
          > 1e15  → nanosecondes epoch     → soustrait start_ns, divise par 1e6
          > 1e12  → microsecondes epoch    → convertit en ns, puis idem
          > 1e9   → millisecondes epoch    → soustrait start_ns/1e6
          > 0.5   → millisecondes relative → ramène à 0 (début = premier point)
          ≤ 0.5   → secondes relatives     → ×1000, ramène à 0
        Garantit que le résultat commence à ≥ 0 et ne contient pas de NaN/inf.
        """
        a = np.asarray(arr, dtype=np.float64)
        # Valeur représentative (médiane ignore les outliers)
        valid = a[np.isfinite(a)]
        if len(valid) == 0:
            return [0.0] * len(a)
        sample = float(np.median(valid))

        if sample > 1e15:                        # nanosecondes epoch
            out = (a - start_ns) / 1_000_000
        elif sample > 1e12:                      # microsecondes epoch
            out = (a * 1_000 - start_ns) / 1_000_000
        elif sample > 1e9:                       # millisecondes epoch
            out = a - (start_ns / 1_000_000)
        elif sample > 0.5:                       # ms relatif (offset inconnu)
            out = a - a[np.isfinite(a)][0]
        else:                                    # secondes relatives
            out = (a - a[np.isfinite(a)][0]) * 1_000

        # Sécurité : remplacer NaN/inf par interpolation linéaire
        out = np.where(np.isfinite(out), out, np.nan)
        nans = np.isnan(out)
        if nans.any() and (~nans).any():
            idx = np.arange(len(out))
            out[nans] = np.interp(idx[nans], idx[~nans], out[~nans])

        return out.tolist()

    # ── Passe 2 : construire les séries temporelles ──

    # ── Trackers ──
    if trk_df is not None:
        df = trk_df
        # Colonne temporelle : override UI ou fallback "timestamp_ns"
        trk_t_col = time_cols.get("tracker", "timestamp_ns")
        if trk_t_col not in df.columns:
            trk_t_col = "timestamp_ns"
        t_ms = _auto_to_ms(df[trk_t_col].to_numpy())
        trk: dict = {"t_ms": t_ms}
        for role in ("head", "left", "right"):
            for ax in ("x", "y", "z"):
                col = f"tracker_{role}_{ax}"
                if col in df.columns:
                    trk[f"{role}_{ax}"] = df[col].tolist()
            # Magnitude 3D = déplacement
            cols_xyz = [f"tracker_{role}_{ax}" for ax in ("x", "y", "z")]
            if all(c in df.columns for c in cols_xyz):
                xyz = df[cols_xyz].to_numpy(dtype=float)
                # Vitesse frame-à-frame (dérivée approximative)
                vel = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[:1]), axis=1)
                trk[f"{role}_speed"] = vel.tolist()
                # Position absolue (norme)
                trk[f"{role}_pos_norm"] = np.linalg.norm(xyz, axis=1).tolist()
        result["tracker"] = trk

    # ── Grippers ──
    for side in ("left", "right"):
        if side not in grip_dfs:
            continue
        df = grip_dfs[side]
        # Colonne temporelle : override UI ou auto-détection, toujours via _auto_to_ms
        grip_t_col = time_cols.get(f"gripper_{side}")
        if not (grip_t_col and grip_t_col in df.columns):
            # Fallback prioritaire
            for candidate in ("timestamp_ns", "t_ms_corrected_ns", "t_ms", "time_seconds"):
                if candidate in df.columns:
                    grip_t_col = candidate
                    break
        if grip_t_col and grip_t_col in df.columns:
            t_ms = _auto_to_ms(df[grip_t_col].to_numpy())
        else:
            t_ms = [0.0] * len(df)
        grip: dict = {"t_ms": t_ms}
        if "opening_mm" in df.columns:
            grip["opening_mm"] = df["opening_mm"].tolist()
        if "angle_deg" in df.columns:
            grip["angle_deg"] = df["angle_deg"].tolist()
        result[f"gripper_{side}"] = grip

    # ── JSONL vidéo (timestamps frames) ──
    for side, rows in jsonl_rows.items():
        t_ms = _epoch_ms_to_ms([r["capture_time"] for r in rows])
        idx  = [r["index"] for r in rows]
        result["videos"][side] = {"t_ms": t_ms, "frame_idx": idx}

    # ── Flux CSV (optical flow) ──
    if vid_dir.exists():
        for side in ("head", "left", "right"):
            for fname in (f"{side}_flux.csv", f"{side}.csv"):
                fp = vid_dir / fname
                if fp.exists():
                    df = pd.read_csv(fp)
                    cols_lower = {c.lower(): c for c in df.columns}
                    # Colonne temps : préférence décroissante
                    t_col = next((cols_lower[k] for k in (
                        "timestamp_abs_ms", "t_ms", "time_ms", "t", "time",
                        "time_seconds",
                    ) if k in cols_lower), None)
                    # Colonne signal : préférence décroissante
                    v_col = next((cols_lower[k] for k in (
                        "motion_mean_smooth", "motion_mean", "motion_median_smooth",
                        "motion_median", "motion", "flow", "value", "flux", "signal",
                    ) if k in cols_lower), None)
                    if t_col and v_col:
                        t_vals = df[t_col].to_numpy(dtype=float)
                        # timestamp_abs_ms est en epoch ms → convertir en ms depuis start
                        if "abs_ms" in t_col.lower() or "timestamp" in t_col.lower():
                            start_ms = start_ns / 1_000_000
                            t_vals = t_vals - start_ms
                        elif "second" in t_col.lower():
                            t_vals = t_vals * 1000.0
                        result["flux"][side] = {
                            "t_ms":   t_vals.tolist(),
                            "motion": df[v_col].tolist(),
                        }
                    break

    # ── Durée totale ──
    all_t: list = []
    if result["tracker"].get("t_ms"):
        all_t += result["tracker"]["t_ms"]
    for side in ("head", "left", "right"):
        if result["videos"].get(side, {}).get("t_ms"):
            all_t += result["videos"][side]["t_ms"]
    result["duration_ms"] = float(max(all_t)) if all_t else 0.0

    return result


def _extract_video_frame(session_path: str, side: str, t_ms: float) -> Optional[bytes]:
    """
    Extrait la frame vidéo la plus proche de t_ms (en ms depuis start de session)
    et la retourne encodée JPEG.
    Retourne None si impossible.
    """
    try:
        import cv2
        sess = Path(session_path)

        # Utiliser le vrai t=0 (minimum des timestamps réels de tous les flux)
        start_ns = _compute_session_start_ns(session_path)

        # Charger le JSONL pour trouver l'index de frame le plus proche
        jl_path = sess / "videos" / f"{side}.jsonl"
        if not jl_path.exists():
            return None

        rows = _parse_jsonl(jl_path)
        if not rows:
            return None

        start_ms = start_ns / 1_000_000
        target_epoch_ms = start_ms + t_ms

        # Frame la plus proche
        times  = np.array([r["capture_time"] for r in rows], dtype=float)
        idx    = int(np.argmin(np.abs(times - target_epoch_ms)))
        frame_number = rows[idx]["index"]

        # Ouvrir la vidéo et extraire la frame
        mp4_path = sess / "videos" / f"{side}.mp4"
        if not mp4_path.exists():
            return None

        cap = cv2.VideoCapture(str(mp4_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return None

        # Redimensionner pour l'UI (max 640px de large)
        h, w = frame.shape[:2]
        if w > 640:
            scale  = 640 / w
            frame  = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return bytes(buf)

    except Exception:
        return None


@app.get("/api/session/data")
async def session_data(
    session_path: str,
    time_col_tracker: Optional[str] = None,
    time_col_gripper_left: Optional[str] = None,
    time_col_gripper_right: Optional[str] = None,
):
    """
    Retourne toutes les séries temporelles alignées d'une session.
    Utilisé par l'onglet Visualisation.

    time_col_tracker / time_col_gripper_left / time_col_gripper_right :
        Colonne temporelle à utiliser pour chaque flux (override auto-détection).
    """
    sess = Path(session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")
    try:
        time_cols = {}
        if time_col_tracker:
            time_cols["tracker"] = time_col_tracker
        if time_col_gripper_left:
            time_cols["gripper_left"] = time_col_gripper_left
        if time_col_gripper_right:
            time_cols["gripper_right"] = time_col_gripper_right
        data = _load_session_timeseries(session_path, time_cols=time_cols)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(500, str(e))


# Verrous pour éviter les conversions faststart parallèles sur le même fichier
_faststart_locks: Dict[str, threading.Lock] = {}
_faststart_locks_mu = threading.Lock()

def _get_faststart_lock(key: str) -> threading.Lock:
    with _faststart_locks_mu:
        if key not in _faststart_locks:
            _faststart_locks[key] = threading.Lock()
        return _faststart_locks[key]


def _run_faststart(src: Path) -> bool:
    """
    Remuxe src en place avec -movflags +faststart (déplace le moov atom en tête).
    Opération rapide (pas de re-encodage). Retourne True si succès.
    """
    lock = _get_faststart_lock(str(src))
    if not lock.acquire(blocking=False):
        return False  # conversion déjà en cours
    try:
        tmp = src.with_suffix(".tmp_fs.mp4")
        try:
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src),
                 "-c", "copy", "-movflags", "+faststart", str(tmp)],
                capture_output=True, timeout=300,
            )
            if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                tmp.rename(src)
                return True
            tmp.unlink(missing_ok=True)
        except Exception:
            tmp.unlink(missing_ok=True)
        return False
    finally:
        lock.release()


@app.get("/api/session/video/{side}")
async def session_video(side: str, session_path: str, request: Request):
    """
    Sert le fichier mp4 avec support HTTP Range natif (Starlette FileResponse).
    Cache fort (1h) + ETag pour 304 Not Modified.
    Utilise automatiquement la version faststart (moov atom en tête) si disponible,
    sinon sert l'original et lance la conversion en arrière-plan.
    side : head | left | right
    """
    if side not in ("head", "left", "right"):
        raise HTTPException(400, "side doit être head, left ou right")
    sess = Path(session_path)
    vid_dir = sess / "videos"

    mp4 = vid_dir / f"{side}.mp4"

    if not mp4.exists():
        raise HTTPException(404, f"Vidéo {side} introuvable")

    # Lancer la conversion faststart en arrière-plan si le moov atom n'est pas en tête
    # (détecté par la présence du flag ; on tente une seule fois)
    threading.Thread(target=_run_faststart, args=(mp4,), daemon=True).start()

    stat = mp4.stat()
    etag = f'"{stat.st_size}-{int(stat.st_mtime)}"'

    # 304 Not Modified — évite tout re-transfert
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={
            "ETag": etag,
            "Cache-Control": "public, max-age=3600",
        })

    # FileResponse gère Range nativement (Starlette 0.36+) via sendfile
    return FileResponse(
        str(mp4),
        media_type="video/mp4",
        headers={
            "Accept-Ranges":  "bytes",
            "Cache-Control":  "public, max-age=3600",
            "ETag":           etag,
        },
    )



@app.get("/api/session/video_info")
async def session_video_info(session_path: str):
    """
    Retourne pour chaque caméra le t0 en secondes (offset entre le vrai t=0 de la session
    — minimum réel de tous les flux — et la première capture_time du JSONL),
    pour synchroniser video.currentTime avec la timeline.
    """
    sess = Path(session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")

    # t=0 = minimum réel de tous les flux (indépendant de start_time_ns metadata)
    start_ns = _compute_session_start_ns(session_path)
    start_ms = start_ns / 1_000_000

    result = {}
    for side in ("head", "left", "right"):
        jl = sess / "videos" / f"{side}.jsonl"
        mp4 = sess / "videos" / f"{side}.mp4"
        if not jl.exists() or not mp4.exists():
            continue
        rows = _parse_jsonl(jl)
        if not rows:
            continue
        first_capture_ms = float(rows[0]["capture_time"])
        # offset = temps de la première frame relative au vrai t=0 (en secondes)
        t0_s = (first_capture_ms - start_ms) / 1000.0
        result[side] = {
            "t0_s": t0_s,
            "available": True,
        }

    return JSONResponse(result)


class CameraSwapRequest(BaseModel):
    session_path: str
    # mapping complet slot→nouvelle_source, e.g. {"head": "left", "left": "head", "right": "right"}
    mapping: dict


@app.post("/api/session/camera_swap")
async def session_camera_swap(req: CameraSwapRequest):
    """
    Renomme physiquement les fichiers vidéo (.mp4, .jsonl, _flux.csv) pour appliquer
    un échange de caméras, puis met à jour metadata.json["cameras"] en conséquence.

    req.mapping = { slot_destination: slot_source_actuel }
    Exemple : {"head": "left", "left": "head", "right": "right"}
      → le fichier left.mp4 devient head.mp4, head.mp4 devient left.mp4, right.* inchangé

    L'opération est atomique via fichiers temporaires : en cas d'erreur partielle,
    les originaux sont restaurés.
    """
    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")

    mapping = req.mapping  # { dest: src }
    sides = ("head", "left", "right")

    # Valider le mapping : doit être une permutation valide des 3 côtés
    if set(mapping.keys()) != set(sides) or set(mapping.values()) != set(sides):
        raise HTTPException(400, "Le mapping doit être une permutation complète de head/left/right")

    # Si identité, rien à faire
    if all(mapping[s] == s for s in sides):
        return JSONResponse({"ok": True, "renamed": []})

    vid_dir = sess / "videos"
    extensions = [".mp4", ".jsonl", "_flux.csv"]

    # Construire la liste des renommages nécessaires (seulement les côtés qui changent)
    # On passe par des noms temporaires pour éviter les collisions (ex: head↔left)
    renames: list[tuple[Path, Path]] = []  # (src_path, dst_path)
    tmp_renames: list[tuple[Path, Path]] = []  # (original, tmp)

    for dst_side, src_side in mapping.items():
        if src_side == dst_side:
            continue
        for ext in extensions:
            src_name = f"{src_side}{ext}"
            dst_name = f"{dst_side}{ext}"
            src_path = vid_dir / src_name
            dst_path = vid_dir / dst_name
            if src_path.exists():
                renames.append((src_path, dst_path))

    if not renames:
        return JSONResponse({"ok": True, "renamed": []})

    # Étape 1 : déplacer tous les fichiers source vers des noms temporaires
    tmp_map: dict[Path, Path] = {}  # original_src → tmp_path
    try:
        for src_path, dst_path in renames:
            if src_path not in tmp_map:
                tmp_path = src_path.with_name(f"__swap_tmp_{src_path.name}")
                src_path.rename(tmp_path)
                tmp_map[src_path] = tmp_path
    except Exception as e:
        # Restaurer ce qui a déjà été déplacé
        for orig, tmp in tmp_map.items():
            if tmp.exists():
                tmp.rename(orig)
        raise HTTPException(500, f"Erreur lors du déplacement temporaire : {e}")

    # Étape 2 : déplacer les temporaires vers leurs destinations finales
    done: list[tuple[Path, Path]] = []
    try:
        for src_path, dst_path in renames:
            tmp_path = tmp_map[src_path]
            tmp_path.rename(dst_path)
            done.append((src_path, dst_path))
    except Exception as e:
        # Restaurer les temporaires restants
        for orig, tmp in tmp_map.items():
            if tmp.exists():
                tmp.rename(orig)
        raise HTTPException(500, f"Erreur lors du renommage final : {e}")

    # Étape 3 : mettre à jour metadata.json["cameras"] (échanger les positions)
    meta_path = sess / "metadata.json"
    renamed_files = [str(dst_path.name) for _, dst_path in done]
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if "cameras" in meta:
                # Reconstruire cameras avec les nouvelles positions
                # mapping[dst] = src → la caméra qui était à src va maintenant à dst
                # On cherche dans cameras quelle entrée a position == src_side
                new_cameras = {}
                for cam_id, cam_info in meta["cameras"].items():
                    cam_pos = cam_info.get("position")
                    # Trouver si ce cam_pos est une source dans le mapping
                    new_pos = next(
                        (dst for dst, src in mapping.items() if src == cam_pos),
                        cam_pos
                    )
                    new_cameras[cam_id] = {**cam_info, "position": new_pos}
                meta["cameras"] = new_cameras
            # Nettoyer un éventuel camera_remap résiduel
            meta.pop("camera_remap", None)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    except Exception as e:
        # Les fichiers sont déjà renommés, on signale juste l'erreur metadata
        return JSONResponse({"ok": True, "renamed": renamed_files, "meta_warning": str(e)})

    return JSONResponse({"ok": True, "renamed": renamed_files})


class TrackerSwapRequest(BaseModel):
    session_path: str
    # mapping complet role→nouvelle_source, e.g. {"head": "left", "left": "head", "right": "right"}
    mapping: dict


@app.post("/api/session/tracker_swap")
async def session_tracker_swap(req: TrackerSwapRequest):
    """
    Échange les rôles des trackers dans tracker_positions.csv en renommant les colonnes.
    Les colonnes tracker_{src}_* deviennent tracker_{dst}_*.
    Opération atomique via fichier temporaire.
    Met aussi à jour metadata.json["trackers"] si les entrées ont un champ "role".
    """
    import pandas as pd

    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")

    mapping = req.mapping  # { dst: src }
    sides = ("head", "left", "right")

    if set(mapping.keys()) != set(sides) or set(mapping.values()) != set(sides):
        raise HTTPException(400, "Le mapping doit être une permutation complète de head/left/right")

    if all(mapping[s] == s for s in sides):
        return JSONResponse({"ok": True, "swapped": []})

    trk_path = sess / "tracker_positions.csv"
    if not trk_path.exists():
        raise HTTPException(404, "tracker_positions.csv introuvable")

    df = pd.read_csv(trk_path)

    # Construire le renommage des colonnes
    # Pour chaque colonne tracker_{src}_{suffix}, la renommer en tracker_{dst}_{suffix}
    # où mapping[dst] = src
    rename_map = {}
    for dst, src in mapping.items():
        if src == dst:
            continue
        for col in df.columns:
            if col.startswith(f"tracker_{src}_"):
                suffix = col[len(f"tracker_{src}_"):]
                new_col = f"tracker_{dst}_{suffix}"
                rename_map[col] = new_col

    if not rename_map:
        return JSONResponse({"ok": True, "swapped": []})

    df = df.rename(columns=rename_map)

    # Écrire via fichier temporaire
    tmp_path = trk_path.with_name("__swap_tmp_tracker_positions.csv")
    try:
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(trk_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(500, f"Erreur écriture CSV : {e}")

    # Mettre à jour metadata.json["trackers"] si les entrées ont un champ "role"
    meta_path = sess / "metadata.json"
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if "trackers" in meta:
                updated = False
                new_trackers = {}
                for tid, tinfo in meta["trackers"].items():
                    role = tinfo.get("role")
                    if role and role in mapping.values():
                        # Trouver le nouveau rôle : dst tel que mapping[dst] == role
                        new_role = next(
                            (dst for dst, src in mapping.items() if src == role),
                            role
                        )
                        new_trackers[tid] = {**tinfo, "role": new_role}
                        updated = True
                    else:
                        new_trackers[tid] = tinfo
                if updated:
                    meta["trackers"] = new_trackers
                    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    except Exception:
        pass  # CSV déjà mis à jour, on ignore l'erreur metadata

    swapped = list(rename_map.keys())
    return JSONResponse({"ok": True, "swapped": swapped})


# ──────────────────────────────────────────────────────────────────────────────
# Bulk swap (camera + tracker) — traite N sessions en parallèle
# ──────────────────────────────────────────────────────────────────────────────

class BulkCameraSwapRequest(BaseModel):
    sessions: List[str]
    mapping: dict   # permutation complète head/left/right


@app.post("/api/session/camera_swap_bulk")
async def session_camera_swap_bulk(req: BulkCameraSwapRequest):
    """
    Applique le même camera_swap sur une liste de sessions en parallèle.
    Retourne { ok, done, errors } sans crasher si certaines sessions échouent.
    """
    sides = ("head", "left", "right")
    if set(req.mapping.keys()) != set(sides) or set(req.mapping.values()) != set(sides):
        raise HTTPException(400, "Le mapping doit être une permutation complète de head/left/right")
    if all(req.mapping[s] == s for s in sides):
        return JSONResponse({"ok": True, "done": len(req.sessions), "errors": []})

    def _do_one(sess_path: str):
        sess = Path(sess_path)
        if not sess.exists():
            return {"path": sess_path, "error": "introuvable"}
        vid_dir = sess / "videos"
        extensions = [".mp4", ".jsonl", "_flux.csv"]
        mapping = req.mapping
        renames = []
        for dst_side, src_side in mapping.items():
            if src_side == dst_side:
                continue
            for ext in extensions:
                src_path = vid_dir / f"{src_side}{ext}"
                dst_path = vid_dir / f"{dst_side}{ext}"
                if src_path.exists():
                    renames.append((src_path, dst_path))
        if not renames:
            return {"path": sess_path, "error": None}
        # Phase 1 : vers tmp
        tmp_map = {}
        try:
            for src_path, _ in renames:
                if src_path not in tmp_map:
                    tmp = src_path.with_name(f"__bswap_{src_path.name}")
                    src_path.rename(tmp)
                    tmp_map[src_path] = tmp
        except Exception as e:
            for orig, tmp in tmp_map.items():
                if tmp.exists():
                    try: tmp.rename(orig)
                    except Exception: pass
            return {"path": sess_path, "error": f"tmp: {e}"}
        # Phase 2 : tmp → dest
        try:
            for src_path, dst_path in renames:
                tmp_map[src_path].rename(dst_path)
        except Exception as e:
            for orig, tmp in tmp_map.items():
                if tmp.exists():
                    try: tmp.rename(orig)
                    except Exception: pass
            return {"path": sess_path, "error": f"rename: {e}"}
        # Phase 3 : metadata
        try:
            meta_path = sess / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if "cameras" in meta:
                    new_cameras = {}
                    for cam_id, cam_info in meta["cameras"].items():
                        cam_pos = cam_info.get("position")
                        new_pos = next(
                            (dst for dst, src in mapping.items() if src == cam_pos),
                            cam_pos
                        )
                        new_cameras[cam_id] = {**cam_info, "position": new_pos}
                    meta["cameras"] = new_cameras
                meta.pop("camera_remap", None)
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        except Exception:
            pass
        return {"path": sess_path, "error": None}

    loop = asyncio.get_event_loop()
    max_workers = min(16, max(2, (os.cpu_count() or 4)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [loop.run_in_executor(pool, _do_one, sp) for sp in req.sessions]
        results = await asyncio.gather(*futures)

    errors = [r for r in results if r["error"]]
    done   = len(results) - len(errors)
    return JSONResponse({"ok": True, "done": done, "errors": errors})


class BulkTrackerSwapRequest(BaseModel):
    sessions: List[str]
    mapping: dict


@app.post("/api/session/tracker_swap_bulk")
async def session_tracker_swap_bulk(req: BulkTrackerSwapRequest):
    """
    Applique le même tracker_swap sur une liste de sessions en parallèle.
    """
    import pandas as pd

    sides = ("head", "left", "right")
    if set(req.mapping.keys()) != set(sides) or set(req.mapping.values()) != set(sides):
        raise HTTPException(400, "Le mapping doit être une permutation complète de head/left/right")
    if all(req.mapping[s] == s for s in sides):
        return JSONResponse({"ok": True, "done": len(req.sessions), "errors": []})

    def _do_one(sess_path: str):
        import pandas as _pd
        sess = Path(sess_path)
        if not sess.exists():
            return {"path": sess_path, "error": "introuvable"}
        trk_path = sess / "tracker_positions.csv"
        if not trk_path.exists():
            return {"path": sess_path, "error": "tracker_positions.csv introuvable"}
        mapping = req.mapping
        try:
            df = _pd.read_csv(trk_path)
            rename_map = {}
            for dst, src in mapping.items():
                if src == dst:
                    continue
                for col in df.columns:
                    if col.startswith(f"tracker_{src}_"):
                        suffix = col[len(f"tracker_{src}_"):]
                        rename_map[col] = f"tracker_{dst}_{suffix}"
            if rename_map:
                df = df.rename(columns=rename_map)
                tmp = trk_path.with_suffix(".tmp.csv")
                df.to_csv(tmp, index=False)
                tmp.replace(trk_path)
        except Exception as e:
            return {"path": sess_path, "error": str(e)}
        # metadata
        try:
            meta_path = sess / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if "trackers" in meta:
                    new_trackers = {}
                    updated = False
                    for tid, tinfo in meta["trackers"].items():
                        role = tinfo.get("role")
                        new_role = next(
                            (dst for dst, src in mapping.items() if src == role),
                            role
                        )
                        if new_role != role:
                            new_trackers[tid] = {**tinfo, "role": new_role}
                            updated = True
                        else:
                            new_trackers[tid] = tinfo
                    if updated:
                        meta["trackers"] = new_trackers
                        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        except Exception:
            pass
        return {"path": sess_path, "error": None}

    loop = asyncio.get_event_loop()
    max_workers = min(16, max(2, (os.cpu_count() or 4)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [loop.run_in_executor(pool, _do_one, sp) for sp in req.sessions]
        results = await asyncio.gather(*futures)

    errors = [r for r in results if r["error"]]
    done   = len(results) - len(errors)
    return JSONResponse({"ok": True, "done": done, "errors": errors})


@app.get("/api/session/frame")
async def session_frame(session_path: str, side: str, t_ms: float):
    """
    Retourne une frame vidéo JPEG extraite à t_ms ms depuis le début de session.
    side : head | left | right
    """
    if side not in ("head", "left", "right"):
        raise HTTPException(400, "side doit être head, left ou right")

    frame_bytes = _extract_video_frame(session_path, side, t_ms)
    if frame_bytes is None:
        # Retourner une image placeholder 640×360 grise
        try:
            import cv2
            placeholder = np.full((360, 640, 3), 30, dtype=np.uint8)
            cv2.putText(placeholder, f"No frame — {side} @ {t_ms:.0f}ms",
                        (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
            _, buf = cv2.imencode(".jpg", placeholder)
            frame_bytes = bytes(buf)
        except Exception:
            raise HTTPException(404, "Frame introuvable")

    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/api/session/stream")
async def session_stream(session_path: str, fps: float = 30.0, t_start: float = 0.0):
    """
    SSE stream léger : uniquement tracker + grippers (pas de frames vidéo).
    Les vidéos sont lues nativement par <video> via /api/session/video/{side}.

    Chaque event data = JSON :
    {
      "t_ms": float,
      "duration_ms": float,
      "tracker": {"head": [x,y,z], "left": [x,y,z], "right": [x,y,z]},
      "gripper": {"left": float|null, "right": float|null}
    }
    Quand la fin est atteinte envoie {"done": true, "t_ms": duration_ms}.
    """
    sess = Path(session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")

    fps = max(1.0, min(fps, 60.0))
    interval = 1.0 / fps

    try:
        ts_data = _load_session_timeseries(session_path)
    except Exception as e:
        raise HTTPException(500, f"Erreur chargement session : {e}")

    duration_ms: float = ts_data.get("duration_ms", 0.0)
    if duration_ms <= 0:
        raise HTTPException(400, "Session vide (durée nulle)")

    def _build_arr(series: dict, key: str):
        v = series.get(key)
        return np.asarray(v, dtype=np.float64) if v else None

    trk = ts_data.get("tracker", {})
    trk_t   = _build_arr(trk, "t_ms")
    trk_data = {}
    for role in ("head", "left", "right"):
        xs = _build_arr(trk, f"{role}_x")
        ys = _build_arr(trk, f"{role}_y")
        zs = _build_arr(trk, f"{role}_z")
        if xs is not None and ys is not None and zs is not None:
            trk_data[role] = (xs, ys, zs)

    grip_data = {}
    for side in ("left", "right"):
        g = ts_data.get(f"gripper_{side}", {})
        gt = _build_arr(g, "t_ms")
        ga = _build_arr(g, "angle_deg")
        if gt is not None and ga is not None:
            grip_data[side] = (gt, ga)

    def _interp(t_arr, v_arr, t):
        if t_arr is None or v_arr is None:
            return None
        idx = int(np.searchsorted(t_arr, t))
        if idx == 0:
            return float(v_arr[0])
        if idx >= len(t_arr):
            return float(v_arr[-1])
        t0, t1 = t_arr[idx-1], t_arr[idx]
        v0, v1 = v_arr[idx-1], v_arr[idx]
        alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
        return float(v0 + alpha * (v1 - v0))

    async def _generate():
        import time as _time
        t_ms = t_start
        loop_start = _time.perf_counter()
        frame = 0
        while True:
            tracker_pos = {}
            if trk_t is not None:
                for role, (xs, ys, zs) in trk_data.items():
                    tracker_pos[role] = [
                        _interp(trk_t, xs, t_ms),
                        _interp(trk_t, ys, t_ms),
                        _interp(trk_t, zs, t_ms),
                    ]

            gripper_val = {}
            for side, (gt, ga) in grip_data.items():
                gripper_val[side] = _interp(gt, ga, t_ms)

            payload = {
                "t_ms":        t_ms,
                "duration_ms": duration_ms,
                "tracker":     tracker_pos,
                "gripper":     gripper_val,
            }
            yield f"data: {json.dumps(payload)}\n\n"

            frame += 1
            t_ms = t_start + frame * interval * 1000.0
            if t_ms >= duration_ms:
                yield f"data: {json.dumps({'done': True, 't_ms': duration_ms})}\n\n"
                return

            # Sleep précis : compense le temps pris par le calcul/sérialisation
            next_wake = loop_start + frame * interval
            delay = next_wake - _time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation — diff avant / après corrections
# ──────────────────────────────────────────────────────────────────────────────

def _load_diff_data(session_path: str) -> dict:
    """
    Calcule les données AVANT et APRÈS les deux corrections :
      A) fix.py  — réassignation des rôles tracker (head/left/right)
      B) IA sync — décalage temporel des grippers / vidéos

    Retourne :
    {
      "has_role_swap":  bool,
      "has_ia_shift":   bool,
      "fix": {
          "confidence": float,
          "swaps": [["left","right"], ...],
          "before": { tracker series avec anciens rôles },
          "after":  { tracker series avec rôles corrigés },
      },
      "ia": {
          "pairs": [
              { "name": "gripper_left",
                "shift_ms": float,
                "before": {"t_ms":[…], "signal":[…]},
                "after":  {"t_ms":[…], "signal":[…]},
                "ref":    {"t_ms":[…], "signal":[…], "name": "tracker_left"},
              },
              ...
          ]
      },
      "duration_ms": float,
      "start_ns": int,
    }
    """
    import pandas as pd
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from fix import find_tracker_blocks, infer_head, infer_left_right, build_mapping, global_confidence

    sess = Path(session_path)
    result: dict = {
        "has_role_swap": False, "has_ia_shift": False,
        "fix": {}, "ia": {"pairs": []},
        "duration_ms": 0, "start_ns": 0,
    }

    meta_path = sess / "metadata.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    # t=0 = minimum réel de tous les flux (indépendant de start_time_ns metadata)
    start_ns = _compute_session_start_ns(session_path)
    result["start_ns"] = start_ns

    def _ns_to_ms(ns_arr):
        return ((np.asarray(ns_arr, dtype=np.float64) - start_ns) / 1_000_000).tolist()

    # ── A) Rôles tracker ─────────────────────────────────────────────
    trk_path = sess / "tracker_positions.csv"
    if trk_path.exists():
        df = pd.read_csv(trk_path)
        t_ms = _ns_to_ms(df["timestamp_ns"].to_numpy())

        try:
            blocks    = find_tracker_blocks(df)
            head_info = infer_head(blocks)
            lr_info   = infer_left_right(blocks, head_info)
            mapping   = build_mapping(head_info, lr_info)
            conf      = global_confidence(head_info, lr_info)

            # Vérifier si swap nécessaire
            swapped_pairs = []
            for b in blocks:
                if b.role_hint and mapping[b.gid] != b.role_hint:
                    swapped_pairs.append([b.role_hint, mapping[b.gid]])
            has_swap = len(swapped_pairs) > 0
            result["has_role_swap"] = has_swap

            def _build_trk_series(role_map):
                """role_map : {gid -> role} — construit les séries avec ce mapping."""
                gid_to_block = {b.gid: b for b in blocks}
                series: dict = {"t_ms": t_ms}
                for gid, role in role_map.items():
                    b = gid_to_block[gid]
                    for ai, ax in enumerate("xyz"):
                        series[f"{role}_{ax}"] = b.pos[:, ai].tolist()
                    # Vitesse frame-à-frame
                    vel = np.linalg.norm(
                        np.diff(b.pos, axis=0, prepend=b.pos[:1]), axis=1
                    )
                    series[f"{role}_speed"] = vel.tolist()
                    series[f"{role}_pos_norm"] = np.linalg.norm(b.pos, axis=1).tolist()
                return series

            # Avant = rôles actuels dans le CSV (role_hint)
            hint_map   = {b.gid: b.role_hint for b in blocks if b.role_hint}
            # Après = rôles inférés corrects
            inferred_map = mapping

            result["fix"] = {
                "confidence":  conf,
                "swaps":       swapped_pairs,
                "before":      _build_trk_series(hint_map),
                "after":       _build_trk_series(inferred_map),
            }
        except Exception as e:
            result["fix"] = {"error": str(e)}

        result["duration_ms"] = float(t_ms[-1]) if t_ms else 0

    # ── B) Shifts IA ────────────────────────────────────────────────
    # Chercher les résultats de sync (sync_results.json ou sync_ml_advanced_results.json)
    shift_files = [
        sess / "sync_ml_advanced_results.json",
        sess / "sync_results.json",
    ]
    shift_data = None
    for sf in shift_files:
        if sf.exists():
            try:
                shift_data = json.loads(sf.read_text())
                break
            except Exception:
                pass

    if shift_data:
        pairs_out = []

        # Normaliser les deux formats de résultats
        pairs_raw = []
        if isinstance(shift_data, dict) and "pairs" in shift_data:
            # sync_results.json format
            pairs_raw = shift_data["pairs"]
        elif isinstance(shift_data, list):
            # sync_ml_advanced_results.json format (liste de PairEstimate)
            pairs_raw = shift_data

        for p in pairs_raw:
            # Extraire le nom cible et le shift
            tgt   = p.get("target") or p.get("tgt_name") or ""
            shift = p.get("offset_rec_ms") or p.get("shift_to_apply_ms") or 0.0
            ref_name = p.get("ref") or p.get("ref_name") or ""
            corr_ok  = p.get("corr_validated", True) and abs(float(shift)) > 1.0

            if not tgt or abs(float(shift)) < 0.5:
                continue

            result["has_ia_shift"] = True

            # Charger le signal cible — gripper ou video
            tgt_series = {"t_ms": [], "signal": [], "name": tgt}
            ref_series = {"t_ms": [], "signal": [], "name": ref_name}

            # Signal cible : gripper angle_deg
            for side in ("left", "right"):
                if side in tgt.lower():
                    gp = sess / f"gripper_{side}_data.csv"
                    if gp.exists():
                        gdf = pd.read_csv(gp)
                        if "timestamp_ns" in gdf.columns:
                            gt = _ns_to_ms(gdf["timestamp_ns"].to_numpy())
                        elif "t_ms_corrected_ns" in gdf.columns:
                            gt = _ns_to_ms(gdf["t_ms_corrected_ns"].to_numpy())
                        else:
                            gt = (gdf["time_seconds"].to_numpy(dtype=float) * 1000).tolist()
                        tgt_series["t_ms"]   = gt
                        tgt_series["signal"] = gdf["angle_deg"].tolist() if "angle_deg" in gdf.columns else []
                    break

            # Signal référence : tracker position norm
            if ref_name and trk_path.exists():
                for role in ("head", "left", "right"):
                    if role in ref_name.lower():
                        cols = [f"tracker_{role}_{ax}" for ax in "xyz"]
                        if all(c in df.columns for c in cols):
                            xyz = df[cols].to_numpy(dtype=float)
                            ref_series["t_ms"]   = t_ms
                            ref_series["signal"] = np.linalg.norm(
                                np.diff(xyz, axis=0, prepend=xyz[:1]), axis=1
                            ).tolist()
                        break

            if not tgt_series["t_ms"]:
                continue

            # Série "après" = signal cible décalé de shift ms
            t_after = (np.array(tgt_series["t_ms"]) - float(shift)).tolist()

            pairs_out.append({
                "name":     tgt,
                "ref_name": ref_name,
                "shift_ms": float(shift),
                "reliable": bool(corr_ok),
                "before":   {"t_ms": tgt_series["t_ms"], "signal": tgt_series["signal"]},
                "after":    {"t_ms": t_after,             "signal": tgt_series["signal"]},
                "ref":      ref_series,
            })

        result["ia"]["pairs"] = pairs_out

    return result


@app.get("/api/session/diff")
async def session_diff(session_path: str):
    """
    Retourne les données AVANT / APRÈS les corrections fix.py et IA sync.
    Utilisé par l'onglet Viewer → onglet Diff.
    """
    sess = Path(session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")
    try:
        data = _load_diff_data(session_path)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(500, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation — arborescence fichiers d'une session
# ──────────────────────────────────────────────────────────────────────────────

# Fichiers attendus dans une session complète (relatifs à la racine session)
_EXPECTED_FILES = [
    "metadata.json",
    "tracker_positions.csv",
    "gripper_left_data.csv",
    "gripper_right_data.csv",
    "videos/head.mp4",
    "videos/head.jsonl",
    "videos/left.mp4",
    "videos/left.jsonl",
    "videos/right.mp4",
    "videos/right.jsonl",
]


def _fmt_size(nb: int) -> str:
    if nb < 1024:
        return f"{nb} B"
    if nb < 1024 ** 2:
        return f"{nb/1024:.1f} KB"
    if nb < 1024 ** 3:
        return f"{nb/1024**2:.1f} MB"
    return f"{nb/1024**3:.2f} GB"


def _scan_dir(base: Path, rel: Path) -> list:
    """Retourne récursivement l'arborescence sous base/rel."""
    entries = []
    target = base / rel
    try:
        items = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return entries
    for item in items:
        item_rel = rel / item.name
        if item.is_dir():
            entries.append({
                "name":     item.name,
                "rel":      str(item_rel),
                "type":     "dir",
                "children": _scan_dir(base, item_rel),
            })
        else:
            stat = item.stat()
            entries.append({
                "name":     item.name,
                "rel":      str(item_rel),
                "type":     "file",
                "size":     stat.st_size,
                "size_fmt": _fmt_size(stat.st_size),
                "expected": str(item_rel).replace("\\", "/") in _EXPECTED_FILES,
            })
    return entries


@app.get("/api/session/files")
async def session_files(session_path: str):
    """
    Retourne l'arborescence complète des fichiers d'une session,
    avec taille et flag 'expected' (fichier attendu dans une session complète).
    """
    sess = Path(session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")

    tree = _scan_dir(sess, Path(""))

    # Fichiers attendus manquants
    missing = [
        f for f in _EXPECTED_FILES
        if not (sess / f).exists()
    ]

    return JSONResponse({
        "tree":    tree,
        "missing": missing,
    })


@app.get("/api/session/file")
async def session_file(session_path: str, filename: str):
    """Retourne le contenu JSON d'un fichier dans la session."""
    sess = Path(session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")
    # Sécurité : interdire les traversées de répertoire
    target = (sess / filename).resolve()
    if not str(target).startswith(str(sess.resolve())):
        raise HTTPException(403, "Accès refusé")
    if not target.exists():
        raise HTTPException(404, f"{filename} introuvable dans la session")
    return JSONResponse(json.loads(target.read_text(encoding="utf-8")))


# ──────────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────────

class ExportDestLocal(BaseModel):
    path: str

class ExportDestS3(BaseModel):
    bucket: str
    prefix: str = ""
    region: str = "eu-west-1"
    access_key: str
    secret_key: str

class ExportDestSftp(BaseModel):
    host: str
    port: int = 22
    user: str
    password: str
    remote_path: str

class LeRobotOptions(BaseModel):
    enabled: bool = False
    dataset_name: str = "robot_dataset"
    robot_type: str = "so100"
    fps: int = 30
    chunks_size: int = 1000  # nombre d'épisodes par chunk
    batch_size: int = 5      # épisodes traités en mémoire à la fois (anti-crash)
    push_to_hub: bool = False
    hf_token: str = ""
    repo_id: str = ""

class ExportRequest(BaseModel):
    sessions: List[str]
    dest_type: str                          # "local" | "s3" | "sftp"
    dest_local: Optional[ExportDestLocal] = None
    dest_s3: Optional[ExportDestS3] = None
    dest_sftp: Optional[ExportDestSftp] = None
    lerobot: Optional[LeRobotOptions] = None


def _worker_export(job: Job, req: ExportRequest):
    """Worker d'export : copie les sessions vers la destination configurée,
    avec conversion optionnelle au format LeRobot v3.

    En mode LeRobot, TOUTES les sessions sélectionnées forment un seul dataset
    (chaque session = un épisode), découpé en chunks de opts.chunks_size épisodes.
    """
    import shutil

    sessions = req.sessions
    total = len(sessions)
    _log_job(job, f"Export de {total} session(s) — destination: {req.dest_type}", "INFO")
    _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=0.0)

    errors = []
    exported_count = 0

    if req.lerobot and req.lerobot.enabled:
        # ── Mode LeRobot : un dataset multi-épisodes ─────────────────────────
        valid_sessions = []
        for sess_path in sessions:
            sess = Path(sess_path)
            if not sess.exists():
                errors.append(f"{sess_path}: introuvable")
                _log_job(job, f"SKIP {sess.name} — dossier introuvable", "WARN")
                continue
            missing_grippers = [
                f"gripper_{side}_data.csv"
                for side in ("left", "right")
                if not (sess / f"gripper_{side}_data.csv").exists()
            ]
            if missing_grippers:
                _log_job(job, f"SKIP {sess.name} — fichiers manquants : {', '.join(missing_grippers)}", "WARN")
                continue
            valid_sessions.append(sess)

        if valid_sessions:
            # Répertoire de travail sur T9 (même disque que les données source) pour éviter
            # de saturer /tmp du système. N'est supprimé qu'après un export réussi.
            work_dir = INGEST_DIR.parent / "_lerobot_work" / f"lerobot_{job.id}"
            work_dir.mkdir(parents=True, exist_ok=True)
            try:
                from lerobot_convert import build_lerobot_dataset, ConvertOptions
                _opts = ConvertOptions(
                    dataset_name=req.lerobot.dataset_name,
                    robot_type=req.lerobot.robot_type,
                    fps=req.lerobot.fps,
                    chunks_size=req.lerobot.chunks_size,
                    batch_size=req.lerobot.batch_size,
                    push_to_hub=req.lerobot.push_to_hub,
                    hf_token=req.lerobot.hf_token,
                    repo_id=req.lerobot.repo_id,
                )
                dataset_dir = build_lerobot_dataset(
                    valid_sessions, _opts, work_dir,
                    log_fn=lambda msg, level="INFO": _log_job(job, msg, level),
                )
                dataset_name = req.lerobot.dataset_name
                # Pour l'export local, on copie le contenu du dataset_dir directement dans
                # dest/dataset_name/ (pas dest/dataset_name/dataset_name/).
                # Le répertoire parent contient déjà la structure LeRobot complète (meta/, data/, videos/).
                if req.dest_type == "local":
                    _export_local_lerobot(job, dataset_dir, req.dest_local, dataset_name)
                elif req.dest_type == "s3":
                    _export_s3(job, dataset_dir, req.dest_s3, dataset_name)
                elif req.dest_type == "sftp":
                    _export_sftp(job, dataset_dir, req.dest_sftp, dataset_name)
                else:
                    raise ValueError(f"dest_type inconnu: {req.dest_type}")
                exported_count = len(valid_sessions)
                _log_job(job, f"✓ Dataset LeRobot exporté ({len(valid_sessions)} épisodes)", "OK")
                # Nettoyage uniquement en cas de succès complet
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception as e:
                errors.append(f"LeRobot build: {e}")
                _log_job(job, f"✗ Erreur build LeRobot: {e}", "ERROR")
                _log_job(job, f"  Travail partiel conservé dans {work_dir} — relancer le job pour reprendre", "WARN")
        _update_job(job, progress=100.0)
    else:
        # ── Mode raw : une session = un export ───────────────────────────────
        exported_count = 0
        for i, sess_path in enumerate(sessions):
            sess = Path(sess_path)
            if not sess.exists():
                errors.append(f"{sess.name}: introuvable")
                _log_job(job, f"[{i+1}/{total}] SKIP {sess.name} — dossier introuvable", "WARN")
                _update_job(job, progress=round((i + 1) / total * 100, 1))
                continue

            sess_name = sess.name
            # Chemin relatif depuis INGEST_DIR pour préserver la hiérarchie Silver
            # ex: /mnt/storage/silver///mnt/storage/silver//session_X → /mnt/storage/silver//session_X
            try:
                sess_rel = str(sess.relative_to(INGEST_DIR))
            except ValueError:
                sess_rel = sess.name
            # Log allégé : seulement tous les 50 pour les gros jobs
            if total <= 200 or (i % 50 == 0):
                _log_job(job, f"[{i+1}/{total}] Export {sess_rel}…", "INFO")
            try:
                if req.dest_type == "local":
                    _export_local(job, sess, req.dest_local, sess_rel)
                elif req.dest_type == "s3":
                    _export_s3(job, sess, req.dest_s3, sess_rel)
                elif req.dest_type == "sftp":
                    _export_sftp(job, sess, req.dest_sftp, sess_rel)
                else:
                    raise ValueError(f"dest_type inconnu: {req.dest_type}")
                exported_count += 1
                # Notifier le client que cette session a été exportée
                if _loop:
                    asyncio.run_coroutine_threadsafe(
                        _broadcast({"type": "session_removed", "session_path": sess_path, "reason": "exported"}),
                        _loop,
                    )
            except Exception as e:
                # Limiter la taille de la liste d'erreurs pour éviter l'OOM
                if len(errors) < 500:
                    errors.append(f"{sess_rel}: {e}")
                elif len(errors) == 500:
                    errors.append("... (trop d'erreurs, liste tronquée)")
                _log_job(job, f"[{i+1}/{total}] ✗ {sess_rel}: {e}", "ERROR")
            _update_job(job, progress=round((i + 1) / total * 100, 1))

    result = {"errors": errors, "error_count": len(errors), "total": total, "exported": exported_count}
    status = JobStatus.DONE if not errors else JobStatus.ERROR
    # Tronquer le message d'erreur global pour éviter les champs JSON géants
    if errors:
        err_sample = errors[:5]
        err_msg = "; ".join(err_sample) + (f" ... (+{len(errors)-5} autres)" if len(errors) > 5 else "")
    else:
        err_msg = None
    _update_job(job, status=status, ended_at=_now(), progress=100.0, result=result, error=err_msg)
    _log_job(job, f"Export terminé — {len(errors)} erreur(s)", "OK" if not errors else "WARN")



def _export_local_lerobot(job: Job, src: Path, dest: ExportDestLocal, dataset_name: str):
    """Export d'un dataset LeRobot complet.

    src est déjà le répertoire dataset (contient meta/, data/, videos/).
    On copie son contenu dans dest.path/dataset_name/ en écrasant les fichiers existants
    (comportement de mise à jour — permet la reprise après un export partiel).
    """
    import shutil
    out = Path(dest.path) / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
    _log_job(job, f"    → local: {out} ({n} fichiers)", "INFO")


def _export_local(job: Job, src: Path, dest: ExportDestLocal, sess_name: str):
    import shutil
    out = Path(dest.path) / sess_name
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(out))
    _log_job(job, f"    → local (move): {out}", "INFO")


def _export_s3(job: Job, src: Path, dest: ExportDestS3, sess_name: str):
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        raise RuntimeError("boto3 non installé — pip install boto3")
    boto_session = boto3.Session(
        aws_access_key_id=dest.access_key,
        aws_secret_access_key=dest.secret_key,
        region_name=dest.region,
    )
    s3 = boto_session.client("s3")
    prefix = (dest.prefix.rstrip("/") + "/" + sess_name + "/").lstrip("/")
    files = [f for f in src.rglob("*") if f.is_file()]
    n_ok = 0
    n_err = 0
    file_errors = []
    for f in files:
        key = prefix + str(f.relative_to(src)).replace("\\", "/")
        # Retry 3x sur les erreurs réseau/throttling transitoires
        last_exc = None
        for attempt in range(3):
            try:
                s3.upload_file(str(f), dest.bucket, key)
                last_exc = None
                break
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                # Ne pas retry sur les erreurs d'auth / bucket inexistant
                if code in ("AccessDenied", "NoSuchBucket", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
                    last_exc = e
                    break
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        if last_exc is not None:
            n_err += 1
            file_errors.append(f"{f.name}: {last_exc}")
        else:
            n_ok += 1
    if file_errors:
        sample = "; ".join(file_errors[:3]) + (f" ... (+{len(file_errors)-3} autres)" if len(file_errors) > 3 else "")
        raise RuntimeError(f"{n_err} fichier(s) non uploadé(s) sur {len(files)}: {sample}")
    _log_job(job, f"    → S3 s3://{dest.bucket}/{prefix} ({n_ok} fichiers)", "INFO")


def _export_sftp(job: Job, src: Path, dest: ExportDestSftp, sess_name: str):
    try:
        import paramiko
    except ImportError:
        raise RuntimeError("paramiko non installé — pip install paramiko")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(dest.host, port=dest.port, username=dest.user, password=dest.password)
    sftp = ssh.open_sftp()
    remote_base = dest.remote_path.rstrip("/") + "/" + sess_name
    n = 0
    for f in src.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(src)).replace("\\", "/")
            remote_path = remote_base + "/" + rel
            # Créer les répertoires parents distants
            remote_dir = remote_path.rsplit("/", 1)[0]
            parts = remote_dir.split("/")
            cur = ""
            for p in parts:
                if not p:
                    continue
                cur = (cur + "/" + p) if cur else ("/" + p)
                try:
                    sftp.stat(cur)
                except FileNotFoundError:
                    sftp.mkdir(cur)
            sftp.put(str(f), remote_path)
            n += 1
    sftp.close()
    ssh.close()
    _log_job(job, f"    → SFTP {dest.host}:{remote_base} ({n} fichiers)", "INFO")


@app.post("/api/export")
async def export_sessions(req: ExportRequest):
    """Lance un job d'export asynchrone pour les sessions sélectionnées."""
    if not req.sessions:
        raise HTTPException(400, "Aucune session fournie")
    if req.dest_type not in ("local", "s3", "sftp"):
        raise HTTPException(400, f"dest_type invalide: {req.dest_type}")
    job = _new_job("export")
    threading.Thread(target=_worker_export, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


# ──────────────────────────────────────────────────────────────────────────────
# Reject
# ──────────────────────────────────────────────────────────────────────────────

class RejectRequest(BaseModel):
    sessions: List[str]
    reject_path: str


def _worker_reject(job: Job, req: RejectRequest):
    import shutil
    dest_root = Path(req.reject_path)
    total = len(req.sessions)
    _log_job(job, f"Rejet de {total} session(s) vers {dest_root}", "WARN")
    _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=0.0)

    moved, errors = [], []
    for i, sess_path in enumerate(req.sessions):
        sess = Path(sess_path)
        if not sess.exists():
            errors.append(f"{sess_path}: introuvable")
            _log_job(job, f"[{i+1}/{total}] SKIP {sess.name} — introuvable", "WARN")
            _update_job(job, progress=round((i + 1) / total * 100, 1))
            continue
        try:
            dest_root.mkdir(parents=True, exist_ok=True)
            dest = dest_root / sess.name
            if dest.exists():
                # Évite l'écrasement silencieux : suffixe timestamp
                dest = dest_root / f"{sess.name}_{int(time.time())}"
            shutil.move(str(sess), str(dest))
            moved.append(sess.name)
            _log_job(job, f"[{i+1}/{total}] ✓ {sess.name} → {dest}", "OK")
            # Notifier le client que cette session a été rejetée
            if _loop:
                asyncio.run_coroutine_threadsafe(
                    _broadcast({"type": "session_removed", "session_path": sess_path, "reason": "rejected"}),
                    _loop,
                )
        except Exception as e:
            errors.append(f"{sess.name}: {e}")
            _log_job(job, f"[{i+1}/{total}] ✗ {sess.name}: {e}", "ERROR")
        _update_job(job, progress=round((i + 1) / total * 100, 1))

    result = {"moved": moved, "errors": errors, "total": total}
    status = JobStatus.DONE if not errors else JobStatus.ERROR
    _update_job(job, status=status, ended_at=_now(), progress=100.0,
                result=result, error="; ".join(errors) if errors else None)
    _log_job(job, f"Rejet terminé — {len(moved)}/{total} déplacée(s), {len(errors)} erreur(s)",
             "OK" if not errors else "WARN")


@app.post("/api/reject")
async def reject_sessions(req: RejectRequest):
    """Déplace les sessions sélectionnées vers le dossier de rejet."""
    if not req.sessions:
        raise HTTPException(400, "Aucune session fournie")
    if not req.reject_path:
        raise HTTPException(400, "reject_path manquant")
    job = _new_job("reject")
    threading.Thread(target=_worker_reject, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


# ──────────────────────────────────────────────────────────────────────────────
# Fix camera offset
# ──────────────────────────────────────────────────────────────────────────────

class FixCameraOffsetRequest(BaseModel):
    session: str
    force: bool = False


def _worker_fix_camera_offset(job: Job, req: FixCameraOffsetRequest):
    import importlib.util, sys as _sys
    _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=0.0)
    try:
        sess_path = Path(req.session)

        # ── Étape 0 : détecter si les vidéos sont à l'envers ─────────────
        rotate_result = None
        try:
            import importlib.util as _ilu, sys as _sys2
            _sc_spec = _ilu.spec_from_file_location(
                "session_check", _ROOT / "verification" / "session_check.py"
            )
            _sc_mod = _ilu.module_from_spec(_sc_spec)
            _sys2.modules.setdefault("session_check", _sc_mod)
            _sc_spec.loader.exec_module(_sc_mod)

            _log_job(job, "Diagnostic orientation vidéo…", "INFO")
            dim_vo = _sc_mod._dim_video_orientation(sess_path)
            upside_down = dim_vo.details.get("upside_down", False) if hasattr(dim_vo, "details") else False

            if upside_down:
                _log_job(job, "Vidéos détectées à l'envers — rotation 180° en cours…", "WARN")
                from utils.data_prep import rotate_session_videos
                rotate_result = rotate_session_videos(sess_path, force=req.force, log=lambda m, l="INFO": _log_job(job, m, l))
                rotated = rotate_result.get("rotated", [])
                errors  = rotate_result.get("errors", [])
                if rotated:
                    _log_job(job, f"✓ Rotation appliquée : {', '.join(rotated)}", "OK")
                if errors:
                    _log_job(job, f"Erreurs rotation : {errors}", "WARN")
            else:
                _log_job(job, "Orientation vidéo correcte — pas de rotation nécessaire", "INFO")
        except Exception as e_rot:
            _log_job(job, f"[WARN] Vérification orientation ignorée : {e_rot}", "WARN")

        _update_job(job, progress=40.0)

        # ── Étape 1 : corriger l'offset caméra ───────────────────────────
        fix_script = _ROOT / "fix_camera_offset.py"
        spec = importlib.util.spec_from_file_location("fix_camera_offset", fix_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        _log_job(job, f"Correction offset caméra : {sess_path.name}", "INFO")
        report = mod.fix_session(sess_path, dry_run=False, force=req.force)

        status = report.get("status", "?")
        if status == "corrected":
            cams = report.get("cameras_fixed", [])
            for c in cams:
                _log_job(job, f"  {c['camera']}: offset {c['offset_ms']:.1f} ms → {c['offset_applied_ms']} ms ({c['frames']} frames)", "OK")
            _log_job(job, f"✓ Correction terminée ({len(cams)} caméra(s))", "OK")
        else:
            reason = report.get("reason", "")
            _log_job(job, f"[{status.upper()}] {reason}", "WARN" if status in ("skipped", "ok") else "ERROR")

        if rotate_result is not None:
            report["rotate_videos"] = rotate_result

        _update_job(job, progress=100.0, status=JobStatus.DONE, ended_at=_now(), result=report)
    except Exception as e:
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=str(e))
        _log_job(job, f"Erreur : {e}", "ERROR")


@app.post("/api/session/fix_camera_offset")
async def fix_camera_offset(req: FixCameraOffsetRequest):
    """Corrige le décalage temporel des caméras d'une session."""
    if not req.session:
        raise HTTPException(400, "session manquante")
    if not Path(req.session).exists():
        raise HTTPException(404, f"Session introuvable : {req.session}")
    job = _new_job("fix_camera_offset")
    threading.Thread(target=_worker_fix_camera_offset, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


class FixCameraOffsetBulkRequest(BaseModel):
    sessions: List[str]
    force: bool = False


def _worker_fix_camera_offset_bulk(job: Job, sessions: List[str], force: bool):
    import importlib.util, sys as _sys
    _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=0.0)
    try:
        fix_script = _ROOT / "fix_camera_offset.py"
        spec = importlib.util.spec_from_file_location("fix_camera_offset", fix_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        total = len(sessions)
        results = []
        for i, sess_str in enumerate(sessions):
            sess_path = Path(sess_str)
            _log_job(job, f"[{i+1}/{total}] {sess_path.name}…", "INFO")
            try:
                report = mod.fix_session(sess_path, dry_run=False, force=force)
                status = report.get("status", "?")
                if status == "corrected":
                    cams = report.get("cameras_fixed", [])
                    _log_job(job, f"  ✓ {sess_path.name}: {len(cams)} caméra(s) corrigée(s)", "OK")
                else:
                    reason = report.get("reason", "")
                    _log_job(job, f"  [{status.upper()}] {sess_path.name}: {reason}", "WARN")
                results.append({"session": sess_str, "status": status, "result": report})
            except Exception as e:
                _log_job(job, f"  Erreur {sess_path.name}: {e}", "ERROR")
                results.append({"session": sess_str, "status": "error", "error": str(e)})
            _update_job(job, progress=round((i + 1) / total * 100, 1))

        corrected = sum(1 for r in results if r["status"] == "corrected")
        _log_job(job, f"Fix terminé : {corrected}/{total} session(s) corrigée(s)", "OK")
        _update_job(job, status=JobStatus.DONE, ended_at=_now(),
                    result={"corrected": corrected, "total": total, "details": results})
    except Exception as e:
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=str(e))
        _log_job(job, f"Erreur bulk fix: {e}", "ERROR")


@app.post("/api/session/fix_camera_offset_bulk")
async def fix_camera_offset_bulk(req: FixCameraOffsetBulkRequest):
    """Corrige l'offset caméra sur plusieurs sessions en parallèle."""
    if not req.sessions:
        raise HTTPException(400, "sessions manquantes")
    job = _new_job("fix_camera_offset_bulk")
    threading.Thread(
        target=_worker_fix_camera_offset_bulk,
        args=(job, req.sessions, req.force),
        daemon=True,
    ).start()
    return {"job_id": job.id}


# ── Fix complet (tous les scripts) ────────────────────────────────────────────

class FixAllBulkRequest(BaseModel):
    sessions: List[str]
    force: bool = False
    # Quels modules activer (True par défaut)
    run_tracker_labels:    bool = True
    run_camera_labels:     bool = True
    run_camera_offset:     bool = True
    run_timestamp_sync:    bool = True   # vérification uniquement (pas de modification)
    run_gripper_sync:      bool = True   # vérification uniquement (pas de modification)


def _worker_fix_all_bulk(job: Job, req: FixAllBulkRequest):
    """Worker qui enchaîne les 4 nouveaux fix + camera_offset sur chaque session."""
    import sys as _sys
    _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=0.0)

    # Importer les modules fix
    fix_root = _ROOT
    if str(fix_root) not in _sys.path:
        _sys.path.insert(0, str(fix_root))

    try:
        from fix.fix_tracker_labels  import fix_tracker_labels
        from fix.fix_camera_labels   import fix_camera_labels
        from fix.fix_timestamp_sync  import analyse_timestamp_sync
        from fix.fix_gripper_video_sync import analyse_gripper_video_sync
    except ImportError as e:
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=str(e))
        _log_job(job, f"Erreur import fix modules : {e}", "ERROR")
        return

    # Charger fix_camera_offset (legacy)
    import importlib.util
    _cam_fix_mod = None
    if req.run_camera_offset:
        try:
            fix_script = _ROOT / "fix_camera_offset.py"
            spec = importlib.util.spec_from_file_location("fix_camera_offset", fix_script)
            _cam_fix_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_cam_fix_mod)
        except Exception as e:
            _log_job(job, f"[WARN] fix_camera_offset.py introuvable : {e}", "WARN")

    total = len(req.sessions)
    results = []

    for i, sess_str in enumerate(req.sessions):
        sess_path = Path(sess_str)
        name = sess_path.name
        _log_job(job, f"", "INFO")
        _log_job(job, f"═══════════════════════════════════════════════════", "INFO")
        _log_job(job, f"[{i+1}/{total}] {name}", "INFO")

        # Snapshot metadata avant tous les fix
        _meta_before: dict = {}
        try:
            mp = sess_path / "metadata.json"
            if mp.exists():
                _meta_before = json.loads(mp.read_text(encoding="utf-8"))
                dur = _meta_before.get("duration_seconds", _meta_before.get("duration", 0))
                n_trk = len(_meta_before.get("trackers", {}))
                n_cam = len(_meta_before.get("cameras", {}))
                prev_score = _meta_before.get("check_score")
                prev_grade = _meta_before.get("check_grade", "")
                score_str  = f"  score={prev_score:.0f}% ({prev_grade})" if prev_score is not None else ""
                _log_job(job,
                    f"  durée={dur:.1f}s  trackers={n_trk}  caméras={n_cam}{score_str}",
                    "INFO")
        except Exception:
            pass
        _log_job(job, f"═══════════════════════════════════════════════════", "INFO")

        sess_result = {"session": sess_str, "fixes": {}}

        # ── 1. Labels trackers ────────────────────────────────────────────────
        if req.run_tracker_labels:
            _log_job(job, "  ┌─ TRACKERS ─────────────────────────────────────", "INFO")
            try:
                r = fix_tracker_labels(sess_path, dry_run=False, force=req.force)
                s = r.status

                # Avant
                old = r.old_assignment or {"head": "head", "left": "left", "right": "right"}
                _log_job(job,
                    f"  │ AVANT  head={old.get('head','?')}  left={old.get('left','?')}  right={old.get('right','?')}",
                    "INFO")

                # Détail des 4 tests
                for t in r.tests:
                    ev = t.evidence
                    if t.name == "height_Y":
                        med = ev.get("medians_Y", {})
                        med_str = "  ".join(f"{k}={v:.3f}m" for k, v in med.items())
                        _log_job(job,
                            f"  │ Test1 height_Y    head={t.head_vote:8s} conf={t.confidence:.2f}"
                            f"  z={ev.get('zscore', 0):.1f}σ  Δy={ev.get('delta_Y_m', 0):.3f}m"
                            f"  [{med_str}]",
                            "INFO")
                    elif t.name == "centrality_3D":
                        dist = ev.get("mean_dist_m", {})
                        d_str = "  ".join(f"{k}={v:.3f}m" for k, v in dist.items())
                        _log_job(job,
                            f"  │ Test2 centrality  head={t.head_vote:8s} conf={t.confidence:.2f}"
                            f"  sep={ev.get('separation', 0):.2f}  [{d_str}]",
                            "INFO")
                    elif t.name == "mobility":
                        spd = ev.get("median_speed_m_per_frame", {})
                        s_str = "  ".join(f"{k}={v:.5f}" for k, v in spd.items())
                        _log_job(job,
                            f"  │ Test3 mobility    head={t.head_vote:8s} conf={t.confidence:.2f}"
                            f"  sep={ev.get('separation', 0):.2f}  [{s_str}]",
                            "INFO")
                    elif t.name == "lateral_projection":
                        proj = ev.get("projections_m", {})
                        p_str = "  ".join(f"{k}={v:+.3f}m" for k, v in proj.items())
                        _log_job(job,
                            f"  │ Test4 lateral     left={t.left_vote:8s} right={t.right_vote:8s}"
                            f"  conf={t.confidence:.2f}  sep={ev.get('separation_m', 0):.3f}m  [{p_str}]",
                            "INFO")

                # Consensus
                z = next((t.evidence.get("zscore", 0) for t in r.tests if t.name == "height_Y"), 0)
                _log_job(job,
                    f"  │ Accord {r.agreement_count}/4 tests  z={z:.1f}σ"
                    f"  → {'CERTAIN' if s in ('ok','corrected') else 'INCERTAIN'}",
                    "OK" if s in ("ok", "corrected") else "WARN")

                # Après
                if s == "corrected":
                    pred = r.predicted
                    _log_job(job,
                        f"  │ APRÈS  head={pred.get('head','?')}  left={pred.get('left','?')}  right={pred.get('right','?')}"
                        f"  ← CORRIGÉ",
                        "OK")
                elif s == "ok":
                    _log_job(job, f"  │ APRÈS  inchangé (labels corrects)", "INFO")
                elif s == "uncertain":
                    _log_job(job, f"  │ APRÈS  inchangé — {r.reason}", "WARN")
                elif s == "skipped":
                    _log_job(job, f"  │ APRÈS  ignoré (déjà vérifié)", "INFO")
                else:
                    _log_job(job, f"  │ [{s}] {r.reason}", "WARN")

                sess_result["fixes"]["tracker_labels"] = {"status": s, "reason": r.reason,
                                                           "predicted": r.predicted}
            except Exception as e:
                _log_job(job, f"  │ ✗ erreur : {e}", "ERROR")
                sess_result["fixes"]["tracker_labels"] = {"status": "error", "error": str(e)}
            _log_job(job, "  └────────────────────────────────────────────────", "INFO")

        # ── 2. Labels caméras ─────────────────────────────────────────────────
        if req.run_camera_labels:
            _log_job(job, "  ┌─ CAMERAS ──────────────────────────────────────", "INFO")
            try:
                r = fix_camera_labels(sess_path, dry_run=False, force=req.force)
                s = r.status

                # Avant
                cur = r.current or {}
                if cur:
                    cur_str = "  ".join(f"{pos}={lbl}" for pos, lbl in sorted(cur.items()))
                    _log_job(job, f"  │ AVANT  {cur_str}", "INFO")

                # Trackers corrigés
                if r.tracker_prediction:
                    trk_str = "  ".join(
                        f"{role}←{csv}" for role, csv in sorted(r.tracker_prediction.items())
                        if role != csv
                    )
                    if trk_str:
                        _log_job(job, f"  │ Trackers swap : {trk_str}", "INFO")

                # Scores flux optique
                for cam_file, scores in (r.flow_scores or {}).items():
                    if cam_file == "head":
                        continue
                    scores_str = "  ".join(f"{role}={v:.2f}" for role, v in sorted(scores.items()))
                    _log_job(job, f"  │ Flux [{cam_file:5s}] {scores_str}", "INFO")

                # Après
                pred = r.predicted or {}
                if s == "corrected":
                    pred_str = "  ".join(f"{cam}→{role}" for cam, role in sorted(pred.items()))
                    _log_job(job, f"  │ APRÈS  {pred_str}  ← CORRIGÉ", "OK")
                elif s == "ok":
                    _log_job(job, f"  │ APRÈS  inchangé (labels corrects)", "INFO")
                elif s == "uncertain":
                    _log_job(job, f"  │ APRÈS  inchangé — {r.reason}", "WARN")
                elif s == "skipped":
                    _log_job(job, f"  │ APRÈS  ignoré (déjà vérifié)", "INFO")
                else:
                    _log_job(job, f"  │ [{s}] {r.reason}", "WARN")

                sess_result["fixes"]["camera_labels"] = {"status": s, "reason": r.reason}
            except Exception as e:
                _log_job(job, f"  │ ✗ erreur : {e}", "ERROR")
                sess_result["fixes"]["camera_labels"] = {"status": "error", "error": str(e)}
            _log_job(job, "  └────────────────────────────────────────────────", "INFO")

        # ── 3. Offset caméra (fix_camera_offset legacy) ───────────────────────
        if req.run_camera_offset and _cam_fix_mod is not None:
            _log_job(job, "  ┌─ CAMERA OFFSET ────────────────────────────────", "INFO")
            try:
                report = _cam_fix_mod.fix_session(sess_path, dry_run=False, force=req.force)
                s = report.get("status", "?")
                if s == "corrected":
                    cams = report.get("cameras_fixed", [])
                    shifts = report.get("shifts", report.get("offsets", {}))
                    for cam_name, cam_info in (shifts.items() if isinstance(shifts, dict) else {}):
                        before_ms = cam_info.get("before_ms", cam_info.get("old_ms", "?"))
                        after_ms  = cam_info.get("after_ms",  cam_info.get("new_ms", cam_info.get("shift_ms", "?")))
                        _log_job(job,
                            f"  │ {cam_name:12s}  AVANT={before_ms}ms  APRÈS={after_ms}ms",
                            "OK")
                    if not shifts:
                        _log_job(job, f"  │ {len(cams)} caméra(s) recalée(s) : {cams}", "OK")
                else:
                    _log_job(job, f"  │ [{s}] {report.get('reason','')}", "INFO")
                sess_result["fixes"]["camera_offset"] = {"status": s}
            except Exception as e:
                _log_job(job, f"  │ ✗ erreur : {e}", "ERROR")
                sess_result["fixes"]["camera_offset"] = {"status": "error", "error": str(e)}
            _log_job(job, "  └────────────────────────────────────────────────", "INFO")

        # ── 4. Vérification timestamps (lecture seule) ─────────────────────────
        if req.run_timestamp_sync:
            _log_job(job, "  ┌─ TIMESTAMPS ───────────────────────────────────", "INFO")
            try:
                r = analyse_timestamp_sync(sess_path)

                # Détail par flux
                for st in r.streams:
                    gaps_str = f"{st.n_gaps} gap(s) +{st.total_gap_ms:.0f}ms" if st.n_gaps else "0 gaps"
                    drift_str = f"  drift={st.drift_ppm:+.0f}ppm" if st.drift_ppm is not None else ""
                    cov_str   = f"  cov={st.coverage*100:.0f}%" if st.coverage is not None else ""
                    icon = "✓" if st.status == "ok" else ("⚠" if st.status == "warn" else "✗")
                    _log_job(job,
                        f"  │ {icon} {st.name:20s}  {st.n_samples:5d} samples"
                        f"  {st.duration_ms/1000:.1f}s  {st.sample_rate_hz:.1f}Hz"
                        f"  {gaps_str}{drift_str}{cov_str}",
                        "INFO" if st.status == "ok" else "WARN")

                # Alignements paires
                for pa in r.pairs:
                    aligned_str = "aligné" if pa.aligned else f"DECALE Δstart={pa.delta_start_ms:+.0f}ms"
                    _log_job(job,
                        f"  │   {pa.stream_a:15s} ↔ {pa.stream_b:15s}"
                        f"  overlap={pa.overlap_ms/1000:.1f}s  ±lag={pa.max_lag_searchable_ms:.0f}ms  {aligned_str}",
                        "INFO" if pa.aligned else "WARN")

                # Issues
                for issue in r.issues:
                    _log_job(job, f"  │ ⚠ {issue}", "WARN")

                if not r.issues:
                    _log_job(job, f"  │ ✓ Tous les flux sont synchronisés", "INFO")

                deltas = r.summary.get("camera_start_deltas_ms", {})
                sess_result["fixes"]["timestamp_sync"] = {
                    "status": r.status, "issues": r.issues,
                    "camera_deltas_ms": deltas,
                }
            except Exception as e:
                _log_job(job, f"  │ ✗ erreur : {e}", "ERROR")
                sess_result["fixes"]["timestamp_sync"] = {"status": "error", "error": str(e)}
            _log_job(job, "  └────────────────────────────────────────────────", "INFO")

        # ── 5. Vérification gripper-vidéo (lecture seule) ─────────────────────
        if req.run_gripper_sync:
            _log_job(job, "  ┌─ GRIPPER SYNC ─────────────────────────────────", "INFO")
            try:
                r = analyse_gripper_video_sync(sess_path, level=1)
                if r.status == "no_data":
                    _log_job(job, "  │ – pas de données gripper", "INFO")
                else:
                    for res in r.results:
                        aligned_str = "aligné" if res.temporal_aligned else "DECALE"
                        _log_job(job,
                            f"  │ {res.side:6s}  Δstart={res.temporal_delta_start_ms:+7.0f}ms"
                            f"  overlap={res.temporal_overlap_ms/1000:.1f}s  {aligned_str}",
                            "INFO" if res.temporal_aligned else "WARN")
                    for issue in r.issues:
                        _log_job(job, f"  │ ⚠ {issue}", "WARN")
                    if not r.issues:
                        _log_job(job, f"  │ ✓ Gripper synchronisé", "INFO")

                sess_result["fixes"]["gripper_sync"] = {
                    "status": r.status, "issues": r.issues,
                    "results": [{"side": res.side,
                                 "delta_ms": res.temporal_delta_start_ms,
                                 "overlap_ms": res.temporal_overlap_ms,
                                 "aligned": res.temporal_aligned}
                                for res in r.results],
                }
            except Exception as e:
                _log_job(job, f"  │ ✗ erreur : {e}", "ERROR")
                sess_result["fixes"]["gripper_sync"] = {"status": "error", "error": str(e)}
            _log_job(job, "  └────────────────────────────────────────────────", "INFO")

        results.append(sess_result)
        _update_job(job, progress=round((i + 1) / total * 100, 1))

    # Résumé final
    n_corr = sum(
        1 for r in results
        if any(f.get("status") == "corrected" for f in r["fixes"].values())
    )
    n_warn = sum(
        1 for r in results
        if any(f.get("status") in ("uncertain", "warn")
               for f in r["fixes"].values())
    )
    _log_job(job,
             f"Fix terminé : {n_corr} session(s) modifiée(s), "
             f"{n_warn} avertissement(s) sur {total} session(s)",
             "OK")
    _update_job(job, status=JobStatus.DONE, ended_at=_now(),
                result={"corrected": n_corr, "warned": n_warn,
                        "total": total, "details": results})


@app.post("/api/session/fix_all_bulk")
async def fix_all_bulk(req: FixAllBulkRequest):
    """Applique tous les fix (labels trackers, labels caméras, offset caméra,
    vérification timestamps, vérification sync gripper) sur les sessions sélectionnées."""
    if not req.sessions:
        raise HTTPException(400, "sessions manquantes")
    job = _new_job("fix_all_bulk")
    threading.Thread(
        target=_worker_fix_all_bulk,
        args=(job, req),
        daemon=True,
    ).start()
    return {"job_id": job.id}


class TrimStreamsBulkRequest(BaseModel):
    sessions: List[str]
    force: bool = False


def _worker_trim_streams_bulk(job: Job, sessions: List[str], force: bool):
    import importlib.util
    _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=0.0)
    try:
        fix_script = _ROOT / "fix_camera_offset.py"
        spec = importlib.util.spec_from_file_location("fix_camera_offset", fix_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        total = len(sessions)
        results = []
        for i, sess_str in enumerate(sessions):
            sess_path = Path(sess_str)
            _log_job(job, f"[{i+1}/{total}] {sess_path.name}…", "INFO")
            try:
                report = mod.trim_session(sess_path, dry_run=False, force=force)
                status = report.get("status", "?")
                if status == "trimmed":
                    streams = report.get("streams", {})
                    total_removed = sum(s.get("rows_removed", s.get("frames_removed", 0)) for s in streams.values())
                    trims = {k: round(v, 0) for k, v in report.get("trims_ms", {}).items() if v > 1}
                    _log_job(job, f"  ✓ {sess_path.name}: {total_removed} lignes rognées — {trims}", "OK")
                else:
                    _log_job(job, f"  [{status.upper()}] {sess_path.name}: {report.get('reason', '')}", "WARN")
                results.append({"session": sess_str, "status": status, "result": report})
            except Exception as e:
                _log_job(job, f"  Erreur {sess_path.name}: {e}", "ERROR")
                results.append({"session": sess_str, "status": "error", "error": str(e)})
            _update_job(job, progress=round((i + 1) / total * 100, 1))

        trimmed = sum(1 for r in results if r["status"] == "trimmed")
        _log_job(job, f"Trim terminé : {trimmed}/{total} session(s) rognée(s)", "OK")
        _update_job(job, status=JobStatus.DONE, ended_at=_now(),
                    result={"trimmed": trimmed, "total": total, "details": results})
    except Exception as e:
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=str(e))
        _log_job(job, f"Erreur bulk trim: {e}", "ERROR")


@app.post("/api/session/trim_streams_bulk")
async def trim_streams_bulk(req: TrimStreamsBulkRequest):
    """Rogne le début de chaque flux pour aligner temporellement les sessions sélectionnées."""
    if not req.sessions:
        raise HTTPException(400, "sessions manquantes")
    job = _new_job("trim_streams_bulk")
    threading.Thread(
        target=_worker_trim_streams_bulk,
        args=(job, req.sessions, req.force),
        daemon=True,
    ).start()
    return {"job_id": job.id}


class ApplyGripperOffsetRequest(BaseModel):
    session:   str
    offset_ms: float  # décalage à ajouter aux timestamps gripper (en ms)


@app.post("/api/session/apply_gripper_offset")
async def apply_gripper_offset(req: ApplyGripperOffsetRequest):
    """
    Applique un décalage temporel permanent aux fichiers gripper_*_data.csv
    de la session indiquée. Le décalage (offset_ms) est ajouté à la colonne
    timestamp_ns (convertie depuis ms). Opération atomique via fichier temporaire.
    """
    import pandas as pd

    if not req.session:
        raise HTTPException(400, "session manquante")
    sess = Path(req.session)
    if not sess.exists():
        raise HTTPException(404, f"Session introuvable : {req.session}")
    if req.offset_ms == 0:
        return JSONResponse({"ok": True, "files_patched": []})

    offset_ns = int(req.offset_ms * 1_000_000)  # ms → ns
    files_patched: list[str] = []

    for side in ("left", "right"):
        grip_path = sess / f"gripper_{side}_data.csv"
        if not grip_path.exists():
            continue
        df = pd.read_csv(grip_path)
        ts_col = None
        if "timestamp_ns" in df.columns:
            ts_col = "timestamp_ns"
        elif "t_ms_corrected_ns" in df.columns:
            ts_col = "t_ms_corrected_ns"
        if ts_col is None:
            continue  # pas de colonne timestamp connue — ignorer
        df[ts_col] = df[ts_col] + offset_ns
        tmp_path = Path("/tmp") / f"__tmp_gripper_{side}_data.csv"
        try:
            df.to_csv(tmp_path, index=False)
            shutil.move(str(tmp_path), str(grip_path))
            files_patched.append(grip_path.name)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise HTTPException(500, f"Erreur écriture {grip_path.name} : {e}")

    if not files_patched:
        raise HTTPException(400, "Aucune colonne timestamp_ns trouvée dans les fichiers gripper")

    return JSONResponse({"ok": True, "files_patched": files_patched, "offset_ms": req.offset_ms})


# ──────────────────────────────────────────────────────────────────────────────
# Trim to SW — tronque tous les fichiers sur la fenêtre sw=ON → sw=OFF (gripper droit)
# ──────────────────────────────────────────────────────────────────────────────

class TrimToSwRequest(BaseModel):
    session: str
    force:   bool = False


def _worker_trim_to_sw(job: Job, req: TrimToSwRequest):
    import pandas as pd
    _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=0.0)
    try:
        sess = Path(req.session)
        meta_path = sess / "metadata.json"

        # Marqueur anti-double-application
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not req.force and meta.get("trim_to_sw_applied"):
            _log_job(job, "Déjà tronqué (trim_to_sw_applied=true) — utilise --force pour ré-appliquer", "WARN")
            _update_job(job, progress=100.0, status=JobStatus.DONE, ended_at=_now(),
                        result={"status": "skipped", "reason": "déjà appliqué"})
            return

        # ── 1. Lire gripper droit, trouver t0 (1er sw=ON) et t1 (dernier sw=OFF) ──
        grip_right = sess / "gripper_right_data.csv"
        if not grip_right.exists():
            raise FileNotFoundError("gripper_right_data.csv introuvable")

        df_gr = pd.read_csv(grip_right)
        if "sw" not in df_gr.columns or "timestamp_ns" not in df_gr.columns:
            raise ValueError("Colonnes 'sw' ou 'timestamp_ns' manquantes dans gripper_right_data.csv")

        on_rows  = df_gr[df_gr["sw"] == "ON"]
        off_rows = df_gr[df_gr["sw"] == "OFF"]
        if on_rows.empty:
            raise ValueError("Aucune ligne sw=ON dans gripper_right_data.csv")
        if off_rows.empty:
            raise ValueError("Aucune ligne sw=OFF dans gripper_right_data.csv")

        t0_ns = int(on_rows["timestamp_ns"].iloc[0])
        t1_ns = int(off_rows["timestamp_ns"].iloc[-1])
        duration_s = (t1_ns - t0_ns) / 1e9

        _log_job(job, f"Fenêtre sw : t0={t0_ns}  t1={t1_ns}  durée={duration_s:.2f}s", "INFO")
        _update_job(job, progress=10.0)

        report = {
            "status":     "trimmed",
            "t0_ns":      t0_ns,
            "t1_ns":      t1_ns,
            "duration_s": round(duration_s, 3),
            "files":      [],
        }

        # ── helper : tronque un CSV sur [t0_ns, t1_ns] ──────────────────────────
        def trim_csv(path: Path, ts_col: str):
            if not path.exists():
                return
            df = pd.read_csv(path)
            if ts_col not in df.columns:
                _log_job(job, f"  {path.name} : colonne '{ts_col}' absente — ignoré", "WARN")
                return
            before = len(df)
            df = df[(df[ts_col] >= t0_ns) & (df[ts_col] <= t1_ns)].reset_index(drop=True)
            after = len(df)
            tmp = Path("/tmp") / f"__trim_{path.name}"
            df.to_csv(tmp, index=False)
            shutil.move(str(tmp), str(path))
            report["files"].append({"file": path.name, "rows_before": before, "rows_after": after})
            _log_job(job, f"  {path.name} : {before} → {after} lignes", "OK")

        # ── 2. Tronquer gripper left et right ───────────────────────────────────
        trim_csv(sess / "gripper_right_data.csv", "timestamp_ns")
        _update_job(job, progress=30.0)
        trim_csv(sess / "gripper_left_data.csv",  "timestamp_ns")
        _update_job(job, progress=45.0)

        # ── 3. Tronquer tracker ──────────────────────────────────────────────────
        trim_csv(sess / "tracker_positions.csv", "timestamp_ns")
        _update_job(job, progress=60.0)

        # ── 4. Tronquer les JSONL caméras ────────────────────────────────────────
        # capture_time est en ms → convertir t0/t1 en ms pour comparaison
        t0_ms = t0_ns / 1_000_000
        t1_ms = t1_ns / 1_000_000

        def trim_jsonl(path: Path):
            if not path.exists():
                return
            with open(path, "rb") as f:
                raw = f.read()
            frames_in = []
            for line in raw.split(b"\r\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    frames_in.append(json.loads(line.decode("utf-8")))
                except Exception:
                    continue
            frames_out = [fr for fr in frames_in if t0_ms <= fr.get("capture_time", 0) <= t1_ms]
            lines = [json.dumps(fr, separators=(",", ":")) + "\r\n" for fr in frames_out]
            tmp = Path("/tmp") / f"__trim_{path.name}"
            with open(tmp, "wb") as f:
                f.write("".join(lines).encode("utf-8"))
            shutil.move(str(tmp), str(path))
            report["files"].append({
                "file": path.name,
                "rows_before": len(frames_in),
                "rows_after":  len(frames_out),
            })
            _log_job(job, f"  {path.name} : {len(frames_in)} → {len(frames_out)} frames", "OK")

        videos_dir = sess / "videos"
        for cam in ("head", "left", "right"):
            trim_jsonl(videos_dir / f"{cam}.jsonl")
        _update_job(job, progress=90.0)

        # ── 5. Marquer dans metadata.json ────────────────────────────────────────
        meta["trim_to_sw_applied"] = True
        meta["trim_to_sw_t0_ns"]   = t0_ns
        meta["trim_to_sw_t1_ns"]   = t1_ns
        meta["trim_to_sw_duration_s"] = round(duration_s, 3)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        _log_job(job, f"✓ Trim terminé — durée conservée : {duration_s:.2f}s", "OK")
        _update_job(job, progress=100.0, status=JobStatus.DONE, ended_at=_now(), result=report)

    except Exception as e:
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=str(e))
        _log_job(job, f"Erreur trim_to_sw : {e}", "ERROR")


# ──────────────────────────────────────────────────────────────────────────────
# Rotation 180° des vidéos (depuis le viewer)
# ──────────────────────────────────────────────────────────────────────────────

class RotateVideosRequest(BaseModel):
    session: str
    force: bool = False


def _worker_rotate_videos(job: Job, req: RotateVideosRequest):
    _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=0.0)
    try:
        from utils.data_prep import rotate_session_videos
        sess_path = Path(req.session)

        def _log(msg, level="INFO"):
            _log_job(job, msg, level)

        result = rotate_session_videos(sess_path, force=req.force, log=_log)
        _update_job(job, progress=100.0, status=JobStatus.DONE, ended_at=_now(), result=result)

    except Exception as e:
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=str(e))
        _log_job(job, f"Erreur rotate_videos : {e}", "ERROR")


@app.post("/api/session/rotate_videos")
async def rotate_videos(req: RotateVideosRequest):
    """Applique la rotation 180° à toutes les vidéos d'une session."""
    if not req.session:
        raise HTTPException(400, "session manquante")
    if not Path(req.session).exists():
        raise HTTPException(404, f"Session introuvable : {req.session}")
    job = _new_job("rotate_videos")
    threading.Thread(target=_worker_rotate_videos, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


@app.post("/api/session/trim_to_sw")
async def trim_to_sw(req: TrimToSwRequest):
    """Tronque tous les fichiers de la session sur la fenêtre sw=ON → sw=OFF du gripper droit."""
    if not req.session:
        raise HTTPException(400, "session manquante")
    if not Path(req.session).exists():
        raise HTTPException(404, f"Session introuvable : {req.session}")
    job = _new_job("trim_to_sw")
    threading.Thread(target=_worker_trim_to_sw, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


# ──────────────────────────────────────────────────────────────────────────────
# Vérification alignement pinces (timestamps absolus)
# ──────────────────────────────────────────────────────────────────────────────

class CheckPincesRequest(BaseModel):
    session: str


def _worker_check_pinces(job: Job, req: CheckPincesRequest):
    _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=0.0)
    try:
        from utils.sync import (
            _PincesThresholds, _process_side_pinces,
        )
        sess_path = Path(req.session)
        thr       = _PincesThresholds()
        results   = {}

        for i, side in enumerate(("left", "right"), 1):
            _log_job(job, f"Analyse côté {side}…", "INFO")
            r = _process_side_pinces(sess_path, side, thr)
            _update_job(job, progress=i * 50.0)

            if not r.success:
                results[side] = {"status": "FAILED", "error": r.error}
                _log_job(job, f"  {side}: FAILED — {r.error[:120]}", "ERROR")
                continue

            alerts_list = [
                {"code": a.code, "level": a.level, "message": a.message,
                 "value": round(a.value, 3), "threshold": round(a.threshold, 3)}
                for a in r.alerts
            ]
            aln = r.alignment
            results[side] = {
                "status":            r.status,
                "n_errors":          r.n_errors,
                "n_warnings":        r.n_warnings,
                "alerts":            alerts_list,
                "offset_start_ms":   round(aln.offset_start_ms, 2),
                "latency_max_ms":    round(aln.latency_max_abs_ms, 2) if aln.latency_max_abs_ms == aln.latency_max_abs_ms else None,
                "latency_p95_ms":    round(aln.latency_p95_abs_ms, 2) if aln.latency_p95_abs_ms == aln.latency_p95_abs_ms else None,
                "sensor_gap_max_ms": round(aln.sensor_gap_max_ms, 1) if aln.sensor_gap_max_ms != float("inf") else None,
                "sensor_gap_count":  aln.sensor_gap_count,
                "overlap_s":         round(aln.overlap_s, 2),
                "frame_drops":       r.video.frame_drops,
                "jitter_std_ms":     round(r.video.jitter_std_ms, 3),
                "neg_dt_count":      r.sensor.neg_dt_count,
            }
            sym = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(r.status, "?")
            _log_job(job, f"  {side}: {sym} {r.status}  off={aln.offset_start_ms:+.1f}ms  "
                          f"lat_max={aln.latency_max_abs_ms:.1f}ms  "
                          f"errs={r.n_errors}  warn={r.n_warnings}",
                     "OK" if r.status == "OK" else ("WARN" if r.status == "WARNING" else "ERROR"))

        overall = "OK"
        for s in results.values():
            st = s.get("status", "FAILED")
            if st == "FAILED" or st == "ERROR":
                overall = "ERROR"; break
            if st == "WARNING":
                overall = "WARNING"

        _log_job(job, f"Vérification pinces terminée — {overall}", "OK" if overall == "OK" else "WARN")
        _update_job(job, progress=100.0, status=JobStatus.DONE, ended_at=_now(),
                    result={"overall": overall, "sides": results})
    except Exception as e:
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=str(e))
        _log_job(job, f"Erreur check_pinces : {e}", "ERROR")


@app.post("/api/session/check_pinces")
async def check_pinces(req: CheckPincesRequest):
    """Vérifie l'alignement pince/vidéo d'une session par comparaison des timestamps absolus."""
    if not req.session:
        raise HTTPException(400, "session manquante")
    if not Path(req.session).exists():
        raise HTTPException(404, f"Session introuvable : {req.session}")
    job = _new_job("check_pinces")
    threading.Thread(target=_worker_check_pinces, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


# ──────────────────────────────────────────────────────────────────────────────
# Modifier scénario / mode d'une session (déplace le dossier + met à jour metadata)
# ──────────────────────────────────────────────────────────────────────────────

class SetScenarioModeRequest(BaseModel):
    session_path: str
    scenario: str       # nouveau scénario (clé slug, ex: "pants", "towel", …)
    mode: str           # "do" ou "reset"


@app.post("/api/session/set_scenario_mode")
async def set_scenario_mode(req: SetScenarioModeRequest):
    """Déplace le dossier session vers {parent_racine}/{scenario}/{mode}/ et met à jour metadata.json."""
    if not req.session_path:
        raise HTTPException(400, "session_path manquant")
    if req.mode not in ("do", "reset"):
        raise HTTPException(400, "mode doit être 'do' ou 'reset'")
    if not req.scenario.strip():
        raise HTTPException(400, "scenario vide")

    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, f"Session introuvable : {req.session_path}")

    # Déterminer la racine (3 niveaux au-dessus si structure {root}/{scenario}/{mode}/{session})
    current_mode_dir = sess.parent
    current_scenario_dir = current_mode_dir.parent
    root_dir = current_scenario_dir.parent

    scenario_slug = req.scenario.strip()
    new_mode_dir = root_dir / scenario_slug / req.mode
    new_sess = new_mode_dir / sess.name

    if new_sess == sess:
        return {"status": "ok", "moved": False, "new_path": str(sess)}

    if new_sess.exists():
        raise HTTPException(409, f"Destination déjà occupée : {new_sess}")

    new_mode_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(sess), str(new_sess))

    # Récupérer les champs do/reset depuis MongoDB pour les mettre dans scenario
    scenario_instruction: str | None = None
    try:
        from pymongo import MongoClient as _MongoClient
        _mongo_uri  = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        _mongo_db   = os.getenv("MONGO_DB", "physical_data")
        _mongo_coll = os.getenv("MONGO_SCENARIOS_COLLECTION", "scenarios")
        _client = _MongoClient(_mongo_uri, serverSelectionTimeoutMS=3000)
        _doc = _client[_mongo_db][_mongo_coll].find_one(
            {"name": {"$regex": f"^{scenario_slug}$", "$options": "i"}},
            {"_id": 0, "do": 1, "reset": 1},
        )
        if _doc:
            scenario_instruction = _doc.get(req.mode)
        _client.close()
    except Exception:
        pass

    # Mettre à jour metadata.json
    meta_path = new_sess / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["scenario_folder"] = scenario_slug
            meta["mode"] = req.mode
            if scenario_instruction:
                meta["scenario"] = scenario_instruction
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    return {
        "status": "ok",
        "moved": True,
        "new_path": str(new_sess),
        "scenario_instruction": scenario_instruction,
    }


# ──────────────────────────────────────────────────────────────────────────────
# S3 Input Source — browse / stream / move
# ──────────────────────────────────────────────────────────────────────────────

def _s3_client(cfg: dict):
    """Crée un client boto3 à partir de la config S3 fournie par le frontend."""
    import boto3
    kwargs: dict = {"region_name": cfg.get("region") or "us-east-1"}
    if cfg.get("access_key") and cfg.get("secret_key"):
        kwargs["aws_access_key_id"]     = cfg["access_key"]
        kwargs["aws_secret_access_key"] = cfg["secret_key"]
    return boto3.client("s3", **kwargs)


class S3BrowseRequest(BaseModel):
    bucket:     str
    prefix:     str  = ""
    access_key: str  = ""
    secret_key: str  = ""
    region:     str  = "us-east-1"


class S3MoveRequest(BaseModel):
    bucket:      str
    src_prefix:  str   # ex: "sessions/my_session/"
    dest_prefix: str   # ex: "rejected/my_session/"
    access_key:  str = ""
    secret_key:  str = ""
    region:      str = "us-east-1"


class S3StreamRequest(BaseModel):
    bucket:      str
    prefix:      str           # ex: "sessions/my_session/"
    local_dir:   str           # dossier local de destination
    access_key:  str = ""
    secret_key:  str = ""
    region:      str = "us-east-1"


@app.post("/api/s3/browse")
async def s3_browse(req: S3BrowseRequest):
    """
    Liste les dossiers (prefixes) et fichiers directs sous req.prefix dans le bucket S3.
    Retourne { folders: [...], files: [...] }
    """
    try:
        s3  = _s3_client(req.dict())
        prefix = req.prefix
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        paginator = s3.get_paginator("list_objects_v2")
        folders, files = [], []

        for page in paginator.paginate(Bucket=req.bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes") or []:
                name = cp["Prefix"][len(prefix):].rstrip("/")
                if name:
                    folders.append({"name": name, "prefix": cp["Prefix"]})
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if key == prefix:
                    continue
                name = key[len(prefix):]
                if name:
                    files.append({
                        "name":          name,
                        "key":           key,
                        "size":          obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                    })

        return {"folders": folders, "files": files, "prefix": prefix, "bucket": req.bucket}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/s3/move")
async def s3_move(req: S3MoveRequest):
    """
    Déplace tous les objets de src_prefix vers dest_prefix dans le même bucket
    (copy + delete, S3 n'a pas de rename natif).
    """
    try:
        s3 = _s3_client(req.dict())
        src  = req.src_prefix  if req.src_prefix.endswith("/")  else req.src_prefix  + "/"
        dest = req.dest_prefix if req.dest_prefix.endswith("/") else req.dest_prefix + "/"

        paginator = s3.get_paginator("list_objects_v2")
        moved = []

        for page in paginator.paginate(Bucket=req.bucket, Prefix=src):
            for obj in page.get("Contents") or []:
                old_key = obj["Key"]
                new_key = dest + old_key[len(src):]
                s3.copy_object(
                    Bucket=req.bucket,
                    CopySource={"Bucket": req.bucket, "Key": old_key},
                    Key=new_key,
                )
                s3.delete_object(Bucket=req.bucket, Key=old_key)
                moved.append({"from": old_key, "to": new_key})

        return {"moved": len(moved), "objects": moved}
    except Exception as e:
        raise HTTPException(500, str(e))


def _worker_s3_stream(job: Job, req_dict: dict):
    """Thread : télécharge tous les objets d'un prefix S3 vers un dossier local."""
    try:
        import boto3
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=2)

        bucket     = req_dict["bucket"]
        prefix     = req_dict["prefix"]
        local_dir  = Path(req_dict["local_dir"])
        access_key = req_dict.get("access_key", "")
        secret_key = req_dict.get("secret_key", "")
        region     = req_dict.get("region", "us-east-1")

        kwargs: dict = {"region_name": region}
        if access_key and secret_key:
            kwargs["aws_access_key_id"]     = access_key
            kwargs["aws_secret_access_key"] = secret_key
        s3 = boto3.client("s3", **kwargs)

        if prefix and not prefix.endswith("/"):
            prefix += "/"

        # Lister tous les objets
        paginator = s3.get_paginator("list_objects_v2")
        all_keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                if obj["Key"] != prefix:
                    all_keys.append((obj["Key"], obj["Size"]))

        if not all_keys:
            raise RuntimeError(f"Aucun objet trouvé sous s3://{bucket}/{prefix}")

        _log_job(job, f"{len(all_keys)} fichiers à télécharger depuis s3://{bucket}/{prefix}", "INFO")
        total_bytes = sum(s for _, s in all_keys)

        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        done_bytes = 0

        for key, size in all_keys:
            rel = key[len(prefix):]
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            downloaded += 1
            done_bytes += size
            progress = 2 + int(95 * done_bytes / max(total_bytes, 1))
            _update_job(job, progress=progress)
            _log_job(job, f"↓ {rel}  ({size/1024:.0f} Ko)", "INFO")

        _log_job(job, f"Streaming terminé — {downloaded} fichiers → {local_dir}", "OK")
        _update_job(
            job,
            status    = JobStatus.DONE,
            ended_at  = _now(),
            progress  = 100,
            result    = {
                "bucket":      bucket,
                "prefix":      prefix,
                "local_dir":   str(local_dir),
                "n_files":     downloaded,
                "total_bytes": done_bytes,
            },
        )
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


@app.post("/api/s3/stream")
async def s3_stream(req: S3StreamRequest):
    """
    Lance le téléchargement (streaming) d'un prefix S3 complet vers un dossier local.
    Retourne un job_id pour suivre la progression.
    """
    job = _new_job("s3_stream")
    threading.Thread(
        target=_worker_s3_stream,
        args=(job, req.dict()),
        daemon=True,
    ).start()
    return {"job_id": job.id}


# ──────────────────────────────────────────────────────────────────────────────
# Inbox → Bronze
# ──────────────────────────────────────────────────────────────────────────────

class InboxPromoteRequest(BaseModel):
    session_path: str
    bronze_dir:   Optional[str] = None


def _worker_inbox_promote(job: Job, session_path: str, bronze_dir: Optional[str]):
    try:
        from pipeline.inbox_bronze import promote_session, BRONZE_DIR, _INBOX_CONFIG
        from pathlib import Path as _Path

        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=5)
        _log_job(job, f"Vérification + promotion de {_Path(session_path).name}… "
                      f"[threads={_INBOX_CONFIG.n_threads} procs={_INBOX_CONFIG.n_processes}]")

        b_dir = _Path(bronze_dir) if bronze_dir else BRONZE_DIR

        def _cb(msg, level="INFO"):
            _log_job(job, msg, level)

        report = promote_session(session_path, bronze_dir=b_dir, log_cb=_cb,
                                 config=_INBOX_CONFIG)

        _update_job(
            job,
            status   = JobStatus.DONE if (report.all_ok and report.promoted_to) else JobStatus.ERROR,
            ended_at = _now(),
            progress = 100,
            result   = report.to_dict(),
            error    = report.error,
        )
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


@app.get("/api/inbox/scan")
async def inbox_scan(inbox_dir: Optional[str] = None):
    """
    Scanne /mnt/storage/silver/ (ou inbox_dir si fourni) et retourne le rapport de vérification
    de chaque session trouvée. Parallélisé selon _INBOX_CONFIG.
    """
    try:
        from pipeline.inbox_bronze import scan_inbox, INBOX_DIR, _INBOX_CONFIG
        from pathlib import Path as _Path

        root = _Path(inbox_dir) if inbox_dir else INBOX_DIR
        reports = scan_inbox(root, config=_INBOX_CONFIG)
        return {
            "inbox_dir": str(root),
            "sessions":  [r.to_dict() for r in reports],
            "config":    _INBOX_CONFIG.to_dict(),
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/api/inbox/check")
async def inbox_check(req: dict):
    """
    Exécute les 5 checks sur une session précise (sans la déplacer).
    body: { "session_path": "/mnt/storage/silver//session_xxx" }
    """
    try:
        from pipeline.inbox_bronze import run_checks, _INBOX_CONFIG
        from pathlib import Path as _Path

        session_path = req.get("session_path", "")
        if not session_path:
            raise HTTPException(status_code=400, detail="session_path requis")

        report = run_checks(_Path(session_path), config=_INBOX_CONFIG)
        return report.to_dict()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/api/inbox/promote")
async def inbox_promote(req: InboxPromoteRequest):
    """
    Lance vérification + déplacement vers bronze en arrière-plan.
    Retourne un job_id pour suivre la progression via WebSocket.
    """
    job = _new_job("inbox_promote")
    threading.Thread(
        target=_worker_inbox_promote,
        args=(job, req.session_path, req.bronze_dir),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@app.get("/api/inbox/config")
async def inbox_config_get():
    """Retourne la configuration de concurrence inbox → bronze."""
    from pipeline.inbox_bronze import _INBOX_CONFIG
    import os as _os
    return {
        **_INBOX_CONFIG.to_dict(),
        "cpu_count": _os.cpu_count() or 1,
    }


@app.post("/api/inbox/config")
async def inbox_config_set(req: dict):
    """
    Met à jour à chaud la configuration de concurrence.
    body: { "n_threads": 4, "n_processes": 2 }
    """
    import pipeline.inbox_bronze as _ib
    import os as _os
    cpu = _os.cpu_count() or 1
    if "n_threads" in req:
        _ib._INBOX_CONFIG.n_threads   = max(1, min(int(req["n_threads"]),   cpu * 4))
    if "n_processes" in req:
        _ib._INBOX_CONFIG.n_processes = max(1, min(int(req["n_processes"]), cpu))
    return {**_ib._INBOX_CONFIG.to_dict(), "cpu_count": cpu}


# ──────────────────────────────────────────────────────────────────────────────
# Service systemd (Debian Linux) — daemon inbox → bronze 24/7
# Le serveur tourne sans privilèges (NoNewPrivileges) → il ne peut pas écrire
# dans /etc/systemd/system/ ni appeler sudo/systemctl --user.
# Ces endpoints génèrent le fichier unit + les commandes shell à coller dans
# un terminal root. Rien n'est exécuté côté serveur.
# ──────────────────────────────────────────────────────────────────────────────

_SERVICE_NAME = "syncml-inbox-bronze"
_SERVICE_FILE = Path(f"/etc/systemd/system/{_SERVICE_NAME}.service")


def _unit_content(python_bin: str, server_script: str, inbox_dir: str,
                  bronze_dir: str, host: str, port: int) -> str:
    import getpass as _gp
    user = _gp.getuser()
    return f"""[Unit]
Description=SyncML Inbox → Bronze daemon
After=network.target
Wants=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={Path(server_script).parent}
ExecStart={python_bin} {server_script} --host {host} --port {port} --root {bronze_dir} --inbox {inbox_dir}
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""


@app.get("/api/inbox/service/status")
async def inbox_service_status():
    """
    Vérifie l'état du service via systemctl (lecture seule, pas de D-Bus requis).
    """
    import subprocess as _sp

    unit_exists   = _SERVICE_FILE.exists()
    active        = False
    enabled       = False
    pid           = None
    status_output = ""

    # systemctl status/is-active est lisible sans privilèges
    try:
        r = _sp.run(["systemctl", "is-active", _SERVICE_NAME],
                    capture_output=True, text=True, timeout=5)
        active = r.stdout.strip() == "active"
    except Exception:
        pass
    try:
        r = _sp.run(["systemctl", "is-enabled", _SERVICE_NAME],
                    capture_output=True, text=True, timeout=5)
        enabled = r.stdout.strip() == "enabled"
    except Exception:
        pass
    try:
        r = _sp.run(["systemctl", "show", _SERVICE_NAME,
                     "--property=MainPID", "--value"],
                    capture_output=True, text=True, timeout=5)
        v = r.stdout.strip()
        if v and v != "0":
            pid = int(v)
    except Exception:
        pass
    try:
        r = _sp.run(["systemctl", "status", "--no-pager", "--lines=8", _SERVICE_NAME],
                    capture_output=True, text=True, timeout=5)
        status_output = r.stdout[:800]
    except Exception:
        pass

    return {
        "service_name":  _SERVICE_NAME,
        "service_file":  str(_SERVICE_FILE),
        "unit_exists":   unit_exists,
        "active":        active,
        "enabled":       enabled,
        "pid":           pid,
        "status_output": status_output,
    }


@app.post("/api/inbox/service/install")
async def inbox_service_install(req: dict):
    """
    Génère le contenu du fichier unit systemd et les commandes shell à exécuter
    en root pour installer le service. Ne touche pas au système.
    body: { inbox_dir, bronze_dir, host, port, python_bin (opt), server_script (opt) }
    """
    import shutil as _sh

    inbox_dir     = req.get("inbox_dir",     "/mnt/storage/silver/")
    bronze_dir    = req.get("bronze_dir",    "/mnt/storage/silver/")
    host          = req.get("host",          "0.0.0.0")
    port          = int(req.get("port",      8000))
    python_bin    = req.get("python_bin")    or _sh.which("python3") or sys.executable
    server_script = req.get("server_script") or str(Path(__file__).resolve())

    unit = _unit_content(python_bin, server_script, inbox_dir, bronze_dir, host, port)

    shell_cmd = (
        f"cat > {_SERVICE_FILE} << 'UNIT'\n{unit}UNIT\n"
        f"systemctl daemon-reload\n"
        f"systemctl enable {_SERVICE_NAME}\n"
        f"systemctl restart {_SERVICE_NAME}\n"
        f"systemctl --no-pager status {_SERVICE_NAME}"
    )

    return {
        "ok":           True,
        "service_file": str(_SERVICE_FILE),
        "unit_content": unit,
        "shell_cmd":    shell_cmd,
        "message":      "Copiez et exécutez les commandes ci-dessous en root pour installer le service.",
    }


@app.post("/api/inbox/service/uninstall")
async def inbox_service_uninstall():
    """Génère les commandes shell à exécuter en root pour désinstaller le service."""
    shell_cmd = (
        f"systemctl stop {_SERVICE_NAME}\n"
        f"systemctl disable {_SERVICE_NAME}\n"
        f"rm -f {_SERVICE_FILE}\n"
        f"systemctl daemon-reload"
    )
    return {
        "ok":        True,
        "shell_cmd": shell_cmd,
        "message":   "Exécutez ces commandes en root pour désinstaller le service.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scripts personnalisés
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/scripts/list")
async def scripts_list():
    """Liste les scripts .py disponibles à la racine du projet (hors .venv, __pycache__)."""
    scripts = []
    for p in sorted(_ROOT.rglob("*.py")):
        # Exclure .venv, __pycache__, server/, et fichiers __init__
        parts = p.relative_to(_ROOT).parts
        if any(part in (".venv", "__pycache__", ".git") for part in parts):
            continue
        if p.name == "__init__.py":
            continue
        rel = str(p.relative_to(_ROOT))
        scripts.append({
            "path": rel,
            "name": p.stem,
            "label": rel,
        })
    return {"scripts": scripts}


@app.post("/api/scripts/run")
async def scripts_run(req: dict):
    """
    Lance un ou plusieurs scripts sur une liste de sessions.
    body: { sessions: [str], scripts: [str] }
    """
    import subprocess as _sp
    import importlib.util as _ilu

    session_paths = req.get("sessions", [])
    script_rels   = req.get("scripts", [])

    if not session_paths or not script_rels:
        raise HTTPException(400, "sessions et scripts requis")

    job_id = str(uuid.uuid4())[:8]
    job    = Job(id=job_id, kind="scripts")
    with _jobs_lock:
        _jobs[job_id] = job

    def _run():
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=5)
        total = len(session_paths) * len(script_rels)
        done  = 0
        for sess in session_paths:
            for rel in script_rels:
                script_path = _ROOT / rel
                if not script_path.exists():
                    _log_job(job, f"Script introuvable : {rel}", level="ERROR")
                    done += 1
                    continue
                try:
                    _log_job(job, f"[{Path(sess).name}] → {rel}")
                    result = _sp.run(
                        [sys.executable, str(script_path), sess],
                        capture_output=True, text=True, timeout=300
                    )
                    if result.stdout:
                        _log_job(job, result.stdout.strip()[:500])
                    if result.returncode != 0:
                        _log_job(job, result.stderr.strip()[:300] or "Erreur", level="ERROR")
                    else:
                        _log_job(job, f"OK : {rel}", level="OK")
                except Exception as e:
                    _log_job(job, f"Exception: {e}", level="ERROR")
                done += 1
                _update_job(job, progress=5 + 90 * done // total)
        _update_job(job, status=JobStatus.DONE, ended_at=_now(), progress=100)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    async with _ws_lock:
        _ws_clients.append(ws)
    # Snapshot initial de tous les jobs
    with _jobs_lock:
        snapshot = [_job_to_dict(j) for j in _jobs.values()]
    await ws.send_json({"type": "snapshot", "jobs": snapshot})
    try:
        while True:
            # Garder la connexion vivante (ping côté client)
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


# ──────────────────────────────────────────────────────────────────────────────
# Download sessions (ZIP)
# ──────────────────────────────────────────────────────────────────────────────

class DownloadSessionsRequest(BaseModel):
    sessions:    List[str]   # chemins absolus des sessions
    include_mp4: bool = True # inclure les fichiers MP4


@app.post("/api/session/download")
async def download_sessions(req: DownloadSessionsRequest):
    """
    Crée un ZIP de toutes les sessions sélectionnées et le retourne en streaming.
    - MP4  → ZIP_STORED  (déjà compressés, DEFLATE serait lent et inutile)
    - Reste → ZIP_DEFLATED
    Les fichiers MP4 peuvent être exclus via include_mp4=false.
    Un fichier _video_report.json est ajouté au ZIP indiquant l'état des vidéos
    par session (présentes / manquantes).
    """
    import tempfile
    import zipfile

    VIDEO_POSITIONS = {"left", "right", "head"}

    sessions = [Path(p) for p in req.sessions if Path(p).exists()]
    if not sessions:
        raise HTTPException(404, "Aucune session trouvée")

    # ── Vérification des vidéos par session ──────────────────────────────────
    video_report: list = []
    for sess in sessions:
        vid_dir = sess / "videos"
        present: list = []
        missing: list = []
        for pos in sorted(VIDEO_POSITIONS):
            mp4 = vid_dir / f"{pos}.mp4"
            if mp4.exists():
                present.append(pos)
            else:
                missing.append(pos)
        video_report.append({
            "session":  sess.name,
            "present":  present,
            "missing":  missing,
            "ok":       len(missing) == 0,
        })

    # Nom du fichier ZIP
    zip_name = f"{sessions[0].name}.zip" if len(sessions) == 1 \
               else f"sessions_{len(sessions)}.zip"

    # Écriture dans un fichier temporaire (évite de tout charger en RAM)
    tmp_dir = Path("/mnt/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=tmp_dir)
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        with zipfile.ZipFile(tmp_path, mode="w", allowZip64=True) as zf:
            # Rapport vidéo en tête de ZIP
            report_json = json.dumps(video_report, ensure_ascii=False, indent=2)
            zf.writestr("_video_report.json", report_json, compress_type=zipfile.ZIP_DEFLATED)

            for sess in sessions:
                for f in sorted(sess.rglob("*")):
                    if not f.is_file():
                        continue
                    if not req.include_mp4 and f.suffix.lower() == ".mp4":
                        continue
                    compression = (zipfile.ZIP_STORED
                                   if f.suffix.lower() in (".mp4", ".mkv", ".avi")
                                   else zipfile.ZIP_DEFLATED)
                    arcname = f.relative_to(sess.parent)
                    zf.write(str(f), str(arcname), compress_type=compression)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Erreur ZIP : {e}")

    # Résumé dans l'en-tête HTTP (sessions sans vidéos)
    sessions_missing = [r["session"] for r in video_report if r["missing"]]
    missing_header = ",".join(sessions_missing) if sessions_missing else ""

    # Streaming par chunks de 4 Mo + suppression du temp après envoi
    def _stream_and_cleanup():
        try:
            with open(tmp_path, "rb") as fh:
                while True:
                    chunk = fh.read(4 * 1024 * 1024)  # 4 Mo
                    if not chunk:
                        break
                    yield chunk
        finally:
            tmp_path.unlink(missing_ok=True)

    headers = {"Content-Disposition": f'attachment; filename="{zip_name}"'}
    if missing_header:
        headers["X-Missing-Videos"] = missing_header

    return StreamingResponse(
        _stream_and_cleanup(),
        media_type="application/zip",
        headers=headers,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SyncML Studio — Serveur web")
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      type=int, default=8000)
    p.add_argument("--reload",    action="store_true")
    # Arguments legacy ignorés (chemins désormais hardcodés dans pipeline.py)
    p.add_argument("--root",      default=None, help=argparse.SUPPRESS)
    p.add_argument("--model-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--nas",       default=None, help=argparse.SUPPRESS)
    p.add_argument("--watch",     default=None, help=argparse.SUPPRESS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        "server:app",
        host   = args.host,
        port   = args.port,
        reload = args.reload,
    )
