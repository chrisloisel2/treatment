#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SyncML Studio — Serveur web FastAPI.

Architecture 3 chemins :
  /mnt/storage/bronze  → source brute, lecture seule
  /mnt/storage/bronze/    → espace de travail (copie de travail)
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
    INGEST_DIR = Path("/mnt/storage/bronze")
    SILVER_DIR = Path("/home/ia/silver")
    MODEL_DIR  = INGEST_DIR / "_sync_ml_model"

DEFAULT_WATCH_DIR = "/mnt/storage/bronze"

# Répertoire de persistance des jobs sur disque
JOBS_DIR = INGEST_DIR / "_server_jobs"

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
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
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


def _log_job(job: Job, msg: str, level: str = "INFO"):
    entry = {"ts": _now(), "msg": msg, "level": level}
    job.logs.append(entry)
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


def _job_to_dict(job: Job) -> dict:
    return {
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
    }


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

def _worker_scan(job: Job):
    try:
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=10)
        _log_job(job, f"Scan de {INGEST_DIR}…")

        import utils.sync as ia
        # Découverte dans /mnt/storage/bronze/ (sessions déposées par l'opérateur)
        sessions = [
            s for s in (INGEST_DIR.iterdir() if INGEST_DIR.exists() else [])
            if s.is_dir() and not s.name.startswith("_")
        ]
        model_exists = (MODEL_DIR / "model.pt").exists()
        result = []
        for s in sorted(sessions, key=lambda p: p.name):
            meta_path = s / "metadata.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    pass
            has_tracker  = (s / "tracker_positions.csv").exists()
            has_gripper  = (s / "gripper_left_data.csv").exists() or (s / "gripper_right_data.csv").exists()
            has_ux       = (s / "ux_data.csv").exists()
            has_flux_csv = any((s / "videos").glob("*_flux.csv")) if (s / "videos").exists() else False
            has_jsonl    = any((s / "videos").glob("*.jsonl"))    if (s / "videos").exists() else False
            has_subtitle = (s / "episode_subtitle.json").exists()
            # Statut pipeline
            pipeline_steps = {}
            pipeline_done  = False
            ps_path = s / "pipeline_state.json"
            if ps_path.exists():
                try:
                    ps = json.loads(ps_path.read_text())
                    pipeline_steps = ps.get("steps", {})
                    pipeline_done  = ps.get("finished", False) and ps.get("success", False)
                except Exception:
                    pass
            result_json = s / ia.RESULTS_JSON if hasattr(ia, "RESULTS_JSON") else None
            last_result = None
            if result_json and result_json.exists():
                try:
                    last_result = json.loads(result_json.read_text())
                except Exception:
                    pass

            result.append({
                "name":           s.name,
                "path":           str(s),
                "has_tracker":    has_tracker,
                "has_gripper":    has_gripper,
                "has_ux":         has_ux,
                "has_flux_csv":   has_flux_csv,
                "has_jsonl":      has_jsonl,
                "has_subtitle":   has_subtitle,
                "meta":           meta,
                "last_result":    last_result,
                "pipeline_done":  pipeline_done,
                "pipeline_steps": pipeline_steps,
            })

        _log_job(job, f"{len(result)} sessions trouvées dans {INGEST_DIR}.", "OK")
        _update_job(
            job,
            status    = JobStatus.DONE,
            ended_at  = _now(),
            progress  = 100,
            result    = {
                "sessions":     result,
                "ingest_dir":   str(INGEST_DIR),
                "silver_dir":   str(SILVER_DIR),
                "model_exists": model_exists,
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

# ── Watcher singleton ──
_watcher: Optional["IngestionWatcher"] = None  # type: ignore[name-defined]

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
    # Le watcher NE démarre PAS automatiquement.
    # Il doit être activé explicitement depuis l'interface (POST /api/watcher/start).


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


def _worker_watcher_scan(job: "Job", params: dict, write_mode: bool,  # type: ignore[name-defined]
                          delete_after_store: bool, auto_start: bool,
                          watch_dir: str = DEFAULT_WATCH_DIR):
    """Worker qui démarre le watcher sur le répertoire configuré."""
    global _watcher
    try:
        from pipeline.pipeline import IngestionWatcher
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=10)

        def _log(msg, level="INFO"):
            _log_job(job, msg, level)

        _watcher = IngestionWatcher(
            params             = params,
            write_mode         = write_mode,
            delete_after_store = delete_after_store,
            log_callback       = _log,
            poll_interval      = 8.0,
            auto_start         = auto_start,
            watch_dir          = watch_dir,
        )
        _watcher.start()
        _log(f"Watcher démarré sur {watch_dir}", "OK")
        _update_job(job, status=JobStatus.DONE, ended_at=_now(), progress=100,
                    result={"watch_dir": watch_dir, "auto_start": auto_start,
                            "write_mode": write_mode, "delete_after_store": delete_after_store})
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
    session:       str        # chemin dans /mnt/storage/bronze/
    apply:         bool  = False
    dry_run:       bool  = True
    resample_ms:   float = 5.0
    max_lag_ms:    float = 400.0
    window_ms:     float = 2200.0
    signal_config: Optional[Dict[str, List[str]]] = None

class PipelineRunRequest(BaseModel):
    session:            str          # nom ou chemin de session dans /mnt/storage/bronze/
    write_mode:         bool  = False
    delete_after_store: bool  = False
    force_flux:         bool  = False
    resample_ms:        float = 5.0
    max_lag_ms:         float = 400.0
    window_ms:          float = 2200.0
    steps:              Optional[List[str]] = None  # None = toutes les étapes

class WatcherStartRequest(BaseModel):
    watch_dir:          str   = DEFAULT_WATCH_DIR
    write_mode:         bool  = False
    delete_after_store: bool  = False
    auto_start:         bool  = False
    resample_ms:        float = 5.0
    max_lag_ms:         float = 400.0
    window_ms:          float = 2200.0

class PipelineStateRequest(BaseModel):
    session_path: str


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(_static_dir / "index.html"))


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
async def scan():
    """Lance un scan asynchrone de /mnt/storage/bronze."""
    job = _new_job("scan")
    threading.Thread(target=_worker_scan, args=(job,), daemon=True).start()
    return {"job_id": job.id}


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
    """Lance un entraînement asynchrone sur les sessions de /mnt/storage/bronze/."""
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
    """Lance une inférence asynchrone sur une session de /mnt/storage/bronze/."""
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
    """Lance la pipeline complète (9 étapes) sur une session de /mnt/storage/bronze/."""
    params = {
        "resample_ms": req.resample_ms,
        "max_lag_ms":  req.max_lag_ms,
        "window_ms":   req.window_ms,
    }
    # req.session peut être un nom ou un chemin complet dans /mnt/storage/bronze/
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
    """Lance la pipeline sur plusieurs sessions de /mnt/storage/bronze/ en parallèle."""
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
    """Worker thread : exécute pipeline_align_pro.align_session en sous-processus."""
    import subprocess, sys
    try:
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=5)
        _log_job(job, f"align_pro démarré sur {sess.name}", "INFO")

        script = Path(__file__).resolve().parent.parent / "pipeline_align_pro.py"
        cmd = [sys.executable, str(script), str(sess)]
        if force:
            cmd.append("--force")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                _log_job(job, line, "INFO")

        proc.wait()
        if proc.returncode == 0:
            _update_job(job, status=JobStatus.DONE, ended_at=_now(), progress=100)
            _log_job(job, "align_pro terminé avec succès", "OK")
        else:
            _update_job(job, status=JobStatus.ERROR, ended_at=_now(),
                        error=f"align_pro a retourné code {proc.returncode}")
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


@app.post("/api/watcher/start")
async def watcher_start(req: WatcherStartRequest):
    """Démarre le watcher d'ingestion automatique sur le répertoire configuré."""
    global _watcher
    if _watcher and _watcher._running:
        _watcher.stop()
        _watcher = None
    params = {
        "resample_ms": req.resample_ms,
        "max_lag_ms":  req.max_lag_ms,
        "window_ms":   req.window_ms,
    }
    job = _new_job("watcher")
    threading.Thread(
        target = _worker_watcher_scan,
        args   = (job, params, req.write_mode, req.delete_after_store, req.auto_start,
                  req.watch_dir),
        daemon = True,
    ).start()
    return {"job_id": job.id}


@app.post("/api/watcher/stop")
async def watcher_stop():
    global _watcher
    if _watcher:
        _watcher.stop()
        _watcher = None
        return {"stopped": True}
    return {"stopped": False}


@app.get("/api/watcher/status")
async def watcher_status():
    if _watcher is None:
        return {"running": False}
    return {
        "running":   _watcher._running,
        "watch_dir": str(_watcher.watch_dir),
        "queue":     _watcher.get_queue(),
        "seen":      len(_watcher.get_seen()),
    }


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

def _load_session_timeseries(session_path: str) -> dict:
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
    """
    import pandas as pd

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
    start_ns = int(meta.get("start_time_ns", 0))
    result["start_ns"] = start_ns

    def _ns_to_ms(ns_arr):
        """Convertit des timestamps ns en ms depuis start_ns."""
        return ((np.asarray(ns_arr, dtype=np.float64) - start_ns) / 1_000_000).tolist()

    def _epoch_ms_to_ms(epoch_ms_arr):
        """Convertit des timestamps epoch-ms en ms depuis start_ns."""
        start_ms = start_ns / 1_000_000
        return (np.asarray(epoch_ms_arr, dtype=np.float64) - start_ms).tolist()

    # ── Trackers ──
    trk_path = sess / "tracker_positions.csv"
    if trk_path.exists():
        df = pd.read_csv(trk_path)
        t_ms = _ns_to_ms(df["timestamp_ns"].to_numpy())
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
        grip_path = sess / f"gripper_{side}_data.csv"
        if grip_path.exists():
            df = pd.read_csv(grip_path)
            # t_ms est relatif à start_time_ns dans certaines versions
            # On utilise timestamp_ns si disponible, sinon t_ms corrigé
            if "timestamp_ns" in df.columns:
                t_ms = _ns_to_ms(df["timestamp_ns"].to_numpy())
            elif "t_ms_corrected_ns" in df.columns:
                t_ms = _ns_to_ms(df["t_ms_corrected_ns"].to_numpy())
            else:
                # t_ms brut depuis début
                t_ms = df["time_seconds"].to_numpy(dtype=float) * 1000
                t_ms = t_ms.tolist()
            grip: dict = {"t_ms": t_ms}
            if "opening_mm" in df.columns:
                grip["opening_mm"] = df["opening_mm"].tolist()
            if "angle_deg" in df.columns:
                grip["angle_deg"] = df["angle_deg"].tolist()
            result[f"gripper_{side}"] = grip

    # ── JSONL vidéo (timestamps frames) ──
    vid_dir = sess / "videos"
    if vid_dir.exists():
        for side in ("head", "left", "right"):
            jl = vid_dir / f"{side}.jsonl"
            if jl.exists():
                rows = _parse_jsonl(jl)
                if rows:
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

        # Lire start_ns depuis metadata pour la conversion
        meta_path = sess / "metadata.json"
        start_ns = 0
        if meta_path.exists():
            start_ns = int(json.loads(meta_path.read_text()).get("start_time_ns", 0))

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
async def session_data(session_path: str):
    """
    Retourne toutes les séries temporelles alignées d'une session.
    Utilisé par l'onglet Visualisation.
    """
    sess = Path(session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")
    try:
        data = _load_session_timeseries(session_path)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/session/video/{side}")
async def session_video(side: str, session_path: str, request: Request):
    """
    Sert le fichier mp4 brut avec support HTTP Range pour lecture native <video>.
    side : head | left | right
    """
    if side not in ("head", "left", "right"):
        raise HTTPException(400, "side doit être head, left ou right")
    sess = Path(session_path)
    mp4 = sess / "videos" / f"{side}.mp4"
    if not mp4.exists():
        raise HTTPException(404, f"Vidéo {side} introuvable")

    file_size = mp4.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end"
        try:
            ranges = range_header.replace("bytes=", "").split("-")
            start = int(ranges[0])
            end   = int(ranges[1]) if ranges[1] else file_size - 1
        except Exception:
            raise HTTPException(416, "Range invalide")
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def _iter_file(path, s, length):
            with open(path, "rb") as f:
                f.seek(s)
                remaining = length
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            _iter_file(mp4, start, chunk_size),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range":  f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges":  "bytes",
                "Content-Length": str(chunk_size),
                "Cache-Control":  "no-cache",
            },
        )

    # Pas de Range → envoyer tout
    return FileResponse(str(mp4), media_type="video/mp4", headers={
        "Accept-Ranges":  "bytes",
        "Content-Length": str(file_size),
        "Cache-Control":  "no-cache",
    })


@app.get("/api/session/video_info")
async def session_video_info(session_path: str):
    """
    Retourne pour chaque caméra le t0 en secondes (offset entre start_time_ns de la session
    et la première capture_time du JSONL), pour synchroniser video.currentTime avec la timeline.
    """
    sess = Path(session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")

    meta_path = sess / "metadata.json"
    start_ns = 0
    if meta_path.exists():
        try:
            start_ns = int(json.loads(meta_path.read_text()).get("start_time_ns", 0))
        except Exception:
            pass

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
        # offset = temps de la première frame relative à start_ns de session (en secondes)
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
        t_ms = t_start
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

            t_ms += interval * 1000.0
            if t_ms >= duration_ms:
                yield f"data: {json.dumps({'done': True, 't_ms': duration_ms})}\n\n"
                return

            await asyncio.sleep(interval)

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
    start_ns = int(meta.get("start_time_ns", 0))
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
    avec conversion optionnelle au format LeRobot v3."""
    import shutil

    sessions = req.sessions
    total = len(sessions)
    _log_job(job, f"Export de {total} session(s) — destination: {req.dest_type}", "INFO")
    _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=0.0)

    exported = []
    errors = []

    for i, sess_path in enumerate(sessions):
        sess = Path(sess_path)
        if not sess.exists():
            errors.append(f"{sess_path}: introuvable")
            _log_job(job, f"[{i+1}/{total}] SKIP {sess.name} — dossier introuvable", "WARN")
            continue

        sess_name = sess.name
        _log_job(job, f"[{i+1}/{total}] Export {sess_name}…", "INFO")

        try:
            # ── Conversion LeRobot v3 ────────────────────────────────────────
            if req.lerobot and req.lerobot.enabled:
                sess_path = _convert_lerobot(job, sess, req.lerobot)

            # ── Destination ─────────────────────────────────────────────────
            if req.dest_type == "local":
                _export_local(job, Path(sess_path), req.dest_local, sess_name)
            elif req.dest_type == "s3":
                _export_s3(job, Path(sess_path), req.dest_s3, sess_name)
            elif req.dest_type == "sftp":
                _export_sftp(job, Path(sess_path), req.dest_sftp, sess_name)
            else:
                raise ValueError(f"dest_type inconnu: {req.dest_type}")

            exported.append(sess_name)
            _log_job(job, f"[{i+1}/{total}] ✓ {sess_name}", "OK")

        except Exception as e:
            errors.append(f"{sess_name}: {e}")
            _log_job(job, f"[{i+1}/{total}] ✗ {sess_name}: {e}", "ERROR")

        _update_job(job, progress=round((i + 1) / total * 100, 1))

    result = {"exported": exported, "errors": errors, "total": total}
    status = JobStatus.DONE if not errors else JobStatus.ERROR
    err_msg = "; ".join(errors) if errors else None
    _update_job(job, status=status, ended_at=_now(), progress=100.0, result=result, error=err_msg)
    _log_job(job, f"Export terminé — {len(exported)}/{total} OK, {len(errors)} erreur(s)", "OK" if not errors else "WARN")


def _convert_lerobot(job: Job, sess: Path, opts: LeRobotOptions) -> str:
    """Convertit une session au format LeRobot v3 HuggingFace dans un dossier temp."""
    import tempfile, shutil

    _log_job(job, f"  → Conversion LeRobot v3 : {sess.name}", "INFO")
    tmp_dir = Path(tempfile.mkdtemp(prefix="lerobot_"))
    dataset_dir = tmp_dir / opts.dataset_name / sess.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # ── Structure LeRobot v3 ─────────────────────────────────────────────────
    # meta/info.json
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    # Chercher les fichiers CSV de la session
    csv_files = list(sess.glob("*.csv"))
    video_files = list(sess.glob("*.mp4"))

    # Construire info.json (format LeRobot v3)
    features: dict = {}
    for csv_f in csv_files:
        col_name = csv_f.stem.replace("-", "_").replace(" ", "_")
        features[col_name] = {"dtype": "float32", "shape": [1], "names": None}
    for vid_f in video_files:
        feat_key = f"observation.images.{vid_f.stem}"
        features[feat_key] = {
            "dtype": "video",
            "shape": [opts.fps, 480, 640, 3],
            "names": ["time", "height", "width", "channel"],
            "video_info": {"video.fps": opts.fps, "video.codec": "av1", "video.pix_fmt": "yuv420p"}
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": opts.robot_type,
        "total_episodes": 1,
        "total_frames": 0,
        "total_tasks": 1,
        "total_videos": len(video_files),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": opts.fps,
        "splits": {"train": f"0:{1}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    # tasks.jsonl
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": f"episode from {sess.name}"}) + "\n",
        encoding="utf-8"
    )

    # episodes.jsonl
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": [0], "length": 0}) + "\n",
        encoding="utf-8"
    )

    # Copier les vidéos dans videos/chunk-000/<key>/episode_000000.mp4
    if video_files:
        for vid_f in video_files:
            vid_key = f"observation.images.{vid_f.stem}"
            vid_out = dataset_dir / "videos" / "chunk-000" / vid_key
            vid_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vid_f, vid_out / "episode_000000.mp4")
            _log_job(job, f"    vidéo copiée : {vid_f.name}", "INFO")

    # Copier les CSV et autres fichiers de données dans data/
    data_dir = dataset_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    for f in csv_files:
        shutil.copy2(f, data_dir / f.name)

    # Copier les fichiers JSON de métadonnées
    for jf in sess.glob("*.json"):
        shutil.copy2(jf, meta_dir / jf.name)

    # Push HuggingFace Hub si demandé
    if opts.push_to_hub and opts.hf_token and opts.repo_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=opts.hf_token)
            api.create_repo(repo_id=opts.repo_id, repo_type="dataset", exist_ok=True)
            api.upload_folder(folder_path=str(dataset_dir), repo_id=opts.repo_id, repo_type="dataset")
            _log_job(job, f"    ✓ Poussé sur HF Hub : {opts.repo_id}", "OK")
        except Exception as e:
            _log_job(job, f"    ✗ HF Hub push échoué : {e}", "WARN")

    return str(dataset_dir)


def _export_local(job: Job, src: Path, dest: ExportDestLocal, sess_name: str):
    import shutil
    out = Path(dest.path) / sess_name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(str(src), str(out))
    _log_job(job, f"    → local: {out}", "INFO")


def _export_s3(job: Job, src: Path, dest: ExportDestS3, sess_name: str):
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 non installé — pip install boto3")
    session = boto3.Session(
        aws_access_key_id=dest.access_key,
        aws_secret_access_key=dest.secret_key,
        region_name=dest.region,
    )
    s3 = session.client("s3")
    prefix = (dest.prefix.rstrip("/") + "/" + sess_name + "/").lstrip("/")
    n = 0
    for f in src.rglob("*"):
        if f.is_file():
            key = prefix + str(f.relative_to(src)).replace("\\", "/")
            s3.upload_file(str(f), dest.bucket, key)
            n += 1
    _log_job(job, f"    → S3 s3://{dest.bucket}/{prefix} ({n} fichiers)", "INFO")


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
