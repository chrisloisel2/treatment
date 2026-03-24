#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SyncML Studio — Serveur web FastAPI.

Intégration dans une pipeline big data :
  - API REST JSON pour automatisation / orchestrateurs (Airflow, Prefect, etc.)
  - WebSocket pour streaming temps-réel des logs et progress
  - Endpoints de statut machine-readable pour health checks
  - Tous les jobs tournent dans des threads séparés, non-bloquants

Lancement :
  python server.py [--root /path/to/sessions] [--host 0.0.0.0] [--port 8000]
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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


# Registre en mémoire des jobs
_jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()

# WebSocket : liste des clients connectés
_ws_clients: List[WebSocket] = []
_ws_lock = asyncio.Lock()

# Root par défaut (modifiable via CLI ou API)
_root_dir: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _new_job(kind: str) -> Job:
    j = Job(id=str(uuid.uuid4())[:8], kind=kind)
    with _jobs_lock:
        _jobs[j.id] = j
    return j


def _log_job(job: Job, msg: str, level: str = "INFO"):
    entry = {"ts": _now(), "msg": msg, "level": level}
    job.logs.append(entry)
    # Broadcast async (non-bloquant depuis un thread)
    asyncio.run_coroutine_threadsafe(
        _broadcast({"type": "log", "job_id": job.id, "entry": entry}),
        _loop,
    )


def _update_job(job: Job, **kwargs):
    for k, v in kwargs.items():
        setattr(job, k, v)
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

def _worker_scan(job: Job, root: str):
    try:
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=10)
        _log_job(job, f"Scan de {root}…")

        import IA as ia
        sessions = ia.discover_sessions(Path(root), None)
        result = []
        for s in sessions:
            meta_path = s / "metadata.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    pass
            has_tracker  = (s / "tracker_positions.csv").exists()
            has_gripper  = (s / "gripper_left_data.csv").exists() or (s / "gripper_right_data.csv").exists()
            has_flux_csv = any((s / "videos").glob("*_flux.csv")) if (s / "videos").exists() else False
            has_jsonl    = any((s / "videos").glob("*.jsonl"))    if (s / "videos").exists() else False
            model_exists = (Path(root) / ia.MODEL_DIRNAME / "model.pt").exists()
            result_json  = s / ia.RESULTS_JSON
            last_result  = None
            if result_json.exists():
                try:
                    last_result = json.loads(result_json.read_text())
                except Exception:
                    pass

            result.append({
                "name":         s.name,
                "path":         str(s),
                "has_tracker":  has_tracker,
                "has_gripper":  has_gripper,
                "has_flux_csv": has_flux_csv,
                "has_jsonl":    has_jsonl,
                "meta":         meta,
                "last_result":  last_result,
            })

        _log_job(job, f"{len(result)} sessions trouvées.", "OK")
        _update_job(
            job,
            status    = JobStatus.DONE,
            ended_at  = _now(),
            progress  = 100,
            result    = {"sessions": result, "root": root, "model_exists": model_exists},
        )
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


def _worker_train(job: Job, root: str, sessions: List[str], params: dict):
    try:
        import IA as ia
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

        model_dir = Path(root) / ia.MODEL_DIRNAME
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


def _worker_infer(job: Job, root: str, session: str, params: dict, apply: bool, dry_run: bool):
    try:
        import IA as ia

        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=5)
        _log_job(job, f"Inférence sur {Path(session).name}…")

        root_p    = Path(root)
        sess_p    = Path(session)
        model_dir = root_p / ia.MODEL_DIRNAME

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
_nas_root:   str = ""
_model_dir:  str = ""

# Servir le frontend statique
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── Event loop référence (pour run_coroutine_threadsafe depuis threads) ──
_loop: asyncio.AbstractEventLoop = None

@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_running_loop()


# ──────────────────────────────────────────────────────────────────────────────
# Worker pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _pipeline_log_cb(job: "Job") -> "Callable":  # type: ignore[name-defined]
    def _cb(msg: str, level: str = "INFO"):
        _log_job(job, msg, level)
    return _cb


def _pipeline_step_cb(job: "Job") -> "Callable":  # type: ignore[name-defined]
    def _cb(state):
        from pipeline import SessionPipelineState, StepStatus
        # Calculer progression basée sur étapes complétées
        step_names = ["detect", "rotate", "tracker", "video", "flux_csv", "ia_sync", "validate", "store"]
        done = sum(
            1 for n in step_names
            if state.steps.get(n) and state.steps[n].status
               in (StepStatus.DONE, StepStatus.SKIPPED)
        )
        progress = int(done / len(step_names) * 100)
        # Sérialiser l'état pipeline
        steps_serial = {
            n: {
                "status":     str(state.steps[n].status) if n in state.steps else "pending",
                "duration_s": state.steps[n].duration_s if n in state.steps else 0,
                "message":    state.steps[n].message    if n in state.steps else "",
            }
            for n in step_names
        }
        _update_job(
            job,
            progress = progress,
            result   = {
                "session":    state.session_name,
                "steps":      steps_serial,
                "n_reliable": state.n_reliable,
                "mean_conf":  state.mean_conf,
                "nas_path":   state.nas_path,
                "success":    state.success,
                "error":      state.error,
            } if state.finished else {
                "session": state.session_name,
                "steps":   steps_serial,
                "current": state.current_step,
            },
        )
        # Broadcast dédié pour les étapes pipeline
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "pipeline_step", "job_id": job.id, "steps": steps_serial,
                        "current": state.current_step, "finished": state.finished,
                        "success": getattr(state, "success", False)}),
            _loop,
        )
    return _cb


def _worker_pipeline(job: "Job", session: str, root: str, model_dir: str,  # type: ignore[name-defined]
                     nas_root: str, params: dict, force_flux: bool):
    try:
        from pipeline import PipelineRunner
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=2)
        runner = PipelineRunner(
            session_path   = session,
            ingestion_root = root,
            model_dir      = model_dir,
            nas_root       = nas_root,
            params         = params,
            log_callback   = _pipeline_log_cb(job),
            step_callback  = _pipeline_step_cb(job),
            force_flux     = force_flux,
            resume         = True,
        )
        state = runner.run()
        final_status = JobStatus.DONE if state.success else JobStatus.ERROR
        _update_job(
            job,
            status   = final_status,
            ended_at = _now(),
            progress = 100 if state.success else job.progress,
            error    = state.error,
        )
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


def _worker_watcher_scan(job: "Job", watch_dir: str, model_dir: str, nas_root: str,  # type: ignore[name-defined]
                          params: dict, auto_start: bool):
    """Worker qui démarre le watcher et écoute les nouvelles sessions."""
    global _watcher
    try:
        from pipeline import IngestionWatcher
        _update_job(job, status=JobStatus.RUNNING, started_at=_now(), progress=10)

        def _log(msg, level="INFO"):
            _log_job(job, msg, level)

        _watcher = IngestionWatcher(
            watch_dir    = watch_dir,
            model_dir    = model_dir,
            nas_root     = nas_root,
            params       = params,
            log_callback = _log,
            poll_interval= 8.0,
            auto_start   = auto_start,
        )
        _watcher.start()
        _log(f"Watcher démarré sur {watch_dir}", "OK")
        _update_job(job, status=JobStatus.DONE, ended_at=_now(), progress=100,
                    result={"watch_dir": watch_dir, "auto_start": auto_start})
    except Exception:
        err = traceback.format_exc()
        _log_job(job, err, "ERROR")
        _update_job(job, status=JobStatus.ERROR, ended_at=_now(), error=err)


# ──────────────────────────────────────────────────────────────────────────────
# Modèles Pydantic
# ──────────────────────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    root: str

class TrainRequest(BaseModel):
    root:          str
    sessions:      List[str]
    epochs:        int   = 18
    batch_size:    int   = 64
    lr:            float = 1e-3
    resample_ms:   float = 5.0
    max_lag_ms:    float = 400.0
    window_ms:     float = 2200.0
    signal_config: Optional[Dict[str, List[str]]] = None  # {flux_name: [col1, col2]}

class InferRequest(BaseModel):
    root:          str
    session:       str
    apply:         bool  = False
    dry_run:       bool  = True
    resample_ms:   float = 5.0
    max_lag_ms:    float = 400.0
    window_ms:     float = 2200.0
    signal_config: Optional[Dict[str, List[str]]] = None

class RootRequest(BaseModel):
    root: str

class PipelineRunRequest(BaseModel):
    session:     str
    root:        str
    model_dir:   str         = "_sync_ml_model"
    nas_root:    str         = ""
    force_flux:  bool        = False
    resample_ms: float       = 5.0
    max_lag_ms:  float       = 400.0
    window_ms:   float       = 2200.0

class WatcherStartRequest(BaseModel):
    watch_dir:   str
    model_dir:   str  = "_sync_ml_model"
    nas_root:    str  = ""
    auto_start:  bool = False
    resample_ms: float = 5.0
    max_lag_ms:  float = 400.0
    window_ms:   float = 2200.0

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
        import IA as ia
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
async def scan(req: ScanRequest):
    """Lance un scan asynchrone du répertoire racine."""
    global _root_dir
    _root_dir = req.root
    job = _new_job("scan")
    threading.Thread(target=_worker_scan, args=(job, req.root), daemon=True).start()
    return {"job_id": job.id}


@app.post("/api/train")
async def train(req: TrainRequest):
    """Lance un entraînement asynchrone."""
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
        args=(job, req.root, req.sessions, params),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@app.post("/api/infer")
async def infer(req: InferRequest):
    """Lance une inférence asynchrone sur une session."""
    params = {
        "resample_ms":   req.resample_ms,
        "max_lag_ms":    req.max_lag_ms,
        "window_ms":     req.window_ms,
        "signal_config": req.signal_config,
    }
    job = _new_job("infer")
    threading.Thread(
        target=_worker_infer,
        args=(job, req.root, req.session, params, req.apply, req.dry_run),
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
    """Lance la pipeline complète (7 étapes) sur une session."""
    global _model_dir, _nas_root
    _model_dir = req.model_dir
    _nas_root  = req.nas_root

    params = {
        "resample_ms": req.resample_ms,
        "max_lag_ms":  req.max_lag_ms,
        "window_ms":   req.window_ms,
    }
    # Résoudre model_dir relatif à root
    model_dir = req.model_dir
    if not Path(model_dir).is_absolute():
        model_dir = str(Path(req.root) / model_dir)

    job = _new_job("pipeline")
    threading.Thread(
        target = _worker_pipeline,
        args   = (job, req.session, req.root, model_dir, req.nas_root, params, req.force_flux),
        daemon = True,
    ).start()
    return {"job_id": job.id}


@app.post("/api/pipeline/run_batch")
async def pipeline_run_batch(req: dict):
    """Lance la pipeline sur plusieurs sessions en parallèle."""
    sessions   = req.get("sessions", [])
    root       = req.get("root", "")
    model_dir  = req.get("model_dir", "_sync_ml_model")
    nas_root   = req.get("nas_root", "")
    force_flux = req.get("force_flux", False)
    params = {
        "resample_ms": req.get("resample_ms", 5.0),
        "max_lag_ms":  req.get("max_lag_ms", 400.0),
        "window_ms":   req.get("window_ms", 2200.0),
    }
    if not Path(model_dir).is_absolute() and root:
        model_dir = str(Path(root) / model_dir)

    job_ids = []
    for sess in sessions:
        job = _new_job("pipeline")
        threading.Thread(
            target = _worker_pipeline,
            args   = (job, sess, root, model_dir, nas_root, params, force_flux),
            daemon = True,
        ).start()
        job_ids.append(job.id)

    return {"job_ids": job_ids, "count": len(job_ids)}


@app.get("/api/pipeline/state")
async def pipeline_state(session_path: str):
    """Retourne l'état pipeline persisté d'une session."""
    from pipeline import SessionPipelineState
    state = SessionPipelineState.load(Path(session_path))
    if state is None:
        raise HTTPException(404, "Aucun état pipeline trouvé pour cette session")
    return state.to_dict()


@app.post("/api/pipeline/rollback")
async def pipeline_rollback(req: PipelineStateRequest):
    """Force un rollback sur une session (restaure les .bak_syncml)."""
    from pipeline import _restore_bak_files, _restore_backup, SessionPipelineState
    sess = Path(req.session_path)
    if not sess.exists():
        raise HTTPException(404, "Session introuvable")

    # Essayer d'abord le backup complet
    ok = _restore_backup(sess, "ia_sync")
    if not ok:
        # Fallback sur les .bak_syncml
        _restore_bak_files(sess)

    # Mettre à jour l'état
    state = SessionPipelineState.load(sess)
    if state:
        from pipeline import StepStatus
        for step_name in ["ia_sync", "validate", "store"]:
            if step_name in state.steps:
                state.steps[step_name].status  = StepStatus.ROLLED_BACK
                state.steps[step_name].message = "Rollback manuel"
        state.finished = False
        state.success  = False
        state.save()

    return {"rolled_back": True, "session": str(sess.name)}


@app.post("/api/watcher/start")
async def watcher_start(req: WatcherStartRequest):
    """Démarre le watcher d'ingestion automatique."""
    global _model_dir, _nas_root
    _model_dir = req.model_dir
    _nas_root  = req.nas_root
    params = {
        "resample_ms": req.resample_ms,
        "max_lag_ms":  req.max_lag_ms,
        "window_ms":   req.window_ms,
    }
    model_dir = req.model_dir
    if not Path(model_dir).is_absolute():
        model_dir = str(Path(req.watch_dir) / model_dir)

    job = _new_job("watcher")
    threading.Thread(
        target = _worker_watcher_scan,
        args   = (job, req.watch_dir, model_dir, req.nas_root, params, req.auto_start),
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
    """Vide les jobs terminés."""
    with _jobs_lock:
        done = [k for k, v in _jobs.items() if v.status in (JobStatus.DONE, JobStatus.ERROR)]
        for k in done:
            del _jobs[k]
    return {"cleared": len(done)}


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
      - gripper_left   : {t_ms, opening_mm, angle_deg}
      - gripper_right  : {t_ms, opening_mm, angle_deg}
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
            for col in ("opening_mm", "angle_deg"):
                if col in df.columns:
                    grip[col] = df[col].tolist()
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

            # Signal cible : gripper opening_mm
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
                        tgt_series["t_ms"]    = gt
                        tgt_series["signal"]  = gdf["opening_mm"].tolist()
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
    p.add_argument("--root",      default=None, help="Répertoire racine par défaut")
    p.add_argument("--model-dir", default="_sync_ml_model")
    p.add_argument("--nas",       default="",   help="Répertoire NAS")
    p.add_argument("--watch",     default=None, help="Démarrer le watcher sur ce répertoire")
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      type=int, default=8000)
    p.add_argument("--reload",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.root:
        _root_dir  = args.root
    if args.nas:
        _nas_root  = args.nas
    if args.model_dir:
        _model_dir = args.model_dir
    uvicorn.run(
        "server:app",
        host   = args.host,
        port   = args.port,
        reload = args.reload,
    )
