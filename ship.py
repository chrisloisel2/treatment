#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ship.py — Pipeline final : scan récursif → validation complète → zip → Mistral

Découvre récursivement toutes les sessions dans un dossier racine, les valide
avec l'ensemble des vérifications du repo (structure, CSV, complétude, tracker
placement, portes structurelles check.py + score IA), zippe chaque session
valide et l'envoie vers le serveur Mistral.

Un historique persistant (JSON) empêche d'envoyer deux fois la même session.

Usage
-----
    python ship.py <racine> [options]

Options
-------
    --workers N          Threads de validation parallèle (défaut : 4)
    --min-score F        Score IA minimum pour l'envoi (0–100, défaut : 0)
    --skip-ia            Sauter le calcul du score IA (contrôles structurels seuls)
    --dry-run            Valider sans zipper ni envoyer
    --out-dir DIR        Dossier de sortie pour les zips (défaut : <racine>/_zips)
    --keep-zips          Conserver les zips après l'envoi
    --json               Émettre un rapport JSON sur stdout à la fin
    --history-file FILE  Chemin du fichier d'historique (défaut : <racine>/_ship_history.json)
    --force              Renvoyer même les sessions déjà présentes dans l'historique
    --show-history       Afficher l'historique des envois et quitter
    --accepted-dir DIR   Dossier où copier les sessions envoyées avec succès (grade A/B)
    --rejected-dir DIR   Dossier où copier les sessions rejetées (fichiers manquants ou grade C/D/F)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import requests

# ── sys.path : permet de lancer ship.py depuis n'importe quel répertoire ──────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Rich TUI ──────────────────────────────────────────────────────────────────
try:
    from rich import box as _box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("ERREUR: 'rich' n'est pas installé. Lancez : pip install rich")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Upload Mistral (inline de tomistral.py — autonomie standalone)
# ══════════════════════════════════════════════════════════════════════════════

_MISTRAL_BASE_URL = "http://13.62.206.125:5001"
_MISTRAL_USERNAME = os.getenv("MISTRAL_USERNAME", os.getenv("USERNAME", "pd_umi"))
_MISTRAL_PASSWORD = os.getenv("MISTRAL_PASSWORD", os.getenv("PASSWORD", "sqiu763hQP1"))


def upload_zip_to_mistral(zip_path: str) -> bool:
    """Envoie un fichier .zip vers le serveur Mistral via URL signée."""
    path = Path(zip_path)
    if not path.exists() or path.suffix.lower() != ".zip":
        logging.error("Pas un fichier .zip valide : %s", zip_path)
        return False

    session  = requests.Session()
    payload  = {
        "username": _MISTRAL_USERNAME,
        "password": _MISTRAL_PASSWORD,
        "repo_id":  path.stem,
        "filename": path.name,
    }

    logging.info("Demande d'URL signée pour '%s'…", path.name)
    try:
        r = session.post(
            url=f"{_MISTRAL_BASE_URL}/pd/upload",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
    except requests.RequestException as exc:
        logging.error("Erreur de connexion Mistral : %s", exc)
        return False

    if r.status_code != 200:
        try:
            err = r.json().get("error", "Unknown error")
        except Exception:
            err = r.text
        logging.error("Erreur serveur Mistral [%s] : %s", r.status_code, err)
        return False

    signed_url = r.json().get("url")
    if not signed_url:
        logging.error("Pas d'URL signée reçue du serveur Mistral.")
        return False

    logging.info("Upload de '%s' vers Mistral…", path.name)
    try:
        with open(path, "rb") as f:
            resp = session.put(
                signed_url,
                data=f,
                headers={"Content-Type": "application/zip"},
                timeout=120,
            )
    except requests.RequestException as exc:
        logging.error("Erreur upload : %s", exc)
        return False

    if resp.status_code in (200, 201, 204):
        logging.info("Upload réussi : %s", path.name)
        return True

    logging.error("Upload échoué [%s] : %s", resp.status_code, resp.text[:200])
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Modèles de données
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckEntry:
    name:    str
    ok:      bool
    message: str
    details: List[str] = field(default_factory=list)


@dataclass
class ShipResult:
    session_name: str
    session_path: str
    checks:       List[CheckEntry] = field(default_factory=list)

    ia_score:        float         = 0.0
    final_score:     float         = 0.0
    blocking_reason: Optional[str] = None

    zip_path:       Optional[str] = None
    zip_size_bytes: int           = 0
    uploaded:       bool          = False
    already_sent:   bool          = False
    skipped:        bool          = False
    error:          Optional[str] = None

    accepted_copy: Optional[str] = None
    rejected_copy: Optional[str] = None

    @property
    def valid(self) -> bool:
        return all(c.ok for c in self.checks) and not self.blocking_reason

    @property
    def grade(self) -> str:
        if not self.valid:
            return "Z"   # fichiers manquants ou données incomplètes
        s = self.final_score
        if s >= 90: return "A"
        if s >= 75: return "B"
        if s >= 60: return "C"
        if s >= 45: return "D"
        return "F"

    def to_dict(self) -> dict:
        return {
            "session":         self.session_name,
            "path":            self.session_path,
            "valid":           self.valid,
            "grade":           self.grade,
            "ia_score":        self.ia_score,
            "final_score":     self.final_score,
            "blocking_reason": self.blocking_reason,
            "uploaded":        self.uploaded,
            "already_sent":    self.already_sent,
            "skipped":         self.skipped,
            "zip_path":        self.zip_path,
            "accepted_copy":   self.accepted_copy,
            "rejected_copy":   self.rejected_copy,
            "error":           self.error,
            "checks": [
                {"name": c.name, "ok": c.ok, "message": c.message, "details": c.details}
                for c in self.checks
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Historique persistant
# ══════════════════════════════════════════════════════════════════════════════

class ShipHistory:
    """
    Registre JSON des sessions déjà envoyées à Mistral.
    Thread-safe : plusieurs workers peuvent appeler mark_sent() en parallèle.
    """

    def __init__(self, path: Path) -> None:
        self.path   = path
        self._lock  = threading.Lock()
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def already_sent(self, session_name: str) -> bool:
        return session_name in self._data

    def mark_sent(self, result: ShipResult) -> None:
        with self._lock:
            self._data[result.session_name] = {
                "session":     result.session_name,
                "path":        result.session_path,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "score":       result.final_score,
                "grade":       result.grade,
            }
            self._save()

    def entries(self) -> list[dict]:
        return sorted(self._data.values(), key=lambda e: e.get("uploaded_at", ""))

    def __len__(self) -> int:
        return len(self._data)


# ══════════════════════════════════════════════════════════════════════════════
# Découverte des sessions — streaming via os.walk
# ══════════════════════════════════════════════════════════════════════════════

def discover_sessions_stream(root: Path) -> Iterator[Path]:
    """
    Yield chaque session dès qu'elle est trouvée par os.walk.
    Ne descend pas dans les sous-dossiers d'une session reconnue
    (évite de traverser videos/, _zips/, etc. sur des dizaines de téras).
    """
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames.sort()  # ordre déterministe
        if "metadata.json" in filenames or "tracker_positions.csv" in filenames:
            yield Path(dirpath)
            dirnames.clear()  # ne pas descendre dans les sous-dossiers de la session


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Vérifications structurelles (inlinées — 0 dépendance externe)
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_FILES      = ["metadata.json", "tracker_positions.csv"]
_VIDEO_SIDES         = ("head", "left", "right")
_GRIPPER_SIDES       = ("left", "right")
_GRIPPER_ZERO_MAX    = 0.95   # >95 % de zéros → pince bloquée
_MIN_TRACKER_ROWS    = 10
_MIN_JSONL_LINES     = 5


def _chk_structure(p: Path) -> CheckEntry:
    missing = [f for f in _REQUIRED_FILES if not (p / f).exists()]
    vd = p / "videos"
    if not vd.is_dir():
        missing.append("videos/")
    else:
        for s in _VIDEO_SIDES:
            if not (vd / f"{s}.mp4").exists():
                missing.append(f"videos/{s}.mp4")
    if missing:
        return CheckEntry("structure", False,
                          f"{len(missing)} fichier(s) manquant(s)",
                          [f"Manquant : {m}" for m in missing])
    return CheckEntry("structure", True, "Tous les fichiers requis présents")


def _chk_tracker_placement(p: Path) -> CheckEntry:
    csv = p / "tracker_positions.csv"
    if not csv.exists():
        return CheckEntry("tracker_placement", False, "tracker_positions.csv introuvable")
    try:
        df   = pd.read_csv(csv, nrows=0)
        cols = " ".join(str(c).lower() for c in df.columns)
        missing_labels = [l for l in ("head", "left", "right") if l not in cols]
        if missing_labels:
            return CheckEntry("tracker_placement", False,
                              f"Labels manquants : {', '.join(missing_labels)}",
                              [f"Colonnes : {list(df.columns)[:12]}"])
        if len(df.columns) < 3 + 3 * 7:
            return CheckEntry("tracker_placement", False,
                              f"Trop peu de colonnes ({len(df.columns)})")
        return CheckEntry("tracker_placement", True, "Labels head/left/right présents")
    except Exception as exc:
        return CheckEntry("tracker_placement", False, f"Erreur lecture CSV : {exc}")


def _chk_csv_validity(p: Path) -> CheckEntry:
    details: List[str] = []
    ok = True

    csv = p / "tracker_positions.csv"
    if csv.exists():
        try:
            df = pd.read_csv(csv)
            if len(df) < _MIN_TRACKER_ROWS:
                ok = False
                details.append(f"tracker_positions.csv : {len(df)} lignes < {_MIN_TRACKER_ROWS}")
            else:
                details.append(f"tracker_positions.csv : {len(df)} lignes OK")
        except Exception as exc:
            ok = False
            details.append(f"tracker_positions.csv illisible : {exc}")
    else:
        ok = False
        details.append("tracker_positions.csv absent")

    for side in _GRIPPER_SIDES:
        gp = p / f"gripper_{side}_data.csv"
        if not gp.exists():
            details.append(f"gripper_{side}_data.csv absent (optionnel)")
            continue
        try:
            gdf = pd.read_csv(gp)
            vcols = [c for c in gdf.columns
                     if any(kw in c.lower() for kw in
                            ("position", "value", "angle", "open", "close",
                             "grip", "force", "opening", "mm", "deg"))
                     and pd.api.types.is_numeric_dtype(gdf[c])]
            if not vcols:
                vcols = [c for c in gdf.columns if pd.api.types.is_numeric_dtype(gdf[c])]
            for col in vcols[:3]:
                vals = gdf[col].dropna().to_numpy(dtype=float)
                if not len(vals):
                    continue
                zr = float(np.sum(vals == 0.0) / len(vals))
                if zr > _GRIPPER_ZERO_MAX:
                    ok = False
                    details.append(f"gripper_{side}[{col}] : {zr*100:.1f}% zéros → pince bloquée")
                else:
                    details.append(f"gripper_{side}[{col}] : {zr*100:.1f}% zéros — OK")
        except Exception as exc:
            details.append(f"gripper_{side} erreur : {exc}")

    return CheckEntry("csv_validity", ok,
                      "CSV valides" if ok else "Problème(s) CSV", details)


def _chk_completude(p: Path) -> CheckEntry:
    details: List[str] = []
    ok = True

    if not (p / "tracker_positions.csv").exists():
        ok = False
        details.append("tracker_positions.csv manquant")
    else:
        details.append("tracker_positions.csv présent")

    vd = p / "videos"
    if not vd.is_dir():
        ok = False
        details.append("Dossier videos/ manquant")
    else:
        for side in _VIDEO_SIDES:
            jl = vd / f"{side}.jsonl"
            if not jl.exists():
                ok = False
                details.append(f"videos/{side}.jsonl manquant")
                continue
            n = 0
            try:
                for line in jl.read_bytes().decode("utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                        n += 1
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass
            if n < _MIN_JSONL_LINES:
                ok = False
                details.append(f"videos/{side}.jsonl : {n} lignes < {_MIN_JSONL_LINES}")
            else:
                details.append(f"videos/{side}.jsonl : {n} lignes OK")

    return CheckEntry("completude", ok,
                      "Toutes les données présentes" if ok else "Données incomplètes", details)


def _run_inbox_checks(session_path: Path) -> List[CheckEntry]:
    entries: List[CheckEntry] = []
    for fn in (_chk_structure, _chk_csv_validity, _chk_completude, _chk_tracker_placement):
        try:
            entries.append(fn(session_path))
        except Exception as exc:
            entries.append(CheckEntry(fn.__name__, False, f"Exception : {exc}"))
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — check.py (portes structurelles + score IA)
# ══════════════════════════════════════════════════════════════════════════════

_IA_MODEL        = None
_IA_MODEL_LOADED = False


def _get_ia_model():
    global _IA_MODEL, _IA_MODEL_LOADED
    if not _IA_MODEL_LOADED:
        try:
            from check import load_model
            _IA_MODEL = load_model()
        except Exception:
            _IA_MODEL = None
        _IA_MODEL_LOADED = True
    return _IA_MODEL


def _run_check_session(session_path: Path, skip_ia: bool) -> tuple[float, float, Optional[str]]:
    """
    Retourne (ia_score, final_score, blocking_reason|None).
    Si check.py n'est pas disponible (mode standalone sans le repo complet),
    retourne un score neutre sans blocage.
    """
    try:
        from check import check_session  # _HERE est déjà dans sys.path
    except Exception:
        return 0.0, 50.0, None           # check.py absent → structurel seul

    model = None if skip_ia else _get_ia_model()

    try:
        report   = check_session(session_path, model=model)
        blocking = getattr(report, "blocking_reason", None) or None  # '' → None
        ia_score = float(getattr(report, "ia_score", 0.0))
        score    = float(getattr(report, "score",    0.0))
        return ia_score, score, blocking
    except Exception as exc:
        return 0.0, 0.0, f"check_session exception : {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# Validation complète d'une session
# ══════════════════════════════════════════════════════════════════════════════

def validate_session(session_path: Path, skip_ia: bool = False) -> ShipResult:
    result = ShipResult(
        session_name=session_path.name,
        session_path=str(session_path),
    )

    result.checks = _run_inbox_checks(session_path)

    if not all(c.ok for c in result.checks):
        return result

    ia_score, final_score, blocking = _run_check_session(session_path, skip_ia)
    result.ia_score        = ia_score
    result.final_score     = final_score
    result.blocking_reason = blocking

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Création du ZIP
# ══════════════════════════════════════════════════════════════════════════════

def zip_session(session_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{session_path.name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in sorted(session_path.rglob("*")):
            if file.is_file():
                arcname = file.relative_to(session_path.parent)
                zf.write(file, arcname)

    return zip_path


# ══════════════════════════════════════════════════════════════════════════════
# Tri physique (copie vers accepted / rejected)
# ══════════════════════════════════════════════════════════════════════════════

def _is_rejected(result: ShipResult) -> bool:
    """Rejetée si structure invalide (Z) OU grade en dessous de B (C/D/F)."""
    return result.grade not in ("A", "B")


def _copy_to(src: Path, dest_dir: Path) -> Optional[str]:
    dest = dest_dir / src.name
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return str(dest)
    except Exception as exc:
        return f"ERREUR copie : {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal pour une session
# ══════════════════════════════════════════════════════════════════════════════

def process_session(
    session_path: Path,
    out_dir: Path,
    skip_ia: bool,
    min_score: float,
    dry_run: bool,
    keep_zips: bool,
    history: Optional[ShipHistory] = None,
    force: bool = False,
    accepted_dir: Optional[Path] = None,
    rejected_dir: Optional[Path] = None,
) -> ShipResult:

    # Court-circuit historique
    if history and not force and history.already_sent(session_path.name):
        return ShipResult(
            session_name=session_path.name,
            session_path=str(session_path),
            already_sent=True,
            skipped=True,
        )

    result = validate_session(session_path, skip_ia=skip_ia)

    # Copie vers rejected si qualité insuffisante
    if not dry_run and rejected_dir and _is_rejected(result):
        result.rejected_copy = _copy_to(session_path, rejected_dir)

    if not result.valid:
        return result

    if result.final_score < min_score:
        result.skipped = True
        result.error = f"Score {result.final_score:.1f} < seuil {min_score:.1f}"
        return result

    if dry_run:
        result.skipped = True
        return result

    # Zip
    try:
        zip_path = zip_session(session_path, out_dir)
        result.zip_path       = str(zip_path)
        result.zip_size_bytes = zip_path.stat().st_size
    except Exception as exc:
        result.error = f"Erreur zip : {exc}"
        return result

    # Envoi Mistral
    try:
        success = upload_zip_to_mistral(str(zip_path))
        result.uploaded = success
        if success:
            if history:
                history.mark_sent(result)
            if accepted_dir:
                result.accepted_copy = _copy_to(session_path, accepted_dir)
        else:
            result.error = "Upload Mistral échoué (voir logs ci-dessus)"
    except Exception as exc:
        result.error = f"Erreur upload : {exc}"

    # Nettoyage zip
    if not keep_zips and result.zip_path:
        try:
            Path(result.zip_path).unlink()
            result.zip_path = None
        except Exception:
            pass

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Interface graphique terminal — Rich
# ══════════════════════════════════════════════════════════════════════════════

def _session_duration_hours(session_path: Path) -> float:
    """Lit duration_seconds dans metadata.json et retourne des heures."""
    try:
        meta = json.loads((session_path / "metadata.json").read_text(encoding="utf-8"))
        return float(meta.get("duration_seconds", 0.0)) / 3600.0
    except Exception:
        return 0.0


class ShipUI:
    """TUI rich — live table, barre de progression, codes couleur par grade."""

    _GRADE: dict[str, tuple[str, str]] = {
        "A": ("bright_green",   "bold"),
        "B": ("green",          "bold"),
        "C": ("yellow",         "bold"),
        "D": ("dark_orange",    "bold"),
        "F": ("red",            "bold"),
        "Z": ("bright_magenta", "bold"),   # fichiers / data manquants
        "?": ("dim",            ""),
    }

    _STATUS: dict[str, tuple[str, str]] = {
        "EN ATTENTE":  ("dim",          "·"),
        "EN COURS":    ("cyan",         "⟳"),
        "ENVOYÉ":      ("bright_green", "✓"),
        "DÉJÀ ENVOYÉ": ("yellow",       "◎"),
        "DRY-RUN":     ("cyan",         "○"),
        "INVALIDE":    ("red",          "✗"),
        "IGNORÉ":      ("dark_orange",  "⊘"),
        "ÉCHEC":       ("red",          "!"),
    }

    def __init__(
        self,
        history: ShipHistory,
        config: dict,
    ) -> None:
        self._lock         = threading.Lock()
        self._history      = history
        self._config       = config
        self._done         = 0
        self._scan_done    = False
        self._rows: dict[str, dict] = {}
        # ── Stats cumulées ────────────────────────────────────────────────
        self._sent_gb:     float          = 0.0
        self._sent_hours:  float          = 0.0
        self._grade_counts: dict[str,int] = {}
        self.console       = Console()

    # ── Mises à jour thread-safe ───────────────────────────────────────────

    def add_session(self, path: Path) -> None:
        """Appelé dès qu'une session est découverte sur le disque."""
        with self._lock:
            if path.name not in self._rows:
                self._rows[path.name] = {"status": "EN ATTENTE", "result": None}

    def mark_scan_done(self) -> None:
        with self._lock:
            self._scan_done = True

    def on_result(self, result: ShipResult) -> None:
        with self._lock:
            self._rows[result.session_name] = {
                "status": self._status_of(result),
                "result": result,
            }
            self._done += 1
            # Stats cumulées
            g = result.grade
            self._grade_counts[g] = self._grade_counts.get(g, 0) + 1
            if result.uploaded:
                self._sent_gb    += result.zip_size_bytes / (1024 ** 3)
                self._sent_hours += _session_duration_hours(Path(result.session_path))

    def __rich__(self):
        """Permet à Live(ui) de ré-appeler render() à chaque rafraîchissement."""
        return self.render()

    @staticmethod
    def _status_of(r: ShipResult) -> str:
        if r.already_sent and not r.uploaded: return "DÉJÀ ENVOYÉ"
        if r.uploaded:                        return "ENVOYÉ"
        if r.skipped and not r.error:         return "DRY-RUN"
        if r.skipped:                         return "IGNORÉ"
        if not r.valid:                       return "INVALIDE"
        return "ÉCHEC"

    # ── Blocs de rendu ─────────────────────────────────────────────────────

    def _header(self) -> Panel:
        cfg   = self._config
        lines = [
            f"[dim]Racine[/]     [white]{cfg['root']}[/]",
            f"[dim]Historique[/] [white]{cfg['history_path']}[/]"
            f"  [yellow]{len(self._history)} entrée(s)[/]",
        ]
        if cfg.get("accepted_dir"):
            lines.append(f"[dim]Acceptées[/]  [bright_green]{cfg['accepted_dir']}[/]")
        if cfg.get("rejected_dir"):
            lines.append(f"[dim]Rejetées[/]   [red]{cfg['rejected_dir']}[/]")

        tags: list[str] = [f"[cyan]workers={cfg['workers']}[/]"]
        if cfg.get("dry_run"):   tags.append("[yellow]DRY-RUN[/]")
        if cfg.get("force"):     tags.append("[yellow]--force[/]")
        if cfg.get("skip_ia"):   tags.append("[yellow]structurel seul[/]")
        if cfg.get("min_score"): tags.append(f"[cyan]score ≥ {cfg['min_score']:.0f}%[/]")
        lines.append("  ".join(tags))

        return Panel(
            "\n".join(lines),
            title="[bold bright_cyan]🚀  SHIP · Pipeline Mistral[/]",
            border_style="bright_cyan",
            padding=(0, 2),
        )

    # Lignes réservées hors tableau : header (~7) + stats (~3) + footer (~4) + en-têtes (2) + marges (2)
    _OVERHEAD = 18

    def _visible_rows(self) -> int:
        """Nombre de lignes disponibles pour les données dans le terminal."""
        try:
            h = self.console.height or 40
        except Exception:
            h = 40
        return max(5, h - self._OVERHEAD)

    def _build_row(self, name: str, entry: dict) -> tuple:
        r:      Optional[ShipResult] = entry["result"]
        status: str                  = entry["status"]

        grade = r.grade if r else "?"
        score = f"{r.final_score:.1f}%" if (r and r.final_score) else "—"

        gc, gb     = self._GRADE.get(grade, ("dim", ""))
        grade_cell = Text(f"  {grade}  ", style=f"{gc} {gb}".strip())

        sc, si      = self._STATUS.get(status, ("dim", "·"))
        status_cell = Text(f"{si}  {status}", style=sc)

        detail = ""
        if r:
            if r.blocking_reason:
                detail = r.blocking_reason[:48]
            elif r.error:
                detail = r.error[:48]
            else:
                failed = [c.name for c in r.checks if not c.ok]
                if failed:
                    detail = "✗ " + ", ".join(failed)
                elif r.accepted_copy:
                    detail = f"↳ {Path(r.accepted_copy).name}"
                elif r.rejected_copy:
                    detail = f"↳ {Path(r.rejected_copy).name}"

        return name, grade_cell, score, status_cell, detail

    def _table(self) -> Table:
        t = Table(
            box=_box.SIMPLE_HEAVY,
            header_style="bold dim",
            show_lines=False,
            expand=True,
            padding=(0, 1),
        )
        t.add_column("Session",  style="white",  min_width=38, no_wrap=True)
        t.add_column("Grade",    justify="center", min_width=7)
        t.add_column("Score",    justify="right",  min_width=7)
        t.add_column("Statut",   min_width=16)
        t.add_column("Détail",   style="dim",      min_width=30, no_wrap=True)

        with self._lock:
            all_rows = list(self._rows.items())

        visible = self._visible_rows()
        hidden  = max(0, len(all_rows) - visible)

        if hidden:
            # Résumé des sessions masquées au-dessus de la fenêtre
            above = all_rows[:hidden]
            counts: dict[str, int] = {}
            for _, e in above:
                r = e["result"]
                g = r.grade if r else "?"
                counts[g] = counts.get(g, 0) + 1

            parts: list[str] = []
            for g in ("A", "B", "C", "D", "F", "Z"):
                n = counts.get(g, 0)
                if n:
                    gc, _ = self._GRADE.get(g, ("dim", ""))
                    parts.append(f"[{gc}]{g}×{n}[/]")

            summary = "  ".join(parts) if parts else ""
            indicator = Text.assemble(
                ("  ↑ ", "dim"),
                (f"{hidden} session(s) au-dessus", "bold dim"),
                ("   ", ""),
                Text.from_markup(summary),
            )
            t.add_row(indicator, "", "", "", "")

        for name, entry in all_rows[hidden:]:
            t.add_row(*self._build_row(name, entry))

        return t

    def _footer(self) -> Panel:
        with self._lock:
            done       = self._done
            total      = len(self._rows)
            scan_done  = self._scan_done
            rows       = list(self._rows.values())

        pct  = done / total if total else 0.0
        W    = 44
        fill = int(W * pct)

        # Barre pleine si scan pas terminé (total inconnu) → animation pointillée
        if not scan_done:
            # position glissante basée sur le nb de sessions découvertes
            offset = total % W
            bar = (
                f"[dim]{'░' * offset}[/]"
                f"[bright_cyan]{'█' * min(8, W - offset)}[/]"
                f"[dim]{'░' * max(0, W - offset - 8)}[/]"
            )
            total_str = f"[bold]{total}[/][dim]+[/]"
        else:
            bar = f"[bright_cyan]{'█' * fill}[/][dim]{'░' * (W - fill)}[/]"
            total_str = f"[bold]{total}[/]"

        n_sent    = sum(1 for e in rows if e["status"] == "ENVOYÉ")
        n_invalid = sum(1 for e in rows if e["status"] in ("INVALIDE", "ÉCHEC"))
        n_already = sum(1 for e in rows if e["status"] == "DÉJÀ ENVOYÉ")
        n_dry     = sum(1 for e in rows if e["status"] == "DRY-RUN")
        n_ign     = sum(1 for e in rows if e["status"] == "IGNORÉ")

        scan_tag = (
            "[dim]scan terminé[/]" if scan_done
            else "[cyan]⟳ scan en cours…[/]"
        )

        parts: list[str] = [scan_tag]
        if n_sent:    parts.append(f"[bright_green]✓ {n_sent} envoyée(s)[/]")
        if n_already: parts.append(f"[yellow]◎ {n_already} déjà envoyée(s)[/]")
        if n_dry:     parts.append(f"[cyan]○ {n_dry} dry-run[/]")
        if n_ign:     parts.append(f"[dark_orange]⊘ {n_ign} ignorée(s)[/]")
        if n_invalid: parts.append(f"[red]✗ {n_invalid} rejetée(s)[/]")

        return Panel(
            f"{bar}  [bold]{done}[/] / {total_str}  [dim]{pct * 100:.0f} %[/]\n"
            + "  ".join(parts),
            border_style="dim",
            padding=(0, 2),
        )

    def _stats_panel(self) -> Panel:
        with self._lock:
            discovered  = len(self._rows)
            sent_gb     = self._sent_gb
            sent_hours  = self._sent_hours
            gc          = dict(self._grade_counts)

        # ── Taille envoyée ────────────────────────────────────────────────
        if sent_gb >= 1.0:
            size_str = f"[bold white]{sent_gb:.2f}[/] [dim]Go[/]"
        else:
            size_str = f"[bold white]{sent_gb * 1024:.1f}[/] [dim]Mo[/]"

        # ── Durée envoyée ─────────────────────────────────────────────────
        total_min = int(sent_hours * 60)
        if total_min >= 60:
            h, m  = divmod(total_min, 60)
            dur_str = f"[bold white]{h}[/][dim]h[/][bold white]{m:02d}[/][dim]min[/]"
        else:
            dur_str = f"[bold white]{total_min}[/][dim]min[/]"

        # ── Répartition des grades ────────────────────────────────────────
        grade_parts: list[str] = []
        for g in ("A", "B", "C", "D", "F", "Z"):
            n = gc.get(g, 0)
            if n:
                col, _ = self._GRADE.get(g, ("dim", ""))
                grade_parts.append(f"[{col}]{g}[/][bold]{n}[/]")
        grades_str = "  ".join(grade_parts) if grade_parts else "[dim]—[/]"

        content = (
            f"[dim]Envoyé[/]  📦 {size_str}   ⏱ {dur_str}"
            f"   [dim]│[/]   [dim]Découvertes[/] [bold white]{discovered}[/]"
            f"   [dim]│[/]   {grades_str}"
        )
        return Panel(content, border_style="dim", padding=(0, 2))

    def render(self) -> Group:
        return Group(
            self._header(),
            self._stats_panel(),
            self._table(),
            self._footer(),
        )

    # ── Vues statiques ─────────────────────────────────────────────────────

    def show_history(self) -> None:
        entries = self._history.entries()
        if not entries:
            self.console.print("\n[yellow]● Historique vide — aucune session envoyée.[/]\n")
            return

        t = Table(
            title=f"[bold bright_cyan]Historique des envois  ({len(entries)} session(s))[/]",
            box=_box.ROUNDED,
            header_style="bold dim",
            show_lines=True,
            expand=False,
        )
        t.add_column("Date envoi",  style="dim",       min_width=19, no_wrap=True)
        t.add_column("Grade",       justify="center",  min_width=7)
        t.add_column("Score",       justify="right",   min_width=7)
        t.add_column("Session",     min_width=38)

        for e in entries:
            grade = e.get("grade", "?")
            gc, gb = self._GRADE.get(grade, ("dim", ""))
            ts     = e.get("uploaded_at", "")[:19].replace("T", " ")
            score  = float(e.get("score", 0.0))
            t.add_row(
                ts,
                Text(f"  {grade}  ", style=f"{gc} {gb}".strip()),
                f"{score:.1f}%",
                e.get("session", ""),
            )

        self.console.print()
        self.console.print(t)
        self.console.print()

    def show_summary(self, results: List[ShipResult]) -> None:
        n_total   = len(results)
        n_already = sum(1 for r in results if r.already_sent and not r.uploaded)
        n_valid   = sum(1 for r in results if r.valid)
        n_invalid = n_total - n_valid - n_already
        n_sent    = sum(1 for r in results if r.uploaded)
        n_skipped = sum(1 for r in results if r.skipped and not r.already_sent)
        n_failed  = sum(1 for r in results if r.valid and not r.uploaded and not r.skipped)

        # Tableau de bilan
        t = Table(box=_box.MINIMAL, show_header=False, expand=False, padding=(0, 2))
        t.add_column("", style="dim",  min_width=26)
        t.add_column("", style="bold", min_width=10)

        t.add_row("Sessions découvertes", str(n_total))
        if n_already:
            t.add_row("Déjà envoyées",    Text(str(n_already), style="yellow"))
        t.add_row("Valides",              Text(str(n_valid),   style="bright_green"))
        t.add_row("Invalides",            Text(str(n_invalid), style="red" if n_invalid else "dim"))

        if self._config.get("dry_run"):
            t.add_row("Mode", Text("DRY-RUN — aucun envoi effectué", style="yellow"))
        else:
            t.add_row("Envoyées à Mistral", Text(str(n_sent),   style="bright_green"))
            if n_skipped:
                t.add_row("Ignorées (score)", str(n_skipped))
            if n_failed:
                t.add_row("Échecs d'envoi",  Text(str(n_failed), style="red"))

        # Distribution des grades
        grade_counts: dict[str, int] = {}
        for r in results:
            grade_counts[r.grade] = grade_counts.get(r.grade, 0) + 1

        grade_cell = Text()
        for g in ("A", "B", "C", "D", "F", "Z"):
            n = grade_counts.get(g, 0)
            if n:
                gc, gb = self._GRADE.get(g, ("dim", ""))
                if grade_cell._length:
                    grade_cell.append("   ")
                grade_cell.append(f"{g} × {n}", style=f"{gc} {gb}".strip())
        if grade_cell._length:
            t.add_row("Grades", grade_cell)

        self.console.print()
        self.console.print(
            Panel(t, title="[bold]Bilan final[/]", border_style="bright_cyan", padding=(0, 1))
        )
        self.console.print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ship.py",
        description="Scan récursif → validation → zip → Mistral",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("root",            type=Path, help="Dossier racine à scanner récursivement")
    p.add_argument("--workers",       type=int,   default=4,   metavar="N")
    p.add_argument("--min-score",     type=float, default=0.0, metavar="F")
    p.add_argument("--skip-ia",       action="store_true")
    p.add_argument("--dry-run",       action="store_true")
    p.add_argument("--out-dir",       type=Path,  default=None, metavar="DIR")
    p.add_argument("--keep-zips",     action="store_true")
    p.add_argument("--json",          action="store_true")
    p.add_argument("--history-file",  type=Path,  default=None, metavar="FILE")
    p.add_argument("--force",         action="store_true")
    p.add_argument("--show-history",  action="store_true")
    p.add_argument("--accepted-dir",  type=Path,  default=None, metavar="DIR")
    p.add_argument("--rejected-dir",  type=Path,  default=None, metavar="DIR")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        Console().print(f"[red]✗  Dossier introuvable : {root}[/]", highlight=False)
        sys.exit(1)

    out_dir      = args.out_dir.resolve()      if args.out_dir      else root / "_zips"
    accepted_dir = args.accepted_dir.resolve() if args.accepted_dir else None
    rejected_dir = args.rejected_dir.resolve() if args.rejected_dir else None
    history_path = args.history_file.resolve() if args.history_file else root / "_ship_history.json"
    history      = ShipHistory(history_path)

    config = {
        "root":         str(root),
        "history_path": str(history_path),
        "workers":      args.workers,
        "dry_run":      args.dry_run,
        "force":        args.force,
        "skip_ia":      args.skip_ia,
        "min_score":    args.min_score if args.min_score > 0 else None,
        "accepted_dir": str(accepted_dir) if accepted_dir else None,
        "rejected_dir": str(rejected_dir) if rejected_dir else None,
    }

    ui = ShipUI(history, config)

    # ── show-history ──────────────────────────────────────────────────────
    if args.show_history:
        ui.show_history()
        sys.exit(0)

    # ── Pipeline streaming : scan + traitement en parallèle ───────────────
    #
    # Architecture :
    #   [Thread scan]  os.walk → yield session → pool.submit(job)
    #                                          → result_q.put(result)
    #   [Main thread]  result_q.get() → ui.on_result() → Live auto-refresh
    #
    result_q    : queue.Queue[ShipResult] = queue.Queue()
    n_submitted = 0
    n_sub_lock  = threading.Lock()
    scan_event  = threading.Event()   # levé quand le scan est terminé

    def _scan_and_submit(pool: ThreadPoolExecutor) -> None:
        nonlocal n_submitted
        try:
            for sess in discover_sessions_stream(root):
                with n_sub_lock:
                    n_submitted += 1
                ui.add_session(sess)

                def _job(s: Path = sess) -> None:
                    try:
                        r = process_session(
                            s, out_dir, args.skip_ia, args.min_score,
                            args.dry_run, args.keep_zips, history, args.force,
                            accepted_dir, rejected_dir,
                        )
                    except Exception as exc:
                        r = ShipResult(
                            session_name=s.name,
                            session_path=str(s),
                            error=f"Exception inattendue : {exc}",
                        )
                    result_q.put(r)

                pool.submit(_job)
        finally:
            scan_event.set()
            ui.mark_scan_done()

    results: List[ShipResult] = []
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        scan_thread = threading.Thread(
            target=_scan_and_submit, args=(pool,), daemon=True, name="scan"
        )
        scan_thread.start()

        with Live(ui, refresh_per_second=8, console=ui.console):
            while True:
                try:
                    r = result_q.get(timeout=0.2)
                except queue.Empty:
                    # Vérifier si tout est terminé
                    with n_sub_lock:
                        submitted = n_submitted
                    if scan_event.is_set() and n_done >= submitted:
                        break
                    continue

                results.append(r)
                n_done += 1
                ui.on_result(r)

                # Vérifier la condition de fin après chaque résultat
                with n_sub_lock:
                    submitted = n_submitted
                if scan_event.is_set() and n_done >= submitted:
                    break

        scan_thread.join(timeout=2)

    # ── Bilan final ───────────────────────────────────────────────────────
    results.sort(key=lambda r: r.session_name)
    ui.show_summary(results)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))

    sys.exit(1 if any(r.valid and not r.uploaded and not r.skipped for r in results) else 0)


if __name__ == "__main__":
    main()
