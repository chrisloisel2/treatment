#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Inbox → Bronze

Scanne /mnt/storage/silver/, applique 5 vérifications, puis déplace les sessions valides
vers /mnt/storage/silver//{scenario}.

Étapes :
  1. STRUCTURE        — intégrité des fichiers/dossiers requis
  2. CSV              — validité des CSV + assurance que la pince n'est jamais bloquée à 0
  3. COMPLETUDE       — pas de tracker manquant, pas de JSONL manquant
  4. CONTINUITÉ       — timestamps capteurs triés, gaps détectés et classifiés (fixable/non)
  5. TRAKEUR          — vérification géométrique via trakeur.py (head/left/right)
  6. MOVE             — déplacement vers /mnt/storage/silver//{scenario}
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Chemins ───────────────────────────────────────────────────────────────────
INBOX_DIR  = Path("/mnt/storage/silver/")
BRONZE_DIR = Path("/mnt/storage/silver/")


# ── Concurrence ───────────────────────────────────────────────────────────────

@dataclass
class InboxConfig:
    """
    Paramètres de concurrence de la pipeline Inbox → Bronze.
    Modifiable à chaud sans redémarrage du serveur.
    """
    n_threads:   int = 4   # threads pour le scan/checks I/O-bound (par session)
    n_processes: int = 2   # processus pour le check trakeur (CPU-bound)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InboxConfig":
        return cls(
            n_threads   = max(1, int(d.get("n_threads",   4))),
            n_processes = max(1, int(d.get("n_processes", 2))),
        )


# Config globale mutable — modifiée à chaud via l'API
_INBOX_CONFIG = InboxConfig()

# ── Fichiers requis dans une session ──────────────────────────────────────────
REQUIRED_FILES = [
    "metadata.json",
    "tracker_positions.csv",
]
REQUIRED_VIDEOS_DIR = "videos"
VIDEO_SIDES         = ("head", "left", "right")
GRIPPER_SIDES       = ("left", "right")

# ── Seuils ────────────────────────────────────────────────────────────────────
GRIPPER_ZERO_RATIO_MAX = 0.95   # si >95 % des valeurs à 0 → pince bloquée
MIN_TRACKER_ROWS       = 10     # nb de lignes minimum dans tracker_positions.csv
MIN_JSONL_LINES        = 5      # nb minimum de lignes JSON dans chaque JSONL


# ══════════════════════════════════════════════════════════════════════════════
# Modèles
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    name:    str
    ok:      bool
    message: str
    details: List[str] = field(default_factory=list)


@dataclass
class SessionReport:
    session_name: str
    session_path: str
    checks:       List[CheckResult] = field(default_factory=list)
    promoted_to:  Optional[str]     = None
    error:        Optional[str]     = None

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def scenario(self) -> Optional[str]:
        meta_path = Path(self.session_path) / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return meta.get("scenario")
        except Exception:
            return None

    def to_dict(self) -> dict:
        return {
            "session_name": self.session_name,
            "session_path": self.session_path,
            "all_ok":       self.all_ok,
            "promoted_to":  self.promoted_to,
            "error":        self.error,
            "checks": [
                {
                    "name":    c.name,
                    "ok":      c.ok,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Checks individuels
# ══════════════════════════════════════════════════════════════════════════════

def check_structure(session_path: Path) -> CheckResult:
    """
    Vérifie que tous les fichiers et dossiers requis sont présents.
    """
    missing = []

    for f in REQUIRED_FILES:
        if not (session_path / f).exists():
            missing.append(f)

    videos_dir = session_path / REQUIRED_VIDEOS_DIR
    if not videos_dir.is_dir():
        missing.append(f"{REQUIRED_VIDEOS_DIR}/")
    else:
        for side in VIDEO_SIDES:
            mp4 = videos_dir / f"{side}.mp4"
            if not mp4.exists():
                missing.append(f"videos/{side}.mp4")

    if missing:
        return CheckResult(
            name="structure",
            ok=False,
            message=f"{len(missing)} fichier(s)/dossier(s) manquant(s)",
            details=[f"Manquant : {m}" for m in missing],
        )
    return CheckResult(name="structure", ok=True, message="Tous les fichiers requis présents")


def check_tracker_placement(session_path: Path) -> CheckResult:
    """
    Vérifie que les en-têtes du CSV permettent d'identifier head/left/right.
    """
    csv_path = session_path / "tracker_positions.csv"
    if not csv_path.exists():
        return CheckResult(name="tracker_placement", ok=False,
                           message="tracker_positions.csv introuvable")
    try:
        df = pd.read_csv(csv_path, nrows=0)
        cols = " ".join(str(c).lower() for c in df.columns)
        found = []
        for label in ("head", "left", "right"):
            if label in cols:
                found.append(label)

        missing_labels = [l for l in ("head", "left", "right") if l not in found]
        if missing_labels:
            return CheckResult(
                name="tracker_placement",
                ok=False,
                message=f"Labels manquants dans les en-têtes : {', '.join(missing_labels)}",
                details=[f"Colonnes trouvées : {list(df.columns)[:12]}"],
            )

        if len(df.columns) < 3 + 3 * 7:
            return CheckResult(
                name="tracker_placement",
                ok=False,
                message=f"Pas assez de colonnes ({len(df.columns)}) — 3 trackers × 7 colonnes attendues",
            )

        return CheckResult(
            name="tracker_placement",
            ok=True,
            message="Labels head/left/right trouvés dans le CSV",
        )
    except Exception as e:
        return CheckResult(name="tracker_placement", ok=False,
                           message=f"Erreur lecture CSV : {e}")


def check_csv_validity(session_path: Path) -> CheckResult:
    """
    Vérifie :
    - tracker_positions.csv lisible et suffisamment long
    - gripper_*_data.csv : aucune pince n'est bloquée à 0 sur >95 % des lignes
    """
    details = []
    ok = True

    # Tracker
    csv_path = session_path / "tracker_positions.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if len(df) < MIN_TRACKER_ROWS:
                ok = False
                details.append(f"tracker_positions.csv trop court : {len(df)} lignes < {MIN_TRACKER_ROWS}")
            else:
                details.append(f"tracker_positions.csv : {len(df)} lignes OK")
        except Exception as e:
            ok = False
            details.append(f"tracker_positions.csv illisible : {e}")
    else:
        ok = False
        details.append("tracker_positions.csv absent")

    # Grippers
    for side in GRIPPER_SIDES:
        grip_path = session_path / f"gripper_{side}_data.csv"
        if not grip_path.exists():
            # non bloquant — gripper optionnel
            details.append(f"gripper_{side}_data.csv absent (optionnel)")
            continue
        try:
            gdf = pd.read_csv(grip_path)
            # Chercher la colonne de valeur principale (position, value, angle…)
            value_cols = [c for c in gdf.columns
                          if any(kw in c.lower() for kw in
                                 ("position", "value", "angle", "open", "close", "grip", "force",
                                  "opening", "mm", "deg"))]
            # Garder uniquement les colonnes numériques
            value_cols = [c for c in value_cols
                          if pd.api.types.is_numeric_dtype(gdf[c])]
            if not value_cols:
                value_cols = [c for c in gdf.columns
                              if pd.api.types.is_numeric_dtype(gdf[c])]

            for col in value_cols[:3]:  # vérifier max 3 colonnes
                vals = gdf[col].dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                zero_ratio = float(np.sum(vals == 0.0) / len(vals))
                if zero_ratio > GRIPPER_ZERO_RATIO_MAX:
                    ok = False
                    details.append(
                        f"gripper_{side}[{col}] : {zero_ratio*100:.1f}% de zéros "
                        f"→ pince probablement bloquée"
                    )
                else:
                    details.append(
                        f"gripper_{side}[{col}] : {zero_ratio*100:.1f}% zéros — OK"
                    )
        except Exception as e:
            details.append(f"gripper_{side}_data.csv erreur lecture : {e}")

    if ok:
        return CheckResult(name="csv_validity", ok=True,
                           message="CSV valides", details=details)
    return CheckResult(name="csv_validity", ok=False,
                       message="Problème(s) détecté(s) dans les CSV", details=details)


def check_completude(session_path: Path) -> CheckResult:
    """
    Vérifie :
    - tracker_positions.csv présent et non vide
    - fichiers JSONL présents et non vides pour chaque caméra
    """
    details = []
    ok = True

    # Tracker
    csv_path = session_path / "tracker_positions.csv"
    if not csv_path.exists():
        ok = False
        details.append("tracker_positions.csv manquant")
    else:
        details.append("tracker_positions.csv présent")

    # JSONL
    videos_dir = session_path / "videos"
    if not videos_dir.is_dir():
        ok = False
        details.append("Dossier videos/ manquant")
    else:
        for side in VIDEO_SIDES:
            jsonl_path = videos_dir / f"{side}.jsonl"
            if not jsonl_path.exists():
                ok = False
                details.append(f"videos/{side}.jsonl manquant")
            else:
                # Compter lignes valides
                n_valid = 0
                try:
                    for line in jsonl_path.read_bytes().decode("utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line)
                            n_valid += 1
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    pass

                if n_valid < MIN_JSONL_LINES:
                    ok = False
                    details.append(
                        f"videos/{side}.jsonl trop court : {n_valid} lignes valides < {MIN_JSONL_LINES}"
                    )
                else:
                    details.append(f"videos/{side}.jsonl : {n_valid} lignes OK")

    if ok:
        return CheckResult(name="completude", ok=True,
                           message="Toutes les données présentes", details=details)
    return CheckResult(name="completude", ok=False,
                       message="Données incomplètes", details=details)


def _trakeur_worker(csv_path_str: str) -> dict:
    """
    Fonction top-level picklable — exécutée dans un sous-processus dédié.
    Retourne un dict sérialisable {ok, message, details}.
    """
    import sys as _sys
    from pathlib import Path as _Path
    import numpy as _np
    import pandas as _pd

    _root = _Path(__file__).resolve().parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))

    import script.trakeur as trakeur

    csv_path = _Path(csv_path_str)
    df       = _pd.read_csv(csv_path)
    blocks   = trakeur.split_blocks(df)
    truth    = trakeur.parse_truth_from_headers(df)

    pred_head, head_score = trakeur.detect_head(blocks)

    details = [
        f"Head prédit : tracker {pred_head}  (vérité : tracker {truth['head']})",
        f"Score head : {_np.round(head_score, 3).tolist()}",
    ]

    if pred_head != truth["head"]:
        return {"ok": False,
                "message": f"Head mal identifié : prédit={pred_head}, attendu={truth['head']}",
                "details": details}

    quat_mode, axis, sign = "wxyz", 0, 1
    try:
        pred_left, pred_right = trakeur.predict_hands_with_rule(
            blocks, pred_head, quat_mode, axis, sign
        )
        details.append(
            f"Mains prédites : left={pred_left} right={pred_right} "
            f"(vérité : left={truth['left']} right={truth['right']})"
        )
    except Exception as he:
        details.append(f"Prédiction mains ignorée : {he}")

    return {"ok": True, "message": "Trackers validés géométriquement", "details": details}


def check_sensor_continuity(session_path: Path) -> CheckResult:
    """
    Vérifie la continuité temporelle des CSV capteurs (gripper_*_data.csv).

    Pour chaque côté :
    - Tri des timestamps (bug systémique : premiers ~12 enregistrements arrivent
      en retard car le buffer de démarrage est vidé après le reste du flux)
    - Détection des gaps > 4 × dt_nominal
    - Classification : gap fixable (pince immobile, Δopening ≤ 2 mm) ou non

    Un gap fixable ne bloque pas la session — il sera interpolé lors du
    traitement aval (verify.py / session_pinces.py).
    Un gap non fixable (signal dynamique ou durée > 1 500 ms) génère un WARNING.
    """
    GAP_FACTOR         = 4.0    # seuils gaps en multiples de dt_nominal
    MAX_OPENING_CHANGE = 2.0    # mm — au-delà, gap considéré dynamique
    MAX_GAP_MS         = 1500.0 # ms — au-delà, gap non interpolable

    details = []
    ok = True

    for side in GRIPPER_SIDES:
        grip_path = session_path / f"gripper_{side}_data.csv"
        if not grip_path.exists():
            details.append(f"gripper_{side} : absent (optionnel)")
            continue

        try:
            gdf = pd.read_csv(grip_path)
            if "timestamp_ns" not in gdf.columns or "opening_mm" not in gdf.columns:
                details.append(f"gripper_{side} : colonnes timestamp_ns/opening_mm absentes — skip")
                continue

            gdf = (gdf[["timestamp_ns", "opening_mm"]]
                   .apply(pd.to_numeric, errors="coerce")
                   .dropna()
                   .sort_values("timestamp_ns")
                   .reset_index(drop=True))

            if len(gdf) < 3:
                details.append(f"gripper_{side} : moins de 3 échantillons — skip")
                continue

            ts   = gdf["timestamp_ns"].values.astype(np.int64)
            op   = gdf["opening_mm"].values.astype(float)
            dts  = np.diff(ts) / 1e6   # ms
            dt_nominal_ms = float(np.median(dts))

            if dt_nominal_ms <= 0:
                details.append(f"gripper_{side} : dt_nominal invalide ({dt_nominal_ms:.2f}ms) — skip")
                continue

            gap_thresh = GAP_FACTOR * dt_nominal_ms
            gap_mask   = dts > gap_thresh
            n_gaps     = int(gap_mask.sum())

            if n_gaps == 0:
                details.append(
                    f"gripper_{side} : {len(ts)} échant., dt_nom={dt_nominal_ms:.1f}ms, "
                    f"aucun gap — OK"
                )
                continue

            n_fixable   = 0
            n_unfixable = 0
            worst_gap   = 0.0

            for i in np.where(gap_mask)[0]:
                gap_ms    = float(dts[i])
                delta_op  = abs(float(op[i + 1]) - float(op[i]))
                worst_gap = max(worst_gap, gap_ms)

                if gap_ms <= MAX_GAP_MS and delta_op <= MAX_OPENING_CHANGE:
                    n_fixable += 1
                else:
                    n_unfixable += 1

            msg = (
                f"gripper_{side} : {n_gaps} gap(s) détecté(s) "
                f"[fixables={n_fixable}, non-fixables={n_unfixable}] "
                f"pire={worst_gap:.0f}ms, dt_nom={dt_nominal_ms:.1f}ms"
            )

            if n_unfixable > 0:
                ok = False
                details.append(f"[WARN] {msg}")
            else:
                details.append(f"[INFO] {msg} → interpolation possible")

        except Exception as e:
            details.append(f"gripper_{side} : erreur lecture ({e})")

    if ok:
        return CheckResult(
            name="sensor_continuity",
            ok=True,
            message="Continuité capteurs OK (gaps fixables ou absents)",
            details=details,
        )
    return CheckResult(
        name="sensor_continuity",
        ok=False,
        message="Gap(s) capteur non interpolable(s) détecté(s)",
        details=details,
    )


def check_trakeur(session_path: Path, n_processes: int = 1) -> CheckResult:
    """
    Appelle la logique de trakeur.py sur la session pour valider
    que head/left/right sont correctement identifiés géométriquement.

    n_processes > 1 : le calcul est délégué à un sous-processus dédié
    (utile quand plusieurs sessions sont traitées en parallèle pour éviter
    la contention NumPy sur le GIL).
    """
    csv_path = session_path / "tracker_positions.csv"
    if not csv_path.exists():
        return CheckResult(name="trakeur", ok=False,
                           message="tracker_positions.csv absent")

    try:
        if n_processes > 1:
            with ProcessPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_trakeur_worker, str(csv_path))
                res = fut.result(timeout=60)
        else:
            res = _trakeur_worker(str(csv_path))

        return CheckResult(
            name="trakeur",
            ok=res["ok"],
            message=res["message"],
            details=res["details"],
        )
    except Exception as e:
        return CheckResult(
            name="trakeur",
            ok=False,
            message=f"Erreur trakeur : {e}",
            details=[traceback.format_exc()],
        )


# ══════════════════════════════════════════════════════════════════════════════
# Déplacement
# ══════════════════════════════════════════════════════════════════════════════

def promote_to_bronze(session_path: Path, bronze_dir: Path = BRONZE_DIR) -> str:
    """
    Déplace la session vers bronze_dir/{scenario}/{session_name}.
    Lit le scénario depuis metadata.json.
    Retourne le chemin de destination.
    """
    meta_path = session_path / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        scenario = meta.get("scenario") or "unknown_scenario"
    except Exception:
        scenario = "unknown_scenario"

    # Normaliser le nom de scénario en nom de dossier
    scenario_slug = scenario.strip().replace(" ", "_").replace("/", "_")

    dest_dir = bronze_dir / scenario_slug
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / session_path.name
    if dest.exists():
        dest = dest_dir / f"{session_path.name}_{int(os.times().elapsed)}"

    shutil.move(str(session_path), str(dest))
    return str(dest)


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrateur principal
# ══════════════════════════════════════════════════════════════════════════════

def run_checks(session_path: Path,
               config: Optional[InboxConfig] = None) -> SessionReport:
    """
    Exécute les 5 checks de vérification (sans le move).
    Le check trakeur est délégué en sous-processus si config.n_processes > 1.
    """
    cfg = config or _INBOX_CONFIG
    report = SessionReport(
        session_name=session_path.name,
        session_path=str(session_path),
    )

    # Checks séquentiels (les 5 premiers sont I/O-bound et rapides)
    for label, fn in [
        ("1 · Structure",           check_structure),
        ("2 · Validité CSV",        check_csv_validity),
        ("3 · Complétude",          check_completude),
        ("4 · Continuité capteur",  check_sensor_continuity),
    ]:
        result = fn(session_path)
        result.name = label
        report.checks.append(result)

    # Check trakeur — peut utiliser un sous-processus dédié
    trak_result = check_trakeur(session_path, n_processes=cfg.n_processes)
    trak_result.name = "6 · Trakeur"
    report.checks.append(trak_result)

    return report


def scan_inbox(inbox_dir: Path = INBOX_DIR,
               config: Optional[InboxConfig] = None) -> List[SessionReport]:
    """
    Scanne inbox_dir et retourne un rapport par session détectée.
    Les sessions sont traitées en parallèle via ThreadPoolExecutor
    (n_threads = config.n_threads).
    """
    cfg = config or _INBOX_CONFIG

    if not inbox_dir.exists():
        return []

    entries = sorted(
        e for e in inbox_dir.iterdir()
        if e.is_dir() and not e.name.startswith(".") and not e.name.startswith("_")
    )

    if not entries:
        return []

    reports: List[SessionReport] = [None] * len(entries)  # type: ignore

    with ThreadPoolExecutor(max_workers=cfg.n_threads) as pool:
        futures = {
            pool.submit(run_checks, entry, cfg): idx
            for idx, entry in enumerate(entries)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                reports[idx] = fut.result()
            except Exception as e:
                reports[idx] = SessionReport(
                    session_name=entries[idx].name,
                    session_path=str(entries[idx]),
                    error=f"Erreur inattendue : {e}",
                )

    return reports


def promote_session(session_path: str,
                    bronze_dir: Path = BRONZE_DIR,
                    log_cb: Optional[Callable[[str, str], None]] = None,
                    config: Optional[InboxConfig] = None) -> SessionReport:
    """
    Vérifie puis promeut une session vers bronze.
    log_cb(message, level)
    """
    def _log(msg, level="INFO"):
        if log_cb:
            log_cb(msg, level)

    cfg  = config or _INBOX_CONFIG
    sess = Path(session_path)
    report = run_checks(sess, config=cfg)

    if not report.all_ok:
        failed = [c for c in report.checks if not c.ok]
        _log(f"Session {sess.name} : {len(failed)} check(s) échoué(s) — promotion annulée", "WARN")
        report.error = f"{len(failed)} check(s) échoués : " + ", ".join(c.name for c in failed)
        return report

    _log(f"Session {sess.name} : tous les checks OK — déplacement vers bronze…", "OK")
    try:
        dest = promote_to_bronze(sess, bronze_dir)
        report.promoted_to = dest
        _log(f"→ {dest}", "OK")
    except Exception as e:
        report.error = f"Erreur lors du déplacement : {e}"
        _log(report.error, "ERROR")

    return report
