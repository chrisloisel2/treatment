#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/ — Module de gestion complète de la réparation des sessions robot.

Architecture :
  problems.py              : catalogue des problèmes (ProblemCode, DiagnosedProblem)
  diagnosis.py             : SessionReport → list[DiagnosedProblem]
  sftp_utils.py            : accès SFTP au serveur HDD (bronze/silver/gold)
  fix_camera_offset.py     : correction offset caméra/tracker (cross-corrélation)
  fix_clock_drift.py       : correction dérive linéaire d'horloge
  fix_camera_gaps.py       : correction gaps dans le flux caméra
  fix_tracker_gaps.py      : correction gaps dans le flux tracker (SLERP)
  fix_quaternions.py       : réparation quaternions NaN/inf (SLERP)
  fix_tracker_placement.py : re-identification head/left/right des trackers (legacy)
  fix_tracker_labels.py    : re-identification CERTAINE head/left/right (4 tests)
  fix_camera_labels.py     : vérification/correction labels caméra (serial + calibration)
  fix_timestamp_sync.py    : analyse synchronisation temporelle (gaps, drift, décalages)
  fix_gripper_video_sync.py: vérification gripper capteur ↔ flux vidéo (IA)
  fix_sync_lag.py          : correction lag résiduel de synchronisation
  repair.py                : orchestrateur principal (check → diagnose → fix → recheck)

Usage rapide :
    # CLI
    python -m fix.repair /path/to/session --verbose
    python -m fix.repair /path/to/root --batch --json-out results.json

    # Python
    from pathlib import Path
    from fix.repair import process_session
    import check as chk

    model = chk.load_model()
    result = process_session(Path("/path/to/session"), model, verbose=True)
    print(result)

    # Diagnostic seul
    import check as chk
    from fix.diagnosis import diagnose_session, format_diagnosis

    report = chk.check_session(Path("/path/to/session"), model)
    problems = diagnose_session(Path("/path/to/session"), report)
    print(format_diagnosis(problems))
"""

from fix.problems import (
    ProblemCode,
    DiagnosedProblem,
    make_problem,
    get_spec,
)
from fix.diagnosis import diagnose_session, format_diagnosis
from fix.repair import process_session
from fix.fix_tracker_labels import fix_tracker_labels
from fix.fix_camera_labels import fix_camera_labels
from fix.fix_timestamp_sync import analyse_timestamp_sync
from fix.fix_gripper_video_sync import analyse_gripper_video_sync

__all__ = [
    "ProblemCode",
    "DiagnosedProblem",
    "make_problem",
    "get_spec",
    "diagnose_session",
    "format_diagnosis",
    "process_session",
    "fix_tracker_labels",
    "fix_camera_labels",
    "analyse_timestamp_sync",
    "analyse_gripper_video_sync",
]
