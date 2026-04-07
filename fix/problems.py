#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/problems.py — Catalogue exhaustif des problèmes détectables dans une session.

Chaque problème est décrit par :
  - code        : identifiant unique (snake_case)
  - description : description humaine courte
  - recoverable : True si un fix automatique est possible
  - fix_module  : nom du module de fix à appeler (None si non récupérable)
  - priority    : ordre d'application des fixes (plus petit = appliqué en premier)

L'ordre des fixes est crucial :
  1. Fichiers/metadata       → fatal si absent
  2. Quaternions corrompus   → les autres analyses dépendent de données saines
  3. Placement tracker       → la sync check dépend de l'identification correcte
  4. Offset caméra (grossier)→ aligne les timestamps avant la recherche fine
  5. Drift d'horloge         → correction linéaire après alignement grossier
  6. Gaps caméra             → trim après alignement
  7. Gaps tracker            → interpolation
  8. Lag fin de sync         → affinage sub-frame après tous les autres fixes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProblemCode(str, Enum):
    # ── Problèmes fatals (non récupérables automatiquement) ───────────────────
    MISSING_FILES        = "missing_files"
    METADATA_CORRUPT     = "metadata_corrupt"

    # ── Corruption de données ──────────────────────────────────────────────────
    QUATERNION_CORRUPT   = "quaternion_corrupt"

    # ── Placement physique ────────────────────────────────────────────────────
    TRACKER_MISPLACED    = "tracker_misplaced"
    CAMERA_MISPLACED     = "camera_misplaced"

    # ── Synchronisation temporelle ────────────────────────────────────────────
    CAMERA_OFFSET        = "camera_offset"       # décalage grossier caméra/tracker
    CLOCK_DRIFT          = "clock_drift"          # dérive linéaire progressive
    SYNC_LAG             = "sync_lag"             # lag résiduel sub-seconde

    # ── Continuité du flux ────────────────────────────────────────────────────
    CAMERA_GAPS          = "camera_gaps"
    TRACKER_GAPS         = "tracker_gaps"

    # ── Score IA faible (après tous les autres fixes) ─────────────────────────
    LOW_IA_SCORE         = "low_ia_score"


@dataclass
class ProblemSpec:
    code:        ProblemCode
    description: str
    recoverable: bool
    fix_module:  Optional[str]  # nom du module dans fix/
    priority:    int            # ordre d'application (croissant)


# Registre complet : code → spec
PROBLEM_REGISTRY: dict[ProblemCode, ProblemSpec] = {
    ProblemCode.MISSING_FILES: ProblemSpec(
        code=ProblemCode.MISSING_FILES,
        description="Fichiers requis manquants (jsonl, csv, metadata)",
        recoverable=False,
        fix_module=None,
        priority=0,
    ),
    ProblemCode.METADATA_CORRUPT: ProblemSpec(
        code=ProblemCode.METADATA_CORRUPT,
        description="metadata.json illisible ou corrompu",
        recoverable=False,
        fix_module=None,
        priority=1,
    ),
    ProblemCode.QUATERNION_CORRUPT: ProblemSpec(
        code=ProblemCode.QUATERNION_CORRUPT,
        description="Quaternions NaN/inf dans tracker_positions.csv",
        recoverable=True,
        fix_module="fix_quaternions",
        priority=10,
    ),
    ProblemCode.TRACKER_MISPLACED: ProblemSpec(
        code=ProblemCode.TRACKER_MISPLACED,
        description="Trackers head/left/right mal assignés (mauvaises colonnes CSV)",
        recoverable=True,
        fix_module="fix_tracker_placement",
        priority=20,
    ),
    ProblemCode.CAMERA_MISPLACED: ProblemSpec(
        code=ProblemCode.CAMERA_MISPLACED,
        description="Caméras head/left/right mal assignées (mauvais JSONL)",
        recoverable=True,
        fix_module="fix_camera_placement",
        priority=21,
    ),
    ProblemCode.CAMERA_OFFSET: ProblemSpec(
        code=ProblemCode.CAMERA_OFFSET,
        description="Décalage grossier caméra/tracker (caméra démarre avant/après le tracker)",
        recoverable=True,
        fix_module="fix_camera_offset",
        priority=30,
    ),
    ProblemCode.CLOCK_DRIFT: ProblemSpec(
        code=ProblemCode.CLOCK_DRIFT,
        description="Dérive linéaire progressive des timestamps caméra par rapport au tracker",
        recoverable=True,
        fix_module="fix_clock_drift",
        priority=35,
    ),
    ProblemCode.CAMERA_GAPS: ProblemSpec(
        code=ProblemCode.CAMERA_GAPS,
        description="Trop de gaps dans le flux caméra (frames manquantes)",
        recoverable=True,
        fix_module="fix_camera_gaps",
        priority=40,
    ),
    ProblemCode.TRACKER_GAPS: ProblemSpec(
        code=ProblemCode.TRACKER_GAPS,
        description="Trop de gaps dans le flux tracker",
        recoverable=True,
        fix_module="fix_tracker_gaps",
        priority=41,
    ),
    ProblemCode.SYNC_LAG: ProblemSpec(
        code=ProblemCode.SYNC_LAG,
        description="Lag résiduel sub-seconde entre caméra et tracker après alignement grossier",
        recoverable=True,
        fix_module="fix_sync_lag",
        priority=50,
    ),
    ProblemCode.LOW_IA_SCORE: ProblemSpec(
        code=ProblemCode.LOW_IA_SCORE,
        description="Score IA faible même après tous les fixes — session potentiellement inutilisable",
        recoverable=False,
        fix_module=None,
        priority=99,
    ),
}


@dataclass
class DiagnosedProblem:
    """Problème détecté sur une session spécifique."""
    code:        ProblemCode
    spec:        ProblemSpec
    message:     str                # description détaillée du problème
    details:     dict = field(default_factory=dict)  # données brutes mesurées

    @property
    def recoverable(self) -> bool:
        return self.spec.recoverable

    @property
    def fix_module(self) -> Optional[str]:
        return self.spec.fix_module

    @property
    def priority(self) -> int:
        return self.spec.priority

    def __repr__(self) -> str:
        return f"Problem({self.code.value}: {self.message})"


def get_spec(code: ProblemCode) -> ProblemSpec:
    return PROBLEM_REGISTRY[code]


def make_problem(code: ProblemCode, message: str, **details) -> DiagnosedProblem:
    return DiagnosedProblem(
        code=code,
        spec=PROBLEM_REGISTRY[code],
        message=message,
        details=details,
    )
