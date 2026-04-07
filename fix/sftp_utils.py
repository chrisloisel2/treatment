#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix/sftp_utils.py — Utilitaires SFTP pour accéder aux sessions du serveur HDD.

Permet :
  - Lister les sessions sur bronze/silver/gold
  - Télécharger une session complète (avec cache local)
  - Uploader une session corrigée
  - Lister et télécharger des sessions pour l'entraînement de modèles

Structure du serveur :
  /mnt/storage/bronze/  — sessions brutes ingérées
  /mnt/storage/silver/  — sessions partiellement traitées
  /mnt/storage/gold/    — sessions validées

Chaque session : {tier}/{session_id}/
  ├── metadata.json
  ├── tracker_positions.csv
  └── videos/
      ├── head.jsonl  head.mp4
      ├── left.jsonl  left.mp4
      └── right.jsonl right.mp4
"""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

# ── Dépendance SFTP ───────────────────────────────────────────────────────────
try:
    import paramiko
    _PARAMIKO = True
except ImportError:
    _PARAMIKO = False


# ══════════════════════════════════════════════════════════════════════════════
# Configuration serveur
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HDDConfig:
    host:         str = "192.168.88.82"
    port:         int = 22
    username:     str = "exoria"
    password:     str = "Admin123456"
    inbox_base:   str = "/mnt/inbox"
    bronze_base:  str = "/mnt/storage/bronze"
    silver_base:  str = "/mnt/storage/silver"
    gold_base:    str = "/mnt/storage/gold"
    send_base:    str = "/mnt/storage/send"
    retry_base:   str = "/mnt/storage/retry"

    def tier_path(self, tier: str) -> str:
        return {
            "inbox":  self.inbox_base,
            "bronze": self.bronze_base,
            "silver": self.silver_base,
            "gold":   self.gold_base,
            "send":   self.send_base,
            "retry":  self.retry_base,
        }[tier]


_DEFAULT_CONFIG = HDDConfig()

# Fichiers requis par session (pour vérifier qu'une session est complète)
SESSION_REQUIRED_FILES = [
    "metadata.json",
    "tracker_positions.csv",
    "videos/head.jsonl",
    "videos/left.jsonl",
    "videos/right.jsonl",
]

SESSION_OPTIONAL_FILES = [
    "videos/head.mp4",
    "videos/left.mp4",
    "videos/right.mp4",
    "gripper_left_data.csv",
    "gripper_right_data.csv",
]


# ══════════════════════════════════════════════════════════════════════════════
# Client SFTP
# ══════════════════════════════════════════════════════════════════════════════

class SFTPClient:
    """Client SFTP avec reconnexion automatique et cache local."""

    def __init__(self, config: HDDConfig = _DEFAULT_CONFIG):
        if not _PARAMIKO:
            raise ImportError("paramiko non installé — pip install paramiko")
        self.config = config
        self._ssh: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None

    # ── Connexion ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.connect(
            hostname=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self.config.password,
            timeout=15,
            banner_timeout=30,
            look_for_keys=False,
            allow_agent=False,
        )
        self._sftp = self._ssh.open_sftp()

    def disconnect(self) -> None:
        if self._sftp:
            self._sftp.close()
            self._sftp = None
        if self._ssh:
            self._ssh.close()
            self._ssh = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def _ensure_connected(self):
        if self._sftp is None:
            self.connect()

    # ── Listing ────────────────────────────────────────────────────────────────

    def list_sessions(
        self,
        tier: str = "bronze",
        require_complete: bool = True,
    ) -> List[str]:
        """
        Liste les sessions disponibles sur le serveur.

        Args:
            tier              : bronze / silver / gold / inbox
            require_complete  : si True, ne retourne que les sessions avec tous les fichiers requis

        Returns:
            Liste de session_id
        """
        self._ensure_connected()
        base = self.config.tier_path(tier)
        try:
            entries = self._sftp.listdir(base)
        except FileNotFoundError:
            return []

        sessions = []
        for entry in sorted(entries):
            session_path = f"{base}/{entry}"
            try:
                st = self._sftp.stat(session_path)
                if not stat.S_ISDIR(st.st_mode):
                    continue
            except Exception:
                continue

            if require_complete:
                missing = []
                for rel in SESSION_REQUIRED_FILES:
                    try:
                        self._sftp.stat(f"{session_path}/{rel}")
                    except FileNotFoundError:
                        missing.append(rel)
                if missing:
                    continue

            sessions.append(entry)

        return sessions

    def list_all_tiers(self) -> dict[str, List[str]]:
        """Liste les sessions de tous les tiers."""
        result = {}
        for tier in ("bronze", "silver", "gold"):
            try:
                result[tier] = self.list_sessions(tier)
            except Exception as e:
                result[tier] = []
                print(f"[sftp] Erreur listing {tier}: {e}")
        return result

    # ── Téléchargement ─────────────────────────────────────────────────────────

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Télécharge un fichier distant vers local_path."""
        self._ensure_connected()
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._sftp.get(remote_path, str(local_path))
            return True
        except Exception as e:
            print(f"[sftp] Erreur download {remote_path}: {e}")
            return False

    def download_session(
        self,
        session_id: str,
        local_dir: Path,
        tier: str = "bronze",
        with_videos: bool = False,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Télécharge une session complète dans local_dir/{session_id}/.

        Args:
            session_id  : identifiant de la session
            local_dir   : dossier local de destination
            tier        : bronze / silver / gold
            with_videos : si True, télécharge aussi les MP4 (volumineux)
            force       : si True, re-télécharge même si déjà présent

        Returns:
            Path vers la session locale, ou None si échec
        """
        self._ensure_connected()
        local_session = local_dir / session_id
        remote_session = f"{self.config.tier_path(tier)}/{session_id}"

        # Vérifier si déjà téléchargé (complet)
        if not force and local_session.exists():
            all_ok = all(
                (local_session / rel).exists()
                for rel in SESSION_REQUIRED_FILES
            )
            if all_ok:
                return local_session

        local_session.mkdir(parents=True, exist_ok=True)

        files_to_download = SESSION_REQUIRED_FILES.copy()
        if with_videos:
            files_to_download.extend(SESSION_OPTIONAL_FILES)
        else:
            # Toujours prendre les gripper CSV si présents
            for rel in SESSION_OPTIONAL_FILES:
                if not rel.startswith("videos/"):
                    files_to_download.append(rel)

        ok_count = 0
        for rel in files_to_download:
            remote_file = f"{remote_session}/{rel}"
            local_file  = local_session / rel
            try:
                self._sftp.stat(remote_file)
                if self.download_file(remote_file, local_file):
                    ok_count += 1
            except FileNotFoundError:
                pass  # Fichier optionnel absent
            except Exception as e:
                print(f"[sftp] {session_id}/{rel}: {e}")

        # Vérifier que les fichiers requis sont là
        required_ok = all((local_session / rel).exists() for rel in SESSION_REQUIRED_FILES)
        if not required_ok:
            shutil.rmtree(local_session, ignore_errors=True)
            return None

        return local_session

    def download_sessions_for_training(
        self,
        local_dir: Path,
        tier: str = "bronze",
        max_sessions: Optional[int] = None,
        with_videos: bool = False,
    ) -> List[Path]:
        """
        Télécharge des sessions pour l'entraînement d'un modèle.

        Returns:
            Liste de chemins locaux des sessions téléchargées
        """
        self._ensure_connected()
        sessions = self.list_sessions(tier, require_complete=True)
        if max_sessions is not None:
            sessions = sessions[:max_sessions]

        print(f"[sftp] {len(sessions)} sessions disponibles sur {tier}")

        local_paths = []
        for i, session_id in enumerate(sessions, 1):
            print(f"  [{i}/{len(sessions)}] {session_id}...", end="", flush=True)
            path = self.download_session(
                session_id, local_dir, tier=tier, with_videos=with_videos
            )
            if path:
                local_paths.append(path)
                print(" OK")
            else:
                print(" FAIL")

        print(f"[sftp] {len(local_paths)}/{len(sessions)} sessions téléchargées → {local_dir}")
        return local_paths

    # ── Upload ─────────────────────────────────────────────────────────────────

    def upload_session(
        self,
        local_session: Path,
        tier: str = "bronze",
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Upload une session locale vers le serveur.

        Args:
            local_session : chemin local de la session
            tier          : tier de destination
            session_id    : ID remote (par défaut = nom du dossier local)

        Returns:
            True si succès
        """
        self._ensure_connected()
        sid = session_id or local_session.name
        remote_base = f"{self.config.tier_path(tier)}/{sid}"

        try:
            self._mkdir_remote(remote_base)
            self._mkdir_remote(f"{remote_base}/videos")
        except Exception as e:
            print(f"[sftp] Impossible de créer {remote_base}: {e}")
            return False

        ok_count = 0
        for rel in SESSION_REQUIRED_FILES + SESSION_OPTIONAL_FILES:
            local_file = local_session / rel
            if not local_file.exists():
                continue
            remote_file = f"{remote_base}/{rel}"
            try:
                self._sftp.put(str(local_file), remote_file)
                ok_count += 1
            except Exception as e:
                print(f"[sftp] Upload {rel}: {e}")

        return ok_count >= len(SESSION_REQUIRED_FILES)

    def _mkdir_remote(self, path: str) -> None:
        """Crée un répertoire distant (et ses parents si nécessaire)."""
        try:
            self._sftp.stat(path)
            return  # Existe déjà
        except FileNotFoundError:
            pass
        parts = path.rstrip("/").split("/")
        current = ""
        for part in parts:
            if not part:
                continue
            current = f"/{part}" if not current else f"{current}/{part}"
            try:
                self._sftp.stat(current)
            except FileNotFoundError:
                try:
                    self._sftp.mkdir(current)
                except Exception:
                    pass

    # ── Lecture directe (sans téléchargement) ─────────────────────────────────

    def read_metadata(self, session_id: str, tier: str = "bronze") -> Optional[dict]:
        """Lit le metadata.json d'une session sans tout télécharger."""
        self._ensure_connected()
        remote = f"{self.config.tier_path(tier)}/{session_id}/metadata.json"
        try:
            buf = io.BytesIO()
            self._sftp.getfo(remote, buf)
            return json.loads(buf.getvalue().decode("utf-8"))
        except Exception:
            return None

    def read_jsonl_times(self, session_id: str, cam: str,
                          tier: str = "bronze") -> Optional[list]:
        """Lit les capture_time d'un JSONL caméra sans télécharger la session."""
        self._ensure_connected()
        remote = f"{self.config.tier_path(tier)}/{session_id}/videos/{cam}.jsonl"
        try:
            buf = io.BytesIO()
            self._sftp.getfo(remote, buf)
            times = []
            for line in buf.getvalue().replace(b"\r\n", b"\n").split(b"\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ct = obj.get("capture_time")
                    if ct is not None:
                        times.append(float(ct))
                except Exception:
                    pass
            return times if times else None
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Fonctions de haut niveau
# ══════════════════════════════════════════════════════════════════════════════

def download_training_data(
    local_dir: Path,
    tiers: tuple[str, ...] = ("bronze", "silver", "gold"),
    max_per_tier: Optional[int] = None,
    with_videos: bool = False,
    config: HDDConfig = _DEFAULT_CONFIG,
) -> List[Path]:
    """
    Télécharge les données d'entraînement depuis tous les tiers du serveur.

    Usage :
        paths = download_training_data(Path("/tmp/training_sessions"))
        # → entraîner un modèle sur paths
    """
    if not _PARAMIKO:
        print("[sftp] paramiko non installé — téléchargement impossible")
        return []

    local_dir.mkdir(parents=True, exist_ok=True)
    all_paths = []

    with SFTPClient(config) as client:
        for tier in tiers:
            tier_dir = local_dir / tier
            tier_dir.mkdir(exist_ok=True)
            paths = client.download_sessions_for_training(
                tier_dir, tier=tier,
                max_sessions=max_per_tier,
                with_videos=with_videos,
            )
            all_paths.extend(paths)

    return all_paths


def list_server_sessions(
    config: HDDConfig = _DEFAULT_CONFIG,
) -> dict[str, List[str]]:
    """Liste toutes les sessions disponibles sur le serveur, par tier."""
    if not _PARAMIKO:
        print("[sftp] paramiko non installé")
        return {}
    with SFTPClient(config) as client:
        return client.list_all_tiers()
