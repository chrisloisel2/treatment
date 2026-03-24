#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rotate_videos.py — Rotation à 180° des vidéos d'une session.

Utilise FFmpeg (vf transpose=2,transpose=2 ou rotate=PI) pour effectuer
la rotation sans re-encodage de qualité si possible (via -c:v copy avec
rotation metadata), ou avec re-encodage minimal (libx264 crf=18) sinon.

La rotation est appliquée dans le flux vidéo (pixel transform), pas
uniquement via un flag de métadonnée, pour une compatibilité maximale
avec les lecteurs et OpenCV.

Les fichiers originaux sont sauvegardés en .bak_rotate avant modification.
Un fichier .rotate_done est écrit dans videos/ pour marquer la completion
et permettre à la pipeline de passer cette étape si déjà faite.

Usage standalone :
    python rotate_videos.py /path/to/session [--sides head left right] [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

VIDEO_SIDES = ("head", "left", "right")
ROTATE_MARKER = ".rotate_done"


def _ffmpeg_available() -> Optional[str]:
    """Retourne le chemin de ffmpeg ou None si absent."""
    for candidate in ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if shutil.which(candidate):
            return candidate
    return None


def _probe_rotation(ffmpeg: str, mp4: Path) -> int:
    """
    Lit la métadonnée de rotation dans le fichier MP4.
    Retourne 0, 90, 180 ou 270.
    """
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if not shutil.which(ffprobe):
        ffprobe = "ffprobe"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_streams", str(mp4)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(r.stdout)
        for stream in data.get("streams", []):
            # FFmpeg < 5.x : rotation dans les tags
            tags = stream.get("tags", {})
            rot = tags.get("rotate", tags.get("Rotate", "0"))
            try:
                return int(rot)
            except (ValueError, TypeError):
                pass
            # FFmpeg ≥ 5.x : side_data_list
            for sd in stream.get("side_data_list", []):
                if sd.get("side_data_type") == "Display Matrix":
                    try:
                        return int(sd.get("rotation", 0))
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return 0


def rotate_video_180(
    ffmpeg: str,
    src: Path,
    dst: Path,
    log=None,
) -> bool:
    """
    Retourne src à 180° et écrit le résultat dans dst.
    Stratégie : vf hflip,vflip (équivalent rotation 180° pixel-exact).
    Encodeur : libx264 crf=18 preset=fast pour la compatibilité maximale.

    Retourne True si succès, False sinon.
    """
    def _log(msg, level="INFO"):
        if log:
            log(msg, level)
        else:
            print(f"[{level}] {msg}")

    # Fichier temporaire pour éviter d'écraser src si dst == src
    tmp = dst.with_suffix(".tmp_rotate.mp4")

    cmd = [
        ffmpeg,
        "-y",                       # overwrite
        "-i", str(src),
        "-vf", "hflip,vflip",       # rotation 180° (hflip + vflip = rotate PI)
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "copy",             # audio inchangé
        # Effacer les métadonnées de rotation pour éviter double-rotation
        "-metadata:s:v:0", "rotate=0",
        "-movflags", "+faststart",
        str(tmp),
    ]

    _log(f"Rotation 180° : {src.name} …")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max
        )
    except subprocess.TimeoutExpired:
        _log(f"Timeout rotation {src.name}", "ERROR")
        tmp.unlink(missing_ok=True)
        return False
    except Exception as e:
        _log(f"Erreur FFmpeg {src.name}: {e}", "ERROR")
        tmp.unlink(missing_ok=True)
        return False

    if result.returncode != 0:
        err = result.stderr.strip()[-600:]
        _log(f"FFmpeg erreur ({src.name}): {err}", "ERROR")
        tmp.unlink(missing_ok=True)
        return False

    # Vérifier que le fichier de sortie a une taille raisonnable
    if not tmp.exists() or tmp.stat().st_size < 1024:
        _log(f"Fichier de sortie invalide : {tmp}", "ERROR")
        tmp.unlink(missing_ok=True)
        return False

    tmp.rename(dst)
    elapsed = time.time() - t0
    _log(f"Rotation OK : {dst.name} ({elapsed:.1f}s)", "OK")
    return True


def rotate_session_videos(
    session_dir: Path,
    sides: List[str] = list(VIDEO_SIDES),
    force: bool = False,
    log=None,
) -> dict:
    """
    Applique la rotation 180° à toutes les vidéos d'une session.

    - Sauvegarde les originaux en <side>.mp4.bak_rotate
    - Écrit le fichier .rotate_done dans videos/ après succès total
    - Si force=False et .rotate_done existe, passe sans rien faire

    Retourne un dict avec "rotated", "skipped", "errors", "already_done".
    """
    def _log(msg, level="INFO"):
        if log:
            log(msg, level)
        else:
            print(f"[{level}] {msg}")

    vid_dir = session_dir / "videos"
    marker  = vid_dir / ROTATE_MARKER

    if marker.exists() and not force:
        _log("Rotation déjà effectuée (.rotate_done présent) — skip", "INFO")
        return {"rotated": [], "skipped": list(sides), "errors": [], "already_done": True}

    ffmpeg = _ffmpeg_available()
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg introuvable. Installez-le avec : brew install ffmpeg  "
            "ou  apt install ffmpeg"
        )

    if not vid_dir.exists():
        raise FileNotFoundError(f"Dossier vidéo absent : {vid_dir}")

    rotated = []
    skipped = []
    errors  = []

    for side in sides:
        mp4 = vid_dir / f"{side}.mp4"
        bak = vid_dir / f"{side}.mp4.bak_rotate"

        if not mp4.exists():
            _log(f"{side}.mp4 absent — skip", "WARN")
            skipped.append(side)
            continue

        # Vérifier si déjà une rotation 180° en métadonnée (signalerait double-rotation)
        existing_rot = _probe_rotation(ffmpeg, mp4)
        if existing_rot == 180 and not force:
            _log(f"{side}.mp4 : rotation 180° déjà présente en métadonnée, "
                 f"rotation pixel appliquée quand même (--force pour ignorer)", "WARN")

        # Backup de l'original
        if not bak.exists():
            _log(f"Backup : {mp4.name} → {bak.name}")
            shutil.copy2(mp4, bak)
        else:
            _log(f"Backup existant conservé : {bak.name}", "INFO")

        # Rotation en place : on écrit dans un tmp puis on remplace
        ok = rotate_video_180(ffmpeg, bak, mp4, log=log)
        if ok:
            rotated.append(side)
        else:
            errors.append(side)
            # Restaurer le backup si rotation échouée
            shutil.copy2(bak, mp4)
            _log(f"Restauration backup {side}.mp4 après échec", "WARN")

    # Écrire le marqueur seulement si toutes les rotations réussies
    if not errors:
        marker.write_text(
            json.dumps({
                "rotated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sides": rotated,
                "ffmpeg": ffmpeg,
            }, indent=2),
            encoding="utf-8",
        )
        _log(f"Marqueur .rotate_done écrit ({len(rotated)} vidéo(s))", "OK")
    else:
        _log(f"Erreurs sur {errors} — marqueur NON écrit", "WARN")

    return {
        "rotated":      rotated,
        "skipped":      skipped,
        "errors":       errors,
        "already_done": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI standalone
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Retourne à 180° les vidéos d'une session d'ingestion."
    )
    p.add_argument("session", type=str, help="Chemin de la session")
    p.add_argument(
        "--sides", nargs="+", default=list(VIDEO_SIDES),
        choices=list(VIDEO_SIDES),
        help="Faces à traiter (défaut: toutes)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Ré-appliquer même si .rotate_done existe",
    )
    args = p.parse_args()

    sess = Path(args.session)
    if not sess.exists():
        print(f"ERREUR : session introuvable : {sess}", file=sys.stderr)
        sys.exit(1)

    result = rotate_session_videos(sess, sides=args.sides, force=args.force)

    print(f"\nRésultat :")
    print(f"  Rotées    : {result['rotated']}")
    print(f"  Sautées   : {result['skipped']}")
    print(f"  Erreurs   : {result['errors']}")
    sys.exit(0 if not result["errors"] else 1)


if __name__ == "__main__":
    main()
