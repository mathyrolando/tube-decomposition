"""Procedencia de resultados: commit de git con el que se lanzó la corrida.

GIT se calcula una sola vez, al importar el módulo (es decir, al lanzar el script),
así que no cambia aunque el código se edite mientras la corrida está en marcha.
dirty=True significa que había cambios sin commitear en el repositorio (sin contar results/):
esa cifra no es reproducible desde el commit.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=LAB_DIR, capture_output=True, text=True, check=True).stdout.strip()


def git_info() -> dict:
    try:
        commit = _git("rev-parse", "--short", "HEAD")
        changes = _git("status", "--porcelain", "--", ".", ":(exclude)results")
        return {"commit": commit, "dirty": bool(changes)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


GIT = git_info()
