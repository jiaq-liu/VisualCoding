"""Paths, interpreter and API-key resolution shared by every baseline adapter.

Design rule: this package is import-safe from ANY virtualenv (stdlib only).
Everything that needs the verify stack (executor rollouts, the LLM judge, image
preprocessing) is reached through a subprocess running the MAIN environment's
interpreter (VC_PY), so a baseline may live in its own venv with
its own dependency pins.
"""
from __future__ import annotations

import os
from pathlib import Path

BASELINES_ROOT = Path(__file__).resolve().parent.parent          # <repo>/baselines
REPO_ROOT = BASELINES_ROOT.parent                                # <repo>
VC_ROOT = Path(os.environ.get("VC_ROOT", REPO_ROOT / "visualcoding"))       # the verify package + its output tree
VC_PY = Path(os.environ.get("VC_PY", REPO_ROOT / ".venv" / "bin" / "python"))   # interpreter of the MAIN environment (setup_env.sh)
DATA_SPLIT = Path(os.environ.get("DATA_SPLIT", os.environ.get("VC_DATA", REPO_ROOT / "data")))
IMAGE_ROOT = Path(os.environ.get("IMAGE_ROOT", os.environ.get("VC_IMAGE_ROOT", DATA_SPLIT / "images")))
OUT_ROOT = Path(os.environ.get("BASELINE_OUT", REPO_ROOT / "output" / "baselines"))

# Rollout trees scanned for reusable rollouts (newest-looking last is fine; all are scanned).
ROLLOUT_ROOTS = [Path(p) for p in os.environ.get(
    "ROLLOUT_ROOTS", str(VC_ROOT / "output" / "verify")).split(os.pathsep) if p]

MODEL = os.environ.get("BASELINE_MODEL", "gemini-3.1-pro-preview")
JUDGE_MODEL = os.environ.get("BASELINE_JUDGE_MODEL", "gemini-3.1-flash-lite")

_ENV_CANDIDATES = [os.environ.get("VC_ENV_FILE"), REPO_ROOT / ".env", VC_ROOT / ".env"]


def gemini_api_key() -> str:
    """GEMINI_API_KEY from the environment, else the first .env that carries one (never logged)."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    for cand in _ENV_CANDIDATES:
        if not cand:
            continue
        p = Path(cand)
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip().lower() == "gemini_api_key":
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    raise RuntimeError("no Gemini key: set GEMINI_API_KEY or VC_ENV_FILE")


def subprocess_env(extra: dict | None = None) -> dict:
    """os.environ + the API key, for calls into the verify stack."""
    env = dict(os.environ)
    env["GEMINI_API_KEY"] = gemini_api_key()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def check() -> list[str]:
    """Return a list of problems with the local setup ([] when everything resolves)."""
    problems = []
    if not (VC_ROOT / "verify" / "run.py").is_file():
        problems.append(f"verify package not found under {VC_ROOT}")
    if not VC_PY.is_file():
        problems.append(f"main-environment interpreter not found: {VC_PY} (set VC_PY)")
    if not DATA_SPLIT.is_dir():
        problems.append(f"data_split not found: {DATA_SPLIT}")
    if not IMAGE_ROOT.is_dir():
        problems.append(f"image root not found: {IMAGE_ROOT}")
    try:
        gemini_api_key()
    except RuntimeError as e:
        problems.append(str(e))
    return problems
