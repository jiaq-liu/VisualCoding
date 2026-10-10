"""Load API keys from <repo>/.env, visualcoding/.env or $VC_ENV_FILE and export what the vendored code reads.

Vendored the agent harness modules read:
  engine.api_caller.call_vision_api : REASONING_API_KEY, REASONING_END_POINT ('' -> google-genai SDK)
                                      REASONING_API_KEY_2 / REASONING_END_POINT_2 (optional fallback)
  utils.llm_judger                  : VERIFIER_API_KEY, VERIFIER_END_POINT, VERIFIER_MODEL_NAME
  utils.gemini_sdk                  : GEMINI_THINKING_LEVEL / GEMINI_THINKING_BUDGET / GEMINI_THINKING_HEADROOM
                                      (defaults: 3.x pro -> thinking_level=low, flash -> budget 0)
Variables already present in the process environment take precedence.
"""
from __future__ import annotations

import os
from pathlib import Path

from ._paths import PKG_ROOT


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_keys() -> dict[str, str]:
    """Merge the .env files (GEMINI_API_KEY or gemini_api_key); later files win.  Keys are never logged."""
    keys: dict[str, str] = {}
    keys.update(_read_env_file(PKG_ROOT.parent / ".env"))    # <repo>/.env (release layout)
    keys.update(_read_env_file(PKG_ROOT / ".env"))           # visualcoding/.env
    if os.environ.get("VC_ENV_FILE"):
        keys.update(_read_env_file(Path(os.environ["VC_ENV_FILE"])))
    return keys


def setup_env(model_name: str, judge_model: str | None = None) -> None:
    keys = load_keys()
    gemini = (os.environ.get("GEMINI_API_KEY") or keys.get("gemini_api_key")
              or keys.get("GEMINI_API_KEY") or "")
    if not gemini:
        raise RuntimeError("no Gemini API key: export GEMINI_API_KEY or put GEMINI_API_KEY=... into <repo>/.env")
    os.environ.setdefault("REASONING_API_KEY", gemini)
    os.environ.setdefault("REASONING_END_POINT", "")          # empty -> google-genai SDK path
    os.environ.setdefault("REASONING_MODEL_NAME", model_name)
    os.environ.setdefault("VERIFIER_API_KEY", gemini)
    os.environ.setdefault("VERIFIER_END_POINT", "")
    os.environ.setdefault("VERIFIER_MODEL_NAME", judge_model or model_name)
    os.environ.setdefault("GEMINI_DISABLE_SAFETY", "1")       # parity with the agent harness runs
    # gemini-3.5 flash / flash-lite reject thinking_budget=0 (400 INVALID_ARGUMENT); the SDK default for
    # 3.x flash is budget 0, so pin the lowest accepted level unless the caller set one.  3.1 flash-lite keeps thinking off.
    if "gemini-3.5" in model_name and not os.environ.get("GEMINI_THINKING_LEVEL") and not os.environ.get("GEMINI_THINKING_BUDGET"):
        os.environ["GEMINI_THINKING_LEVEL"] = "low"
