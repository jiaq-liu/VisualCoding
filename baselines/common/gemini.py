"""Gemini through the google-genai SDK, for the baselines' own optimizer / reflection calls.

This is the LM the baselines call from inside their own loops (GEPA's reflection_lm,
TextGrad's backward engine, SkillOpt's optimizer role, XSkill's critique / skill builder,
Trace2Skill's analysts).  It runs inside the baseline's venv, so that venv needs
`pip install google-genai`; everything that touches the verify stack goes through a
subprocess instead (see executor.py / vlm.py).

Accepts a plain string, an OpenAI-style message list, or a multimodal message list
(content parts with image_url data: URIs) — that is exactly GEPA's LanguageModel protocol
and close enough to TextGrad's engine contract to wrap.

Conventions copied from visualcoding/verify/vendor/harness/utils/gemini_sdk.py (the tested pro path):
  * gemini-3.x pro cannot disable thinking; its floor is thinking_level="low",
  * thinking tokens are billed inside max_output_tokens, so the cap gets head-room,
  * empty replies are retried with a temperature bump (deterministic empties are real).
"""
from __future__ import annotations

import base64
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field

from .paths import MODEL, gemini_api_key

_CLIENTS: dict[str, object] = {}
_LOCK = threading.Lock()
USAGE = {"n_calls": 0, "in_tokens": 0, "out_tokens": 0, "think_tokens": 0, "retries": 0}

_DATA_URI = re.compile(r"^data:(?P<mime>[\w./+-]+);base64,(?P<b64>.*)$", re.S)


def _sdk():
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:                                  # pragma: no cover
        raise ImportError("pip install google-genai in this venv") from e
    return genai, types


def client(api_key: str | None = None):
    key = api_key or gemini_api_key()
    with _LOCK:
        c = _CLIENTS.get(key)
        if c is None:
            genai, _ = _sdk()
            c = genai.Client(api_key=key)
            _CLIENTS[key] = c
        return c


# ----------------------------------------------------------------------------- message conversion
def _part_from_image_url(types, url: str):
    m = _DATA_URI.match(url.strip())
    if m:
        return types.Part.from_bytes(data=base64.b64decode(m.group("b64")), mime_type=m.group("mime"))
    if os.path.isfile(url):
        ext = os.path.splitext(url)[1].lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/png")
        with open(url, "rb") as f:
            return types.Part.from_bytes(data=f.read(), mime_type=mime)
    return types.Part.from_uri(file_uri=url, mime_type="image/png")


def to_contents(prompt, types) -> tuple[list, str | None]:
    """(contents, system_instruction) from a string or an OpenAI-style message list."""
    if isinstance(prompt, str):
        return [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])], None
    system, contents = [], []
    for msg in prompt:
        role = msg.get("role", "user")
        content = msg.get("content")
        parts = []
        if isinstance(content, str):
            if content.strip():
                parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    parts.append(types.Part.from_text(text=str(item)))
                elif item.get("type") == "text":
                    parts.append(types.Part.from_text(text=item.get("text", "")))
                elif item.get("type") in ("image_url", "image"):
                    url = (item.get("image_url") or {}).get("url") if item.get("type") == "image_url" else item.get("url")
                    if url:
                        parts.append(_part_from_image_url(types, url))
        if not parts:
            continue
        if role == "system":
            system += [p.text for p in parts if getattr(p, "text", None)]
            continue
        contents.append(types.Content(role="model" if role == "assistant" else "user", parts=parts))
    return contents, ("\n\n".join(system) if system else None)


def _thinking(types, model: str, explicit):
    if explicit is not None:
        return explicit
    name = (model or "").lower()
    if "gemini-3" not in name:
        return None
    if "pro" in name:                       # cannot be switched off; low is the floor
        return types.ThinkingConfig(thinking_level=os.environ.get("GEMINI_THINKING_LEVEL", "low"))
    return types.ThinkingConfig(thinking_budget=int(os.environ.get("GEMINI_THINKING_BUDGET", "0")))


@dataclass
class Gemini:
    """Callable LM. `Gemini()(prompt)` -> text; conforms to GEPA's LanguageModel protocol."""
    model: str = MODEL
    temperature: float = 1.0
    max_tokens: int = 8192
    top_p: float = 1.0
    retries: int = 5
    timeout_s: float = 600.0
    thinking_config: object | None = None
    api_key: str | None = None
    empty_temp_bump: float = 0.3
    log: object = None
    usage: dict = field(default_factory=lambda: dict(n_calls=0, in_tokens=0, out_tokens=0, think_tokens=0))

    def __call__(self, prompt, **kw) -> str:
        return self.generate(prompt, **kw)

    def generate(self, prompt, *, system: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None, images: list | None = None) -> str:
        genai, types = _sdk()
        if images:
            parts = [{"type": "image_url", "image_url": {"url": u}} for u in images]
            prompt = ([{"role": "user", "content": parts + [{"type": "text", "text": prompt}]}]
                      if isinstance(prompt, str) else prompt)
        contents, sys_from_msgs = to_contents(prompt, types)
        system_instruction = system or sys_from_msgs
        temp = self.temperature if temperature is None else temperature
        cap = self.max_tokens if max_tokens is None else max_tokens
        think = _thinking(types, self.model, self.thinking_config)
        if think is not None and "pro" in self.model.lower():
            cap += int(os.environ.get("GEMINI_THINKING_HEADROOM", "4096"))
        last_err = None
        for attempt in range(self.retries + 1):
            cfg = types.GenerateContentConfig(
                temperature=temp, top_p=self.top_p, max_output_tokens=cap,
                system_instruction=system_instruction, thinking_config=think,
                http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))
            try:
                resp = client(self.api_key).models.generate_content(
                    model=self.model, contents=contents, config=cfg)
                text = (getattr(resp, "text", None) or "").strip()
                self._account(resp)
                if text:
                    return text
                last_err = f"empty content (finish={_finish(resp)})"
                temp = min(1.0, temp + self.empty_temp_bump)        # deterministic empties need a nudge
            except Exception as e:                                   # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                if not _retryable(e) or attempt == self.retries:
                    if attempt == self.retries:
                        break
                    raise
            USAGE["retries"] += 1
            if attempt < self.retries:
                time.sleep(min(30.0, 2 ** attempt + random.random()))
        if self.log:
            self.log(f"[gemini] giving up after {self.retries + 1} attempts: {last_err}")
        return ""

    def _account(self, resp) -> None:
        u = getattr(resp, "usage_metadata", None)
        vals = {"in_tokens": getattr(u, "prompt_token_count", 0) or 0,
                "out_tokens": getattr(u, "candidates_token_count", 0) or 0,
                "think_tokens": getattr(u, "thoughts_token_count", 0) or 0} if u else {}
        with _LOCK:
            USAGE["n_calls"] += 1
            self.usage["n_calls"] += 1
            for k, v in vals.items():
                USAGE[k] += int(v)
                self.usage[k] += int(v)


def _finish(resp) -> str:
    try:
        return str(resp.candidates[0].finish_reason)
    except Exception:                                            # noqa: BLE001
        return "?"


def _retryable(e) -> bool:
    s = f"{type(e).__name__}: {e}".lower()
    return any(k in s for k in ("429", "resource_exhausted", "500", "503", "unavailable",
                                "deadline", "timeout", "internal"))


def complete(prompt, model: str = MODEL, **kw) -> str:
    """One-shot helper."""
    lm_kw = {k: kw.pop(k) for k in ("temperature", "max_tokens", "top_p", "thinking_config", "retries")
             if k in kw}
    return Gemini(model=model, **lm_kw)(prompt, **kw)
