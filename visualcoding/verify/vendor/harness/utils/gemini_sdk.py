"""
Shared helpers for calling Gemini through the ``google-genai`` SDK.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from google.protobuf.json_format import MessageToDict
except ImportError:
    MessageToDict = None

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
except ImportError:
    _genai = None
    _genai_types = None


_CLIENT_CACHE: Dict[str, Any] = {}


def _safe_get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely read SDK attributes that may raise accessor errors."""
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def ensure_google_genai():
    if _genai is None or _genai_types is None:
        raise ImportError(
            "google-genai is required for Gemini models. "
            "Install it with `pip install -r requirements.txt`."
        )
    return _genai, _genai_types


def get_gemini_client(api_key: str):
    """Return a cached ``google.genai.Client`` keyed by API key."""
    genai, _ = ensure_google_genai()
    client = _CLIENT_CACHE.get(api_key)
    if client is None:
        client = genai.Client(api_key=api_key)
        _CLIENT_CACHE[api_key] = client
    return client


def _is_thinking_capable_model(model_name: Optional[str]) -> bool:
    """Models where thinking is enabled by default and must be opt-out for judge-style calls."""
    name = (model_name or "").lower()
    return "gemini-3" in name


def _thinking_cannot_be_disabled(model_name: Optional[str]) -> bool:
    """Gemini 3.x *Pro* rejects thinking_budget=0 ("Budget 0 is invalid. This model
    only works in thinking mode") and does not accept thinking_level=minimal; its
    lowest setting is thinking_level=low."""
    name = (model_name or "").lower()
    return "gemini-3" in name and "pro" in name


def _thinking_explicitly_off() -> bool:
    """True when the env forces thinking fully off (budget 0 / level minimal)."""
    level = os.environ.get("GEMINI_THINKING_LEVEL", "").strip().lower()
    budget = os.environ.get("GEMINI_THINKING_BUDGET", "").strip()
    return level == "minimal" or budget == "0"


def _minimum_thinking_config(types: Any, model_name: Optional[str]) -> Any:
    """Least-thinking config this model accepts (used for every Gemini call in the
    pipeline: agent, selector/skill-writer, LLM judge).

    Overrides (env, applied to all Gemini calls):
      GEMINI_THINKING_LEVEL   minimal|low|medium|high   -> ThinkingConfig(thinking_level=...)
      GEMINI_THINKING_BUDGET  <int>                     -> ThinkingConfig(thinking_budget=...)
    Defaults: gemini-3.x flash/flash-lite -> thinking_budget=0 (off);
              gemini-3.x pro              -> thinking_level="low" (cannot be turned off).
    """
    level = os.environ.get("GEMINI_THINKING_LEVEL", "").strip().lower()
    budget = os.environ.get("GEMINI_THINKING_BUDGET", "").strip()
    if level:
        return types.ThinkingConfig(thinking_level=level)
    if budget:
        try:
            return types.ThinkingConfig(thinking_budget=int(budget))
        except ValueError:
            pass
    if _thinking_cannot_be_disabled(model_name):
        return types.ThinkingConfig(thinking_level="low")
    return types.ThinkingConfig(thinking_budget=0)


# Configurable harm categories whose filters we disable. These benchmarks
# (AEGIS2 safety classification, crime-case facts, etc.) legitimately contain
# language the default filters flag — we are CLASSIFYING such text, not
# generating harmful content — so the default BLOCK_MEDIUM_AND_ABOVE thresholds
# reject otherwise-valid requests. Setting BLOCK_NONE removes those rejections.
#
# NOTE: the non-configurable PROHIBITED_CONTENT block CANNOT be turned off via
# safety_settings; those residual blocks are handled gracefully by the callers
# (treated as "no output for this sample" rather than crashing/looping).
_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)


def _default_safety_settings(types: Any) -> Optional[list]:
    """Build BLOCK_NONE safety settings for all configurable categories.

    Returns None when explicitly disabled via ``GEMINI_DISABLE_SAFETY=0`` (so
    the caller falls back to the API defaults). String enum values are used for
    robustness across google-genai versions, with a plain-dict fallback.
    """
    if os.environ.get("GEMINI_DISABLE_SAFETY", "1").strip().lower() in ("0", "false", "no"):
        return None
    settings: list = []
    for cat in _SAFETY_CATEGORIES:
        try:
            settings.append(types.SafetySetting(category=cat, threshold="BLOCK_NONE"))
        except Exception:
            settings.append({"category": cat, "threshold": "BLOCK_NONE"})
    return settings or None


def _build_generation_config(
    generation_config: Any,
    system_instruction: Any = None,
    tools: Any = None,
    tool_config: Any = None,
    model_name: Optional[str] = None,
    disable_thinking: bool = False,
    timeout: Optional[float] = None,
) -> Any:
    """Merge per-call params into a ``GenerateContentConfig`` object for the new SDK."""
    _, types = ensure_google_genai()

    if generation_config is None:
        config_kwargs: Dict[str, Any] = {}
    elif isinstance(generation_config, dict):
        config_kwargs = dict(generation_config)
    else:
        # Already a GenerateContentConfig (or dataclass-like). Take a shallow copy via model_dump.
        try:
            config_kwargs = generation_config.model_dump(exclude_none=True)
        except Exception:
            config_kwargs = {}

    if system_instruction is not None and "system_instruction" not in config_kwargs:
        config_kwargs["system_instruction"] = system_instruction
    if tools is not None and "tools" not in config_kwargs:
        config_kwargs["tools"] = tools
    if tool_config is not None and "tool_config" not in config_kwargs:
        config_kwargs["tool_config"] = tool_config

    if disable_thinking or _is_thinking_capable_model(model_name):
        config_kwargs.setdefault(
            "thinking_config", _minimum_thinking_config(types, model_name)
        )
        # Gemini bills thinking tokens INSIDE max_output_tokens. On models whose
        # thinking cannot be switched off (3.x Pro), a small cap meant for a
        # one-line / JSON reply (64-512) is eaten by the thoughts and the visible
        # text comes back truncated (finish_reason=MAX_TOKENS, e.g. "Extract").
        # Add head-room for the thoughts so the caller's cap applies to the
        # visible answer. GEMINI_THINKING_HEADROOM (tokens) overrides; 0 disables.
        if _thinking_cannot_be_disabled(model_name) and not _thinking_explicitly_off():
            cap = config_kwargs.get("max_output_tokens")
            if isinstance(cap, int) and cap > 0:
                try:
                    headroom = int(os.environ.get("GEMINI_THINKING_HEADROOM", "4096"))
                except ValueError:
                    headroom = 4096
                if headroom > 0:
                    config_kwargs["max_output_tokens"] = cap + headroom

    if "safety_settings" not in config_kwargs:
        _ss = _default_safety_settings(types)
        if _ss is not None:
            config_kwargs["safety_settings"] = _ss

    if timeout is not None and "http_options" not in config_kwargs:
        try:
            config_kwargs["http_options"] = types.HttpOptions(
                timeout=int(float(timeout) * 1000)
            )
        except Exception:
            pass

    return types.GenerateContentConfig(**config_kwargs)


def gemini_generate_content(
    client: Any,
    model_name: str,
    contents: Any,
    *,
    system_instruction: Any = None,
    tools: Any = None,
    tool_config: Any = None,
    generation_config: Any = None,
    timeout: Optional[float] = None,
    disable_thinking: bool = False,
) -> Any:
    """
    Invoke ``client.models.generate_content`` with the merged GenerateContentConfig.

    ``timeout`` is in seconds (matching legacy call sites) and is translated to
    ``HttpOptions(timeout=<milliseconds>)`` within the config.
    """
    stream = os.environ.get("GEMINI_STREAM", "").strip() == "1"
    if stream:
        try:
            timeout = float(os.environ.get("GEMINI_STREAM_TIMEOUT", "900"))
        except ValueError:
            timeout = 900.0
    config = _build_generation_config(
        generation_config=generation_config,
        system_instruction=system_instruction,
        tools=tools,
        tool_config=tool_config,
        model_name=model_name,
        disable_thinking=disable_thinking,
        timeout=timeout,
    )
    if not stream:
        return client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
    return _generate_streamed(client, model_name, contents, config)


class _MergedCandidate:
    def __init__(self, content, finish_reason):
        self.content = content
        self.finish_reason = finish_reason
        self.safety_ratings = None


class _MergedResponse:
    """What the stream adds up to, shaped like GenerateContentResponse for the parser."""

    def __init__(self, parts, finish_reason, usage_metadata, prompt_feedback, n_chunks, n_thought_chunks):
        _, types = ensure_google_genai()
        self.candidates = [_MergedCandidate(types.Content(role="model", parts=parts), finish_reason)]
        self.usage_metadata = usage_metadata
        self.prompt_feedback = prompt_feedback
        self.stream_stats = {"chunks": n_chunks, "thought_chunks": n_thought_chunks}

    @property
    def text(self):
        return "".join(getattr(p, "text", "") or "" for p in self.candidates[0].content.parts)


def _generate_streamed(client, model_name, contents, config):
    """generate_content_stream merged into one response (GEMINI_STREAM=1).  Thought parts (part.thought) keep the
    connection alive during thinking and are dropped; text parts are concatenated in order; function-call parts are
    kept as they come; the last chunk's finish_reason / usage_metadata are reported."""
    _, types = ensure_google_genai()
    try:
        tc = getattr(config, "thinking_config", None)
        if tc is not None and getattr(tc, "include_thoughts", None) is None:
            tc.include_thoughts = True
    except Exception:
        pass
    texts, others = [], []
    finish_reason, usage, feedback = None, None, None
    n_chunks = n_thought = 0
    for chunk in client.models.generate_content_stream(model=model_name, contents=contents, config=config):
        n_chunks += 1
        usage = getattr(chunk, "usage_metadata", None) or usage
        feedback = getattr(chunk, "prompt_feedback", None) or feedback
        cands = getattr(chunk, "candidates", None) or []
        if not cands:
            continue
        cand = cands[0]
        fr = getattr(cand, "finish_reason", None)
        if fr is not None:
            finish_reason = fr
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "thought", False):
                n_thought += 1
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str):
                texts.append(text)
            elif getattr(part, "function_call", None) is not None:
                others.append(part)
    parts = []
    if texts:
        parts.append(types.Part(text="".join(texts)))
    parts.extend(others)
    return _MergedResponse(parts, finish_reason, usage, feedback, n_chunks, n_thought)


def camel_to_snake(name: str) -> str:
    if not isinstance(name, str):
        return name
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def rest_to_sdk_value(value: Any) -> Any:
    """
    Convert Gemini REST-style camelCase dicts to SDK/proto-friendly snake_case dicts.
    Non-container values are returned unchanged.
    """
    if isinstance(value, dict):
        return {camel_to_snake(k): rest_to_sdk_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rest_to_sdk_value(v) for v in value]
    return value


def _to_json_compatible(value: Any) -> Any:
    """
    Recursively convert Gemini SDK / protobuf values into plain JSON-compatible
    Python types.

    ``bytes`` values (e.g. ``Part.thought_signature``) are encoded as base64
    strings so they round-trip safely through JSON storage and back into the
    google-genai pydantic validators (which accept base64 for ``bytes`` fields).
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")

    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(v) for v in value]

    # google-genai returns pydantic models for parts/content; prefer their
    # native ``model_dump`` for a clean dict form.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_json_compatible(model_dump(exclude_none=True))
        except Exception:
            pass

    if MessageToDict is not None:
        try:
            converted = MessageToDict(value, preserving_proto_field_name=True)
            return _to_json_compatible(converted)
        except Exception:
            pass

    if hasattr(value, "items"):
        try:
            return {str(k): _to_json_compatible(v) for k, v in value.items()}
        except Exception:
            pass

    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return [_to_json_compatible(v) for v in value]
        except Exception:
            pass

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def sdk_args_to_dict(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return _to_json_compatible(args)
    model_dump = getattr(args, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_json_compatible(model_dump(exclude_none=True))
        except Exception:
            pass
    if MessageToDict is not None:
        try:
            return _to_json_compatible(MessageToDict(args, preserving_proto_field_name=True))
        except Exception:
            pass
    if hasattr(args, "items"):
        try:
            return _to_json_compatible({k: v for k, v in args.items()})
        except Exception:
            pass
    converted = _to_json_compatible(args)
    if isinstance(converted, dict):
        return converted
    return {"raw_arguments": converted}


def _snake_to_camel(name: str) -> str:
    if not isinstance(name, str) or "_" not in name:
        return name
    head, *tail = name.split("_")
    return head + "".join(part.title() for part in tail if part)


def _dict_snake_to_camel(value: Any) -> Any:
    if isinstance(value, dict):
        return {_snake_to_camel(k): _dict_snake_to_camel(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dict_snake_to_camel(v) for v in value]
    return value


def sdk_part_to_rest_dict(part: Any) -> Dict[str, Any]:
    """
    Convert a Gemini SDK part object into a plain dict using the lowerCamelCase
    field names expected by the existing REST-style payload builder.
    """
    if part is None:
        return {}
    if isinstance(part, dict):
        return _dict_snake_to_camel(_to_json_compatible(part))

    # google-genai returns pydantic models — ``model_dump`` produces snake_case;
    # convert back to camelCase so downstream REST-shaped payloads stay stable.
    model_dump = getattr(part, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_none=True)
            return _dict_snake_to_camel(_to_json_compatible(dumped))
        except Exception:
            pass

    if MessageToDict is not None:
        try:
            converted = MessageToDict(part, preserving_proto_field_name=False)
            return _to_json_compatible(converted)
        except Exception:
            pass

    converted = _to_json_compatible(part)
    if isinstance(converted, dict):
        return _dict_snake_to_camel(converted)
    return {}


def sdk_parts_to_rest_list(parts: Any) -> List[Dict[str, Any]]:
    """Convert a Gemini SDK parts container into a list of plain REST-style dicts."""
    if not parts:
        return []

    converted_parts: List[Dict[str, Any]] = []
    for part in parts:
        converted = sdk_part_to_rest_dict(part)
        if converted:
            converted_parts.append(converted)
    return converted_parts


def extract_text_from_sdk_response(resp: Any) -> str:
    candidates = _safe_get_attr(resp, "candidates", None) or []
    if not candidates:
        return ""

    content = _safe_get_attr(candidates[0], "content", None)
    parts = _safe_get_attr(content, "parts", None) if content is not None else None
    if parts:
        texts = []
        for part in parts:
            # google-genai marks internal reasoning parts with ``thought=True`` — skip them
            # so only the final answer text is returned.
            if _safe_get_attr(part, "thought", False):
                continue
            text = _safe_get_attr(part, "text", "")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        combined = "".join(texts).strip()
        if combined:
            return combined

    # Fall back to the SDK's own ``text`` convenience accessor.
    try:
        fallback = _safe_get_attr(resp, "text", None)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    except Exception:
        pass
    return ""


def extract_block_reason(resp: Any) -> str:
    """Return the prompt-feedback block reason when the request was filtered."""
    feedback = _safe_get_attr(resp, "prompt_feedback", None)
    if feedback is None:
        return ""
    reason = _safe_get_attr(feedback, "block_reason", None)
    if reason is None:
        return ""
    name = _safe_get_attr(reason, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(reason)
    return text if text and text != "None" else ""


def extract_finish_reason(resp: Any) -> str:
    """Extract the first candidate finish reason without triggering unsafe SDK accessors."""
    candidates = _safe_get_attr(resp, "candidates", None) or []
    if not candidates:
        return ""

    finish_reason = _safe_get_attr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        return ""

    name = _safe_get_attr(finish_reason, "name", None)
    if isinstance(name, str) and name:
        return name

    text = str(finish_reason)
    return text if text != "None" else ""


def normalize_sdk_usage(resp: Any) -> Dict[str, int]:
    metadata = _safe_get_attr(resp, "usage_metadata", None)
    if isinstance(metadata, dict):
        prompt_token_count = metadata.get("prompt_token_count")
        candidates_token_count = metadata.get("candidates_token_count")
        total_token_count = metadata.get("total_token_count")
    else:
        prompt_token_count = _safe_get_attr(metadata, "prompt_token_count", None) if metadata is not None else None
        candidates_token_count = _safe_get_attr(metadata, "candidates_token_count", None) if metadata is not None else None
        total_token_count = _safe_get_attr(metadata, "total_token_count", None) if metadata is not None else None

    def _to_int(value: Any) -> int:
        try:
            if value in (None, ""):
                return 0
            return int(value)
        except Exception:
            return 0

    input_tokens = _to_int(prompt_token_count)
    output_tokens = _to_int(candidates_token_count)
    total_tokens = _to_int(total_token_count) or (input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "calls": 1 if (input_tokens or output_tokens or total_tokens) else 0,
    }


def extract_status_code(exc: Exception) -> Optional[int]:
    # google-genai surfaces errors as ``ClientError`` / ``ServerError`` with ``code`` attr.
    direct_code = getattr(exc, "code", None)
    if callable(direct_code):
        try:
            direct_code = direct_code()
        except Exception:
            direct_code = None
    try:
        if direct_code is not None:
            return int(direct_code)
    except Exception:
        pass

    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "status_code"):
        try:
            return int(response.status_code)
        except Exception:
            pass

    for attr in ("status_code",):
        value = getattr(exc, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    return None


def extract_response_text(exc: Exception, limit: int = 500) -> str:
    body = ""
    response = getattr(exc, "response", None)
    if response is not None:
        body = getattr(response, "text", "") or ""

    if not body:
        message = getattr(exc, "message", None)
        if message:
            body = str(message)

    if not body:
        details = getattr(exc, "details", None)
        if details:
            try:
                body = json.dumps(details, ensure_ascii=False)
            except Exception:
                body = str(details)

    if not body:
        body = str(exc)

    if len(body) <= limit:
        return body
    return body[: limit - 3] + "..."
