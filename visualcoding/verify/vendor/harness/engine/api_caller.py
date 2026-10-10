"""
API caller utilities for vision API inference.
"""

import copy
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from utils.gemini_sdk import (
    extract_finish_reason as extract_sdk_finish_reason,
    extract_response_text as extract_sdk_response_text,
    extract_status_code as extract_sdk_status_code,
    gemini_generate_content,
    get_gemini_client,
    normalize_sdk_usage,
    rest_to_sdk_value,
    sdk_args_to_dict,
    sdk_parts_to_rest_list,
)

# API configuration constants
API_TIMEOUT = 240  # API request timeout (seconds)
BASE_WAIT_TIME = 1.0  # Base wait time (seconds)
MAX_WAIT_TIME = 15.0  # Maximum wait time (seconds)
ROUND_ROBIN_429_WAIT_MIN = 5.0  # Round-Robin mode 429 error minimum wait time
ROUND_ROBIN_429_WAIT_MAX = 10.0  # Round-Robin mode 429 error maximum wait time
HIGH_TOKEN_THRESHOLD = 25000  # High token number warning threshold
MAX_NON_429_RETRIES = 3  # Non-rate-limit failures should fail fast

# Gemini 3.x (esp. Pro, which always thinks) occasionally returns finish_reason=STOP
# with an EMPTY text part and no function call. Re-sending the identical request
# tends to reproduce the empty reply, so from the 2nd attempt on we append this
# user nudge; the nudge is NOT written into the stored conversation history.
EMPTY_CONTENT_NUDGE = (
    "Your previous reply came back empty. Continue the task now: either make a "
    "tool call, or give your final answer wrapped in <answer></answer> tags."
)


def _with_empty_nudge(messages: list) -> list:
    """Return a shallow copy of ``messages`` with the empty-content nudge appended."""
    return list(messages) + [{"role": "user", "content": EMPTY_CONTENT_NUDGE}]

# HTTP status codes treated as transient rate-limit / overload signals and
# retried indefinitely (same handling as 429). 503 UNAVAILABLE ("model is
# currently experiencing high demand") is a transient overload like 429.
RATE_LIMIT_STATUS_CODES = (429, 503)


def _is_rate_limit_status(status_code) -> bool:
    """True for transient rate-limit / overload HTTP codes (retry forever)."""
    return status_code in RATE_LIMIT_STATUS_CODES


def _is_gemini_model(model_name: str = "") -> bool:
    """Return True when the selected model belongs to the Gemini family."""
    return "gemini" in (model_name or "").lower()


def _should_use_gemini_sdk(endpoint: str, model_name: str = "") -> bool:
    """
    Use the Gemini Python SDK only when the model is Gemini and no API URL is provided.
    """
    return _is_gemini_model(model_name) and not (endpoint or "").strip()


def _normalize_chat_completions_endpoint(endpoint: str) -> str:
    """
    Normalize an OpenAI-compatible chat endpoint.

    Accepts either a base URL like ``https://host/v1`` or a full endpoint like
    ``https://host/v1/chat/completions`` and always returns the latter.
    """
    endpoint = (endpoint or "").rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return f"{endpoint}/chat/completions"


def _safe_json_loads(payload: Any) -> Any:
    """Best-effort JSON parsing."""
    if isinstance(payload, (dict, list)):
        return payload
    if not isinstance(payload, str):
        return payload
    try:
        return json.loads(payload)
    except Exception:
        return payload


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from OpenAI-style content blocks."""
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    text_parts: List[str] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                text_parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
    return "\n".join(text_parts).strip()


def _data_url_to_gemini_part(data_url: str) -> Optional[Dict[str, Any]]:
    """Convert a data URL into Gemini inlineData part."""
    match = re.match(r"^data:([^;]+);base64,(.+)$", data_url, re.DOTALL)
    if not match:
        return None
    mime_type, raw_data = match.groups()
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": raw_data,
        }
    }


def _openai_content_to_gemini_parts(content: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI-style content into Gemini parts."""
    if isinstance(content, str):
        return [{"text": content}] if content.strip() else []

    if not isinstance(content, list):
        return []

    parts: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                parts.append({"text": item})
            continue

        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append({"text": text})
        elif item_type == "image_url":
            image_url = item.get("image_url") or {}
            url = image_url.get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            gemini_part = _data_url_to_gemini_part(url)
            if gemini_part:
                parts.append(gemini_part)
            else:
                # Native Gemini payloads do not directly accept arbitrary remote URLs,
                # so keep the reference as text instead of dropping it silently.
                parts.append({"text": f"[Image URL] {url}"})
    return parts


def _openai_tools_to_gemini_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI function-calling schema into Gemini functionDeclarations."""
    if not tools:
        return None

    declarations: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function_info = tool.get("function") or {}
        name = function_info.get("name")
        if not name:
            continue
        declaration: Dict[str, Any] = {"name": name}
        description = function_info.get("description")
        parameters = function_info.get("parameters")
        if description:
            declaration["description"] = description
        if parameters:
            declaration["parameters"] = parameters
        declarations.append(declaration)

    if not declarations:
        return None

    return [{"functionDeclarations": declarations}]


def _build_gemini_payload(model_name: str, messages: list, sampling_params: dict, tools: list = None) -> dict:
    """Build Gemini native ``generateContent`` payload."""
    system_chunks: List[str] = []
    contents: List[Dict[str, Any]] = []
    tool_name_by_id: Dict[str, str] = {}

    for message in messages:
        role = (message.get("role") or "").lower()
        content = message.get("content")

        if role == "system":
            system_text = _extract_text_from_content(content)
            if system_text:
                system_chunks.append(system_text)
            continue

        if role == "user":
            parts = _openai_content_to_gemini_parts(content)
            if parts:
                contents.append({"role": "user", "parts": parts})
            continue

        if role == "assistant":
            raw_parts = message.get("gemini_parts")
            parts: List[Dict[str, Any]] = []

            if isinstance(raw_parts, list) and raw_parts:
                # Reuse Gemini-native parts verbatim so tool-call metadata such as
                # thoughtSignature survives the next round-trip intact.
                parts = copy.deepcopy(raw_parts)
            else:
                parts = _openai_content_to_gemini_parts(content)
                for tool_call in message.get("tool_calls") or []:
                    function_info = tool_call.get("function") or {}
                    function_name = function_info.get("name")
                    if not function_name:
                        continue
                    arguments = _safe_json_loads(function_info.get("arguments", {}))
                    if not isinstance(arguments, dict):
                        arguments = {"raw_arguments": arguments}

                    function_call_part = {
                        "functionCall": {
                            "name": function_name,
                            "args": arguments,
                        }
                    }
                    thought_signature = tool_call.get("thought_signature")
                    if thought_signature:
                        function_call_part["functionCall"]["thoughtSignature"] = thought_signature
                    parts.append(function_call_part)

            for tool_call in message.get("tool_calls") or []:
                function_info = tool_call.get("function") or {}
                function_name = function_info.get("name")
                tool_call_id = tool_call.get("id")
                if function_name and tool_call_id:
                    tool_name_by_id[tool_call_id] = function_name
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            tool_name = tool_name_by_id.get(tool_call_id or "")
            tool_payload = _safe_json_loads(content)
            if not isinstance(tool_payload, dict):
                tool_payload = {"result": str(content)}

            if tool_name:
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_name,
                            "response": tool_payload,
                        }
                    }],
                })
            else:
                text_fallback = _extract_text_from_content(content) if isinstance(content, list) else str(content)
                if text_fallback.strip():
                    contents.append({"role": "user", "parts": [{"text": text_fallback}]})

    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Please continue."}]}]

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": sampling_params["temperature"],
            "topP": sampling_params["top_p"],
            "maxOutputTokens": sampling_params["max_tokens"],
        }
    }

    if system_chunks:
        payload["system_instruction"] = {
            "parts": [{"text": "\n\n".join(system_chunks)}]
        }

    gemini_tools = _openai_tools_to_gemini_tools(tools)
    if gemini_tools:
        payload["tools"] = gemini_tools
        payload["toolConfig"] = {
            "functionCallingConfig": {
                "mode": "AUTO"
            }
        }

    return payload


def _sanitize_openai_messages(messages: list) -> list:
    """
    Drop any assistant tool_calls that are not followed by a matching `tool`
    response message. OpenAI rejects the whole request with
    "An assistant message with 'tool_calls' must be followed by tool messages
    responding to each 'tool_call_id'..." if any orphan exists. This walks the
    history once and keeps only tool_calls whose ids show up in a subsequent
    `tool` message before the next assistant turn.
    """
    if not messages:
        return messages

    sanitized = []
    dropped = 0
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue

        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            sanitized.append(msg)
            continue

        answered_ids = set()
        for j in range(i + 1, len(messages)):
            nxt = messages[j]
            if not isinstance(nxt, dict):
                continue
            role = nxt.get("role")
            if role == "tool":
                tcid = nxt.get("tool_call_id")
                if tcid:
                    answered_ids.add(tcid)
            elif role == "assistant":
                break

        kept = [tc for tc in msg["tool_calls"] if tc.get("id") in answered_ids]
        if len(kept) == len(msg["tool_calls"]):
            sanitized.append(msg)
            continue

        dropped += len(msg["tool_calls"]) - len(kept)
        new_msg = dict(msg)
        if kept:
            new_msg["tool_calls"] = kept
        else:
            new_msg.pop("tool_calls", None)
            if not new_msg.get("content"):
                new_msg["content"] = "[Orphan tool call(s) removed before resend]"
        sanitized.append(new_msg)

    if dropped:
        print(f"[Sanitize] Removed {dropped} orphan tool_call(s) without follow-up tool messages")
    return sanitized


def _build_payload(model_name: str, messages: list, sampling_params: dict, tools: list = None, end_point: str = None):
    """
    Build API request payload for either OpenAI-compatible or Gemini SDK calls.
    """
    if _should_use_gemini_sdk(end_point, model_name):
        return _build_gemini_payload(model_name, messages, sampling_params, tools)

    messages = _sanitize_openai_messages(messages)

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
        "max_tokens": sampling_params["max_tokens"],
    }

    if tools:
        payload["tools"] = tools
        if "o4-mini" not in model_name:
            payload["parallel_tool_calls"] = False

    # Endpoint-specific request fields (e.g. DeepSeek `{"thinking": {"type": "disabled"}}`), merged from a JSON env var.
    extra = os.environ.get("REASONING_EXTRA_BODY")
    if extra:
        try:
            payload.update(json.loads(extra))
        except json.JSONDecodeError:
            print(f"[API] REASONING_EXTRA_BODY is not valid JSON - ignored: {extra[:80]}")
        payload = {k: v for k, v in payload.items() if v is not None}   # a null in REASONING_EXTRA_BODY removes a default field (e.g. max_tokens -> max_completion_tokens)

    return payload


def _add_reasoning_param(payload: dict, model_name: str, end_point: str = None):
    """
    Add reasoning parameters for OpenAI-compatible requests.
    """
    model_lower = model_name.lower()
    is_gpt_model = "gpt-5" in model_lower or "o1" in model_lower or "o3" in model_lower

    is_openrouter = False
    if end_point:
        is_openrouter = "openrouter" in end_point.lower()

    if "gemini" in model_lower:
        if is_openrouter:
            reasoning_max_tokens = os.environ.get("REASONING_MAX_TOKENS")
            if reasoning_max_tokens and reasoning_max_tokens.lower() not in ["none", "false", ""]:
                payload["reasoning"] = {
                    "max_tokens": int(reasoning_max_tokens),
                    "exclude": False
                }
        else:
            reasoning_effort = os.environ.get("REASONING_EFFORT")
            if reasoning_effort and reasoning_effort.lower() not in ["none", "false", ""]:
                payload["reasoning"] = {
                    "effort": reasoning_effort,
                    "exclude": False
                }
    elif is_gpt_model:
        reasoning_effort = os.environ.get("REASONING_EFFORT")
        if reasoning_effort and reasoning_effort.lower() not in ["none", "false", ""]:
            payload["reasoning_effort"] = reasoning_effort


def _make_api_request(end_point: str, headers: dict, payload: dict, api_name: str = "API"):
    """
    Send API request and return response.

    Returns:
        tuple: (response, error_type)
               response: requests.Response object or None
               error_type: "429", "timeout", "network", "other", None(Success)
    """
    try:
        response = requests.post(end_point, headers=headers, json=payload, timeout=API_TIMEOUT)
        if _is_rate_limit_status(response.status_code):
            return response, "429"
        return response, None
    except requests.exceptions.Timeout:
        print(f"[{api_name}] API timeout")
        return None, "timeout"
    except requests.exceptions.RequestException as e:
        print(f"[{api_name}] API call failed: {e}")
        return None, "network"
    except Exception as e:
        print(f"[{api_name}] Unexpected error: {e}")
        return None, "other"


def _parse_openai_response(response, api_name: str, attempt: int = 0):
    """
    Parse an OpenAI-compatible API response.

    Returns:
        tuple: (result, is_429, error_type)
    """
    if response.status_code != 200:
        try:
            error_info = response.json()
            print(f"[{api_name}] API Error {response.status_code} on attempt {attempt}: {error_info}")
        except Exception:
            print(f"[{api_name}] API Error {response.status_code} on attempt {attempt}: {response.text}")

        is_429 = _is_rate_limit_status(response.status_code)
        return None, is_429, "http_error"

    try:
        result = response.json()
    except json.JSONDecodeError as e:
        print(f"[{api_name}] Failed to parse JSON response: {e}")
        return None, False, "invalid_format"

    if "choices" in result and result["choices"]:
        first_choice = result["choices"][0]
        if first_choice is None:
            print(f"[{api_name}] API returned None in choices[0] on attempt {attempt}")
            return None, False, "invalid_format"

        if not isinstance(first_choice, dict) or "message" not in first_choice:
            print(f"[{api_name}] API response missing 'message' field in choices[0] on attempt {attempt}")
            print(f"[{api_name}] choices[0] content: {first_choice}")
            return None, False, "invalid_format"

        message = first_choice["message"]
        if message is None or not isinstance(message, dict):
            print(f"[{api_name}] API returned invalid message type on attempt {attempt}: {type(message)}")
            return None, False, "invalid_format"

        _u = result.get("usage") or {}
        if isinstance(_u, dict):        # cost accounting (OpenAI-style endpoints); DeepSeek adds cache hit / miss + reasoning counts
            _d = _u.get("completion_tokens_details") or {}
            message["usage"] = {"input_tokens": int(_u.get("prompt_tokens") or 0), "output_tokens": int(_u.get("completion_tokens") or 0),
                                "thinking_tokens": int((_d.get("reasoning_tokens") if isinstance(_d, dict) else 0) or 0),
                                "cache_hit_tokens": int(_u.get("prompt_cache_hit_tokens") or ((_u.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(_u.get("prompt_tokens_details"), dict) else 0) or 0)}
        if "tool_calls" in message and message["tool_calls"]:
            return message, False, None

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return message, False, None          # always the message dict (content + usage); verify.loop reads .get("content")

        finish_reason = first_choice.get("finish_reason")
        usage = result.get("usage", {}) or {}
        print(f"[{api_name}] API returned empty content on attempt {attempt}: "
              f"finish_reason={finish_reason}, prompt_tokens={usage.get('prompt_tokens')}, "
              f"completion_tokens={usage.get('completion_tokens')}")
        if finish_reason == "length":
            print(f"[{api_name}] finish_reason=length with empty text — likely the whole "
                  f"completion budget went to reasoning; consider raising max_tokens")
        elif finish_reason in ("content_filter", "safety"):
            print(f"[{api_name}] blocked by content filter — not retryable client-side")
        try:
            print(f"[{api_name}] Empty message detail: {json.dumps(message, ensure_ascii=False)}")
        except Exception:
            print(f"[{api_name}] Empty message detail: {message}")
        return None, False, "empty_content"

    if "choices" in result and len(result["choices"]) == 0:
        print(f"[{api_name}] Model refused to generate (empty choices) on attempt {attempt}")
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        print(f"[{api_name}] Usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
        if "error" in result:
            print(f"[{api_name}] Error details: {result['error']}")
        if prompt_tokens > HIGH_TOKEN_THRESHOLD:
            print(f"[{api_name}] Warning: Very high prompt_tokens ({prompt_tokens}), may have hit token limit")
        print(f"[{api_name}] Possible causes: content filtering, safety policy, token limit, or model overload")
        return None, False, "empty_choices"

    print(f"[{api_name}] Invalid API response format on attempt {attempt}: {result}")
    return None, False, "invalid_format"


def _parse_gemini_sdk_response(response, api_name: str, attempt: int = 0):
    """
    Parse a ``google-genai`` SDK response and map it to the existing
    OpenAI-compatible structure expected by the rest of the pipeline.
    """
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        print(f"[{api_name}] Gemini SDK returned no candidates on attempt {attempt}: {prompt_feedback}")
        return None, False, "empty_choices"

    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for idx, part in enumerate(parts):
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            text_parts.append(text)

        function_call = getattr(part, "function_call", None)
        function_name = getattr(function_call, "name", None) if function_call is not None else None
        if function_name:
            arguments = sdk_args_to_dict(getattr(function_call, "args", None))
            try:
                serialized_arguments = json.dumps(arguments, ensure_ascii=False)
            except TypeError:
                serialized_arguments = json.dumps(
                    {"raw_arguments": str(arguments)},
                    ensure_ascii=False,
                )
            thought_signature = getattr(part, "thought_signature", None)
            if isinstance(thought_signature, (bytes, bytearray)):
                import base64 as _b64
                thought_signature = _b64.b64encode(bytes(thought_signature)).decode("ascii")
            tool_calls.append({
                "id": f"call_{idx}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": serialized_arguments,
                },
                "thought_signature": thought_signature,
            })

    content_text = "\n".join(text_parts).strip()
    gemini_parts = sdk_parts_to_rest_list(parts)
    # cost accounting: prompt / candidate / thought token counts of this call
    try:
        usage = normalize_sdk_usage(response)
        _um = getattr(response, "usage_metadata", None)
        _tt = (_um.get("thoughts_token_count") if isinstance(_um, dict) else getattr(_um, "thoughts_token_count", None)) if _um is not None else None
        usage["thinking_tokens"] = int(_tt or 0)
    except Exception:  # noqa: BLE001
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "thinking_tokens": 0, "calls": 0}
    if tool_calls:
        return {
            "content": content_text,
            "tool_calls": tool_calls,
            "gemini_parts": gemini_parts,
            "usage": usage,
        }, False, None

    if content_text:
        return {
            "content": content_text,
            "gemini_parts": gemini_parts,
            "usage": usage,
        }, False, None

    finish_reason = extract_sdk_finish_reason(response)
    print(f"[{api_name}] Gemini SDK returned empty content on attempt {attempt}: finish_reason={finish_reason}, candidate={candidate}")
    return None, False, "empty_content"


def _try_single_attempt(
    api_key: str,
    end_point: str,
    model_name: str,
    messages: list,
    sampling_params: dict,
    api_name: str = "API",
    tools: list = None,
    attempt: int = 1,
    nudge_empty: bool = False,
):
    """
    Try a single API attempt. The outer caller is responsible for infinite retry.

    ``nudge_empty``: the previous attempt returned empty content — append the
    EMPTY_CONTENT_NUDGE user turn to this request (request-local only).

    Returns:
        Tuple of (result, is_429, error_info)
    """
    use_gemini_sdk = _should_use_gemini_sdk(end_point, model_name)
    if nudge_empty:
        print(f"[{api_name}] previous attempt returned empty content — retrying with a continue-nudge")
        messages = _with_empty_nudge(messages)

    if use_gemini_sdk:
        normalized_endpoint = None
    else:
        normalized_endpoint = _normalize_chat_completions_endpoint(end_point)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    payload = _build_payload(model_name, messages, sampling_params, tools, end_point=end_point)
    if not use_gemini_sdk:
        _add_reasoning_param(payload, model_name, normalized_endpoint)

    response = None
    request_error = None
    if use_gemini_sdk:
        try:
            client = get_gemini_client(api_key)
            response = gemini_generate_content(
                client,
                model_name=model_name,
                contents=rest_to_sdk_value(payload.get("contents")),
                system_instruction=rest_to_sdk_value(payload.get("system_instruction")),
                tools=rest_to_sdk_value(payload.get("tools")),
                tool_config=rest_to_sdk_value(payload.get("toolConfig")),
                generation_config=rest_to_sdk_value(payload.get("generationConfig")),
                timeout=API_TIMEOUT,
            )
        except Exception as exc:
            status_code = extract_sdk_status_code(exc)
            if _is_rate_limit_status(status_code):
                wait_time = ROUND_ROBIN_429_WAIT_MIN + random.uniform(
                    0,
                    ROUND_ROBIN_429_WAIT_MAX - ROUND_ROBIN_429_WAIT_MIN,
                )
                print(f"[{api_name}] Gemini SDK rate limit / overload hit ({status_code}), waiting {wait_time:.2f}s before next attempt")
                time.sleep(wait_time)
                return None, True, f"{api_name}: HTTP 429 (rate limit)"
            error_msg = f"{api_name}: sdk_error"
            print(f"[{api_name}] Gemini SDK Error on attempt {attempt}: {exc}")
            body = extract_sdk_response_text(exc)
            if body:
                print(f"[{api_name}] Gemini SDK Error detail: {body}")
            return None, False, error_msg
    else:
        response, request_error = _make_api_request(normalized_endpoint, headers, payload, api_name)

        if request_error == "429" or (response and _is_rate_limit_status(response.status_code)):
            body_text = ""
            if response is not None:
                try:
                    body_text = response.text or ""
                except Exception as body_exc:
                    body_text = f"<failed to read body: {body_exc}>"
                print(f"[{api_name}] 429 body: {body_text[:1000]}")

            # Some proxies wrap non-rate-limit failures (invalid_request_error,
            # context_length_exceeded, insufficient credit, etc.) as HTTP 429.
            # Retrying these is pointless — surface them as non-retriable so the
            # round-robin loop counts toward `max_non_429_retries` and fails fast.
            non_retriable_markers = (
                "invalid_request_error",
                "context_length_exceeded",
                "context length",
                "insufficient_balance",
                "insufficient_quota",
                "billing_hard_limit_reached",
                "model_not_found",
            )
            lowered = body_text.lower()
            for marker in non_retriable_markers:
                if marker in lowered:
                    error_msg = f"{api_name}: HTTP 429 (non-retriable: {marker})"
                    print(f"[{api_name}] Treating 429 as non-retriable ({marker}); will not retry forever")
                    return None, False, error_msg

            wait_time = ROUND_ROBIN_429_WAIT_MIN + random.uniform(
                0,
                ROUND_ROBIN_429_WAIT_MAX - ROUND_ROBIN_429_WAIT_MIN,
            )
            print(f"[{api_name}] API rate limit hit (429), waiting {wait_time:.2f} seconds before next attempt")
            time.sleep(wait_time)
            return None, True, f"{api_name}: HTTP 429 (rate limit)"

        if request_error:
            error_msg = f"{api_name}: {request_error}"
            print(f"[{api_name}] Request failed: {request_error}")
            return None, False, error_msg

    if use_gemini_sdk:
        result, is_429, error_type = _parse_gemini_sdk_response(response, api_name, attempt=attempt)
    else:
        result, is_429, error_type = _parse_openai_response(response, api_name, attempt=attempt)

    error_info = None
    if result is None:
        if error_type == "http_error":
            try:
                error_detail = response.json() if response else {}
                error_info = f"{api_name}: HTTP {response.status_code} - {error_detail}"
            except Exception:
                error_info = f"{api_name}: HTTP {response.status_code}"
        elif error_type:
            error_info = f"{api_name}: {error_type}"

    return result, is_429, error_info


def call_vision_api(model_name: str, messages: list, sampling_params: dict, max_retries: int = None, tools: list = None):
    """
    Call the reasoning API.

    Rate-limit / overload failures (HTTP 429 and 503) keep retrying indefinitely.
    All other failures are capped at ``max_retries`` attempts, defaulting to 3.
    """
    max_non_429_retries = max_retries if max_retries is not None else MAX_NON_429_RETRIES
    api_key_1 = os.environ.get("REASONING_API_KEY")
    end_point_1 = os.environ.get("REASONING_END_POINT")

    primary_uses_gemini_sdk = _should_use_gemini_sdk(end_point_1, model_name)
    if not api_key_1 or (not end_point_1 and not primary_uses_gemini_sdk):
        raise ValueError(
            "REASONING_API_KEY must be set, and REASONING_END_POINT is required unless using a Gemini model with an empty endpoint."
        )

    api_key_2 = os.environ.get("REASONING_API_KEY_2")
    end_point_2 = os.environ.get("REASONING_END_POINT_2")
    has_fallback = bool(api_key_2 and (end_point_2 or _should_use_gemini_sdk(end_point_2, model_name)))

    if has_fallback:
        print(
            "[API Round-Robin] Starting round-robin polling: "
            "HTTP 429 retries are unbounded, other failures are capped"
        )
        consecutive_429_count = 0
        non_429_round_count = 0
        round_idx = 0
        nudge_empty = False

        while True:
            round_idx += 1

            result_1, is_429_1, error_1 = _try_single_attempt(
                api_key_1,
                end_point_1,
                model_name,
                messages,
                sampling_params,
                api_name="Primary API",
                tools=tools,
                attempt=round_idx,
                nudge_empty=nudge_empty,
            )
            if result_1 is not None:
                return result_1
            nudge_empty = bool(error_1 and str(error_1).endswith("empty_content"))

            result_2, is_429_2, error_2 = _try_single_attempt(
                api_key_2,
                end_point_2,
                model_name,
                messages,
                sampling_params,
                api_name="Fallback API",
                tools=tools,
                attempt=round_idx,
                nudge_empty=nudge_empty,
            )
            if result_2 is not None:
                return result_2
            nudge_empty = nudge_empty or bool(error_2 and str(error_2).endswith("empty_content"))

            if error_1:
                print(f"[API Round-Robin] Round {round_idx} primary failed: {error_1}")
            if error_2:
                print(f"[API Round-Robin] Round {round_idx} fallback failed: {error_2}")

            if is_429_1 and is_429_2:
                consecutive_429_count += 1
                extra_wait = min(
                    ROUND_ROBIN_429_WAIT_MIN + consecutive_429_count * 5,
                    60.0,
                )
                print(f"[API Round-Robin] Both APIs are rate-limited, waiting {extra_wait:.2f}s before retrying")
                time.sleep(extra_wait)
            else:
                consecutive_429_count = 0
                non_429_round_count += 1
                if non_429_round_count >= max_non_429_retries:
                    last_errors = [err for err in [error_1, error_2] if err]
                    last_error_text = " | ".join(last_errors) if last_errors else "unknown_error"
                    return (
                        f"Error: Non-rate-limit API attempts failed after "
                        f"{max_non_429_retries} tries. Last errors: {last_error_text}"
                    )
                time.sleep(BASE_WAIT_TIME)

    print(
        "[API] No fallback API configured: "
        "HTTP 429 retries are unbounded, other failures are capped"
    )
    attempt = 0
    non_429_attempt_count = 0
    nudge_empty = False
    while True:
        attempt += 1
        result, is_429, error = _try_single_attempt(
            api_key_1,
            end_point_1,
            model_name,
            messages,
            sampling_params,
            api_name="Primary API",
            tools=tools,
            attempt=attempt,
            nudge_empty=nudge_empty,
        )
        if result is not None:
            return result
        nudge_empty = bool(error and str(error).endswith("empty_content"))

        if error:
            print(f"[API] Primary API attempt {attempt} failed: {error}")

        if not is_429:
            non_429_attempt_count += 1
            if non_429_attempt_count >= max_non_429_retries:
                return (
                    f"Error: Non-rate-limit API attempts failed after "
                    f"{max_non_429_retries} tries. Last error: {error or 'unknown_error'}"
                )

        wait_time = min(BASE_WAIT_TIME * (2 ** min(attempt - 1, 4)), MAX_WAIT_TIME) if is_429 else BASE_WAIT_TIME
        time.sleep(wait_time)
