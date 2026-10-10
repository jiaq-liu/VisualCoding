# Vendored agent-harness modules

Copies of the model-call, image-budget and LLM-judge utilities of the agent harness that produced the executor
baselines (the same prompts, sampling settings and judge as the no-tool / code-interpreter agent our executors run in).
They keep their original top-level package names (`engine`, `utils`, `prompts`), so `verify/_paths.py:enable_vendor()`
puts this directory at the front of `sys.path` before import.  The `__init__.py` files are empty stubs.

| file | used for |
|---|---|
| engine/api_caller.py | `call_vision_api`: Gemini SDK call (OpenAI-style messages -> generateContent, thought-signature preservation, 429/503 retry) and the OpenAI-compatible HTTP path (`REASONING_END_POINT`, `REASONING_EXTRA_BODY`) used for DeepSeek / relay executors; usage accounting |
| utils/gemini_sdk.py | google-genai wrapper; thinking config (3.x pro -> `thinking_level=low`, flash-lite -> thinking off), safety off |
| utils/context_utils.py | `process_image` (pixel budget), `pil_to_base64_data_uri`, `estimate_tokens` |
| utils/image_resize.py | `fetch_image` (smart_resize to 28-pixel multiples) for the input image |
| utils/llm_judger.py | `compute_score` / `evaluate_answer`: the LLM-as-judge of every executor baseline |
| prompts/llm_as_judge_prompts.py | judge prompts |

Local changes: `api_caller.py` merges `REASONING_EXTRA_BODY` (JSON) into the request payload, drops `null` fields, and records
`prompt_tokens_details.cached_tokens` / reasoning tokens in `usage`.  Env vars these modules read are exported by `verify/env.py`.
