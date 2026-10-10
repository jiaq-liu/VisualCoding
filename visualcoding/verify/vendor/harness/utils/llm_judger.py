"""
LLM-as-Judge evaluator for assessing answer correctness.

This module provides an LLM-based evaluation system that compares predicted
answers against ground truth using configurable prompts.
"""

import re
import os
import time
from typing import Any, Tuple, List, Optional, Dict

import openai

from prompts.llm_as_judge_prompts import SYSTEM_PROMPT, QUERY_PROMPT
from utils.gemini_sdk import (
    extract_block_reason as extract_sdk_block_reason,
    extract_finish_reason as extract_sdk_finish_reason,
    extract_response_text as extract_sdk_response_text,
    extract_status_code as extract_sdk_status_code,
    extract_text_from_sdk_response,
    gemini_generate_content,
    get_gemini_client,
)


# Hard cap on judge retries. The judge is called per-sample; without a cap a
# silently rate-limited / safety-blocked request can spin forever (see the
# unbounded loop that previously sat in ``evaluate``).
DEFAULT_MAX_RETRIES = 8


# Global client instance (lazily initialized)
_client = None
def _is_gemini_model(model_name: str = "") -> bool:
    """Return True when the selected model belongs to the Gemini family."""
    return "gemini" in (model_name or "").lower()


def _should_use_gemini_sdk(endpoint: str, model_name: str = "") -> bool:
    """
    Use the Gemini Python SDK only when the model is Gemini and no endpoint is provided.
    """
    return _is_gemini_model(model_name) and not (endpoint or "").strip()


def _normalize_openai_base_url(endpoint: str) -> str:
    """Normalize an OpenAI-compatible base URL for the OpenAI SDK client."""
    endpoint = (endpoint or "").rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint[: -len("/chat/completions")]
    return endpoint


def _get_client():
    """Get or create the global LLM judge client instance."""
    global _client
    if _client is None:
        _client = LLMJudgeClient()
    return _client


class LLMJudgeClient:
    """Client for LLM-based answer evaluation using OpenAI-compatible or Gemini-native APIs."""

    def __init__(self):
        """Initialize the LLM judge client with environment variables."""
        api_key = os.environ.get("VERIFIER_API_KEY")
        endpoint = os.environ.get("VERIFIER_END_POINT")
        self.model_name = os.environ.get("VERIFIER_MODEL_NAME", "gpt-4o-2024-11-20")
        self.use_gemini_sdk = _should_use_gemini_sdk(endpoint, self.model_name)
        self.is_gemini_native_api = self.use_gemini_sdk
        self.api_key = api_key
        self.endpoint = endpoint

        if not api_key or (not endpoint and not self.use_gemini_sdk):
            raise ValueError(
                "VERIFIER_API_KEY must be set, and VERIFIER_END_POINT is required unless using a Gemini model with an empty endpoint."
            )

        self.client = None
        if not self.is_gemini_native_api:
            self.client = openai.OpenAI(
                base_url=_normalize_openai_base_url(endpoint),
                api_key=api_key,
            )

    def _call_gemini_sdk(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 32768,
    ) -> str:
        """Call Gemini native via the ``google-genai`` SDK for judge inference."""
        client = get_gemini_client(self.api_key)
        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        response = gemini_generate_content(
            client,
            model_name=self.model_name,
            contents=[prompt],
            system_instruction=system_prompt,
            tool_config={"function_calling_config": {"mode": "NONE"}},
            generation_config=generation_config,
            timeout=120,
            disable_thinking=True,
        )
        response_text = extract_text_from_sdk_response(response)
        if not response_text:
            finish_reason = extract_sdk_finish_reason(response)
            block_reason = extract_sdk_block_reason(response)
            details = []
            if finish_reason:
                details.append(f"finish_reason={finish_reason}")
            if block_reason:
                details.append(f"block_reason={block_reason}")
            suffix = f" ({', '.join(details)})" if details else ""
            raise ValueError(f"Invalid response: empty content{suffix}")
        return response_text

    def evaluate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES
    ) -> Tuple[str, str]:
        """
        Evaluate an answer using the LLM judge.

        Retries on transient failures up to ``max_retries`` attempts. When the
        cap is reached the most recent exception is re-raised so the caller can
        fall back (typically scoring the sample as 0).
        """
        if max_retries < 1:
            max_retries = 1

        messages = []

        if system_prompt is not None:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        })

        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            try:
                if self.is_gemini_native_api:
                    response_text = self._call_gemini_sdk(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=min(0.2 * (attempt - 1), 1.0),
                        max_tokens=32768,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=min(0.2 * (attempt - 1), 1.0),
                        max_tokens=8192,
                        timeout=120,
                    )

                    if not response or not hasattr(response, 'choices') or not response.choices:
                        raise ValueError("Invalid response: missing choices field")

                    if len(response.choices) == 0:
                        raise ValueError("Invalid response: empty choices array")

                    first_choice = response.choices[0]
                    if not first_choice or not hasattr(first_choice, 'message'):
                        raise ValueError("Invalid response: missing message in choices[0]")

                    message = first_choice.message
                    if not message or not hasattr(message, 'content'):
                        raise ValueError("Invalid response: missing content in message")

                    response_text = message.content
                    if not response_text:
                        raise ValueError("Invalid response: empty content")

                if 'score:' not in response_text.lower():
                    raise ValueError("No 'score:' found in response")

                score_str = (
                    response_text.lower()
                    .split("score:")[-1]
                    .strip()
                    .split("\n")[0]
                    .strip()
                    .split(" ")[0]
                )

                if "1" not in score_str and '0' not in score_str:
                    raise ValueError(f"No valid score ('0' or '1') found: {score_str}")

                return score_str, response_text

            except openai.RateLimitError as e:
                last_exc = e
                print(f"[RateLimitError] Attempt {attempt}/{max_retries}: {str(e)}")
                if attempt >= max_retries:
                    break
                time.sleep(min(3 * attempt, 10))
                continue

            except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as e:
                last_exc = e
                error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
                print(f"[APIError] Attempt {attempt}/{max_retries}: {str(e)} (code: {error_code})")
                if attempt >= max_retries:
                    break
                delay = min(2 ** min(attempt - 1, 4), 10)
                time.sleep(delay)
                continue

            except Exception as e:
                last_exc = e
                if self.use_gemini_sdk:
                    status_code = extract_sdk_status_code(e)
                    detail = extract_sdk_response_text(e)
                    if status_code is not None:
                        print(f"[GeminiAPIStatus] Attempt {attempt}/{max_retries}: {status_code}")
                    if detail:
                        print(f"[GeminiAPIDetail] {detail}")
                print("=" * 100)
                print(f"[Error] Attempt {attempt}/{max_retries}: {str(e)}")
                print(f"Messages length: {len(str(messages))}")
                print("=" * 100)
                if attempt >= max_retries:
                    break
                delay = min(2 ** min(attempt - 1, 4), 10)
                time.sleep(delay)
                continue

        # Loop exhausted without a successful return. Re-raise the last error so
        # the caller (evaluate_answer) can fall back to scoring the sample 0.
        assert last_exc is not None
        raise last_exc


def _extract_answer_from_text(
    text: str,
    predict_str_list: List[str],
    extraction_mode: str
) -> Tuple[float, str]:
    """
    Extract answer from prediction text using specified extraction mode.

    Args:
        text: Full prediction text
        predict_str_list: List of prediction strings (for fallback)
        extraction_mode: Extraction mode ('split', 'strict', or 'strict_v2')

    Returns:
        Tuple of (reward, extracted_answer) where reward is 0.0 on failure
    """
    if extraction_mode == 'split':
        return 0.0, text.split("<answer>")[-1].split("</answer>")[0].strip()

    elif extraction_mode in ['strict', 'strict_v2']:
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return 0.0, matches[-1].strip()

        if predict_str_list:
            last_turn = predict_str_list[-1].strip()
            if last_turn.startswith("Error:") or "API attempts failed" in last_turn:
                return 0.0, ""

            return 0.0, last_turn

        return 0.0, ""

    else:
        raise ValueError(f"Unknown extraction mode: {extraction_mode}")


def evaluate_answer(
    question: str,
    prediction_list: List[str],
    ground_truth: str,
    extract_answer: bool = False,
    extra_info: Optional[Dict] = None
) -> Tuple[float, str]:
    """
    Evaluate a predicted answer against ground truth using LLM-as-judge.
    """
    prediction_text = ' '.join(prediction_list)

    if extract_answer and extra_info:
        extraction_mode = extra_info.get('extract_answer_tags', 'strict')
        reward, prediction_text = _extract_answer_from_text(
            prediction_text,
            prediction_list,
            extraction_mode
        )

        if reward == 0.0 and not prediction_text:
            return 0.0, "Answer extraction failed"

    eval_prompt = QUERY_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction_text
    )

    if len(eval_prompt) > 10000:
        print(f"[evaluate_answer] Warning: Very long prompt ({len(eval_prompt)} chars)")

    try:
        client = _get_client()
        score_str, response_text = client.evaluate(
            prompt=eval_prompt,
            system_prompt=SYSTEM_PROMPT
        )
    except Exception as e:
        print(f"[evaluate_answer] Error calling LLM judge: {e}")
        return 0.0, f"Evaluation API error: {str(e)}"

    if not score_str:
        return 0.0, "Evaluation failed: Empty response from API"

    try:
        reward = 1.0 if '1' in score_str else 0.0

        analysis = ""
        if response_text and "explanation:" in response_text.lower():
            explanation_part = response_text.lower().split("explanation:")[-1].strip()
            if explanation_part:
                analysis = explanation_part
        elif not response_text:
            analysis = "No explanation provided in response"

        return reward, analysis

    except Exception as e:
        print(f"[evaluate_answer] Error processing response: {e}")
        return 0.0, f"Error processing evaluation response: {str(e)}"


def _split_sheet_qualified_range(range_str: str, fallback_sheet: str = "") -> Tuple[str, str]:
    """
    Parse an Excel range that may carry a sheet qualifier.

    Accepts (155/400 SpreadsheetBench samples carry such qualifiers):
      "A1:B5"               -> ("", "A1:B5")
      "'Sheet1'!A1:B5"      -> ("Sheet1", "A1:B5")
      "Sheet1!A1:B5"        -> ("Sheet1", "A1:B5")
      "Analyse'!D5:K12"     -> ("Analyse", "D5:K12")  (malformed trailing quote)
      "A1"                  -> ("", "A1")             (single cell)

    When ``range_str`` carries a sheet name and ``fallback_sheet`` is empty,
    the parsed sheet name wins. When both are present, the parsed name wins
    (the qualifier is more specific than the dataset's separate field).
    """
    if not range_str:
        return fallback_sheet, ""
    if "!" not in range_str:
        return fallback_sheet, range_str.strip()
    sheet_part, _, cells_part = range_str.partition("!")
    sheet_part = sheet_part.strip().strip("'").strip('"').strip()
    cells_part = cells_part.strip()
    return (sheet_part or fallback_sheet), cells_part


def _cells_equal(a, b, float_tol: float = 1e-6) -> bool:
    """Compare two openpyxl cell values with float / numeric tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    # Numeric comparison with tolerance
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= float_tol or (fa != 0 and abs(fa - fb) / abs(fa) <= 1e-4)
    except (TypeError, ValueError):
        pass
    # String / fallback: strip whitespace, case-insensitive
    sa, sb = str(a).strip(), str(b).strip()
    return sa == sb or sa.casefold() == sb.casefold()


def _evaluate_spreadsheet_range(extra: Dict) -> Tuple[float, str]:
    """
    Compare a cell range in the agent's output xlsx against the golden xlsx.
    Score = matching_cells / total_cells_in_golden_range. Partial credit.

    Required keys in `extra`:
      agent_output_path : absolute path the agent saved its xlsx to
      golden_xlsx       : absolute path to the read-only golden file
      answer_position   : "A3:D32" or "A3" (single cell)
      answer_sheet      : sheet name (empty string => use active sheet of golden)

    Failure modes (return 0.0):
      - agent never saved an output file
      - agent_output is unreadable (corrupt xlsx)
      - sheet missing in agent output
      - range doesn't parse
    """
    try:
        import openpyxl
    except ImportError:
        return 0.0, "openpyxl not installed; cannot verify spreadsheet"

    agent_path = extra.get("agent_output_path", "")
    golden_path = extra.get("golden_xlsx", "")
    raw_rng = (extra.get("answer_position") or "").strip()
    raw_sheet = (extra.get("answer_sheet") or "").strip()

    if not agent_path or not os.path.exists(agent_path):
        return 0.0, f"agent_output_path missing: {agent_path}"
    if not golden_path or not os.path.exists(golden_path):
        return 0.0, f"golden_xlsx missing: {golden_path}"
    if not raw_rng:
        return 0.0, "answer_position empty"

    # 38.7% of samples carry a sheet qualifier inside answer_position
    # (e.g. "'WHP'!E7:G8"). Strip it and prefer the parsed sheet name.
    sheet, rng = _split_sheet_qualified_range(raw_rng, raw_sheet)

    try:
        # data_only=True returns evaluated formula values when the workbook
        # was last saved with cached values (which the dataset's golden
        # files all are). Agent-saved files often have only formula text,
        # so we also load with data_only=False and prefer evaluated values
        # but fall back to raw if unavailable.
        wb_a = openpyxl.load_workbook(agent_path, data_only=True, read_only=True)
        wb_g = openpyxl.load_workbook(golden_path, data_only=True, read_only=True)
    except Exception as e:
        return 0.0, f"failed to open xlsx: {e}"

    # Resolve sheet name. Empty name => first sheet (some samples don't
    # specify one). Non-empty name => MUST match exactly: this is how we
    # detect when the agent wrote to the wrong sheet.
    def _resolve_sheet(wb, name: str, strict: bool):
        if not name:
            return wb[wb.sheetnames[0]] if wb.sheetnames else None
        if name in wb.sheetnames:
            return wb[name]
        # Strict: missing required sheet => caller treats as error
        return None if strict else (wb[wb.sheetnames[0]] if wb.sheetnames else None)

    # Golden's sheet name is canonical; if missing, fall back to first.
    ws_g = _resolve_sheet(wb_g, sheet, strict=False)
    # Agent must hit the requested sheet exactly when one is specified.
    ws_a = _resolve_sheet(wb_a, sheet, strict=bool(sheet))
    if ws_g is None:
        return 0.0, "golden sheet not found"
    if ws_a is None:
        return 0.0, f"agent output missing requested sheet '{sheet}'"

    # Extract cells from the requested range. openpyxl's ws[range] returns a
    # tuple of tuples for multi-cell, or a Cell for a single cell.
    def _cells(ws, range_str):
        try:
            block = ws[range_str]
        except Exception:
            return None
        if hasattr(block, "value"):  # single cell
            return [block.value]
        out = []
        for row in block:
            if hasattr(row, "value"):  # single cell tuple
                out.append(row.value)
                continue
            for cell in row:
                out.append(cell.value)
        return out

    cells_g = _cells(ws_g, rng)
    cells_a = _cells(ws_a, rng)
    if cells_g is None:
        return 0.0, f"could not parse range '{rng}' in golden"
    if cells_a is None:
        return 0.0, f"could not parse range '{rng}' in agent output"

    if len(cells_g) == 0:
        return 0.0, "golden range is empty"
    # If shapes differ (rare), score the overlap.
    n = min(len(cells_a), len(cells_g))
    matches = sum(1 for a, b in zip(cells_a[:n], cells_g[:n]) if _cells_equal(a, b))
    score = matches / len(cells_g)
    return float(score), f"{matches}/{len(cells_g)} cells match"


# ============================================================================
# MCE text-task verifiers (deterministic; ported from mce-artifact/env/*).
#
# These mirror each MCE environment's exact comparison so this pipeline reproduces the
# same metric. Dispatched from compute_score by ``extra_info['verifier_type']``,
# which api_processors sets from each sample's ``data_source``.
#
# Answer contract (see scripts/convert_mce_datasets.py): the agent is asked to
# put its final answer between <answer> and </answer>. We extract that first,
# then fall back to each task's native marker, then to the raw text.
# ============================================================================

def _join_predictions(predict_str_list: List[str]) -> str:
    if isinstance(predict_str_list, str):
        return predict_str_list
    return "\n".join(p for p in (predict_str_list or []) if p)


def _extract_answer_tag(text: str) -> Optional[str]:
    """Return the content of the LAST <answer>...</answer> block, if any."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _eval_diagnosis(predict_str_list, ground_truth, extra) -> Tuple[float, str]:
    """symptom_diagnosis: normalized exact match against the disease label."""
    text = _join_predictions(predict_str_list)
    pred = _extract_answer_tag(text)
    if pred is None:
        m = re.search(r"\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]", text, re.IGNORECASE | re.DOTALL)
        pred = m.group(1).strip() if m else text.strip().split("\n")[-1]

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower().strip()).rstrip(".!?")

    correct = norm(pred) == norm(ground_truth)
    return (1.0 if correct else 0.0), f"pred={pred!r} gt={ground_truth!r}"


def _extract_crimes(text: str) -> set:
    """Port of CrimePredictionEnvironment._extract_crimes."""
    m = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", text)
    if m:
        crimes_str = m.group(1).strip()
    else:
        m = re.search(r"罪名[:：](.*?)(?:\n|$)", text)
        crimes_str = m.group(1).strip() if m else text.strip()
    crimes_str = re.sub(r"<eoa>.*", "", crimes_str).strip()
    crimes = re.split(r"[;；,，、]", crimes_str)
    return {c.strip() for c in crimes if c.strip()}


def _normalize_crime_set(crimes: set) -> set:
    """Normalize charge names so a semantically-correct answer matches the GT.

    The dataset's ground truth stores charge names WITHOUT the trailing 罪
    suffix (e.g. "破坏电力设备", not "破坏电力设备罪") and splits on the full set
    of delimiters incl. the Chinese enumeration comma 、. Models reliably emit
    the natural "...罪" form, which made every prediction mismatch on an
    otherwise-correct answer. We strip a single trailing 罪 from each token
    (both sides, so it is a no-op on GT) before set comparison.
    """
    out = set()
    for c in crimes:
        c = c.strip()
        if len(c) > 1 and c.endswith("罪"):
            c = c[:-1]
        if c:
            out.add(c)
    return out


def _eval_crime(predict_str_list, ground_truth, extra) -> Tuple[float, str]:
    """crime_prediction: set-equality of crime names (罪-suffix normalized)."""
    text = _join_predictions(predict_str_list)
    ans = _extract_answer_tag(text)
    pred_crimes = _normalize_crime_set(_extract_crimes(ans) if ans is not None else _extract_crimes(text))
    gt_crimes = _normalize_crime_set(_extract_crimes(ground_truth))
    correct = pred_crimes == gt_crimes
    return (1.0 if correct else 0.0), f"pred={sorted(pred_crimes)} gt={sorted(gt_crimes)}"


def _eval_finer(predict_str_list, ground_truth, extra) -> Tuple[float, str]:
    """finer: token-level accuracy (fractional). Port of _finer_answer_is_correct."""
    text = _join_predictions(predict_str_list)
    ans = _extract_answer_tag(text)
    predicted = ans if ans is not None else text.strip().split("\n")[-1]

    pred = [v.lower().strip() for v in predicted.split(",")]
    label = [v.lower().strip() for v in ground_truth.split(",")]
    if len(pred) != len(label):
        if len(pred) > len(label):
            pred = pred[: len(label)]
        else:
            pred += [""] * (len(label) - len(pred))

    count = 0
    for prediction, gt in zip(pred, label):
        try:
            gt_v = eval(gt)
            pred_v = eval(prediction.replace(",", "").replace("$", ""))
        except Exception:
            gt_v, pred_v = gt, prediction
        if gt_v == pred_v:
            count += 1
    score = count / len(pred) if pred else 0.0
    return float(score), f"{count}/{len(pred)} tokens match; pred={predicted!r} gt={ground_truth!r}"


def _eval_aegis(predict_str_list, ground_truth, extra) -> Tuple[float, str]:
    """aegis2: safe/unsafe binary. Port of _aegis2_answer_is_correct."""
    text = _join_predictions(predict_str_list)
    ans = _extract_answer_tag(text)
    raw = (ans if ans is not None else text).lower()
    if "safe" in raw and "unsafe" not in raw:
        pred = "safe"
    elif "unsafe" in raw:
        pred = "unsafe"
    else:
        pred = "unknown"
    correct = pred == ground_truth.lower()
    return (1.0 if correct else 0.0), f"pred={pred!r} gt={ground_truth!r}"


def _eval_uspto(predict_str_list, ground_truth, extra) -> Tuple[float, str]:
    """uspto: SMILES reactant set-equality (strip+lower, '.'-separated)."""
    text = _join_predictions(predict_str_list)
    ans = _extract_answer_tag(text)
    predicted = ans if ans is not None else text.strip().split("\n")[-1]

    def parse(s: str) -> set:
        return {r.strip().lower() for r in s.split(".") if r.strip()}

    pred_set, gt_set = parse(predicted), parse(ground_truth)
    correct = pred_set == gt_set
    return (1.0 if correct else 0.0), f"exact_match={correct}; n_pred={len(pred_set)} n_gt={len(gt_set)}"


# data_source / verifier_type -> verifier fn
_MCE_TEXT_VERIFIERS = {
    "mce_diagnosis": _eval_diagnosis,
    "mce_crime": _eval_crime,
    "mce_finer": _eval_finer,
    "mce_aegis": _eval_aegis,
    "mce_uspto": _eval_uspto,
}


def compute_score(
    prompt: str,
    predict_str_list: List[str],
    ground_truth: str,
    extra_info: Optional[Dict] = None
) -> Tuple[float, str]:
    """
    Compute accuracy score and return analysis text.

    Dispatches to a custom verifier when ``extra_info['verifier_type']`` is
    set; otherwise falls through to the LLM-as-judge path.
    """
    extra = extra_info or {}

    # ----- Custom verifier dispatch ---------------------------------
    verifier_type = extra.get("verifier_type")
    if verifier_type == "spreadsheet_range":
        try:
            score, analysis = _evaluate_spreadsheet_range(extra)
            acc_weight = extra.get("acc_reward_weight", 1.0)
            return float(acc_weight) * float(score), analysis
        except Exception as e:
            print(f"[compute_score] spreadsheet verifier error: {e}")
            return 0.0, f"spreadsheet verifier error: {e}"

    if verifier_type in _MCE_TEXT_VERIFIERS:
        try:
            score, analysis = _MCE_TEXT_VERIFIERS[verifier_type](
                predict_str_list, ground_truth, extra
            )
            acc_weight = extra.get("acc_reward_weight", 1.0)
            return float(acc_weight) * float(score), analysis
        except Exception as e:
            print(f"[compute_score] {verifier_type} verifier error: {e}")
            return 0.0, f"{verifier_type} verifier error: {e}"

    # ----- Default: LLM-as-judge ------------------------------------
    extract_answer = extra.get("gpt_extract_answer", False)
    acc_weight = extra.get('acc_reward_weight', 1.0)

    try:
        reward, analysis = evaluate_answer(
            question=prompt,
            prediction_list=predict_str_list,
            ground_truth=ground_truth,
            extract_answer=extract_answer,
            extra_info=extra
        )

        if not isinstance(reward, (int, float)):
            print(f"[compute_score] Warning: Invalid reward type: {type(reward)}, defaulting to 0.0")
            reward = 0.0
            if not analysis:
                analysis = f"Invalid reward value: {reward}"

        accuracy_score = acc_weight * float(reward)
        return accuracy_score, analysis

    except Exception as e:
        print(f"[compute_score] Error: {e}")
        return 0.0, f"Evaluation error: {str(e)}"


if __name__ == '__main__':
    question = "What is the name of the store with a blue sign?"
    prediction = [
        "<think>Looking at the image, I can see a blue sign on the right.</think> "
        "<answer>The name is J&optica.</answer>"
    ]
    ground_truth = "Jptica"
