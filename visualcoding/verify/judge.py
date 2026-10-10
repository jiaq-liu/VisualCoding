"""Grading: the agent harness's LLM-as-judge (vendored; the judge of every reported executor rollout) or exact match.

`judge_llm` reproduces the agent harness _evaluate_trajectory: prompt = question text, prediction list = all
assistant texts, extraction 'strict' (last <answer> tag, else last assistant turn unless it starts with 'Error:').
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ._paths import enable_vendor
from .data import Task

enable_vendor()
from utils.llm_judger import compute_score  # noqa: E402

EXTRA_INFO = {"acc_reward_weight": 1.0, "gpt_extract_answer": True, "extract_answer_tags": "strict"}
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LETTER_RE = re.compile(r"^\(?([A-J])\)?(?:[.:)]|\s|$)")
_MULTI_RE = re.compile(r"^[A-J](?:\s*[,、]?\s*[A-J])+$")


@dataclass
class ParsedAnswer:
    raw: str
    letter: str | None = None          # MC letter (single)
    letters: str | None = None         # multi-select letters, e.g. "AB"
    value: float | None = None         # numeric value (option's number for MC)
    text: str = ""


def parse_answer(raw: str | None, task: Task) -> ParsedAnswer:
    """Letter / multi-select letters / number out of a final answer string (used by the exact judge)."""
    raw = (raw or "").strip()
    pa = ParsedAnswer(raw=raw, text=raw)
    if not raw:
        return pa
    if task.is_mc:
        m = _LETTER_RE.match(raw)
        if m:
            pa.letter = m.group(1)
        elif _MULTI_RE.match(raw.replace(" ", "")):
            pa.letters = "".join(sorted(set(re.findall(r"[A-J]", raw))))
        else:
            low = raw.lower().strip(" .")
            for k, v in task.options.items():
                if low == v.lower().strip(" .") or low == f"{k.lower()}. {v.lower()}":
                    pa.letter = k
                    break
        num = task.options_numeric
        if pa.letter and num and pa.letter in num:
            pa.value = num[pa.letter]
            return pa
    m = _NUM_RE.search(raw.replace(",", ""))
    if m and not pa.letter:
        try:
            pa.value = float(m.group())
        except ValueError:
            pass
    return pa


def judge_llm(task: Task, assistant_responses: list[str]) -> tuple[float, str]:
    try:
        score, analysis = compute_score(prompt=task.question_text, predict_str_list=assistant_responses,
                                        ground_truth=task.answer, extra_info=EXTRA_INFO)
        return float(score), analysis or "Evaluation completed."
    except Exception as e:  # noqa: BLE001
        return 0.0, f"Evaluation failed: {e}"


def judge_exact(task: Task, final_answer: str) -> tuple[float, str] | None:
    """Deterministic grading where the format allows it (MC letter, single number); None otherwise."""
    pa = parse_answer(final_answer, task)
    gt = task.answer.strip()
    if task.is_mc:
        gtl = task.gt_letter
        if gtl is None and re.fullmatch(r"[A-J]+", gt):        # multi-select like "AB"
            got = pa.letters or pa.letter or ""
            return (1.0 if "".join(sorted(got)) == "".join(sorted(gt)) else 0.0), f"multi-select {got} vs {gt}"
        if gtl:
            return (1.0 if pa.letter == gtl else 0.0), f"letter {pa.letter} vs {gtl}"
        return None
    m = _NUM_RE.fullmatch(gt.replace(",", ""))
    if m and pa.value is not None:
        return (1.0 if abs(pa.value - float(m.group())) < 1e-6 else 0.0), f"number {pa.value} vs {gt}"
    return None


def judge(task: Task, assistant_responses: list[str], final_answer: str, mode: str = "llm") -> tuple[float, str, str]:
    """Returns (score, analysis, judge_used)."""
    if mode in ("exact", "auto"):
        r = judge_exact(task, final_answer)
        if r is not None:
            return r[0], r[1], "exact"
        if mode == "exact":
            return 0.0, "exact judge not applicable", "exact"
    s, a = judge_llm(task, assistant_responses)
    return s, a, "llm"
