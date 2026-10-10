"""Records of the frozen split -> `Task`.

Record format (data/<dataset>/<family>_{train,test}.json, see data/README.md):
  {"images": ["tirbench/data/maze/110.png"], "doc_id": "tirbench_maze_91", "problem": "<image>\n<question>[\n\nOptions:\nA. ..]",
   "solution": "D" | "3" | "NCA" | ..., "data_source": "tirbench", "dataset": "tirbench", "family": "maze", "split": "test", ...}
Image paths are relative to the image root (`_paths.IMAGE_ROOT`).  The family comes from the record; multiple-choice options
are parsed out of the question text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ._paths import IMAGE_ROOT

DEFAULT_IMAGE_ROOT = IMAGE_ROOT
_FAMILY_BY_ID: dict[str, str] = {}     # doc_id -> record["family"] (frozen split records carry it; ids such as mira_mr_gear_rotation_2_1 do not parse)

_DOC_RE = re.compile(r"^([a-z0-9]+)_(.+?)_(\d+)$")
# "A. 3" / "A: 4%" / "A.ab and cd" at line start (TIR-Bench math / refcoco / symbolic)
_OPT_BLOCK_RE = re.compile(r"^\s*([A-J])[.:)]\s*(.+?)\s*$", re.M)
# "(A) 12 (B) 11 (C) 13" inline (TIR-Bench color, ...)
_OPT_INLINE_RE = re.compile(r"\(([A-J])\)\s*(.+?)(?=\s*\([A-J]\)|\s*$)", re.S)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_OPT_BARE_RE = re.compile(r"(?:(?<=\s)|^)([A-J])\s+(-?\d[\d.,]*(?:\s*[-–]\s*-?\d[\d.,]*)?(?:\s?(?:%|mA|ml|mL|cm|mm|kV|psi|bar|°C|°F|kg/cm²|N|g|kg|V|A))?"
                          r"|None of the above|none of the above)(?=\s|$)")


@dataclass
class Task:
    task_id: str
    question: str            # raw problem text (with <image> placeholders)
    question_text: str       # placeholders removed, stripped
    images: list[Path]       # absolute paths
    answer: str              # ground truth ("D. 1", "A", "NCA", ...)
    source: str              # dataset name (record["data_source"])
    family: str              # family name (record["family"])
    options: dict[str, str] = field(default_factory=dict)   # letter -> text ({} if free-form)

    @property
    def is_mc(self) -> bool:
        return bool(self.options)

    @property
    def options_numeric(self) -> dict[str, float] | None:
        """letter -> float if every option is a single number (units/% allowed), else None."""
        if not self.options:
            return None
        out = {}
        for k, v in self.options.items():
            s = v.replace(",", "")
            nums = _NUM_RE.findall(s)
            if len(nums) != 1 or re.search(r"[A-Za-z]{3,}", s):
                return None
            out[k] = float(nums[0])
        return out

    @property
    def gt_letter(self) -> str | None:
        m = re.match(r"^\(?([A-J])\)?(?:[.:)]|\s|$)", self.answer.strip())
        return m.group(1) if (m and self.is_mc) else None

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "family": self.family, "source": self.source,
                "question_text": self.question_text, "answer": self.answer,
                "options": self.options, "images": [str(p) for p in self.images]}


def parse_options(question: str) -> dict[str, str]:
    """Return letter->text when the question carries a well-formed option list."""
    pairs = _OPT_BLOCK_RE.findall(question)
    if len(pairs) >= 2:
        letters = [k for k, _ in pairs]
        if letters[0] == "A" and letters == sorted(letters) and len(set(letters)) == len(letters):
            return {k: v.strip() for k, v in pairs}
    if "(A)" in question and "(B)" in question:
        tail = question[question.index("(A)"):]
        pairs = _OPT_INLINE_RE.findall(tail)
        if len(pairs) >= 2:
            return {k: v.strip() for k, v in pairs}
    # "A 36.3 B 36.4 C 36.5 ... F None of the above" (tir instrument): bare letters followed by a number / range
    pairs = _OPT_BARE_RE.findall(question)
    if len(pairs) >= 3:
        letters = [k for k, _ in pairs]
        if letters[0] == "A" and letters == sorted(letters) and len(set(letters)) == len(letters) \
                and all(ord(b) - ord(a) == 1 for a, b in zip(letters, letters[1:])):
            return {k: v.strip() for k, v in pairs}
    return {}


def family_of(doc_id: str) -> str:
    if doc_id in _FAMILY_BY_ID:
        return _FAMILY_BY_ID[doc_id]
    m = _DOC_RE.match(doc_id or "")
    return m.group(2).lower() if m else "unknown"


def load_tasks(paths: Iterable[str | Path], image_root: str | Path | None = None) -> list[Task]:
    root = Path(image_root) if image_root else DEFAULT_IMAGE_ROOT
    tasks: list[Task] = []
    for p in paths:
        for rec in json.load(open(p)):
            q = rec.get("problem") or rec.get("question") or ""
            qt = re.sub(r"<image>\s*", "", q, flags=re.I).strip()
            doc_id = rec.get("doc_id") or rec.get("question_id")
            if rec.get("family"):
                _FAMILY_BY_ID[doc_id] = str(rec["family"]).lower()
            tasks.append(Task(
                task_id=doc_id,
                question=q,
                question_text=qt,
                images=[(root / im).resolve() for im in rec.get("images", [])],
                answer=str(rec.get("solution", "")),
                source=rec.get("data_source") or doc_id.split("_")[0],
                family=family_of(doc_id),
                options=parse_options(qt),
            ))
    return tasks
