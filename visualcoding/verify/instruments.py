"""The Instrument contract: a fixed measurement procedure whose reading is converted to an answer by code.

An instrument is NOT a rollout: it is a fixed procedure (pure image code, optionally one or more single-turn JSON-only
model calls with the options hidden) that returns a structured reading, a derived answer and a range flag.  The operating
range is a rule fitted on train (`acquire.fit_rule`); outside the range the harness falls back to the executor's own
rollout (hard switch).  Acquired programs are wrapped as instruments by `acquire.deploy`.

Registry: `INSTRUMENTS[key]`; each instrument declares the families it serves (family names as in verify.data.Task.family,
optionally qualified by dataset, "tirbench:ocr").

    Reading  = value (number / list / str), answer (derived key), in_range, diag, n_calls, contents
    Instrument.measure(task, img, call) -> Reading         (img = native-resolution PIL image)
    Instrument.answer_from_value(value, task) -> key
    Instrument.gt_key(task) -> key                         (what a correct answer key looks like)
    Instrument.is_correct(answer, task) -> 1 / 0 / None    (exact normalised match; None = no answer)

The module also holds the image / prompt / answer helpers shared with the `lib` primitives (json_load, msgs, part,
upscale, orientation_panels, apply_variant, stretch_contrast).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

from .data import Task
from .model import image_part

MAX_PIXELS, MIN_PIXELS = 2_000_000, 40_000
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
SYS = "You are a precise visual measurement instrument. You reply with JSON only, no prose, no markdown fences."
Call = Callable[..., str]   # call(messages, **sampling_overrides) -> content ("" on failure)


@dataclass
class Reading:
    value: object = None
    answer: str | None = None
    in_range: bool | None = None
    diag: dict = field(default_factory=dict)
    n_calls: int = 0
    contents: list = field(default_factory=list)


def json_load(content: str):
    s = (content or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        m = re.search(r"[\[{].*[\]}]", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group())
            except Exception:  # noqa: BLE001
                return None
    return None


def strip_question(task: Task) -> str:
    """Question text without the option lines (so the instrument is not anchored on the options)."""
    lines = []
    for ln in task.question_text.splitlines():
        s = ln.strip()
        if not s or re.match(r"^\(?[A-J][\).:]\s", s) or s.lower().startswith("options"):
            continue
        lines.append(s)
    txt = " ".join(lines)
    txt = re.sub(r"\s*\([A-J]\)\s*[^()]*", "", txt) if "(A)" in txt else txt   # inline "(A) .. (B) .." options
    txt = re.sub(r"\s*[A-J]\)\s*[^()]*?(?=\s[A-J]\)|$)", "", txt) if re.search(r"\sA\)\s", txt) else txt
    return txt.strip()


def nearest_letter(value: float, task: Task) -> str | None:
    opts = task.options_numeric
    if not opts or value is None:
        return None
    return min(opts, key=lambda k: (abs(opts[k] - value), k))


def gt_value(task: Task) -> float | None:
    if task.is_mc:
        gl = task.gt_letter
        if gl and gl in task.options:
            m = _NUM_RE.search(task.options[gl].replace(",", ""))
            return float(m.group()) if m else None
        return None
    m = _NUM_RE.fullmatch(task.answer.strip().replace(",", ""))
    return float(m.group()) if m else None


def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^[\(\[]?[a-j][\)\].:]\s*", "", s) if False else s
    return re.sub(r"[\s\.,;:'\"“”‘’`\-_/\\()\[\]{}]+", "", s)


def msgs(img_parts: list, text: str, sys: str = SYS) -> list[dict]:
    return [{"role": "system", "content": sys},
            {"role": "user", "content": [*img_parts, {"type": "text", "text": text}]}]


def part(img: Image.Image, max_pixels: int = MAX_PIXELS) -> dict:
    return image_part(img, max_pixels, MIN_PIXELS)[0]


class Instrument:
    key: str = ""
    families: tuple[str, ...] = ()           # "refcoco" or "tirbench:ocr"
    deterministic: bool = False              # pure code -> one sample is enough
    cost_note: str = ""

    def serves(self, task: Task) -> bool:
        return task.family in self.families or f"{task.source}:{task.family}" in self.families

    def measure(self, task: Task, img: Image.Image, call: Call) -> Reading:  # pragma: no cover
        raise NotImplementedError

    # answer keys ---------------------------------------------------------------------------------
    def answer_from_value(self, value, task: Task):
        if value is None:
            return None
        if task.is_mc:
            return nearest_letter(float(value), task) if isinstance(value, (int, float)) else None
        return f"{float(value):g}" if isinstance(value, (int, float)) else str(value)

    def gt_key(self, task: Task):
        if task.is_mc:
            return task.gt_letter
        return task.answer.strip()

    def is_correct(self, answer, task: Task) -> int | None:
        if answer is None:
            return None
        gk = self.gt_key(task)
        if gk is None:
            return None
        if task.is_mc:
            return int(str(answer).strip().upper() == gk)
        a, g = str(answer), str(gk)
        try:
            return int(abs(float(a.replace(",", "")) - float(g.replace(",", ""))) < 1e-6)
        except ValueError:
            return int(norm_text(a) == norm_text(g))

    def range_from_row(self, row: dict, task: Task) -> bool | None:
        """Recompute the range flag of a stored reading from its diag (train-fitted rules applied post hoc by
        `instrument_eval refresh`); None = keep the stored flag."""
        return None

    # range: default = the reading parsed
    def finalize(self, rd: Reading, task: Task) -> Reading:
        if rd.answer is None and rd.value is not None:
            rd.answer = self.answer_from_value(rd.value, task)
        if rd.in_range is None:
            rd.in_range = rd.answer is not None
        return rd


INSTRUMENTS: dict[str, Instrument] = {}


def for_task(task: Task, keys: list[str] | None = None) -> list[Instrument]:
    return [i for k, i in INSTRUMENTS.items() if (keys is None or k in keys) and i.serves(task)]


def upscale(img: Image.Image, min_side: int = 400) -> Image.Image:
    s = max(1.0, min_side / min(img.size))
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS) if s > 1 else img


def orientation_panels(img: Image.Image) -> tuple[Image.Image, list]:
    from PIL import ImageDraw, ImageFont, ImageOps
    base = img
    if max(base.size) > 480:
        s = 480 / max(base.size)
        base = base.resize((int(base.width * s), int(base.height * s)), Image.LANCZOS)
    variants = []
    for k in range(4):
        variants.append(("rot", k))
    for k in range(4):
        variants.append(("mirror", k))
    ims = []
    for kind, k in variants:
        v = base.rotate(-90 * k, expand=True)
        if kind == "mirror":
            v = ImageOps.mirror(v)
        ims.append(v)
    pad, lab = 20, 44
    cw = max(i.width for i in ims) + pad
    ch = max(i.height for i in ims) + lab + pad
    out = Image.new("RGB", (cw * 4 + pad, ch * 2 + pad), (255, 255, 255))
    dr = ImageDraw.Draw(out)
    try:
        f = ImageFont.load_default(size=30)
    except Exception:  # noqa: BLE001
        f = None
    for i, im in enumerate(ims):
        x, y = pad + (i % 4) * cw, pad + (i // 4) * ch
        dr.rectangle((x, y, x + 160, y + lab - 8), fill=(0, 0, 0))
        dr.text((x + 8, y + 4), f"PANEL {i}", fill=(255, 255, 255), font=f)
        out.paste(im, (x, y + lab))
    return out, variants


def apply_variant(img: Image.Image, variant) -> Image.Image:
    from PIL import ImageOps
    kind, k = variant
    v = img.rotate(-90 * k, expand=True)
    return ImageOps.mirror(v) if kind == "mirror" else v


def stretch_contrast(img: Image.Image) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    a = np.clip((a - lo) / max(1.0, hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(a)
