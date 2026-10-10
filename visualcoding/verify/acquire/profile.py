"""Automatic family profile: what the writer is shown before hypothesising (sizes, palette, components, periodicity,
question templates, answer types), rendered as text for the model plus a labelled contact sheet of samples."""
from __future__ import annotations

import collections
import json
import re
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import instruments as I
from .primitives import Lib

COLOUR_WORDS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "magenta", "black", "white",
                "gray", "grey", "brown")


def _q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[i]


def template(q: str) -> str:
    t = I.strip_question_text(q) if hasattr(I, "strip_question_text") else q
    t = t.lower()
    t = re.sub(r"\"[^\"]*\"|'[^']*'", '"…"', t)
    t = re.sub(r"-?\d+(?:\.\d+)?", "#", t)
    t = re.sub(r"\b(" + "|".join(COLOUR_WORDS) + r")\b", "<colour>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def answer_type(task) -> str:
    if task.is_mc:
        return "MC-numeric-options" if task.options_numeric else "MC-text-options"
    a = task.answer.strip()
    if a.startswith("["):
        return "free-list"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", a.replace(",", "")):
        return "free-number"
    return "free-text"


def image_stats(img: Image.Image, lib: Lib) -> dict:
    a, s = lib.arr(img, 1000)
    bg = lib.background(a)
    fg = lib.fg_mask(a, 60, bg)
    comps, _ = lib.components(fg, min_area=int(20 * s * s) or 5)
    ink = lib.ink_mask(a)
    pr = lib.period(lib.projection(ink, 1))
    pc = lib.period(lib.projection(ink, 0))
    return {"w": img.width, "h": img.height, "bg": [int(v) for v in bg], "fg_frac": round(float(fg.mean()), 3),
            "n_components": len(comps), "n_colors16": lib.n_colors(a), "period_strength": round(max(pr["strength"], pc["strength"]), 3),
            "period_px": (pr["period"] or pc["period"])}


def profile_family(tasks: list, out_dir: str | Path, n_thumbs: int = 8, n_examples: int = 4) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lib = Lib()
    stats = []
    for t in tasks:
        try:
            img = Image.open(t.images[0]).convert("RGB")
            stats.append(image_stats(img, lib))
        except Exception as e:  # noqa: BLE001
            stats.append({"error": str(e)})
    ok = [s for s in stats if "error" not in s]
    sizes = collections.Counter((s["w"], s["h"]) for s in ok)
    tmpl = collections.Counter()
    example_of = {}
    for t in tasks:
        k = template(I.strip_question(t))
        tmpl[k] += 1
        example_of.setdefault(k, t.task_id)
    atypes = collections.Counter(answer_type(t) for t in tasks)
    n_opts = collections.Counter(len(t.options) for t in tasks if t.is_mc)
    prof = {
        "family": f"{tasks[0].source}:{tasks[0].family}" if tasks else "?",
        "n_tasks": len(tasks), "sources": dict(collections.Counter(t.source for t in tasks)),
        "sizes_top": [[list(k), v] for k, v in sizes.most_common(5)], "n_distinct_sizes": len(sizes),
        "bg_typical": [int(statistics.median(s["bg"][i] for s in ok)) for i in range(3)] if ok else None,
        "fg_frac_q": [_q([s["fg_frac"] for s in ok], p) for p in (0.1, 0.5, 0.9)],
        "n_components_q": [_q([s["n_components"] for s in ok], p) for p in (0.1, 0.5, 0.9)],
        "n_colors16_q": [_q([s["n_colors16"] for s in ok], p) for p in (0.1, 0.5, 0.9)],
        "periodic_share": round(sum(1 for s in ok if s["period_strength"] >= 0.5) / max(len(ok), 1), 2),
        "templates": [{"template": k[:220], "n": v, "example": example_of[k]} for k, v in tmpl.most_common(8)],
        "n_templates": len(tmpl),
        "answer_types": dict(atypes), "n_options": dict(n_opts),
        "examples": [{"task_id": t.task_id, "question": t.question_text[:500], "options": t.options, "answer": t.answer}
                     for t in tasks[:n_examples]],
        "per_task": {t.task_id: s for t, s in zip(tasks, stats)},
    }
    # contact sheet
    thumbs = []
    for t in tasks[:n_thumbs]:
        try:
            im = Image.open(t.images[0]).convert("RGB")
            im.thumbnail((256, 256))
            canvas = Image.new("RGB", (260, 276), (245, 245, 245))
            canvas.paste(im, (2 + (256 - im.width) // 2, 2 + (256 - im.height) // 2))
            ImageDraw.Draw(canvas).text((4, 262), t.task_id[-24:], fill=(0, 0, 0))
            thumbs.append(canvas)
        except Exception:  # noqa: BLE001
            pass
    if thumbs:
        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 260, rows * 276), (255, 255, 255))
        for i, th in enumerate(thumbs):
            sheet.paste(th, ((i % cols) * 260, (i // cols) * 276))
        sheet.save(out / "contact.png")
        prof["contact_sheet"] = str(out / "contact.png")
    (out / "profile.json").write_text(json.dumps(prof, ensure_ascii=False, indent=1))
    return prof


def render(prof: dict) -> str:
    L = [f"# Family profile: {prof['family']} ({prof['n_tasks']} train tasks; sources {prof['sources']})",
         f"- image sizes (w,h) top: {prof['sizes_top']} ({prof['n_distinct_sizes']} distinct)",
         f"- typical background RGB {prof['bg_typical']}; foreground fraction p10/p50/p90 {prof['fg_frac_q']}",
         f"- foreground components p10/p50/p90 {prof['n_components_q']}; distinct colours (16-level quantised) {prof['n_colors16_q']}",
         f"- share of images with a strong periodic (grid-like) structure: {prof['periodic_share']}",
         f"- answer types {prof['answer_types']}; number of options {prof['n_options']}"
         + ("" if prof.get("n_options") else
            " - NO OPTIONS: task.options is {} on every task, so a verdict dict over options is impossible; the value must be "
            "the FULL answer in the format of the examples below (a free-list answer = the complete list, in the order the "
            "question defines; a free-number answer = the number)"),
         f"- question templates ({prof['n_templates']} distinct; numbers -> #, colours -> <colour>):"]
    for t in prof["templates"]:
        L.append(f"    [{t['n']:2d}] {t['template']}   (e.g. {t['example']})")
    L.append("- examples (train; the code must never receive answers):")
    for e in prof["examples"]:
        opts = "; ".join(f"{k}. {v}" for k, v in e["options"].items()) if e["options"] else "(free-form)"
        L.append(f"    {e['task_id']}: Q: {e['question'][:300]!r}  options: {opts}  answer: {e['answer']!r}")
    return "\n".join(L)
