"""Answer adapters: measurement value -> answer key, and correctness, so that `measure()` never touches options.

number  -> nearest numeric option (MC) | formatted number (free-form)
string  -> option letter (if the value is a letter of the options), option whose text equals / contains / best
           token-overlaps the value (unique), or the string itself (free-form)
list    -> JSON list (free-form); conventions: 'identity' or 'inverse' (permutation inverse), chosen on TRAIN
list of numbers on an MC task with numeric options -> CANDIDATES: the option nearest to any candidate (a
           measurement known up to a symmetry, e.g. an angle mod 90); the harness records opt_d1 / opt_margin
bool    -> the option whose text is yes / no (MC)
dict {letter: bool} -> OPTION VERDICTS: the code evaluated each option independently (a move string simulated
           on the parsed maze, a claimed permutation checked against seam costs); the answer is the unique True letter
"""
from __future__ import annotations

import json
import re

from .. import instruments as I

_BASE = I.Instrument()


def inverse_perm(lst):
    """Inverse of a 1-based permutation given as a list (position i holds piece p  ->  piece p is at position i)."""
    n = len(lst)
    inv = [None] * n
    for pos, p in enumerate(lst, start=1):
        try:
            q = int(p)
        except (TypeError, ValueError):
            return None
        if not 1 <= q <= n or inv[q - 1] is not None:
            return None
        inv[q - 1] = pos
    return inv


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def option_for_text(value: str, task) -> str | None:
    v = (value or "").strip()
    if not v or not task.is_mc:
        return None
    if re.fullmatch(r"\(?[A-J]\)?[.:]?", v.upper()):
        letter = re.sub(r"[^A-J]", "", v.upper())
        return letter if letter in task.options else None
    nv = I.norm_text(v)
    exact = [k for k, t in task.options.items() if I.norm_text(t) == nv]
    if len(exact) == 1:
        return exact[0]
    contains = [k for k, t in task.options.items() if nv and (nv in I.norm_text(t) or I.norm_text(t) in nv)]
    if len(contains) == 1:
        return contains[0]
    tv = _tokens(v)
    scored = sorted(((len(tv & _tokens(t)), k) for k, t in task.options.items()), reverse=True)
    if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def answer_from_value(value, task, convention: str = "identity"):
    if value is None:
        return None
    if isinstance(value, bool):
        if task.is_mc:
            want = "yes" if value else "no"
            hits = [k for k, t in task.options.items() if I.norm_text(t) == want]
            return hits[0] if len(hits) == 1 else None
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if task.is_mc:
            return I.nearest_letter(float(value), task) if task.options_numeric else None
        return f"{float(value):g}"
    if isinstance(value, str):
        return option_for_text(value, task) if task.is_mc else value.strip()
    if isinstance(value, (list, tuple)):
        lst = list(value)
        if task.is_mc and task.options_numeric and lst and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in lst):
            opts = task.options_numeric
            return min(opts, key=lambda k: (min(abs(opts[k] - float(x)) for x in lst), k))
        if convention == "inverse":
            lst = inverse_perm(lst)
            if lst is None:
                return None
        if task.is_mc:
            return option_for_text(json.dumps(lst), task)
        return json.dumps([int(x) if isinstance(x, float) and float(x).is_integer() else x for x in lst])
    if isinstance(value, dict):
        if task.is_mc and value and all(isinstance(v, bool) for v in value.values()):
            hits = [k for k, v in value.items() if v and str(k).strip().upper() in task.options]
            return str(hits[0]).strip().upper() if len(hits) == 1 else None
        return None
    return str(value)


def is_correct(answer, task):
    """Same semantics as Instrument.is_correct (letters for MC, numeric tolerance, normalised text) plus list equality."""
    if answer is None:
        return None
    if not task.is_mc:
        try:
            a = json.loads(answer) if isinstance(answer, str) and answer.strip().startswith("[") else None
            g = json.loads(task.answer) if task.answer.strip().startswith("[") else None
            if isinstance(a, list) and isinstance(g, list):
                return int([str(x) for x in a] == [str(x) for x in g])
        except (ValueError, TypeError):
            pass
    return _BASE.is_correct(answer, task)


def gt_key(task):
    return _BASE.gt_key(task)


def choose_convention(values: dict, tasks: dict) -> tuple[str, dict]:
    """For list-valued readings: pick identity vs inverse by TRAIN accuracy (2 candidates; reported)."""
    stats = {}
    for conv in ("identity", "inverse"):
        n = right = 0
        for tid, v in values.items():
            if not isinstance(v, (list, tuple)):
                continue
            ans = answer_from_value(v, tasks[tid], conv)
            c = is_correct(ans, tasks[tid])
            if c is not None:
                n += 1
                right += c
        stats[conv] = {"n": n, "right": right}
    if not stats["identity"]["n"]:
        return "identity", stats
    best = max(stats, key=lambda k: (stats[k]["right"], k == "identity"))
    return best, stats
