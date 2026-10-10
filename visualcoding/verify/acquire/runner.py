"""Sandboxed execution of a model-written `measure(img, task, lib)` on a task list (subprocess, per-task timeout).

The code sees a TaskView (id, question text, options, family, source) and never the answer; imports are restricted
by a static check to numpy / cv2 / PIL / scipy / skimage / math / re / json / collections / itertools / statistics / functools
(scipy / skimage added - the library is a starting point, not a boundary; no text recogniser).
Output rows: task_id, value, diag, n_calls, error, elapsed (JSON-safe).  The parent (gate.py) turns them into
instrument_eval-style rows with answers and correctness.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ALLOWED_IMPORTS = {"numpy", "cv2", "PIL", "scipy", "skimage", "math", "re", "json", "collections", "itertools", "statistics", "functools"}
PIXEL_MODULES = {"numpy", "cv2", "scipy", "skimage", "PIL"}     # a call into one of these makes a value pixel-derived (model_read)
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "getattr", "setattr", "vars"}
FORBIDDEN_ATTRS = {"answer", "solution", "gt", "__dict__", "__class__", "__subclasses__"}


@dataclass
class TaskView:
    task_id: str
    question_text: str
    options: dict
    family: str
    source: str

    @property
    def is_mc(self) -> bool:
        return bool(self.options)


ASK_QUESTION_WORDS = re.compile(r"how many|how much|what is the (answer|number|angle|count|value|color|colour)|answer the question|"
                                r"which (option|answer|choice)|degrees|correct answer|select (from|the)", re.I)
# a transcription request ("what is the number written on this piece?", "read the printed label") is the one legitimate
# model input that overlaps the question words; it is NOT a violation
# - model_read() still flags the candidate when the transcription is the value itself
TRANSCRIBE_CUES = re.compile(r"written|printed|label|digit|transcri|read (the|this|these) (text|number|numbers|label|digits|characters)|"
                             r"characters", re.I)


def _strings_in(node) -> list[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
    return out


def _is_re_call(n) -> bool:
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "re"


def _question_sources(node, assigns: dict, seen: set) -> tuple[set, list]:
    """(task-question sources, string constants) that flow into `node`: direct uses of task.question_text / task.options /
    lib.question(task), and, through simple def-use on local names, values assigned earlier (`prompt = f"... {task.question_text}"`).
    Routing uses do not count: a source under a comparison (`'nested' in q`) or inside a `re.*` call (phrase extraction)."""
    out, strings = set(), []
    stack = [node]
    while stack:
        m = stack.pop()
        if isinstance(m, ast.Compare) or _is_re_call(m):
            continue
        if isinstance(m, ast.Attribute) and m.attr in ("question_text", "options"):
            out.add("task." + m.attr)
        elif isinstance(m, ast.Call) and isinstance(m.func, ast.Attribute) and m.func.attr == "question":
            out.add("lib.question(task)")
        elif isinstance(m, ast.Constant) and isinstance(m.value, str):
            strings.append(m.value)
        elif isinstance(m, ast.Name) and m.id in assigns and m.id not in seen:
            seen.add(m.id)
            for v in assigns[m.id]:
                o, st = _question_sources(v, assigns, seen)
                out |= o
                strings += st
        stack.extend(ast.iter_child_nodes(m))
    return out, strings


def _ask_violations(tree, questions: list[str] | None = None) -> list[str]:
    """`lib.ask` may only request a DISCRETE INPUT (a phrase, a box, a panel index, a transcription).  Passing the task
    question / options to it, or asking the task's own question ("how many ...", "what is the angle ...") makes the model
    the instrument.
    With `questions` (the family's train questions) a question-word match counts only when the matched phrase occurs in
    one of them: "what is the color of the legend item for X" is a discrete input on chart (the code then measures the
    hue); on the colour family the same words are the task's own question."""
    errs = []
    qtext = " ".join(q.lower() for q in (questions or []))
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)] or [tree]
    for fn in fns:
        assigns: dict = {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    for nm in ast.walk(t):
                        if isinstance(nm, ast.Name):
                            assigns.setdefault(nm.id, []).append(n.value)
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and isinstance(n.target, ast.Name) and n.value is not None:
                assigns.setdefault(n.target.id, []).append(n.value)
        for n in ast.walk(fn):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "ask"):
                continue
            args = list(n.args) + [k.value for k in n.keywords]
            src_names, strings = set(), []
            for a in args:
                o, st = _question_sources(a, assigns, set())
                src_names |= o
                strings += st
            if src_names:
                errs.append(f"lib.ask receives {sorted(src_names)}: the model must not be asked the task question (an instrument "
                            "measures; lib.ask is for a discrete input such as a phrase, a box, a panel index or a transcription)")
            for st in strings:
                m = ASK_QUESTION_WORDS.search(st)
                if m and questions is not None and m.group(0).lower() not in qtext and not \
                        re.match(r"answer the question|correct answer|which (option|answer|choice)|select (from|the)", m.group(0), re.I):
                    continue                          # the phrase is not how this family's questions are asked
                if m and not TRANSCRIBE_CUES.search(st):
                    errs.append(f"lib.ask prompt {st[:60]!r} asks the task's own question ({m.group(0)!r}): not allowed; ask only for a "
                                "discrete input (phrase / box / panel index / transcription) and let code compute the value")
                    break
    return errs


def _abstains_everywhere(tree) -> list[str]:
    """`measure()` whose every `return` carries `value: None`: the candidate declares the whole family unsupported.
    That is not an instrument (nothing is measured, so nothing can be certified) and it wastes a candidate slot: 80 of
    the 270 candidates of the early null runs were such stubs, 55 of them from 3.5 flash-lite."""
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "measure"]
    if not fns:
        return []
    rets = [n for n in ast.walk(fns[0]) if isinstance(n, ast.Return)]
    if not rets:
        return []
    for n in rets:
        v = n.value
        if not isinstance(v, ast.Dict):
            return []                              # returns a variable / call: cannot tell statically
        for k, x in zip(v.keys, v.values):
            if isinstance(k, ast.Constant) and k.value == "value" and not (isinstance(x, ast.Constant) and x.value is None):
                return []
    return ["measure() returns value None on every path: declaring every template unsupported is not an instrument. At least one "
            "route must compute a reading from pixels (pick the template the library can measure, abstain on the others)"]


def _hardcoded_values(tree) -> list[str]:
    """`return {'value': <numeric literal>}`: a per-template constant is not a measurement (playbook, anti-patterns)."""
    errs = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)):
            continue
        for k, v in zip(n.value.keys, n.value.values):
            if isinstance(k, ast.Constant) and k.value == "value" and isinstance(v, ast.Constant) and \
                    isinstance(v.value, (int, float)) and not isinstance(v.value, bool):
                errs.append(f"hardcoded value {v.value!r} returned as the measurement (line {n.lineno}): a constant per question "
                            "template is not a measurement; compute the value from the image")
                break
    return errs


PIXEL_PRIMS = {"arr", "gray", "background", "fg_mask", "ink_mask", "hue_mask", "stretch", "upscale", "rotate", "components",
               "fill_holes", "holes", "contour_stats", "classify", "min_area_rect", "color_name", "n_colors", "color_share",
               "mean_color", "projection", "period", "detect", "count_objects", "apply_variant"}


def _pixel_aliases(tree) -> set:
    """Names under which the pixel modules are bound in this code (`import numpy as np`, `from scipy import ndimage as ndi`,
    `from skimage.measure import label`), plus the usual spellings."""
    names = {"np", "numpy", "cv2", "scipy", "ndimage", "ndi", "skimage", "ski"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in PIXEL_MODULES:
                    names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in PIXEL_MODULES:
            for a in n.names:
                names.add(a.asname or a.name)
    return names


def _root_name(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def model_read(tree) -> bool:
    """True when some returned `value` is derived from a `lib.ask` result WITHOUT any pixel-derived quantity entering it
    (`value = numbers_in(ask(...))[0]`, `value = 90 * panel_index`): the model's judgement is the reading.  A diagnostic, not a
    rejection: transcription of printed text is the one legitimate case."""
    pix_names = _pixel_aliases(tree)
    fns_all = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    img_names = {fn.args.args[0].arg for fn in fns_all if fn.name == "measure" and fn.args.args}
    for fn in fns_all or [tree]:
        assigns: dict = {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    for nm in ast.walk(t):
                        if isinstance(nm, ast.Name):
                            assigns.setdefault(nm.id, []).append(n.value)
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and isinstance(n.target, ast.Name) and n.value is not None:
                assigns.setdefault(n.target.id, []).append(n.value)
            elif isinstance(n, (ast.For, ast.comprehension)):
                for nm in ast.walk(n.target):
                    if isinstance(nm, ast.Name):
                        assigns.setdefault(nm.id, []).append(n.iter)

        def taints(node, seen):
            """(ask, pix): a numpy / cv2 / scipy / skimage call is TRANSPARENT (it propagates the taint of its
            arguments, it does not create pixel taint by itself): an early measure `np.linalg.lstsq` over label positions and
            an object box that all came from lib.ask was counted as pixel-computed.  Pixel sources = the image argument of
            measure() and the library's pixel primitives."""
            ask = pix = False
            stack = [node]
            while stack:
                m = stack.pop()
                if isinstance(m, ast.Call) and isinstance(m.func, ast.Attribute):
                    if m.func.attr == "ask":
                        ask = True
                        continue                      # the image / crop handed to the model does not make the value pixel-computed
                    elif m.func.attr in PIXEL_PRIMS and _root_name(m.func.value) not in pix_names:
                        pix = True                    # lib.arr / lib.fg_mask / ... (np.<same name> falls through to the arguments)
                elif isinstance(m, ast.Name) and m.id in img_names:
                    pix = True                        # the image itself (np.asarray(img), img.crop(...))
                elif isinstance(m, ast.Name) and m.id in assigns and m.id not in seen:
                    seen.add(m.id)
                    for v in assigns[m.id]:
                        a, p_ = taints(v, seen)
                        ask, pix = ask or a, pix or p_
                stack.extend(ast.iter_child_nodes(m))
            return ask, pix
        for n in ast.walk(fn):
            if not (isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)):
                continue
            for k, v in zip(n.value.keys, n.value.values):
                if isinstance(k, ast.Constant) and k.value == "value" and not (isinstance(v, ast.Constant) and v.value is None):
                    a, p_ = taints(v, set())
                    if a and not p_:
                        return True
    return False


def _no_image_use(tree) -> list[str]:
    """measure() must read its image argument somewhere.  one maze run picked the option string with the fewest
    turns (22/37 right, then 'certified' 9/40 behind a hard-coded margin) without touching the image: that is the
    executor's answer prior, not an instrument, and the GAIN tier (precision >= 0.5, better than the executor) is exactly
    the bar such a prior clears.  185 of 2318 stored candidates never used the image; all 20 accepted ones were that run."""
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "measure"]:
        if not fn.args.args:
            continue
        img = fn.args.args[0].arg
        if not any(isinstance(n, ast.Name) and n.id == img and isinstance(n.ctx, ast.Load) for n in ast.walk(fn)):
            return [f"measure() never uses its image argument `{img}`: the value must be computed from pixels; a rule over the "
                    "option texts alone (fewest turns, longest string, ...) is the executor's answer prior, not an instrument"]
    return []


def _options_on_free_form(tree) -> list[str]:
    """on a family whose tasks have NO options, code that reads `task.options` (iterating them, scoring them,
    returning a verdict dict) reads nothing on every task."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "options" and isinstance(n.value, ast.Name) and n.value.id == "task":
            return ["this family has NO options (task.options == {} on every train task): code that reads task.options - iterating "
                    "them, scoring them, returning a verdict dict over them - measures nothing here. The value must be the FULL "
                    "answer computed from the image (see the profile's answer type and examples)"]
    return []


def static_check(code: str, questions: list[str] | None = None, has_options: bool | None = None, probe: bool = False) -> list[str]:
    """probe=True: a probe snippet is not an instrument - the stub / constant / no-image / options checks do not
    apply; the ask-violation, import and name checks do."""
    errs = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    errs += _ask_violations(tree, questions)
    if not probe:
        if has_options is False:
            errs += _options_on_free_form(tree)
        errs += _hardcoded_values(tree)
        errs += _abstains_everywhere(tree)
        errs += _no_image_use(tree)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "measure" not in names:
        errs.append("no top-level `def measure(img, task, lib)`")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    errs.append(f"import of `{m}` not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errs.append(f"use of `{node.id}` not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            errs.append(f"attribute `.{node.attr}` not allowed")
    return sorted(set(errs))


def error_signature(err: str) -> str:
    """First line of an error with quoted names and numbers masked, so that `'list' object has no attribute 'replace'`
    and `'dict' object ...` are the same signature (the ledger warns when consecutive candidates share one)."""
    first = (err or "").strip().splitlines()
    if not first:
        return ""
    line = re.sub(r"'[^']*'", "'<x>'", first[0])
    line = re.sub(r"\d+(\.\d+)?", "#", line)
    return line[:120]


def lib_call_in(tb: str) -> str | None:
    """Name of the library primitive on a traceback (None when the error is in the candidate's own code)."""
    m = re.findall(r"primitives\.py\", line \d+, in (\w+)", tb or "")
    return m[-1] if m else None


def _jsonable(v, depth=0):
    if isinstance(v, (bool, int, float, str)) or v is None:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist(), depth + 1) if v.size <= 64 else f"<array {v.shape}>"
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, depth + 1) for x in list(v)[:64]] if depth < 4 else "<list>"
    if isinstance(v, dict):
        return {str(k): _jsonable(x, depth + 1) for k, x in list(v.items())[:64]} if depth < 4 else "<dict>"
    return str(v)[:80]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _exec_measure(code: str):
    ns: dict = {}
    exec(compile(code, "<measure>", "exec"), ns)      # noqa: S102 (static-checked code, subprocess)
    fn = ns.get("measure")
    if not callable(fn):
        raise RuntimeError("measure not defined")
    return fn


def run_file(code_path: str, tasks_path: str, out_path: str, model: str | None, timeout: int):
    from PIL import Image
    from .primitives import Lib
    code = Path(code_path).read_text()
    tasks = json.loads(Path(tasks_path).read_text())
    call = None
    if model:
        from ..env import setup_env
        from ..instrument_eval import make_call
        setup_env(model, model)
        counter = {}
        call = make_call(model, {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048}, counter)
    if not model:
        counter = {}
    fn = _exec_measure(code)
    try:
        Lib.warm_detectors(code)       # weights load once here, not inside a task's alarm
    except Exception as e:  # noqa: BLE001
        print(f"[runner] detector warm-up failed: {type(e).__name__}: {e}", file=sys.stderr)
    signal.signal(signal.SIGALRM, _alarm)
    with open(out_path, "w") as fh:
        for t in tasks:
            tv = TaskView(t["task_id"], t["question_text"], t["options"], t["family"], t["source"])
            lib = Lib(call, debug_dir=str(Path(out_path).parent / "debug" / t["task_id"]),
                      auto_render=os.environ.get("ACQ_AUTO_RENDER") == "1")
            row = {"task_id": t["task_id"], "value": None, "diag": {}, "n_calls": 0, "error": None, "elapsed": 0.0}
            tk0 = (counter.get("in_tokens", 0), counter.get("out_tokens", 0))
            t0 = time.time()
            signal.alarm(timeout)
            try:
                img = Image.open(t["image"]).convert("RGB")
                res = fn(img, tv, lib)
                if not isinstance(res, dict) or "value" not in res:
                    raise ValueError("measure must return {'value': ..., 'diag': {...}}")
                row["value"] = _jsonable(res.get("value"))
                d = res.get("diag") or {}
                row["diag"] = _jsonable(d) if isinstance(d, dict) else {"diag": _jsonable(d)}
            except _Timeout:
                row["error"] = f"timeout after {timeout}s"
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-800:]
            finally:
                signal.alarm(0)
            row["n_calls"] = lib.n_calls
            row["tokens"] = {"in": counter.get("in_tokens", 0) - tk0[0], "out": counter.get("out_tokens", 0) - tk0[1]}
            row["contents"] = [c[:2000] for c in lib.contents]
            row["debug"] = lib.debug_files + lib.auto_files       # the model's own saves first, then the harness renders
            row["checks"] = lib.checks                            # sub-step assertions (lib.check)
            row["elapsed"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()


def run_candidate(code: str, tasks: list, work_dir: str | Path, model: str | None = None,
                  timeout_task: int = 30, timeout_total: int = 1800) -> tuple[list[dict], str | None]:
    """Run `code` on `tasks` (verify.data.Task list) in a subprocess; returns (rows, fatal_error)."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "code.py").write_text(code)
    (work / "tasks.json").write_text(json.dumps([{"task_id": t.task_id, "image": str(t.images[0]),
                                                  "question_text": t.question_text, "options": t.options,
                                                  "family": t.family, "source": t.source} for t in tasks]))
    out = work / "raw.jsonl"
    cmd = [sys.executable, "-m", "verify.acquire.runner", "--code", str(work / "code.py"), "--tasks", str(work / "tasks.json"),
           "--out", str(out), "--timeout", str(timeout_task)] + (["--model", model] if model else [])
    fatal = None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_total,
                           cwd=str(Path(__file__).resolve().parents[2]))
        if p.returncode != 0:
            fatal = (p.stderr or p.stdout)[-1500:]
    except subprocess.TimeoutExpired:
        fatal = f"total timeout {timeout_total}s"
    rows = []
    if out.exists():
        rows = [json.loads(l) for l in open(out)]
    return rows, fatal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=int, default=30)
    a = ap.parse_args()
    run_file(a.code, a.tasks, a.out, a.model, a.timeout)


if __name__ == "__main__":
    main()
