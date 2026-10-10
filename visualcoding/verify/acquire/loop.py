"""Acquisition loop: family profile -> rounds of K candidate instruments (JSON: hypothesis, witnesses, code) ->
sandboxed train run -> adapters -> gate -> feedback table -> next round edits the best; freeze the best.

    python -m verify.acquire.loop --family tirbench:rotation_game --model gemini-3.1-flash-lite \
        --rounds 5 --candidates 3 --out output/verify/acq/rotation_game_flashlite

The model only ever sees TRAIN tasks; the test readings are taken afterwards through `verify.acquire.deploy`.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import re
import time
from pathlib import Path

from PIL import Image

from .. import instrument_eval as E
from .. import instruments as I
from ..env import setup_env
from . import adapters as A
from . import fit_rule as F
from . import gate as G
from . import hypothesis as H
from . import profile as P
from . import runner as R
from .primitives import Lib

PLAYBOOK = (Path(__file__).with_name("playbook.md")).read_text()


RUNTIME = {"runner_s": 0.0, "runner_runs": 0, "runner_calls": 0, "runner_in": 0, "runner_out": 0}   # cost accounting


def run_timed(code, tasks, wd, model, timeout):
    """R.run_candidate with wall-clock and lib.ask token accounting into RUNTIME."""
    t0 = time.time()
    raw, fatal = R.run_candidate(code, tasks, wd, model, timeout)
    RUNTIME["runner_s"] += time.time() - t0
    RUNTIME["runner_runs"] += 1
    for x in raw:
        RUNTIME["runner_calls"] += int(x.get("n_calls") or 0)
        tk = x.get("tokens") or {}
        RUNTIME["runner_in"] += int(tk.get("in") or 0)
        RUNTIME["runner_out"] += int(tk.get("out") or 0)
    return raw, fatal


def est_usd(model: str, tin: int, tout: int, tthink: int = 0) -> tuple[float | None, str]:
    """Estimated USD from verify/acquire/prices.json (ASSUMED prices; thoughts billed as output).  (usd, note)."""
    try:
        pr = json.loads((Path(__file__).with_name("prices.json")).read_text())
    except Exception:  # noqa: BLE001
        return None, "no prices.json"
    key = next((k for k in pr if k != "_note" and k in (model or "")), None)
    if key is None:
        return None, f"no price for {model}"
    p = pr[key]
    return round(tin / 1e6 * p["in_per_M"] + (tout + tthink) / 1e6 * p["out_per_M"], 4), pr.get("_note", "")


def train_baselines(tasks, eye: str = "auto") -> list[str]:
    """Executor rollout dirs on TRAIN (pass@1 per task = the gain reference of the GAIN tier and the number shown to the writer).
    Given explicitly with --eye-dirs (the pipeline always does), else from the environment variable VC_TRAIN_ROLLOUTS
    (os.pathsep-separated verify.run output dirs)."""
    dirs = [d for d in os.environ.get("VC_TRAIN_ROLLOUTS", "").split(os.pathsep) if d]
    if not dirs:
        raise SystemExit("no executor TRAIN rollouts: pass --eye-dirs <verify.run output dir ...> (or set VC_TRAIN_ROLLOUTS)")
    return dirs


def probe_doc(n: int) -> str:
    if n <= 0:
        return ""
    return (f"\n\n# Probes\nBefore submitting a candidate you may run up to {n} PROBES per slot. A probe is a reply of the form "
            "{\"probe\": {\"code\": \"def probe(img, task, lib):\\n    ...\\n    return {<anything JSON-able>}\", \"tasks\": [\"<train task id>\", ...]}} "
            "(at most 3 ids, taken from the per-task table or the profile). The harness runs it on those tasks and replies in this "
            "conversation with the dict your probe returned per task, the original of each task and every picture saved with "
            "lib.debug (plus the auto-rendered first mask). Use it to check ONE task before writing the candidate: print the "
            "intermediate numbers, save the mask / crop / assembled result and look at it. The same rules apply as for measure() "
            "(no task question to lib.ask, allowed imports only); a probe is not judged and costs no round. After the probes, reply "
            "with the candidate JSON as usual.\n"
            "A FACTS probe is the same request with \"tasks\": \"all\": the harness runs it on EVERY train task and replies with the "
            "distribution of each key of the returned dict over the tasks (numbers: min / median / max with the task ids at the "
            "extremes; strings and booleans: counts; errors), a per-task line, and the pictures (original + your lib.debug saves) of "
            "the extreme tasks. Round 0 of the loop is such a probe (measure the generator before writing); later you may send one "
            "whenever a stage of your construction must be verified on all tasks at once.")


def system_prompt(lib: Lib, probes: int = 0) -> str:
    return (PLAYBOOK + "\n\n# Library\n" + lib.doc() + probe_doc(probes) +
            "\n\n# Output\nReply with the JSON object of playbook section 7 only (or a probe request, see # Probes). No markdown fences, no prose outside the JSON. "
            "Escape newlines inside the \"code\" string as \\n. Put every import you use (import numpy as np, import cv2, "
            "from scipy import ndimage, ...) inside the code: nothing is pre-imported. Optional keys: \"require\" (list of hard witness atoms such as "
            "\"agree==True\" or \"fill>=0.75\"), \"features\" (diag names the range fit may threshold; include \"opt_margin\" when "
            "the answer mapping must be unique), \"route_key\" (diag key naming the question route, default \"route\"). "
            "Required keys \"pixels_compute\" and \"model_supplies\" state the division of labour: the value must come from what "
            "code computes from pixels; the model may only supply a discrete input. Optional key \"notes\": up to 12 short lines "
            "(a list of strings) with the facts you have VERIFIED on train (sizes, geometry, colours, label positions, constants "
            "read off the facts report), what you tried and what it showed, what is still open - carried verbatim into every "
            "later round (the ledger is the harness's memory; notes are yours). A line is a FACT only when a facts probe, a "
            "lib.check or the per-task table measured it; mark anything else '(assumed)' - the harness does not verify notes.")


def picture_lines(images: list[str] | None) -> list[str]:
    """captions of the pictures attached to the message, in attachment order (contact sheet first)."""
    if not images:
        return []
    L = ["Attached pictures, in this order - LOOK at them before editing (a wrong reading is diagnosed from the picture, not "
         "guessed from the numbers):"]
    L += [f"  [picture {i + 1}] {c}" for i, c in enumerate(images)]
    L.append("  Save what you want to see next round with lib.debug(mask_or_image_or_labels, 'name') inside measure() (max 3 per "
             "task): the harness attaches those pictures for the worst tasks of your instrument next to their originals.")
    if PROBE_HINT:
        L.append("  " + PROBE_HINT)
    return L


PROBE_HINT = ""      # set in main() when probes are enabled
FACTS_ONCE = False   # ablation --facts-once: the facts report is measured once (slot r1c0) and fixed for every later slot
HIDE_EYE = False     # ablation --hide-rollout-stats: no executor rollout numbers in any writer message
_GAIN_RE = re.compile(r"gain over the executor [^;]*")


def _gain_str(train: dict) -> str:
    """', gain +x.x' (the train gain over the executor) or '' when executor statistics are hidden."""
    return "" if HIDE_EYE else f", gain {train.get('gain', 0):+.1f}"


def _why_str(why) -> str:
    """Gate reason with the executor-gain numbers removed when they are hidden (GAIN-tier reasons quote them)."""
    if not HIDE_EYE or not why:
        return why
    return _GAIN_RE.sub("gain over the executor: not shown in this run", str(why))


def _exec_line(ev: dict) -> str:
    head = ("" if HIDE_EYE else
            f"Executor eyeball pass@1 on these tasks: {ev['eye_mean'] if ev['eye_mean'] is None else round(ev['eye_mean'], 2)}; ")
    return head + f"{'E' if HIDE_EYE else 'e'}rrors {ev['errors']} (timeouts {ev['timeouts']}); model calls {ev['n_calls']}."


def first_images(tasks: list, n: int = 3, max_pixels: int = 1_000_000) -> tuple[list, list[str], list[dict]]:
    """Round 1: the originals of the first n train tasks at usable resolution (the contact sheet is 256 px thumbnails)."""
    parts, caps, manifest = [], [], []
    for t in tasks[:n]:
        try:
            im = Image.open(t.images[0]).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        parts.append(I.part(im, max_pixels))
        caps.append(f"task {t.task_id}: original ({im.width}x{im.height})")
        manifest.append({"task_id": t.task_id, "kind": "original", "path": str(t.images[0])})
    return parts, caps, manifest


def round_images(rows: list[dict], tasks: dict, n_tasks: int = 3, max_debug: int = 3, max_pixels: int = 1_000_000) -> tuple[list, list[str], list[dict]]:
    """pictures for the writer = the originals of the worst tasks of the candidate whose table is shown + what its
    code saved on them with lib.debug.  Worst first: wrong readings, then abstentions / unparsed, then errors (the
    traceback already says what happened there), then right ones."""
    def rank(r):
        if r.get("error"):
            return 2
        if r["answer"] is None:
            return 1
        return 0 if r["correct"] == 0 else 3
    kinds = {0: "WRONG reading", 1: "no reading (abstained / unparsed)", 2: "error", 3: "right"}
    parts, caps, manifest = [], [], []
    for r in sorted(rows, key=lambda r: (rank(r), r["task_id"]))[:n_tasks]:
        t = tasks[r["task_id"]]
        try:
            im = Image.open(t.images[0]).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        parts.append(I.part(im, max_pixels))
        caps.append(f"task {t.task_id}: original ({im.width}x{im.height}) - {kinds[rank(r)]}")
        manifest.append({"task_id": t.task_id, "kind": "original", "path": str(t.images[0])})
        for d in (r.get("debug") or [])[:max_debug]:
            try:
                dim = Image.open(d["path"]).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            parts.append(I.part(dim, max_pixels))
            if d.get("auto"):
                caps.append(f"task {t.task_id}: harness rendering of your code's first {d['name'][5:]} call ({dim.width}x{dim.height}; "
                            "masks white on black, labels rescaled) - this is what your code segmented on this task")
            else:
                caps.append(f"task {t.task_id}: your lib.debug {d['name']!r} ({dim.width}x{dim.height})")
            manifest.append({"task_id": t.task_id, "kind": "auto" if d.get("auto") else "debug", "name": d["name"], "path": d["path"]})
    return parts, caps, manifest


FACTS_PARA = ("Round 0 - MEASURE BEFORE YOU WRITE. Your FIRST reply is a facts probe over ALL train tasks: "
              "{\"probe\": {\"code\": \"def probe(img, task, lib):\\n    ...\\n    return {<facts>}\", \"tasks\": \"all\"}}. Measure what "
              "the instrument will rely on - image size, background colour, the grid / axis / piece / cell geometry you detect "
              "and how many lines / cells / components are found, label positions, anything you would otherwise assume - and "
              "save with lib.debug what you want to see on the extreme tasks. The harness replies with the distribution of every "
              "key over the train tasks (min / median / max with the task ids at the extremes, counts of strings and booleans, "
              "errors), a per-task line and the pictures of the extreme tasks. You may send up to {n} such probes; then reply with "
              "your `notes` (the verified facts, short lines) and the round-1 candidate.")


def first_prompt(prof_text: str, images: list[str] | None = None, facts: int = 0,
                 facts_text: str | None = None) -> str:
    return (prof_text + "\n\n" + "\n".join(picture_lines(images)) + ("\n\n" if images else "") +
            (FACTS_PARA.replace("{n}", str(facts)) + "\n\n" if facts > 0 else "") +
            (("Facts measured on ALL train tasks by a facts probe run before this slot (harness summary - the generator's "
              "constants your code may rely on; the report is fixed - no probe can be run in this slot, reply with the candidate JSON):\n"
              + facts_text + "\n\n") if facts_text else "") +
            "Round 1. Write the instrument for this family following the playbook. First state the division of "
            "labour (\"pixels_compute\": the quantity code computes from pixels; \"model_supplies\": the discrete input the model "
            "supplies, or 'nothing'), then one hypothesis, a measurement built from the library or written from numpy / cv2 / scipy / "
            "skimage yourself, witnesses as numbers in diag, unsupported templates declared. Return the JSON object.")


def _f2(x) -> str:
    return "None" if x is None else f"{x:.2f}"


def gate_text(ev: dict) -> str:
    """the gate line names the tier - certified, or GAIN (better than the fallback, not a certificate)."""
    t = ev.get("tier")
    why = _why_str(ev["why"])
    if t == "certified" or (ev["accepted"] and not t):
        return f"Gate: ACCEPTED - certified ({why})."
    if t == "gain":
        against = ("(the executor's own rate on these tasks is not shown in this run)" if HIDE_EYE else
                   f"against the executor's {_f2(ev.get('eye_mean'))} on these tasks")
        return (f"Gate: ACCEPTED at the GAIN tier only, NOT certified ({why}). In range the instrument is right "
                f"{_f2(ev['loo'].get('precision'))} of the time (leave-one-out) {against}: a 'better than the fallback' claim, not a certificate.")
    return f"Gate: REJECTED ({why})."


def tier_word(ev: dict) -> str:
    if ev.get("tier") == "gain":
        return "accepted at the GAIN tier, not certified"
    return "accepted (certified)" if ev["accepted"] else "not accepted"


def _compact(d: dict, n: int = 140) -> str:
    s = json.dumps({k: v for k, v in d.items() if k not in ("exception",)}, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def error_summary(rows: list[dict]) -> tuple[int, str]:
    """(number of erroring tasks, most common first error line) for one candidate's rows."""
    errs = [str(r["error"]).splitlines()[0][:110] for r in rows if r.get("error")]
    if not errs:
        return 0, ""
    top = max(set(errs), key=errs.count)
    return len(errs), top


def repeated_error(ledger: list[dict], rows: list[dict], err_ex: str) -> tuple[str, int, str]:
    """(signature, streak, warning) for the candidate whose rows / most common error are given: the streak counts
    consecutive ledger entries (this one included) that end in the same error signature; a warning text is returned
    when the streak is >= 2 (in the early runs every such streak was a library boundary condition - gray(a) with a
    PIL-only gray, numbers_in(dict), ask(img=array) - that the model re-wrote because the call is a reasonable API
    expectation; the traceback alone did not make it change the call)."""
    sig = R.error_signature(err_ex)
    if not sig:
        return "", 0, ""
    streak = 1
    for e in reversed(ledger):
        if e.get("error_sig") == sig and e.get("errors"):
            streak += 1
        else:
            break
    if streak < 2:
        return sig, 1, ""
    tb = next((str(r["error"]) for r in rows if r.get("error")), "")
    call = R.lib_call_in(tb)
    where = f"inside lib.{call}()" if call else "in the candidate's own code"
    prev = ledger[-1]
    return sig, streak, (f"REPEATED ERROR x{streak}: the same error as r{prev['round']}c{prev['cand']} ({err_ex[:90]}), raised {where}. "
                         "Consecutive candidates failing on one error means a boundary condition of the call, not a wrong hypothesis: "
                         + ("check the lib doc line for that primitive (input type: PIL image vs array vs the parsed reply of ask; "
                            "mask polarity; None returns) and change the CALL, then keep the same hypothesis."
                            if call else "fix that line (types / indices / None) and keep the same hypothesis."))


def ledger_lines(ledger: list[dict]) -> list[str]:
    L = []
    for e in ledger:
        if e.get("fatal"):
            L.append(f"  r{e['round']}c{e['cand']} [not run]: {e['hypothesis'][:100]!r} | {e['fatal'][:160]}")
            continue
        a = (f"A {e['right']}/{e['parsed']} ({e['agreement']:.2f}) {'passed' if e['verdict'] else ('FAILED (gain level)' if e.get('level') == 'gain' else 'FAILED')}" +
             (" via route" if e.get("verdict_by") == "route" else ""))
        b = ("accepted" + (" [GAIN tier]" if e.get("tier") == "gain" else "")) if e["accepted"] else f"B: {_why_str(e['why'])[:60]}"
        sy = f" | systematic: {'; '.join(e['systematic'])[:80]}" if e.get("systematic") else ""
        er = f" | ERRORS {e['errors']}/{e.get('n_tasks', '?')}: {e['error_example']}" if e.get("errors") else ""
        mr = " | MODEL-READ (value = the model's reply, not a measurement)" if e.get("model_read") else ""
        rw = f" | {e['warning']}" if e.get("warning") else ""
        pr = e["primitives"]
        pr = pr if len(pr) <= 10 else pr[:10] + [f"+{len(pr) - 10} more"]
        d = e.get("delta")
        dl = (f" | vs best: lost {d['lost']} right, gained {d['gained']}, {e.get('n_lines', '?')} lines changed" if d else
              (f" | {e.get('n_lines')} lines changed" if e.get("n_lines") is not None else ""))
        L.append(f"  r{e['round']}c{e['cand']} [{e['change_type']}]: {e['hypothesis'][:100]!r} | uses {pr} | {a}{sy}{er}{mr}{rw}{dl} | {b}")
    return L


def n_lines_changed(prev_code: str | None, code: str) -> int:
    """size of an edit = lines inserted / replaced / deleted between the code the model was shown and its reply."""
    if not prev_code:
        return len((code or "").splitlines())
    sm = difflib.SequenceMatcher(None, prev_code.splitlines(), (code or "").splitlines(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


def delta_vs_best(prev_rows: list[dict], rows: list[dict], prev_ev: dict, ev: dict, tasks: dict) -> dict:
    """per-task flips of a candidate against the best candidate so far - the diff a person reads after every
    edit: readings lost (right -> not right) / gained, and tasks that left / entered the fitted range."""
    P = {r["task_id"]: r for r in prev_rows}
    lost, gained, left, entered = [], [], [], []
    for r in rows:
        p = P.get(r["task_id"])
        if p is None:
            continue
        t = tasks[r["task_id"]]
        pr, nr = p["correct"] == 1, r["correct"] == 1
        if pr and not nr:
            lost.append(r["task_id"])
        if nr and not pr:
            gained.append(r["task_id"])
        try:
            pi = bool(prev_ev.get("rule")) and F.range_from_row(prev_ev["rule"], p, t)
            ni = bool(ev.get("rule")) and F.range_from_row(ev["rule"], r, t)
        except Exception:  # noqa: BLE001
            pi = ni = False
        if pi and not ni:
            left.append(r["task_id"])
        if ni and not pi:
            entered.append(r["task_id"])
    return {"lost": len(lost), "gained": len(gained), "lost_ids": lost[:10], "gained_ids": gained[:10],
            "left_range": len(left), "entered_range": len(entered), "left_ids": left[:10], "entered_ids": entered[:10]}


def delta_lines(delta: dict | None, adopted: bool, r: int, change: str) -> list[str]:
    if not delta:
        return []

    def ids(xs):
        return ", ".join(xs) if xs else "-"
    regress = delta["lost"] > 0 or delta["left_range"] > 0
    head = (f"Your round-{r} edit ({change[:120]!r}) was ADOPTED as the new best" if adopted else
            (f"REGRESSION - your round-{r} edit ({change[:120]!r}) was NOT adopted" if regress else
             f"NOT ADOPTED (no reading improved) - your round-{r} edit ({change[:120]!r}) changed nothing the gate measures"))
    L = [f"{head}: compared with the previous best it lost {delta['lost']} right reading(s) [{ids(delta['lost_ids'])}] and gained "
         f"{delta['gained']} [{ids(delta['gained_ids'])}]; {delta['left_range']} task(s) left the range [{ids(delta['left_ids'])}], "
         f"{delta['entered_range']} entered [{ids(delta['entered_ids'])}]."]
    if not adopted and delta["lost"] > 0:
        L.append("A change that loses readings the best candidate got right is a MEASUREMENT change, not an extension: the best "
                 "candidate's code is shown below again - keep everything it does on the tasks it reads right and add the new "
                 "mechanism only where it abstains or is wrong (the tasks at the top of the table). If the lost tasks share a "
                 "property, guard the new branch by it instead of replacing the old one.")
    return L


def checks_lines(rows: list[dict]) -> list[str]:
    """lib.check sub-step assertions aggregated over train, split by right / wrong readings."""
    agg: dict = {}
    for r in rows:
        for k, v in (r.get("checks") or {}).items():
            a = agg.setdefault(k, {"n": 0, "ok": 0, "fail": [], "right_ok": 0, "right_n": 0, "wrong_ok": 0, "wrong_n": 0})
            ok = bool(v.get("ok")) if isinstance(v, dict) else bool(v)
            a["n"] += 1
            a["ok"] += int(ok)
            if not ok:
                a["fail"].append(r["task_id"])
            if r["answer"] is not None:
                key = "right" if r["correct"] == 1 else "wrong"
                a[key + "_n"] += 1
                a[key + "_ok"] += int(ok)
    if not agg:
        return []
    L = ["Sub-step checks (lib.check) on train - pass / tasks, split by right and wrong readings. A check that passes on the right "
         "readings and fails on the wrong ones is a witness; a check that fails on most tasks is a broken stage - fix it before "
         "touching anything built on top of it:"]
    for k, a in agg.items():
        fails = f"; fails on {', '.join(a['fail'][:8])}{', …' if len(a['fail']) > 8 else ''}" if a["fail"] else ""
        L.append(f"  {k}: {a['ok']}/{a['n']} pass (right readings {a['right_ok']}/{a['right_n']}, wrong {a['wrong_ok']}/{a['wrong_n']}){fails}")
    return L


def facts_report(raw: list[dict], tasks_by: dict, max_keys: int = 24, max_rows: int = 40) -> tuple[str, str, list, list[dict]]:
    """report of an all-task probe = distribution of every key of the returned dict over the train tasks (what
    a person reads off a per-task table before writing a solver) with the ids at the extremes, a per-task line, and the
    pictures (original + saves) of up to 4 extreme tasks.  Returns (full text, summary carried into later rounds, parts, manifest)."""
    flat, errs = {}, []
    for x in raw:
        if x.get("error"):
            errs.append((x["task_id"], str(x["error"]).splitlines()[0][:160]))
            continue
        flat[x["task_id"]] = F._flatten(x.get("diag") or {})
    keys: list[str] = []
    for d in flat.values():
        for k in d:
            if k not in keys:
                keys.append(k)
    S = [f"Facts probe over {len(raw)} train tasks ({len(errs)} errors) - distribution of every key of the returned dict (numbers: "
         "min / median / max with the task ids at the extremes, distinct values when few; strings and booleans: counts):"]
    extreme: list[str] = []
    for k in keys[:max_keys]:
        vals = [(tid, d[k]) for tid, d in flat.items() if k in d]
        nums = [(tid, v) for tid, v in vals if isinstance(v, float)]
        if nums and len(nums) == len(vals):
            s = sorted(nums, key=lambda z: z[1])
            med = s[len(s) // 2][1]
            distinct = sorted({round(v, 4) for _, v in nums})
            S.append(f"  {k}: n {len(s)}, min {s[0][1]:.5g} ({s[0][0]}), median {med:.5g}, max {s[-1][1]:.5g} ({s[-1][0]}), "
                     f"{len(distinct)} distinct" + (f" {[float(f'{v:.5g}') for v in distinct]}" if len(distinct) <= 8 else ""))
            extreme += [s[0][0], s[-1][0]]
        else:
            c = collections.Counter(v if isinstance(v, str) else json.dumps(v) for _, v in vals)
            top = ", ".join(f"{kk[:30]!r}: {n}" for kk, n in c.most_common(5))
            S.append(f"  {k}: n {len(vals)}; {top}" + (f"; +{len(c) - 5} more values" if len(c) > 5 else ""))
    if len(keys) > max_keys:
        S.append(f"  (+{len(keys) - max_keys} more keys not shown)")
    for tid, e in errs[:5]:
        S.append(f"  ERROR {tid}: {e}")
    if len(errs) > 5:
        S.append(f"  (+{len(errs) - 5} more errors)")
    summary = "\n".join(S)
    L = [summary, "Per task (dict truncated):"]
    for x in raw[:max_rows]:
        if x.get("error"):
            continue
        L.append(f"  {x['task_id']}: {_compact(x.get('diag') or {}, 160)}")
    if len(raw) > max_rows:
        L.append(f"  (+{len(raw) - max_rows} more tasks)")
    pick: list[str] = []
    for tid in extreme + [x["task_id"] for x in raw]:
        if tid not in pick and tid in tasks_by:
            pick.append(tid)
        if len(pick) >= 4:
            break
    by = {x["task_id"]: x for x in raw}
    parts, caps, manifest = [], [], []
    for tid in pick:
        t, x = tasks_by[tid], by[tid]
        try:
            im = Image.open(t.images[0]).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        parts.append(I.part(im, 1_000_000))
        caps.append(f"task {tid}: original ({im.width}x{im.height}) - an extreme task of the report above")
        manifest.append({"task_id": tid, "kind": "original", "path": str(t.images[0])})
        for d in (x.get("debug") or [])[:2]:
            try:
                dim = Image.open(d["path"]).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            parts.append(I.part(dim, 1_000_000))
            what = (f"harness rendering of the first {d['name'][5:]} call" if d.get("auto") else f"your lib.debug {d['name']!r}")
            caps.append(f"task {tid}: {what} ({dim.width}x{dim.height})")
            manifest.append({"task_id": tid, "kind": "auto" if d.get("auto") else "debug", "name": d["name"], "path": d["path"]})
    L += [f"  [picture {i + 1}] {c}" for i, c in enumerate(caps)]
    L.append("Reply with the next facts probe (if a budget is left), a probe on <= 3 named tasks, or with your `notes` (the verified "
             "facts, short lines) and the candidate JSON.")
    return "\n".join(L), summary, parts, manifest


def feedback_prompt_v2(prev: dict, ev: dict, rows: list[dict], tasks: dict, round_idx: int, best_note: str,
                       ledger: list[dict], forced: bool, ext_left: int = 0,
                       eye: dict | None = None, images: list[str] | None = None,
                       delta: dict | None = None, adopted: bool = False, delta_change: str = "",
                       facts: str | None = None, notes: str | None = None) -> str:
    """ext_left = rounds left in the extension stage (while it is > 0 an accepted instrument is never offered
    'final' - one maze run returned identical code 12 times after its first certified acceptance at 9/40).
    eye = executor rollouts per task (shown per task so the writer aims at the tasks the executor misses);
    images = captions of the pictures attached to this message."""
    h = ev["hypothesis"]
    L = [f"Round {round_idx}. Your current instrument (hypothesis: {prev.get('hypothesis', '')!r}; change type of your last edit: "
         f"{ev.get('change_type', '?')}) was run on the train tasks. Every round is judged in two stages.",
         H.render(h)]
    if h.get("level") == "gain" and ev.get("tier") != "gain" and ev.get("gain_eval"):
        L.append(f"STAGE B - certified tier unavailable (stage A below {h.get('h_min')}); only the GAIN tier was tried. " + gate_text(ev))
    elif h["verdict"] or (h.get("level") == "gain" and ev.get("gain_eval")):
        L.append(f"STAGE B - witnesses and range (declared require {ev['require']}, features {ev['features']}, route key {ev['group_by']}): "
                 f"{ev['rule_str']} -> train {ev['train']['cover']} in range, {ev['train']['right']} right{_gain_str(ev['train'])}; "
                 f"leave-one-out {ev['loo']['cover']} in, {ev['loo']['right']} right, precision "
                 f"{ev['loo']['precision'] if ev['loo']['precision'] is None else round(ev['loo']['precision'], 2)}, "
                 f"rule stability {ev['loo']['rule_stability']:.2f}; searched {ev['searched']} rules. " + gate_text(ev))
        ge = ev.get("gain_eval")
        if ge and ev.get("tier") != "gain":
            st = ge.get("stats") or {}
            L.append(f"  GAIN-tier fit (precision >= 0.5, better than the executor): {ge['rule_str']} -> train {ge['train']['cover']} in range, "
                     f"{ge['train']['right']} right; leave-one-out {ge['loo']['cover']} in, {ge['loo']['right']} right, precision "
                     f"{_f2(ge['loo'].get('precision'))}"
                     + ("" if HIDE_EYE else f", gain over the executor per reading {_f2(st.get('mean'))} (one-sided 95 % lower bound {_f2(st.get('lower'))})")
                     + f" -> {'ACCEPTED' if ge.get('accepted') else 'rejected'}: {_why_str(ge['why'])}.")
        if ev.get("require_dropped"):
            L.append(f"  Ignored `require` atoms {ev['require_dropped']}: hard witnesses must be booleans or strings your code computes; "
                     "numeric thresholds are fitted by the harness from `features`.")
        if ev["hints"]:
            L.append("  Single conditions on your diag that would keep >= 3 readings at >= 95 % precision (what a witness would have to look like; "
                     "a witness must be a principled check, not a threshold that happens to exclude the wrong tasks):")
            L += [f"    - {x}" for x in ev["hints"]]
    else:
        L.append("STAGE B - skipped: the measurement itself does not agree with the answers often enough. Witnesses and thresholds "
                 "cannot fix that; change the measurement.")
    L.append(_exec_line(ev))
    if ev.get("warning"):
        L.append("WARNING - " + ev["warning"])
    L.append(best_note)
    L += delta_lines(delta, adopted, round_idx - 1, delta_change)          # the diff against the best, per task
    L.append("Hypothesis ledger (every candidate so far, including the ones that crashed; never repeat a FAILED hypothesis; "
             "a crashed candidate tells you what NOT to write, e.g. a missing import):")
    L += ledger_lines(ledger)
    if forced:
        prims = sorted({p for e in ledger for p in e["primitives"]})
        if h["verdict"]:
            L.append("FORCED CHANGE OF APPROACH: the measurement passes stage A but no principled witness has certified it for several "
                     f"rounds, and no new primitive was introduced. Bring in a second, independent method ({prims} are used up: use another "
                     "library primitive, e.g. a different representation of the same quantity) whose agreement with the current reading is "
                     "the witness, or split the route so that the failing template is measured differently. Do not tune thresholds.")
        else:
            L.append("FORCED HYPOTHESIS CHANGE: the last rounds only tuned constants or kept the same measurement while stage A kept failing. "
                     f"Propose a DIFFERENT measurement now: a library primitive you have not used ({prims} are used up), or a different "
                     "quantity / region / representation. Do not repeat any hypothesis in the ledger.")
    def _ex(r):
        cs = (eye or {}).get(r["task_id"])
        return f"{sum(c['correct'] for c in cs)}/{len(cs)}" if cs else "-"

    if eye and not HIDE_EYE:
        weak = [r for r in rows if (eye.get(r["task_id"]) and sum(c["correct"] for c in eye[r["task_id"]]) / len(eye[r["task_id"]]) <= 0.25)]
        if weak:
            rd = sum(1 for r in weak if r["correct"] == 1)
            ab = sum(1 for r in weak if r["answer"] is None and not r.get("error"))
            L.append(f"Where an instrument pays: the executor is right on at most 1 of its rollouts on {len(weak)} of {len(rows)} train tasks "
                     f"({', '.join(r['task_id'] for r in weak[:10])}{', …' if len(weak) > 10 else ''}); your instrument reads {rd} of them right "
                     f"and abstains on {ab}. A task the executor already answers on every rollout adds nothing: aim the measurement at the "
                     "ones it misses (the 'executor' column).")
    L += checks_lines(rows)                                                 # lib.check sub-step assertions
    if facts:
        L.append("Facts measured on ALL train tasks by your round-0 probe (harness summary - the generator's constants your code "
                 + ("may rely on; the facts report is fixed for this run - no further probe can be run):\n" if FACTS_ONCE else
                    "may rely on; send another {\"tasks\": \"all\"} probe to verify a new stage on every task):\n") + facts)
    if notes:
        L.append("Your notes (carried verbatim from your latest candidate; rewrite them every round - verified facts, what was "
                 "tried and what it showed, what is still open):\n" + notes)
    L += picture_lines(images)
    L.append("Per-task table (wrong / unparsed / errors first). value | answer (option text) | gt (option text) | ok | executor "
             + ("right/rollouts (not shown in this run) | diag:" if HIDE_EYE else "right/rollouts | diag:"))
    order = sorted(rows, key=lambda r: (r["correct"] == 1, r["answer"] is not None, r["task_id"]))

    def _opt(t, key):
        if key is None:
            return "None"
        if t.is_mc and key in t.options:
            return f"{key} ({t.options[key][:24]})"
        return str(key)[:40]

    for r in order[:40]:
        t = tasks[r["task_id"]]
        ok = "-" if r["correct"] is None else ("Y" if r["correct"] else "N")
        err = f" ERROR: {str(r['error']).splitlines()[0][:120]}" if r.get("error") else ""
        v = r["value"]
        if isinstance(v, float):
            val = f"{v:.4g}"
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            val = "[" + ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v[:12]) + ("…" if len(v) > 12 else "") + "]"
        else:
            val = json.dumps(v, ensure_ascii=False)
        extra = ""
        gv = r.get("gt_value")
        if isinstance(v, list) and gv is not None and v and all(isinstance(x, (int, float)) for x in v):
            k = min(range(len(v)), key=lambda i: abs(v[i] - gv))
            extra = f" | nearest candidate to gt value {gv:g}: index {k} (off by {abs(v[k] - gv):.3g})"
        dbg = f" | saved {', '.join(repr(d['name']) for d in r['debug'][:4])}" if r.get("debug") else ""
        L.append(f"  {t.task_id} | {val[:120]} | {_opt(t, r['answer'])} | {_opt(t, r['gt_key'])} | {ok} | {_ex(r)} | "
                 f"{_compact(r['diag'], 140)}{extra}{dbg}{err}")
    errs = [r for r in rows if r.get("error")]
    if errs:
        L.append("First error traceback:")
        L.append("  " + str(errs[0]["error"])[-600:].replace("\n", "\n  "))
    L.append("Your current code:\n" + prev.get("code", ""))
    if ev.get("model_read") and not ev["accepted"]:
        L.append("Instruction: the value of this candidate is the model's own reply (MODEL-READ): that is the executor answering, not "
                 "an instrument, and it cannot be certified. Write a measurement whose value is COMPUTED from pixels by code "
                 "(the model may at most supply a discrete input such as a phrase, a box, a panel index or a transcription of "
                 "printed text). The allowed pattern on pictures with printed numbers (axes, scales, dials, rulers): the model "
                 "TRANSCRIBES the printed labels with their pixel positions - e.g. lib.ask('Transcribe every number printed along "
                 "the axis / scale as a JSON list of {\"text\": ..., \"x\": ..., \"y\": ...} with pixel coordinates in this image', "
                 "img) - and CODE finds the geometry in pixels (tick marks, bar tops, the needle, the pointer, the line) and converts "
                 "pixels to units with a linear fit through the transcribed labels; witnesses = the fit residual, the number of labels, "
                 "the agreement of two conversions (ticks vs labels). Return the JSON object.")
    elif not h["verdict"] and not ev["accepted"]:
        L.append("Instruction: change the MEASUREMENT (a different primitive or a different quantity), keeping the code minimal; "
                 "witnesses and thresholds are irrelevant until stage A passes. Return the JSON object.")
    elif h.get("systematic") and not ev["accepted"]:
        L.append(f"Instruction: stage A passed but the measurement has a systematic error ({'; '.join(h['systematic'])}). Fix the "
                 "measurement's convention or quantity with ONE change (this is a measurement change, not a witness): e.g. remove the "
                 "candidate family that only produces wrong answers, apply the offset / scale, or change what is counted. Return the JSON object.")
    elif not ev["accepted"] and ev.get("insufficient"):
        L.append("Instruction: the measurement is not rejected, train simply has too few tasks on that route to certify it. "
                 "Extend coverage with ONE change: route one more template that the same kind of measurement can answer; "
                 "do not touch the working route. Return the JSON object.")
    elif not ev["accepted"]:
        L.append("Instruction: keep the measurement; make ONE change: add or replace a principled witness (emitted as a number or boolean "
                 "in diag), declare `require` / `features` / `route_key`, or mark unsupported templates. Do not hand-tune thresholds: "
                 "the harness fits them. Return the JSON object.")
    elif h.get("systematic"):
        # accepted, but stage A still reports a systematic error: the accepted witness is then doing the convention's job
        L.append(f"Instruction: accepted, but the measurement still has a systematic error ({'; '.join(h['systematic'])}): fix the "
                 "convention with ONE change (remove the candidate family / apply the offset) so that the witness no longer has to "
                 "exclude it; coverage will rise. Return the JSON object.")
    elif ev.get("tier") == "gain":
        L.append("Instruction: accepted at the GAIN tier only: in range the instrument beats the executor but is not right often enough "
                 "to be certified. CERTIFY it first with ONE change: the wrong in-range readings are at the top of the table - add a "
                 "principled witness (emitted in diag) that separates them, or fix the measurement's convention, so that in-range "
                 "precision reaches 0.95 WITHOUT abstaining on most tasks (a witness that keeps only a handful of readings trades the "
                 "instrument's coverage for its certificate); extend coverage only after that. " + _no_final(ext_left) + "Return the JSON object.")
    elif h.get("routes_fail"):
        L.append(f"Instruction: accepted on route(s) {h.get('routes_pass')}; route(s) {list(h['routes_fail'])} fail stage A and their "
                 "readings are excluded from the range. Make ONE change on a failing route only (a different measurement for it, or "
                 "declare its templates unsupported); do not touch the passing route(s). " + _no_final(ext_left) + "Return the JSON object.")
    elif ext_left > 0:
        L.append(f"Instruction: accepted. Extend coverage with ONE change - more tasks in range (the rule now covers "
                 f"{ev['train']['cover']} of {ev['n_tasks']} train tasks: look at the abstained / unparsed tasks in the table and make the "
                 "measurement read them without losing precision - e.g. a looser but still principled consistency check, a tolerance "
                 "at the ends, a second threshold tried when the first yields no candidate) or one more template. " + _no_final(ext_left) +
                 "Return the JSON object.")
    else:
        L.append(f"Instruction: accepted. Either extend coverage with ONE change - more tasks in range (the rule now covers "
                 f"{ev['train']['cover']} of {ev['n_tasks']} train tasks: look at the abstained / unparsed tasks in the table and make the "
                 "measurement read them without losing precision) or one more template - or return the same code with "
                 "\"change\": \"final\". Return the JSON object.")
    return "\n".join(L)


def feedback_prompt_scalar(prev: dict, ev: dict, round_idx: int, best_note: str, last: list[str], rounds_left: int,
                           images: list[str] | None, facts: str | None, notes: str | None, fail_note: str = "") -> str:
    """ablation (scalar feedback): coarse execution feedback only - the best candidate's code and its aggregate counts
    (parsed, right, errors, timeouts, model calls, gate verdict without reasons); no per-task table, no stage-A / stage-B
    diagnostics, no hints, no ledger, no delta lists, no pictures of failing tasks (the round-1 pictures are re-attached).
    `last` = one aggregate line per slot of the previous round."""
    L = [f"Round {round_idx}. Your current instrument (hypothesis: {prev.get('hypothesis', '')!r}) was run on the {ev['n_tasks']} train tasks.",
         f"Aggregate result: {ev['parsed']}/{ev['n_tasks']} readings parsed, {ev['parsed_right']} of the parsed right "
         f"(accuracy among parsed {ev['parsed_acc'] if ev['parsed_acc'] is None else round(ev['parsed_acc'], 2)}); "
         f"errors {ev['errors']} (timeouts {ev['timeouts']}); model calls {ev['n_calls']}"
         + ("" if HIDE_EYE or ev.get('eye_mean') is None else f"; executor eyeball pass@1 on these tasks {round(ev['eye_mean'], 2)}") + ".",
         "Gate: " + (("ACCEPTED at the GAIN tier (not certified)" if ev.get("tier") == "gain" else "ACCEPTED (certified)") if ev["accepted"]
                     else "REJECTED") + " - this run reports the verdict only; per-task results and reasons are not shown.",
         best_note]
    if last:
        L.append("Your candidates of the previous round (aggregate only):")
        L += last
    L.append(f"Budget: {rounds_left} round(s) left after this one.")
    if facts:
        L.append("Facts measured on ALL train tasks by your round-0 probe (harness summary):\n" + facts)
    if notes:
        L.append("Your notes (carried verbatim from your latest candidate; rewrite them every round - verified facts, what was "
                 "tried and what it showed, what is still open):\n" + notes)
    if images:
        L.append("Attached pictures, in this order (the same pictures as in round 1):")
        L += [f"  [picture {i + 1}] {c}" for i, c in enumerate(images)]
    L.append("Your current code:\n" + prev.get("code", ""))
    L.append("Make ONE change (or declare the instrument final by returning the same code with \"change\": \"final\"). "
             "Return the JSON object.")
    if fail_note:
        L.append(fail_note)
    return "\n".join(L)


def _no_final(ext_left: int) -> str:
    if ext_left <= 0:
        return ""
    return (f"Do NOT return \"final\": the extension stage has {ext_left} round(s) left, and a change that loses precision is simply "
            "rejected while the current instrument is kept, so there is nothing to lose by trying. ")


PROBE_WRAP = "\n\ndef measure(img, task, lib):\n    return {'value': None, 'diag': probe(img, task, lib)}\n"


def parse_probe(content: str) -> dict | None:
    """{"probe": {"code": "def probe(img, task, lib): ...", "tasks": [...]}} -> the probe dict, else None."""
    obj = I.json_load(content or "")
    if isinstance(obj, dict) and isinstance(obj.get("probe"), dict) and isinstance(obj["probe"].get("code"), str):
        return obj["probe"]
    return None


def run_probe(code: str, sel: list, ids: list[str], wd: Path, a, tasks_by: dict, facts: bool = False) -> tuple[str, list, list[dict], str | None]:
    """Run a probe snippet on the selected train tasks.  Returns (result text, image parts, manifest, facts summary or None).
    facts=True: `sel` is every train task and the reply is the facts report (distributions) instead of per-task dicts."""
    if not sel:
        return (f"Probe not run: none of {ids} is a train task id of this family (use the ids shown in the per-task table). "
                "Reply with a corrected probe request or with the candidate JSON."), [], [], None
    if "def probe" not in code:
        return "Probe rejected: the code must define a top-level `def probe(img, task, lib)` returning a dict. Reply again.", [], [], None
    wrapped = code + PROBE_WRAP
    errs = R.static_check(wrapped, [t.question_text for t in sel], probe=True)
    if errs:
        return "Probe rejected by the static check: " + "; ".join(errs)[:600] + ". Fix it or send the candidate JSON.", [], [], None
    raw, fatal = run_timed(wrapped, sel, wd, a.lib_model or a.model, a.timeout_task)
    if fatal and not raw:
        return "Probe crashed before producing results: " + fatal[-500:] + "\nReply with a corrected probe or the candidate JSON.", [], [], None
    if facts:
        text, summary, parts, manifest = facts_report(raw, tasks_by)
        return text, parts, manifest, summary
    L = [f"Probe results on {len(raw)} task(s) (the dict your probe returned; pictures attached in this order):"]
    parts, caps, manifest = [], [], []
    for x in raw:
        t = tasks_by[x["task_id"]]
        if x.get("error"):
            L.append(f"  {t.task_id}: ERROR {str(x['error']).splitlines()[0][:200]}")
        else:
            L.append(f"  {t.task_id}: {json.dumps(x.get('diag'), ensure_ascii=False)[:900]}")
        try:
            im = Image.open(t.images[0]).convert("RGB")
            parts.append(I.part(im, 1_000_000))
            caps.append(f"task {t.task_id}: original ({im.width}x{im.height})")
            manifest.append({"task_id": t.task_id, "kind": "original", "path": str(t.images[0])})
        except Exception:  # noqa: BLE001
            pass
        for d in (x.get("debug") or [])[:3]:
            try:
                dim = Image.open(d["path"]).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            parts.append(I.part(dim, 1_000_000))
            what = (f"harness rendering of the first {d['name'][5:]} call" if d.get("auto") else f"your lib.debug {d['name']!r}")
            caps.append(f"task {t.task_id}: {what} ({dim.width}x{dim.height})")
            manifest.append({"task_id": t.task_id, "kind": "auto" if d.get("auto") else "debug", "name": d["name"], "path": d["path"]})
    L += [f"  [picture {i + 1}] {c}" for i, c in enumerate(caps)]
    if len(raw) < len(ids):
        L.append(f"  (ids not in this family's train set were skipped: {[i for i in ids if i not in tasks_by]})")
    L.append("Reply with the next probe request or with the candidate JSON.")
    return "\n".join(L), parts, manifest, None


def _is_all(ids) -> bool:
    return ids == "all" or (isinstance(ids, list) and [str(x).lower() for x in ids] == ["all"])


def _dump_convo(convo: list[dict], path: Path) -> None:
    """audit: the messages actually sent (text parts verbatim, images as placeholders)."""
    msgs = []
    for m in convo:
        c = m.get("content")
        if isinstance(c, list):
            c = [x if x.get("type") == "text" else {"type": x.get("type", "image"), "omitted": True} for x in c]
        msgs.append({"role": m.get("role"), "content": c})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(msgs, ensure_ascii=False, indent=1))


def ask_candidate(call, sysp: str, img_parts: list, prompt: str, a, tasks_by: dict, out: Path, r: int, k: int,
                  facts_required: bool = False, facts_budget: int | None = None, dump: bool = False) -> tuple[str, dict]:
    """one candidate slot = a short conversation: the writer may run up to a.probes probes (a probe(img, task, lib)
    snippet on <= 3 named train tasks; the harness answers in the same conversation with the returned dict per task and
    the pictures saved) before it submits the candidate.  up to a.facts FACTS probes ({"tasks": "all"}) per slot,
    answered with the facts report; when facts_required (round 1, first slot) a reply that is not a facts probe is sent
    back once with a reminder.  Returns (the candidate reply, {"probes", "facts", "facts_text"})."""
    convo = I.msgs(img_parts, prompt, sys=sysp)
    n = n_facts = 0
    facts_text = None
    nudged = False
    all_tasks = list(tasks_by.values())
    n_budget = a.facts if facts_budget is None else facts_budget      # ablation --facts-once: 0 after slot r1c0
    n_call = 0
    while True:
        n_call += 1
        if dump:
            _dump_convo(convo, out / "writer_messages" / f"r{r}_c{k}_call{n_call}.json")
        content = call(convo) or ""
        pr = parse_probe(content) if (a.probes > 0 or n_budget > 0) else None
        if pr is None and FACTS_ONCE and n_budget == 0 and a.probes <= 0 and not nudged and parse_probe(content) is not None:
            # ablation: the playbook tells the writer to send facts probes, but after slot r1c0 the report is fixed;
            # a probe request here is answered once with a reminder (no slot burnt), then the next reply is taken as the candidate
            nudged = True
            (out / f"r{r}_c{k}_probe_refused.txt").write_text(content)
            print(f"[loop] r{r} c{k}: probe request in a zero-budget slot (facts-once) -> reminder sent", flush=True)
            convo = convo + [{"role": "assistant", "content": content},
                             {"role": "user", "content": [{"type": "text", "text":
                                 "No probe can be run in this slot: the facts report shown above was measured once for this run and is "
                                 "fixed. Reply with the candidate JSON now (measure() plus its declaration), using the facts you already have."}]}]
            continue
        if pr is None:
            if facts_required and n_facts == 0 and not nudged:
                nudged = True
                (out / f"r{r}_c{k}_nofacts_reply.txt").write_text(content)
                print(f"[loop] r{r} c{k}: round-0 facts probe missing -> reminder sent", flush=True)
                convo = convo + [{"role": "assistant", "content": content},
                                 {"role": "user", "content": [{"type": "text", "text":
                                     "Round 0 comes first: reply with a FACTS probe over ALL train tasks ({\"probe\": {\"code\": "
                                     "\"def probe(img, task, lib): ...\", \"tasks\": \"all\"}}) that measures what your instrument will "
                                     "rely on; the candidate comes after the harness's report."}]}]
                continue
            return content, {"probes": n, "facts": n_facts, "facts_text": facts_text}
        if _is_all(pr.get("tasks")):
            if n_facts >= n_budget:
                (out / f"r{r}_c{k}_facts{n_facts + 1}_request.txt").write_text(content)
                convo = convo + [{"role": "assistant", "content": content},
                                 {"role": "user", "content": [{"type": "text", "text": f"The facts-probe budget of this slot ({n_budget}) is "
                                                                                        "used up. Reply with a probe on <= 3 named tasks or with the candidate JSON now."}]}]
                continue
            n_facts += 1
            (out / f"r{r}_c{k}_facts{n_facts}_request.txt").write_text(content)
            wd = out / f"r{r}_c{k}_facts{n_facts}"
            text, parts, manifest, summary = run_probe(pr.get("code") or "", all_tasks, ["all"], wd, a, tasks_by, facts=True)
            if summary:
                facts_text = ((facts_text + "\n\n") if facts_text else "") + summary     # all facts reports of the slot
            wd.mkdir(parents=True, exist_ok=True)
            (wd / "result.txt").write_text(text)
            (wd / "images.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
            first = text.splitlines()[1] if len(text.splitlines()) > 1 else text
            print(f"[loop] r{r} c{k}: facts probe {n_facts}/{n_budget} on all {len(all_tasks)} tasks -> {first[:140]}", flush=True)
            convo = convo + [{"role": "assistant", "content": content},
                             {"role": "user", "content": [*parts, {"type": "text", "text": text}]}]
            if getattr(a, "facts_gen", False) and n_facts >= n_budget:      # ablation: the pre-stage ends with the report
                if dump:
                    _dump_convo(convo, out / "writer_messages" / f"r{r}_c{k}_call{n_call + 1}.json")
                return "", {"probes": n, "facts": n_facts, "facts_text": facts_text, "gen_done": True}
            continue
        if a.probes <= 0 or n >= a.probes:
            (out / f"r{r}_c{k}_probe{n + 1}_request.txt").write_text(content)
            convo = convo + [{"role": "assistant", "content": content},
                             {"role": "user", "content": [{"type": "text", "text": f"The probe budget of this slot ({a.probes}) is used up. "
                                                                                    "Reply with the candidate JSON now."}]}]
            if dump:
                _dump_convo(convo, out / "writer_messages" / f"r{r}_c{k}_call{n_call + 1}.json")
            return call(convo) or "", {"probes": n, "facts": n_facts, "facts_text": facts_text}
        n += 1
        (out / f"r{r}_c{k}_probe{n}_request.txt").write_text(content)
        ids = [str(x) for x in (pr.get("tasks") or [])][:3]
        sel = [tasks_by[i] for i in ids if i in tasks_by]
        wd = out / f"r{r}_c{k}_probe{n}"
        text, parts, manifest, _ = run_probe(pr.get("code") or "", sel, ids, wd, a, tasks_by)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "result.txt").write_text(text)
        (wd / "images.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        first = text.splitlines()[1] if len(text.splitlines()) > 1 else text
        print(f"[loop] r{r} c{k}: probe {n}/{a.probes} on {ids} -> {first[:140]}", flush=True)
        convo = convo + [{"role": "assistant", "content": content},
                         {"role": "user", "content": [*parts, {"type": "text", "text": text}]}]


def notes_text(cand: dict) -> str | None:
    """the candidate's `notes` (list of lines or a string), bounded."""
    n = cand.get("notes")
    if isinstance(n, list):
        n = "\n".join(f"- {str(x)[:200]}" for x in n[:12] if str(x).strip())
    if not isinstance(n, str) or not n.strip():
        return None
    return n.strip()[:2000]


def parse_candidate(content: str) -> tuple[dict | None, str | None]:
    """(candidate, error)."""
    obj = I.json_load(content)
    if not isinstance(obj, dict):
        return None, "unparseable JSON"
    if not isinstance(obj.get("code"), str) or "def measure" not in obj["code"]:
        return None, "unparseable JSON / no measure()"
    return obj, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, help="family or source:family, e.g. tirbench:rotation_game")
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--lib-model", default=None, help="model behind lib.ask (default: --model)")
    ap.add_argument("--profile", default="auto", choices=["auto", "simple", "solver"],
                    help="frozen recipe: 'simple' = 6 rounds x 3 candidates, stale 1, extension 8, patience 4, no thinking; "
                    "'solver' (deep mode) = 18 rounds x 1 candidate, stale 6, extension 8, patience 4, thinking high; "
                    "'auto' = solver when the family's answers are free-form lists / paths (a whole arrangement), else simple. "
                    "Any of --rounds / --candidates / --stale-rounds / --extend-rounds / --extend-patience / --thinking given explicitly overrides the profile")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--candidates", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-task", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--h-min", type=float, default=H.H_MIN, help="stage A pass threshold on agreement among parsed readings")
    ap.add_argument("--stale-rounds", type=int, default=None, help="rounds without a real hypothesis change (while failing) before a change is forced")
    ap.add_argument("--seed", default=None, help="frozen dir of an accepted instrument: its code + declaration are evaluated as round 0 "
                    "and the model is asked to extend it (coverage extension test without re-discovery)")
    ap.add_argument("--extend-rounds", type=int, default=None, help="on the FIRST acceptance the accepted candidate becomes the seed "
                    "and this many more rounds are granted for coverage extension (forced changes off; only an accepted candidate can replace the best)")
    ap.add_argument("--extend-patience", type=int, default=None, help="stop the extension stage after this many rounds without a coverage gain "
                    "(0 = use the whole budget; gains were observed after 3-5 flat rounds)")
    ap.add_argument("--eye-dirs", nargs="*", default=None,
                    help="default off. Explicit executor TRAIN rollout dirs (the pipeline always passes them)")
    ap.add_argument("--eye", default="auto", choices=["auto", "pro", "flashlite"],
                    help="executor rollouts used as the train reference (gain tier, feedback): auto = flash-lite once finished, else pro")
    ap.add_argument("--thinking", default=None, help="Gemini thinking level for the WRITER's calls only (low / medium / high); "
                    "default = the SDK default (3.x pro: low). lib.ask calls in the runner keep the default")
    ap.add_argument("--thinking-headroom", type=int, default=32768, help="extra output tokens reserved for thoughts when --thinking is set")
    ap.add_argument("--no-auto-render", action="store_true", help="do not auto-render the first mask / labels map per task")
    ap.add_argument("--probes", type=int, default=3, help="probes the writer may run per candidate slot before submitting "
                    "(a probe(img, task, lib) snippet on <= 3 named train tasks; results + pictures come back in the same conversation); 0 = off")
    ap.add_argument("--facts", type=int, default=2, help="FACTS probes ({\"tasks\": \"all\"}: a probe run on EVERY train task, "
                    "answered with the distribution of each returned key) allowed per slot; round 1's first slot must start with one; 0 = off")
    ap.add_argument("--max-rounds", type=int, default=0, help="hard cap on the round count when the extension window is "
                    "re-anchored at every new best (0 = rounds + 2 x extend-rounds)")
    ap.add_argument("--round-images", type=int, default=3, help="originals of this many worst tasks (+ their lib.debug pictures) "
                    "are attached to every round's message (0 = contact sheet only, as before)")
    ap.add_argument("--train-n", type=int, default=0, help="ablation: use only a random subset of N train tasks (0 = all)")
    ap.add_argument("--train-seed", type=int, default=1, help="seed of the --train-n subsample")
    ap.add_argument("--feedback", default="full", choices=["full", "iid", "scalar"],
                    help="ablation (inactive by default = 'full', the v2.9 feedback): 'iid' = every slot is an independent sample "
                         "from the round-1 prompt (no previous code, ledger, notes or results are passed on; the harness still gates and "
                         "selects); 'scalar' = only the best code and its aggregate counts are passed on (no per-task table / diagnostics)")
    ap.add_argument("--hide-rollout-stats", action="store_true",
                    help="ablation: remove every executor rollout number (per-task k/4, pass@1 means, "
                         "train gains, GAIN-tier gain bounds) from the writer's messages; the gate and selection still use them")
    ap.add_argument("--facts-once", action="store_true",
                    help="ablation: only slot r1c0 may run facts probes; its report is inserted verbatim into every later "
                         "slot's prompt (paired arms share one fixed initial facts report)")
    ap.add_argument("--facts-gen", action="store_true",
                    help="ablation (shared facts pre-stage): run only the round-1 facts probe(s) of slot r1c0 (the writer's probe, the "
                         "harness report) and stop; writes <out>/facts.txt (the summary later prompts carry) + shared_facts.json (hash, files)")
    ap.add_argument("--facts-load", default=None,
                    help="ablation: <dir> of a --facts-gen run whose facts.txt is the FIXED facts report of every slot of this run "
                         "(no facts probe is run; facts budget 0 in every slot, probe requests get the facts-once reminder)")
    a = ap.parse_args()
    if a.facts_load:                     # ablation: the report is loaded, never measured in this run
        a.facts_once, a.facts = True, 0
        if not (Path(a.facts_load) / "facts.txt").exists():
            raise SystemExit(f"--facts-load: {a.facts_load}/facts.txt missing")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fam = a.family
    tasks = E.split_tasks("train", {fam})
    if ":" in fam:
        src, f = fam.split(":", 1)
        tasks = [t for t in tasks if t.source == src and t.family == f]
    if not tasks:
        raise SystemExit(f"no train tasks for {fam}")
    if a.train_n and a.train_n < len(tasks):                       # ablation: train-size ablation (inactive by default)
        import random as _rnd
        tasks = sorted(_rnd.Random(a.train_seed).sample(tasks, a.train_n), key=lambda t: t.task_id)
        (out / "train_subset.json").write_text(json.dumps({"n": a.train_n, "seed": a.train_seed, "task_ids": [t.task_id for t in tasks]}, indent=1))
    tasks_by = {t.task_id: t for t in tasks}
    # frozen recipe: profile-based defaults (explicit flags win)
    PROFILES = {"simple": dict(rounds=6, candidates=3, stale_rounds=1, extend_rounds=8, extend_patience=8, thinking=None),   # patience = window
                "solver": dict(rounds=18, candidates=1, stale_rounds=6, extend_rounds=8, extend_patience=4, thinking="high")}
    profile = a.profile
    if profile == "auto":
        n_list = sum(1 for t in tasks if P.answer_type(t) == "free-list")
        profile = "solver" if n_list >= len(tasks) / 2 else "simple"
    a.profile_resolved = profile
    for k, v in PROFILES[profile].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if str(a.thinking).lower() in ("none", "off", "no"):      # explicit "no thinking" under the solver profile (e.g. a flash-lite writer)
        a.thinking = None
    print(f"[loop] profile {profile}{' (auto)' if a.profile == 'auto' else ''}: rounds {a.rounds} x {a.candidates} candidates, stale {a.stale_rounds}, "
          f"extension {a.extend_rounds} (patience {a.extend_patience}), thinking {a.thinking}", flush=True)
    global HIDE_EYE, FACTS_ONCE
    HIDE_EYE = bool(a.hide_rollout_stats)
    FACTS_ONCE = bool(a.facts_once)
    eye_dirs = [str(d) for d in a.eye_dirs] if a.eye_dirs else train_baselines(tasks, a.eye)
    eye = E.load_eyeball(eye_dirs, tasks_by)
    _ms = set()                                     # label = the model that produced the reference rollouts
    for _d in eye_dirs:
        _p = Path(_d) / "results.jsonl"
        if _p.exists():
            with open(_p) as _fh:
                _ms |= {json.loads(_l).get("model") for _l in _fh}
    eye_ref = "+".join(sorted(str(m) for m in _ms)) or "none"
    print(f"[loop] executor reference on train: {eye_ref} ({len(eye)}/{len(tasks_by)} tasks with rollouts)", flush=True)
    if a.feedback != "full" or a.hide_rollout_stats or a.facts_once:
        print(f"[loop] ablation controls: feedback={a.feedback} hide_rollout_stats={a.hide_rollout_stats} facts_once={a.facts_once}", flush=True)
    import hashlib as _hl
    (out / "config.json").write_text(json.dumps({**{k: v for k, v in vars(a).items()}, "profile_resolved": profile, "eye_dirs": eye_dirs,
                                                  "eye_ref": eye_ref, "loop_sha256": _hl.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                  "gate_defaults": G.DEFAULTS, "n_train": len(tasks)}, indent=1, default=str))
    (out / "train_manifest.json").write_text(json.dumps({"family": fam, "split": "train", "task_ids": [t.task_id for t in tasks],
                                                          "eye_ref": eye_ref, "eye_dirs": eye_dirs,
                                                          "rollouts_per_task": {t.task_id: len(eye.get(t.task_id) or []) for t in tasks}}, indent=1))
    prof_dir = out / "profile"
    if (prof_dir / "profile.json").exists():
        prof = json.loads((prof_dir / "profile.json").read_text())
    else:
        prof = P.profile_family(tasks, prof_dir)
    prof_text = P.render(prof)
    (out / "profile.txt").write_text(prof_text)
    lib = Lib()
    sysp = system_prompt(lib, probes=a.probes)
    global PROBE_HINT
    if a.probes > 0:
        PROBE_HINT = (f"Not sure WHY a reading is wrong? Send a probe request first (up to {a.probes} per slot, see # Probes in the "
                      "system prompt): run a snippet on that one task, get its numbers and pictures back, then write the candidate.")
    (out / "system_prompt.txt").write_text(sysp)
    setup_env(a.model, a.model)
    import os
    os.environ["ACQ_AUTO_RENDER"] = "0" if a.no_auto_render else "1"     # read by the runner subprocess
    counter: dict = {}
    call0 = E.make_call(a.model, {"temperature": a.temperature, "top_p": 1.0, "max_tokens": a.max_tokens}, counter, retries=6)

    def call(messages):
        """The writer's call: the thinking level is set around the call only (the runner subprocess, spawned later, inherits
        the environment as it is at spawn time, so lib.ask keeps the SDK default)."""
        import os
        if not a.thinking:
            return call0(messages)
        keep = {k: os.environ.get(k) for k in ("GEMINI_THINKING_LEVEL", "GEMINI_THINKING_HEADROOM", "GEMINI_STREAM")}
        os.environ["GEMINI_THINKING_LEVEL"] = a.thinking
        os.environ["GEMINI_THINKING_HEADROOM"] = str(a.thinking_headroom)      # thoughts eat the output cap (SDK default 4096)
        os.environ["GEMINI_STREAM"] = "1"                                      # streamed, so the server deadline does not cut long thinking
        try:
            return call0(messages)
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    sheet_parts, sheet_caps = [], []
    if prof.get("contact_sheet"):
        sheet_parts = [I.part(Image.open(prof["contact_sheet"]).convert("RGB"))]
        sheet_caps = ["family contact sheet: 256 px thumbnails of the first train tasks, labelled with their task ids"]
    n_img = a.round_images
    imgs, caps, manifest = first_images(tasks, n_img) if n_img > 0 else ([], [], [])
    img_parts = sheet_parts + imgs
    (out / "round1_images.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    log = open(out / "log.jsonl", "a")
    best = None            # (score, round, cand, candidate json, ev, rows)
    best_gain = None       # best GAIN-tier candidate seen (LOO gain, round, cand, train cover, LOO precision) - information only
    prev_prompt = first_prompt(prof_text, images=sheet_caps + caps, facts=(a.facts if not a.seed else 0))
    first_caps = sheet_caps + caps
    dump_msgs = a.feedback != "full" or a.hide_rollout_stats or a.facts_once
    first_accept = None
    n_fatal = 0

    def first_with_facts():
        """ablation: the round-1 prompt with the fixed facts report of slot r1c0 (no facts-probe request)."""
        return first_prompt(prof_text, images=first_caps, facts=0, facts_text=facts_text)

    def scalar_note():
        return f"Best so far: round {best[1]} ({tier_word(best[3])})."
    ledger: list[dict] = []
    facts_text: str | None = None      # the last facts report summary (carried into every later prompt)
    notes: str | None = None           # the writer's own notes (carried verbatim)
    if a.facts_load:                   # ablation: one facts report per (family, replicate), shared by every arm
        facts_text = (Path(a.facts_load) / "facts.txt").read_text()
        prev_prompt = first_with_facts()
        (out / "facts.txt").write_text(facts_text)
        (out / "shared_facts_used.json").write_text(json.dumps({"dir": str(Path(a.facts_load).resolve()),
                                                                 "sha256": _hl.sha256(facts_text.encode()).hexdigest()}, indent=1))
        print(f"[loop] shared facts loaded from {a.facts_load} ({len(facts_text)} chars, sha256 {_hl.sha256(facts_text.encode()).hexdigest()[:16]})", flush=True)
    n_regress = 0
    shown_code = None      # code the model was asked to edit in this round
    stale = 0
    t_start = time.time()
    base_prompt = prev_prompt      # the round prompt without "all candidates failed" notes
    seed = None
    if a.seed:
        sd = Path(a.seed)
        smeta = json.loads((sd / "meta.json").read_text()) if (sd / "meta.json").exists() else {}
        seed = {k: v for k, v in (smeta.get("declared") or {}).items() if v is not None}
        seed.update({k: v for k, v in (smeta.get("division") or {}).items() if v})
        seed["code"] = (sd / "code.py").read_text()
        seed["change"] = f"seed: frozen instrument {sd}"
        seed.setdefault("hypothesis", "")
        print(f"[loop] round 0 = seed instrument {sd} ({smeta.get('key')}, train {smeta.get('gate', {}).get('train')})", flush=True)
    max_round = a.rounds
    ext = {"on": bool(seed), "since": 0 if seed else None, "last_gain": 0 if seed else None, "cover": None}
    r = (0 if seed else 1) - 1
    while True:
        r += 1
        if r > max_round:
            break
        (out / f"round{r}_prompt.txt").write_text(prev_prompt)
        cands, fatal_reasons = [], []
        for k in range(1 if r == 0 else a.candidates):
            pst = {"probes": 0, "facts": 0, "facts_text": None}
            if r == 0:
                content, cand, perr = json.dumps(seed, ensure_ascii=False), dict(seed), None
            else:
                slot_prompt = prev_prompt
                fb = a.facts
                if a.facts_once and not (r == 1 and k == 0):
                    fb = 0
                    if r == 1:
                        slot_prompt = first_with_facts()
                if slot_prompt != prev_prompt:
                    (out / f"r{r}_c{k}_prompt.txt").write_text(slot_prompt)
                content, pst = ask_candidate(call, sysp, img_parts, slot_prompt, a, tasks_by, out, r, k,
                                             facts_required=(a.facts > 0 and r == 1 and k == 0 and seed is None),
                                             facts_budget=fb, dump=dump_msgs)
                if pst.get("gen_done"):                                   # ablation --facts-gen: products + stop
                    ft = pst.get("facts_text") or ""
                    (out / "facts.txt").write_text(ft)
                    files = sorted(str(q.relative_to(out)) for q in out.glob(f"r{r}_c{k}_facts*") if q.is_file()) + \
                            sorted(str(q.relative_to(out)) for q in out.glob(f"r{r}_c{k}_facts*/**/*") if q.is_file())
                    (out / "shared_facts.json").write_text(json.dumps({
                        "family": fam, "n_train": len(tasks), "eye_ref": eye_ref, "n_facts_probes": pst["facts"], "writer_calls": counter.get("n", 0),
                        "sha256": _hl.sha256(ft.encode()).hexdigest(), "chars": len(ft), "files": files, "model": a.model,
                        "writer_in_tokens": counter.get("in_tokens", 0), "writer_out_tokens": counter.get("out_tokens", 0),
                        "writer_think_tokens": counter.get("think_tokens", 0), "elapsed_s": round(time.time() - t_start)}, indent=1))
                    (out / "acquisition_status.json").write_text(json.dumps({"family": fam, "facts_gen": True, "frozen": False, "accepted": False,
                                                                             "writer_calls": counter.get("n", 0), "elapsed_s": round(time.time() - t_start)}, indent=1))
                    print(f"[loop] facts-gen done: {len(ft)} chars, sha256 {_hl.sha256(ft.encode()).hexdigest()[:16]}", flush=True)
                    log.close()
                    return
                if pst.get("facts_text"):
                    # every facts report is kept (the pro smoke's first probe measured n_pieces, the second the canvas: both
                    # must stay in later prompts); the newest last, bounded
                    facts_text = ((facts_text + "\n\n") if facts_text else "") + pst["facts_text"]
                    facts_text = facts_text[-6000:]
                    (out / "facts.txt").write_text(facts_text)
                cand, perr = parse_candidate(content or "")
            (out / f"r{r}_c{k}_response.txt").write_text(content or "")
            rec = {"round": r, "cand": k, "parsed": cand is not None, "api_calls": counter.get("n", 0), "elapsed": round(time.time() - t_start),
                   "probes": pst["probes"], "facts": pst["facts"], "notes": (notes_text(cand) if cand else None)}

            def _fatal(msg, hyp=""):
                rec["error"] = msg
                fatal_reasons.append(msg)
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[loop] r{r} c{k}: {msg[:200]}", flush=True)
                ledger.append({"round": r, "cand": k, "hypothesis": str(hyp)[:160], "primitives": [], "fatal": msg[:300],
                               "parsed": 0, "right": 0, "agreement": 0.0, "verdict": False, "accepted": False, "why": msg[:120],
                               "change_type": "?", "systematic": [], "errors": 0, "error_example": ""})
                (out / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=1))
            if cand is None:
                obj = I.json_load(content or "")
                _fatal(perr or "unparseable JSON / no measure()", obj.get("hypothesis", "") if isinstance(obj, dict) else "")
                n_fatal += 1
                continue
            errs = R.static_check(cand["code"], [t.question_text for t in tasks], has_options=any(t.is_mc for t in tasks))
            if errs:
                _fatal("static check: " + "; ".join(errs), cand.get("hypothesis", ""))
                cands.append((None, cand, {"accepted": False, "why": rec["error"]}, [], k))
                n_fatal += 1
                continue
            wd = out / f"r{r}_c{k}"
            raw, fatal = run_timed(cand["code"], tasks, wd, a.lib_model or a.model, a.timeout_task)
            if fatal and not raw:
                _fatal("runner: " + fatal[-300:], cand.get("hypothesis", ""))
                cands.append((None, cand, {"accepted": False, "why": rec["error"]}, [], k))
                n_fatal += 1
                continue
            values = {x["task_id"]: x.get("value") for x in raw}
            conv, cstats = ("identity", None)
            if any(isinstance(v, list) for v in values.values()) and not tasks[0].is_mc:
                conv, cstats = A.choose_convention(values, tasks_by)
            rows = G.rows_from_raw(raw, tasks_by, conv)
            hyp = H.check(rows, tasks_by, h_min=a.h_min, route_key=cand.get("route_key") or "route")
            ev = G.evaluate(rows, tasks_by, eye, declared=cand, hypothesis=hyp, gain_tier=True)
            ev["convention"], ev["convention_stats"] = conv, cstats
            ev["change_type"] = H.change_type(shown_code, cand["code"])
            ev["_cand"] = k
            ev["primitives"] = H.primitives_used(cand["code"])
            try:
                ev["model_read"] = R.model_read(__import__("ast").parse(cand["code"]))
            except SyntaxError:
                ev["model_read"] = False
            n_err, err_ex = error_summary(rows)
            ev["error_example"] = err_ex
            ev["n_lines"] = n_lines_changed(shown_code, cand["code"])
            ev["delta"] = delta_vs_best(best[4], rows, best[3], ev, tasks_by) if best is not None else None
            if ev["delta"] and ev["delta"]["lost"] > 0:
                n_regress += 1
            err_sig, err_streak, warning = repeated_error(ledger, rows, err_ex) if n_err else ("", 0, "")
            ev["warning"] = warning
            if warning:
                print(f"[loop] r{r} c{k}: WARNING {warning[:200]}", flush=True)
            if hyp is not None:
                ledger.append({"round": r, "cand": k, "hypothesis": str(cand.get("hypothesis", ""))[:160], "primitives": ev["primitives"],
                               "parsed": hyp["parsed"], "right": hyp["right"], "agreement": hyp["agreement"], "verdict": hyp["verdict"],
                               "accepted": ev["accepted"], "why": ev["why"], "change_type": ev["change_type"],
                               "systematic": hyp.get("systematic", []), "errors": n_err, "error_example": err_ex, "n_tasks": len(rows),
                               "model_read": ev.get("model_read", False), "pixels_compute": str(cand.get("pixels_compute", ""))[:160],
                               "model_supplies": str(cand.get("model_supplies", ""))[:120],
                               "error_sig": err_sig, "error_streak": err_streak, "warning": warning,
                               "tier": ev.get("tier"), "verdict_by": hyp.get("verdict_by"), "level": hyp.get("level"),
                               "delta": ev["delta"], "n_lines": ev["n_lines"]})
                (out / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=1))
            with open(wd / "rows.jsonl", "w") as fh:
                for x in rows:
                    fh.write(json.dumps(x, ensure_ascii=False) + "\n")
            (wd / "gate.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1))
            (wd / "candidate.json").write_text(json.dumps(cand, ensure_ascii=False, indent=1))
            sc = G.score(ev)
            if ev.get("tier") == "gain":
                bg = (ev["loo"].get("gain", 0.0), r, k, ev["train"]["cover"], ev["loo"].get("precision"))
                if best_gain is None or bg[0] > best_gain[0]:
                    best_gain = bg
            rec.update({"score": sc, "accepted": ev["accepted"], "tier": ev.get("tier"), "why": ev["why"], "parsed": ev["parsed"], "parsed_right": ev["parsed_right"],
                        "cover": ev["train"]["cover"], "gain": ev["train"].get("gain"), "loo_precision": ev["loo"]["precision"],
                        "rule": ev["rule_str"], "change": cand.get("change", "")[:200], "hypothesis": cand.get("hypothesis", "")[:200],
                        "delta": ev["delta"], "n_lines": ev["n_lines"],
                        "tokens": {"in": counter.get("in_tokens", 0), "out": counter.get("out_tokens", 0), "think": counter.get("think_tokens", 0)},
                        "runner": dict(RUNTIME)})
            log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
            hs = "" if hyp is None else f"A {'pass' if hyp['verdict'] else ('gain-level' if hyp.get('level') == 'gain' else 'FAIL')} {hyp['agreement']:.2f} [{ev['change_type']}{' MODEL-READ' if ev.get('model_read') else ''}] | "
            print(f"[loop] r{r} c{k}: parsed {ev['parsed']}/{ev['n_tasks']} right {ev['parsed_right']} | {hs}rule {ev['rule_str']} | "
                  f"train {ev['train']['cover']} in gain {ev['train'].get('gain', 0):+.1f} | LOO prec {ev['loo']['precision']} | "
                  f"{('ACCEPT[' + str(ev.get('tier')) + ']') if ev['accepted'] else 'reject'}: {ev['why']}", flush=True)
            if ev["accepted"] and first_accept is None:
                first_accept = {"round": r, "cand": k, "slot": (r - 1) * a.candidates + k + 1, "tier": ev.get("tier")}
            cands.append((sc, cand, ev, rows, k))
        scored = [c for c in cands if c[0] is not None]
        if not scored:
            # every candidate of this round was fatal (unparseable / static check / runner crash): the next
            # prompt must still show the ledger (with these entries) and force a change, and name the reasons
            reasons = "; ".join(x[:160] for x in fatal_reasons) or "no candidate parsed"
            fail_note = (f"(All {a.candidates} candidates of round {r} failed before running: {reasons}. "
                         "Return valid JSON with a runnable measure().)")
            if a.feedback == "iid":
                prev_prompt = first_with_facts()          # no memory between slots
                continue
            if a.feedback == "scalar":
                last = [f"  r{r}c{i}: did not run ({x[:120]})" for i, x in enumerate(fatal_reasons)]
                prev_prompt = (feedback_prompt_scalar(best[2], best[3], r + 1, scalar_note(), last, max_round - r, first_caps, facts_text, notes)
                               if best is not None else first_with_facts() + "\n\n" + fail_note)
                continue
            if best is None:
                prev_prompt = base_prompt + "\n\n" + fail_note
            else:
                stale += 1
                forced = stale >= a.stale_rounds
                note = (f"Best so far: round {best[1]} ({tier_word(best[3])}; "
                        f"{best[3]['train']['cover']} in range{_gain_str(best[3]['train'])}).")
                imgs, caps, manifest = round_images(best[4], tasks_by, n_img) if n_img > 0 else ([], [], [])
                img_parts = sheet_parts + imgs
                (out / f"round{r + 1}_images.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
                prev_prompt = feedback_prompt_v2(best[2], best[3], best[4], tasks_by, r + 1, note, ledger, forced,
                                                 ext_left=(max_round - r) if ext["on"] else 0, eye=(None if HIDE_EYE else eye),
                                                 images=sheet_caps + caps, facts=facts_text, notes=notes) + "\n\n" + fail_note
                if forced:
                    stale = 0
            continue
        rb = max(scored, key=lambda c: c[0])
        prev_best_score = best[0] if best is not None else None
        prev_prims = set(best[3].get("primitives", [])) if best is not None else set()
        # among equally scored NON-accepted candidates the most recent one becomes the code the model edits next
        # (otherwise the model would be shown the same early dummy again and again; an accepted best is still only replaced by a better one)
        if best is None or rb[0] > best[0] or (rb[0] == best[0] and not best[3]["accepted"] and not rb[2]["accepted"]):
            best = (rb[0], r, rb[1], rb[2], rb[3])
        stop_ext = False
        if best[3]["accepted"]:
            cov = best[3]["train"]["cover"]
            if not ext["on"] and a.extend_rounds > 0:
                ext.update(on=True, since=r, last_gain=r, cover=cov)
                max_round = max(max_round, r + a.extend_rounds)     # never SHORTEN the budget (deep mode rep3: certified at r3 -> 11 rounds instead of 18)
                print(f"[loop] r{r}: first acceptance ({best[3].get('tier')}, train cover {cov}) -> extension stage: {a.extend_rounds} more rounds "
                      f"(until r{max_round})", flush=True)
            elif ext["on"]:
                if ext["cover"] is None or cov > ext["cover"]:
                    ext["cover"], ext["last_gain"] = cov, r
                elif a.extend_patience and r - ext["last_gain"] >= a.extend_patience:
                    print(f"[loop] r{r}: extension stage: no coverage gain for {a.extend_patience} rounds (best {cov}) -> stop", flush=True)
                    stop_ext = True
            # the window is re-anchored at EVERY new best (one run: GAIN at r4 opened the 8-round window,
            # the certified candidate came at r15 with 3 rounds left), under a hard cap
            if ext["on"] and a.extend_rounds > 0 and best[1] == r and prev_best_score is not None and rb[0] > prev_best_score:
                cap = a.max_rounds if a.max_rounds > 0 else a.rounds + 2 * a.extend_rounds
                new_max = min(r + a.extend_rounds, cap)
                if new_max > max_round:
                    print(f"[loop] r{r}: new best ({tier_word(best[3])}, cover {cov}) -> extension window re-anchored: until r{new_max} "
                          f"(cap {cap})", flush=True)
                    max_round = new_max
        note = (f"Best so far: round {best[1]} ({tier_word(best[3])}; "
                f"{best[3]['train']['cover']} in range{_gain_str(best[3]['train'])}).")
        if a.feedback == "iid":
            prev_prompt = first_with_facts()
        elif a.feedback == "scalar":
            last = [f"  r{r}c{c[4]}: parsed {c[2]['parsed']}/{c[2]['n_tasks']}, right {c[2]['parsed_right']}, errors {c[2]['errors']} -> "
                    f"{'ACCEPTED' if c[2]['accepted'] else 'rejected'}" + ("; adopted as the new best" if (best[1] == r and c is rb) else "")
                    for c in scored]
            last += [f"  (did not run: {x[:120]})" for x in fatal_reasons]
            nt = notes_text(rb[1])
            if nt:
                notes = nt
            prev_prompt = feedback_prompt_scalar(best[2], best[3], r + 1, scalar_note(), last, max_round - r, first_caps, facts_text, notes)
        else:
            # stale = rounds whose best is not accepted and did not change the measurement (constants / identical code, or a
            # failing hypothesis with the same primitives as before)
            rev = rb[2]
            progressed = prev_best_score is not None and rb[0] > prev_best_score
            new_prims = set(rev.get("primitives", [])) - set(prev_prims)
            if rev["accepted"] or (progressed and not rev.get("model_read")) or \
                    (new_prims and rev.get("change_type") == "structural" and not rev.get("model_read")):
                stale = 0
            else:
                stale += 1          # no progress and no new primitive: threshold / witness tinkering
            if ext["on"]:
                stale = 0           # extension stage: the accepted instrument is edited, never forced away from
            forced = stale >= a.stale_rounds
            # the round prompt is built from the best-so-far candidate; a repeated-error warning raised by another candidate
            # of THIS round must still be shown at the top (one maze run: the warned candidates were never the best,
            # so only the ledger lines carried it and flash-lite rewrote `if not boxes` in the next round)
            ev_best = dict(best[3])
            round_warnings = [c[2]["warning"] for c in scored if c[2].get("warning")]
            if round_warnings and not ev_best.get("warning"):
                ev_best["warning"] = round_warnings[-1]
            imgs, caps, manifest = round_images(best[4], tasks_by, n_img) if n_img > 0 else ([], [], [])
            img_parts = sheet_parts + imgs
            (out / f"round{r + 1}_images.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
            nt = notes_text(rb[1])
            if nt:
                notes = nt
            prev_prompt = feedback_prompt_v2(best[2], ev_best, best[4], tasks_by, r + 1, note, ledger, forced,
                                             ext_left=(max_round - r) if ext["on"] else 0, eye=(None if HIDE_EYE else eye), images=sheet_caps + caps,
                                             delta=rb[2].get("delta"), adopted=(best[1] == r), delta_change=str(rb[1].get("change", "")),
                                             facts=facts_text, notes=notes)
            if forced:
                stale = 0
        (out / f"round{r + 1}_prompt.txt").write_text(prev_prompt)
        base_prompt = prev_prompt
        shown_code = None if a.feedback == "iid" else best[2]["code"]
        if stop_ext:
            break
    def _status(extra: dict) -> None:
        (out / "acquisition_status.json").write_text(json.dumps({
            "family": fam, "feedback": a.feedback, "hide_rollout_stats": a.hide_rollout_stats, "facts_once": a.facts_once,
            "rounds_run": min(r, max_round), "candidates_per_round": a.candidates, "n_fatal": n_fatal, "first_accept": first_accept,
            "writer_calls": counter.get("n", 0), "elapsed_s": round(time.time() - t_start), **extra}, indent=1))
    if best is None:
        print("[loop] no runnable candidate", flush=True)
        _status({"accepted": False, "tier": None, "frozen": False, "why": "no runnable candidate"})
        return
    fz = out / "frozen"
    fz.mkdir(exist_ok=True)
    (fz / "code.py").write_text(best[2]["code"])
    (fz / "rule.json").write_text(json.dumps(best[3]["rule"]))
    tag = a.model.replace("gemini-3.1-", "").replace("-preview", "")          # 3.1 keeps the keys (flashlite / pro)
    tag = re.sub(r"[^a-z0-9]", "", tag.replace("gemini-", ""))                # others: 35flashlite, ...
    fam_key = fam.split(":")[-1]
    meta = {"key": f"acq.{fam_key}.{tag}", "families": [fam], "model": a.model, "rounds": a.rounds,
            "profile": a.profile_resolved, "profile_arg": a.profile, "stale_rounds": a.stale_rounds, "extend_patience": a.extend_patience,
            "candidates": a.candidates, "best_round": best[1], "convention": best[3].get("convention", "identity"),
            "n_calls_train": best[3]["n_calls"], "timeout_task": a.timeout_task,
            "gate": {k: best[3].get(k) for k in ("train", "loo", "accepted", "why", "rule_str", "searched", "parsed", "parsed_right", "eye_mean",
                                                 "tier", "certified", "cert_why", "gain_why", "gain_stats", "excluded_routes")},
            "tier": best[3].get("tier"), "eye_ref": eye_ref, "gain_tier": True,
            "best_gain": (dict(zip(("loo_gain", "round", "cand", "train_cover", "loo_precision"), best_gain)) if best_gain else None),
            "declared": {k: best[2].get(k) for k in ("hypothesis", "measurement", "witnesses", "unsupported", "require", "features", "route_key")},
            "api_calls": counter.get("n", 0), "elapsed_s": round(time.time() - t_start),
            "feedback": "hypothesis-v2.9", "h_min": a.h_min, "n_ledger": len(ledger),
            "round_images": n_img, "thinking": a.thinking, "auto_render": not a.no_auto_render, "probes": a.probes,
            "n_probes_run": sum(1 for _ in out.glob("r*_c*_probe*_request.txt")),
            "facts": a.facts, "n_facts_run": sum(1 for _ in out.glob("r*_c*_facts*_request.txt")),
            "facts_load": (str(Path(a.facts_load).resolve()) if a.facts_load else None),
            "max_rounds_cap": (a.max_rounds if a.max_rounds > 0 else a.rounds + 2 * a.extend_rounds),
            "n_regressions": n_regress, "notes": notes,
            "cost": {"writer_calls": counter.get("n", 0), "writer_in_tokens": counter.get("in_tokens", 0),
                     "writer_out_tokens": counter.get("out_tokens", 0), "writer_think_tokens": counter.get("think_tokens", 0),
                     "runner_runs": RUNTIME["runner_runs"], "runner_s": round(RUNTIME["runner_s"], 1), "runner_calls": RUNTIME["runner_calls"],
                     "runner_in_tokens": RUNTIME["runner_in"], "runner_out_tokens": RUNTIME["runner_out"],
                     "elapsed_s": round(time.time() - t_start),
                     "est_usd_writer": est_usd(a.model, counter.get("in_tokens", 0), counter.get("out_tokens", 0), counter.get("think_tokens", 0))[0],
                     "est_usd_runner": est_usd(a.lib_model or a.model, RUNTIME["runner_in"], RUNTIME["runner_out"])[0],
                     "price_note": est_usd(a.model, 0, 0)[1]},
            "hypothesis": best[3].get("hypothesis"), "change_type": best[3].get("change_type"), "primitives": best[3].get("primitives"),
            "model_read": best[3].get("model_read", False),
            "division": {k: best[2].get(k) for k in ("pixels_compute", "model_supplies")},
            "seed": a.seed, "extension": {"extend_rounds": a.extend_rounds, "since_round": ext["since"], "last_round": min(r, max_round),
                                          "max_round": max_round, "final_cover": ext["cover"]},
            "feedback_mode": a.feedback, "hide_rollout_stats": a.hide_rollout_stats, "facts_once": a.facts_once,
            "first_accept": first_accept, "n_fatal": n_fatal, "best_cand": best[3].get("_cand")}
    (fz / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    _status({"accepted": bool(best[3]["accepted"]), "tier": best[3].get("tier"), "frozen": True, "why": best[3]["why"],
             "best_round": best[1], "train": best[3]["train"], "loo": {k: v for k, v in best[3]["loo"].items() if k != "pairs"}})
    lo_print = {k: v for k, v in best[3]["loo"].items() if k != "pairs"}
    print(f"[loop] frozen {meta['key']} from round {best[1]}: {best[3]['rule_str']} | train {best[3]['train']} | LOO {lo_print} | "
          f"{('ACCEPTED [' + str(best[3].get('tier')) + ']') if best[3]['accepted'] else 'NOT accepted'} ({best[3]['why']}) | api calls {meta['api_calls']}", flush=True)
    c = meta["cost"]
    print(f"[loop] cost: writer {c['writer_calls']} calls, {c['writer_in_tokens']} in / {c['writer_out_tokens']} out / {c['writer_think_tokens']} thinking tokens "
          f"~ ${c['est_usd_writer']} (assumed prices); runner {c['runner_runs']} runs {c['runner_s']} s, {c['runner_calls']} lib.ask calls "
          f"~ ${c['est_usd_runner']}; wall-clock {c['elapsed_s']} s", flush=True)


if __name__ == "__main__":
    main()
