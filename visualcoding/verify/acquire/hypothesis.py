"""Stage A of a round: the hypothesis check (playbook procedure, step B).

Before any witness or threshold is considered, does f(m) agree with the train answers?  `check()` reports the
agreement among parsed readings (overall, per route, per question template), a verdict (>= H_MIN with >= MIN_PARSED
parsed), and cheap systematic-error diagnostics that tell the model what kind of change is needed:
constant offset / scale factor (numeric), over- / under-count histogram (integer counts), which candidate index the
answer sits at (candidate lists: exposes a wrong sign / convention family), unparsed templates, string mismatches.
`change_type()` classifies a candidate's edit against the previous best (identical / constants-only / structural)
so the loop can tell threshold tuning from a real hypothesis change.
"""
from __future__ import annotations

import ast
import collections
import json
import re
import statistics

from .. import instruments as I
from .profile import template as _template

H_MIN, MIN_PARSED = 0.6, 3
H_GAIN, MIN_PARSED_GAIN = 0.5, 5      # stage A bar for the GAIN tier (gate.DEFAULTS gain_precision / gain_min_cover)


PIXEL_MODULE_RE = re.compile(r"\b(np|numpy|cv2|scipy|ndimage|ndi|skimage|ski|signal|optimize|morphology|filters|feature)"
                             r"(?:\.[A-Za-z_]\w*)*\.([A-Za-z_]\w*)\s*\(")


def primitives_used(code: str) -> list[str]:
    """Library primitives called, plus the numpy / cv2 / scipy / skimage functions called as `mod:name` - a solver
    written from those modules never introduces a `lib.` primitive, and without this every structural growth of such a
    candidate counted as 'no new primitive' (stale -> forced change)."""
    names = set(re.findall(r"lib\.([A-Za-z_]\w*)\s*\(", code or "")) - {"question", "debug"}
    mod = {"np": "np", "numpy": "np", "cv2": "cv2", "scipy": "scipy", "ndimage": "scipy", "ndi": "scipy", "signal": "scipy",
           "optimize": "scipy", "skimage": "ski", "ski": "ski", "morphology": "ski", "filters": "ski", "feature": "ski"}
    for m, fn in PIXEL_MODULE_RE.findall(code or ""):
        if fn not in ("array", "asarray", "zeros", "ones", "abs", "sum", "mean", "max", "min", "argmax", "argmin", "arange",
                      "sqrt", "float32", "float64", "uint8", "int32", "clip", "round", "where", "concatenate", "stack", "linspace"):
            names.add(f"{mod[m]}:{fn}")
    return sorted(names)


def change_type(prev_code: str | None, code: str) -> str:
    if prev_code is None:
        return "initial"
    if (prev_code or "").strip() == (code or "").strip():
        return "identical"

    def mask(src):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                node.value = 0
        return ast.dump(tree)
    try:
        return "constants-only" if mask(prev_code) == mask(code) else "structural"
    except SyntaxError:
        return "structural"


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check(rows: list[dict], tasks: dict, h_min: float = H_MIN, min_parsed: int = MIN_PARSED, route_key: str = "route",
          h_gain: float = H_GAIN, min_parsed_gain: int = MIN_PARSED_GAIN) -> dict:
    """Verdict = pooled agreement >= h_min with >= min_parsed parsed, OR (per route) any route named in
    diag[route_key] that reaches the same bar on its own readings.  A route that does not is reported in `routes_fail`
    and its readings are excluded from stage B (gate.evaluate), so that a good route is never dragged down by a bad
    one the model added for coverage (symbolic: 114 of 115 extension candidates never tried a second route)."""
    parsed = [r for r in rows if r["answer"] is not None]
    right = [r for r in parsed if r["correct"] == 1]
    agreement = (len(right) / len(parsed)) if parsed else 0.0
    pooled = len(parsed) >= min_parsed and agreement >= h_min
    pooled_gain = len(parsed) >= min_parsed_gain and agreement >= h_gain
    out = {"parsed": len(parsed), "right": len(right), "n": len(rows), "agreement": round(agreement, 3), "verdict": pooled,
           "h_min": h_min, "min_parsed": min_parsed, "route_key": route_key, "h_gain": h_gain, "min_parsed_gain": min_parsed_gain}
    # per route / per template
    by_route = collections.defaultdict(lambda: [0, 0])
    by_tmpl = collections.defaultdict(lambda: [0, 0, 0])      # n, parsed, right
    for r in rows:
        t = tasks[r["task_id"]]
        k = _template(I.strip_question(t))[:70]
        by_tmpl[k][0] += 1
        if r["answer"] is not None:
            by_tmpl[k][1] += 1
            by_tmpl[k][2] += int(r["correct"] == 1)
            rt = (r.get("diag") or {}).get(route_key)
            if isinstance(rt, str):
                by_route[rt][0] += 1
                by_route[rt][1] += int(r["correct"] == 1)
    out["per_route"] = {k: f"{v[1]}/{v[0]}" for k, v in sorted(by_route.items())}
    routes_pass = sorted(k for k, v in by_route.items() if v[0] >= min_parsed and v[1] / v[0] >= h_min)
    routes_fail = {k: f"{v[1]}/{v[0]}" for k, v in sorted(by_route.items()) if k not in routes_pass}
    verdict = pooled or bool(routes_pass)
    # GAIN level: the same test at the gain tier's bar (agreement >= h_gain with >= min_parsed_gain parsed, pooled or
    # per route); a candidate below the 0.6 bar but at this level can only be accepted at the GAIN tier of stage B
    routes_pass_gain = sorted(k for k, v in by_route.items() if v[0] >= min_parsed_gain and v[1] / v[0] >= h_gain)
    routes_fail_gain = {k: f"{v[1]}/{v[0]}" for k, v in sorted(by_route.items()) if k not in routes_pass_gain}
    verdict_gain = pooled_gain or bool(routes_pass_gain)
    level = "certified" if verdict else ("gain" if verdict_gain else None)
    out.update({"verdict": verdict, "verdict_by": ("pooled" if pooled else ("route" if routes_pass else None)),
                "routes_pass": routes_pass, "routes_fail": routes_fail, "verdict_gain": verdict_gain, "level": level,
                "routes_pass_gain": routes_pass_gain, "routes_fail_gain": routes_fail_gain})
    out["per_template"] = [{"template": k, "n": v[0], "parsed": v[1], "right": v[2]} for k, v in
                           sorted(by_tmpl.items(), key=lambda kv: -kv[1][0])[:10]]
    diags, systematic = [], []
    # constant reading (34 of 36 flash-lite maze candidates in a 12-round run read the same fallback option on
    # 40/40 tasks because their consistency check killed every candidate answer; stage A only said "3/40 right")
    if len(parsed) >= min_parsed and not verdict:
        top, n_top = collections.Counter(json.dumps(r["value"], sort_keys=True, ensure_ascii=False) for r in parsed).most_common(1)[0]
        if n_top >= 5 and n_top >= 0.8 * len(parsed):
            diags.append(f"CONSTANT READING: the value is {top[:30]} on {n_top} of {len(parsed)} parsed tasks, so the measurement does "
                         "not discriminate between tasks. Either the consistency check rejects every candidate answer (check its polarity, "
                         "geometry and off-by-one at the ends: a check that fails everywhere is broken, not evidence) or a fallback is read "
                         "where the code should abstain (value None). Fix the check; a constant cannot pass stage A.")
            systematic.append(f"constant reading {top[:20]} on {n_top}/{len(parsed)} parsed")
    # the abstention form of the same failure (48 of 51 flash-lite maze candidates abstained on 40/40 tasks with
    # n_valid = 0 / n_cands = 0 on every task; stage A only said "0 readings parsed" and the model kept the broken check)
    if len(parsed) < min_parsed and len(rows) >= 5:
        keys = collections.defaultdict(collections.Counter)
        for r in rows:
            for k, v in (r.get("diag") or {}).items():
                if isinstance(v, (int, float, bool, str)) and k != "route":
                    keys[k][json.dumps(v)] += 1
        const = [(k, c.most_common(1)[0]) for k, c in keys.items() if c.most_common(1)[0][1] >= 0.9 * len(rows)]
        zero = [(k, vn) for k, vn in const if vn[0] in ("0", "0.0", "false")]
        pick = zero or const
        if pick:
            k, (v, n) = pick[0]
            diags.append(f"CHECK FAILS EVERYWHERE: diag {k} = {v} on {n} of {len(rows)} tasks and no reading is given, so the "
                         "consistency check rejects every candidate answer on every task. A check that never passes is broken (mask "
                         "polarity, geometry, step size, off-by-one at the ends), not selective: fix the check first; abstaining on "
                         "every task certifies nothing.")
            systematic.append(f"check fails everywhere ({k} = {v} on {n}/{len(rows)})")
    # the route-valued form - `route: no_opts` / `no_n` / `bad_grid` on every task with no reading (the line above
    # skips `route` because a single template legitimately has one route)
    if len(parsed) < min_parsed and len(rows) >= 5:
        rc = collections.Counter(str((r.get("diag") or {}).get(route_key)) for r in rows if r["answer"] is None and not r.get("error"))
        if rc:
            rv, rn = rc.most_common(1)[0]
            if rn >= 0.9 * len(rows) and rv not in ("None", "unsupported"):
                diags.append(f"ABSTAINS EVERYWHERE: {route_key} = {rv!r} on {rn} of {len(rows)} tasks and no reading is given. A reason "
                             "that fires on EVERY task is a wrong assumption about the family (options that do not exist, a number "
                             "expected in the question text, a grid size, an input format), not a property of the tasks: re-read the "
                             "family profile (answer type, options, question templates, examples) and remove the assumption.")
                systematic.append(f"abstains everywhere ({route_key} = {rv} on {rn}/{len(rows)})")
    unparsed = [f"[{v[0]}] {k}" for k, v in by_tmpl.items() if v[1] == 0]
    if unparsed:
        diags.append("templates with no reading at all (declare them unsupported or add a route): " + "; ".join(unparsed[:5]))
    # numeric values vs numeric gt
    nums = [(r, float(r["value"]), float(r["gt_value"])) for r in parsed if _num(r["value"]) and r.get("gt_value") is not None]
    if len(nums) >= 3:
        errs = [v - g for _, v, g in nums]
        exact = sum(1 for e in errs if abs(e) < 1e-6)
        over = sum(1 for e in errs if e > 1e-6)
        under = sum(1 for e in errs if e < -1e-6)
        med = statistics.median(errs)
        mad = statistics.median(abs(e - med) for e in errs)
        if exact < len(nums):
            if abs(med) > 1e-6 and mad <= max(0.5, 0.05 * abs(med)):
                diags.append(f"constant offset: value - gt = {med:+g} in most parsed tasks (median; spread {mad:g}) -> the quantity is right up to a fixed offset")
                systematic.append(f"constant offset {med:+g}")
            ratios = [g / v for _, v, g in nums if abs(v) > 1e-9 and abs(g) > 1e-9]
            if len(ratios) >= 3:
                rm = statistics.median(ratios)
                rmad = statistics.median(abs(x - rm) for x in ratios)
                if abs(rm - 1) > 0.02 and rmad <= 0.05 * abs(rm):
                    diags.append(f"scale factor: gt / value = {rm:.3g} consistently -> the quantity is right up to a scale")
                    systematic.append(f"scale factor {rm:.3g}")
            if all(float(v).is_integer() and float(g).is_integer() for _, v, g in nums):
                hist = collections.Counter(int(round(e)) for e in errs)
                items = ", ".join(f"{'exact' if k == 0 else ('over by ' + str(k) if k > 0 else 'under by ' + str(-k))}: {c}"
                                  for k, c in sorted(hist.items(), key=lambda kv: -kv[1])[:6])
                diags.append(f"count errors: {items} (over {over}, under {under}, exact {exact} of {len(nums)})")
                if exact < len(nums) and (over == 0 or under == 0) and (over + under) >= 2:
                    systematic.append(f"all {over + under} wrong counts are {'over' if over else 'under'}-counts")
            else:
                # generic difference histogram: a reading that is right up to one of a few fixed offsets or a sign
                # flip shows up as a small set of gt - value differences (e.g. {0, 90, 180, 270} for an orientation)
                hist = collections.Counter(int(round(g - v)) for _, v, g in nums)
                top = sorted(hist.items(), key=lambda kv: -kv[1])[:6]
                if sum(c for _, c in top) >= 0.6 * len(nums) and len(hist) <= 8:
                    diags.append("gt - value (rounded) takes only a few values: " + ", ".join(f"{k:+d}: {c}" for k, c in top) +
                                 f" of {len(nums)} -> the quantity is right up to a fixed set of offsets / a sign; return the candidate list")
                negs = [g for _, v, g in nums if abs(g + v) < 1e-6 or abs((g + v) % 360) < 1e-6 and g != v]
                if len(negs) >= 2 and len(negs) >= 0.3 * len(nums):
                    diags.append(f"{len(negs)} of {len(nums)} readings match the answer with the OPPOSITE sign (gt = -value, possibly mod 360)")
    # candidate lists vs gt value
    cands = [(r, [float(x) for x in r["value"]], float(r["gt_value"])) for r in parsed
             if isinstance(r["value"], list) and r["value"] and all(_num(x) for x in r["value"]) and r.get("gt_value") is not None]
    if len(cands) >= 3:
        idx_right, idx_wrong, idx_taken, offs = collections.Counter(), collections.Counter(), collections.Counter(), []
        for r, vs, g in cands:
            k = min(range(len(vs)), key=lambda i: abs(vs[i] - g))
            (idx_right if r["correct"] == 1 else idx_wrong)[k] += 1
            offs.append(abs(vs[k] - g))
            if r["correct"] != 1:
                t = tasks[r["task_id"]]
                on = t.options_numeric if t.is_mc else None
                if on and r["answer"] in on:
                    cv = on[r["answer"]]
                    idx_taken[min(range(len(vs)), key=lambda i: abs(vs[i] - cv))] += 1
        n_c = max(len(vs) for _, vs, _ in cands)
        lens = sorted({len(vs) for _, vs, _ in cands})
        if len(lens) > 1:
            diags.append(f"candidate lists have varying length ({lens[0]}-{lens[-1]} entries, e.g. after set() / dedup): index-based "
                         "diagnosis below is unreliable; build the list in a FIXED order (one entry per convention x symmetry) so that "
                         "index i always means the same candidate family")
        never = sorted(set(range(n_c)) - set(idx_right) - set(idx_wrong))
        steal = sorted(set(idx_taken) - set(idx_right))
        msg = (f"candidate lists: in right tasks the answer sits at candidate index {dict(sorted(idx_right.items()))}; "
               f"in wrong tasks the candidate nearest to the gt value is at index {dict(sorted(idx_wrong.items()))} "
               f"(median distance {statistics.median(offs):.3g}) while the wrong answer was produced by candidate index {dict(sorted(idx_taken.items()))}")
        if never:
            msg += f"; indices {never} never match anything"
        if steal:
            msg += (f"; indices {steal} produce wrong answers and never right ones -> they belong to a wrong convention (a wrong sign or "
                    "offset family): remove them from the list")
            systematic.append(f"candidate indices {steal} only ever produce wrong answers")
        diags.append(msg)
    # string readings
    strs = [r for r in parsed if isinstance(r["value"], str) and r["correct"] == 0]
    if strs:
        ex = "; ".join(f"{r['task_id'].split('_')[-1]}: read {r['value'][:30]!r} vs gt {str(r['gt_key'])[:30]!r}" for r in strs[:3])
        diags.append(f"string mismatches ({len(strs)}): {ex}")
    out["diagnostics"] = diags
    out["systematic"] = systematic
    if pooled:
        why = "ok"
    elif routes_pass:
        why = (f"ok via route(s) {', '.join(routes_pass)}; failing route(s) " +
               ", ".join(f"{k} {v}" for k, v in routes_fail.items()))
    elif len(parsed) < min_parsed:
        why = f"only {len(parsed)} readings parsed (need >= {min_parsed})"
    else:
        why = f"agreement {agreement:.2f} < {h_min}"
        if by_route:
            why += f"; no route reaches {h_min} with >= {min_parsed} parsed"
    if level == "gain":
        why += (f" - but at the GAIN level (agreement >= {h_gain} with >= {min_parsed_gain} parsed" +
                (f", via route(s) {', '.join(routes_pass_gain)}" if not pooled_gain else "") +
                "): only the GAIN tier of stage B is available")
    out["why"] = why
    return out


def render(h: dict) -> str:
    L = [f"STAGE A - hypothesis check: {h['parsed']}/{h['n']} readings parsed; f(m) agrees with the train answer in "
         f"{h['right']} of {h['parsed']} parsed (agreement {h['agreement']:.2f}) -> {'PASSED' if h['verdict'] else 'FAILED'} ({h['why']}; "
         f"pass = agreement >= {h['h_min']} with >= {h['min_parsed']} parsed)."]
    if h["per_route"]:
        L.append("  per route (right/parsed): " + ", ".join(f"{k} {v}" for k, v in h["per_route"].items()))
        if h.get("routes_pass") is not None:
            L.append(f"  stage A is judged PER ROUTE (diag {h.get('route_key', 'route')}): passed {h['routes_pass'] or 'none'}; "
                     f"failed {h.get('routes_fail') or 'none'}" +
                     (" - the readings of a failed route are excluded from stage B: fix that route's measurement or declare its "
                      "templates unsupported; the passing routes are not affected." if h.get("routes_fail") else "."))
    if h.get("level") == "gain":
        L.append(f"  GAIN level: the measurement agrees with the answers >= {h.get('h_gain')} of the time on >= {h.get('min_parsed_gain')} "
                 "readings, so stage B may still accept it at the GAIN tier (in range: better than the executor, not certified); "
                 "the certified tier needs the measurement fixed first.")
    L.append("  per question template (n | parsed | right): " + "; ".join(f"[{t['n']}|{t['parsed']}|{t['right']}] {t['template'][:60]}" for t in h["per_template"][:6]))
    for d in h["diagnostics"]:
        L.append("  - " + d)
    if h.get("systematic"):
        L.append("  SYSTEMATIC ERROR in the measurement itself: " + "; ".join(h["systematic"]) +
                 " -> this is fixed by changing the measurement's convention / quantity, not by a witness.")
    return "\n".join(L)
