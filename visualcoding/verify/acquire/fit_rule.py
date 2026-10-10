"""Range-rule fitter for instrument readings.

Given the stored TRAIN readings of one instrument (results.jsonl rows: value / answer / correct / diag), search a
small, fixed rule space over the witnesses in `diag` for the rule that keeps the most readings in range subject to
the in-range readings being right at least `min_precision` of the time.  The size of the search is reported, together
with a leave-one-task-out (LOO) estimate, so that a rule that only fits the train tasks is visible as such.

Rule space: a conjunction of at most `max_atoms` atoms over the flattened features
    numeric  f >= t | f <= t     t = midpoints between consecutive distinct observed values (<= 30 per feature)
    boolean  f == True | False
    string   f == v | f != v     (<= 10 observed values per feature; a missing feature satisfies no atom)
plus the implicit atom "answer parsed" (a reading without an answer is never in range).  Features = diag flattened
(nested dicts with dots, lists as `.len`), the numeric value itself, and for numeric MC readings the distance to
the nearest option (`opt_d1`) and the margin to the second nearest (`opt_margin`, `opt_margin_rel`).
Among feasible rules: max coverage, then fewer atoms, then higher precision, then the larger threshold margin
(gap between the neighbouring observed values, relative to the feature range), then a fixed order (deterministic).

Structure the model can declare (playbook §7): `require` = hard witnesses applied before the search (a witness
that never fails on train cannot be learned from train, e.g. "exactly one candidate"); `group_by` = the routing
feature (one rule per route, e.g. symbolic: nested / grid / shapes / orange); `features` = the declared witnesses
the search may use (`value` itself is excluded unless allowed: a threshold on the reading is a regime check the
model must declare, not something to be found by coverage maximisation).

CLI (train readings only; never point this at test):
    python -m verify.acquire.fit_rule --results output/verify/<run>/train/results.jsonl --instrument <key>
    python -m verify.acquire.fit_rule --results ... --all [--min-precision 0.95] [--max-atoms 2] [--out-dir rules/]
Library:
    load_records(path, key) -> records; fit(records, ...) -> dict; loo(records, ...) -> dict
    apply_rule(rule, feats) -> bool; range_from_row(rule, row, task) -> bool   (drop-in for Instrument.range_from_row)
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

import numpy as np

MAX_THR, MAX_STR, MAX_DEPTH, MAX_STRLEN = 30, 10, 3, 40


# ----------------------------------------------------------------------------- features
def _flatten(d: dict, prefix: str = "", depth: int = 0, out: dict | None = None) -> dict:
    out = {} if out is None else out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = v
        elif isinstance(v, (int, float)):
            if math.isfinite(v):
                out[key] = float(v)
        elif isinstance(v, str):
            if 0 < len(v) <= MAX_STRLEN:
                out[key] = v
        elif isinstance(v, (list, tuple)):
            out[f"{key}.len"] = float(len(v))
        elif isinstance(v, dict) and depth < MAX_DEPTH:
            _flatten(v, f"{key}.", depth + 1, out)
    return out


def features_from_row(row: dict, task=None) -> dict:
    """Witness features of one stored reading: flattened diag + value + option-margin features (MC numeric)."""
    f = _flatten(row.get("diag") or {})
    f["has_answer"] = row.get("answer") is not None
    val = row.get("value")
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
        f["value"] = float(val)
        opts = task.options_numeric if (task is not None and task.is_mc) else None
        if opts and len(opts) >= 2:
            d = sorted(abs(o - val) for o in opts.values())
            f["opt_d1"] = d[0]
            f["opt_margin"] = d[1] - d[0]
            f["opt_margin_rel"] = (d[1] - d[0]) / max(abs(val), 1e-9)
    elif isinstance(val, (list, tuple)):
        f["value.len"] = float(len(val))
        opts = task.options_numeric if (task is not None and task.is_mc) else None
        nums = [float(x) for x in val if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if opts and len(opts) >= 2 and nums and len(nums) == len(val):
            d = sorted(min(abs(o - x) for x in nums) for o in opts.values())
            f["opt_d1"] = d[0]
            f["opt_margin"] = d[1] - d[0]
    return f


# ----------------------------------------------------------------------------- atoms
def _allowed(k: str, features, allow_value: bool, allow_opt: bool = False) -> bool:
    if k == "has_answer":
        return False
    if features is not None:
        return k in features
    if k == "value":
        return allow_value
    if k.startswith("opt_"):
        return allow_opt
    return True


def _atoms(records: list[dict], features=None, allow_value: bool = False, allow_opt: bool = False) -> list[tuple]:
    """Atoms (feature, op, threshold, margin); margin = robustness of a numeric threshold (gap between the two
    observed values it separates, relative to the feature range), 1.0 for boolean / string atoms."""
    numeric, boolean, string = collections.defaultdict(list), set(), collections.defaultdict(set)
    for rec in records:
        for k, v in rec["feats"].items():
            if not _allowed(k, features, allow_value, allow_opt):
                continue
            if isinstance(v, bool):
                boolean.add(k)
            elif isinstance(v, float):
                numeric[k].append(v)
            elif isinstance(v, str):
                string[k].add(v)
    atoms = []
    for k in sorted(numeric):
        u = sorted(set(numeric[k]))
        if len(u) < 2:
            continue
        rng = max(u[-1] - u[0], 1e-12)
        mids = [((a + b) / 2, (b - a) / rng) for a, b in zip(u, u[1:])]
        if len(mids) > MAX_THR:
            idx = np.linspace(0, len(mids) - 1, MAX_THR).round().astype(int)
            mids = [mids[i] for i in sorted(set(idx))]
        for t, mg in mids:
            atoms.append((k, ">=", t, mg))
            atoms.append((k, "<=", t, mg))
    for k in sorted(boolean):
        atoms.append((k, "==", True, 1.0))
        atoms.append((k, "==", False, 1.0))
    for k in sorted(string):
        vs = sorted(string[k])
        if len(vs) > MAX_STR:
            continue
        for v in vs:
            atoms.append((k, "==", v, 1.0))
            atoms.append((k, "!=", v, 1.0))
    return atoms


def _atom_true(atom, feats: dict) -> bool:
    k, op, t = atom[0], atom[1], atom[2]
    if k not in feats:
        return False
    v = feats[k]
    if op == ">=":
        return isinstance(v, float) and v >= t
    if op == "<=":
        return isinstance(v, float) and v <= t
    if op == "==":
        return v == t
    if op == "!=":
        return v != t
    raise ValueError(op)


def apply_rule(rule, feats: dict) -> bool:
    """rule = {"require": [[feature, op, value], ...], "group_by": <feature or None>, "rules": {<group>: [atoms]}}
    (a plain list of atoms is accepted as a single-group rule).  The implicit 'answer parsed' atom is checked via
    feats['has_answer']; a group without a rule is out of range."""
    if not feats.get("has_answer", False):
        return False
    if rule is None:
        return False
    if isinstance(rule, list):
        rule = {"require": [], "group_by": None, "rules": {"*": rule}}
    if not all(_atom_true(a, feats) for a in rule.get("require", [])):
        return False
    g = feats.get(rule["group_by"]) if rule.get("group_by") else "*"
    atoms = rule["rules"].get(str(g))        # a missing routing feature is the group "None" (frozen rules without it: out of range)
    if atoms is None:
        return False
    return all(_atom_true(a, feats) for a in atoms)


def parse_atom(text: str):
    """'agree==True' | 'hits.len<=1' | 'route!=none' -> atom tuple."""
    for op in ("<=", ">=", "==", "!="):
        if op in text:
            k, v = text.split(op, 1)
            k, v = k.strip(), v.strip()
            if v in ("True", "False"):
                return (k, op, v == "True", 1.0)
            try:
                return (k, op, float(v), 1.0)
            except ValueError:
                return (k, op, v, 1.0)
    raise ValueError(f"bad atom {text!r}")


def range_from_row(rule: list, row: dict, task=None) -> bool:
    return apply_rule(rule, features_from_row(row, task))


# ----------------------------------------------------------------------------- search
def _counts(sel: np.ndarray, right: np.ndarray):
    cover = int(sel.sum())
    r = int((sel & right).sum())
    return cover, r, cover - r


def _feasible(cover, r, min_precision, min_cover, max_wrong):
    if cover < min_cover:
        return False
    if max_wrong is not None and cover - r > max_wrong:
        return False
    return r / cover >= min_precision


def fit_flat(records: list[dict], min_precision: float = 0.95, min_cover: int = 3, max_atoms: int = 2,
             max_wrong: int | None = None, features=None, allow_value: bool = False, allow_opt: bool = False,
             objective: str = "cover", min_margin: float = 0.0) -> dict:
    """Best atom list on `records` (each: feats, correct in {0,1,None}, has_answer, eye = executor pass@1 or None).
    correct None with an answer counts as wrong (conservative).  objective 'cover' = most readings in range;
    'gain' = largest sum over in-range readings of (correct - eye), i.e. the train policy gain over the fallback
    (readings without an eyeball reference contribute 0).  Both under the precision constraint.
    Returns atoms + counts + search size; atoms None if nothing is feasible."""
    n = len(records)
    has = np.array([bool(r["has_answer"]) for r in records], dtype=bool).reshape(n)
    right = np.array([r["correct"] == 1 for r in records], dtype=bool).reshape(n) & has
    eyev = np.array([(r.get("eye") if r.get("eye") is not None else float("nan")) for r in records], dtype=float)
    gvec = np.where(np.isnan(eyev), 0.0, right.astype(float) - np.nan_to_num(eyev))
    atoms = [a for a in _atoms(records, features, allow_value, allow_opt) if a[3] >= min_margin]
    A = np.array([[_atom_true(a, r["feats"]) for r in records] for a in atoms], dtype=bool).reshape(len(atoms), n) & has
    # dedupe atoms with identical masks (keep the first in the deterministic order)
    keep, seen = [], set()
    for i in range(len(atoms)):
        key = A[i].tobytes()
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    atoms = [atoms[i] for i in keep]
    A = A[keep] if keep else np.zeros((0, n), dtype=bool)
    m = len(atoms)
    marg = np.array([a[3] for a in atoms], dtype=float)
    searched = 1
    best = None          # key = (cover, -n_atoms, precision, margin, -order)

    def consider(sel, rule, order, margin):
        nonlocal best
        cover, r, w = _counts(sel, right)
        if not _feasible(cover, r, min_precision, min_cover, max_wrong):
            return
        g = float((gvec * sel).sum())
        key = ((g, cover) if objective == "gain" else (cover, g), -len(rule), r / cover, margin, -order)
        if best is None or key > best[0]:
            best = (key, rule, (cover, r, w, g))

    consider(has, [], 0, 1.0)
    if max_atoms >= 1:
        for i in range(m):
            consider(A[i], [atoms[i]], i + 1, float(marg[i]))
        searched += m
    if max_atoms >= 2 and m >= 2:
        order = m + 1
        for i in range(m - 1):
            S = A[i] & A[i + 1:]
            covers = S.sum(1)
            rights = (S & right).sum(1)
            gains = (S * gvec).sum(1)
            for j_off in np.nonzero(covers >= min_cover)[0]:
                cover, r = int(covers[j_off]), int(rights[j_off])
                if max_wrong is not None and cover - r > max_wrong:
                    continue
                if r / cover < min_precision:
                    continue
                j = i + 1 + j_off
                g = float(gains[j_off])
                key = ((g, cover) if objective == "gain" else (cover, g), -2, r / cover, float(min(marg[i], marg[j])),
                       -(order + j_off))
                if best is None or key > best[0]:
                    best = (key, [atoms[i], atoms[j]], (cover, r, cover - r, g))
            order += m - 1 - i
        searched += m * (m - 1) // 2
    out = {"n_rows": n, "n_atoms": m, "searched": searched}
    if best is None:
        out.update({"atoms": None, "train": {"cover": 0, "right": 0, "wrong": 0, "precision": None, "gain": 0.0}})
    else:
        cover, r, w, g = best[2]
        out.update({"atoms": [list(a) for a in best[1]],
                    "train": {"cover": cover, "right": r, "wrong": w, "precision": r / cover, "gain": g}})
    return out


def fit(records: list[dict], min_precision: float = 0.95, min_cover: int = 3, max_atoms: int = 2,
        max_wrong: int | None = None, features=None, allow_value: bool = False, group_by: str | None = None,
        require: list | None = None, allow_opt: bool = False, objective: str = "cover",
        min_margin: float = 0.0) -> dict:
    """Rule object for `records`: hard `require` atoms first, then one atom list per `group_by` value (or one
    overall).  A group with no feasible list is out of range.  Counts are pooled over groups."""
    require = [tuple(a) if len(a) == 4 else (a[0], a[1], a[2], 1.0) for a in (require or [])]
    kept = [r for r in records if all(_atom_true(a, r["feats"]) for a in require)] if require else list(records)
    groups = collections.defaultdict(list)
    for r in kept:
        g = r["feats"].get(group_by) if group_by else "*"
        if g is None:
            g = "None"      # readings without the routing feature form their own group (a `route` set only on
                            # abstentions, e.g. "ambiguous", must not drop every actual reading from the fit)
        groups[str(g)].append(r)
    rules, searched, n_atoms = {}, 0, 0
    cover = right = 0
    gain = 0.0
    for g in sorted(groups):
        f = fit_flat(groups[g], min_precision, min_cover, max_atoms, max_wrong, features, allow_value, allow_opt,
                     objective, min_margin)
        searched += f["searched"]
        n_atoms += f["n_atoms"]
        if f["atoms"] is not None:
            rules[g] = f["atoms"]
            cover += f["train"]["cover"]
            right += f["train"]["right"]
            gain += f["train"]["gain"]
    rule = {"require": [list(a) for a in require], "group_by": group_by, "rules": rules} if rules else None
    return {"n_rows": len(records), "n_atoms": n_atoms, "searched": searched, "min_precision": min_precision,
            "min_cover": min_cover, "max_atoms": max_atoms, "objective": objective, "rule": rule,
            "train": {"cover": cover, "right": right, "wrong": cover - right,
                      "precision": (right / cover) if cover else None, "gain": gain}}


def loo(records: list[dict], **kw) -> dict:
    """Leave-one-task-out: fit on the other tasks, apply to the held-out task's rows.  Reports pooled counts and the
    share of folds whose rule equals the full-data rule (rule stability)."""
    full = fit(records, **kw)
    by_task = collections.defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
    cover = right = 0
    gain = 0.0
    same = 0
    pairs = []          # (instrument correct, executor pass@1 or None) per LOO in-range reading (gain tier)
    for tid, held in by_task.items():
        rest = [r for r in records if r["task_id"] != tid]
        f = fit(rest, **kw)
        if f["rule"] is not None and f["rule"] == full["rule"]:
            same += 1
        rule = f["rule"]
        if rule is None:
            continue
        for r in held:
            if apply_rule(rule, r["feats"]):
                cover += 1
                right += int(r["correct"] == 1)
                if r.get("eye") is not None:
                    gain += int(r["correct"] == 1) - r["eye"]
                pairs.append([int(r["correct"] == 1), r.get("eye")])
    return {"cover": cover, "right": right, "wrong": cover - right, "precision": (right / cover) if cover else None,
            "gain": gain, "n_tasks": len(by_task), "rule_stability": same / max(len(by_task), 1), "pairs": pairs}


# ----------------------------------------------------------------------------- data
def _dedupe(path: Path, key: str) -> list[dict]:
    seen, rows = set(), []
    for line in open(path):
        r = json.loads(line)
        if r["instrument"] != key:
            continue
        k = (r["task_id"], r["sample"])
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
    return rows


def load_records(results: str | Path, key: str, tasks: dict | None = None, baseline: list[str] | None = None) -> list[dict]:
    """Records for one instrument from a results.jsonl (train!).  Correctness = the stored row-level flag, except
    free-form tasks with a judge_cache.json next to the file (judge verdict wins)."""
    from .. import instrument_eval as E
    from .. import instruments as I
    path = Path(results)
    rows = _dedupe(path, key)
    if not rows:
        return []
    inst = I.INSTRUMENTS[key]
    if tasks is None:
        tasks = {t.task_id: t for t in E.split_tasks(rows[0]["split"], set(inst.families))}
    jpath = path.parent / "judge_cache.json"
    jcache = json.loads(jpath.read_text()) if jpath.exists() else {}
    eye = E.load_eyeball(baseline, tasks) if baseline else {}
    recs = []
    for r in rows:
        t = tasks.get(r["task_id"])
        if t is None:
            continue
        ans, corr = r.get("answer"), r.get("correct")
        if ans is not None and not t.is_mc:
            jk = f"{t.task_id}\t{ans}"
            if jk in jcache:
                corr = int(jcache[jk]["score"] >= 0.5)
        cs = eye.get(r["task_id"])
        recs.append({"task_id": r["task_id"], "sample": r["sample"], "feats": features_from_row(r, t),
                     "correct": corr, "has_answer": ans is not None, "stored_in_range": bool(r.get("in_range")),
                     "eye": (sum(c["correct"] for c in cs) / len(cs)) if cs else None})
    return recs


def _sel_counts(sel: list[dict]) -> dict:
    right = sum(1 for r in sel if r["correct"] == 1)
    gain = sum((int(r["correct"] == 1) - r["eye"]) for r in sel if r.get("eye") is not None)
    return {"cover": len(sel), "right": right, "wrong": len(sel) - right,
            "precision": (right / len(sel)) if sel else None, "gain": gain}


def stored_counts(records: list[dict]) -> dict:
    return _sel_counts([r for r in records if r["stored_in_range"] and r["has_answer"]])


def _atoms_str(atoms) -> str:
    if not atoms:
        return "answer parsed"
    return " and ".join(f"{a[0]} {a[1]} {a[2]:.4g}" if isinstance(a[2], float) else f"{a[0]} {a[1]} {a[2]!r}" for a in atoms)


def rule_str(rule) -> str:
    if rule is None:
        return "(none feasible)"
    parts = []
    if rule.get("require"):
        parts.append("require " + _atoms_str(rule["require"]))
    if rule.get("group_by"):
        parts.append("; ".join(f"{rule['group_by']}={g}: {_atoms_str(a)}" for g, a in rule["rules"].items()))
    else:
        parts.append(_atoms_str(rule["rules"].get("*")))
    return " | ".join(parts)


def _fmt(c: dict) -> str:
    p = c.get("precision")
    g = f" gain {c['gain']:+5.1f}" if c.get("gain") is not None else ""
    base = f"{c['cover']:4d} in, {c['right']:3d} right, {c['wrong']:2d} wrong, "
    return base + (f"prec {p:.3f}" if p is not None else "prec  -   ") + g


def _mean_eye(recs):
    xs = [r["eye"] for r in recs if r.get("eye") is not None]
    return (sum(xs) / len(xs)) if xs else None


def report_one(results, key, min_precision, min_cover, max_atoms, max_wrong, do_loo=True, group_by="auto",
               require=None, features=None, allow_value=False, allow_opt=False, objective="cover",
               baseline=None, min_margin=0.0) -> dict:
    recs = load_records(results, key, baseline=baseline)
    if not recs:
        return {"instrument": key, "n_rows": 0}
    if group_by == "auto":
        group_by = "route" if any(isinstance(r["feats"].get("route"), str) for r in recs) else None
    kw = dict(min_precision=min_precision, min_cover=min_cover, max_atoms=max_atoms, max_wrong=max_wrong,
              features=features, allow_value=allow_value, group_by=group_by or None, allow_opt=allow_opt,
              objective=objective, min_margin=min_margin, require=[parse_atom(a) for a in (require or [])])
    f = fit(recs, **kw)
    out = {"instrument": key, "n_tasks": len({r["task_id"] for r in recs}), **f,
           "stored": stored_counts(recs), "parsed": _sel_counts([r for r in recs if r["has_answer"]]),
           "eye_mean": _mean_eye(recs)}
    if do_loo:
        out["loo"] = loo(recs, **kw)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results.jsonl of a TRAIN instrument run")
    ap.add_argument("--instrument", help="instrument key (or --all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-precision", type=float, default=0.95)
    ap.add_argument("--min-cover", type=int, default=3)
    ap.add_argument("--max-atoms", type=int, default=2)
    ap.add_argument("--max-wrong", type=int, default=None)
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--group-by", default="auto", help="routing feature: one rule per value ('auto' = 'route' if present, '' = none)")
    ap.add_argument("--require", action="append", default=[], help="hard atom, e.g. agree==True or hits.len<=1 (repeatable)")
    ap.add_argument("--features", help="comma list of declared witness features the search may use (default: all diag features)")
    ap.add_argument("--allow-value", action="store_true", help="let the search threshold the reading value itself")
    ap.add_argument("--allow-opt", action="store_true", help="let the search use the option-margin features opt_*")
    ap.add_argument("--objective", choices=["cover", "gain"], default="cover")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="numeric thresholds must separate observed values at least this fraction of the feature range apart")
    ap.add_argument("--baseline", nargs="+", help="executor baseline dirs (train) for the eyeball pass@1 -> gain")
    ap.add_argument("--out-dir", help="write <key>.json per instrument")
    a = ap.parse_args()
    if "val" in Path(a.results).parts[-2] or "test" in Path(a.results).parts[-2]:
        raise SystemExit("refusing: rules are fitted on TRAIN readings only")
    keys = [a.instrument] if a.instrument else []
    if a.all:
        keys = sorted({json.loads(l)["instrument"] for l in open(a.results)})
    for key in keys:
        o = report_one(a.results, key, a.min_precision, a.min_cover, a.max_atoms, a.max_wrong, do_loo=not a.no_loo,
                       group_by=a.group_by, require=a.require,
                       features=set(a.features.split(",")) if a.features else None, allow_value=a.allow_value,
                       allow_opt=a.allow_opt, objective=a.objective, baseline=a.baseline, min_margin=a.min_margin)
        if not o.get("n_rows"):
            print(f"== {key}: no rows")
            continue
        ey = f", eyeball pass@1 {o['eye_mean']:.3f}" if o.get("eye_mean") is not None else ""
        print(f"== {key}: {o['n_tasks']} tasks, {o['n_rows']} rows, {o['n_atoms']} atoms, {o['searched']} rules searched"
              f" ({o['objective']}){ey}")
        print(f"   parsed only : {_fmt(o['parsed'])}")
        print(f"   stored rule : {_fmt(o['stored'])}")
        print(f"   fitted      : {_fmt(o['train'])}   {rule_str(o['rule'])}")
        if "loo" in o:
            print(f"   fitted LOO  : {_fmt(o['loo'])}   rule stability {o['loo']['rule_stability']:.2f}")
        if a.out_dir:
            Path(a.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(a.out_dir) / f"{key}.json").write_text(json.dumps(o, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
