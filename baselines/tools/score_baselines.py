#!/usr/bin/env python
"""Macro-accuracy table for the five baselines, both executors, against their own arm's control.

Each family contributes one number (mean accuracy over its test tasks at 1 rollout) and the
table averages families, not tasks -- the paper's macro convention.  The two arms are read
differently because they are stored differently: the tool arm writes one directory per
family (`metrics_at_k.json`), the tool-free arm writes one content-addressed directory per
PROMPT (`_vlm/<model>/<prompt_sha>/<task>/rollout_0/result.json`), shared across families,
so a no-tool family's score is gathered by task id from its own prompt's directory.

    python tools/score_baselines.py [--datasets tirbench,puzzlevqa,mira,gameqa]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, driver, vlm                                  # noqa: E402
from common.paths import OUT_ROOT                                     # noqa: E402

TOOL = ("xskill", "trace2skill", "skillopt")
NO_TOOL = ("gepa", "textgrad")
MODELS = {"pro": "gemini-3.1-pro-preview", "flashlite": "gemini-3.1-flash-lite"}


def tool_acc(method: str, fam, tag: str) -> float | None:
    p = driver.family_dir(method, fam) / f"deploy_test_{tag}" / "metrics_at_k.json"
    if not p.is_file():
        return None
    try:
        return json.load(open(p))["overall"]["pass@1"]
    except Exception:                                                  # noqa: BLE001
        return None


def tool_control(fam, tag: str) -> float | None:
    p = Path(OUT_ROOT) / "_control_tool" / tag / "test" / fam.slug / "metrics_at_k.json"
    if not p.is_file():
        return None
    try:
        return json.load(open(p))["overall"]["pass@1"]
    except Exception:                                                  # noqa: BLE001
        return None


def vlm_acc(prompt: str | None, fam, model: str) -> float | None:
    """Mean accuracy of one family under one prompt -- the cache holds every family together."""
    recs = fam.records("test")
    s = vlm.mean_scores(vlm.out_dir(prompt, model), recs, 1)
    return st.mean(s.values()) if s else None


def rows(fams, method: str, tag: str) -> tuple[list, list]:
    model = MODELS[tag]
    meth, ctrl = [], []
    for f in fams:
        if method in TOOL:
            a, c = tool_acc(method, f, tag), tool_control(f, tag)
        else:
            art = driver.load_artifact(method, f)
            a = vlm_acc(art.text, f, model) if art else None
            c = vlm_acc(None, f, model)
        if a is not None and c is not None:
            meth.append((f, a))
            ctrl.append((f, c))
    return meth, ctrl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="tirbench,puzzlevqa,mira,gameqa")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    fams = data.families(args.datasets.split(","))
    print(f"{len(fams)} families, {sum(len(f.records('test')) for f in fams)} test tasks\n")
    out = {}
    for tag in ("pro", "flashlite"):
        print(f"=== {MODELS[tag]} executor")
        print(f"{'method':12s} {'arm':8s} {'n':>3s} {'method':>8s} {'control':>8s} {'delta':>8s}   per-dataset delta")
        for method in TOOL + NO_TOOL:
            meth, ctrl = rows(fams, method, tag)
            if not meth:
                print(f"{method:12s} {'-':8s} {0:3d}   (no results)")
                continue
            by = defaultdict(list)
            for (f, a), (_, c) in zip(meth, ctrl):
                by[f.key.split(":")[0]].append(a - c)
            m, c = st.mean(a for _, a in meth), st.mean(x for _, x in ctrl)
            per = " ".join(f"{k}:{st.mean(v):+.3f}" for k, v in sorted(by.items()))
            arm = "tool" if method in TOOL else "no_tool"
            print(f"{method:12s} {arm:8s} {len(meth):3d} {m:8.4f} {c:8.4f} {m - c:+8.4f}   {per}")
            out[f"{method}_{tag}"] = {"n_families": len(meth), "method": m, "control": c,
                                      "delta": m - c,
                                      "per_dataset": {k: st.mean(v) for k, v in by.items()},
                                      "per_family": {f.key: a for f, a in meth}}
        print()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
