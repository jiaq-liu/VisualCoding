#!/usr/bin/env python
"""Offline mechanism test for the GEPA driver: no API calls, no rollouts.

`common.vlm.run` is replaced by a stub that writes synthetic result.json files (a task is
"solved" only when the candidate prompt contains a magic word) and the reflection LM by a
stub that proposes that word.  A real gepa.optimize() therefore has to climb from 0 to 1,
which exercises the adapter protocol, the reflective-dataset schema, the trainset-as-valset
path and artifact saving.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES / "gepa" / "src"))
sys.path.insert(0, str(BASELINES / "drivers"))

from common import vlm, gemini, driver                      # noqa: E402

MAGIC = "MEASURE-THE-PIXELS"
tmp = Path(tempfile.mkdtemp(prefix="gepa_offline_"))
vlm.CACHE = tmp / "_vlm"
driver.OUT_ROOT = tmp / "out"
import common.paths as paths                                # noqa: E402
paths.OUT_ROOT = tmp / "out"

calls = {"vlm": 0, "lm": 0}


def fake_vlm_run(recs, prompt=None, *, rollouts=1, model="stub", **kw):
    calls["vlm"] += 1
    d = vlm.out_dir(prompt, model)
    score = 1.0 if prompt and MAGIC in prompt else 0.0
    for rec in recs:
        for i in range(rollouts):
            p = d / rec["doc_id"] / f"rollout_{i}"
            p.mkdir(parents=True, exist_ok=True)
            (p / "result.json").write_text(json.dumps({
                "question_id": rec["doc_id"], "_rollout_idx": i, "accuracy_score": score,
                "final_answer": "X", "ground_truth": rec["solution"], "family": rec["family"],
                "trajectory_text": "reasoning ... <answer>X</answer>",
                "trajectory_analysis": "stub judge"}))
    return {"out_dir": str(d), "n_tasks": len(recs), "rollouts": rollouts, "to_run": 0, "cached": 0}


def fake_lm(self, prompt, **kw):
    calls["lm"] += 1
    return f"```\nRead the question, then {MAGIC} before answering. Put the answer in <answer></answer>.\n```"


vlm.run = fake_vlm_run
gemini.Gemini.generate = fake_lm

import run_gepa                                             # noqa: E402
run_gepa.vlm.run = fake_vlm_run

from common import data                                     # noqa: E402


class Args:
    passes, minibatch, reflect_k, rollouts, workers, seed = 2, 3, 24, 2, 2, 0
    model = reflection_model = "stub"
    force = True
    dry_run = False


fam = data.family("mira:mr_billiards")
out = {}
ok = True
for mode, passes in (("single_pass", 0), ("search", 2)):
    calls["vlm"] = calls["lm"] = 0
    a = Args()
    a.passes = passes
    rep = run_gepa.train_family(fam, a, log=lambda *x: None)
    art = driver.load_artifact("gepa", fam)
    good = art is not None and MAGIC in art.text
    # single pass must be exactly one seed evaluation and one reflection call
    if mode == "single_pass":
        good = good and calls["lm"] == 1 and calls["vlm"] == 1
    ok = ok and good
    out[mode] = {"report": rep, "artifact_has_magic": bool(art and MAGIC in art.text),
                 "vlm_calls": calls["vlm"], "reflection_calls": calls["lm"], "ok": good}

print(json.dumps(out, indent=1, default=str))
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if ok else 1)
