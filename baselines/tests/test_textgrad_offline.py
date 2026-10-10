#!/usr/bin/env python
"""Offline mechanism test for the TextGrad driver: no API calls.

A stub EngineLM answers every call (forward VLM call, evaluator, backward, TGD step) and
`common.vlm.run` writes synthetic scores that only reward a prompt containing the magic
word.  A real textgrad graph therefore has to carry a gradient from the loss back into the
system-prompt Variable and the TGD step has to rewrite it.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES / "textgrad"))
sys.path.insert(0, str(BASELINES / "drivers"))

from common import vlm, driver, data                       # noqa: E402
import common.paths as paths                               # noqa: E402

MAGIC = "MEASURE-THE-PIXELS"
tmp = Path(tempfile.mkdtemp(prefix="tg_offline_"))
vlm.CACHE = tmp / "_vlm"
paths.OUT_ROOT = tmp / "out"
driver.OUT_ROOT = tmp / "out"
calls = {"engine": 0, "vlm": 0}


def fake_vlm_run(recs, prompt=None, *, rollouts=1, model="stub", **kw):
    calls["vlm"] += 1
    d = vlm.out_dir(prompt, model)
    score = 1.0 if prompt and MAGIC in prompt else 0.0
    for rec in recs:
        for i in range(rollouts):
            p = d / rec["doc_id"] / f"rollout_{i}"
            p.mkdir(parents=True, exist_ok=True)
            (p / "result.json").write_text(json.dumps(
                {"question_id": rec["doc_id"], "_rollout_idx": i, "accuracy_score": score,
                 "final_answer": "X", "ground_truth": rec["solution"], "family": rec["family"],
                 "trajectory_text": "...", "trajectory_analysis": "stub"}))
    return {"out_dir": str(d), "n_tasks": len(recs), "rollouts": rollouts, "to_run": 0}


vlm.run = fake_vlm_run
import run_textgrad                                        # noqa: E402
run_textgrad.vlm.run = fake_vlm_run

from textgrad.engine.base import EngineLM                  # noqa: E402


class StubEngine(EngineLM):
    model_string = "gemini-3.1-pro-preview"                # multimodal per the registry patch
    system_prompt = "stub"
    n_calls = 0

    def __call__(self, content, **kw):
        return self.generate(content, **kw)

    def generate(self, content, system_prompt=None, **kw):
        StubEngine.n_calls += 1
        calls["engine"] += 1
        text = " ".join(str(c) for c in content) if isinstance(content, list) else str(content)
        if "<IMPROVED_VARIABLE>" in text or "improved" in text.lower():
            return (f"<IMPROVED_VARIABLE>Read the question, then {MAGIC} before answering. "
                    f"Put the answer in <answer></answer>.</IMPROVED_VARIABLE>")
        if "ACCURACY" in text:
            return "The response misses the measurement step. <ACCURACY>0</ACCURACY>"
        return "I looked at the image. <answer>X</answer>"


run_textgrad._engine = lambda model: _install(StubEngine())


def _install(eng):
    import textgrad as tg
    tg.set_backward_engine(eng, override=True)
    return eng


class Args:
    epochs, batch_size, measure_rollouts, workers, seed = 1, 3, 0, 2, 0
    model = "gemini-3.1-pro-preview"
    force, dry_run = True, False


fam = data.family("mira:mr_billiards")
out = {}
ok = True
for mode, measure in (("single_pass", 0), ("measured", 2)):
    calls["engine"] = calls["vlm"] = 0
    a = Args()
    a.measure_rollouts = measure
    rep = run_textgrad.train_family(fam, a, log=lambda *x: None)
    art = driver.load_artifact("textgrad", fam)
    good = art is not None and MAGIC in art.text
    if mode == "single_pass":
        # measurement off => no tool-free rollouts at all, and the artifact is the last prompt
        good = good and calls["vlm"] == 0 and (art.notes or {}).get("mode") == "single_pass"
    ok = ok and good
    out[mode] = {"report": rep, "engine_calls": calls["engine"], "vlm_calls": calls["vlm"],
                 "ok": good}

print(json.dumps(out, indent=1, default=str))
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if ok else 1)
