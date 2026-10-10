#!/usr/bin/env python
"""Offline mechanism test for the XSkill driver: no API calls, no rollouts.

A synthetic executor output tree (the shape verify.run writes) is built on disk, the
executor call is stubbed out and ExperienceLLM is replaced by canned responses.  The real
XSkill modules then run: summarise -> critique -> batch_merge -> per-sample skill ->
merge_skills, and the artifact must end up carrying both the skill document and the
experience block.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES / "XSkill" / "eval"))
sys.path.insert(0, str(BASELINES / "drivers"))

from common import data, driver, executor                    # noqa: E402
import common.paths as paths                                 # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="xskill_offline_"))
paths.OUT_ROOT = tmp / "out"
driver.OUT_ROOT = tmp / "out"

SKILL_TEXT = "## Skill: measure before answering\n1. Crop the region.\n2. Read the pixels."
EXPERIENCE = "Sample the HSV value with code instead of eyeballing the colour."


def fake_exec_run(recs, out_dir, skill_text=None, **kw):
    """Write the rollout tree verify.run would have written."""
    out_dir = Path(out_dir)
    for rec in recs:
        for i in range(2):
            d = out_dir / rec["doc_id"] / f"rollout_{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "injected_system_prompt.txt").write_text("You are a visual reasoning agent.")
            events = [
                {"doc_id": rec["doc_id"], "initial_prompt": rec["problem"],
                 "ground_truth": rec["solution"], "family": rec["family"], "rollout": i},
                {"turn_idx": 1, "text_output": "Let me measure the colour.", "tool_name": "code_interpreter"},
                {"turn_idx": 1, "tool_call": {"name": "code_interpreter", "code": "print(img.getpixel((5,5)))"},
                 "tool_result": "(31, 49, 89)", "tool_images": [], "meta": {"error": False}, "evidence": []},
                {"turn_idx": 2, "text_output": "<answer>D</answer>"},
                {"turn_idx": 2, "final_answer": "D", "accuracy_score": 1.0 if i == 0 else 0.0,
                 "committed_by": "model", "termination": "answered"},
            ]
            (d / "traj.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            (d / "result.json").write_text(json.dumps(
                {"question_id": rec["doc_id"], "_rollout_idx": i, "accuracy_score": 1.0 if i == 0 else 0.0,
                 "final_answer": "D", "ground_truth": rec["solution"], "family": rec["family"],
                 "trajectory_analysis": "stub judge"}))
            (d / "metrics.json").write_text(json.dumps({"accuracy_score": 1.0 if i == 0 else 0.0}))
    return {"out_dir": str(out_dir), "n_tasks": len(recs), "rollouts": 2,
            "reuse": {"reused": 0, "to_run": 0}}


import run_xskill                                            # noqa: E402
run_xskill.executor.run = fake_exec_run

from exskill import llm_client, gemini_bridge                # noqa: E402

# Keep the test fully offline: the experience merger embeds candidate experiences.
gemini_bridge.embed = lambda texts, model=None: [[float(len(t) % 7), 1.0, 0.5] for t in texts]

calls = {"chat": 0, "image": 0}


def fake_chat(self, prompt, max_tokens=None, temperature=None, top_p=1.0):
    calls["chat"] += 1
    p = str(prompt)
    if "json" in p.lower() and ("experience" in p.lower() or "critique" in p.lower()):
        return "```json\n" + json.dumps([{"option": "add", "experience": EXPERIENCE}]) + "\n```"
    if "skill" in p.lower():
        return SKILL_TEXT
    return "The agent cropped the swatch and read the pixel values, then answered D."


def fake_chat_with_image(self, prompt, image, **kw):
    calls["image"] += 1
    return "A colour swatch on a white background."


llm_client.ExperienceLLM.chat = fake_chat
llm_client.ExperienceLLM.chat_with_image = fake_chat_with_image


class Args:
    rollouts, workers, analysts, seed = 2, 2, 2, 0
    model = "stub-model"
    experience_max_ops, experience_max_items, skill_max_length = 2, 16, 1000
    experience_refine = skill_refine = False
    force, dry_run = True, False


fam = data.family("tirbench:color")
recs = fam.records("train")[:2]
fam_records = data.Family.records
data.Family.records = lambda self, split: recs if split == "train" else fam_records(self, split)

rep = run_xskill.train_family(fam, Args(), log=lambda *a: None)
art = driver.load_artifact("xskill", fam)
# The merger may rewrite an experience's wording (it calls the LLM), so assert the shape:
# the skill document and a non-empty experience block both reached the injected artifact.
ok = (art is not None and SKILL_TEXT.split("\n")[0] in art.text
      and "[E" in art.text and len(art.text) > len(SKILL_TEXT) + 20)
print(json.dumps({"report": rep, "artifact_ok": ok, "llm_text_calls": calls["chat"],
                  "llm_image_calls": calls["image"],
                  "artifact_head": (art.text[:120] if art else None)}, indent=1, default=str))
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if ok else 1)
