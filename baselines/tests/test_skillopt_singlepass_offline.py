#!/usr/bin/env python
"""Offline test of the SINGLE-PASS SkillOpt loop: runs the real trainer, no API calls.

This is the expensive branch to get wrong.  Under the study's protocol SkillOpt must roll
out exactly once per family -- the whole train split, under an EMPTY skill, which the reuse
store serves from the acq pool -- and never again.  Two things could silently break that:

  * `evaluation.sel_env_num > 0` puts a selection pass after every edit (and a baseline pass
    before the first), which is 9 x 4 fresh agent rollouts per family;
  * `train.batch_size > 0` splits the family into several steps, each with its own rollout
    under an edited (hence uncached) skill.

So the test drives `ReflACTTrainer.train()` for real with the reflection/aggregation/apply
stages stubbed, and asserts on the number and the content of the rollout calls.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES / "SkillOpt"))
sys.path.insert(0, str(BASELINES / "drivers"))

from common import data                                          # noqa: E402
import common.paths as paths                                     # noqa: E402

SKILLOPT = BASELINES / "SkillOpt"
tmp = Path(tempfile.mkdtemp(prefix="skillopt_singlepass_"))
paths.OUT_ROOT = tmp / "out"

NEW_SKILL = "# Learned skill\n\nMeasure the pixels before answering.\n"

from skillopt.config import flatten_config, is_structured, load_config   # noqa: E402
from skillopt.envs.visqa import rollout as visqa_rollout                 # noqa: E402
from skillopt.envs.visqa.adapter import VisQAAdapter                     # noqa: E402
import skillopt.engine.trainer as trainer_mod                            # noqa: E402

fam = data.family("tirbench:color")
recs = fam.records("train")[:6]

# ---------------------------------------------------------------- stub the executor
seen: list[dict] = []


def fake_exec_run(recs_, out_dir, skill_text=None, spec=None, **kw):
    seen.append({"n": len(recs_), "skill": skill_text, "dir": str(out_dir)})
    out_dir = Path(out_dir)
    for rec in recs_:
        for i in range(2):
            d = out_dir / rec["doc_id"] / f"rollout_{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "traj.jsonl").write_text("\n".join(json.dumps(e) for e in [
                {"doc_id": rec["doc_id"], "initial_prompt": rec["problem"],
                 "ground_truth": rec["solution"], "family": rec["family"], "rollout": i},
                {"turn_idx": 1, "tool_call": {"name": "code_interpreter", "code": "print(1)"},
                 "tool_result": "1", "tool_images": [], "meta": {"error": False}},
                {"turn_idx": 2, "final_answer": "D", "accuracy_score": float(i),
                 "committed_by": "model", "termination": "answered"}]))
            (d / "result.json").write_text(json.dumps(
                {"question_id": rec["doc_id"], "_rollout_idx": i, "accuracy_score": float(i),
                 "final_answer": "D", "ground_truth": rec["solution"], "family": rec["family"]}))
            (d / "metrics.json").write_text("{}")
    return {"out_dir": str(out_dir)}


visqa_rollout.executor.run = fake_exec_run

# ------------------------------------------------- stub reflect / aggregate / apply
trainer_mod._normalise_patches = lambda raw, **k: ([{"op": "add", "content": "x"}], [])
trainer_mod.merge_patches = lambda *a, **k: {"edits": [{"op": "add", "content": "x"}]}
trainer_mod.get_payload_items = lambda *a, **k: [{"op": "add", "content": "x"}]
trainer_mod.rank_and_select = lambda *a, **k: {"edits": [{"op": "add", "content": "x"}]}
trainer_mod.apply_patch_with_report = lambda skill, patch: (NEW_SKILL, [])
trainer_mod._extract_failure_patterns = lambda *a, **k: []

# ------------------------------------------------------------------ config + splits
import run_skillopt                                              # noqa: E402
data.Family.records = lambda self, split: recs if split == "train" else []
d = tmp / "fam"
d.mkdir(parents=True, exist_ok=True)
splits = run_skillopt.make_splits(fam, d)

cfg = load_config(str(SKILLOPT / "configs" / "visqa" / "default.yaml"), overrides=[
    f"env.split_dir={splits}", f"env.image_root={paths.IMAGE_ROOT}",
    "env.rollouts=2", "env.workers=1", "train.num_epochs=1",
])
flat = flatten_config(cfg) if is_structured(cfg) else cfg
flat["out_root"] = str(tmp / "run")
flat["env"] = "visqa"
flat["skill_init"] = str(SKILLOPT / flat["skill_init"]) if flat.get("skill_init") else ""

import inspect                                                   # noqa: E402
accepted = set(inspect.signature(VisQAAdapter.__init__).parameters) - {"self"}
adapter = VisQAAdapter(**{k: flat[k] for k in accepted if k in flat})
adapter.reflect = lambda *a, **k: [{"op": "add", "content": "x"}]

summary = trainer_mod.ReflACTTrainer(flat, adapter).train()

best = (tmp / "run" / "best_skill.md")
checks = {
    "one_rollout_call": len(seen) == 1,
    "rollout_is_whole_family": bool(seen) and seen[0]["n"] == len(recs),
    "rollout_skill_empty": bool(seen) and not (seen[0]["skill"] or ""),
    "batch_size_resolved": flat.get("batch_size") == len(recs),
    "sel_env_num_zero": int(flat.get("sel_env_num", -1)) == 0,
    "slow_update_off": flat.get("use_slow_update") is False,
    "meta_skill_off": flat.get("use_meta_skill") is False,
    "eval_test_off": flat.get("eval_test") is False,
    "best_skill_is_the_edit": best.is_file() and NEW_SKILL.strip() in best.read_text(),
}
print(json.dumps({"checks": checks, "rollout_calls": seen,
                  "summary": {k: summary.get(k) for k in ("train_hard", "best_val", "steps")}},
                 indent=1, default=str))
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if all(checks.values()) else 1)
