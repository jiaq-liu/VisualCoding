#!/usr/bin/env python
"""Offline test for the SkillOpt wiring: no API calls, no rollouts, no subprocess.

Checks the three things this study adds: the `gemini_chat` backend is selectable through
SkillOpt's own config machinery, the `visqa` env normalises data_split records and turns
executor rollouts into SkillOpt rows plus the conversation the reflection stage reads, and
the driver lays out splits (val/test = train) and the training command.
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

from common import data, driver                                  # noqa: E402
import common.paths as paths                                     # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="skillopt_offline_"))
paths.OUT_ROOT = tmp / "out"
driver.OUT_ROOT = tmp / "out"

import skillopt.model as model                                   # noqa: E402
from skillopt.envs.visqa.dataloader import _normalize            # noqa: E402
from skillopt.envs.visqa import rollout as visqa_rollout         # noqa: E402
from skillopt.envs.visqa.evaluator import aggregate              # noqa: E402

CODE = "print(img.getpixel((5,5)))"
fam = data.family("tirbench:color")
recs = fam.records("train")[:2]


def fake_exec_run(recs_, out_dir, skill_text=None, spec=None, **kw):
    out_dir = Path(out_dir)
    for rec in recs_:
        for i in range(2):
            d = out_dir / rec["doc_id"] / f"rollout_{i}"
            d.mkdir(parents=True, exist_ok=True)
            events = [
                {"doc_id": rec["doc_id"], "initial_prompt": rec["problem"],
                 "ground_truth": rec["solution"], "family": rec["family"], "rollout": i},
                {"turn_idx": 1, "tool_call": {"name": "code_interpreter", "code": CODE},
                 "tool_result": "(31, 49, 89)", "tool_images": [], "meta": {"error": False}},
                {"turn_idx": 2, "final_answer": "D", "accuracy_score": 1.0,
                 "committed_by": "model", "termination": "answered"},
            ]
            (d / "traj.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            (d / "result.json").write_text(json.dumps(
                {"question_id": rec["doc_id"], "_rollout_idx": i, "accuracy_score": 1.0,
                 "final_answer": "D", "ground_truth": rec["solution"], "family": rec["family"]}))
            (d / "metrics.json").write_text("{}")
    return {"out_dir": str(out_dir)}


visqa_rollout.executor.run = fake_exec_run
items = [_normalize(r, Path(paths.IMAGE_ROOT)) for r in recs]
rows = visqa_rollout.run_batch(items, str(tmp / "rb"), "SKILL TEXT", rollouts=2, workers=1)
# the analyst reads <rollout_dir>/predictions/<id>/conversation.json -- one level up
# and it finds nothing, calls no model, and the whole step is silently skipped
conv_path = tmp / "rb" / "predictions" / items[0]["id"] / "conversation.json"
conv = json.loads(conv_path.read_text())

import run_skillopt                                              # noqa: E402
cmds = []
SKILL = "# Learned skill\n\nMeasure the pixels before answering.\n"


def fake_train(cmd, **kw):
    """Stand in for `scripts/train.py`: record the command and leave a best_skill.md.

    It has to actually produce the skill, because the driver now refuses to save an
    artifact for a run that failed or produced nothing (a real run once left 33 empty
    artifacts behind, and `train` skips a family that already has one).
    """
    cmds.append(cmd)
    out_root = next((c.split("=", 1)[1] for c in cmd if c.startswith("out_root=")), None)
    if out_root:
        p = Path(out_root)
        p.mkdir(parents=True, exist_ok=True)
        (p / "best_skill.md").write_text(SKILL)
    return type("R", (), {"returncode": 0})()


run_skillopt.subprocess.run = fake_train
data.Family.records = lambda self, split: recs if split == "train" else []


class Args:
    rollouts, workers, epochs, batch_size, sel_env_num, seed = 2, 2, 1, 0, 0, 0
    model = "gemini-3.1-pro-preview"
    force, dry_run = True, False


rep = run_skillopt.train_family(fam, Args(), log=lambda *a: None)
d = driver.family_dir("skillopt", fam)
cmd = " ".join(cmds[0]) if cmds else ""
splits = {p.name: sorted(x.name for x in p.glob("*.json")) for p in (d / "splits").iterdir() if p.is_dir()}
val_recs = json.loads((d / "splits" / "val" / f"{fam.slug}.json").read_text())

checks = {
    "gemini_chat_selectable": model.set_backend("gemini_chat") == "gemini_chat",
    "row_schema": set(rows[0]) >= {"id", "question", "task_type", "hard", "soft",
                                   "predicted_answer", "response", "gold_answer", "phase"},
    "hard_soft": (rows[0]["hard"], rows[0]["soft"]) == (1, 1.0) and aggregate([1.0, 0.0]) == (1, 0.5),
    "conversation_has_code": any(CODE in c["content"] for c in conv),
    "conversation_per_rollout": len(conv) == 2,
    "conversation_in_predictions_dir": conv_path.is_file(),
    "item_normalised": items[0]["id"] == recs[0]["doc_id"] and "<image>" not in items[0]["question"],
    "splits_written": set(splits) == {"train", "val", "test"},
    "val_is_train": [r["doc_id"] for r in val_recs] == [r["doc_id"] for r in recs],
    "cmd_uses_visqa_config": "configs/visqa/default.yaml" in cmd,
    "cmd_sets_split_dir": "env.split_dir=" in cmd and "env.image_root=" in cmd,
    "artifact_saved": driver.load_artifact("skillopt", fam) is not None,
    "artifact_is_the_skill": (driver.artifact_text("skillopt", fam) or "").strip() == SKILL.strip(),
    # single pass: one step over the whole family, no selection rollouts, and an empty
    # initial skill so that one rollout pass is served from the acq pool
    "cmd_single_pass": "train.batch_size=0" in cmd and "evaluation.sel_env_num=0" in cmd,
    "initial_skill_empty": not (BASELINES / "SkillOpt" / "skillopt" / "envs" / "visqa"
                                / "skills" / "initial.md").read_text().strip(),
    "trainer_selection_guard": (BASELINES / "SkillOpt" / "skillopt" / "engine" / "trainer.py"
                                ).read_text().count("selection_off") >= 5,
}
# a FAILED train must NOT leave an artifact behind, or the retry skips the family
import shutil as _sh
_sh.rmtree(driver.family_dir("skillopt", fam), ignore_errors=True)
run_skillopt.subprocess.run = lambda cmd, **kw: type("R", (), {"returncode": 1})()
rep_fail = run_skillopt.train_family(fam, Args(), log=lambda *a: None)
checks["failure_saves_no_artifact"] = (
    "error" in rep_fail and driver.load_artifact("skillopt", fam) is None)

model.set_backend("openai_chat")
print(json.dumps({"report": rep, "checks": checks, "row": rows[0] | {"response": "..."}}, indent=1, default=str))
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if all(checks.values()) else 1)
