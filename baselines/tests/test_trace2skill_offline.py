#!/usr/bin/env python
"""Offline test for the Trace2Skill wiring: no API calls, no subprocesses.

Covers the parts this study adds — trajectory logs in the naming Trace2Skill's own
discovery expects (and carrying the code, not "[Function Call]"), the two analysis calls
and the evolver call, the skill text that gets injected, and the shape of the two Gemini
clients.  The upstream analysis / evolution logic itself is exercised by its own repo.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES / "Trace2Skill"))
sys.path.insert(0, str(BASELINES / "drivers"))

from common import data, driver                                  # noqa: E402
import common.paths as paths                                     # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="t2s_offline_"))
paths.OUT_ROOT = tmp / "out"
driver.OUT_ROOT = tmp / "out"

CODE = "print(img.getpixel((5,5)))"


def fake_exec_run(recs, out_dir, skill_text=None, **kw):
    out_dir = Path(out_dir)
    for rec in recs:
        for i in range(2):
            d = out_dir / rec["doc_id"] / f"rollout_{i}"
            d.mkdir(parents=True, exist_ok=True)
            events = [
                {"doc_id": rec["doc_id"], "initial_prompt": rec["problem"],
                 "ground_truth": rec["solution"], "family": rec["family"], "rollout": i},
                {"turn_idx": 1, "text_output": "Measuring.", "tool_name": "code_interpreter"},
                {"turn_idx": 1, "tool_call": {"name": "code_interpreter", "code": CODE},
                 "tool_result": "(31, 49, 89)", "tool_images": [], "meta": {"error": False}},
                {"turn_idx": 2, "final_answer": "D", "accuracy_score": 1.0 if i == 0 else 0.0,
                 "committed_by": "model", "termination": "answered"},
            ]
            (d / "traj.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            (d / "result.json").write_text(json.dumps(
                {"question_id": rec["doc_id"], "_rollout_idx": i,
                 "accuracy_score": 1.0 if i == 0 else 0.0, "final_answer": "D",
                 "ground_truth": rec["solution"], "family": rec["family"]}))
            (d / "metrics.json").write_text("{}")
    return {"out_dir": str(out_dir), "reuse": {"reused": 0, "to_run": 0}}


import run_trace2skill                                           # noqa: E402
run_trace2skill.executor.run = fake_exec_run
cmds = []
EVOLVED = "Measure the pixel colour with the code interpreter before choosing an option."


def fake_run(cmd, log_file, log=print):
    """Record the command; when it is the evolver, actually evolve the skill directory.

    The driver now refuses to save an artifact whose skill is empty *or still identical to
    the seed*, because either would ship as "what Trace2Skill learned" and, since `train`
    skips a family that already has an artifact, survive every retry.
    """
    cmds.append(cmd)
    if "skill_evolution" in " ".join(cmd):
        sd = Path(next(c for i, c in enumerate(cmd) if cmd[i - 1] == "--skill-dir"))
        md = sd / "SKILL.md"
        md.write_text((md.read_text() if md.is_file() else "") + f"\n\n## Learned\n{EVOLVED}\n")
    return 0


run_trace2skill._run = fake_run


class Args:
    rollouts, workers, analysts, seed = 2, 2, 2, 0
    model = "gemini-3.1-pro-preview"
    batch_size, merge_batch_size, max_skill_lines = 1, 5, 500
    force, dry_run = True, False


fam = data.family("tirbench:color")
recs = fam.records("train")[:2]
data.Family.records = lambda self, split: recs if split == "train" else []
rep = run_trace2skill.train_family(fam, Args(), log=lambda *a: None)

d = driver.family_dir("trace2skill", fam)
logs = sorted(p.name for p in (d / "logs").glob("*.md"))
sys.path.insert(0, str(BASELINES / "Trace2Skill" / "analysis"))
from analysis.run_error_analysis_llm import parse_log_filename   # noqa: E402
parsed = [parse_log_filename(n) for n in logs]
body = (d / "logs" / logs[0]).read_text()

# a reference file must survive into the injected text
(d / "skill" / "references").mkdir(parents=True, exist_ok=True)
(d / "skill" / "references" / "colour.md").write_text("Sample HSV with code.")
text = run_trace2skill.skill_text(d / "skill")

from src.react_agent.gemini_client import GeminiClient, GeminiOpenAIShim   # noqa: E402
import common.gemini as cg                                        # noqa: E402
cg.Gemini.generate = lambda self, prompt, **kw: "STUB REPLY"
gc = GeminiClient(model="stub")
shim = GeminiOpenAIShim(model="stub")


class M:
    role, content = "user", "hi"


checks = {
    "logs_written": len(logs) == 4,
    "log_names_parse": all(p is not None for p in parsed),
    "instance_ids_intact": all("tirbench-color" in (p[0] or "") for p in parsed),
    "both_outcomes": {p[1] for p in parsed} == {"SUCCEED", "FAILED"},
    "log_has_code": CODE in body,
    "error_analysis_called": any("run_error_analysis_llm.py" in c for c in map(" ".join, cmds)),
    "success_analysis_called": any("run_success_analysis_llm.py" in c for c in map(" ".join, cmds)),
    "analysis_uses_gemini": all("--llm_client gemini" in c for c in map(" ".join, cmds)
                                if "analysis_llm.py" in c),
    "evolver_called_with_gemini": any("run_parallel_combined_skill_evolution.py" in c
                                      and "--llm-client gemini" in c for c in map(" ".join, cmds)),
    "skill_text_no_frontmatter": not text.startswith("---"),
    "skill_text_has_reference": "Sample HSV with code." in text,
    "artifact_saved": driver.load_artifact("trace2skill", fam) is not None,
    "artifact_is_evolved": EVOLVED in (driver.artifact_text("trace2skill", fam) or ""),
    "gemini_client_chat": gc.chat([M()]) == "STUB REPLY",
    "openai_shim": shim.chat.completions.create(
        model="stub", messages=[{"role": "user", "content": "hi"}]).choices[0].message.content == "STUB REPLY",
}
print(json.dumps({"report": rep, "checks": checks, "logs": logs}, indent=1, default=str))
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if all(checks.values()) else 1)
