"""The tool arm: rollouts through our own executor (`verify.run`, code_interpreter agent).

Every baseline that is allowed tools (XSkill, Trace2Skill, SkillOpt) produces its
rollouts here, so all of them share one executor, one turn budget, one judge and
one no-skill control with the main method.  The call is a subprocess running
the main environment's interpreter (VC_PY): baselines keep their own virtualenvs.

    from common import data, executor
    fam = data.family("tirbench:color")
    r = executor.run(fam.records("train"), out_dir, skill_text=None)      # seed pass: fully reused
    rows = executor.results(out_dir)                                      # one dict per rollout
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import data
from .paths import VC_PY, IMAGE_ROOT, JUDGE_MODEL, MODEL, VC_ROOT, subprocess_env
from .store import RolloutStore, prompt_sha, signature


@dataclass
class ExecSpec:
    """The acq protocol. Changing any field changes the signature, i.e. disables reuse of old runs."""
    model: str = MODEL
    judge_model: str = JUDGE_MODEL
    rollouts: int = 4
    workers: int = 8
    max_turns: int = 20
    max_tokens: int = 8192
    temperature: float = 0.6
    judge: str = "llm"
    max_passes: int = 3            # resumable retries of verify.run until every rollout exists

    def signature(self, skill_text: str | None = None) -> tuple[str, str]:
        sig = signature(model=self.model, max_turns=self.max_turns,
                        max_tokens=self.max_tokens, temperature=self.temperature)
        return sig, prompt_sha(skill_text)


DEFAULT = ExecSpec()


def run(recs: list[dict], out_dir, skill_text: str | None = None, *, spec: ExecSpec = DEFAULT,
        inject_dir=None, reuse: bool = True, dry_run: bool = False, store: RolloutStore | None = None,
        log=print) -> dict:
    """Roll out `recs` (data_split records) under an optional skill; returns a report dict.

    `skill_text` is appended to the system prompt exactly as `--system-prompt-extra` does.
    `inject_dir` holds per-task `<task_id>.txt` files (XSkill's per-sample adaptation); it makes
    the prompt task-dependent, so reuse is restricted to rollouts with a matching prompt.
    `dry_run=True` reports what would be reused / run without calling the model.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = data.write_input(recs, out_dir / "_input.json")
    task_ids = [r["doc_id"] for r in recs]
    sig, sha = spec.signature(skill_text)

    skill_file = None
    if skill_text:
        skill_file = out_dir / "_skill.md"
        skill_file.write_text(skill_text)

    store = store or RolloutStore()
    reuse_report = {"reused": 0, "already_present": 0, "to_run": len(task_ids) * spec.rollouts}
    if reuse and inject_dir is None:
        reuse_report = store.materialize(out_dir, task_ids, sig, sha, spec.rollouts, dry_run=dry_run)
        if dry_run:                      # the full link map is noise in a planning report
            reuse_report = {k: v for k, v in reuse_report.items() if k != "map"}

    report = {"out_dir": str(out_dir), "n_tasks": len(task_ids), "rollouts": spec.rollouts,
              "model": spec.model, "sig": sig, "prompt_sha": sha, "reuse": reuse_report,
              "skill": bool(skill_text), "inject_dir": str(inject_dir) if inject_dir else None}
    if dry_run:
        report["dry_run"] = True
        return report

    cmd = [str(VC_PY), "-u", "-m", "verify.run",
           "--input", str(input_file), "--image-root", str(IMAGE_ROOT),
           "--model", spec.model, "--judge-model", spec.judge_model, "--judge", spec.judge,
           "--rollouts", str(spec.rollouts), "--workers", str(spec.workers),
           "--max-turns", str(spec.max_turns), "--max-tokens", str(spec.max_tokens),
           "--temperature", str(spec.temperature),
           "--output-dir", str(out_dir)]
    if skill_file:
        cmd += ["--system-prompt-extra", str(skill_file)]
    if inject_dir:
        cmd += ["--inject-dir", str(inject_dir)]

    t0 = time.time()
    log_file = out_dir.with_suffix(".log") if out_dir.suffix else Path(str(out_dir) + ".log")
    passes = 0
    for passes in range(1, spec.max_passes + 1):
        with open(log_file, "a") as fh:
            fh.write(f"\n=== pass {passes} {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
            fh.flush()
            subprocess.run(cmd, cwd=str(VC_ROOT), env=subprocess_env(), stdout=fh, stderr=fh)
        if complete(out_dir, task_ids, spec.rollouts):
            break
    report.update(passes=passes, wall_s=round(time.time() - t0),
                  complete=complete(out_dir, task_ids, spec.rollouts), log=str(log_file))
    (out_dir / "_run_report.json").write_text(json.dumps(report, indent=1))
    return report


def complete(out_dir, task_ids, rollouts: int) -> bool:
    out_dir = Path(out_dir)
    return all((out_dir / t / f"rollout_{i}" / "result.json").is_file()
               for t in task_ids for i in range(rollouts))


def results(out_dir) -> list[dict]:
    """One dict per rollout (verify's result.json fields), read straight from the tree."""
    out_dir = Path(out_dir)
    rows = []
    for p in sorted(out_dir.glob("*/rollout_*/result.json")):
        try:
            r = json.loads(p.read_text())
            r["_dir"] = str(p.parent)
            rows.append(r)
        except Exception:
            pass
    return rows


def scores(out_dir) -> dict:
    """task_id -> mean accuracy over its rollouts."""
    per: dict[str, list] = {}
    for r in results(out_dir):
        per.setdefault(r["question_id"], []).append(float(r["accuracy_score"]))
    return {k: sum(v) / len(v) for k, v in per.items() if v}


def accuracy(out_dir) -> float:
    s = scores(out_dir)
    return sum(s.values()) / len(s) if s else 0.0
