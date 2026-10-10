#!/usr/bin/env python
"""Trace2Skill baseline — trajectory-local lessons distilled into a skill, tool arm.

Trace2Skill's published stages are used as they are: per-trajectory LLM analysis of
failures and successes (`analysis/run_*_analysis_llm.py`), then parallel patch proposal and
hierarchical conflict-free consolidation into a skill directory
(`skill_evolver/run_parallel_combined_skill_evolution.py`).  Only the trajectory *source*
changes: instead of its own bash agent on SpreadsheetBench, the traces come from our
executor, so the arm is identical to XSkill / SkillOpt / the main method and the seed pass
is reused from the acq runs.

The rendered trace is rebuilt from `traj.jsonl` (common/traj.py), not from
`conversation_history`, whose tool calls carry no code — an analyst reading those would be
told only that "a tool was called".

Model calls go to gemini-3.1-pro via `--llm_client gemini` (src/react_agent/gemini_client.py).

  python drivers/run_trace2skill.py train  --families all
  python drivers/run_trace2skill.py deploy --families all --split test
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINES))

from common import data, driver, executor, traj                      # noqa: E402
from common.paths import MODEL, gemini_api_key, subprocess_env       # noqa: E402

METHOD = "trace2skill"
ARM = "tool"
T2S = BASELINES / "Trace2Skill"
SEED_SKILL = T2S / "vqa_skill_seed"
MAX_SKILL_CHARS = 12000


def _py() -> str:
    v = BASELINES / ".venvs" / "trace2skill" / "bin" / "python"
    return str(v if v.is_file() else sys.executable)


def _run(cmd: list[str], log_file: Path, log=print) -> int:
    env = subprocess_env({"PYTHONPATH": f"{T2S}:{BASELINES}", "OPENAI_API_KEY": gemini_api_key()})
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd[:6])} ...\n")
        fh.flush()
        return subprocess.run(cmd, cwd=str(T2S), env=env, stdout=fh, stderr=fh).returncode


def write_logs(train_dir: Path, logs_dir: Path) -> dict:
    """One markdown log per rollout, named as Trace2Skill's discovery expects.

    `<prefix>_<instance_id>_<SUCCEED|FAILED>.md`, where instance_id is the last
    underscore-separated field — so doc_ids (which contain underscores) are written with
    dashes, keeping them intact as a single field.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    n = {"SUCCEED": 0, "FAILED": 0}
    for t in traj.load_run(train_dir):
        tag = "SUCCEED" if t.success else "FAILED"
        iid = f"{t.task_id}-r{t.rollout}".replace("_", "-")
        body = (f"# Task {t.task_id} (rollout {t.rollout})\n\n"
                f"Question: {t.question.strip()}\n\nGround truth: {t.ground_truth}\n\n"
                f"## Trajectory\n\n{t.render(with_header=False)}\n")
        (logs_dir / f"t2s_agent_{iid}_{tag}.md").write_text(body)
        n[tag] += 1
    return n


def train_family(fam: data.Family, args, log=print) -> dict:
    d = driver.family_dir(METHOD, fam)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "artifact.json").is_file() and not args.force:
        return {"family": fam.key, "skipped": "artifact exists"}

    recs = fam.records("train")
    train_dir = d / "train"
    spec = executor.ExecSpec(model=args.model, rollouts=args.rollouts, workers=args.workers)
    rep = executor.run(recs, train_dir, skill_text=None, spec=spec, dry_run=args.dry_run, log=log)
    if args.dry_run:
        return {"family": fam.key, "n_train": len(recs), "rollouts": rep["reuse"], "dry_run": True}

    t0 = time.time()
    logs_dir = d / "logs"
    counts = write_logs(train_dir, logs_dir)
    log(f"   logs: {counts['SUCCEED']} SUCCEED, {counts['FAILED']} FAILED")

    err_dir, suc_dir = d / "error_analysis", d / "success_analysis"
    common = ["--model", args.model, "--llm_client", "gemini", "--max_workers", str(args.analysts)]
    if counts["FAILED"]:
        _run([_py(), "analysis/run_error_analysis_llm.py", "--logs_dir", str(logs_dir),
              "--output_dir", str(err_dir), *common], d / "analysis.log", log)
    if counts["SUCCEED"]:
        _run([_py(), "analysis/run_success_analysis_llm.py", "--logs_dir", str(logs_dir),
              "--output_dir", str(suc_dir), *common], d / "analysis.log", log)

    # The evolver takes both inputs as required arguments and resolves each to an existing
    # file.  A family whose training rollouts all passed has no error analysis, and one where
    # they all failed has no success analysis -- that was 10 of 69 families on the first run,
    # each dying with "Input path not found".  Give the missing side an empty record list, so
    # "there were no failures to learn from" is expressed as data instead of as a crash.
    for dir_, name in ((err_dir, "parsed_error_records.json"),
                       (suc_dir, "parsed_success_records.json")):
        dir_.mkdir(parents=True, exist_ok=True)
        if not (dir_ / name).is_file():
            (dir_ / name).write_text("[]")

    skill_dir = d / "skill"
    if not skill_dir.is_dir():
        shutil.copytree(SEED_SKILL, skill_dir)
    rc = _run([_py(), "skill_evolver/run_parallel_combined_skill_evolution.py",
               "--error-json", str(err_dir), "--success-json", str(suc_dir),
               "--skill-dir", str(skill_dir), "--model", args.model, "--llm-client", "gemini",
               "--max-workers", str(args.analysts), "--batch-size", str(args.batch_size),
               "--merge-batch-size", str(args.merge_batch_size),
               "--max-skill-lines", str(args.max_skill_lines), "--continue-evolving"],
              d / "evolve.log", log)

    text = skill_text(skill_dir)
    if rc != 0 or not text.strip() or text.strip() == skill_text(SEED_SKILL).strip():
        # No artifact on failure, and none when the evolver left the seed skill untouched:
        # `train` skips a family that already has an artifact, so either would quietly ship
        # as "what Trace2Skill learned" and survive every retry.
        return {"family": fam.key, "n_train": len(recs), **counts,
                "error": f"evolver returncode {rc}" if rc != 0 else "skill unchanged from seed",
                "skill_chars": len(text), "log": str(d / "evolve.log")}
    driver.Artifact(
        method=METHOD, family=fam.key, arm=ARM, text=text, model=args.model, seed=args.seed,
        train_tasks=len(recs), rollouts_used=len(recs) * args.rollouts,
        rollouts_new=int(rep["reuse"].get("to_run", 0)), lm_calls=-1,
        notes={"logs": counts, "evolver_returncode": rc, "skill_dir": str(skill_dir),
               "chars": len(text), "wall_s": round(time.time() - t0)}).save()
    return {"family": fam.key, "n_train": len(recs), **counts, "evolver_rc": rc,
            "skill_chars": len(text), "wall_s": round(time.time() - t0)}


def skill_text(skill_dir: Path) -> str:
    """SKILL.md plus its reference files — what gets injected into the system prompt."""
    parts = []
    md = skill_dir / "SKILL.md"
    if md.is_file():
        body = md.read_text()
        if body.startswith("---"):                      # drop the YAML front matter
            body = body.split("---", 2)[-1]
        parts.append(body.strip())
    for ref in sorted((skill_dir / "references").glob("*.md")) if (skill_dir / "references").is_dir() else []:
        parts.append(f"### {ref.stem}\n{ref.read_text().strip()}")
    text = "\n\n".join(p for p in parts if p)
    return text[:MAX_SKILL_CHARS]


def main(argv=None) -> int:
    ap = driver.parser(METHOD, ARM)
    ap.add_argument("--analysts", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--merge-batch-size", type=int, default=5)
    ap.add_argument("--max-skill-lines", type=int, default=500)
    args = ap.parse_args(argv)
    fams = driver.select_families(args.families)[: args.limit]
    log = print
    if args.verb == "status":
        print(json.dumps(driver.status(METHOD, fams), indent=1))
        return 0
    args.rollouts = args.rollouts if args.rollouts is not None else (4 if args.verb == "train" else 1)
    reports = []
    for i, fam in enumerate(fams, 1):
        log(f"[{i}/{len(fams)}] {args.verb} {fam.key}")
        if args.verb == "train":
            rep = train_family(fam, args, log=log)
        elif args.verb == "deploy":
            rep = driver.deploy_family(METHOD, fam, ARM, args.split, rollouts=args.rollouts,
                                       workers=args.workers, model=args.model,
                                       dry_run=args.dry_run, log=log)
        else:
            rep = driver.control_family(fam, ARM, args.split, rollouts=args.rollouts,
                                        workers=args.workers, model=args.model,
                                        dry_run=args.dry_run, log=log)
        rep["family"] = fam.key
        reports.append(rep)
        log("   " + json.dumps({k: v for k, v in rep.items() if k != "map"}, default=str))
    out = driver.method_root(METHOD) / (
        f"_{args.verb}_report.json" if args.verb == "train" else
        f"_{args.verb}_{args.split}_{driver.model_tag(args.model)}_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=1, default=str))
    log(f"\n{args.verb}: {len(reports)} families; report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
