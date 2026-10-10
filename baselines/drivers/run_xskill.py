#!/usr/bin/env python
"""XSkill baseline — experience library + skill document, tool arm.

XSkill's learning modules are used as published (`eval/exskill/*`: visually grounded
trajectory summarisation, cross-rollout critique, experience merging, skill synthesis and
consolidation).  What is replaced is the *rollout producer*: instead of XSkill's own agent
loop, trajectories come from our executor, so XSkill, Trace2Skill, SkillOpt and the main
method all share one agent, one tool, one turn budget, one judge — and the same no-skill
control.  That also makes the seed pass free: with an empty skill the system prompt is
byte-identical to the acq runs, so those rollouts are reused (see common/store.py).

This works because the two trees are the same lineage: XSkill's summariser reads
`<sample>/rollout_*/traj.jsonl` with `initial_prompt`/`ground_truth` on the first line, an
`injected_system_prompt.txt`, and `original_image.jpg` / `tool_image_*.jpg` next to them —
which is exactly what verify.run writes.

Model calls go to gemini-3.1-pro through google-genai (eval/exskill/gemini_bridge.py).

  python drivers/run_xskill.py train  --families all
  python drivers/run_xskill.py deploy --families all --split test        # 1 rollout, skill injected
  python drivers/run_xskill.py control --families all --split test       # shared tool-arm control
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES / "XSkill" / "eval"))

from common import data, driver, executor                            # noqa: E402
from common.paths import MODEL, gemini_api_key                       # noqa: E402

METHOD = "xskill"
ARM = "tool"


def _setup_env(model: str) -> None:
    key = gemini_api_key()
    os.environ.setdefault("EXPERIENCE_MODEL_NAME", model)
    os.environ.setdefault("EXPERIENCE_API_KEY", key)
    os.environ.setdefault("EXPERIENCE_END_POINT", "")        # empty -> google-genai SDK
    os.environ.setdefault("EXPERIENCE_USE_GENAI", "1")
    os.environ.setdefault("EXPERIENCE_EMBEDDING_API_KEY", key)
    os.environ.setdefault("GEMINI_API_KEY", key)


def _sample_args(args) -> argparse.Namespace:
    """The attribute bag XSkill's modules read with getattr()."""
    return argparse.Namespace(
        skill_enable=True, skill_refine=args.skill_refine, skill_max_length=args.skill_max_length,
        experience_max_ops=args.experience_max_ops, experience_max_items=args.experience_max_items,
        experience_refine=args.experience_refine, output_dir=str(driver.method_root(METHOD)))


def _process_sample(rec: dict, sample_dir: Path, sargs, model: str) -> dict:
    """One task: summarise its rollouts, critique them into experience ops, synthesise a skill."""
    from exskill import (ExperienceLLM, generate_skill_for_sample, intra_sample_experiences,
                         summarize_rollouts)
    out = {"doc_id": rec["doc_id"], "ops": [], "skill": None, "error": None}
    try:
        traj_paths = sorted(str(p) for p in sample_dir.glob("rollout_*/traj.jsonl"))
        if not traj_paths:
            out["error"] = "no trajectories"
            return out
        llm = ExperienceLLM(model_name=model)
        merged = summarize_rollouts(traj_paths, llm, sample_dir=str(sample_dir))
        if not merged:
            out["error"] = "empty summary"
            return out
        summaries_only = {k: v for k, v in merged.items()
                          if k not in ("question", "ground_truth", "system_prompt")}
        raw_ops = intra_sample_experiences(
            merged.get("question") or rec["problem"], merged.get("ground_truth") or rec["solution"],
            summaries_only, llm, max_ops=sargs.experience_max_ops, debug_dir=str(sample_dir),
            system_prompt=merged.get("system_prompt"))
        if isinstance(raw_ops, str):
            try:
                raw_ops = json.loads(raw_ops)
            except Exception:                                        # noqa: BLE001
                raw_ops = []
        if isinstance(raw_ops, dict) and "experiences" in raw_ops:
            # batch_merge reads {"option": add|modify, "experience": text}
            raw_ops = [{"option": "add", "experience": v} for v in raw_ops["experiences"].values()
                       if isinstance(v, str) and v.strip()]
        out["ops"] = raw_ops if isinstance(raw_ops, list) else []
        if out["ops"]:
            (sample_dir / "exp_items.json").write_text(json.dumps(out["ops"], ensure_ascii=False, indent=1))
        sk = generate_skill_for_sample({"question_id": rec["doc_id"], "sample_dir": str(sample_dir)},
                                       llm, sargs, ground_truth=str(rec["solution"]))
        if sk.get("success"):
            out["skill"] = sk["skill_content"]
        else:
            out["error"] = sk.get("error")
    except Exception as e:                                           # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


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

    _setup_env(args.model)
    from exskill import (ExperienceLLM, batch_merge, format_for_prompt, merge_skills,
                         refine_experience_library, refine_skill_document, save_library)
    sargs = _sample_args(args)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.analysts) as ex:
        futs = {ex.submit(_process_sample, r, train_dir / r["doc_id"], sargs, args.model): r
                for r in recs}
        for fut in as_completed(futs):
            results.append(fut.result())
    ops = [o for r in results for o in r["ops"]]
    skills = [r["skill"] for r in results if r["skill"]]
    log(f"   {len(results)} samples -> {len(ops)} experience ops, {len(skills)} sample skills")

    llm = ExperienceLLM(model_name=args.model)
    library = batch_merge({}, ops, llm, debug_dir=None, experience_limit=args.experience_max_items)
    if args.experience_refine and library:
        library = refine_experience_library(library, llm, debug_dir=str(d))
    save_library(str(d / "experience_library.json"), library)

    skill_doc = merge_skills("", skills, llm, sargs) if skills else ""
    if args.skill_refine and skill_doc:
        skill_doc = refine_skill_document(skill_doc, llm, skill_path=str(d / "skill.md"),
                                          word_threshold=args.skill_max_length, force_refine=True)
    (d / "skill.md").write_text(skill_doc or "")

    exp_block = format_for_prompt(library, max_items=args.experience_max_items) if library else ""
    text = "\n\n".join(x for x in (skill_doc.strip(), exp_block.strip()) if x)
    if not text.strip():
        # Never save an empty artifact: `train` skips a family that already has one, so an
        # empty artifact survives every retry and then deploys as "the method learned this".
        return {"family": fam.key, "n_train": len(recs), "error": "empty artifact",
                "errors": [r["error"] for r in results if r["error"]][:3]}
    driver.Artifact(
        method=METHOD, family=fam.key, arm=ARM, text=text, model=args.model, seed=args.seed,
        train_tasks=len(recs), rollouts_used=len(recs) * args.rollouts,
        rollouts_new=int(rep["reuse"].get("to_run", 0)), lm_calls=-1,
        notes={"n_experience_ops": len(ops), "library_size": len(library),
               "n_sample_skills": len(skills), "errors": [r["error"] for r in results if r["error"]],
               "experience_max_items": args.experience_max_items,
               "experience_max_ops": args.experience_max_ops,
               "wall_s": round(time.time() - t0)}).save()
    return {"family": fam.key, "n_train": len(recs), "ops": len(ops), "library": len(library),
            "skills": len(skills), "rollouts_new": int(rep["reuse"].get("to_run", 0)),
            "wall_s": round(time.time() - t0)}


def adapt_dir(fam: data.Family, args, split: str, log=print) -> Path:
    """XSkill Phase II: adapt the skill to each test task, materialised for --inject-dir."""
    from exskill import ExperienceLLM, adapt_skill_for_task, load_existing, format_for_prompt
    d = driver.family_dir(METHOD, fam)
    out = d / f"inject_{split}"
    out.mkdir(parents=True, exist_ok=True)
    _setup_env(args.model)
    llm = ExperienceLLM(model_name=args.model)
    base = (d / "skill.md").read_text() if (d / "skill.md").is_file() else ""
    library = load_existing(str(d / "experience_library.json"))
    exp_block = format_for_prompt(library, max_items=args.experience_max_items) if library else ""
    for rec in fam.records(split):
        f = out / f"{rec['doc_id']}.txt"
        if f.is_file():
            continue
        task = rec["problem"].replace("<image>", "").strip()
        try:
            f.write_text(adapt_skill_for_task(base, exp_block, task, llm) or base)
        except Exception as e:                                       # noqa: BLE001
            log(f"   adapt failed for {rec['doc_id']}: {type(e).__name__}: {e}")
            f.write_text(base)
    return out


def main(argv=None) -> int:
    ap = driver.parser(METHOD, ARM)
    ap.add_argument("--analysts", type=int, default=4, help="parallel per-sample analysis workers")
    ap.add_argument("--experience-max-ops", type=int, default=2)
    ap.add_argument("--experience-max-items", type=int, default=16)
    ap.add_argument("--experience-refine", action="store_true")
    ap.add_argument("--skill-refine", action="store_true")
    ap.add_argument("--skill-max-length", type=int, default=1000)
    ap.add_argument("--adapt", action="store_true",
                    help="deploy with per-task skill adaptation (XSkill Phase II) via --inject-dir")
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
            if args.adapt and not args.dry_run:
                inject = adapt_dir(fam, args, args.split, log=log)
                art = driver.load_artifact(METHOD, fam)
                tag = driver.model_tag(args.model)
                rep = executor.run(fam.records(args.split),
                                   driver.family_dir(METHOD, fam) / f"deploy_{args.split}_{tag}_adapt",
                                   skill_text=(art.text if art else None), inject_dir=inject,
                                   spec=executor.ExecSpec(model=args.model, rollouts=args.rollouts,
                                                          workers=args.workers), log=log)
            else:
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
