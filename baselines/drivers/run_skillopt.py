#!/usr/bin/env python
"""SkillOpt baseline — the skill document as trainable state, tool arm.

SkillOpt's training loop is used as published (rollout -> reflect -> aggregate -> update)
through a new environment package, `skillopt/envs/visqa/`, modelled on the shipped `docvqa`
env.  Three things are specific to this study:

  * the env's rollout helper calls the shared code-interpreter executor instead of a single
    chat completion, so SkillOpt's skill is trained against the same agent as every other
    method here;
  * SINGLE PASS: `--batch-size 0` makes the family's train set one batch, so a run is one
    rollout -> reflect -> aggregate -> rewrite -> accept.  The initial skill is empty, so
    that single rollout pass is served entirely from the acq pool and SkillOpt pays for no
    new training rollouts at all;
  * no held-out validation: the family's own train records are written into the `val/`
    split, `evaluation.use_gate=false` force-accepts each edit, and `--sel-env-num 0` turns
    the selection rollouts off — with the gate disabled they only bought a number nobody
    read (see the `selection_off` branch in skillopt/engine/trainer.py).

The optimizer role talks to gemini-3.1-pro through `skillopt/model/gemini_backend.py`.

  python drivers/run_skillopt.py train  --families all
  python drivers/run_skillopt.py deploy --families all --split test
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

from common import data, driver, executor                          # noqa: E402
from common.paths import IMAGE_ROOT, MODEL, subprocess_env         # noqa: E402

METHOD = "skillopt"
ARM = "tool"
SKILLOPT = BASELINES / "SkillOpt"


def _py() -> str:
    v = BASELINES / ".venvs" / "skillopt" / "bin" / "python"
    return str(v if v.is_file() else sys.executable)


def make_splits(fam: data.Family, d: Path) -> Path:
    """train/ val/ test/ for SkillOpt's split_dir loader.

    val/ and test/ both hold the family's TRAIN records: the protocol gives these methods no
    held-out data, and writing train into test/ means that even an accidental test-time
    evaluation inside SkillOpt cannot touch the held-out split.
    """
    splits = d / "splits"
    train_recs = fam.records("train")
    for name in ("train", "val", "test"):
        p = splits / name
        p.mkdir(parents=True, exist_ok=True)
        data.write_input(train_recs, p / f"{fam.slug}.json")
    (splits / "README.txt").write_text(
        "val/ and test/ are copies of train/: this study gives SkillOpt no held-out data and\n"
        "the real test split is scored outside SkillOpt (drivers/run_skillopt.py deploy).\n")
    return splits


def train_family(fam: data.Family, args, log=print) -> dict:
    d = driver.family_dir(METHOD, fam)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "artifact.json").is_file() and not args.force:
        return {"family": fam.key, "skipped": "artifact exists"}

    recs = fam.records("train")
    if args.dry_run:
        pre = executor.run(recs, d / "_dry", skill_text=None,
                           spec=executor.ExecSpec(model=args.model, rollouts=args.rollouts),
                           dry_run=True, log=log)
        return {"family": fam.key, "n_train": len(recs), "seed_pass": pre["reuse"],
                "epochs": args.epochs, "dry_run": True}

    splits = make_splits(fam, d)
    out_root = d / "run"
    t0 = time.time()
    cmd = [_py(), "-u", "scripts/train.py", "--config", "configs/visqa/default.yaml",
           "--cfg-options",
           f"env.split_dir={splits}", f"env.image_root={IMAGE_ROOT}",
           f"env.rollouts={args.rollouts}", f"env.workers={args.workers}",
           f"out_root={out_root}", f"train.num_epochs={args.epochs}",
           f"train.batch_size={min(args.batch_size, len(recs)) if args.batch_size > 0 else 0}",
           f"evaluation.sel_env_num={min(args.sel_env_num, len(recs)) if args.sel_env_num > 0 else 0}",
           f"model.optimizer={args.model}", f"model.target={args.model}",
           f"train.seed={args.seed}"]
    with open(d / "train.log", "a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=str(SKILLOPT), env=subprocess_env(), stdout=fh, stderr=fh).returncode

    best = out_root / "best_skill.md"
    text = best.read_text() if best.is_file() else ""
    if rc != 0 or not text.strip():
        # No artifact on failure.  `train` skips a family that already has an artifact, so
        # saving an empty one here makes the failure permanent and silent: the retry skips
        # it and the deployment injects nothing while reporting as if SkillOpt had learned.
        return {"family": fam.key, "n_train": len(recs), "error": f"returncode {rc}",
                "skill_chars": len(text), "log": str(d / "train.log"),
                "tail": (d / "train.log").read_text()[-400:] if (d / "train.log").is_file() else ""}
    summary = {}
    if (out_root / "summary.json").is_file():
        try:
            summary = json.loads((out_root / "summary.json").read_text())
        except Exception:                                          # noqa: BLE001
            summary = {}
    driver.Artifact(
        method=METHOD, family=fam.key, arm=ARM, text=text, model=args.model, seed=args.seed,
        train_tasks=len(recs), rollouts_used=-1, rollouts_new=-1, lm_calls=-1,
        notes={"returncode": rc, "epochs": args.epochs, "batch_size": args.batch_size,
               "sel_env_num": args.sel_env_num, "out_root": str(out_root),
               "summary": {k: summary.get(k) for k in ("train_hard", "best_val", "final_skill_len")},
               "wall_s": round(time.time() - t0)}).save()
    return {"family": fam.key, "n_train": len(recs), "returncode": rc,
            "skill_chars": len(text), "wall_s": round(time.time() - t0)}


def main(argv=None) -> int:
    ap = driver.parser(METHOD, ARM)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 = the whole family in one step (single pass); >0 restores the "
                         "per-batch edit loop, which pays for fresh rollouts after batch 1")
    ap.add_argument("--sel-env-num", type=int, default=0,
                    help="selection-set size; 0 = no selection rollouts (the gate is off)")
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
