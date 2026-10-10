"""Shared driver contract: where artifacts live, how budget is counted, common CLI flags.

Every baseline gets one driver with the same two verbs:

    python run_<method>.py train  --families all      # per family: learn a skill / prompt
    python run_<method>.py deploy --families all --split test

`train` writes <BASELINE_OUT>/<method>/<family_slug>/artifact.md (+ artifact.json with the
provenance and the budget it consumed); `deploy` runs the frozen artifact over a split —
through the tool executor or the tool-free arm, depending on the method's arm.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from . import data, executor, vlm
from .paths import MODEL, OUT_ROOT

ARMS = ("tool", "no_tool")


def method_root(method: str) -> Path:
    return Path(OUT_ROOT) / method


def family_dir(method: str, fam: data.Family) -> Path:
    return method_root(method) / fam.slug


@dataclass
class Artifact:
    method: str
    family: str
    arm: str
    text: str
    model: str = MODEL
    seed: int = 0
    created: str = ""
    train_tasks: int = 0
    rollouts_used: int = 0          # rollouts this family's training actually consumed (new + reused)
    rollouts_new: int = 0           # the ones that cost an API call
    lm_calls: int = 0               # optimizer / reflection calls
    notes: dict = None

    def save(self) -> Path:
        d = family_dir(self.method, data.family(self.family))
        d.mkdir(parents=True, exist_ok=True)
        (d / "artifact.md").write_text(self.text or "")
        self.created = self.created or time.strftime("%Y-%m-%dT%H:%M:%S")
        (d / "artifact.json").write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False))
        return d / "artifact.md"


def load_artifact(method: str, fam: data.Family) -> Artifact | None:
    p = family_dir(method, fam) / "artifact.json"
    if not p.is_file():
        return None
    return Artifact(**json.loads(p.read_text()))


def artifact_text(method: str, fam: data.Family) -> str | None:
    a = load_artifact(method, fam)
    return a.text if a else None


# ----------------------------------------------------------------------------- deployment
def model_tag(model: str) -> str:
    """Short, filesystem-safe name for the executor a run used.

    It goes in every tool-arm output path.  The paper deploys the same (pro-learned)
    artifact on both executors, and `executor.complete()` decides "already done" from the
    output dir alone — without the tag the flash-lite deployment would find the pro results
    sitting there and run nothing at all.
    """
    m = (model or MODEL).lower()
    if "flash-lite" in m or "flashlite" in m:
        return "flashlite"
    if "pro" in m:
        return "pro"
    return "".join(c if c.isalnum() else "-" for c in m).strip("-")


def deploy_family(method: str, fam: data.Family, arm: str, split: str = "test", *,
                  rollouts: int = 1, workers: int = 8, model: str = MODEL, dry_run: bool = False,
                  spec: executor.ExecSpec | None = None, log=print) -> dict:
    """Run the frozen artifact of one family over a split (test = 1 rollout by protocol)."""
    art = load_artifact(method, fam)
    if art is None:
        return {"family": fam.key, "skipped": "no artifact"}
    recs = fam.records(split)
    if arm == "tool":
        sp = spec or executor.ExecSpec(model=model, rollouts=rollouts, workers=workers)
        out = family_dir(method, fam) / f"deploy_{split}_{model_tag(sp.model)}"
        rep = executor.run(recs, out, skill_text=art.text, spec=sp, dry_run=dry_run, log=log)
    else:
        rep = vlm.run(recs, art.text, rollouts=rollouts, model=model, workers=workers,
                      dry_run=dry_run, log=log)
    rep["family"] = fam.key
    return rep


def control_family(fam: data.Family, arm: str, split: str = "test", *, rollouts: int = 1,
                   workers: int = 8, model: str = MODEL, dry_run: bool = False, log=print) -> dict:
    """The no-skill control of an arm — shared by every method on that arm."""
    recs = fam.records(split)
    if arm == "tool":
        out = Path(OUT_ROOT) / "_control_tool" / model_tag(model) / f"{split}" / fam.slug
        rep = executor.run(recs, out, skill_text=None,
                           spec=executor.ExecSpec(model=model, rollouts=rollouts, workers=workers),
                           dry_run=dry_run, log=log)
    else:
        rep = vlm.run(recs, None, rollouts=rollouts, model=model, workers=workers,
                      dry_run=dry_run, log=log)
    rep["family"] = fam.key
    return rep


# ----------------------------------------------------------------------------- CLI
def select_families(spec: str) -> list[data.Family]:
    """'all' | dataset name(s) | family keys ('tirbench:color') | any comma mix of those."""
    if not spec or spec == "all":
        return data.families()
    if spec in data.DATASETS:
        return data.families([spec])
    out: list[data.Family] = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        out += data.families([tok]) if tok in data.DATASETS else [data.family(tok)]
    return out


def parser(method: str, arm: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"run_{method}.py",
                                 description=f"{method} baseline ({arm} arm) on data_split")
    ap.add_argument("verb", choices=["train", "deploy", "control", "status"])
    ap.add_argument("--families", default="all", help="all | <dataset> | comma list of family keys")
    ap.add_argument("--split", default="test", help="split for deploy/control")
    ap.add_argument("--rollouts", type=int, default=None, help="default: 4 for train, 1 for deploy")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="plan only: report reuse / cost, call nothing")
    ap.add_argument("--force", action="store_true", help="retrain families that already have an artifact")
    ap.add_argument("--limit", type=int, default=None, help="first N families (smoke runs)")
    return ap


def status(method: str, families: list[data.Family]) -> dict:
    """What has been learned / deployed so far."""
    out = {"method": method, "families": len(families), "with_artifact": 0, "deployed": {}}
    for f in families:
        a = load_artifact(method, f)
        if a:
            out["with_artifact"] += 1
        for d in sorted(family_dir(method, f).glob("deploy_*")):
            if d.is_dir():
                out["deployed"][d.name] = out["deployed"].get(d.name, 0) + 1
    return out
