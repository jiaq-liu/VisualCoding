"""Adaptive replication of acquisition runs and the frozen deployment route (train-side rule only).

Rule: run replicate 1.  After replicate k, STOP when its frozen candidate was ACCEPTED by the gate and its range rule covers at least
MIN_COVER (50 %) of the train tasks; otherwise run another replicate, up to MAX_REPS (3).  Among the replicates that were run, the
deployed program is the DEPLOYABLE one with the highest leave-one-out gain -> higher train cover -> lower replicate index.
Deployable = accepted by the gate AND its range rule covers at least MIN_COVER (50 %) of the train tasks: an accepted program that still
covers less than half of train after the replicate budget is used is NOT deployed and the family falls through to the text-skill route
(programs certified on a handful of train tasks did not keep their precision on test).  `--min-cover 0` restores selection by acceptance
alone.  Test is never read by this rule.

Route (decided on the train side and frozen in <family>_<tag>.route.json; the test driver only obeys it):
  a replicate is deployable             -> route "instrument": in range = the program's answer, out of range = the executor;
  otherwise                             -> route "executor"; chain.sh then summarises the family's text skill from train rollout
                                            trajectories (verify.textskill.build; never validated) for every executor without a
                                            deployed program, and report.py ranks program > text skill > executor.

    python adaptive_rule.py decide <run_dir> [<run_dir> ...]      # prints STOP or CONTINUE (+ reason) for the runs so far
    python adaptive_rule.py select <run_dir> [...] [--out f.json] # the train-selected run (or none)
    python adaptive_rule.py route <dataset:family> <run_dir> [...] [--out f.json]"""
import os as _os, pathlib as _pl
ROOT = _os.environ.get("VC_ROOT") or str(_pl.Path(__file__).resolve().parents[2])
import json, sys
from pathlib import Path
MIN_COVER, MAX_REPS = 0.5, 3


def info(rd) -> dict:
    rd = Path(rd); mp = rd / "frozen" / "meta.json"
    if not mp.exists():
        return {"run": str(rd), "frozen": False, "accepted": False, "tier": None, "cover": 0, "n_train": None, "loo_gain": 0.0, "why": "no runnable candidate"}
    m = json.loads(mp.read_text()); g = m["gate"]
    n = (g.get("loo") or {}).get("n_tasks")
    if not n and (rd / "train_manifest.json").exists():
        n = len(json.loads((rd / "train_manifest.json").read_text())["task_ids"])
    return {"run": str(rd), "frozen": True, "accepted": bool(g.get("accepted")), "tier": m.get("tier"), "cover": g["train"]["cover"], "n_train": n,
            "loo_gain": (g.get("loo") or {}).get("gain", 0.0) or 0.0, "why": g.get("why"), "families": m.get("families")}


def stop_after(x: dict) -> tuple:
    if not x["accepted"]:
        return False, f"rejected ({str(x['why'])[:80]})"
    if x["n_train"] and x["cover"] < MIN_COVER * x["n_train"]:
        return False, f"accepted but train cover {x['cover']}/{x['n_train']} < {MIN_COVER:.0%}"
    return True, f"accepted ({x['tier']}), train cover {x['cover']}/{x['n_train']}"


def deployable(x: dict, min_cover: float = MIN_COVER) -> bool:
    return bool(x["accepted"]) and not (x["n_train"] and x["cover"] < min_cover * x["n_train"])


def decide(runs: list) -> tuple:
    xs = [info(r) for r in runs]
    ok, why = stop_after(xs[-1])
    if ok:
        return "STOP", why
    if len(xs) >= MAX_REPS:
        return "STOP", f"{why}; replicate budget ({MAX_REPS}) used"
    return "CONTINUE", why


def select(runs: list, min_cover: float = MIN_COVER):
    xs = [info(r) for r in runs]
    for x in xs:
        x["deployable"] = deployable(x, min_cover)
    acc = [(x["loo_gain"], x["cover"], -i, x) for i, x in enumerate(xs) if x["deployable"]]
    return (max(acc, key=lambda t: t[:3])[3] if acc else None), xs


def _rel(p) -> str:
    p = _os.path.abspath(str(p))
    return _os.path.relpath(p, ROOT) if p.startswith(_os.path.abspath(ROOT) + _os.sep) else p


def route(family: str, runs: list, min_cover: float = MIN_COVER) -> dict:
    """Deployment route of one family from train-side files only: "instrument" when a replicate is deployable, else "executor"."""
    s, xs = select(runs, min_cover)
    r = "instrument" if s else "executor"
    why = (f"selected {Path(s['run']).name} ({s['tier']}, train cover {s['cover']}/{s['n_train']})" if s else
           (f"no accepted instrument in {len(xs)} replicate(s)" if not any(x["accepted"] for x in xs) else
            f"accepted instrument(s) cover < {min_cover:.0%} of train in all {len(xs)} replicate(s) ({', '.join(str(x['cover']) + '/' + str(x['n_train']) for x in xs if x['accepted'])}): not deployed"))
    return {"family": family, "route": r, "why": why, "instrument_run": (_rel(s["run"]) if s else None), "out_of_range": ("executor" if s else None),
            "deploy_min_cover": min_cover, "decided_from": "train-side files only (frozen/meta.json of the replicates)", "runs": xs}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    args, opt, it = [], {}, iter(sys.argv[2:])
    for a in it:                                   # --key value options; everything else is positional
        if a.startswith("--"):
            opt[a[2:]] = next(it, None)
        else:
            args.append(a)
    if cmd == "decide":
        d, why = decide(args); print(d, "-", why)
    elif cmd == "select":
        mc = float(opt["min-cover"]) if opt.get("min-cover") is not None else MIN_COVER
        s, xs = select(args, mc)
        res = {"rule": {"min_cover": MIN_COVER, "deploy_min_cover": mc, "max_reps": MAX_REPS, "selection": "deployable (accepted and train cover >= deploy_min_cover): LOO gain -> train cover -> lower index"},
               "runs": xs, "selected": (s["run"] if s else None), "deploy": bool(s)}
        if opt.get("out"):
            Path(opt["out"]).write_text(json.dumps(res, indent=1))
        print(res["selected"] or "NONE")
    elif cmd == "route":
        res = route(args[0], args[1:], float(opt["min-cover"]) if opt.get("min-cover") is not None else MIN_COVER)
        if opt.get("out"):
            Path(opt["out"]).write_text(json.dumps(res, indent=1))
        print(res["route"], "-", res["why"])
    else:
        raise SystemExit("usage: adaptive_rule.py decide | select | route ...")
