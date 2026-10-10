"""Helpers of the route-aware test driver (run_test.sh).  No API call is made here.

    route_tools.py get <route.json> <field>                       # one field of the frozen route ("" when null)
    route_tools.py oor-ids <instrument results.jsonl>             # comma list of the OUT-OF-RANGE task ids of a deployed instrument
    route_tools.py skill-tested <skill.md>                        # prints where this skill was already measured on test (empty = never)
    route_tools.py mark-tested <skill.md> <where>                 # records the test measurement in the skill's skill_meta.json
    route_tools.py report-skill <source:family> <skill rollout dir> --split test --baseline <dir> [...] --out <prefix>
                                                                  # executor + skill vs the stored plain executor rollouts, paired task bootstrap"""
import os as _os, pathlib as _pl
ROOT = _os.environ.get("VC_ROOT") or str(_pl.Path(__file__).resolve().parents[2])
import collections, json, sys
from pathlib import Path


def oor_ids(results_jsonl) -> list:
    by = collections.defaultdict(list)
    for l in open(results_jsonl):
        r = json.loads(l); by[r["task_id"]].append(bool(r["in_range"]))
    return sorted(t for t, v in by.items() if not sum(v) * 2 > len(v))          # majority over the rollouts


def _meta_path(skill_md) -> Path:
    return Path(skill_md).with_name("skill_meta.json")


def report_skill(family: str, skill_dir: str, split: str, baselines: list, out: str) -> dict:
    sys.path.insert(0, ROOT)
    from verify import instrument_eval as E
    src, fam = family.split(":", 1)
    tasks = {t.task_id: t for t in E.split_tasks(split, {fam}) if t.source == src and t.family == fam}

    def per_task(dirs):
        res = {}
        for tid, cs in E.load_eyeball([str(d) for d in dirs if Path(d).exists()], tasks).items():
            if cs:
                mk = E._majority([c["key"] for c in cs])
                res[tid] = {"e1": E._mean([c["correct"] for c in cs]), "n": len(cs), "maj": int(next((c["correct"] for c in cs if c["key"] == mk), 0)),
                            "p4": int(any(c["correct"] for c in cs))}
        return res
    base, sk = per_task(baselines), per_task([skill_dir])
    ids = [t for t in sorted(tasks) if t in base and t in sk]
    pairs = [(base[t]["e1"], sk[t]["e1"]) for t in ids]
    lo, hi = E._boot_ci(pairs) if pairs else (None, None)
    b, s = E._mean([p[0] for p in pairs]), E._mean([p[1] for p in pairs])
    res = {"family": family, "split": split, "n_tasks": len(tasks), "n_paired": len(ids), "n_skill_rollouts": sum(sk[t]["n"] for t in ids),
           "executor_pass1": b, "skill_pass1": s, "delta": (s - b if pairs else None), "ci95": [lo, hi],
           "maj4": [E._mean([base[t]["maj"] for t in ids]), E._mean([sk[t]["maj"] for t in ids])],
           "pass4": [E._mean([base[t]["p4"] for t in ids]), E._mean([sk[t]["p4"] for t in ids])],
           "fix_mass": sum(max(0.0, y - x) for x, y in pairs), "harm_mass": sum(max(0.0, x - y) for x, y in pairs),
           "baseline_dirs": [str(d) for d in baselines], "skill_rollouts": str(skill_dir)}
    L = [f"# {family} — executor + text skill on {split} (measured once)", "",
         "Reference = the stored plain executor rollouts (same executor, judge and settings). Δ = skill pass@1 − plain pass@1, paired bootstrap over tasks.", "",
         "| family | n | rollouts (skill) | plain pass@1 | skill pass@1 | Δ pass@1 [95%] | maj@4 plain → skill | pass@4 plain → skill | fix / harm mass |", "|---|---|---|---|---|---|---|---|---|"]
    if pairs:
        L.append(f"| {family} | {len(ids)} | {res['n_skill_rollouts']} | {b:.3f} | **{s:.3f}** | {s - b:+.3f} [{lo:+.2f},{hi:+.2f}] | {res['maj4'][0]:.3f} → {res['maj4'][1]:.3f} | "
                 f"{res['pass4'][0]:.3f} → {res['pass4'][1]:.3f} | {res['fix_mass']:.2f} / {res['harm_mass']:.2f} |")
    else:
        L.append(f"| {family} | 0 | | | | no task has both plain and skill rollouts | | | |")
    Path(out + ".md").write_text("\n".join(L) + "\n"); Path(out + ".json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))
    return res


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "get":
        v = json.loads(Path(args[0]).read_text()).get(args[1]); print("" if v is None else v)
    elif cmd == "oor-ids":
        print(",".join(oor_ids(args[0])))
    elif cmd == "skill-tested":
        mp = _meta_path(args[0]); print((json.loads(mp.read_text()).get("test_measured") or "") if mp.exists() else "")
    elif cmd == "mark-tested":
        mp = _meta_path(args[0]); m = json.loads(mp.read_text()) if mp.exists() else {}
        m["test_measured"] = args[1]; mp.write_text(json.dumps(m, indent=1))
    elif cmd == "report-skill":
        pos, opt, key = [], collections.defaultdict(list), None
        for a in args:
            if a.startswith("--"):
                key = a[2:]
            elif key:
                opt[key].append(a)
            else:
                pos.append(a)
        report_skill(pos[0], pos[1], (opt.get("split") or ["test"])[0], opt.get("baseline", []), opt["out"][0])
    else:
        raise SystemExit(__doc__)
