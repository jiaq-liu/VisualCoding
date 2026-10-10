"""Per-executor deploy check (train side only, no API).  For every frozen program P and executor X:
    deploy P for X  iff  the frozen range rule keeps >= gain_min_cover (5) stored train readings that have X's train rollouts, and the one-sided 95 %
    bootstrap lower bound of mean(P correct - X pass@1 over 4 train rollouts) over those readings is > 0   (gate.gain_bound = the GAIN-tier test).
The tier's own acceptance (certified / GAIN precision, LOO, stability, cover >= 50 %) is unchanged and already passed; this adds "and P beats THIS executor on train".
The frozen rule is replayed on the stored train readings (fit_rule.range_from_row), nothing is refit.  Not deployed -> the family's route for X is the text skill.
Writes <batch>/deploy_check.json and skill_jobs.txt (lines: <dataset> <family> <executor>; the (family, executor) pairs whose route is the text skill)."""
import json, os, sys
from pathlib import Path
ROOT = Path(os.environ.get("VC_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                           # noqa: E402
from verify import instrument_eval as E                     # noqa: E402
from verify.data import load_tasks                           # noqa: E402
from verify.acquire import gate as G, fit_rule as F          # noqa: E402
OV = ROOT / "output" / "verify"; B = C.B


def frozen_train_rows(rd):
    code = (rd / "frozen" / "code.py").read_text()
    for d in sorted(rd.glob("r*_c*")):
        if d.is_dir() and (d / "code.py").exists() and (d / "code.py").read_text() == code and (d / "rows.jsonl").exists():
            return [json.loads(l) for l in open(d / "rows.jsonl")]
    return None


def train_pass1(dirs, model, tids):
    out = {}
    for d in dirs:
        p = OV / d / "results.jsonl"; got = {}
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l); q = r.get("question_id")
            if q in tids and q not in out and r.get("model") == model:
                got.setdefault(q, []).append(int(float(r.get("accuracy_score") or 0) >= 0.5))
        out.update({q: sum(v) / len(v) for q, v in got.items() if len(v) >= 4})
    return out


def main():
    out = {}; jobs = []
    for fam, (ds, run, tier) in C.PROG.items():
        rd = OV / run; tasks = {t.task_id: t for t in load_tasks([str(C.DATA / ds / f"{fam}_train.json")])}
        rows = frozen_train_rows(rd); rule = json.loads((rd / "frozen" / "rule.json").read_text())
        assert rows, f"no stored train readings of the frozen code in {rd}"
        by = {}
        for r in rows:                                        # one reading per task (sample 0)
            if r["task_id"] in tasks and (r["task_id"] not in by or str(r.get("sample")) < str(by[r["task_id"]].get("sample"))):
                by[r["task_id"]] = r
        inr = [r for r in by.values() if F.range_from_row(rule, r, tasks[r["task_id"]])]
        rec = {"dataset": ds, "run": run, "tier": tier, "n_train": len(tasks), "n_in_range": len(inr),
               "program_train_precision": round(sum(int(r.get("correct") == 1) for r in inr) / max(len(inr), 1), 3)}
        for x in C.EXECUTORS:
            eye = train_pass1(C.TRAIN[x], C.MODEL[x], set(tasks))
            pairs = [(int(r.get("correct") == 1), eye.get(r["task_id"])) for r in inr]; gs = G.gain_bound(pairs, G.DEFAULTS["gain_alpha"])
            ok = gs["n"] >= G.DEFAULTS["gain_min_cover"] and gs["lower"] is not None and gs["lower"] > 0
            rec[x] = {"n_with_reference": gs["n"], "executor_train_pass1": round(E._mean([e for _, e in pairs if e is not None]), 3) if gs["n"] else None,
                      "gain_mean": None if gs["mean"] is None else round(gs["mean"], 4), "gain_lower_95": None if gs["lower"] is None else round(gs["lower"], 4), "deploy": bool(ok)}
            if not ok:
                jobs.append((ds, fam, x))
        out[fam] = rec
        print(f"{fam:28s} {tier:9s} prec {rec['program_train_precision']:.2f} on {len(inr):2d}/{len(tasks):2d} | " + " | ".join(
            f"{x}: exec {rec[x]['executor_train_pass1']} lower {rec[x]['gain_lower_95']} -> {'DEPLOY' if rec[x]['deploy'] else 'skill'}" for x in C.EXECUTORS))
    for ds, fams in C.NO_PROGRAM.items():
        jobs += [(ds, f, x) for f in fams for x in C.EXECUTORS]
    (B / "deploy_check.json").write_text(json.dumps(out, indent=1))
    (B / "skill_jobs.txt").write_text("".join(f"{ds} {f} {x}\n" for ds, f, x in jobs))


if __name__ == "__main__":
    main()
