"""Report of one pipeline batch on the frozen split (no API): every evaluated executor, single-rollout test protocol, with and without the
per-executor deploy check (deploy_check.json).
Per task, executor X, routing R in {check, nocheck}: b = X's plain rollout 0 (LLM judge); p = the frozen program's stored reading when the program is deployed
for X under R and the reading is in range (exact match); X + text skill (rollout 0) when the family's route is the text skill and a skill test exists; else b.
  check   = program deployed for X iff deploy_check.json says so; otherwise text skill.
  nocheck = every train-selected program deployed for every executor.   Also audits every program: distinct test answers, code lines, api_calls (model-read).
    python scripts/pipeline/report.py   -> output/verify/<batch>/{report.md, summary.json, policy_rows.jsonl, costs.json}"""
import collections, json, os, sys
from pathlib import Path
ROOT = Path(os.environ.get("VC_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                           # noqa: E402
from verify import instrument_eval as E                     # noqa: E402
from verify.acquire.loop import est_usd                      # noqa: E402
OV = ROOT / "output" / "verify"; B = C.B; DS = C.DATA
AUDIT_MR = set()
mean = lambda xs: (sum(xs) / len(xs)) if xs else None
f3 = lambda x, s=False: "—" if x is None else (f"{x:+.3f}" if s else f"{x:.3f}")


def rollout0(dirs, model, tids):
    out = {}
    for d in dirs:
        p = OV / d / "results.jsonl"
        if not p.exists():
            continue
        got = {}
        for l in open(p):
            r = json.loads(l); q = r.get("question_id")
            if q in tids and q not in out and r.get("model") == model and int(r.get("_rollout_idx") or 0) == 0:
                got[q] = int(float(r.get("accuracy_score") or 0) >= 0.5)
        out.update(got)
    return out


def program_rows(run):
    out = {}
    for sp in ("test",):
        p = OV / run / sp / "results.jsonl"
        if not p.exists():
            continue
        by = collections.defaultdict(list)
        for l in open(p):
            r = json.loads(l); by[r["task_id"]].append(r)
        for tid, rs in by.items():
            r = sorted(rs, key=lambda r: r["sample"])[0]
            out[tid] = (bool(r["in_range"]), None if r.get("correct") is None else int(r["correct"] == 1))
    return out


def usage(d):
    p = OV / d / "usage.jsonl"; tot = 0.0
    if not p.exists():
        return None
    for l in open(p):
        for m, u in json.loads(l)["usage"].items():
            tot += est_usd(m, u["in_tokens"], u["out_tokens"], u["think_tokens"])[0] or 0.0
    n = sum(1 for _ in open(OV / d / "results.jsonl"))
    return {"usd": round(tot, 2), "rollouts": n, "usd_per_rollout": round(tot / max(n, 1), 4)}


def main():
    chk = json.loads((B / "deploy_check.json").read_text()) if (B / "deploy_check.json").exists() else {}
    global AUDIT_MR
    audit = {}
    for fam, (ds, run, tier) in C.PROG.items():
        rd = OV / run; meta = json.loads((rd / "frozen" / "meta.json").read_text()); code = (rd / "frozen" / "code.py").read_text(); tr = []
        p = rd / "test" / "results.jsonl"
        if p.exists():
            tr = [json.loads(l) for l in open(p)]
        cost = meta.get("cost") or {}
        audit[fam] = {"run": run, "tier": tier, "model_read_meta": meta.get("model_read"), "api_calls_test": sum(int(r.get("api_calls") or 0) for r in tr), "n_test_readings": len(tr),
                      "distinct_test_answers": len({str(r.get("answer")) for r in tr if r.get("answer") is not None}), "code_lines": len([l for l in code.splitlines() if l.strip()]),
                      "train_cover": (meta.get("gate") or {}).get("train", {}).get("cover"), "writer_usd": cost.get("est_usd_writer")}
    AUDIT_MR = {f for f, v in audit.items() if v["api_calls_test"] > 0 or v["model_read_meta"]}
    rows = []; fam_rows = []
    for ds in C.DATASETS:
        recs = [r for r in json.load(open(DS / ds / "test.json")) if r["family"] in C.RUN[ds]]; all_t = {r["doc_id"] for r in recs}
        base = {x: rollout0(C.TEST[x], C.MODEL[x], all_t) for x in C.MODEL}
        for fam in C.RUN[ds]:
            tids = [r["doc_id"] for r in recs if r["family"] == fam]; has_prog = fam in C.PROG; prog = program_rows(C.PROG[fam][1]) if has_prog else {}
            rec = {"dataset": ds, "family": fam, "n": len(tids), "tier": C.PROG[fam][2] if has_prog else None}
            for x in C.MODEL:
                skill = rollout0([f"{C.BATCH}/skilltest_{fam}_{x}"], C.MODEL[x], set(tids))
                dep_check = has_prog and chk.get(fam, {}).get(x, {}).get("deploy", False)
                dep_nocheck = has_prog
                for R, dep in (("check", dep_check), ("nocheck", dep_nocheck)):
                    route = "instrument" if dep else "text_skill"
                    for tid in tids:
                        if tid not in base[x]:
                            continue
                        b = float(base[x][tid]); take = 0; c = None
                        if dep and prog.get(tid) and prog[tid][0] and prog[tid][1] is not None:
                            take, c, p = 1, prog[tid][1], float(prog[tid][1])
                        elif not dep and tid in skill:
                            p = float(skill[tid])
                        else:
                            p = b
                        rows.append({"dataset": ds, "family": fam, "task_id": tid, "executor": x, "routing": R, "route": route, "baseline_acc": b, "takeover": take,
                                     "program_correct": c, "skill_acc": skill.get(tid) if not dep else None, "policy_acc": p})
                    sel = [r for r in rows if r["family"] == fam and r["executor"] == x and r["routing"] == R]; pairs = [(r["baseline_acc"], r["policy_acc"]) for r in sel]
                    lo, hi = E._boot_ci(pairs) if pairs else (None, None); inr = [r for r in sel if r["takeover"]]
                    rec[f"{x}_{R}"] = {"route": route, "n_done": len(pairs), "base": mean([a for a, _ in pairs]), "policy": mean([b for _, b in pairs]), "delta": mean([b - a for a, b in pairs]),
                                       "ci": [lo, hi], "cover": (len(inr) / len(pairs) if pairs else None), "precision_in": mean([r["program_correct"] for r in inr]),
                                       "n_skill": sum(r["skill_acc"] is not None for r in sel)}
            fam_rows.append(rec)

    def pool(keep, off=frozenset()):
        pairs = [(r["baseline_acc"], r["baseline_acc"] if r["family"] in off else r["policy_acc"]) for r in rows if keep(r)]
        lo, hi = E._boot_ci(pairs) if pairs else (None, None)
        return {"n": len(pairs), "base": mean([a for a, _ in pairs]), "policy": mean([b for _, b in pairs]), "delta": mean([b - a for a, b in pairs]), "ci": [lo, hi]}
    cell = lambda p: f"{f3(p['base'])} → **{f3(p['policy'])}** {f3(p['delta'], True)} [{f3(p['ci'][0], True)},{f3(p['ci'][1], True)}]"
    L = [f"# Batch {C.BATCH} on the frozen split: executors {', '.join(C.EXECUTORS)}, single-rollout test, per-executor deploy check", "",
         "b = the executor's plain rollout 0 (LLM judge; pro thinking_level = low, flash-lite thinking off); method = deployed program's reading in range (exact match) / executor + trajectory text skill "
         "(1 rollout) when the route is the text skill / b otherwise. Δ = mean(p − b), paired task bootstrap. `check` = program deployed for X only if it beats X on train "
         "(one-sided 95 % lower bound of the paired train gain > 0); `nocheck` = the original routes.", "", "## Deploy check (train only)", "",
         "| family | tier | program train precision (in range / train) | " + " | ".join(f"{x} train pass@1 | gain lower | deploy" for x in C.EXECUTORS) + " |", "|---|---|---|" + "---|---|---|" * len(C.EXECUTORS)]
    for fam, v in chk.items():
        L.append(f"| {fam} | {v['tier']} | {v['program_train_precision']:.2f} ({v['n_in_range']}/{v['n_train']}) | " + " | ".join(
            f"{f3(v[x]['executor_train_pass1'])} | {f3(v[x]['gain_lower_95'], True)} | {'yes' if v[x]['deploy'] else '**no**'}" for x in C.EXECUTORS) + " |")
    L += ["", "## Program audit", "", "| family | run | tier | train cover | code lines | distinct test answers / readings | api_calls on test (model-read if > 0) | writer USD |", "|---|---|---|---|---|---|---|---|"]
    for fam, v in audit.items():
        L.append(f"| {fam} | {v['run']} | {v['tier']} | {v['train_cover']} | {v['code_lines']} | {v['distinct_test_answers']} / {v['n_test_readings']} | {v['api_calls_test']} | {v['writer_usd']} |")
    L += ["", "No deployable program: " + ", ".join(f"{f} ({C.ROUTE_WHY.get(f)})" for ds in C.DATASETS for f in C.NO_PROGRAM[ds])]
    L.append(""); pooled = {}
    for x in C.MODEL:
        L += [f"## Executor {x}", "", "| family | tier | route (check) | n (done) | executor | with check: method, Δ [95%] | cover | precision in range | route (nocheck) | without check: method, Δ [95%] |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in fam_rows:
            a, b = r[f"{x}_check"], r[f"{x}_nocheck"]
            L.append(f"| {r['family']} | {r['tier'] or ''} | {a['route']} | {r['n']} ({a['n_done']}) | {f3(a['base'])} | **{f3(a['policy'])}** {f3(a['delta'], True)} [{f3(a['ci'][0], True)},{f3(a['ci'][1], True)}] | "
                     f"{f3(a['cover'])} | {f3(a['precision_in'])} | {b['route']} | {f3(b['policy'])} {f3(b['delta'], True)} [{f3(b['ci'][0], True)},{f3(b['ci'][1], True)}] |")
        L += ["", "| pool | n | with check | without check |", "|---|---|---|---|"]
        groups = [(ds, (lambda r, ds=ds: r["dataset"] == ds), frozenset()) for ds in C.DATASETS]
        groups += [(f"{ds}, model-read programs not deployed", (lambda r, ds=ds: r["dataset"] == ds), frozenset(AUDIT_MR)) for ds in C.DATASETS]
        groups += [("families routed to a program (check)", lambda r: r["route"] == "instrument", frozenset()), ("families routed to the skill", lambda r: r["route"] == "text_skill", frozenset()),
                   ("all families of the batch", lambda r: True, frozenset())]
        pooled[x] = {}
        for label, keep, off in groups:
            pc = pool(lambda r, keep=keep: r["executor"] == x and r["routing"] == "check" and keep(r), off); pn = pool(lambda r, keep=keep: r["executor"] == x and r["routing"] == "nocheck" and keep(r), off)
            pooled[x][label] = {"check": pc, "nocheck": pn}
            L.append(f"| **{label}** | {pc['n']} / {pn['n']} | {cell(pc)} | {cell(pn)} |")
        L.append("")
    costs = {d.name: usage(f"{C.BATCH}/{d.name}") for d in sorted(B.iterdir()) if d.is_dir() and (d / "results.jsonl").exists()}
    sk = {}
    for p in C.SKILL_STORE.glob("*/skill_meta.json"):
        m = json.loads(p.read_text()); sk[p.parent.name] = {"usd": (m.get("cost") or {}).get("usd"), "executor": m.get("executor"), "n_rollouts": m.get("n_rollouts"), "train_pass1": m.get("train_pass1")}
    tot = sum(v["usd"] for v in costs.values() if v); tsk = sum(v["usd"] or 0 for v in sk.values())
    L += ["## Cost of the rollouts and skills (writer cost: see the audit table; tokens measured, USD estimated with verify/acquire/prices.json; judge calls not included)", "",
          f"rollouts {tot:.2f} USD: " + ", ".join(f"{k} {v['usd']} ({v['rollouts']}, {v['usd_per_rollout']}/rollout)" for k, v in costs.items() if v) + f"; skill summaries {tsk:.2f} USD ({len(sk)} skills)."]
    (B / "report.md").write_text("\n".join(L) + "\n"); (B / "summary.json").write_text(json.dumps({"families": fam_rows, "pooled": pooled, "deploy_check": chk, "audit": audit}, indent=1))
    (B / "costs.json").write_text(json.dumps({"rollouts": costs, "skills": sk, "total_rollouts_usd": round(tot, 2), "skill_summaries_usd": round(tsk, 2)}, indent=1))
    with open(B / "policy_rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
