"""One advisory text skill per family, summarised from the EXECUTOR'S OWN TRAIN ROLLOUT TRAJECTORIES with the agent harness's
skill-summarisation recipe (no validation split is read; the skill is deployed as written):
(1) SINGLE_ROLLOUT_SUMMARY per rollout; (2) per task SUCCESS / FAILURE structured analysts over the labelled summaries + the task image
-> structured entries -> tips (the agent harness's renderer); (3) FUSE_TIPS_INTO_SKILL_PROMPT over batches of tips; (4) one SKILL_REFINE_PROMPT pass.
Source rollouts = the stored train baselines of the deployment executor (the rollouts the acquisition loop already uses as its reference),
restricted to the family's TRAIN tasks.  Nothing from test is read.

    python -m verify.textskill.build --family tirbench:contrast [mira:mr_billiards ...] [--store output/verify/text_skills] [--dry-run]

Output per family: <store>/<source>__<family>/{summaries.jsonl, tips.jsonl, fuse_steps/, skill_pre_refine.md, skill.md, skill_meta.json}.
A family whose skill.md already exists is skipped (the stored skill is the deployed one).  Exit code 3 = no train rollouts for a family."""
from __future__ import annotations

import argparse, collections, datetime, hashlib, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .. import instrument_eval as E
from .. import instruments as I
from .._paths import PKG_ROOT
from ..acquire.loop import est_usd
from ..env import setup_env
from .harness_prompts import experience_prompts as EP, skill_prompts as SP, structured_render as SR

WRITER, EXECUTOR = "gemini-3.1-pro-preview", "gemini-3.1-flash-lite"
OV = PKG_ROOT / "output" / "verify"
STORE = OV / "text_skills"
MAX_OPS, TIP_BATCH = 3, 12
SYS = "You are a careful analyst of visual-reasoning agent trajectories. Follow the output format requested by the user exactly."


def skill_dir(family: str, store=None) -> Path:
    src, fam = family.split(":", 1)
    return Path(store or STORE) / f"{src}__{fam}"


def default_train_dirs() -> list[Path]:
    """Finished executor train baselines (`baseline_train*` with a _done.flag); rows are filtered by model later."""
    return [d for d in sorted(OV.glob("baseline_train*")) if d.is_dir() and (OV / f"{d.name}_done.flag").exists() and (d / "results.jsonl").exists()]


def rollouts_of(src: str, fam: str, dirs, executor: str) -> tuple[dict, list[str]]:
    by, used = {}, []
    for d in dirs:
        p = Path(d) / "results.jsonl"
        if not p.exists():
            continue
        got = collections.defaultdict(list)
        for l in open(p):
            r = json.loads(l)
            if r.get("source") == src and r.get("family") == fam and r.get("model") == executor and r["question_id"] not in by:
                got[r["question_id"]].append(r)          # the first dir that has a task wins (as load_eyeball)
        if got:
            by.update(got); used.append(str(d))
    return {k: sorted(v, key=lambda r: int(r["_rollout_idx"])) for k, v in by.items()}, used


def _ok(r: dict) -> bool:
    return float(r.get("accuracy_score") or 0) >= 0.5


def traj_text(r: dict) -> str:
    t = r.get("trajectory_text") or ""
    i = t.find("**user**")                      # drop the system prompt (identical for every rollout)
    ok = _ok(r)
    head = (f"Authoritative rollout outcomes: this rollout status={'success' if ok else 'failure'} (accuracy={1.0 if ok else 0.0}); "
            f"final answer: {r.get('final_answer')!r}; correct answer: {r.get('ground_truth')!r}; termination: {r.get('termination')}.\n\n")
    return head + (t[i:] if i >= 0 else t)


def _clean(resp: str) -> str:
    return re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", (resp or "").strip())


def _valid(doc: str) -> bool:
    return doc.startswith("---") and "## Watch out for" in doc


def build_skill(family: str, out: Path, call, counter: dict, train_dirs, executor: str, writer: str, workers: int = 8, limit_tasks: int = 0,
                dry_run: bool = False) -> dict | None:
    src, fam = family.split(":", 1)
    t0 = time.time(); c0 = dict(counter)
    tasks = {t.task_id: t for t in E.split_tasks("train", {fam}) if t.source == src and t.family == fam}
    R, used = rollouts_of(src, fam, train_dirs, executor)
    R = {tid: rs for tid, rs in R.items() if tid in tasks}                     # TRAIN tasks only
    if limit_tasks:
        R = dict(sorted(R.items())[:limit_tasks])
    n_roll = sum(len(v) for v in R.values())
    p1 = sum(_ok(r) for v in R.values() for r in v) / max(1, n_roll)
    print(f"[skill] {family}: {len(tasks)} train tasks, {len(R)} with stored {executor} rollouts ({n_roll} rollouts, pass@1 {p1:.3f}) from {[Path(d).name for d in used]}", flush=True)
    if not R:
        print(f"[skill] {family}: no train rollouts of {executor} -> run the train baseline first (scripts/baselines/run_*_train_baseline_flashlite.sh)", flush=True)
        return None
    if dry_run:
        return {"family": family, "n_tasks": len(R), "n_rollouts": n_roll, "train_pass1": p1, "dry_run": True}
    out.mkdir(parents=True, exist_ok=True)
    # ---- stage 1: per-rollout summaries (cached)
    sp = out / "summaries.jsonl"
    S = {(x["task_id"], x["rollout"]): x for x in map(json.loads, open(sp))} if sp.exists() else {}
    todo = [(tid, r) for tid, rs in sorted(R.items()) for r in rs if (tid, int(r["_rollout_idx"])) not in S]

    def _sum(item):
        tid, r = item
        txt = call(I.msgs([], EP.SINGLE_ROLLOUT_SUMMARY.format(trajectory=traj_text(r)), sys=SYS))
        return {"task_id": tid, "rollout": int(r["_rollout_idx"]), "success": _ok(r), "summary": txt}
    with ThreadPoolExecutor(workers) as ex, open(sp, "a") as fh:
        for x in ex.map(_sum, todo):
            S[(x["task_id"], x["rollout"])] = x; fh.write(json.dumps(x, ensure_ascii=False) + "\n"); fh.flush()
    print(f"[skill] {family}: stage 1 done ({len(S)} summaries, {sum(1 for x in S.values() if not x['summary'])} empty)", flush=True)
    # ---- stage 2: per-task success / failure analysts -> structured entries -> tips (cached)
    tp = out / "tips.jsonl"
    T = [json.loads(l) for l in open(tp)] if tp.exists() else []
    done = {(x["task_id"], x["lens"]) for x in T}
    schema = EP._STRUCTURED_SCHEMA_BLOCK.format(max_ops=MAX_OPS)
    jobs = []
    for tid, rs in sorted(R.items()):
        ok = [_ok(r) for r in rs]
        for lens, want in (("success", any(ok)), ("failure", not all(ok))):
            if want and (tid, lens) not in done:
                jobs.append((tid, lens))

    def _crit(job):
        tid, lens = job
        rs = R[tid]; t = tasks[tid]
        summ = "\n\n".join(f"### Rollout {int(r['_rollout_idx'])} - status={'success' if _ok(r) else 'failure'}, "
                           f"accuracy={1.0 if _ok(r) else 0.0}\n{S[(tid, int(r['_rollout_idx']))]['summary']}" for r in rs)
        tmpl = EP.SUCCESS_ANALYST_STRUCTURED_PROMPT if lens == "success" else EP.FAILURE_ANALYST_STRUCTURED_PROMPT
        kw = dict(schema=schema, question=rs[0].get("prompt") or t.question_text, summaries=summ, groundtruth=str(rs[0].get("ground_truth")))
        if lens == "success":
            kw["max_ops"] = MAX_OPS
        try:
            img = [I.part(Image.open(t.images[0]).convert("RGB"), 1_000_000)]
        except Exception:  # noqa: BLE001
            img = []
        raw = call(I.msgs(img, tmpl.format(**kw), sys=SYS))
        ents = SR._parse_structured_response(raw)[:MAX_OPS]
        return {"task_id": tid, "lens": lens, "n_success": sum(_ok(r) for r in rs), "entries": ents,
                "tips": [x for x in (SR._render_structured_to_text(e) for e in ents) if x], "raw": raw[:4000]}
    with ThreadPoolExecutor(workers) as ex, open(tp, "a") as fh:
        for x in ex.map(_crit, jobs):
            T.append(x); fh.write(json.dumps(x, ensure_ascii=False) + "\n"); fh.flush()
    T = [x for x in T if x["task_id"] in R]
    tips = [tip for x in sorted(T, key=lambda x: (x["task_id"], x["lens"])) for tip in x["tips"]]
    print(f"[skill] {family}: stage 2 done ({len(T)} analyst calls, {len(tips)} tips; failure-lens {sum(len(x['tips']) for x in T if x['lens'] == 'failure')})", flush=True)
    if not tips:
        print(f"[skill] {family}: no tips were produced -> no skill", flush=True)
        return None
    # ---- stage 3: fuse tips into the family skill, batch by batch
    slug = fam.replace("_", "-")
    skill = f"---\nname: {slug}\ndescription: Advisory notes for the '{fam.replace('_', ' ')}' visual task family.\nversion: 1.0.0\n---\n"
    (out / "fuse_steps").mkdir(exist_ok=True)
    n_invalid = 0
    for bi in range(0, len(tips), TIP_BATCH):
        batch = tips[bi:bi + TIP_BATCH]
        resp = _clean(call(I.msgs([], SP.FUSE_TIPS_INTO_SKILL_PROMPT.format(base_skill=skill, tips="\n".join(f"- {x}" for x in batch), tip_count=len(batch),
                                                                           description_rule=SP._DESCRIPTION_RULE_EXPANDABLE), sys=SYS)))
        if _valid(resp):
            skill = resp.strip() + "\n"
        else:
            n_invalid += 1
            print(f"[skill] {family}: fuse batch {bi // TIP_BATCH} returned an invalid document -> kept the previous skill", flush=True)
        (out / "fuse_steps" / f"{bi // TIP_BATCH:02d}.md").write_text(skill)
    (out / "skill_pre_refine.md").write_text(skill)
    if not _valid(skill):
        print(f"[skill] {family}: every fuse batch was invalid -> no skill", flush=True)
        return None
    # ---- stage 4: one refine pass
    body = skill.split("---", 2)[-1]
    resp = _clean(call(I.msgs([], SP.SKILL_REFINE_PROMPT.format(skill_content=skill, word_count=len(body.split()), description_rule=SP._DESCRIPTION_RULE_LOCKED), sys=SYS)))
    if _valid(resp):
        skill = resp.strip() + "\n"
    else:
        n_invalid += 1
        print(f"[skill] {family}: refine returned an invalid document -> kept the fused skill", flush=True)
    (out / "skill.md").write_text(skill)
    d = {k: counter.get(k, 0) - c0.get(k, 0) for k in ("n", "in_tokens", "out_tokens", "think_tokens")}
    meta = {"family": family, "recipe": "the agent harness stage-1 (novalidate): rollout summaries -> success/failure analysts -> fuse -> refine",
            "split_read": "train only", "executor": executor, "writer": writer, "train_dirs": used, "n_train_tasks": len(tasks), "n_tasks_with_rollouts": len(R),
            "n_rollouts": n_roll, "train_pass1": round(p1, 4), "n_tips": len(tips), "n_invalid_documents": n_invalid, "skill_words": len(skill.split()),
            "skill_sha256": hashlib.sha256(skill.encode()).hexdigest()[:16], "limit_tasks": limit_tasks or None,
            "cost": {**d, "usd": est_usd(writer, d["in_tokens"], d["out_tokens"], d["think_tokens"])[0], "seconds": round(time.time() - t0)},
            "built": datetime.datetime.now().isoformat(timespec="seconds"), "validated_on_val": False, "test_measured": None}
    (out / "skill_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[skill] {family}: skill.md written ({meta['skill_words']} words); {meta['cost']}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", nargs="+", required=True, help="source:family (the source is required: math / color / ocr exist in two sources)")
    ap.add_argument("--store", default=str(STORE), help="skill store; one dir per family: <store>/<source>__<family>/")
    ap.add_argument("--train-dirs", nargs="*", default=None, help="executor train rollout dirs (default: every finished output/verify/baseline_train*)")
    ap.add_argument("--executor", default=EXECUTOR, help="only rollouts of this model are read")
    ap.add_argument("--writer", default=WRITER)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-tasks", type=int, default=0, help="smoke test only: use the first N train tasks")
    ap.add_argument("--dry-run", action="store_true", help="count the rollouts that would be read; no API call, nothing written")
    a = ap.parse_args()
    bad = [f for f in a.family if ":" not in f]
    if bad:
        raise SystemExit(f"--family needs source:family, got {bad}")
    dirs = [Path(d) for d in a.train_dirs] if a.train_dirs else default_train_dirs()
    counter: dict = {}
    call = None
    rc = 0
    for family in a.family:
        out = skill_dir(family, a.store)
        if (out / "skill.md").exists() and not a.dry_run:
            print(f"[skill] {family}: {out / 'skill.md'} exists -> kept (the stored skill is the deployed one)", flush=True); continue
        if call is None and not a.dry_run:
            setup_env(a.writer)
            call = E.make_call(a.writer, {"temperature": 0.3, "top_p": 1.0, "max_tokens": 8192}, counter, retries=5)
        if build_skill(family, out, call, counter, dirs, a.executor, a.writer, a.workers, a.limit_tasks, a.dry_run) is None:
            rc = 3
    sys.exit(rc)


if __name__ == "__main__":
    main()
