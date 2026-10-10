"""Run instruments on a split: one reading per task into <out>/results.jsonl (resumable), scored by exact normalised match.

    python -m verify.instrument_eval run --split test --instruments <key> --out output/verify/<run>/test

The pipeline (scripts/pipeline/report.py, check.py, route_tools.py) combines these readings with the executor's stored rollouts
into the hard-switch policy - in range the program's answer is final, out of range the executor's own rollout answers - with
paired task bootstraps; the helpers it uses (split_tasks, load_eyeball, _boot_ci, _mean, _majority, make_call) live here.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from ._paths import enable_vendor, split_files
from .data import Task, load_tasks
from .env import setup_env
from . import instruments as I

enable_vendor()
from engine.api_caller import call_vision_api          # noqa: E402

_LETTER_RE = re.compile(r"^\s*[\(\[]?([A-J])[\)\].:\s]")

def split_tasks(split: str, families: set[str]) -> list[Task]:
    files = []
    for fam in sorted(families):
        files += split_files(fam, split)
    return load_tasks(sorted(set(files)))


def load_image(task: Task) -> Image.Image:
    return Image.open(task.images[0]).convert("RGB")   # no EXIF transpose: parity with the baseline loop


def make_call(model: str, sampling: dict, counter: dict, retries: int = 3):
    def call(messages, **over):
        sp = dict(sampling)
        sp.update(over)
        last = None
        for _ in range(retries):
            resp = call_vision_api(model, messages, sp, tools=None)
            counter["n"] = counter.get("n", 0) + 1
            if resp is None or (isinstance(resp, str) and resp.startswith("Error:")):
                last = str(resp)
                time.sleep(2)
                continue
            content = resp.get("content") or ""
            u = resp.get("usage") if isinstance(resp, dict) else None
            if isinstance(u, dict):             # cost accounting: cumulative tokens of this call site
                for k, kk in (("input_tokens", "in_tokens"), ("output_tokens", "out_tokens"), ("thinking_tokens", "think_tokens")):
                    counter[kk] = counter.get(kk, 0) + int(u.get(k) or 0)
            if content.strip():
                return content
            last = "empty"
            time.sleep(2)
        counter["err"] = last
        return ""
    return call


# ----------------------------------------------------------------------------- run
def cmd_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_env(args.model, args.model)
    keys = list(I.INSTRUMENTS) if args.instruments == "all" else args.instruments.split(",")
    for k in keys:
        if k not in I.INSTRUMENTS:
            raise SystemExit(f"unknown instrument {k}; known: {sorted(I.INSTRUMENTS)}")
    fams = {f for k in keys for f in I.INSTRUMENTS[k].families}
    tasks = split_tasks(args.split, fams)
    if args.task_ids:
        keep = set(args.task_ids.split(","))
        tasks = [t for t in tasks if t.task_id in keep]
    if args.limit:
        by = collections.defaultdict(list)
        for t in tasks:
            by[(t.source, t.family)].append(t)
        tasks = [t for f in by for t in by[f][: args.limit]]
    res_path = out / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in open(res_path):
            r = json.loads(line)
            done.add((r["task_id"], r["instrument"], r["sample"]))
    jobs = []
    for t in tasks:
        for inst in I.for_task(t, keys):
            ns = 1 if inst.deterministic else args.samples
            for s in range(ns):
                if (t.task_id, inst.key, s) not in done:
                    jobs.append((t, inst, s))
    print(f"[instrument_eval] split={args.split} {len(tasks)} tasks, {len(jobs)} readings to do ({len(done)} done); "
          f"instruments={keys} model={args.model}", flush=True)
    sampling = {"temperature": args.temperature, "top_p": 1.0, "max_tokens": args.max_tokens}
    lock = threading.Lock()
    images: dict[str, Image.Image] = {}

    def get_image(t: Task):
        with lock:
            if t.task_id not in images:
                images[t.task_id] = load_image(t)
            return images[t.task_id]

    def work(job):
        t, inst, s = job
        counter: dict = {}
        call = make_call(args.model, sampling, counter)
        t0 = time.time()
        err = None
        try:
            rd = inst.measure(t, get_image(t), call)
            rd = inst.finalize(rd, t)
        except Exception as e:  # noqa: BLE001
            import traceback
            rd = I.Reading(diag={"exception": traceback.format_exc()[-2000:]}, in_range=False)
            err = f"{type(e).__name__}: {e}"
        correct = inst.is_correct(rd.answer, t)
        return {"task_id": t.task_id, "source": t.source, "family": t.family, "instrument": inst.key, "split": args.split,
                "sample": s, "value": rd.value, "answer": rd.answer, "correct": correct, "in_range": bool(rd.in_range),
                "diag": rd.diag, "n_calls": rd.n_calls, "api_calls": counter.get("n", 0), "gt_key": inst.gt_key(t),
                "gt_value": I.gt_value(t), "error": err or counter.get("err"), "elapsed": round(time.time() - t0, 1),
                "contents": [c[:4000] for c in rd.contents]}

    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(res_path, "a") as fh:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            n_ok += int(r["correct"] == 1)
            if i % 20 == 0 or i == len(jobs):
                print(f"[instrument_eval] {i}/{len(jobs)} done, correct so far {n_ok}", flush=True)
    print("[instrument_eval] run complete", flush=True)


# ----------------------------------------------------------------------------- report helpers
def _majority(vals):
    c = collections.Counter(v for v in vals if v is not None)
    if not c:
        return None
    best = max(c.values())
    for v in vals:
        if v is not None and c[v] == best:
            return v


def _boot_ci(pairs, n=4000, seed=0):
    rng = random.Random(seed)
    if not pairs:
        return (float("nan"), float("nan"))
    ds = []
    for _ in range(n):
        smp = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        ds.append(sum(b - a for a, b in smp) / len(smp))
    ds.sort()
    return ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def eyeball_key(final: str, task: Task) -> str:
    fin = (final or "").strip()
    if task.is_mc:
        m = _LETTER_RE.match(fin)
        return m.group(1) if m else I.norm_text(fin)
    return I.norm_text(fin)


def load_eyeball(dirs: list[str], tasks: dict[str, Task]) -> dict[str, list[dict]]:
    """task_id -> [{'correct','key','final'}] from baseline results.jsonl files (first dir that has the task wins)."""
    out: dict[str, list[dict]] = {}
    for d in dirs:
        p = Path(d) / "results.jsonl"
        if not p.exists():
            continue
        got = collections.defaultdict(list)
        for line in open(p):
            r = json.loads(line)
            tid = r["question_id"]
            if tid in tasks and tid not in out:
                got[tid].append({"correct": int(float(r.get("accuracy_score") or 0) >= 0.5),
                                 "key": eyeball_key(r.get("final_answer") or "", tasks[tid]),
                                 "final": (r.get("final_answer") or "")[:20],
                                 "turns": r.get("n_turns")})
        out.update(got)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--split", default="test", choices=["train", "test"])
    r.add_argument("--instruments", default="all")
    r.add_argument("--samples", type=int, default=4)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--task-ids", default=None)
    r.add_argument("--model", default="gemini-3.1-pro-preview")
    r.add_argument("--temperature", type=float, default=0.6)
    r.add_argument("--max-tokens", type=int, default=8192)
    r.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
