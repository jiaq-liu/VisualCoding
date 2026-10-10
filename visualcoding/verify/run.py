"""CLI: run tasks x rollouts through the executor loop; write a harness-compatible output tree.

  <out>/<doc_id>/rollout_<r>/{traj.jsonl, metrics.json, messages.json, result.json,
                             injected_system_prompt.txt, original_image.jpg, tool_image_N.jpg}
  <out>/results.jsonl        one line per rollout (agent-harness fields + extras)
  <out>/metrics_at_k.json    unbiased pass@k per family and overall
  <out>/run_config.json      arguments and the loop configuration
  <out>/usage.jsonl          executor token usage of every invocation (appended)

Examples (from the package root visualcoding/, with the repository .venv):
  python -m verify.run --input ../data/tirbench/maze_test.json --model gemini-3.1-flash-lite \\
      --judge-model gemini-3.1-pro-preview --judge llm --rollouts 1 --workers 8 --output-dir output/verify/demo_maze
  python -m verify.run --input '../data/tirbench/*_train.json' --rollouts 4 --output-dir output/verify/train_tirbench
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import sys
import threading
import time
from math import comb
from pathlib import Path

from ._paths import IMAGE_ROOT
from .data import load_tasks
from .env import setup_env
from .judge import judge
from .loop import Episode, LoopConfig

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="record files of the frozen split (json lists) or globs")
    ap.add_argument("--image-root", default=str(IMAGE_ROOT))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--judge-model", default=None, help="LLM judge model (default: same as --model)")
    ap.add_argument("--judge", choices=["llm", "exact", "auto"], default="llm",
                    help="llm = the agent harness parity; exact = deterministic only; auto = exact where possible else llm")
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-total-tokens", type=int, default=32768)
    ap.add_argument("--max-images", type=int, default=100)
    ap.add_argument("--sandbox-timeout", type=float, default=60.0)
    ap.add_argument("--system-prompt-extra", default=None, help="file whose text is appended to the system prompt (skill)")
    ap.add_argument("--no-tools", action="store_true", help="ablation: no code_interpreter (tools=None); the model answers in text")
    ap.add_argument("--system-prompt-file", default=None, help="ablation: file whose text REPLACES the base system prompt "
                                                                "(--system-prompt-extra / --inject-dir texts are still appended)")
    ap.add_argument("--prelude-file", default=None, help="ablation: Python source run once per kernel start")
    ap.add_argument("--prelude-dir", default=None, help="ablation: per-task <task_id>.py appended to the kernel prelude")
    ap.add_argument("--inject-dir", default=None,
                    help="ablation: directory of per-task <task_id>.txt files appended to the system prompt of that task")
    ap.add_argument("--families", default=None, help="comma list of families to keep")
    ap.add_argument("--task-ids", default=None, help="comma list of doc_ids to keep")
    ap.add_argument("--limit", type=int, default=None, help="keep the first N tasks (after --shuffle)")
    ap.add_argument("--shuffle", type=int, default=None, help="shuffle tasks with this seed before --limit")
    ap.add_argument("--no-resume", action="store_true", help="re-run rollouts that already have metrics.json")
    ap.add_argument("--sandbox-root", default=None, help="kernel work dirs (default: <out>/_sandbox)")
    return ap


def expand_inputs(patterns) -> list[str]:
    files: list[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        files.extend(hits)
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        sys.exit(f"input files not found: {missing}")
    return files


def collect_results(out: Path) -> list[dict]:
    rows = []
    for p in sorted(out.glob("*/rollout_*/result.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    return rows


def write_metrics(out: Path, rows: list[dict], n_rollouts: int) -> dict:
    per: dict[str, dict[int, float]] = {}
    fam: dict[str, str] = {}
    extras: dict[str, list] = {"termination": [], "committed_by": []}
    for r in rows:
        per.setdefault(r["question_id"], {})[int(r["_rollout_idx"])] = float(r["accuracy_score"])
        fam[r["question_id"]] = r.get("family", "unknown")
        extras["termination"].append(r.get("termination"))
        extras["committed_by"].append(r.get("committed_by"))
    ks = [k for k in (1, 2, 4, 8) if k <= n_rollouts] or [1]
    by_family: dict[str, dict] = {}
    for q, m in per.items():
        n = len(m)
        c = int(sum(m.values()))
        d = by_family.setdefault(fam[q], {"n_tasks": 0, "n_rollouts": 0, **{f"pass@{k}": 0.0 for k in ks}})
        d["n_tasks"] += 1
        d["n_rollouts"] += n
        for k in ks:
            if k <= n:
                d[f"pass@{k}"] += pass_at_k(n, c, k)
    overall = {"n_tasks": 0, "n_rollouts": 0, **{f"pass@{k}": 0.0 for k in ks}}
    for f, d in by_family.items():
        for k in ks:
            overall[f"pass@{k}"] += d[f"pass@{k}"]
            d[f"pass@{k}"] = round(d[f"pass@{k}"] / d["n_tasks"], 4) if d["n_tasks"] else None
        overall["n_tasks"] += d["n_tasks"]
        overall["n_rollouts"] += d["n_rollouts"]
    for k in ks:
        overall[f"pass@{k}"] = round(overall[f"pass@{k}"] / overall["n_tasks"], 4) if overall["n_tasks"] else None
    from collections import Counter
    metrics = {"overall": overall, "by_family": by_family,
               "termination": dict(Counter(extras["termination"])), "committed_by": dict(Counter(extras["committed_by"])),
               "incomplete_tasks": [q for q, m in per.items() if len(m) < n_rollouts]}
    (out / "metrics_at_k.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return metrics


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    setup_env(args.model, args.judge_model)

    files = expand_inputs(args.input)
    tasks = load_tasks(files, args.image_root)
    if args.families:
        keep = set(args.families.split(","))
        tasks = [t for t in tasks if t.family in keep]
    if args.task_ids:
        keep = set(args.task_ids.split(","))
        tasks = [t for t in tasks if t.task_id in keep]
    if args.shuffle is not None:
        random.Random(args.shuffle).shuffle(tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        sys.exit("no tasks selected")

    extra = Path(args.system_prompt_extra).read_text() if args.system_prompt_extra else None
    cfg = LoopConfig(model=args.model, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                     max_turns=args.max_turns, max_images=args.max_images,
                     max_total_tokens=args.max_total_tokens, sandbox_timeout=args.sandbox_timeout,
                     system_prompt_extra=extra, no_tools=args.no_tools,
                     extra_prelude=(Path(args.prelude_file).read_text() if args.prelude_file else ""))
    if args.system_prompt_file:
        cfg.system_prompt_override = Path(args.system_prompt_file).read_text()
    sandbox_root = Path(args.sandbox_root) if args.sandbox_root else out / "_sandbox"
    (out / "run_config.json").write_text(json.dumps({
        "argv": sys.argv[1:], "inputs": files, "n_tasks": len(tasks), "rollouts": args.rollouts,
        "judge": args.judge, "judge_model": args.judge_model or args.model, "config": cfg.__dict__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2, default=str))

    jobs = []
    for t in tasks:
        for r in range(args.rollouts):
            d = out / t.task_id / f"rollout_{r}"
            if not args.no_resume and (d / "metrics.json").exists() and (d / "result.json").exists():
                continue
            jobs.append((t, r))
    log(f"{len(tasks)} tasks x {args.rollouts} rollouts; {len(jobs)} to run ("
        f"model={args.model}, judge={args.judge}/{args.judge_model or args.model}, workers={args.workers})")

    judge_fn = (lambda task, texts, final: judge(task, texts, final, mode=args.judge))

    def task_cfg(t, r):
        """Per-task copy of cfg (--inject-dir / --prelude-dir append per-task text to the prompt / kernel prelude)."""
        import dataclasses
        c = dataclasses.replace(cfg)
        if args.inject_dir:                                   # per-task reference text (ablation)
            f = Path(args.inject_dir) / f"{t.task_id}.txt"
            if f.is_file():
                c.system_prompt_extra = ((c.system_prompt_extra or "").rstrip() + "\n\n" + f.read_text().strip()).strip()
        if args.prelude_dir:                                  # per-task kernel prelude (ablation)
            f = Path(args.prelude_dir) / f"{t.task_id}.py"
            if f.is_file():
                c.extra_prelude = (c.extra_prelude.rstrip("\n") + "\n" + f.read_text()).strip("\n") + "\n"
        return c

    def run_one(t, r):
        d = out / t.task_id / f"rollout_{r}"
        d.mkdir(parents=True, exist_ok=True)
        ep = Episode(t, task_cfg(t, r), d, r, sandbox_root, judge_fn=judge_fn, log=log)
        try:
            res = ep.run()
            log(f"  done {t.task_id} r{r}: acc={res['accuracy_score']:.0f} turns={res['n_turns']} "
                f"term={res['termination']} by={res['committed_by']} ans={str(res['final_answer'])[:30]!r} "
                f"gt={t.answer[:20]!r} ({res['elapsed_s']}s)")
            return res
        except Exception as e:  # noqa: BLE001
            import traceback
            (d / "error.txt").write_text(traceback.format_exc())
            log(f"  FAILED {t.task_id} r{r}: {type(e).__name__}: {e}")
            return None

    t0 = time.time()
    n_done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, t, r) for t, r in jobs]
        for fut in cf.as_completed(futs):
            fut.result()
            n_done += 1
            if n_done % 10 == 0 or n_done == len(futs):
                rows = collect_results(out)
                m = write_metrics(out, rows, args.rollouts)
                log(f"progress {n_done}/{len(futs)} ({(time.time() - t0) / 60:.1f} min); overall {m['overall']}")
    rows = collect_results(out)
    m = write_metrics(out, rows, args.rollouts)
    try:                                           # executor token usage of THIS invocation (judge calls not included); appended, never overwritten
        from .model import USAGE
        with open(out / "usage.jsonl", "a") as fh:
            fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_rollouts_run": len(jobs), "wall_s": round(time.time() - t0), "usage": USAGE}) + "\n")
    except Exception as e:  # noqa: BLE001
        log(f"usage accounting failed: {e}")
    log(f"finished: {json.dumps(m['overall'])}; termination={m['termination']}; committed_by={m['committed_by']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
