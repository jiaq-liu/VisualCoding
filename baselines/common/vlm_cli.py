#!/usr/bin/env python
"""The tool-free arm (GEPA / TextGrad), executed with the main environment's interpreter (VC_PY).

One model call per rollout: system = the candidate prompt, user = image parts + question,
no tools, answer taken from the last <answer> tag, scored by the same LLM judge as the tool
arm.  Image preprocessing, the API client and the judge are verify's, so the only difference
from the tool arm is the absence of the code_interpreter loop.

The output tree has the same shape as `verify.run`'s (<task>/rollout_<i>/result.json), which
makes it readable by common.executor.results()/scores() and content-addressable: the output
dir is keyed by the prompt hash, so re-evaluating a candidate costs nothing.

  python vlm_cli.py --input recs.json --system-prompt-file p.txt --out-dir DIR [--rollouts 1]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VC_ROOT = Path(os.environ.get("VC_ROOT", HERE.parent.parent / "visualcoding"))
sys.path.insert(0, str(VC_ROOT))

from verify.data import load_tasks                       # noqa: E402
from verify.env import setup_env                         # noqa: E402
from verify.judge import judge                           # noqa: E402
from verify.model import call_model, last_answer, load_task_images   # noqa: E402
from verify.run import collect_results, write_metrics    # noqa: E402

# Same answer contract as the tool arm, without the tool sections.
BASE_SYSTEM_PROMPT = """You are a visual reasoning agent. Your goal is to answer questions about images.

# INSTRUCTIONS:

1. Analyze: Carefully observe the image and the user's question.
2. Think: Explain your step-by-step reasoning process.
3. Answer: Once confident, provide the final answer inside <answer>...</answer> tags!

# IMPORTANT:

- Always explain your detailed reasoning process before answering.
- You MUST provide your final answer using complete <answer>...</answer> tags!
"""


def system_prompt(extra: str | None = None) -> str:
    s = BASE_SYSTEM_PROMPT
    if extra:
        s += "\n" + extra.strip() + "\n"
    return s


def run_one(task, sys_prompt, d: Path, rollout_idx: int, args) -> dict:
    t0 = time.time()
    d.mkdir(parents=True, exist_ok=True)
    _, image_list, parts = load_task_images(task.images, args.max_pixels, args.min_pixels)
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": parts + [{"type": "text", "text": task.question_text}]}]
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
    text, err = "", None
    for attempt in range(args.retries + 1):
        resp = call_model(messages, args.model, sampling, tools=None)
        if isinstance(resp, dict):
            text = (resp.get("content") or "").strip()
            if text:
                break
            err = "empty content"
        else:
            err = str(resp)
        sampling = dict(sampling, temperature=min(1.0, sampling["temperature"] + 0.3))
    final = last_answer(text) or (text.strip().splitlines()[-1].strip() if text.strip() else "")
    score, analysis, judge_used = judge(task, [text], final, mode=args.judge)
    res = {"question_id": task.task_id, "_rollout_idx": rollout_idx, "prompt": task.question_text,
           "final_answer": final, "ground_truth": task.answer, "accuracy_score": float(score),
           "trajectory_score": float(score), "trajectory_analysis": analysis,
           "conversation_history": [{"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": task.question_text},
                                    {"role": "assistant", "content": text}],
           "trajectory_text": text, "family": task.family, "source": task.source,
           "n_turns": 1, "n_tool_calls": 0, "n_images": len(image_list),
           "termination": "answered" if text else "empty", "committed_by": "model",
           "judge": judge_used, "elapsed_s": round(time.time() - t0, 1), "model": args.model,
           "arm": "no_tool", "error": err if not text else None}
    (d / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (d / "metrics.json").write_text(json.dumps(
        {k: res[k] for k in ("accuracy_score", "trajectory_score", "trajectory_analysis", "family",
                             "source", "n_turns", "termination", "judge", "elapsed_s", "model")}, indent=1))
    (d / "response.txt").write_text(text)
    (d / "injected_system_prompt.txt").write_text(sys_prompt)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="data_split records json")
    ap.add_argument("--system-prompt-file", default=None, help="candidate prompt appended to the base")
    ap.add_argument("--system-prompt-override", default=None, help="use this file verbatim as the system prompt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-root", default=str(Path(os.environ.get(
        "IMAGE_ROOT", HERE.parent.parent / "data_split" / "image_root"))))
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--judge-model", default=os.environ.get("BASELINE_JUDGE_MODEL", "gemini-3.1-flash-lite"))
    ap.add_argument("--judge", default="llm", choices=["llm", "exact", "auto"])
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-pixels", type=int, default=2_000_000)
    ap.add_argument("--min-pixels", type=int, default=40_000)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    setup_env(args.model, args.judge_model)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks([args.input], args.image_root)
    extra = Path(args.system_prompt_file).read_text() if args.system_prompt_file else None
    sys_prompt = (Path(args.system_prompt_override).read_text() if args.system_prompt_override
                  else system_prompt(extra))
    (out / "run_config.json").write_text(json.dumps({
        "argv": sys.argv[1:], "arm": "no_tool", "n_tasks": len(tasks), "rollouts": args.rollouts,
        "judge": args.judge, "judge_model": args.judge_model,
        "config": {"model": args.model, "temperature": args.temperature, "top_p": args.top_p,
                   "max_tokens": args.max_tokens, "max_pixels": args.max_pixels,
                   "min_pixels": args.min_pixels, "system_prompt_sha": None},
        "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    (out / "system_prompt.txt").write_text(sys_prompt)

    jobs = [(t, r) for t in tasks for r in range(args.rollouts)
            if args.no_resume or not (out / t.task_id / f"rollout_{r}" / "result.json").is_file()]
    print(f"{len(tasks)} tasks x {args.rollouts} rollouts; {len(jobs)} to run (model={args.model}, "
          f"workers={args.workers})", flush=True)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, t, sys_prompt, out / t.task_id / f"rollout_{r}", r, args): (t, r)
                for t, r in jobs}
        for fut in cf.as_completed(futs):
            t, r = futs[fut]
            done += 1
            try:
                res = fut.result()
                if done % 25 == 0 or done == len(futs):
                    print(f"  {done}/{len(futs)} last={t.task_id} acc={res['accuracy_score']:.0f}", flush=True)
            except Exception as e:                      # noqa: BLE001
                d = out / t.task_id / f"rollout_{r}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "error.txt").write_text(f"{type(e).__name__}: {e}")
                print(f"  FAILED {t.task_id} r{r}: {type(e).__name__}: {e}", flush=True)
    rows = collect_results(out)
    m = write_metrics(out, rows, args.rollouts)
    print("finished:", json.dumps(m["overall"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
