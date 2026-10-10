"""Client for the tool-free arm — importable from any venv (stdlib only).

Results are content-addressed by the effective system prompt:

    <BASELINE_OUT>/_vlm/<model>/<prompt_sha>/<task_id>/rollout_<i>/result.json

so a candidate prompt that a prompt optimizer evaluates twice (a repeated minibatch, a
re-evaluated parent, the seed of a second method) is paid for once.  Read the scores back
with common.executor.scores(dir) — the tree shape is the same as the tool arm's.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import time
from pathlib import Path

from . import data
from .paths import VC_PY, IMAGE_ROOT, JUDGE_MODEL, MODEL, OUT_ROOT, subprocess_env

CLI = Path(__file__).resolve().parent / "vlm_cli.py"
CACHE = Path(OUT_ROOT) / "_vlm"


def base_system_prompt() -> str:
    """BASE_SYSTEM_PROMPT out of vlm_cli.py, without importing it (it needs the verify stack)."""
    for node in ast.parse(CLI.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "BASE_SYSTEM_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError("BASE_SYSTEM_PROMPT not found in vlm_cli.py")


def system_prompt(extra: str | None = None) -> str:
    s = base_system_prompt()
    if extra:
        s += "\n" + extra.strip() + "\n"
    return s


def prompt_sha(extra: str | None = None) -> str:
    return hashlib.sha256(system_prompt(extra).encode()).hexdigest()[:16]


def out_dir(extra: str | None = None, model: str = MODEL) -> Path:
    return CACHE / model.replace("/", "_") / prompt_sha(extra)


def run(recs: list[dict], prompt: str | None = None, *, rollouts: int = 1, model: str = MODEL,
        judge_model: str = JUDGE_MODEL, workers: int = 8, temperature: float = 0.6,
        judge: str = "llm", dry_run: bool = False, log=print) -> dict:
    """Score `recs` (data_split records) under a candidate prompt; returns a report dict.

    `prompt` is appended to the tool-free base prompt, exactly like the tool arm's skill text.
    """
    d = out_dir(prompt, model)
    d.mkdir(parents=True, exist_ok=True)
    input_file = data.write_input(recs, d / "_inputs" / f"{_sha_recs(recs)}.json")
    missing = [(r["doc_id"], i) for r in recs for i in range(rollouts)
               if not (d / r["doc_id"] / f"rollout_{i}" / "result.json").is_file()]
    report = {"out_dir": str(d), "n_tasks": len(recs), "rollouts": rollouts, "model": model,
              "prompt_sha": prompt_sha(prompt), "cached": len(recs) * rollouts - len(missing),
              "to_run": len(missing)}
    if dry_run or not missing:
        report["dry_run"] = dry_run
        return report
    if prompt:
        (d / "prompt.md").write_text(prompt)
    cmd = [str(VC_PY), "-u", str(CLI), "--input", str(input_file), "--out-dir", str(d),
           "--image-root", str(IMAGE_ROOT), "--model", model, "--judge-model", judge_model,
           "--judge", judge, "--rollouts", str(rollouts), "--workers", str(workers),
           "--temperature", str(temperature)]
    if prompt:
        cmd += ["--system-prompt-file", str(d / "prompt.md")]
    t0 = time.time()
    with open(str(d) + ".log", "a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {len(missing)} rollouts ===\n")
        fh.flush()
        rc = subprocess.run(cmd, env=subprocess_env(), stdout=fh, stderr=fh).returncode
    report.update(returncode=rc, wall_s=round(time.time() - t0), log=str(d) + ".log")
    return report


def _sha_recs(recs) -> str:
    return hashlib.sha256(",".join(r["doc_id"] for r in recs).encode()).hexdigest()[:12]


def read(out_dir, recs: list[dict], rollouts: int = 1) -> dict:
    """{doc_id: [result rows]} for just these records (the cache dir also holds other families)."""
    out_dir = Path(out_dir)
    got: dict[str, list] = {}
    for r in recs:
        tid = r["doc_id"]
        for i in range(rollouts):
            p = out_dir / tid / f"rollout_{i}" / "result.json"
            if p.is_file():
                try:
                    got.setdefault(tid, []).append(json.loads(p.read_text()))
                except Exception:
                    pass
    return got


def mean_scores(out_dir, recs: list[dict], rollouts: int = 1) -> dict:
    """{doc_id: mean accuracy} (missing rollouts are simply not counted)."""
    return {k: sum(float(r["accuracy_score"]) for r in v) / len(v)
            for k, v in read(out_dir, recs, rollouts).items() if v}
