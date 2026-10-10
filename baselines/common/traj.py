"""Trajectory rendering for the reflection / analysis stages of the baselines.

WHY THIS EXISTS: `result.json:conversation_history` and `trajectory_text` render every
assistant tool call as the literal string "[Function Call]" — the CODE IS NOT IN THEM.
Feeding those to a trajectory-analysing baseline (XSkill's critique, Trace2Skill's error
analysis, SkillOpt's reflection) silently starves it: the model sees an agent that
"called a tool" and never what it computed.  The code, tool output, produced images and
the loop events live in the per-rollout `traj.jsonl`, which is what this module reads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MAX_RESULT_CHARS = 2000
MAX_CODE_CHARS = 4000


@dataclass
class Turn:
    idx: int
    text: str = ""
    code: str | None = None
    tool_result: str | None = None
    tool_images: list = field(default_factory=list)
    error: bool = False
    injected: str | None = None


@dataclass
class Trajectory:
    task_id: str
    question: str
    ground_truth: str
    family: str
    rollout: int
    turns: list[Turn]
    final_answer: str | None
    accuracy: float | None
    termination: str | None
    committed_by: str | None
    judge_analysis: str | None = None
    dir: Path | None = None

    @property
    def success(self) -> bool:
        return bool(self.accuracy and self.accuracy >= 1.0)

    def render(self, max_result_chars: int = MAX_RESULT_CHARS, max_code_chars: int = MAX_CODE_CHARS,
               with_header: bool = True) -> str:
        """Readable transcript with the code and the tool output kept."""
        out = []
        if with_header:
            out += [f"## Task {self.task_id} (family {self.family}, rollout {self.rollout})",
                    f"Question: {self.question.strip()}", ""]
        for t in self.turns:
            if t.injected:
                out.append(f"[Turn {t.idx}] (injected/{t.injected}) {_clip(t.text, 600)}")
                continue
            if t.text.strip():
                out.append(f"[Turn {t.idx}] assistant: {t.text.strip()}")
            if t.code is not None:
                out.append(f"[Turn {t.idx}] code_interpreter:\n```python\n{_clip(t.code, max_code_chars)}\n```")
                res = _clip(t.tool_result or "", max_result_chars)
                imgs = f"  (+{len(t.tool_images)} image(s) shown to the model)" if t.tool_images else ""
                out.append(f"-> {res}{imgs}")
        out.append("")
        out.append(f"Final answer: {self.final_answer!r}   ground truth: {self.ground_truth!r}   "
                   f"correct: {'yes' if self.success else 'no'}   termination: {self.termination}")
        if self.judge_analysis:
            out.append(f"Judge: {_clip(self.judge_analysis, 600)}")
        return "\n".join(out)


def _clip(s: str, n: int) -> str:
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= n else s[: n - 20] + f"\n... [{len(s) - n + 20} chars cut]"


def load(rollout_dir) -> Trajectory:
    """Parse one rollout directory (traj.jsonl + result.json)."""
    d = Path(rollout_dir)
    events = [json.loads(l) for l in (d / "traj.jsonl").read_text().splitlines() if l.strip()]
    head = events[0] if events and "doc_id" in events[0] else {}
    turns: dict[int, Turn] = {}
    final = {}
    for e in events[1:]:
        i = int(e.get("turn_idx", 0))
        t = turns.setdefault(i, Turn(idx=i))
        if "final_answer" in e:
            final = e
            continue
        if "injected" in e:
            t.injected = e.get("injected")
            t.text = e.get("text", "")
            continue
        if "text_output" in e:
            t.text = (t.text + "\n" + (e.get("text_output") or "")).strip()
        if "tool_call" in e:
            call = e.get("tool_call") or {}
            if isinstance(call, str):
                try:
                    call = json.loads(call)
                except Exception:
                    call = {"code": call}
            t.code = call.get("code", "")
            t.tool_result = e.get("tool_result") or ""
            t.tool_images = e.get("tool_images") or []
            t.error = bool((e.get("meta") or {}).get("error"))
    res = {}
    p = d / "result.json"
    if p.is_file():
        try:
            res = json.loads(p.read_text())
        except Exception:
            res = {}
    return Trajectory(
        task_id=head.get("doc_id") or res.get("question_id", d.parent.name),
        question=head.get("initial_prompt") or res.get("prompt", ""),
        ground_truth=head.get("ground_truth") or res.get("ground_truth", ""),
        family=head.get("family") or res.get("family", ""),
        rollout=int(head.get("rollout", res.get("_rollout_idx", 0))),
        turns=[turns[k] for k in sorted(turns)],
        final_answer=final.get("final_answer", res.get("final_answer")),
        accuracy=float(final.get("accuracy_score", res.get("accuracy_score", 0)) or 0),
        termination=final.get("termination", res.get("termination")),
        committed_by=final.get("committed_by", res.get("committed_by")),
        judge_analysis=res.get("trajectory_analysis"),
        dir=d,
    )


def load_run(out_dir, task_ids=None) -> list[Trajectory]:
    """Every rollout of an executor output tree (optionally restricted to some tasks)."""
    out_dir = Path(out_dir)
    keep = set(task_ids) if task_ids else None
    trajs = []
    for p in sorted(out_dir.glob("*/rollout_*/traj.jsonl")):
        if keep and p.parent.parent.name not in keep:
            continue
        try:
            trajs.append(load(p.parent))
        except Exception:
            continue
    return trajs


def group_by_task(trajs: list[Trajectory]) -> dict[str, list[Trajectory]]:
    out: dict[str, list[Trajectory]] = {}
    for t in trajs:
        out.setdefault(t.task_id, []).append(t)
    for v in out.values():
        v.sort(key=lambda t: t.rollout)
    return out


def split_success(trajs: list[Trajectory]) -> tuple[list[Trajectory], list[Trajectory]]:
    """(successes, failures) — the two pools every one of these baselines learns from."""
    return [t for t in trajs if t.success], [t for t in trajs if not t.success]
