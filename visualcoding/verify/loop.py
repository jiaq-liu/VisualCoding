"""The episode loop: one rollout of one task through the multi-turn code-interpreter agent.

One `Episode` = one rollout of one task. Turn = one model call (tool-call turns included), as in
the agent harness's `_run_greedy_loop`. Termination and environment rules (always on):
  * turn budget: the last `commit_reserve` turns carry a "give your final answer now" nudge;
  * loop detection: identical code twice / identical exception twice -> targeted hint; three times -> commit;
  * empty model replies are re-asked by the vendored api_caller (request-local nudge);
  * forced commit when the budget runs out: one temperature-0 answer-only call, else the last answer
    candidate / assistant text — never "Error: Reached max turns";
  * the final answer is the LAST <answer> tag (judge parity).
"""
from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .data import Task
from .model import (ANSWER_NUDGE, NO_TOOLS_ANSWER_NUDGE, approx_tokens, call_model, image_part, last_answer,
                    load_task_images, system_prompt, tool_feedback, tools_schema)
from .sandbox import Sandbox


@dataclass
class LoopConfig:
    model: str = "gemini-3.1-pro-preview"
    temperature: float = 0.6
    top_p: float = 1.0
    max_tokens: int = 8192
    max_turns: int = 20
    commit_reserve: int = 2          # last N turns: answer-now nudge
    max_images: int = 100
    max_total_tokens: int = 32768
    max_pixels: int = 2_000_000
    min_pixels: int = 40_000
    format_nudges: int = 2           # the agent harness parity: untagged text replies get nudged twice
    loop_hint_at: int = 2            # identical code/exception count that triggers a hint
    loop_commit_at: int = 3          # ... that forces a commit
    sandbox_timeout: float = 60.0
    system_prompt_extra: str | None = None   # e.g. a procedural skill text
    system_prompt_override: str | None = None  # parity: use this exact system prompt
    api_retries: int = 2             # consecutive failed model calls before giving up the episode
    no_tools: bool = False           # ablation: answer continuation without any tool (tools=None, text reply only)
    extra_prelude: str = ""          # ablation: Python source run once per kernel start

    def sampling(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens}


_LOOP_HINTS = [
    (r"not enough values to unpack|too many values to unpack", "In OpenCV 5, cv2.HoughLinesP returns shape (N, 4) and "
     "cv2.findContours returns (contours, hierarchy); use np.asarray(x).reshape(-1, 4) for lines."),
    (r"NameError", "A variable from an earlier turn is undefined (the kernel may have restarted). Recompute it in this cell."),
    (r"TimeoutError", "The cell is too slow: use a smaller region (box=...), fewer sweep values, or downscale before processing."),
    (r"MemoryError|Insufficient memory", "Reduce memory: process a crop instead of the full image, or downscale first."),
]


def _strip_images(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m2 = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            m2["content"] = [p if p.get("type") != "image_url" else {"type": "image_url", "image_url": {"url": "<base64 omitted>"}}
                             for p in c]
        m2.pop("gemini_parts", None)
        out.append(m2)
    return out


class Episode:
    def __init__(self, task: Task, cfg: LoopConfig, out_dir: Path, rollout_idx: int,
                 sandbox_root: Path, judge_fn=None, log=print):
        self.task, self.cfg, self.out_dir, self.r = task, cfg, Path(out_dir), rollout_idx
        self.sandbox_root = Path(sandbox_root)
        self.judge_fn = judge_fn
        self.log = log
        self.candidates: list[str] = []
        self.api: list[dict] = []
        self.text_hist: list[dict] = []
        self.image_map: dict[str, Image.Image] = {}
        self.turn = 0
        self.n_tool_calls = 0
        self.termination: str | None = None
        self.committed_by: str | None = None
        self.t0 = time.time()

    # ------------------------------------------------------------------ io helpers
    def _traj(self, **ev) -> None:
        with open(self.out_dir / "traj.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def _inject_user(self, text: str, images: list[tuple[str, Image.Image]] | None = None, tag: str = "inject") -> None:
        if images:
            content = []
            sent = {}
            for n, img in images:
                part, wh = image_part(img, self.cfg.max_pixels, self.cfg.min_pixels)
                sent[n] = wh
                content.append(part)
            content.append({"type": "text", "text": text})
            self.api.append({"role": "user", "content": content})
        else:
            self.api.append({"role": "user", "content": text})
        self.text_hist.append({"role": "user", "content": text})
        self._traj(turn_idx=self.turn, injected=tag, text=text, n_images=len(images or []))

    def _new_image_names(self, n: int) -> list[str]:
        base = len([k for k in self.image_map if k.startswith("tool_image_")])
        return [f"tool_image_{base + i + 1}" for i in range(n)]

    def _run_code(self, code: str, source: str):
        """Execute in the sandbox; register the images it produced; return (result, new_images)."""
        res = self.sandbox.run(code, self.image_map)
        names = self._new_image_names(len(res.images))
        new_images = []
        for n, b in zip(names, res.images):
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
            except Exception:
                continue
            self.image_map[n] = img
            new_images.append((n, img))
            img.save(self.out_dir / f"{n}.jpg", quality=90)
        self._traj(turn_idx=self.turn, tool_call={"name": "code_interpreter", "code": code, "source": source},
                   tool_result=res.text[:4000], tool_images=[n for n, _ in new_images], meta=res.meta)
        return res, new_images

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        cfg, task = self.cfg, self.task
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "traj.jsonl").write_text("")
        self._traj(doc_id=task.task_id, initial_prompt=task.question, ground_truth=task.answer, family=task.family,
                   rollout=self.r)
        self.sandbox = Sandbox(self.sandbox_root, output_timeout=cfg.sandbox_timeout, extra_prelude=cfg.extra_prelude)
        try:
            return self._run_inner()
        finally:
            try:
                self.sandbox.close()
            except Exception:
                pass

    def _run_inner(self) -> dict:
        cfg, task = self.cfg, self.task
        self.image_map, image_list, parts = load_task_images(task.images, cfg.max_pixels, cfg.min_pixels)
        self.original_size = image_list[0][1].size if image_list else None
        if image_list:
            image_list[0][1].save(self.out_dir / "original_image.jpg", quality=90)
        sys_prompt = ((cfg.system_prompt_override.rstrip() + ("\n\n" + cfg.system_prompt_extra.strip() + "\n" if cfg.system_prompt_extra else "\n"))
                      if cfg.system_prompt_override else system_prompt(extra=cfg.system_prompt_extra))
        (self.out_dir / "injected_system_prompt.txt").write_text(sys_prompt)
        self.api = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": parts + [{"type": "text", "text": task.question_text}]}]
        refs = "\n".join(f"[Image: {n}]" for n, _ in image_list)
        self.text_hist = [{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": f"{refs}\n{task.question_text}"}]
        tools = None if cfg.no_tools else tools_schema()
        sampling = cfg.sampling()

        final: str | None = None
        fmt_nudges = 0
        api_errors = 0
        commit_nudged = False
        last_code, same_code, last_err, same_err = None, 0, None, 0

        while final is None:
            if self.turn >= cfg.max_turns:
                self.termination = "max_turns"
                break
            if len(self.image_map) >= cfg.max_images:
                self.termination = "max_images"
                break
            used = approx_tokens(self.api, per_image=0)          # harness-style text-only accounting
            if used >= cfg.max_total_tokens - cfg.max_tokens:
                self.termination = "max_tokens"
                break
            if not commit_nudged and self.turn >= cfg.max_turns - cfg.commit_reserve:
                left = cfg.max_turns - self.turn
                self._inject_user(f"[budget] You have {left} turn(s) left. Give your final answer now in "
                                  f"<answer></answer> tags based on the evidence gathered so far.", tag="commit_nudge")
                commit_nudged = True

            resp = call_model(self.api, cfg.model, sampling, tools)
            self.turn += 1
            if resp is None or (isinstance(resp, str) and resp.startswith("Error:")):
                api_errors += 1
                self._traj(turn_idx=self.turn, error=str(resp))
                self.log(f"    [{task.task_id} r{self.r}] turn {self.turn}: model call failed ({str(resp)[:80]})")
                if api_errors >= cfg.api_retries:
                    self.termination = "api_error"
                    break
                continue
            api_errors = 0
            content = resp.get("content") or ""
            tool_calls = resp.get("tool_calls") or []
            assistant = {"role": "assistant", "content": content}
            if tool_calls:
                assistant["tool_calls"] = [tool_calls[0]]
            if resp.get("gemini_parts"):
                assistant["gemini_parts"] = resp["gemini_parts"]
            self.api.append(assistant)

            if tool_calls:
                tc = tool_calls[0]
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"code": fn.get("arguments", "")}
                self.text_hist.append({"role": "assistant", "content": f"[Function Call] {content}"})
                self._traj(turn_idx=self.turn, text_output=content, tool_name=name)
                if name != "code_interpreter":
                    msgs, summary, _ = tool_feedback(tc.get("id"), f"Error: unknown tool '{name}'. Only code_interpreter is available.",
                                                     [], cfg.max_pixels, cfg.min_pixels)
                    self.api.extend(msgs)
                    self.text_hist.append({"role": "user", "content": summary})
                    continue
                code = args.get("code", "") if isinstance(args, dict) else str(args)
                self.n_tool_calls += 1
                norm = re.sub(r"\s+", " ", code.strip())
                same_code = same_code + 1 if norm == last_code else 0
                last_code = norm
                res, new_images = self._run_code(code, source="model")
                msgs, summary, _sent = tool_feedback(tc.get("id"), res.text, new_images, cfg.max_pixels, cfg.min_pixels)
                self.api.extend(msgs)
                self.text_hist.append({"role": "user", "content": summary})
                sig = res.error_signature
                same_err = same_err + 1 if (sig and sig == last_err) else 0
                last_err = sig
                # loop detection
                n_same = max(same_code, same_err)
                if n_same + 1 >= cfg.loop_commit_at:
                    self.termination = "loop"
                    self._traj(turn_idx=self.turn, loop_detected={"same_code": same_code, "same_err": same_err, "sig": sig})
                    break
                if n_same + 1 >= cfg.loop_hint_at:
                    hint = None
                    if sig:
                        for pat, h in _LOOP_HINTS:
                            if re.search(pat, sig):
                                hint = h
                                break
                        hint = hint or f"The same error occurred again ({sig}). Fix its cause instead of re-running the same code."
                    else:
                        hint = ("You submitted the same code as in your previous turn; it produces the same result. Change the "
                                "approach (region, method or parameters) or give your final answer.")
                    self._inject_user("[loop guard] " + hint, tag="loop_hint")
                continue

            # ---- text reply
            self.text_hist.append({"role": "assistant", "content": content})
            self._traj(turn_idx=self.turn, text_output=content)
            ans = last_answer(content)
            if ans is None:
                if fmt_nudges < cfg.format_nudges:
                    fmt_nudges += 1
                    self._inject_user(NO_TOOLS_ANSWER_NUDGE if cfg.no_tools else ANSWER_NUDGE, tag="format_nudge")
                    continue
                final, self.committed_by, self.termination = content.strip(), "raw_text", "format_fallback"
                break
            self.candidates.append(ans)
            final, self.committed_by, self.termination = ans, "model", "answered"
            break

        if final is None:
            final = self._force_commit()

        # the judge reads assistant texts; make sure the committed value is what it grades
        assistant_texts = [m["content"] for m in self.text_hist if m["role"] == "assistant"]
        if last_answer(assistant_texts[-1] if assistant_texts else "") != final and not final.startswith("Error:"):
            synthetic = f"<answer>{final}</answer>"
            self.text_hist.append({"role": "assistant", "content": synthetic, "synthetic": self.committed_by})
            assistant_texts.append(synthetic)

        score, analysis, judge_used = (0.0, "no judge", "none")
        if self.judge_fn is not None:
            score, analysis, judge_used = self.judge_fn(task, assistant_texts, final)

        result = self._finish(final, score, analysis, judge_used)
        return result

    # ------------------------------------------------------------------ forced commit
    def _force_commit(self) -> str:
        cfg = self.cfg
        # one answer-only call at temperature 0, no tools
        msgs = self.api + [{"role": "user", "content": "The turn budget is exhausted. Based on everything above, reply "
                                                        "with ONLY your final answer wrapped in <answer></answer> tags."}]
        resp = call_model(msgs, cfg.model, {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024}, tools=None)
        self.turn += 1
        if isinstance(resp, dict):
            content = resp.get("content") or ""
            self.text_hist.append({"role": "user", "content": "[forced commit] reply with only the final answer"})
            self.text_hist.append({"role": "assistant", "content": content})
            self._traj(turn_idx=self.turn, text_output=content, forced_commit="oneshot")
            ans = last_answer(content)
            if ans:
                self.committed_by = "forced_oneshot"
                return ans
            if content.strip():
                self.committed_by = "forced_oneshot_raw"
                return content.strip()
        if self.candidates:
            self.committed_by = "forced_last_candidate"
            return self.candidates[-1]
        texts = [m["content"] for m in self.text_hist if m["role"] == "assistant" and not m["content"].startswith("[Function Call]")]
        self.committed_by = "forced_last_text"
        return texts[-1].strip() if texts else ""

    # ------------------------------------------------------------------ outputs
    def _finish(self, final: str, score: float, analysis: str, judge_used: str) -> dict:
        task = self.task
        trajectory_text = "".join(f"**{m['role']}**: {m['content']}\n\n" for m in self.text_hist)
        extras = {
            "family": task.family, "source": task.source, "n_turns": self.turn, "n_tool_calls": self.n_tool_calls,
            "n_images": len(self.image_map), "termination": self.termination, "committed_by": self.committed_by,
            "candidates": self.candidates, "judge": judge_used,
            "elapsed_s": round(time.time() - self.t0, 1), "model": self.cfg.model,
        }
        result = {
            "question_id": task.task_id, "prompt": task.question_text, "final_answer": final, "ground_truth": task.answer,
            "conversation_history": self.text_hist, "accuracy_score": float(score), "trajectory_text": trajectory_text,
            "trajectory_score": float(score), "trajectory_analysis": analysis, "_rollout_idx": self.r, **extras,
        }
        (self.out_dir / "metrics.json").write_text(json.dumps(
            {"accuracy_score": float(score), "trajectory_score": float(score), "trajectory_analysis": analysis, **extras},
            indent=2, ensure_ascii=False))
        (self.out_dir / "messages.json").write_text(json.dumps(_strip_images(self.api), indent=1, ensure_ascii=False))
        (self.out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False))
        self._traj(turn_idx=self.turn, final_answer=final, accuracy_score=float(score), committed_by=self.committed_by,
                   termination=self.termination)
        return result
