"""Reuse of rollouts that already exist on disk.

A rollout is interchangeable with a fresh one when the task, the model, the loop
configuration and the *effective system prompt* all match.  Every `verify.run`
output tree records exactly that (`run_config.json` + per-rollout
`injected_system_prompt.txt`), so this module indexes those trees and symlinks
matching rollout directories into a new output dir.  `verify.run` then skips
them (its resume check is "metrics.json and result.json exist") and only the
gaps cost anything.

The practical consequence: a baseline whose seed skill is empty gets its whole
seed pass — and the no-skill control — for free from the acq runs.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .paths import ROLLOUT_ROOTS, VC_ROOT

# Fields of verify.loop.LoopConfig that change the rollout distribution.
SIG_FIELDS = ("model", "temperature", "top_p", "max_tokens", "max_turns", "commit_reserve",
              "max_images", "max_total_tokens", "max_pixels", "min_pixels",
              "format_nudges", "loop_hint_at", "loop_commit_at", "api_retries")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _const(name: str) -> str:
    """Read a module-level string constant out of verify/model.py without importing it
    (this package must stay importable from venvs that lack the verify dependencies)."""
    src = (VC_ROOT / "verify" / "model.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in verify/model.py")


def system_prompt(skill_text: str | None = None) -> str:
    """Byte-identical to verify.model.system_prompt(extra=skill_text)."""
    s = _const("BASE_SYSTEM_PROMPT")
    if skill_text:
        s += "\n" + skill_text.strip() + "\n"
    return s


def prompt_sha(skill_text: str | None = None) -> str:
    return _sha(system_prompt(skill_text))


def signature(config: dict | None = None, **overrides) -> str:
    """Signature of a loop configuration; defaults are the acq protocol (pro, 4 rollouts)."""
    base = dict(model="gemini-3.1-pro-preview", temperature=0.6, top_p=1.0, max_tokens=8192,
                max_turns=20, commit_reserve=2, max_images=100,
                max_total_tokens=32768, max_pixels=2_000_000, min_pixels=40_000,
                format_nudges=2, loop_hint_at=2, loop_commit_at=3, api_retries=2)
    if config:
        base.update({k: v for k, v in config.items() if k in SIG_FIELDS})
    base.update({k: v for k, v in overrides.items() if k in SIG_FIELDS})
    payload = {k: base[k] for k in SIG_FIELDS}
    return _sha(json.dumps(payload, sort_keys=True))


@dataclass
class Run:
    path: Path
    config: dict
    argv: list
    per_task_prompt: bool            # --inject-dir: prompt differs per task
    sig: str
    prompt_sha: str | None           # None when per_task_prompt
    tasks: set = field(default_factory=set)

    @classmethod
    def load(cls, run_dir: Path):
        try:
            c = json.loads((run_dir / "run_config.json").read_text())
        except Exception:
            return None
        cfg = c.get("config") or {}
        argv = c.get("argv") or []
        per_task = "--inject-dir" in argv or bool(cfg.get("system_prompt_override"))
        return cls(path=run_dir, config=cfg, argv=argv, per_task_prompt=per_task,
                   sig=signature(cfg),
                   prompt_sha=None if per_task else _sha(system_prompt(cfg.get("system_prompt_extra"))))


class RolloutStore:
    """Index of `verify.run` output trees available for reuse."""

    def __init__(self, roots=None, max_depth: int = 2):
        self.roots = [Path(r) for r in (roots or ROLLOUT_ROOTS)]
        self.max_depth = max_depth
        self._runs: list[Run] | None = None

    # ------------------------------------------------------------------ scanning
    def _scan(self) -> list[Run]:
        """Depth-limited walk: run dirs sit at <root>/<name> or <root>/<group>/<name>.

        A recursive glob is not an option: below a run dir there is one directory per
        task per rollout plus the jupyter sandboxes (10^5 entries per tree)."""
        runs, seen = [], set()
        for root in self.roots:
            if not root.is_dir():
                continue
            frontier = [(root, 0)]
            while frontier:
                d, depth = frontier.pop()
                if (d / "run_config.json").is_file():
                    if d not in seen:
                        seen.add(d)
                        r = Run.load(d)
                        if r is not None:
                            runs.append(r)
                    continue                       # never descend into a run dir
                if depth >= self.max_depth:
                    continue
                try:
                    for e in os.scandir(d):
                        if e.is_dir(follow_symlinks=False) and not e.name.startswith((".", "_")):
                            frontier.append((Path(e.path), depth + 1))
                except OSError:
                    continue
        return runs

    @property
    def runs(self) -> list[Run]:
        if self._runs is None:
            self._runs = self._scan()
        return self._runs

    def candidates(self, sig: str, sha: str) -> list[Run]:
        """Runs whose configuration matches; per-task-prompt runs are kept and checked per rollout."""
        return [r for r in self.runs if r.sig == sig and (r.per_task_prompt or r.prompt_sha == sha)]

    # ------------------------------------------------------------------ lookup
    @staticmethod
    def _usable(d: Path, sha: str | None) -> bool:
        if not ((d / "metrics.json").is_file() and (d / "result.json").is_file()):
            return False
        if sha is None:
            return True
        p = d / "injected_system_prompt.txt"
        return p.is_file() and _sha(p.read_text()) == sha

    def find(self, task_id: str, sig: str, sha: str, n: int) -> list[Path]:
        """Up to n reusable rollout dirs for one task (across runs, de-duplicated)."""
        found: list[Path] = []
        for run in self.candidates(sig, sha):
            tdir = run.path / task_id
            if not tdir.is_dir():
                continue
            for d in sorted(tdir.glob("rollout_*"), key=lambda p: p.name):
                if self._usable(d, sha if run.per_task_prompt else None):
                    found.append(d)
                    if len(found) >= n:
                        return found
        return found

    # ------------------------------------------------------------------ materialise
    def materialize(self, out_dir: Path, task_ids, sig: str, sha: str, n: int,
                    dry_run: bool = False) -> dict:
        """Link reusable rollouts into out_dir/<task_id>/rollout_<i>. Returns a reuse report."""
        out_dir = Path(out_dir)
        task_ids = list(task_ids)
        mapping: dict[str, list[str]] = {}
        linked = existing = 0
        for tid in task_ids:
            have = []
            tdir = out_dir / tid
            for i in range(n):
                d = tdir / f"rollout_{i}"
                # lexists, not exists: a symlink whose source went away still occupies the
                # slot, and `verify.run` would then find no result.json behind it.  Clear it.
                if os.path.lexists(d):
                    if not d.exists() and d.is_symlink() and not dry_run:
                        d.unlink()
                        continue
                    have.append(i)
            existing += len(have)
            need = [i for i in range(n) if i not in have]
            if not need:
                continue
            src = self.find(tid, sig, sha, len(need))
            if not src:
                continue
            for i, s in zip(need, src):
                if not dry_run:
                    tdir.mkdir(parents=True, exist_ok=True)
                    try:
                        os.symlink(s, tdir / f"rollout_{i}", target_is_directory=True)
                    except FileExistsError:
                        # The slot was filled between the scan above and here: a second pass
                        # of the same run, or another process pointed at this out_dir.  Keep
                        # whatever is there (a reused link or a real rollout) and move on --
                        # materialising must be safe to repeat, since resume depends on it.
                        existing += 1
                        continue
                mapping.setdefault(tid, []).append(f"rollout_{i} <- {s}")
                linked += 1
        report = {"n_tasks": len(task_ids), "rollouts_per_task": n, "already_present": existing,
                  "reused": linked, "to_run": len(task_ids) * n - existing - linked,
                  "dry_run": dry_run, "map": mapping}
        if not dry_run and mapping:
            out_dir.mkdir(parents=True, exist_ok=True)
            prev = {}
            f = out_dir / "_reuse.json"
            if f.is_file():
                prev = json.loads(f.read_text()).get("map", {})
            prev.update(mapping)
            report_out = dict(report, map=prev)
            f.write_text(json.dumps(report_out, indent=1))
        return report
