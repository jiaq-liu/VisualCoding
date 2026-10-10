"""Stateful Python code interpreter (Jupyter kernel) of the executor loop.

A port of the agent harness `tools/code_interpreter.py` with the same variable contract (`original_image`,
`original_image_N`, `image` alias, `tool_image_N`) and these environment rules:
  * input images are bound ONLY IF the name is not already defined in the kernel, so the
    model's own re-assignments (e.g. `image = image.crop(...)`) survive across turns;
  * an optional prelude (`extra_prelude`, ablations only) is executed once per kernel start;
  * output-silence timeout default 60 s (parameter sweeps are slower than one-off cells);
  * on timeout the kernel is killed and lazily restarted (state lost; reported to the model);
  * one BLAS / OpenCV thread per kernel and an address-space limit (VC_KERNEL_MEM_GB, default 16).
"""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
MAX_CHUNK_CHARS = 20_000
MAX_TOTAL_CHARS = 50_000

_LAUNCH = "from ipykernel import kernelapp as app\napp.launch_new_instance()\n"

def _cap(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    tail = n // 2
    return s[: n - tail] + f"\n...[truncated {len(s) - n} chars]...\n" + s[-tail:]


def _kernel_rlimit() -> None:
    import resource
    gb = float(os.environ.get("VC_KERNEL_MEM_GB", "16"))
    if gb > 0:
        lim = int(gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


@dataclass
class SandboxResult:
    text: str                       # model-visible output
    images: list[bytes] = field(default_factory=list)   # PNG bytes, display order (saved files last)
    meta: dict[str, Any] = field(default_factory=dict)  # restarted / timeout / error / exception / n_images

    @property
    def error_signature(self) -> str | None:
        return self.meta.get("exception")


class Sandbox:
    name = "code_interpreter"

    def __init__(self, work_root: str | Path, output_timeout: float = 60.0, extra_prelude: str = ""):
        self.extra_prelude = extra_prelude
        self.work_dir = Path(work_root).resolve() / uuid.uuid4().hex[:12]
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_timeout = output_timeout
        self.hard_timeout = output_timeout * 3
        self._proc: subprocess.Popen | None = None
        self._kc = None
        self._prelude_done = False
        self.n_restarts = 0
        self._written: dict[str, int] = {}

    # ---------- kernel lifecycle ----------
    def _start(self) -> None:
        from jupyter_client import BlockingKernelClient
        conn = self.work_dir / f"kernel_{uuid.uuid4().hex[:8]}.json"
        launcher = self.work_dir / "launch_kernel.py"
        launcher.write_text(_LAUNCH)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
        env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
        for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                   "OPENCV_FOR_THREADS_NUM"):
            env[_v] = "1"   # one thread per kernel: OpenCV alone spawns nproc workers otherwise
        self._proc = subprocess.Popen(
            [sys.executable, str(launcher), "--IPKernelApp.connection_file", str(conn),
             "--matplotlib=inline", "--quiet"],
            cwd=str(self.work_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
            start_new_session=True, preexec_fn=_kernel_rlimit)
        t0 = time.time()
        while time.time() - t0 < 30:
            if conn.exists():
                try:
                    json.loads(conn.read_text())
                    break
                except (json.JSONDecodeError, OSError):
                    pass
            time.sleep(0.1)
        else:
            self._kill()
            raise RuntimeError("kernel connection file not created within 30 s")
        kc = BlockingKernelClient(connection_file=str(conn))
        kc.load_connection_file()
        kc.start_channels()
        kc.wait_for_ready(timeout=60)
        self._kc = kc
        self._prelude_done = False
        self._written = {}

    def _kill(self) -> None:
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:
                pass
            self._kc = None
        if self._proc is not None:
            try:
                os.killpg(self._proc.pid, 9)
            except Exception:
                pass
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None

    def close(self, remove_work_dir: bool = True) -> None:
        self._kill()
        if remove_work_dir:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)   # per-episode PNG copies of every image

    def __del__(self):
        try:
            self._kill()
        except Exception:
            pass

    def _ensure(self) -> bool:
        """Start the kernel if needed; returns True if a (re)start happened."""
        if self._kc is None or self._proc is None or self._proc.poll() is not None:
            self._kill()
            self._start()
            return True
        return False

    # ---------- preload ----------
    def _preload(self, image_map: dict[str, Image.Image] | None, fresh: bool) -> tuple[str, set[str]]:
        pre = "from PIL import Image\n"
        if fresh or not self._prelude_done:
            if self.extra_prelude:
                pre += self.extra_prelude.rstrip("\n") + "\n"
            self._prelude_done = True
        input_files: set[str] = set()
        if image_map:
            order = sorted(image_map.items(), key=lambda kv: (
                0 if kv[0] == "original_image" else 1 if kv[0].startswith("original_image_") else 2, kv[0]))
            for name, img in order:
                var = re.sub(r"[^a-zA-Z0-9_]", "", name)
                if not var:
                    continue
                p = self.work_dir / f"{var}.png"
                if self._written.get(var) != id(img) or not p.exists():
                    img.save(p)
                    self._written[var] = id(img)
                input_files.add(p.name)
                pre += f"if {var!r} not in globals(): {var} = Image.open({str(p)!r})\n"
            if "original_image" in image_map:
                p = self.work_dir / "image.png"
                if self._written.get("image") != id(image_map["original_image"]) or not p.exists():
                    image_map["original_image"].save(p)
                    self._written["image"] = id(image_map["original_image"])
                input_files.add(p.name)
                pre += "if 'image' not in globals(): image = original_image\n"
        return pre, input_files

    # ---------- execution ----------
    def run(self, code: str, image_map: dict[str, Image.Image] | None = None) -> SandboxResult:
        meta: dict[str, Any] = {"restarted": False, "timeout": False, "error": False, "exception": None}
        m = re.search(r"```[^\n]*\n(.+?)```", code, re.DOTALL)
        if m:
            code = m.group(1)
        if not code.strip():
            return SandboxResult("Error: No code provided", [], meta)
        if re.search(r"\bexit\s*\(|\bquit\s*\(|os\._exit", code):
            t = "Error: Using exit()/quit() is not allowed"
            return SandboxResult(t, [], meta)
        try:
            ast.parse(code)
        except SyntaxError as e:
            meta["error"] = True
            meta["exception"] = f"SyntaxError: {e.msg}"
            t = f"SyntaxError: {e}"
            return SandboxResult(t, [], meta)

        try:
            fresh = self._ensure()
        except Exception as e:
            meta["error"] = True
            t = f"Fatal Error: could not start kernel: {e}"
            return SandboxResult(t, [], meta)
        meta["restarted"] = fresh and self.n_restarts > 0
        pre, input_files = self._preload(image_map, fresh)
        full = pre + code + "\n\n"

        before = {f for f in os.listdir(self.work_dir) if f.lower().endswith(IMAGE_EXTS)}
        kc = self._kc
        try:
            msg_id = kc.execute(full)
        except Exception as e:
            meta["error"] = True
            t = f"Fatal Error: failed to submit code: {e}"
            return SandboxResult(t, [], meta)

        text_parts: list[str] = []
        images: list[bytes] = []
        t_start = last_out = time.time()
        finished = False
        while not finished:
            now = time.time()
            if now - last_out >= self.output_timeout or now - t_start >= self.hard_timeout:
                text_parts.append(f"\nTimeoutError: no output for {now - last_out:.0f}s "
                                  f"(limit {self.output_timeout:.0f}s); the kernel was restarted and "
                                  "all variables were lost (input images are re-loaded automatically). "
                                  "Use smaller regions / fewer sweep values.")
                meta["timeout"] = True
                meta["error"] = True
                meta["exception"] = "TimeoutError"
                self.n_restarts += 1
                self._kill()
                break
            try:
                msg = kc.get_iopub_msg(timeout=min(5.0, max(0.5, self.output_timeout - (now - last_out))))
            except Exception:
                continue
            if msg.get("parent_header", {}).get("msg_id") not in (msg_id, None):
                continue
            mt, c = msg.get("msg_type"), msg.get("content", {})
            if mt == "stream":
                t = c.get("text", "")
                if t.strip():
                    last_out = time.time()
                    text_parts.append(_cap(t, MAX_CHUNK_CHARS))
            elif mt in ("execute_result", "display_data"):
                data = c.get("data", {})
                if "image/png" in data:
                    images.append(base64.b64decode(data["image/png"]))
                    last_out = time.time()
                elif data.get("text/plain", "").strip():
                    last_out = time.time()
                    text_parts.append(_cap(data["text/plain"], MAX_CHUNK_CHARS))
            elif mt == "error":
                meta["error"] = True
                last_out = time.time()
                ename, evalue = c.get("ename", "Error"), c.get("evalue", "")
                meta["exception"] = f"{ename}: {str(evalue)[:120]}"
                tb = "\n".join(re.sub(r"\x1b\[[0-9;]*m", "", ln) for ln in c.get("traceback", []))
                text_parts.append(f"\nerror:\n```\n{_cap(tb, MAX_CHUNK_CHARS)}\n```")
            elif mt == "status" and c.get("execution_state") == "idle" and \
                    msg.get("parent_header", {}).get("msg_id") == msg_id:
                finished = True

        # files saved by the code (the agent harness "auto-detected saved image")
        after = {f for f in os.listdir(self.work_dir) if f.lower().endswith(IMAGE_EXTS)}
        saved = sorted(after - before - input_files)
        if saved:
            disp_keys = set()
            for x in images:
                try:
                    disp_keys.add(hashlib.md5(Image.open(io.BytesIO(x)).convert("RGB").tobytes()).hexdigest())
                except Exception:
                    pass
            for f in saved:
                try:
                    im = Image.open(self.work_dir / f).convert("RGB")
                    key = hashlib.md5(im.tobytes()).hexdigest()
                    if key in disp_keys:
                        continue
                    buf = io.BytesIO()
                    im.save(buf, "PNG")
                    images.append(buf.getvalue())
                    disp_keys.add(key)
                    text_parts.append(f"\n[saved image file {f} captured]")
                except Exception:
                    pass
        meta["n_images"] = len(images)
        text = _cap("".join(text_parts).strip(), MAX_TOTAL_CHARS) or "Finished execution."
        return SandboxResult(text, images, meta)
