"""Deploy a frozen acquired instrument as a regular `Instrument` so instrument_eval can run / report it on test.

    python -m verify.acquire.deploy --frozen output/verify/acq/<run>/frozen [...] -- run --out ... --split test \
        --instruments acq.<family>.<tag> --model <executor> --workers 1
"""
from __future__ import annotations

import json
import signal
import sys
import threading
from pathlib import Path

from .. import instruments as I
from . import adapters as A
from . import fit_rule as F
from .primitives import Lib
from .runner import TaskView, _exec_measure, _jsonable


class Acquired(I.Instrument):
    def __init__(self, frozen_dir: str | Path, key: str | None = None):
        d = Path(frozen_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.rule = json.loads((d / "rule.json").read_text())
        self.fn = _exec_measure((d / "code.py").read_text())
        self.key = key or self.meta["key"]
        self.families = tuple(self.meta["families"])
        self.deterministic = self.meta.get("n_calls_train", 0) == 0
        self.convention = self.meta.get("convention", "identity")
        self.timeout = int(self.meta.get("timeout_task", 60))
        self.cost_note = f"acquired by {self.meta.get('model')} ({self.meta.get('rounds')} rounds)"

    def measure(self, task, img, call):
        lib = Lib(call)
        tv = TaskView(task.task_id, task.question_text, task.options, task.family, task.source)
        rd = I.Reading()
        main = threading.current_thread() is threading.main_thread()
        if main:
            def _alarm(signum, frame):
                raise TimeoutError(f"measure timeout {self.timeout}s")
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(self.timeout)
        try:
            res = self.fn(img, tv, lib)
            rd.value = _jsonable(res.get("value")) if isinstance(res, dict) else None
            d = res.get("diag") if isinstance(res, dict) else None
            rd.diag = _jsonable(d) if isinstance(d, dict) else {}
        finally:
            if main:
                signal.alarm(0)
        rd.n_calls, rd.contents = lib.n_calls, lib.contents
        rd.answer = A.answer_from_value(rd.value, task, self.convention)
        rd.in_range = F.range_from_row(self.rule, {"value": rd.value, "answer": rd.answer, "diag": rd.diag}, task)
        return rd

    def is_correct(self, answer, task):
        return A.is_correct(answer, task)

    def range_from_row(self, row, task):
        return F.range_from_row(self.rule, row, task)


def register(frozen_dir: str | Path, key: str | None = None) -> Acquired:
    inst = Acquired(frozen_dir, key)
    I.INSTRUMENTS[inst.key] = inst
    return inst


def main():
    argv = sys.argv[1:]
    if "--" not in argv or argv[0] != "--frozen":
        raise SystemExit(__doc__)
    i = argv.index("--")
    for d in argv[1:i]:
        inst = register(d)
        print(f"[deploy] registered {inst.key} families={inst.families} deterministic={inst.deterministic}", flush=True)
    from .. import instrument_eval as E
    sys.argv = [sys.argv[0]] + argv[i + 1:]
    E.main()


if __name__ == "__main__":
    main()
