"""verify — the executor harness, frozen-program evaluation, program acquisition and text skills of the method.

Layout (module map with the CLI entry points: README.md in this directory)
  run.py / loop.py / model.py / sandbox.py / judge.py   the executor: CLI, one rollout, prompts + model call, Jupyter-kernel
                                                        code interpreter, LLM judge / exact match
  data.py / env.py / _paths.py / vendor/               records of the frozen split -> Task; key loading; paths; vendored harness modules
  instruments.py / instrument_eval.py                  the Instrument contract (measure -> Reading, range rule, registry) and the
                                                        hard-switch policy evaluation against stored executor rollouts
  acquire/                                             the writer loop (recipe v2.9), the `lib` primitives, rule fitting, acceptance, deploy
  textskill/                                           text skills summarised from stored train trajectories
  shapes.py / detector.py                              solvers behind `lib` primitives (shape geometry, OWLv2 / Grounding DINO)
"""
from ._paths import VERIFY_DIR, PKG_ROOT, VENDOR_DIR, DATA_ROOT, IMAGE_ROOT, enable_vendor

__all__ = ["VERIFY_DIR", "PKG_ROOT", "VENDOR_DIR", "DATA_ROOT", "IMAGE_ROOT", "enable_vendor"]
