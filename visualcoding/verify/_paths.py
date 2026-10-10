"""Path constants and the sys.path shim for the vendored agent-harness modules.

Release layout (override with environment variables):
    <repo>/visualcoding/verify      this package            VERIFY_DIR
    <repo>/visualcoding             package root            PKG_ROOT   (run `python -m verify.<module>` from here; outputs go to PKG_ROOT/output/verify)
    <repo>/data                     frozen train/test split DATA_ROOT  (VC_DATA)        <dataset>/<family>_{train,test}.json
    <repo>/data/images              image root              IMAGE_ROOT (VC_IMAGE_ROOT)  record["images"] are relative to it
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parent
PKG_ROOT = VERIFY_DIR.parent
VENDOR_DIR = VERIFY_DIR / "vendor" / "harness"
DATA_ROOT = Path(os.environ.get("VC_DATA", str(PKG_ROOT.parent / "data")))
IMAGE_ROOT = Path(os.environ.get("VC_IMAGE_ROOT", str(DATA_ROOT / "images")))


def split_files(family: str, split: str) -> list[str]:
    """Record files of one family ("family" or "dataset:family") and split: DATA_ROOT/<dataset>/<family>_<split>.json."""
    import glob
    fam = family.split(":")[-1]
    src = family.split(":")[0] if ":" in family else "*"
    return sorted(set(glob.glob(str(DATA_ROOT / src / f"{fam}_{split}.json"))))


def enable_vendor() -> None:
    """Make `engine.*`, `utils.*`, `prompts.*` resolve to verify/vendor/harness/*.

    The vendored files import each other with their original top-level package
    names, so the vendor dir must be on sys.path (front) before importing them.
    """
    p = str(VENDOR_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
