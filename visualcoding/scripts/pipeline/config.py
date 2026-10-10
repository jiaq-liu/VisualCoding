"""Shared constants of one pipeline batch (no API).

    BATCH      output sub-directory under visualcoding/output/verify (default run1)
    TAG        run tag of the acquisition loops (default A)
    EXECUTORS  space-separated executors to evaluate (default "flashlite pro")
    REF        executor whose TRAIN rollouts are the acquisition reference (default flashlite)
    SKILL_FROM executor whose TRAIN trajectories are summarised into text skills (default pro if evaluated, else REF)

Families of the batch are listed in <batch>/families.txt ("dataset:family" per line, written by make_inputs.py).
PROG is read from the frozen route files (<batch>/<family>_<TAG>.route.json): family -> (dataset, frozen run, tier)."""
import json, os
from pathlib import Path
ROOT = Path(os.environ.get("VC_ROOT") or Path(__file__).resolve().parents[2])
DATA = Path(os.environ.get("VC_DATA") or ROOT.parent / "data")
BATCH = os.environ.get("BATCH", "run1"); TAG = os.environ.get("TAG", "A"); B = ROOT / "output" / "verify" / BATCH
ALL_MODELS = {"flashlite": "gemini-3.1-flash-lite", "pro": "gemini-3.1-pro-preview"}
EXECUTORS = [x for x in os.environ.get("EXECUTORS", "flashlite pro").split() if x]
assert all(x in ALL_MODELS for x in EXECUTORS), f"EXECUTORS must be a subset of {list(ALL_MODELS)}"
MODEL = {x: ALL_MODELS[x] for x in EXECUTORS}
REF = os.environ.get("REF", "flashlite" if "flashlite" in EXECUTORS else EXECUTORS[0])
SKILL_FROM = os.environ.get("SKILL_FROM", "pro" if "pro" in EXECUTORS else REF)
WRITER = os.environ.get("WRITER", "gemini-3.1-pro-preview")
JUDGE = os.environ.get("JUDGE", "gemini-3.1-pro-preview")


def all_families():
    """Every (dataset, family) of the frozen split, in manifest order."""
    man = json.load(open(DATA / "manifest.json"))
    return [(ds, f) for ds, v in man.items() for f in v["families"]]


def parse_families(spec):
    """'all' | dataset name | comma list of dataset:family (or bare family names)."""
    fams = all_families(); out = []
    for s in [x.strip() for x in spec.split(",") if x.strip()]:
        hit = fams if s == "all" else [k for k in fams if s == k[0] or s == f"{k[0]}:{k[1]}" or s == k[1]]
        assert hit, f"unknown family / dataset '{s}'"
        out += [k for k in hit if k not in out]
    return out


_FT = B / "families.txt"
FAMILIES = [tuple(l.strip().split(":", 1)) for l in _FT.read_text().splitlines() if l.strip()] if _FT.exists() else []
DATASETS = tuple(dict.fromkeys(ds for ds, _ in FAMILIES))
RUN = {ds: [f for d, f in FAMILIES if d == ds] for ds in DATASETS}
PROG = {}; ROUTE_WHY = {}
for ds, f in FAMILIES:
    p = B / f"{f}_{TAG}.route.json"
    if not p.exists():
        continue
    r = json.loads(p.read_text()); ROUTE_WHY[f] = r.get("why")
    if r.get("route") == "instrument" and r.get("instrument_run"):
        run = r["instrument_run"].split("output/verify/")[-1]; sel = [x for x in r["runs"] if x["run"].rstrip("/").endswith(run.split("/")[-1])]
        PROG[f] = (ds, run, (sel[0].get("tier") if sel else None) or "certified")
NO_PROGRAM = {ds: [f for f in RUN[ds] if f not in PROG] for ds in DATASETS}
TRAIN = {x: [f"{BATCH}/train_{x}"] for x in EXECUTORS}
TEST = {x: [f"{BATCH}/test_{x}"] for x in EXECUTORS}
SKILL_STORE = B / "text_skills"
