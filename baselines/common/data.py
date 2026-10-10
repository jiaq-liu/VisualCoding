"""data_split access: the 69 families, their train/test records, and input files for the executor.

A *record* is the raw data_split dict ({images, doc_id, problem, solution, data_source, family, ...});
that is also the input format `verify.run --input` reads, so subsets are passed on by writing a json.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .paths import DATA_SPLIT

DATASETS = ("tirbench", "puzzlevqa", "mira", "gameqa")


@dataclass(frozen=True)
class Family:
    dataset: str
    name: str                 # e.g. "color", "pz_venn", "gq_pacman_q1"

    @property
    def key(self) -> str:     # "tirbench:color" — the family key used by the acq runs / reports
        return f"{self.dataset}:{self.name}"

    @property
    def slug(self) -> str:    # filesystem-safe
        return f"{self.dataset}__{self.name}"

    def file(self, split: str) -> Path:
        return DATA_SPLIT / self.dataset / f"{self.name}_{split}.json"

    def records(self, split: str) -> list[dict]:
        return json.loads(self.file(split).read_text())


def families(datasets: tuple[str, ...] | list[str] = DATASETS) -> list[Family]:
    """All families of the given datasets, dataset order then file order."""
    out = []
    for ds in datasets:
        for f in sorted((DATA_SPLIT / ds).glob("*_train.json")):
            name = f.name[: -len("_train.json")]
            if name == "train":          # the merged per-dataset file, not a family
                continue
            out.append(Family(ds, name))
    return out


def family(key: str) -> Family:
    """Look up by "dataset:family" or by bare family name (must be unique)."""
    if ":" in key:
        ds, name = key.split(":", 1)
        return Family(ds, name)
    hits = [f for f in families() if f.name == key]
    if len(hits) != 1:
        raise KeyError(f"{key!r} matches {len(hits)} families")
    return hits[0]


def records(split: str, datasets=DATASETS) -> list[dict]:
    """Every record of a split across the given datasets."""
    return [r for f in families(datasets) for r in f.records(split)]


def write_input(recs: list[dict], path: Path) -> Path:
    """Write records as a `verify.run --input` file (also used as the reproducible subset manifest)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
    return path


def minibatch(recs: list[dict], size: int, seed: int) -> list[dict]:
    """Deterministic sample without replacement (prompt optimizers draw train minibatches)."""
    if size >= len(recs):
        return list(recs)
    return random.Random(seed).sample(recs, size)


def summary() -> dict:
    """{dataset: {n_families, n_train, n_test}} plus totals — used by the setup check."""
    out, tot = {}, {"n_families": 0, "n_train": 0, "n_test": 0}
    for ds in DATASETS:
        fams = families([ds])
        d = {"n_families": len(fams),
             "n_train": sum(len(f.records("train")) for f in fams),
             "n_test": sum(len(f.records("test")) for f in fams)}
        out[ds] = d
        for k in tot:
            tot[k] += d[k]
    out["TOTAL"] = tot
    return out
