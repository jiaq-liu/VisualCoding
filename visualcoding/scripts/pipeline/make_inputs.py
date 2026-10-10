"""Inputs of one batch from the frozen split (no API):  python make_inputs.py <all | dataset | dataset:family,...>
  <batch>/families.txt                      the families of the batch
  <batch>/inputs/train.json  test.json      every train / test task of those families (executor rollouts)
  <batch>/inputs/skill_<family>_test.json   per-family test input (used only for families routed to the text skill)
  <batch>/jobs.txt                          one acquisition job per family: "<dataset:family> <TAG> --eye-dirs <reference train rollouts>" """
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as C   # noqa: E402
fams = C.parse_families(sys.argv[1] if len(sys.argv) > 1 else "all")
(C.B / "inputs").mkdir(parents=True, exist_ok=True)
(C.B / "families.txt").write_text("".join(f"{ds}:{f}\n" for ds, f in fams))


def dump(name, recs):
    json.dump(recs, open(C.B / "inputs" / name, "w"), indent=1, ensure_ascii=False); print(f"{name:44s} {len(recs):5d} tasks  {len(set(r['family'] for r in recs)):3d} families")


D = {sp: [r for ds, f in fams for r in json.load(open(C.DATA / ds / f"{f}_{sp}.json"))] for sp in ("train", "test")}
dump("train.json", D["train"]); dump("test.json", D["test"])
for ds, f in fams:
    dump(f"skill_{f}_test.json", [r for r in D["test"] if r["family"] == f])
(C.B / "jobs.txt").write_text("".join(f"{ds}:{f} {C.TAG} --eye-dirs output/verify/{C.BATCH}/train_{C.REF}\n" for ds, f in fams))
print(f"batch {C.BATCH}: {len(fams)} families, reference executor {C.REF}, executors {C.EXECUTORS}")
