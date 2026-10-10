"""Verify (or build) the image manifest of the frozen split.

    python data/check_images.py                      # check data/images against image_manifest.json
    python data/check_images.py --image-root DIR     # check another image root
    python data/check_images.py --pixels             # on a byte mismatch, also compare decoded RGB pixels (re-encoded PNGs differ in bytes only)
    python data/check_images.py --write              # (maintainers) rebuild image_manifest.json from the image root

image_manifest.json: {"<path relative to the image root>": {"dataset", "bytes", "sha256", "pixel_sha256"}} for every image referenced by the split
(pixel_sha256 = sha256 of the "WxH" size string + the raw RGB bytes of the decoded image)."""
import argparse, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATASETS = ("tirbench", "puzzlevqa", "mira", "gameqa")


def referenced():
    out = {}
    for ds in DATASETS:
        for sp in ("train", "test"):
            for r in json.load(open(HERE / ds / f"{sp}.json")):
                for im in r["images"]:
                    out.setdefault(im, ds)
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pixel_sha(p):
    from PIL import Image
    im = Image.open(p).convert("RGB"); h = hashlib.sha256(f"{im.size[0]}x{im.size[1]}".encode()); h.update(im.tobytes()); return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image-root", default=str(HERE / "images")); ap.add_argument("--write", action="store_true"); ap.add_argument("--pixels", action="store_true")
    a = ap.parse_args(); root = Path(a.image_root); ref = referenced(); mp = HERE / "image_manifest.json"
    if a.write:
        man = {im: {"dataset": ds, "bytes": (root / im).stat().st_size, "sha256": sha(root / im), "pixel_sha256": pixel_sha(root / im)} for im, ds in sorted(ref.items())}
        mp.write_text(json.dumps(man, indent=0)); print(f"wrote {len(man)} entries"); return
    man = json.load(open(mp)); assert set(man) == set(ref), "manifest and split files disagree"
    bad = {ds: [0, 0, 0] for ds in DATASETS}                       # missing, different, ok
    for im, m in man.items():
        p = root / im
        if not p.is_file():
            bad[m["dataset"]][0] += 1
        elif sha(p) == m["sha256"] or (a.pixels and pixel_sha(p) == m["pixel_sha256"]):
            bad[m["dataset"]][2] += 1
        else:
            bad[m["dataset"]][1] += 1
    for ds, (mi, di, ok) in bad.items():
        print(f"{ds:10s} ok {ok:5d}  missing {mi:5d}  different {di:5d}")
    sys.exit(0 if all(v[0] == 0 and v[1] == 0 for v in bad.values()) else 1)


if __name__ == "__main__":
    main()
