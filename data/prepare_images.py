"""Materialise the images of the frozen split under data/images/ from the ORIGINAL public releases of the four benchmarks.

    python data/prepare_images.py --dataset all --download          # fetch the raw releases into data/_raw/ and convert
    python data/prepare_images.py --dataset mira --raw /my/raw      # convert from raw files that are already on disk
    python data/check_images.py [--pixels]                          # verify against image_manifest.json

Only images referenced by the split files are written, at exactly the relative paths stored in the records ("images").
Raw layout under --raw (what --download creates):
    tirbench/data/<task>/<file>                    TIR-Bench images (see README: the benchmark's `data/` directory)       -> tirbench/data/<task>/<file> (copied)
    puzzlevqa/train-00000-of-00001.parquet         HF dataset declare-lab/puzzlevqa (data/train-00000-of-00001.parquet)    -> _candidates/puzzlevqa/img/<category>/<row index>.png (raw bytes of row <row index>)
    mira/<task>/image/<stem>.<ext>                 HF dataset YiyangAiLab/MIRA (input images only, never the cot/ images)  -> _candidates/mira/img/<task>__<stem>.png (RGBA composited on white, saved as RGB PNG)
    gameqa/games_images_test.zip                   HF dataset Gabriel166/GameQA-140K                                       -> _candidates/gameqa/img/<game>__<file> (copied from games_images_test/<game>/<file>)
Image paths keep the `_candidates/` prefix the records use."""
import argparse, io, json, shutil, subprocess, sys, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATASETS = ("tirbench", "puzzlevqa", "mira", "gameqa")
HF = {"puzzlevqa": ("declare-lab/puzzlevqa", ["data/train-00000-of-00001.parquet"]),
      "gameqa": ("Gabriel166/GameQA-140K", ["games_images_test.zip"])}
TIR_REPO = "https://github.com/agents-x-project/TIR-Bench"


def referenced(ds):
    ims = set()
    for sp in ("train", "test"):
        for r in json.load(open(HERE / ds / f"{sp}.json")):
            ims.update(r["images"])
    return sorted(ims)


def download(ds, raw):
    d = raw / ds; d.mkdir(parents=True, exist_ok=True)
    if ds == "tirbench":
        if not (d / "data").is_dir():
            print(f"[tirbench] TIR-Bench distributes its images separately from the code: follow {TIR_REPO} and place (or symlink) the benchmark's "
                  f"`data/` directory at {d / 'data'} (sub-folders color/ contrast/ instrument/ jigsaw/ math/ maze/ ocr/ refcoco/ rotation_game/ symbolic/ visual_search/ word_search/).")
        return
    from huggingface_hub import hf_hub_download, snapshot_download
    if ds == "mira":
        snapshot_download("YiyangAiLab/MIRA", repo_type="dataset", local_dir=str(d), allow_patterns=["*/image/*"])
        return
    repo, files = HF[ds]
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=str(d / "_hf")); shutil.copyfile(p, d / Path(f).name)


def convert(ds, raw, out, force=False):
    ims = referenced(ds); todo = [im for im in ims if force or not (out / im).is_file()]
    print(f"[{ds}] {len(ims)} images referenced, {len(todo)} to write"); d = raw / ds
    if not todo:
        return
    for im in todo:
        (out / im).parent.mkdir(parents=True, exist_ok=True)
    if ds == "tirbench":
        for im in todo:                                          # tirbench/data/<task>/<file>
            shutil.copyfile(d / Path(im).relative_to("tirbench"), out / im)
    elif ds == "gameqa":
        with zipfile.ZipFile(d / "games_images_test.zip") as z:
            for im in todo:
                game, name = Path(im).name.split("__", 1); (out / im).write_bytes(z.read(f"games_images_test/{game}/{name}"))
    elif ds == "puzzlevqa":
        import pyarrow.parquet as pq
        T = pq.read_table(d / "train-00000-of-00001.parquet", columns=["image", "category"])
        for im in todo:
            i = int(Path(im).stem); r = T.slice(i, 1).to_pylist()[0]
            assert r["category"] == Path(im).parent.name, (im, r["category"]); (out / im).write_bytes(r["image"]["bytes"])
    elif ds == "mira":
        from PIL import Image
        for im in todo:
            task, stem = Path(im).stem.split("__", 1)
            src = [q for q in (d / task / "image").glob(stem + ".*") if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
            assert len(src) == 1, (im, src); x = Image.open(src[0])
            if x.mode in ("RGBA", "LA", "P"):
                x = x.convert("RGBA"); bg = Image.new("RGBA", x.size, (255, 255, 255, 255)); x = Image.alpha_composite(bg, x)
            x.convert("RGB").save(out / im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", choices=DATASETS + ("all",)); ap.add_argument("--raw", default=str(HERE / "_raw")); ap.add_argument("--out", default=str(HERE / "images"))
    ap.add_argument("--download", action="store_true"); ap.add_argument("--force", action="store_true", help="rewrite images that already exist")
    a = ap.parse_args(); raw = Path(a.raw); out = Path(a.out)
    for ds in (DATASETS if a.dataset == "all" else (a.dataset,)):
        if a.download:
            download(ds, raw)
        try:
            convert(ds, raw, out, a.force)
        except FileNotFoundError as e:
            print(f"[{ds}] raw files missing under {raw / ds}: {e}"); sys.exit(1)
    print("done; verify with: python data/check_images.py --pixels")


if __name__ == "__main__":
    main()
