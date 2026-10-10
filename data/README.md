# data/ — the frozen train / test split of the four benchmarks

The experiments use four public benchmarks, each organised into **families** (a family = one task template / generator / question type),
with **one train part and one test part per family and no validation split**.  This directory is the single definition of that split:
our method and every baseline read the same files.  Train is used for everything that is learned or selected (programs, range rules,
replicate selection, deploy check, text skills); test is measured once per family.

| dataset | families | train | test | ratio | source |
|---|---|---|---|---|---|
| TIR-Bench | 12 | 369 | 746 | 1 : 2 | github.com/agents-x-project/TIR-Bench (commit f79c756); 12 of the 13 tasks (`spot_difference` has two images per question and is excluded) |
| PuzzleVQA | 20 | 663 | 1337 | 1 : 2 | HF `declare-lab/puzzlevqa`; 20 templates x 100 questions |
| MIRA | 20 | 183 | 363 | 1 : 2 | HF `YiyangAiLab/MIRA` (CC-BY-4.0); 20 task types, 546 questions (input images only) |
| GameQA-17 | 17 | 259 | 514 | 1 : 2 | HF `Gabriel166/GameQA-140K` (MIT), **test split only**; 17 of the 158 (game x question) tasks |
| total | 69 | 1474 | 2960 | | |

Per-family counts: `manifest.json`.  Family names carry a dataset prefix where the source names would collide (`pz_`, `mr_`, `gq_`).

## Files

```
data/
  manifest.json                     per dataset / family: n_train, n_test; per dataset: the image-sharing counts
  <dataset>/<family>_train.json     one record list per family and split
  <dataset>/<family>_test.json
  <dataset>/train.json  test.json   all families of the dataset
  image_manifest.json               sha256 / pixel sha256 / size of the 4 202 images referenced by the records
  prepare_images.py                 materialises data/images/ from the public releases        check_images.py  verifies them
  images/                           (not in git) the image root: record["images"] are paths relative to it
```

Record format:

```json
{"images": ["tirbench/data/maze/110.png"], "doc_id": "tirbench_maze_91", "problem": "<image>\n...", "solution": "D",
 "data_source": "tirbench", "dataset": "tirbench", "family": "maze", "split": "test"}
```

* `images`: one image per question, relative to the image root.  `problem` starts with `<image>`; multiple-choice options are part of the
  question text (`A. ...`, or `[1] ...` for GameQA).  `solution` is the gold answer as the benchmark gives it (letter, number, text,
  permutation list for jigsaw, `H:MM` / two-word strings for some MIRA types).
* `doc_id` is unique across datasets; `family` is the family name (use this field, not the id, to group records — MIRA gear_rotation ids
  such as `mira_mr_gear_rotation_2_1` carry a question suffix).
* Test records with `image_also_in_train` (`same_family` / `other_family`) share their image with a train record (same board, different
  question): TIR-Bench 4, GameQA 14 same-family + 81 other-family; none in the other three.  Report numbers without them for any method
  that memorises images.
* Extra fields are the benchmarks' own metadata (GameQA `game` / `question_id` / `plot_level` ..., MIRA `uid`).

## Images

Images are not redistributed; they are rebuilt from the public releases into `data/images/` at exactly the paths the records use
(the `_candidates/` prefix is part of the path layout):

```bash
pip install huggingface_hub pyarrow pillow          # already in the main environment
python data/prepare_images.py --dataset all --download      # PuzzleVQA, MIRA, GameQA from the Hub; TIR-Bench: see below
python data/check_images.py --pixels                        # every image must be byte- (or pixel-) identical to image_manifest.json
```

TIR-Bench distributes its images separately from the code: obtain the benchmark's `data/` directory as described in its repository and
place or symlink it at `data/_raw/tirbench/data/` (sub-folders `color/ contrast/ ... word_search/`), then run `prepare_images.py --dataset tirbench`.
PuzzleVQA / GameQA images are copied byte for byte (PuzzleVQA: the PNG bytes of parquet row *n* become `img/<category>/<n>.png`);
MIRA input images are re-encoded as RGB PNG with RGBA composited on white (the cubes types are largely transparent) — `check_images.py --pixels`
compares decoded pixels for these.  Verified on the release machine: all 4 202 images regenerate identically.

## How the split was made

* **TIR-Bench, three PuzzleVQA templates**: per family, a 1 : 1 : 1 partition (PuzzleVQA: `random.Random(20260914).shuffle` in file order,
  fixed block sizes 34 / 33 / 33); the first block is train, the other two together are test.
* **The other 17 PuzzleVQA templates**: `random.Random(20260917).shuffle` of the template's 100 questions in file order, first 33 = train.
* **MIRA**: per type, sort by `uid`, `random.Random("20260917:<type>").shuffle`, first round(n/3) = train, done by image (gear_rotation has two
  questions per image; a type that shares images with an already split type inherits that split — trailer_cubes_missing <- trailer_cubes_count,
  cubes_missing <- cubes_count, rolling_dice_sum <- rolling_dice_top), so no test image appears in any train part.
* **GameQA**: from the official test split, the 22 two-dimensional rendered games (120 tasks) were piloted with the flash-lite executor
  (6 questions per task, then 15–25 more for tasks with pass@1 <= 0.5); the **17 tasks with pooled pilot pass@1 <= 0.5** were kept.  Pilot
  questions were excluded, the rest stratified by `plot_level`, at most 60 sampled with `random.Random("20260918:<family>")`, shuffled,
  first round(n/3) = train.  GameQA-17 numbers are therefore not comparable with the benchmark's leaderboard.

Known gold-label issues kept as is: MIRA trailer_cubes_count uids 7 / 11 / 23 have a gold below the maximum derivable from the three views;
trailer_cubes_missing golds are inconsistent with the images for every model tested; multi_piece_puzzle candidate pieces are not to scale.

## Evaluation conventions

* Executor baseline = the executor's own rollout on the same question (1 rollout per test task in the reported tables; train uses 4).
  Free-form answers are scored by an LLM judge; program readings by exact normalised match / numeric equality / option letter
  (`visualcoding/verify/acquire/adapters.py::is_correct`).
* Δ is paired per task; 95 % intervals are paired task bootstraps; numbers are reported per family, per dataset and pooled.
* A method that needs a development set must carve it out of train.

Licences: TIR-Bench — see its repository; PuzzleVQA — see the declare-lab repository; MIRA — CC-BY-4.0;
GameQA — MIT.
