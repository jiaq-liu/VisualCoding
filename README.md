# VisualCoding — executable skills for reliable agentic visual reasoning

Code, data split and baseline implementations of the paper.  Anonymous release; no author information, no keys.

**The method in one paragraph.**  A *program* (instrument) is a fixed measurement procedure `measure(img, task, lib) -> {value, diag}`
that runs *outside* the agent loop: `value` is the reading, `diag` are declared numeric witnesses.  A *range rule* fitted on the family's
TRAIN split decides, from the witnesses, when the reading is trusted: in range, the program's answer is final; out of range, the executor
(a multi-turn code-interpreter VLM agent) answers on its own and never sees the program (hard switch).  Programs are *acquired* by a writer
model (gemini-3.1-pro) in a loop with train-side verification: the model makes discrete decisions (hypothesis, primitives, witnesses, one
edit per round), the harness runs the sandbox, fits the rule, checks leave-one-out precision and stability, and freezes accepted programs.
Families where no program is accepted fall back to a *text skill* summarised from the executor's own train trajectories.  Route
(program > text skill > plain executor) and every threshold are decided on train and frozen before the single test measurement.

## Repository layout

```
visualcoding/           the method
  verify/               executor harness (verify.run), program evaluation (instrument_eval), acquisition loop (acquire/), text skills (textskill/)
  scripts/pipeline/     the end-to-end pipeline on any set of families (chain.sh)
data/                   the frozen train / test split of the four benchmarks (records; images are rebuilt from the public releases)
baselines/              five published baselines (XSkill, Trace2Skill, SkillOpt, GEPA, TextGrad): drivers, patches, tests
environment/            dependency pins            setup_env.sh  .env.example
```

## 1. Setup

```bash
bash setup_env.sh                 # python3.10 -> <repo>/.venv with environment/requirements.txt  (--full: the exact frozen environment, torch cu126)
cp .env.example .env              # GEMINI_API_KEY=...   (or export GEMINI_API_KEY; the key is read by visualcoding/verify/env.py and never printed)
```

Models: executors `gemini-3.1-flash-lite` (thinking off) and `gemini-3.1-pro-preview` (`thinking_level=low`); writer `gemini-3.1-pro-preview`;
LLM judge `gemini-3.1-pro-preview`.
Only `lib.detect` / `lib.count_objects` (used by the refcoco program) need torch + a GPU; everything else is numpy / OpenCV / PIL / scipy / skimage.

## 2. Data

Records are in git; images are rebuilt from the public releases (see `data/README.md` for sources, the split procedure and caveats):

```bash
python data/prepare_images.py --dataset all --download      # PuzzleVQA, MIRA, GameQA from the Hugging Face Hub
#   TIR-Bench: put the benchmark's data/ directory at data/_raw/tirbench/data/ and run  python data/prepare_images.py --dataset tirbench
python data/check_images.py --pixels                        # all 4 202 images must match image_manifest.json
```

| dataset | families | train | test |
|---|---|---|---|
| TIR-Bench | 12 | 369 | 746 |
| PuzzleVQA | 20 | 663 | 1337 |
| MIRA | 20 | 183 | 363 |
| GameQA-17 | 17 | 259 | 514 |

## 3. Run the executor baseline alone

```bash
cd visualcoding
python -m verify.run --input ../data/tirbench/maze_test.json --model gemini-3.1-flash-lite --judge-model gemini-3.1-pro-preview \
    --judge llm --rollouts 1 --workers 8 --output-dir output/verify/demo_maze        # results.jsonl, metrics_at_k.json, per-rollout trajectories
```

`--system-prompt-extra <skill.md>` appends a text skill to the executor's system prompt; the pipeline below produces those skills and the
frozen programs.

## 4. Run the full method

```bash
cd visualcoding
BATCH=run1 nohup setsid bash scripts/pipeline/chain.sh tirbench:maze,tirbench:rotation_game > /dev/null 2>&1 &     # or: all | <dataset>
tail -f output/verify/run1/status.log                       # ends with "report written -> .../report.md"
```

`chain.sh` runs, in order: (1) 4 train rollouts per task of the reference executor (flash-lite) = the acquisition reference;
(2) in the background, 1 test rollout per task of every executor (the baselines) and 4 train rollouts of the other executors;
(3) the writer loop per family with adaptive replication (a second / third replicate only if the previous one was rejected or covers < 50 % of
train; max 3) and the train-side selection -> `<family>_A.route.json`; (4) the selected program's test readings, once; (5) the per-executor
deploy check on train (a program is deployed for an executor only if it beats that executor on train), text skills for every
(family, executor) without a deployed program, and their single-rollout test; (6) `report.py` -> `output/verify/run1/report.md`
(per family / per dataset / pooled Δ with paired task-bootstrap 95 % CIs, with and without the deploy check).  Everything is resumable;
test is never measured twice.  Options (`scripts/pipeline/config.py`): `EXECUTORS="flashlite pro"`, `REF`, `SKILL_FROM`, `WRITER`, `JUDGE`,
`PAR` (families in parallel).  Details: `visualcoding/scripts/README.md`; module map: `visualcoding/verify/README.md`.

Cost guide (measured on the PuzzleVQA + MIRA families, prices of `visualcoding/verify/acquire/prices.json`): writer calls 1.4–3.7 USD per accepted program,
≈ 1.3 USD per text skill; executor rollouts ≈ 0.005 USD (flash-lite) and 0.01–0.6 USD (pro, MIRA types are the expensive ones) each,
judge calls not included.

## 5. Baselines

Five published methods under the same split, executor, judge and single-pass protocol (`baselines/README.md` has the protocol, every
patch to the upstream repositories and the caveats):

```bash
bash baselines/setup_baselines.sh                        # clone the five repos at pinned commits, apply baselines/patches/, create baselines/.venvs/
bash baselines/tests/run_all.sh                          # offline mechanism tests, no API calls
cd baselines
.venvs/xskill/bin/python      drivers/run_xskill.py      train --families all --dry-run     # plan: what is reused, what would be called
.venvs/xskill/bin/python      drivers/run_xskill.py      train --families all               # likewise run_trace2skill / run_skillopt / run_textgrad;
../.venv/bin/python           drivers/run_gepa.py        train --families all               # GEPA runs in the main environment
.venvs/xskill/bin/python      drivers/run_xskill.py      deploy  --families all --split test [--model gemini-3.1-flash-lite]
.venvs/xskill/bin/python      drivers/run_xskill.py      control --families all --split test [--model gemini-3.1-flash-lite]
```

XSkill, Trace2Skill and SkillOpt learn from rollouts of *our* executor (`verify.run`) and are deployed by appending the learned artifact to its
system prompt; GEPA and TextGrad are the tool-free arm (one VLM call per rollout) with their own no-skill control.  Executor rollouts that
already exist under `visualcoding/output/verify` are reused when task, model, loop configuration and effective system prompt match
(`baselines/common/store.py`), so the seed pass and the control are shared with the main method.

## 6. Outputs

`visualcoding/output/verify/<batch>/report.md` (written by `report.py` at the end of `chain.sh`) holds the per-family, per-dataset and pooled
table of a run: executor baseline, policy, Δ with paired task-bootstrap 95 % CIs, with and without the deploy check; `policy_rows.jsonl`
is the per-task table behind it; `<family>_<TAG>.route.json`, `<run>/frozen/{code.py, rule.json, meta.json}` and
`text_skills/<dataset>__<family>/skill.md` are the learned artifacts.

## Protocol notes

* Everything is fitted on train only; test is measured once per family (`run_test.sh` refuses a second run).
* The primitive library is fixed across all experiments; no OCR, reader or detector was added for a dataset.
* Program readings are scored by exact normalised match; executor and skill rollouts by the LLM judge.  Δ is paired per task.
* Programs that read through `lib.ask` are served by the writer model at train and test; the reports mark them.
* The executor kernel hands the agent a resized copy of the image; frozen programs read the original file.

## Licence

MIT (see `LICENSE`).  The benchmarks keep their own licences (`data/README.md`).
