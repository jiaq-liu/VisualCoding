# `verify` — executor harness, frozen-program evaluation, program acquisition, text skills

Run every module from the package root (`visualcoding/`) as `python -m verify.<module>`.  Paths: `_paths.py`
(`VC_DATA` = the frozen split, default `<repo>/data`; `VC_IMAGE_ROOT` = image root, default `<repo>/data/images`;
outputs under `visualcoding/output/verify/`).  Keys: `GEMINI_API_KEY` in the environment or `<repo>/.env` (`env.py`).

## Executor (the agent whose rollouts are the baseline and the fall-back of every policy)

| file | role |
|---|---|
| `run.py` | CLI: `--input <split json...> --model <executor> --judge-model <judge> --judge llm --rollouts R --workers W [--system-prompt-extra skill.md] --output-dir DIR`; parallel rollouts, resume, `results.jsonl`, `metrics_at_k.json`, `usage.jsonl` |
| `loop.py` | one rollout: multi-turn tool loop (code interpreter), turn / token budgets, answer nudges, forced commit |
| `model.py` | system prompt (verbatim from the agent harness) + optional skill text, tool schema, message building, model call |
| `sandbox.py` | Jupyter-kernel code interpreter with the image bound once |
| `judge.py` | LLM judge (parity with the harness) / exact match |
| `data.py` | records of the frozen split -> `Task` (family from the record, options parsed from the question) |
| `env.py` | key loading; exports what the vendored call / judge modules read |
| `vendor/` | model call, image budget, LLM judge (see `vendor/VENDORED.md`) |

Defaults of every reported rollout: temperature 0.6, top_p 1.0, 8192 completion tokens, 20 turns, 32768 total tokens,
LLM judge; pro executor `thinking_level=low`, flash-lite thinking off.

## Programs (instruments)

| file | role |
|---|---|
| `instruments.py` | `Instrument` base class (`measure(task, img, call) -> Reading`, range flag, exact-match scoring), the registry, and the image / prompt / answer helpers the primitives reuse |
| `instrument_eval.py` | `run` a set of instruments over a split (one reading per task, resumable, exact-match scoring); `split_tasks`, `load_eyeball` and the bootstrap helpers the pipeline reports use |
| `acquire/deploy.py` | wraps a frozen acquired program (`frozen/{code.py, rule.json, meta.json}`) as an `Instrument` and forwards to `instrument_eval` |
| `shapes.py`, `detector.py` | solvers behind `lib` primitives: shape geometry, OWLv2 / Grounding DINO behind `lib.detect` |

## Acquisition (`acquire/`, recipe frozen as v2.9)

| file | role |
|---|---|
| `loop.py` | the writer loop: `python -m verify.acquire.loop --family <dataset:family> --model <writer> --out DIR --eye-dirs <executor train rollout dirs> ...`; K candidates x N rounds on TRAIN, feedback, freeze |
| `playbook.md` | the model-facing instructions for writing `measure(img, task, lib)` |
| `profile.py` | family profile shown to the writer (sizes, palette, components, contact sheet) |
| `primitives.py` | `lib`: generic image / colour / component / projection / panel / detector primitives and one discrete model call (`lib.ask`) |
| `runner.py` | sandboxed subprocess run of a candidate over the train tasks: import allow-list, static checks, per-task timeout, model-read marking |
| `hypothesis.py` | stage A: does the reading explain the train answers; systematic-error diagnostics fed back to the writer |
| `fit_rule.py` | stage B: range rule over the declared witnesses, leave-one-out, rule stability |
| `gate.py` | acceptance tiers (certified / GAIN), `gain_bound` (one-sided bootstrap of program − executor on train) |
| `adapters.py` | reading value -> answer (nearest option, text match, permutation, candidate lists); `is_correct` |
| `prices.json` | assumed per-token prices used only for the cost columns of the reports |

## Text skills (`textskill/`)

`python -m verify.textskill.build --family <dataset:family> --store DIR --executor <model> --train-dirs <train rollout dir>`
summarises one advisory skill from the executor's stored TRAIN trajectories (`harness_prompts/` = the prompt copies of the
recipe); nothing is validated, the skill is deployed as written via `verify.run --system-prompt-extra`.
