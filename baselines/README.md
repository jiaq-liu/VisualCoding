# Baselines — five published skill / prompt-learning methods on the frozen split

Five published methods run on the frozen split in `data/` (69 families, 1474 train / 2960 test), every model call
gemini-3.1-pro through the google-genai SDK.  The upstream repositories are not vendored: `setup_baselines.sh` clones
them at the pinned commits, applies `patches/<repo>.patch` and creates one virtualenv per method
(`requirements-<method>.freeze.txt`).

| method | arm | what it learns | rollouts come from |
|---|---|---|---|
| XSkill | tool | experience library + skill document | our executor (`verify.run`) |
| Trace2Skill | tool | skill directory distilled from traces | our executor |
| SkillOpt | tool | skill document trained like a parameter | our executor |
| GEPA | tool-free | evolved system prompt | one VLM call per rollout |
| TextGrad | tool-free | system prompt optimised by textual gradients | one VLM call per rollout |

## Setup

```bash
bash setup_env.sh                      # the main environment (<repo>/.venv): our executor, the judge, GEPA
bash baselines/setup_baselines.sh      # clone + patch gepa / textgrad / XSkill / Trace2Skill / SkillOpt, create baselines/.venvs/<method>
bash baselines/tests/run_all.sh        # offline mechanism tests of the five drivers (no API calls)
```

Paths (`common/paths.py`, all overridable by environment variables): `VC_ROOT` = `visualcoding/`, `VC_PY` = the main interpreter
(`<repo>/.venv/bin/python`), `DATA_SPLIT` = `data/`, `IMAGE_ROOT` = `data/images`, `BASELINE_OUT` = `output/baselines/`,
`ROLLOUT_ROOTS` = the executor rollout trees that may be reused (`visualcoding/output/verify`).  The key comes from `GEMINI_API_KEY`
or `<repo>/.env`.

## Protocol

Per family, **single pass**: one pass over the family's train set at 4 rollouts per task under the seed (empty) skill, one artifact, one
test deployment at 1 rollout per task; no held-out validation, so every artifact is accepted by default; the artifact is appended to the
system prompt (tool arm: `--system-prompt-extra`, XSkill Phase II optionally per task via `--inject-dir`); every arm is reported against a
no-skill control measured in the same arm.

XSkill and Trace2Skill are single pass by construction.  The other three were reduced to it, each with its search still available behind a flag:

| method | single pass means | restore the search |
|---|---|---|
| SkillOpt | `batch_size: 0` (the family is one batch), `sel_env_num: 0` (no selection rollouts), empty initial skill | `--batch-size 16 --sel-env-num 8` |
| GEPA | one call to GEPA's own `InstructionProposalSignature` on the seed traces; no candidate re-evaluation | `--passes N` |
| TextGrad | one epoch, no measurement, ship what the pass produced | `--epochs 3 --measure-rollouts 4` |

**Two executors, one set of artifacts.**  Every artifact is learned from the pro executor's traces and then deployed on both executors
(the same convention as the main method).  Pass `--model gemini-3.1-flash-lite` to `deploy` and `control`; the tool arm puts the executor
tag in every output path (`deploy_test_flashlite`, `_control_tool/flashlite/...`) because `executor.complete()` decides "already done"
from the output dir alone.

**Judge.**  `BASELINE_JUDGE_MODEL` (default `gemini-3.1-flash-lite`; the judge prompt is text-only: question, gold and the client-side
extracted answer).  The judge model is not part of the rollout signature, so a reused rollout keeps the score of the judge that scored it;
set `BASELINE_JUDGE_MODEL=gemini-3.1-pro-preview` to use one judge everywhere.

## Layout

```
baselines/
  common/         shared layer, importable from any venv (stdlib only)
    paths.py      roots, main-environment interpreter, API key resolution
    data.py       the 69 families, their records, executor input files
    store.py      index of existing verify.run output trees -> reusable rollouts
    executor.py   tool arm: runs verify.run (subprocess, main venv) with reuse
    vlm.py/_cli   tool-free arm: one call per rollout, same judge, cached by prompt hash
    traj.py       trajectory rendering from traj.jsonl (WITH the code)
    gemini.py     google-genai wrapper (messages or text, multimodal, retries)
    driver.py     artifact store, deploy/control, shared CLI
  drivers/        run_gepa.py run_textgrad.py run_xskill.py run_trace2skill.py run_skillopt.py
  scripts/        run_tool_arm.sh (XSkill + Trace2Skill + SkillOpt), run_toolfree_arm.sh (GEPA + TextGrad): resumable chains over all stages
  tools/          score_baselines.py: the macro-accuracy table of both arms against their controls
  tests/          offline mechanism tests (no API calls, no rollouts)
  patches/        our changes to the four patched upstream repos (applied by setup_baselines.sh)
  .venvs/         one venv per method (created by setup_baselines.sh, not in git)
  <clone>/        the five upstream repos at the pinned commits (created by setup_baselines.sh, not in git)
```

Outputs go to `<repo>/output/baselines/<method>/<dataset>__<family>/` (artifact.md, artifact.json, train/, deploy_test_<executor>/) and
`_vlm/<model>/<prompt_sha>/` for the tool-free arm.

## Running

```bash
cd baselines
# always plan first: what would be reused, what would cost a call
.venvs/xskill/bin/python drivers/run_xskill.py train --families all --dry-run

# per method, single pass (train uses 4 rollouts, deploy 1) -- defaults are the protocol
.venvs/xskill/bin/python      drivers/run_xskill.py      train  --families all
.venvs/trace2skill/bin/python drivers/run_trace2skill.py train  --families all
.venvs/skillopt/bin/python    drivers/run_skillopt.py    train  --families all
../.venv/bin/python           drivers/run_gepa.py        train  --families all
.venvs/textgrad/bin/python    drivers/run_textgrad.py    train  --families all

# deployment and the per-arm control (1 rollout, test split), on BOTH executors
... drivers/run_<m>.py deploy  --families all --split test
... drivers/run_<m>.py control --families all --split test
... drivers/run_<m>.py deploy  --families all --split test --model gemini-3.1-flash-lite
... drivers/run_<m>.py control --families all --split test --model gemini-3.1-flash-lite

# or the two resumable chains (all stages, all methods of the arm; DS = all | dataset | comma list)
DS=all nohup bash scripts/run_tool_arm.sh > /dev/null 2>&1 &
DS=all nohup bash scripts/run_toolfree_arm.sh > /dev/null 2>&1 &
../.venv/bin/python tools/score_baselines.py            # macro-accuracy table, both executors, against each arm's control
```

`--families` takes `all`, a dataset name, or a comma list of `dataset:family` keys; `--limit N` keeps the first N families;
`status` prints what has been learned so far.

## Rollout reuse

A rollout is interchangeable when task, model, loop configuration and *effective system prompt* match; `verify.run` records all of it
(`run_config.json`, per-rollout `injected_system_prompt.txt`).  `common/store.py` indexes the existing trees under `ROLLOUT_ROOTS` and
symlinks matching rollout dirs into a new output dir, which `verify.run` then skips.  Because every method's seed skill is empty, the seed
pass and the no-skill control are the main method's executor rollouts when those exist (run the main pipeline first to share them);
`_reuse.json` in each output dir records where each one came from.  The tool-free arm has no pool of its own: its seed pass on train and
its control on test are paid once, shared by GEPA and TextGrad because `common/vlm.py` addresses runs by prompt hash.

## Changes to the upstream clones (`patches/`)

* **XSkill** — `eval/exskill/gemini_bridge.py` (new): routes the experience/skill LLM calls and the embedding calls to google-genai
  (embedding batches capped at 100 texts, as the API requires); two call sites patched (`llm_client._call_with_fallback`,
  `experience_retriever._post_embedding_request`).  The learning modules are otherwise untouched; the rollout loop is not used.
* **Trace2Skill** — `src/react_agent/gemini_client.py` (new): `OpenAIClient`-shaped client plus an `openai`-shaped shim; `--llm_client gemini`
  added to the two LLM analysis scripts and `--llm-client gemini` to the two evolvers; `vqa_skill_seed/` is the neutral seed skill;
  `skills/skill-creator/scripts/quick_validate.py` is git-ignored upstream (a fresh clone exits before doing any work without it), so the
  patch adds a copy.
* **SkillOpt** — `skillopt/model/gemini_backend.py` (new backend `gemini_chat`, wired into the dispatch and the backend whitelists) and
  `skillopt/envs/visqa/` (new env whose rollout helper calls our executor), `configs/visqa/default.yaml`, env registered in
  `scripts/train.py` / `scripts/eval_only.py`.  `skillopt/engine/trainer.py` gained two guards: `train.batch_size: 0` resolves to the whole
  train split, and `evaluation.sel_env_num <= 0` skips the baseline and per-step selection rollouts (with `use_gate: false` their score
  decided nothing; the newest skill then wins by construction).
* **TextGrad** — `textgrad/engine/gemini_genai.py` (new multimodal engine on google-genai; the shipped `gemini.py` targets the deprecated
  SDK, is text-only and caches on text only, which would silently serve one image's answer for another); registered in `engine/__init__.py`,
  where the litellm import was also made lazy.
* **GEPA** — unchanged; the adapter, the reflection LM and the single-pass path (which calls GEPA's own
  `strategies.instruction_proposal.InstructionProposalSignature` directly) are all supplied from `drivers/run_gepa.py`.

## Caveats

* The reflection/analysis stages must read `traj.jsonl`, not `conversation_history`: the latter renders tool calls as `[Function Call]`
  with the code stripped.  `common/traj.py` is the only sanctioned renderer.
* SkillOpt's costs hide in three config keys, all pinned in `configs/visqa/default.yaml`: `sel_env_num > 0` buys a selection pass per edit
  plus a baseline pass; `batch_size > 0` splits the family into steps, and every step after the first rolls out under an edited (hence
  uncached) skill; `use_slow_update` / `use_meta_skill` are no-ops at one epoch but cost 2 x 20 x 4 fresh agent rollouts per extra epoch.
  `tests/test_skillopt_singlepass_offline.py` runs the real trainer and asserts there is exactly one rollout call.  `val/` and `test/`
  still hold the family's train records, so an accidental in-SkillOpt evaluation cannot touch the held-out split.
* `SkillOpt/skillopt/envs/visqa/skills/initial.md` must stay **empty**: any content changes `store.prompt_sha` and reuse drops to zero.
* TextGrad with `--measure-rollouts 0` (the default) can ship a drifted prompt — that is what default-accept means here;
  `--measure-rollouts 1` adds a train-side safety net.
* A driver never saves an artifact for a failed or degenerate run (`train` skips a family that already has one, so an empty artifact would
  survive every retry and then deploy as "what the method learned"); the three tool drivers return `FAILED` instead.
* SkillOpt's analyst reads `<rollout_dir>/predictions/<task_id>/conversation.json`; writing the conversation one level up is silent (the
  analyst finds no trajectories, returns 0 edits without calling the model, and the step is skipped with returncode 0).
* A single-pass GEPA is GEPA's reflection without GEPA's search; `--passes 1` keeps one real accept/reject.
* XSkill's per-sample adaptation (Phase II) is `deploy --adapt`; without it the frozen skill plus experience block is injected once for
  the whole family.

## Tests

`tests/test_<method>_offline.py` — no API calls, no rollouts, no subprocesses: stubs the LM and the executor and exercises the real
upstream machinery (GEPA's engine, textgrad's graph, XSkill's critique/merge, the Trace2Skill command wiring, SkillOpt's env + backend).
The GEPA and TextGrad tests cover both the single-pass and the search path; `test_skillopt_singlepass_offline.py` drives the real
`ReflACTTrainer.train()` and asserts the single-pass budget (one rollout call, the whole family, an empty skill).

```bash
bash tests/run_all.sh
```
