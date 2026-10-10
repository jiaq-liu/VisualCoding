# scripts/

Every script derives its roots from its own location (`ROOT` = `visualcoding/`, override with `VC_ROOT`; `PY` = `<repo>/.venv/bin/python`
unless set) and can be run from any working directory.  Outputs go to `visualcoding/output/verify/<batch>/`.

## `pipeline/` — the full method on any set of families of the frozen split

```
BATCH=run1 nohup setsid bash visualcoding/scripts/pipeline/chain.sh tirbench:maze,tirbench:rotation_game > /dev/null 2>&1 &
tail -f visualcoding/output/verify/run1/status.log
```

| script | step | API calls |
|---|---|---|
| `_env.sh` | roots, interpreter, key export (never printed) | – |
| `config.py` | batch constants: `BATCH`, `TAG`, `EXECUTORS` (default `flashlite pro`), `REF` (reference executor, default flash-lite), `SKILL_FROM` (trajectories summarised into skills, default pro), `WRITER`, `JUDGE`; reads the frozen routes | – |
| `make_inputs.py <all \| dataset \| dataset:family,...>` | `families.txt`, `inputs/{train,test}.json`, per-family test inputs, `jobs.txt` | – |
| `rollouts.sh <name> <executor> <input> <workers> <rollouts> [skill.md]` | executor rollouts with `verify.run`, resumable, `<name>_done.flag` | executor + judge |
| `run_one.sh` / `run_adaptive.sh` / `run_batch.sh` | acquisition: one writer loop / adaptive replication + route / `xargs` over `jobs.txt` | writer (+ `lib.ask`) |
| `adaptive_rule.py decide \| select \| route` | the replication stop rule, the selection rule and the route file, all from TRAIN files | – |
| `run_test.sh <dataset:family> <tag>` | test readings of the selected program, once (refuses a second run) | `lib.ask` programs only |
| `check.py` | per-executor deploy check on TRAIN readings vs. TRAIN rollouts -> `deploy_check.json`, `skill_jobs.txt` | – |
| `report.py` | policy per task / family / dataset, with and without the deploy check -> `report.md`, `summary.json`, `policy_rows.jsonl`, `costs.json` | – |
| `chain.sh` | steps 1–6 in order, server side, resumable | all of the above |

What `chain.sh` does, in order (every step reads only TRAIN files until the test measurements of steps 2 and 4):

1. `train_<REF>/`: 4 rollouts per train task of the reference executor.  The writer is told, per task, how often the executor solved it (k/4).
2. in the background: `test_<X>/` (1 rollout per test task, the baseline of executor X) and `train_<X>/` (4 rollouts) for every other executor X.
3. per family: writer loop -> gate -> replicate again only if rejected or train cover < 50 % (max 3) -> select (LOO gain, cover, index) -> `<family>_<TAG>.route.json`
   (`instrument` when a replicate is deployable, else `text_skill`).
4. `run_test.sh` per family: the selected program reads every test task once (`<run>/test/results.jsonl`, exact-match scoring).
5. `check.py`: the program is deployed for executor X only if its in-range train readings beat X's train pass@1 (one-sided 95 % bootstrap lower bound > 0,
   >= 5 readings).  For every (family, executor) without a deployed program a text skill is summarised from the `SKILL_FROM` executor's train
   trajectories (`text_skills/<dataset>__<family>/skill.md`) and tested with 1 rollout per test task (`skilltest_<family>_<X>/`).
6. `report.py`: per test task, `b` = executor's plain rollout; policy = program reading if deployed and in range, executor + skill if routed to the skill,
   else `b`.  Δ = mean(policy − b), paired task bootstrap 95 % CI; reported per family, per dataset and pooled, with (`check`) and without (`nocheck`)
   the deploy check.
