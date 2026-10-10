#!/bin/bash
# Tool-free arm: GEPA + TextGrad (one VLM call per rollout, no code interpreter), with their own no-skill control.
#     DS=all bash baselines/scripts/run_toolfree_arm.sh     (DS = all | dataset | comma list of dataset:family)
# The two arms share nothing but the judge, so this chain pays for its own no-skill control and its own seed pass.  Within the chain,
# `common/vlm.py` content-addresses every rollout by <model>/<prompt_sha>/<task>, so GEPA's seed evaluation is paid once no matter how
# often it is asked for, and vlm_cli resumes (it skips any rollout whose result.json exists), which makes every stage restartable.
# TextGrad is sharded because its loop is strictly sequential per example (forward -> evaluator -> textual gradient -> step);
# shards of whole families run the same computation with no interaction: each family owns its own artifact dir and prompt variable.
set -u
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VC_PY=${VC_PY:-$B/../.venv/bin/python}
TG=$B/.venvs/textgrad/bin/python
GP=$VC_PY                       # the gepa clone has no dependencies of its own
DS=${DS:-all}
FL=gemini-3.1-flash-lite
SHARDS=${SHARDS:-6}
W=${W:-8}                         # vlm workers for the batched (gepa / control / deploy) jobs
RUN=${BASELINE_OUT:-$B/../output/baselines}/_toolfree_arm
mkdir -p "$RUN"
S=$RUN/status.log

say() { echo "$(date -Iseconds) $*" | tee -a "$S"; }

stage() {  # stage <name> <cmd...>
  local name=$1; shift
  if [ -f "$RUN/$name.done" ]; then say "skip $name (done)"; return 0; fi
  say "START $name"
  local t0=$SECONDS
  if "$@" >>"$RUN/$name.log" 2>&1; then
    touch "$RUN/$name.done"
    say "DONE  $name  ($(( (SECONDS-t0)/60 )) min)  $(tail -1 "$RUN/$name.log" | cut -c1-110)"
  else
    say "FAIL  $name  ($(( (SECONDS-t0)/60 )) min)  see $RUN/$name.log"
  fi
}

# ---------------------------------------------------------------- family shards for TextGrad
PYTHONPATH=$B $VC_PY - "$RUN" "$SHARDS" "$DS" <<'PY'
import sys
from pathlib import Path
from common import driver
run, shards, spec = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
keys = [f.key for f in driver.select_families(spec)]
for i in range(shards):
    (run / f"tg_shard_{i}.txt").write_text(",".join(keys[i::shards]))
print(f"{len(keys)} families -> {shards} shards")
PY

say "=== tool-free arm: GEPA + TextGrad, families=$DS, $SHARDS textgrad shards, W=$W"

# 1. learn (pro) + the shared tool-free control on the pro executor, which needs no artifact
stage 50_train_gepa $GP $B/drivers/run_gepa.py train --families $DS --workers $W &
for i in $(seq 0 $((SHARDS-1))); do
  stage 51_train_textgrad_$i $TG $B/drivers/run_textgrad.py train \
        --families "$(cat "$RUN/tg_shard_$i.txt")" &
done
stage 60_control_notool_pro $GP $B/drivers/run_gepa.py control --families $DS --split test --workers 6 &
wait
say "--- learning done"

# 2. deploy the frozen prompts on the pro executor, and the flash-lite control alongside
stage 70_deploy_gepa_pro     $GP $B/drivers/run_gepa.py     deploy --families $DS --split test --workers $W &
stage 71_deploy_textgrad_pro $TG $B/drivers/run_textgrad.py deploy --families $DS --split test --workers $W &
stage 61_control_notool_fl   $GP $B/drivers/run_gepa.py    control --families $DS --split test --model $FL --workers $W &
wait
say "--- pro deployment done"

# 3. the same prompts on the flash-lite executor (the paper's second column)
stage 80_deploy_gepa_fl     $GP $B/drivers/run_gepa.py     deploy --families $DS --split test --model $FL --workers $W &
stage 81_deploy_textgrad_fl $TG $B/drivers/run_textgrad.py deploy --families $DS --split test --model $FL --workers $W &
wait

say "=== tool-free arm finished"
