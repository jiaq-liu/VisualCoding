#!/bin/bash
# Tool arm: XSkill + Trace2Skill + SkillOpt, learned from our executor's rollouts and deployed on both executors.
#     DS=all bash baselines/scripts/run_tool_arm.sh          (DS = all | dataset | comma list of dataset:family; default: all four datasets)
# Independent work runs together; only real dependencies are serialized:
#   * the three trainings read the same (read-only) rollout pool and write to their own dirs
#   * a deployment needs only its own artifact
#   * pro and flash-lite deployments of one method write to different dirs (driver.model_tag)
# Every stage leaves a .done flag, so finished stages are skipped and the chain is resumable.  Executor rollouts that already exist under
# visualcoding/output/verify (same task, model, loop configuration and system prompt) are reused, so run the main method's pipeline first
# if you want the seed pass and the control shared with it.
set -u
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VC_PY=${VC_PY:-$B/../.venv/bin/python}
DS=${DS:-all}
FL=gemini-3.1-flash-lite
W=${W:-6}                  # rollout workers per job; 3 jobs => 18 concurrent rollouts
A=${A:-8}                  # analysis threads per job => 24 concurrent text calls
RUN=${BASELINE_OUT:-$B/../output/baselines}/_tool_arm
mkdir -p "$RUN"
S=$RUN/status.log

say() { echo "$(date -Iseconds) $*" | tee -a "$S"; }

stage() {  # stage <name> <cmd...>   (run in the foreground of whatever shell calls it)
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

XS=$B/.venvs/xskill/bin/python
T2=$B/.venvs/trace2skill/bin/python
SO=$B/.venvs/skillopt/bin/python

say "=== tool arm: 3 methods, families=$DS, W=$W A=$A"

# 1. train -- three independent jobs
stage 10_train_xskill      $XS $B/drivers/run_xskill.py      train --families $DS --analysts $A --workers $W &
stage 11_train_trace2skill $T2 $B/drivers/run_trace2skill.py train --families $DS --analysts $A --workers $W &
stage 12_train_skillopt    $SO $B/drivers/run_skillopt.py    train --families $DS --workers $W &
wait
say "--- training done"

# 2. the shared tool-arm control on test, both executors (pure symlinks, seconds)
stage 20_control_pro       $SO $B/drivers/run_skillopt.py control --families $DS --split test --workers $W
stage 21_control_flashlite $SO $B/drivers/run_skillopt.py control --families $DS --split test --model $FL --workers $W

# 3. deploy on the pro executor -- three independent jobs
stage 30_deploy_xskill_pro      $XS $B/drivers/run_xskill.py      deploy --families $DS --split test --workers $W &
stage 31_deploy_trace2skill_pro $T2 $B/drivers/run_trace2skill.py deploy --families $DS --split test --workers $W &
stage 32_deploy_skillopt_pro    $SO $B/drivers/run_skillopt.py    deploy --families $DS --split test --workers $W &
wait
say "--- pro deployment done"

# 4. the same artifacts on the flash-lite executor (the paper's second column)
stage 40_deploy_xskill_fl      $XS $B/drivers/run_xskill.py      deploy --families $DS --split test --model $FL --workers $W &
stage 41_deploy_trace2skill_fl $T2 $B/drivers/run_trace2skill.py deploy --families $DS --split test --model $FL --workers $W &
stage 42_deploy_skillopt_fl    $SO $B/drivers/run_skillopt.py    deploy --families $DS --split test --model $FL --workers $W &
wait

say "=== tool arm finished"
