#!/bin/bash
# Executor rollouts, resumable (up to 3 passes):  rollouts.sh <name> <pro|flashlite|model id> <input json> <workers> <rollouts per task> [skill.md]
#   -> output/verify/$BATCH/<name>/ (results.jsonl, per-rollout trajectories) and <name>_done.flag once every rollout exists.
#   LLM judge = $JUDGE (default gemini-3.1-pro-preview).  A skill file is appended to the executor's system prompt.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . $SD/_env.sh; need_key; cd $ROOT
NAME=$1; case $2 in pro) MODEL=gemini-3.1-pro-preview ;; flashlite) MODEL=gemini-3.1-flash-lite ;; *) MODEL=$2 ;; esac
IN=$3; W=${4:-8}; R=${5:-1}; SKILL=$6; B=$ROOT/output/verify/$BATCH; O=$B/$NAME; mkdir -p $B
for i in 1 2 3; do
  [ -e ${O}_done.flag ] && break
  $PY -u -m verify.run --input $IN --model $MODEL --judge-model ${JUDGE:-gemini-3.1-pro-preview} --judge llm --rollouts $R --workers $W \
    ${SKILL:+--system-prompt-extra $SKILL} --output-dir $O >> $O.log 2>&1
  N=$($PY -c "print(sum(1 for _ in open('$O/results.jsonl')))" 2>/dev/null || echo 0); T=$($PY -c "import json;print($R*len(json.load(open('$IN'))))")
  echo "$(date -Is) $NAME: $N / $T rollouts (pass $i)" >> $B/status.log
  [ "$N" = "$T" ] && echo ALL_DONE > ${O}_done.flag
done
[ -e ${O}_done.flag ]
