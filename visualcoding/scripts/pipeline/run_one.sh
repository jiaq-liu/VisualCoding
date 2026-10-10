#!/bin/bash
# One acquisition run (frozen recipe):  run_one.sh <dataset:family> <run tag> [extra loop args...]    -> output/verify/$BATCH/<family>_<run tag>/
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . $SD/_env.sh; need_key; cd $ROOT
FAM=$1; RTAG=$2; shift 2
OUT=$ROOT/output/verify/$BATCH/${FAM##*:}_$RTAG; mkdir -p $(dirname $OUT)
export CUDA_VISIBLE_DEVICES=$(freest_gpu)
$PY -u -m verify.acquire.loop --family $FAM --model ${WRITER:-gemini-3.1-pro-preview} --mode code --out $OUT \
  --timeout-task 60 --eye auto --profile auto "$@" > $OUT.log 2>&1
touch $OUT.done
