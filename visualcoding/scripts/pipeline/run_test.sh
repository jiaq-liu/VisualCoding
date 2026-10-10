#!/bin/bash
# TEST readings of one family's train-selected program:  run_test.sh <dataset:family> <tag>.    Measured ONCE: refuses a second run.
# Only the program's test readings are produced here (exact-match scoring).  For which executor the program is deployed is decided
# afterwards by check.py from TRAIN files only; report.py combines readings, executor rollouts and skill rollouts into the policy.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . $SD/_env.sh; cd $ROOT
FAM=$1; TAG=$2; F=${FAM##*:}; B=$ROOT/output/verify/$BATCH; RJ=$B/${F}_${TAG}.route.json; T=$SD/route_tools.py
say() { echo "$(date -Is) $FAM $*" | tee -a $B/${F}_${TAG}.test.log; }
[ -f $RJ ] || { echo "[run_test] no route file $RJ (run run_adaptive.sh first)"; exit 2; }
[ -e $B/${F}_${TAG}.test.done ] && { echo "[run_test] $FAM $TAG already tested - refusing to run test twice"; exit 1; }
ROUTE=$($PY $T get $RJ route); RUN=$($PY $T get $RJ instrument_run); say "route=$ROUTE instrument_run=${RUN:-none}"
if [ "$ROUTE" = instrument ]; then
  [ -e $RUN/test.done ] && { say "$RUN already tested - refusing to run test twice"; exit 1; }
  need_key; export CUDA_VISIBLE_DEVICES=$(freest_gpu)
  KEY=$($PY -c "import json;print(json.load(open('$RUN/frozen/meta.json'))['key'])"); IM=$($PY -c "import json;print(json.load(open('$RUN/frozen/meta.json'))['model'])")
  $PY -u -m verify.acquire.deploy --frozen $RUN/frozen -- run --out $RUN/test --split test --instruments $KEY --model $IM --workers 1 --samples 1 > $RUN/test.log 2>&1
  touch $RUN/test.done; say "program test readings done -> $RUN/test/results.jsonl"
else
  say "no deployable program: nothing to run here (the text-skill route is handled by chain.sh)"
fi
touch $B/${F}_${TAG}.test.done
