#!/bin/bash
# Acquisition with adaptive replication + the frozen deployment route for ONE family:
#     run_adaptive.sh <dataset:family> <tag> [extra loop args, e.g. --eye-dirs output/verify/$BATCH/train_flashlite]
#   rep1 -> adaptive_rule.py decide -> (rep2 -> decide -> rep3) -> select -> route.
#   A further replicate is run only if the previous one was rejected by the gate or accepted with train cover < 50 % (max 3).
#   Selection among accepted replicates: highest leave-one-out gain -> cover -> index.  Everything here reads TRAIN-side files only.
#   Result: <family>_<tag>.route.json = the frozen deployment route; run_test.sh obeys it.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . $SD/_env.sh; cd $ROOT
FAM=$1; TAG=$2; shift 2
case $FAM in *:*) ;; *) echo "run_adaptive.sh needs <dataset:family>, got '$FAM'"; exit 2 ;; esac
B=$ROOT/output/verify/$BATCH; F=${FAM##*:}; RUNS=(); mkdir -p $B
for k in 1 2 3; do
  [ -e $B/${F}_${TAG}_rep$k.done ] || bash $SD/run_one.sh $FAM ${TAG}_rep$k "$@"      # a finished replicate is not re-run (resume)
  RUNS+=($B/${F}_${TAG}_rep$k)
  D=$($PY $SD/adaptive_rule.py decide "${RUNS[@]}")
  echo "$(date -Is) $FAM ${TAG}_rep$k: $D" | tee -a $B/${F}_${TAG}.adaptive.log
  case $D in STOP*) break ;; esac
done
SEL=$($PY $SD/adaptive_rule.py select "${RUNS[@]}" --out $B/${F}_${TAG}.adaptive.json)
echo "$(date -Is) $FAM selected: $SEL" | tee -a $B/${F}_${TAG}.adaptive.log
RT=$($PY $SD/adaptive_rule.py route $FAM "${RUNS[@]}" --out $B/${F}_${TAG}.route.json)
echo "$(date -Is) $FAM route: $RT" | tee -a $B/${F}_${TAG}.adaptive.log
touch $B/${F}_${TAG}.adaptive.done
