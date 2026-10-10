#!/bin/bash
# Full method on a set of families of the frozen split, server-side chain (start it with nohup setsid):
#     BATCH=run1 bash scripts/pipeline/chain.sh <all | dataset | dataset:family,...>
#   1 reference executor TRAIN rollouts (x4) = the acquisition reference ("the executor solved this task k/4 times")
#   2 in the background: TEST rollouts (x1) of every executor = the baselines, TRAIN rollouts (x4) of the other executors (deploy check, skills)
#   3 acquisition batch: writer $WRITER, adaptive replicates (max 3), deploy rule = accepted AND train cover >= 50 %  -> <family>_<TAG>.route.json
#   4 program TEST readings, once per family (run_test.sh refuses a second run)
#   5 check.py (per-executor deploy check, train only) -> text skills summarised from the $SKILL_FROM executor's TRAIN trajectories for every
#     (family, executor) without a deployed program -> single-rollout skill tests for exactly those pairs
#   6 report.py -> output/verify/$BATCH/report.md
# Environment: BATCH TAG EXECUTORS REF SKILL_FROM WRITER JUDGE (see config.py), PAR (families in parallel, 8), W_TRAIN / W_BG (rollout workers).
# Everything is resumable: finished rollouts, replicates and skills are skipped; TEST is never measured twice.  Never edit this file while it runs.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . $SD/_env.sh; need_key; cd $ROOT
B=$ROOT/output/verify/$BATCH; mkdir -p $B; S=$B/status.log; I=$B/inputs
say() { echo "$(date -Is) chain: $*" >> $S; }
say "start ($*)"; $PY $SD/make_inputs.py "${1:-all}" > $B/make_inputs.log 2>&1 || { say "make_inputs failed"; exit 1; }
read -r EXECUTORS REF SKILL_FROM <<< "$($PY -c "import sys;sys.path.insert(0,'$SD');import config as C;print(' '.join(C.EXECUTORS).replace(' ',','), C.REF, C.SKILL_FROM)")"
EXECUTORS=${EXECUTORS//,/ }; STORE=$B/text_skills
bash $SD/rollouts.sh train_$REF $REF $I/train.json ${W_TRAIN:-12} 4 || say "WARNING reference train rollouts incomplete after 3 passes; continuing with the rollouts that exist"
(
  for X in $EXECUTORS; do
    bash $SD/rollouts.sh test_$X $X $I/test.json ${W_BG:-8} 1 &
    [ $X = $REF ] || bash $SD/rollouts.sh train_$X $X $I/train.json ${W_BG:-8} 4 &
  done; wait
  echo "$(date -Is) chain: background rollout jobs ended ($(ls $B/*_done.flag 2>/dev/null | wc -l) done flags)" >> $S
) &
BGPID=$!
say "acquisition batch start (${PAR:-8} families in parallel, adaptive replicates, writer ${WRITER:-gemini-3.1-pro-preview}, train reference $REF)"
bash $SD/run_batch.sh $B/jobs.txt ${PAR:-8} > $B/batch.log 2>&1
say "acquisition batch done: $(cat $B/*_${TAG}.route.json 2>/dev/null | $PY -c "
import sys, re, collections
print(dict(collections.Counter(re.findall(r'\"route\": \"(\w+)\"', sys.stdin.read()))))")"
say "program test start (once per family)"
awk '{print $1, $2}' $B/jobs.txt | xargs -P ${TEST_PAR:-4} -L 1 bash $SD/run_test.sh >> $B/test_all.log 2>&1
say "program test done"
wait $BGPID
for X in $EXECUTORS; do for f in test_$X $([ $X = $REF ] || echo train_$X); do [ -e $B/${f}_done.flag ] || { say "$f incomplete - stopping before the check"; exit 1; }; done; done
$PY $SD/check.py > $B/check.log 2>&1 && say "deploy check done: $(grep -c -- '-> skill' $B/check.log) program families with a blocked executor" || { say "check FAILED (see check.log)"; exit 1; }
cut -d' ' -f1,2 $B/skill_jobs.txt | sort -u > $B/skill_build.txt
say "skill build from $SKILL_FROM train trajectories: $(wc -l < $B/skill_build.txt) families"
SKM=$($PY -c "import sys;sys.path.insert(0,'$SD');import config as C;print(C.ALL_MODELS['$SKILL_FROM'])")
cat $B/skill_build.txt | xargs -P 6 -L 1 sh -c '[ -f '"$STORE"'/$0__$1/skill.md ] || '"$PY"' -u -m verify.textskill.build --family $0:$1 --store '"$STORE"' --executor '"$SKM"' --train-dirs '"$B/train_$SKILL_FROM"' > '"$B"'/skill_$1.log 2>&1'
say "skills built: $(ls $STORE/*/skill.md 2>/dev/null | wc -l) of $(wc -l < $B/skill_build.txt)"
while read DSN F X; do
  SK=$STORE/${DSN}__$F/skill.md; [ -f $SK ] || { say "skill missing for $F"; continue; }
  echo "skilltest_${F}_${X} $X $I/skill_${F}_test.json 5 1 $SK"
done < $B/skill_jobs.txt > $B/skilltest_jobs.txt
xargs -P ${SKILL_PAR:-10} -L 1 bash $SD/rollouts.sh < $B/skilltest_jobs.txt > /dev/null 2>&1
say "skill tests ended: $(ls $B/skilltest_*_done.flag 2>/dev/null | wc -l) of $(wc -l < $B/skilltest_jobs.txt)"
$PY $SD/report.py > $B/report.log 2>&1 && say "report written -> $B/report.md" || say "report failed (see report.log)"
touch $B/chain.done
