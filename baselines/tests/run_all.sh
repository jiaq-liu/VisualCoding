#!/bin/bash
# Offline mechanism tests for the five baseline drivers (no API calls, no rollouts).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
VC_PY=${VC_PY:-$(cd ..; pwd)/.venv/bin/python}     # interpreter of the main environment
export GEMINI_API_KEY=${GEMINI_API_KEY:-offline-test-no-calls}   # the drivers resolve a key at start-up; the stubs never use it
fail=0
for t in gepa textgrad xskill trace2skill skillopt skillopt_singlepass; do
  case $t in gepa) PY=$VC_PY ;; skillopt*) PY=.venvs/skillopt/bin/python ;;
             *) PY=.venvs/$t/bin/python ;; esac
  if timeout 300 "$PY" "tests/test_${t}_offline.py" >"tests/_${t}.log" 2>&1; then
    echo "PASS $t"
  else
    echo "FAIL $t (see tests/_${t}.log)"; fail=1
  fi
done
exit $fail
