#!/bin/bash
# Fetch the five upstream baseline repositories at the pinned commits, apply our patches, and create one virtualenv per method.
#     bash baselines/setup_baselines.sh            # everything
#     bash baselines/setup_baselines.sh xskill     # one method (gepa | textgrad | xskill | trace2skill | skillopt)
# GEPA needs no venv of its own: drivers/run_gepa.py runs in the MAIN environment (<repo>/.venv) with gepa/src on PYTHONPATH.
set -e
B=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); cd $B
PYBIN=${PYBIN:-python3.10}
declare -A URL=( [gepa]=https://github.com/gepa-ai/gepa.git [textgrad]=https://github.com/zou-group/textgrad.git [xskill]=https://github.com/XSkill-Agent/XSkill.git
                 [trace2skill]=https://github.com/Qwen-Applications/Trace2Skill.git [skillopt]=https://github.com/microsoft/SkillOpt.git )
declare -A SHA=( [gepa]=15ee314f9c7d34ec153b809d401f42f55c4dcd76 [textgrad]=75e912e210864b61999781778cdf756d4468120f [xskill]=b65192501767f69fac7acc53321ac16e57fca86a
                 [trace2skill]=3d0b52a140f002a512930252b613c49048f7d5ac [skillopt]=79124b37e9a6371e13b753f8bcd7adb1e493ade1 )
declare -A DIR=( [gepa]=gepa [textgrad]=textgrad [xskill]=XSkill [trace2skill]=Trace2Skill [skillopt]=SkillOpt )
for m in ${@:-gepa textgrad xskill trace2skill skillopt}; do
  d=${DIR[$m]}
  if [ ! -d $d/.git ]; then git clone -q ${URL[$m]} $d; fi
  ( cd $d && git fetch -q origin ${SHA[$m]} 2>/dev/null || true; git checkout -q ${SHA[$m]} )
  if [ -f patches/$d.patch ] && ! ( cd $d && git diff --quiet HEAD -- . 2>/dev/null && [ -z "$(git status --porcelain --untracked-files=all | grep -v __pycache__)" ] ); then
    echo "[$m] $d already has local changes - patch not re-applied"
  elif [ -f patches/$d.patch ]; then
    ( cd $d && git apply --whitespace=nowarn ../patches/$d.patch ) && echo "[$m] patch applied"
  fi
  [ $m = gepa ] && { echo "[gepa] pinned at ${SHA[$m]}; runs in the main environment"; continue; }
  if [ ! -x .venvs/$m/bin/python ]; then
    mkdir -p .venvs; $PYBIN -m venv .venvs/$m; .venvs/$m/bin/pip install -q --upgrade pip
    .venvs/$m/bin/pip install -q -r requirements-$m.freeze.txt || { echo "[$m] exact freeze failed; installing the minimal set"; .venvs/$m/bin/pip install -q google-genai numpy pyyaml requests openai; }
  fi
  echo "[$m] ready: .venvs/$m/bin/python"
done
echo "run the offline tests: bash baselines/tests/run_all.sh"
