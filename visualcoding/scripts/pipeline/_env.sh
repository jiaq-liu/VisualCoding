# Sourced by every pipeline script.  Defines SD-independent roots and exports the API key without ever printing it.
#   ROOT  = visualcoding/ package root (VC_ROOT overrides)      REPO = repository root
#   PY    = python interpreter of the main environment (PY overrides; default: python on PATH)
#   key   = GEMINI_API_KEY from the environment, else from $VC_ENV_FILE, else from $REPO/.env   (line: GEMINI_API_KEY=...)
_PD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${VC_ROOT:-$(cd "$_PD/../.." && pwd)}; REPO=$(cd "$ROOT/.." && pwd); export VC_ROOT=$ROOT
export PY=${PY:-$( [ -x $REPO/.venv/bin/python ] && echo $REPO/.venv/bin/python || echo python )}; export BATCH=${BATCH:-run1}; export TAG=${TAG:-A}
ENVF=${VC_ENV_FILE:-$REPO/.env}
if [ -z "$GEMINI_API_KEY" ] && [ -f "$ENVF" ]; then
  export GEMINI_API_KEY=$(cd $ROOT && $PY -c "
from pathlib import Path
from verify.env import _read_env_file
k=_read_env_file(Path('$ENVF'));print(k.get('gemini_api_key') or k.get('GEMINI_API_KEY') or '')")
fi
need_key() { [ -n "$GEMINI_API_KEY" ] || { echo "no Gemini key: export GEMINI_API_KEY or put GEMINI_API_KEY=... into $ENVF"; exit 1; }; }
freest_gpu() { nvidia-smi 2>/dev/null --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $3-$2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
