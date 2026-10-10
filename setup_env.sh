#!/bin/bash
# Create the main Python environment in <repo>/.venv (python 3.10).
#   bash setup_env.sh              # minimal set (environment/requirements.txt): everything except the detector primitives
#   bash setup_env.sh --full       # the exact frozen environment the experiments ran in (torch 2.13 cu126, transformers 4.55, ...)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); cd "$ROOT"
PYBIN=${PYBIN:-python3.10}
[ -x .venv/bin/python ] || "$PYBIN" -m venv .venv
.venv/bin/python -m pip install --upgrade "pip<24"
if [ "${1:-}" = "--full" ]; then
  .venv/bin/python -m pip install -r environment/requirements.freeze.txt --extra-index-url https://download.pytorch.org/whl/cu126
else
  .venv/bin/python -m pip install -r environment/requirements.txt
fi
# Detector weights (only for lib.detect / lib.count_objects, i.e. the refcoco program) are downloaded on first use into $HF_HOME:
#   google/owlv2-base-patch16-ensemble , IDEA-Research/grounding-dino-base   (override with VC_OWL / VC_GDINO)
.venv/bin/python - <<'PY'
import numpy, cv2, PIL, scipy, skimage, google.genai, ipykernel, jupyter_client
print("ok: numpy", numpy.__version__, "| cv2", cv2.__version__, "| google-genai", google.genai.__version__)
try:
    import torch, transformers; print("ok: torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
except Exception as e:
    print("torch not installed (minimal set): detector primitives unavailable")
PY
[ -f .env ] || echo "[setup_env] create $ROOT/.env from .env.example (GEMINI_API_KEY=...)"
