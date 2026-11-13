#!/usr/bin/env bash
# Full training run on the Virtual Lab GPU machine. Records environment evidence, runs the whole pipeline,
# runs the tests and packs results/ for copying back. See README "Part 1".
#
#   bash run_lab.sh              # full run: 3 seeds x 2 models + case-augmentation extra -> results/
#   bash run_lab.sh --quick      # smoke run first (2-5 min), to check the setup (-> results_quick/)
#   KEEP_TORCH=1 bash run_lab.sh # use the lab's pre-installed CUDA torch instead of the pinned one
set -euo pipefail
cd "$(dirname "$0")"
banner() { echo; echo "=================================================================="; echo "  $*"; echo "=================================================================="; }

banner "STEP 1/5  Python environment"
if [ ! -d .venv ]; then
  if [ "${KEEP_TORCH:-0}" = "1" ]; then python3 -m venv --system-site-packages .venv; else python3 -m venv .venv; fi
fi
source .venv/bin/activate
pip install -q --upgrade pip
if [ "${KEEP_TORCH:-0}" = "1" ]; then
  grep -v '^torch==' requirements.txt > .req_no_torch.txt
  pip install -q -r .req_no_torch.txt
  grep -v -e '^-r ' requirements-dev.txt | pip install -q -r /dev/stdin
  rm -f .req_no_torch.txt
else
  pip install -q -r requirements-dev.txt
fi

banner "STEP 2/5  Environment and GPU evidence -> results/logs/lab_environment.txt"
mkdir -p results/logs
ENV_FILE=results/logs/lab_environment.txt
{
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "user: $(whoami)"
  uname -a
  python --version
  nvidia-smi || echo "nvidia-smi not available"
  python -c "import torch, transformers, datasets; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'); print('transformers', transformers.__version__, '| datasets', datasets.__version__)"
} 2>&1 | tee "$ENV_FILE"

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: CUDA GPU not visible to torch. Try: rm -rf .venv && KEEP_TORCH=1 bash run_lab.sh  (see README troubleshooting)"; exit 1; }

if [ "${1:-}" = "--quick" ]; then
  banner "QUICK SMOKE RUN (500 sentences, 1 epoch) -> results_quick/"
  python run_all.py --quick 2>&1 | tee results/logs/lab_quick_run.log
  banner "QUICK RUN FINISHED - now run: bash run_lab.sh"
  exit 0
fi

banner "STEP 3/5  Full training + predictions + analysis (python run_all.py --case-aug)"
python run_all.py --case-aug 2>&1 | tee results/logs/lab_full_run.log

banner "STEP 4/5  Tests (pytest)"
pytest -q -m "not slow" 2>&1 | tee results/logs/lab_pytest.log || true

banner "STEP 5/5  Headline results + packing results_lab.zip"
python - <<'EOF' 2>&1 | tee results/logs/lab_headline.txt
import json
R = json.load(open("results/results.json"))
print(f"{'model':20s} {'Accuracy':>9s} {'Precision':>9s} {'Recall':>9s} {'F1':>9s} {'F1 lower-case input':>20s}")
for m in ["bert-base-cased", "bert-base-uncased"]:
    t = R["test"][m]
    print(f"{m:20s} {t['accuracy']:9.4f} {t['precision']:9.4f} {t['recall']:9.4f} {t['f1']:9.4f} "
          f"{R['case_variants'][m]['lower']['f1']:20.4f}")
print("training device:", R["training"]["bert-base-cased"]["device"])
EOF
zip -qr results_lab.zip results
banner "ALL DONE. Copy results_lab.zip back to the local project (README Part 2)."
