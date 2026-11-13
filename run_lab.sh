#!/usr/bin/env bash
# Full training run on the Virtual Lab GPU machine. Records environment evidence, runs the whole pipeline,
# and packs results/ for copying back.
#
#   bash run_lab.sh            # full run: 3 seeds x 2 models + case-augmentation extra -> results/
#   bash run_lab.sh --quick    # 2-minute smoke run first, to check the setup (-> results_quick/)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements-dev.txt

mkdir -p results/logs
ENV_FILE=results/logs/lab_environment.txt
{
  echo "date: $(date)"
  echo "host: $(hostname)"
  uname -a
  python --version
  nvidia-smi || echo "nvidia-smi not available"
  python -c "import torch, transformers, datasets; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'); print('transformers', transformers.__version__, '| datasets', datasets.__version__)"
} | tee "$ENV_FILE"

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: CUDA GPU not visible to torch - full training would be very slow. Aborting."; exit 1; }

if [ "${1:-}" = "--quick" ]; then
  python run_all.py --quick 2>&1 | tee results/logs/lab_quick_run.log
  exit 0
fi

python run_all.py --case-aug 2>&1 | tee results/logs/lab_full_run.log
pytest -q 2>&1 | tee results/logs/lab_pytest.log || true
zip -qr results_lab.zip results
echo "DONE. Copy results_lab.zip (or the results/ folder) back to the local project."
