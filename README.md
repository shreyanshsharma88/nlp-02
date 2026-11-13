# NLP - Assignment-2, Set-11 | Group 101

**Token Classification (NER) on CoNLL-2003 — `bert-base-cased` vs `bert-base-uncased`, and the effect of
capitalization on names and organizations.**

| Name | Registration number | Email |
|---|---|---|
| SHREYANSH SHARMA | 2025AG05772 | 2025ag05772@wilp.bits-pilani.ac.in |
| VINAY PRATAP SINGH BHADAURIA | 2025AG05689 | 2025ag05689@wilp.bits-pilani.ac.in |
| LAKSHYA KANCHAN | 2025AG05062 | 2025ag05062@wilp.bits-pilani.ac.in |
| VINAYA RAGHAVA R | 2025AG05187 | 2025ag05187@wilp.bits-pilani.ac.in |

Repository: https://github.com/shreyanshsharma88/nlp-02

## Project summary
Both models are fine-tuned on CoNLL-2003 with **identical** hyper-parameters, data, seeds (13, 42, 1234) and
evaluation code, using Hugging Face `datasets`, `transformers` (`Trainer`) and `seqeval`. We report token-level
Accuracy and entity-level Precision / Recall / F1 (overall, per type, macro), paired-bootstrap confidence intervals,
and then explain the role of capitalization with: case-perturbed test sets (lower / UPPER / Title), slices by
capitalization pattern, case-ambiguous words (may/May, us/US …), a PER-vs-ORG breakdown, 18 custom minimal-pair
sentences and a full error analysis.

## Results (test set, seed 42; generated from `results/results.json`)
<!-- RESULTS_TABLE_START -->
_run `python tools/update_readme.py` after `run_all.py`_
<!-- RESULTS_TABLE_END -->

## Where everything is
| Deliverable | Path |
|---|---|
| Main notebook (executed, with outputs) | `NER_Cased_vs_Uncased.ipynb` (+ `NER_Cased_vs_Uncased.html`, `.pdf`) |
| Report | `report/Report.pdf` (built by `report/build_report.py` from `results/`) |
| All numbers | `results/results.json` |
| Predictions | `results/preds/*.json` (word level), `results/test_predictions_word_level.csv`, `results/custom_predictions.csv` |
| Error analysis tables | `results/disagreements_*.csv`, `results/ambiguous_words.csv` |
| Figures | `results/figures/*.png` |
| Training logs | `results/logs/*.json` (loss, per-epoch validation metrics, time, device) |
| Contribution sheet | `G101_Contribution.xlsx` |
| Plan, audit, status | `PLAN.md` |
| Code | `ner/` (data, tokenize, train, evaluate, predict, analysis), `run_all.py`, `tests/` |

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Python 3.10–3.12
pip install -r requirements-dev.txt                       # requirements.txt = runtime only
```
The dataset is loaded from the Hugging Face Hub (`eriktks/conll2003`, parquet conversion — see PLAN.md for why).
No HF token is needed.

## How to run
```bash
python run_all.py                  # full: 3 seeds x 2 models -> predictions -> analysis -> results/ (resumable)
python run_all.py --seeds 42       # one seed only
python run_all.py --quick          # smoke run: 500 sentences, 1 epoch -> results_quick/ (add --cpu to force CPU)
python run_all.py --stage analyze  # rebuild results.json + figures from saved predictions (no GPU)
jupyter nbconvert --to notebook --execute --inplace NER_Cased_vs_Uncased.ipynb   # rebuild notebook from results/
python report/build_report.py      # -> report/Report.pdf
bash make_zip.sh                   # -> dist/G101_NLP_Assignment2.zip
```
In the notebook, `TRAIN = False` (default) reads `results/`; `TRAIN = True` fine-tunes both models inside it.
Fine-tuned checkpoints go to `outputs/` (≈400 MB each, git-ignored).

### Running on a RunPod GPU pod (optional)
1. Start a PyTorch pod (CUDA ≥ 12, ≥ 16 GB VRAM) and open JupyterLab → Terminal.
2. `git clone https://github.com/shreyanshsharma88/nlp-02.git && cd nlp-02` (or upload the zip and `unzip` it).
3. `pip install -r requirements-dev.txt` (the pod's CUDA torch is replaced by the pinned version; skip `torch` from the file if you prefer the pod's build).
4. `python run_all.py` — fp16 switches on automatically on CUDA.
5. Download `results/` (JupyterLab: right-click → Download, or `zip -r results.zip results`) and copy it into the local
   project; then rebuild the notebook, report and zip locally with the commands above.

### Expected runtime
| Hardware | One model, 3 epochs | Full `run_all.py` (6 runs + analysis) |
|---|---|---|
| Apple M4 (MPS), measured | ≈ 17–20 min | ≈ 2 h |
| CUDA GPU (e.g. A4000/A5000, fp16) | ≈ 3–5 min (estimate) | ≈ 30–40 min (estimate) |
| CPU only | hours — use `--quick` (≈ 5–10 min) | not recommended |

## Tests
```bash
pytest -m "not slow"   # unit + compliance tests (seconds)
pytest                 # + end-to-end smoke test (run_all.py --quick --cpu, several minutes)
```

## BEFORE SUBMISSION
**1. CHECK THE DEADLINE AND THE EXACT SUBMISSION NAMING ON THE PORTAL (THE BRIEF LEFT THE DATE/TIME AS A PLACEHOLDER).**
**2. CONFIRM THE CONTRIBUTION PERCENTAGES AND HEADER WORDING IN `G101_Contribution.xlsx` (DEFAULT: 100 EACH).**
**3. IF THE COURSE REQUIRES VIRTUAL-LAB / VM EXECUTION EVIDENCE: RUN THERE, PUT SCREENSHOTS/VIDEO IN `screenshots/`, CREATE `screenshots/VIRTUAL_LAB.txt`, THEN REBUILD THE REPORT (`python report/build_report.py`).**
**4. OPEN THE NOTEBOOK AND THE REPORT ONCE AND SKIM THEM.**
**5. REBUILD THE ZIP: `bash make_zip.sh` → `dist/G101_NLP_Assignment2.zip`.**
**6. UPLOAD THE ZIP TO THE PORTAL BEFORE THE DEADLINE (16 NOVEMBER).**
