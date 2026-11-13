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

### Running the full training on the Virtual Lab (GPU) — the plan for the final results
1. Open a terminal on the Virtual Lab machine.
2. `git clone https://github.com/shreyanshsharma88/nlp-02.git && cd nlp-02`
3. Optional check (≈ 2–5 min): `bash run_lab.sh --quick`
4. Full run: `bash run_lab.sh`
   - creates `.venv`, installs `requirements-dev.txt`
   - writes `results/logs/lab_environment.txt` (date, host, `nvidia-smi`, torch/CUDA versions)
   - aborts if torch cannot see a CUDA GPU
   - runs `python run_all.py --case-aug`: 3 seeds × 2 models + the case-augmentation extra, then predictions on
     4 case variants, analysis, figures and `results/results.json`. Restart-safe: finished runs are skipped.
   - log of the run: `results/logs/lab_full_run.log`, then `pytest` → `results/logs/lab_pytest.log`
   - packs everything into `results_lab.zip`
5. **Take screenshots while it runs** (the lab desktop or URL visible, the `nvidia-smi` output, training
   progress, the final `done -> results` line). Save them as PNGs.
6. Download `results_lab.zip` (or the whole `results/` folder). The `outputs/` checkpoints are **not** needed.

### Running on a RunPod GPU pod (alternative)
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

## Project status
| Part | Status |
|---|---|
| Pipeline code (`ner/`, `run_all.py`): loading, tokenization/alignment, training, evaluation, predictions, capitalization experiments, error analysis, figures | **Done**; quick end-to-end run passes |
| Unit tests (alignment, metrics, transforms, BIO decoding) | **Done**; 16 pass |
| Notebook structure (`tools/build_notebook.py`) | **Done**; executes end to end on quick results |
| Report layout (`report/build_report.py`) | **Done**; builds a PDF |
| Contribution sheet, zip script, lab runner (`run_lab.sh`) | **Done** |
| Full training + predictions + analysis (`results/`) | **To do**: Virtual Lab run |
| Report and notebook text from the real numbers (`report/narrative.py` is a stub; notebook conclusion is a placeholder) | **To do**: after `results/` exists |
| Executed notebook + HTML/PDF export, `report/Report.pdf`, README results table, final zip | **To do**: after `results/` exists |

## Submission checklist for full marks
Do these in order. **[lab]** = on the Virtual Lab; **[local]** = in this folder after copying `results/` back.

**A. Produce the real results**
1. **[lab]** `bash run_lab.sh` finishes with `done -> results` and no errors (see the Virtual Lab section above).
2. **[lab]** Screenshots taken: lab environment visible, `nvidia-smi`, training in progress, the final output and
   the `pytest` summary. Put them in `screenshots/` (e.g. `01_lab_env.png`, `02_nvidia_smi.png`, `03_training.png`,
   `04_done.png`, `05_pytest.png`). A screen recording is also fine if the course asks for video.
3. **[local]** Copy `results/` back into this folder (replacing the placeholder one), then create
   `screenshots/VIRTUAL_LAB.txt` with one line saying where and when it ran. The report claims lab execution only
   when this file exists.

**B. Rebuild every deliverable from `results/`**
4. **[local]** Sanity-check: `bert-base-cased` test entity F1 should be ≈ 0.91–0.92 (published range). If it is far
   off, stop and investigate before writing anything.
5. **[local]** Write the report and notebook text from the real numbers (`report/narrative.py`, notebook
   conclusion). Every number must come from `results.json`; no hand-typed numbers.
6. **[local]** `python tools/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace NER_Cased_vs_Uncased.ipynb`,
   then export `jupyter nbconvert --to html NER_Cased_vs_Uncased.ipynb` (and PDF if required).
7. **[local]** `python report/build_report.py` → `report/Report.pdf`. Check every page: title block with group and
   members on page 1, no cut-off tables, readable figures, 8–12 pages.
8. **[local]** `python tools/update_readme.py` (fills the results table in this README).
9. **[local]** `pytest`: all tests pass, including the `results.json` sanity and compliance tests.
10. **[local]** Cross-check: the 4 metrics × 2 models are identical in `results.json`, the notebook and the report.

**C. Package and submit**
11. **[local]** `bash make_zip.sh` → `dist/G101_NLP_Assignment2.zip`. Unzip it into a temp folder and confirm it holds
    the code, the executed notebook (+HTML/PDF), `results/`, `report/Report.pdf`, README, requirements,
    `G101_Contribution.xlsx` and `screenshots/`, and **no** `outputs/` checkpoints or `.venv`.
12. **[local]** `git status` (nothing large or private), then commit and push.
13. Upload the zip to the portal **before the deadline (16 November)**.

**What graders look for:** dataset loading with `datasets` · preprocessing and tokenization with label alignment ·
model loading with `transformers` · fine-tuning · Accuracy, Precision, Recall and F1 for both models · predictions on
≥ 5 custom sentences (we have 18) · error analysis · a clear cased-vs-uncased comparison · an evidence-based
explanation of how capitalization affects names (PER) and organizations (ORG). Each has its own notebook heading and
report section.

## BEFORE SUBMISSION
- **CHECK THE DEADLINE AND THE EXACT SUBMISSION NAMING ON THE PORTAL.**
- **CONFIRM THE CONTRIBUTION PERCENTAGES AND HEADER WORDING IN `G101_Contribution.xlsx` (DEFAULT: 100 EACH).**
- **RUN THE FULL TRAINING ON THE VIRTUAL LAB, TAKE SCREENSHOTS, PUT THEM IN `screenshots/` AND CREATE `screenshots/VIRTUAL_LAB.txt`.**
- **REBUILD THE NOTEBOOK, THE REPORT AND THE ZIP FROM THE NEW `results/` (CHECKLIST STEPS 4–11).**
- **OPEN THE NOTEBOOK AND THE REPORT ONCE AND SKIM THEM.**
- **UPLOAD `dist/G101_NLP_Assignment2.zip` TO THE PORTAL BEFORE THE DEADLINE (16 NOVEMBER).**
