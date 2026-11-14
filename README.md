# NLP - Assignment-2, Set-11 | Group 101

**Token Classification (NER) on CoNLL-2003 — `bert-base-cased` vs `bert-base-uncased`, and the effect of
capitalization on names and organizations.**

| Name | Registration number | Email |
|---|---|---|
| SHREYANSH SHARMA | 2025AG05772 | 2025ag05772@wilp.bits-pilani.ac.in |
| VINAY PRATAP SINGH BHADAURIA | 2025AG05689 | 2025ag05689@wilp.bits-pilani.ac.in |
| LAKSHYA KANCHAN | 2025AG05062 | 2025ag05062@wilp.bits-pilani.ac.in |
| VINAYA RAGHAVA R | 2025AG05187 | 2025ag05187@wilp.bits-pilani.ac.in |

Repository: https://github.com/shreyanshsharma88/nlp-02 · Deadline: **16 November** (confirm the exact time on the portal)

---

## Contents
> **Working only on the Virtual Lab with no GPU? Start at [Virtual Lab only: complete workflow (CPU, no GPU)](#virtual-lab-only-complete-workflow-cpu-no-gpu).**

1. [Project summary](#project-summary)
2. [Project status](#project-status)
3. [Step-by-step: from here to submission](#step-by-step-from-here-to-submission)
   - [Part 1: Full training on the Virtual Lab](#part-1-full-training-on-the-virtual-lab-gpu)
   - [Part 2: Bring the results back](#part-2-bring-the-results-back-to-the-local-project)
   - [Part 3: Rebuild every deliverable](#part-3-rebuild-every-deliverable-locally)
   - [Part 4: Package, verify, submit](#part-4-package-verify-and-submit)
4. [Screenshots to take](#screenshots-to-take-execution-evidence)
5. [Final full-marks checklist](#final-full-marks-checklist)
6. [Troubleshooting](#troubleshooting)
7. [Results](#results-test-set-seed-42-generated-from-resultsresultsjson)
8. [Where everything is](#where-everything-is)
9. [Reference: setup, commands, tests, runtimes](#reference-setup-commands-tests-runtimes)

---

## Project summary
Both models are fine-tuned on CoNLL-2003 with **identical** hyper-parameters, data, seeds (13, 42, 1234) and
evaluation code, using Hugging Face `datasets`, `transformers` (`Trainer`) and `seqeval`.

**Metrics:**
- Accuracy is token-level.
- Precision, Recall and F1 are entity-level: overall, per type (PER, ORG, LOC, MISC) and macro.
- Paired-bootstrap confidence intervals compare the two models.

**Explaining capitalization:**
- case-perturbed test sets (lower / UPPER / Title)
- slices by capitalization pattern
- case-ambiguous words (may/May, us/US …)
- a PER-vs-ORG breakdown
- 18 custom minimal-pair sentences
- a full error analysis
- an extra experiment: the cased model trained with case augmentation

| Assignment requirement | Where it is |
|---|---|
| 1. Dataset loading with `datasets` | `ner/data.py` · notebook §1 · report §2 |
| 2. Preprocessing and tokenization | `ner/tokenize.py` · notebook §2 · report §3–4 |
| 3. Model loading with `transformers` | `ner/train.py: build_model` · notebook §3 · report §3 |
| 4. Fine-tuning pipeline | `ner/train.py: train_model`, `run_all.py` · notebook §4 · report §4 |
| 5. Evaluation (Accuracy, Precision, Recall, F1) | `ner/evaluate.py` · notebook §5 · report §5 |
| 6. Predictions on ≥ 5 custom examples (we have 18) | `ner/predict.py`, `ner/analysis.py` · notebook §6 · report §7 |
| 7. Error analysis | `ner/analysis.py` · notebook §5b, §7 · report §6, §8 |

---

## Project status
| Part | Status |
|---|---|
| Pipeline code (`ner/`, `run_all.py`): loading, tokenization and label alignment, training, evaluation, predictions, capitalization experiments, error analysis, figures | ✅ Done; quick end-to-end run passes |
| Unit tests (alignment, metrics, case transforms, BIO decoding) | ✅ Done; 16 pass |
| Notebook structure (`tools/build_notebook.py`) | ✅ Done; executes end to end |
| Report layout (`report/build_report.py`) | ✅ Done; builds a PDF |
| Contribution sheet, zip script, Virtual Lab runner (`run_lab.sh`) | ✅ Done |
| **Full training + predictions + analysis → `results/`** | ⏳ **To do: Part 1 (Virtual Lab)** |
| **Report / notebook text from the real numbers** (`report/narrative.py` is a stub, notebook conclusion is a placeholder) | ⏳ **To do: Part 3** |
| **Executed notebook + HTML export, `report/Report.pdf`, README results table, final zip** | ⏳ **To do: Parts 3–4** |

---

## Virtual Lab only: complete workflow (CPU, no GPU)
**Use this section if you do everything on the Virtual Lab and the lab machine has no GPU** (`nvidia-smi` is not
available). It replaces Parts 1–4 below; those describe the GPU lab + local Mac route.

### Two kinds of terminal
Every command below says which terminal to use.

| Name | How to open it | Prompt looks like |
|---|---|---|
| **Normal terminal** | New terminal (VS Code: Terminal → New Terminal), then run `conda deactivate` until `(base)` disappears. Do **not** activate `.venv`. | `[user@ip-... nlp-02]$` |
| **venv terminal** | Normal terminal, then `cd ~/Desktop/nlp-02 && source .venv/bin/activate` | `(.venv) [user@ip-... nlp-02]$` (no `(base)`) |

`run_lab.sh` creates and activates `.venv` itself, so you start it from a **normal terminal**. Any `python`,
`pytest` or `jupyter` command you type yourself needs the **venv terminal**. If you see both `(.venv)` and `(base)`,
run `conda deactivate` (repeat until `(base)` is gone), then `source .venv/bin/activate` again: conda packages
mixing into the venv cause the `h11` / `httpcore` pip conflict warnings.

### L1. Get the code (normal terminal)
```bash
conda deactivate                      # repeat until "(base)" is gone
cd ~/Desktop
git clone https://github.com/shreyanshsharma88/nlp-02.git    # or download the ZIP from GitHub and unzip
cd nlp-02
rm -rf .venv                          # only if an old/broken .venv exists
git config user.name  "<your name>"   # local to this repo only
git config user.email "<your email>"
```
If you already have an old `nlp-02-main` folder from an earlier ZIP, don't use it; it lacks the CPU support.
📸 **S1**: this terminal after the clone, with the lab desktop/portal visible.

### L2. Quick smoke run, CPU (normal terminal)
```bash
cd ~/Desktop/nlp-02
nproc                                  # number of CPU cores; note it
time ALLOW_CPU=1 SEEDS="42" bash run_lab.sh --quick
```
- First run installs packages (5–10 min). pip may print `dependency conflicts ... h11` warnings; ignore them.
- Must end with `QUICK RUN FINISHED`. Note the `real` time printed by `time`.
- If it stops with a `Traceback`, copy the last 30 lines and send them to the team.

📸 **S2**: `cat results/logs/lab_environment.txt` (machine, `nvidia-smi not available`, torch, `cuda available: False`).
📸 **S3**: the `WARNING: no CUDA GPU - running on CPU` line and the quick run's final lines.

### L3. Full run, CPU (normal terminal, inside tmux)
```bash
cd ~/Desktop/nlp-02
tmux new -s ner                        # if tmux is missing: nohup ALLOW_CPU=1 SEEDS="42" bash run_lab.sh > lab_run.out 2>&1 &
ALLOW_CPU=1 SEEDS="42" bash run_lab.sh
```
- 3 trainings run in order: `bert-base-cased`, `bert-base-uncased`, `bert-base-cased-caseaug` (3 epochs each),
  then predictions, analysis, tests and packing. On CPU this takes **several hours**: start it as early as possible.
- Detach (run keeps going): `Ctrl-b` then `d`. Re-attach: `tmux attach -t ner`.
- If the session dies or the machine restarts: run the same two `cd` / `ALLOW_CPU=...` lines again.
  Finished trainings are skipped (resumable).
- Check progress from another normal terminal: `tail -f ~/Desktop/nlp-02/results/logs/lab_full_run.log`
- Done when you see `ALL DONE` and `results/results.json` exists.

📸 **S4**: `=== training bert-base-cased seed=42 ===` and the progress bar.
📸 **S5**: `top` (or `htop`) in another terminal showing `python` using the CPU.
📸 **S6**: an epoch evaluation line: `grep -iE "eval_f1|val F1" results/logs/lab_full_run.log | tail`
📸 **S7**: `cat results/logs/lab_pytest.log` (the `N passed` line).
📸 **S8**: `cat results/logs/lab_headline.txt` (metrics table for both models) and `ALL DONE`.

### L4. Sanity check (normal terminal)
```bash
cat results/logs/lab_headline.txt
```
`bert-base-cased` test F1 should be about **0.90–0.92**. If it is far lower, stop and tell the team.

### L5. Save the screenshots and the evidence note (normal terminal)
Save the 8 screenshots as PNG in `~/Desktop/nlp-02/screenshots/` with exactly these names:
`S1_lab_environment.png`, `S2_nvidia_smi.png`, `S3_env_check.png`, `S4_training_progress.png`, `S5_gpu_in_use.png`,
`S6_epoch_eval.png`, `S7_pytest.png`, `S8_results.png` (S2/S5 keep their names; on CPU they show the no-GPU
environment and CPU usage). Then:
```bash
cd ~/Desktop/nlp-02
mkdir -p screenshots
echo "Full training run on the Virtual Lab (CPU only, no GPU) on $(date +%F), $(nproc) cores; screenshots S1-S8 in this folder." > screenshots/VIRTUAL_LAB.txt
ls screenshots
```
Create `VIRTUAL_LAB.txt` **only after** the run really finished. The report mentions the lab and shows the
screenshots only if this file exists.

### L6. Hand the results to the team for the write-up (normal terminal)
The report text (`report/narrative.py`) and the notebook conclusion are written from the real numbers in
`results/results.json`. Send the results to whoever writes them (with Claude Code), by **either**:

- **A. Push to GitHub** (needs push access to the repo; when asked for a password, use a GitHub personal access token):
  ```bash
  cd ~/Desktop/nlp-02
  git status                            # outputs/ and .venv/ must NOT be listed (they are gitignored)
  git add results screenshots
  git commit -m "Add Virtual Lab CPU run results and screenshots"
  git pull --rebase && git push
  ```
- **B. Send the files**: `zip -qr lab_handoff.zip results screenshots`, then download `lab_handoff.zip`
  (VS Code Explorer → right-click → Download) and send it to the team.

Wait until the team says the narrative is pushed.

### L7. Rebuild every deliverable (venv terminal)
```bash
cd ~/Desktop/nlp-02
conda deactivate 2>/dev/null; source .venv/bin/activate      # prompt: (.venv) without (base)
git pull                                                      # gets report/narrative.py + notebook conclusion
python tools/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace NER_Cased_vs_Uncased.ipynb
jupyter nbconvert --to html NER_Cased_vs_Uncased.ipynb       # -> NER_Cased_vs_Uncased.html
python report/build_report.py                                 # -> report/Report.pdf
python tools/update_readme.py                                 # fills the README results table
pytest -m "not slow"                                          # all must pass
```
If `git pull` complains about local changes to `README.md` or the notebook: `git stash`, `git pull`, then rerun L7.
Open `report/Report.pdf` and check:
- page 1 has the title, Group 101 and the member table
- the screenshots are in the execution-evidence appendix
- it says the training ran on CPU

The four metrics (Accuracy, Precision, Recall, F1) for both models must match in `results/lab_headline.txt`, the
notebook's Results summary cell, the report's §5 table and the README table.

### L8. Build the submission zip (normal or venv terminal)
```bash
cd ~/Desktop/nlp-02
bash make_zip.sh                                              # -> dist/G101_NLP_Assignment2.zip
unzip -l dist/G101_NLP_Assignment2.zip | grep -E "Report.pdf|ipynb|results.json|S8_results|Contribution" 
unzip -l dist/G101_NLP_Assignment2.zip | grep -cE "outputs/|\.venv/"   # must print 0
```
Download `dist/G101_NLP_Assignment2.zip` (VS Code Explorer → right-click → Download).

### L9. Push and submit (normal terminal)
```bash
git add -A && git status                                      # no outputs/, .venv/, large files
git commit -m "Add executed notebook, report and results table"
git pull --rebase && git push
```
**Upload `G101_NLP_Assignment2.zip` to the course portal before 16 November.** Check `G101_Contribution.xlsx`
matches what the group agreed.

### If the CPU run is too slow
Time the L2 quick run. If the full run cannot finish before the deadline, tell the team before changing anything.
Fewer epochs or dropping the case-augmentation run are possible, but the report must then state exactly what was run.

---

## Step-by-step: from here to submission

### Part 1: Full training on the Virtual Lab (GPU)
> No GPU on the lab machine? Use [Virtual Lab only: complete workflow (CPU, no GPU)](#virtual-lab-only-complete-workflow-cpu-no-gpu) instead.

Needs: a GPU (CUDA), Python 3.10–3.12, `git`, internet access to github.com and huggingface.co, and
**≥ 10 GB free disk** (checkpoints are about 0.4 GB each, plus temporary per-epoch checkpoints).
Expected time: about 30–60 min on a modern GPU (estimate). On an Apple M4 it was about 19 min per training run.

**Step 1.1: Open a terminal on the lab machine and get the code**
```bash
git clone https://github.com/shreyanshsharma88/nlp-02.git
cd nlp-02
```
📸 **Screenshot S1**: the lab environment (lab portal / desktop / URL visible) with this terminal open.

**Step 1.2: Check the GPU**
```bash
nvidia-smi
python3 --version
df -h .
```
📸 **Screenshot S2**: `nvidia-smi` output showing the GPU name and memory.

**Step 1.3 (optional, recommended): Quick smoke run, 2–5 min**
```bash
bash run_lab.sh --quick
```
This creates `.venv`, installs everything and records `results/logs/lab_environment.txt`, then trains on 500
sentences for 1 epoch. It should end with `QUICK RUN FINISHED`. Near-zero F1 here is normal: the run is too short to
learn. The run only checks that the setup works.

If it stops with `CUDA GPU not visible to torch`, see [Troubleshooting](#troubleshooting).

**Step 1.4: Full run**

Use `tmux` (or `nohup`) so a dropped connection does not kill training:
```bash
tmux new -s ner          # if tmux is not installed:  nohup bash run_lab.sh > lab_run.out 2>&1 &
bash run_lab.sh
# detach: Ctrl-b then d   ·   re-attach later: tmux attach -t ner   ·   (nohup: tail -f lab_run.out)
```
The script prints five banners:

| Banner | What happens |
|---|---|
| `STEP 1/5 Python environment` | venv + dependencies |
| `STEP 2/5 Environment and GPU evidence` | writes `results/logs/lab_environment.txt` (date, host, `nvidia-smi`, torch/CUDA) |
| `STEP 3/5 Full training …` | `python run_all.py --case-aug`: 7 fine-tuning runs, then predictions on 4 case variants, analysis, figures, `results/results.json`, log in `results/logs/lab_full_run.log` |
| `STEP 4/5 Tests` | `pytest` → `results/logs/lab_pytest.log` |
| `STEP 5/5 Headline results` | prints the test metrics table → `results/logs/lab_headline.txt`, packs `results_lab.zip` |

The 7 fine-tuning runs are: cased and uncased × seeds 42, 13 and 1234, plus cased with case augmentation.

📸 **Screenshot S3**: `STEP 2/5` output (torch version, `cuda available: True`, GPU name).
📸 **Screenshot S4**: training in progress, with the progress bar and `=== training bert-base-cased seed=42 ===`.
📸 **Screenshot S5**: `nvidia-smi` in a second terminal **while training runs** (GPU utilisation and memory in use).
📸 **Screenshot S6**: an epoch evaluation line (`eval_f1 …`) or a `val F1=… time=… min` line.
📸 **Screenshot S7**: `STEP 4/5` pytest summary (`… passed`).
📸 **Screenshot S8**: `STEP 5/5` headline results table and `ALL DONE`.

If the run is interrupted, run `bash run_lab.sh` again. Finished runs are skipped, so it resumes.

**Step 1.5: Download the results**
Download **`results_lab.zip`** from the lab machine (portal file browser, `scp`, or JupyterLab's
right-click → Download). The `outputs/` folder (checkpoints) is **not** needed. Optional backup:
`zip -r outputs_best.zip outputs/*/best` if you want the trained models.

### Part 2: Bring the results back to the local project
```bash
cd ~/nlp-02
git pull
rm -rf results && unzip -q ~/Downloads/results_lab.zip      # recreates results/
ls results/results.json results/figures results/logs        # must all exist
mkdir -p screenshots                                         # copy S1–S8 here as PNGs (names below)
echo "Full training run on the Virtual Lab GPU on <date>; screenshots S1-S8 in this folder." > screenshots/VIRTUAL_LAB.txt
```
The report states that the code ran on the Virtual Lab **only if** `screenshots/VIRTUAL_LAB.txt` exists. Create it
only after the lab run really happened.

### Part 3: Rebuild every deliverable locally
```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
```
1. **Sanity check.** Open `results/logs/lab_headline.txt`. `bert-base-cased` test F1 should be about **0.91–0.92**,
   the published range. If it is far off, stop and investigate before writing anything.
2. **Write the analysis text from the real numbers**: `report/narrative.py` (report text) and the notebook
   conclusion. Every number is read from `results/results.json`; nothing is typed by hand. Done with Claude Code,
   reviewed by the team.
3. **Notebook**: rebuild, execute, export:
   ```bash
   python tools/build_notebook.py
   jupyter nbconvert --to notebook --execute --inplace NER_Cased_vs_Uncased.ipynb
   jupyter nbconvert --to html NER_Cased_vs_Uncased.ipynb            # -> NER_Cased_vs_Uncased.html
   ```
   For a notebook PDF, open the HTML in a browser and use Print → Save as PDF → `NER_Cased_vs_Uncased.pdf`.
4. **Report**: `python report/build_report.py` → `report/Report.pdf`. Open it and check every page:
   - page 1 has the title, "Group 101" and the member table
   - no table is cut off and all figures are readable
   - 8–12 pages
   - screenshots S1–S8 appear in the execution-evidence appendix
5. **README results table**: `python tools/update_readme.py`
6. **Tests**: `pytest`. All must pass, including the `results.json` sanity and compliance tests (the slow
   end-to-end smoke test takes about 5–10 min on CPU).
7. **Cross-check**: Accuracy, Precision, Recall and F1 for both models must be identical in
   `results/results.json`, the notebook's Results summary cell, the report's §5 table and the README table.

### Part 4: Package, verify and submit
```bash
bash make_zip.sh                                   # -> dist/G101_NLP_Assignment2.zip
mkdir -p /tmp/zipcheck && unzip -q -o dist/G101_NLP_Assignment2.zip -d /tmp/zipcheck && ls -R /tmp/zipcheck | head -50
cp dist/G101_NLP_Assignment2.zip ~/Downloads/
git status                                         # nothing large or private
git add -A && git commit -m "Add full results, executed notebook, report" && git push
```
The zip must contain:
- the code (`ner/`, `run_all.py`, `run_lab.sh`, `tests/`, `tools/`, `report/build_report.py`)
- `NER_Cased_vs_Uncased.ipynb` (with outputs), plus `.html` / `.pdf`
- `results/`, `report/Report.pdf`, `README.md`, `requirements*.txt`
- `G101_Contribution.xlsx`, `screenshots/`

It must **not** contain `outputs/`, `.venv/` or caches.

Then **upload `dist/G101_NLP_Assignment2.zip` to the portal before the deadline (16 November).**

---

## Screenshots to take (execution evidence)
Save as PNG in `screenshots/` with these names. They go into the report's execution-evidence appendix.

| # | File name | When | Must be visible |
|---|---|---|---|
| S1 | `S1_lab_environment.png` | Step 1.1 | the Virtual Lab portal/desktop (lab name or URL), the terminal, `git clone` done |
| S2 | `S2_nvidia_smi.png` | Step 1.2 | `nvidia-smi`: GPU model, driver/CUDA version, memory |
| S3 | `S3_env_check.png` | `STEP 2/5` | torch version, `cuda available: True`, GPU name |
| S4 | `S4_training_progress.png` | `STEP 3/5` | `=== training bert-base-cased seed=42 ===` and the progress bar |
| S5 | `S5_gpu_in_use.png` | during `STEP 3/5` | `nvidia-smi` showing the python process and GPU utilisation |
| S6 | `S6_epoch_eval.png` | during `STEP 3/5` | an `eval_f1` epoch line or `val F1=… time=… min` |
| S7 | `S7_pytest.png` | `STEP 4/5` | the pytest summary line (`N passed`) |
| S8 | `S8_results.png` | `STEP 5/5` | the headline metrics table for both models and `ALL DONE` |

Optional: a short screen recording of S1→S8, if the course asks for video evidence.

---

## Final full-marks checklist
- [ ] Virtual Lab run finished (`results/logs/lab_full_run.log` ends with `done -> results`)
- [ ] Screenshots S1–S8 in `screenshots/`, `screenshots/VIRTUAL_LAB.txt` created
- [ ] `results/results.json` present; cased test F1 ≈ 0.91–0.92
- [ ] Notebook executed top to bottom with outputs; title cell shows "NLP - Assignment-2, Set-11 | Group 101" and all 4 members
- [ ] Notebook has a heading for each of the 7 requirements, plus "Results summary" and "Conclusion: effect of capitalization on names and organizations"
- [ ] Notebook HTML (and PDF) exported
- [ ] `report/Report.pdf` rebuilt, every page checked, 8–12 pages, group and members on page 1
- [ ] All 4 metrics (Accuracy, Precision, Recall, F1) for **both** models; entity- vs token-level clearly stated
- [ ] ≥ 5 custom predictions shown (18 in the table), with which model is right and why
- [ ] Error analysis present: categories, confused pairs, disagreements, seen/unseen, annotation noise
- [ ] Capitalization explained with evidence: case-perturbed test sets, slices, ambiguous words, PER vs ORG
- [ ] Numbers identical across `results.json`, notebook, report, README
- [ ] `pytest` all green
- [ ] `G101_Contribution.xlsx`: headers, 4 members, percentages confirmed by the group
- [ ] `dist/G101_NLP_Assignment2.zip` built, unzipped and checked, copied to `~/Downloads`
- [ ] Everything committed and pushed to GitHub
- [ ] **Zip uploaded on the portal before the deadline**

---

## Troubleshooting
| Problem | Fix |
|---|---|
| `CUDA GPU not visible to torch` | The pinned `torch==2.15.0` may not match the lab's CUDA driver. Use the lab's pre-installed torch: `rm -rf .venv && KEEP_TORCH=1 bash run_lab.sh`. Or install the torch build for the lab's CUDA version from pytorch.org inside `.venv`, then rerun. |
| `nvidia-smi not available` (lab machine has no GPU) | Ask the lab admin for a GPU node first. If none is available, run on CPU with the primary seed only: `ALLOW_CPU=1 SEEDS="42" bash run_lab.sh --quick`, then `ALLOW_CPU=1 SEEDS="42" bash run_lab.sh` in tmux. This is much slower (several hours for the 3 runs) and the seed-variance section then has one seed only. The report states the device from the training logs. |
| `python3` is older than 3.10 | Use `python3.10`/`python3.11`/`python3.12` if available (edit the `python3 -m venv` line in `run_lab.sh`). |
| `CUDA out of memory` | Close other GPU jobs. As a last resort, lower `per_device_train_batch_size` in `ner/train.py` **for both models** (the comparison must stay fair). |
| Hugging Face download fails | Check internet access to huggingface.co. Behind a proxy, set `HTTPS_PROXY`. No token is needed. |
| Disk full | Keep ≥ 10 GB free. Each run's intermediate checkpoints are deleted after it finishes; only `outputs/*/best` remain. |
| Connection dropped mid-run | Re-attach (`tmux attach -t ner`), or rerun `bash run_lab.sh`: finished runs are skipped. |
| `load_dataset("conll2003")` script error in the log | Expected: the code falls back to `eriktks/conll2003` (parquet). See PLAN.md. |

---

## Results (test set, seed 42; generated from `results/results.json`)
<!-- RESULTS_TABLE_START -->
_Filled in by `python tools/update_readme.py` after the Virtual Lab run._
<!-- RESULTS_TABLE_END -->

---

## Where everything is
| Deliverable | Path |
|---|---|
| Main notebook (executed, with outputs) | `NER_Cased_vs_Uncased.ipynb` (+ `NER_Cased_vs_Uncased.html`, `.pdf`) |
| Report | `report/Report.pdf` (built by `report/build_report.py` from `results/`) |
| All numbers | `results/results.json` |
| Predictions | `results/preds/*.json` (word level), `results/test_predictions_word_level.csv`, `results/custom_predictions.csv` |
| Error analysis tables | `results/disagreements_*.csv`, `results/ambiguous_words.csv` |
| Figures | `results/figures/*.png` |
| Training and lab logs | `results/logs/*.json`, `results/logs/lab_*.txt/.log` |
| Execution evidence | `screenshots/` |
| Contribution sheet | `G101_Contribution.xlsx` |
| Plan, audit, status | `PLAN.md` |
| Code | `ner/` (data, tokenize, train, evaluate, predict, analysis), `run_all.py`, `run_lab.sh`, `tests/`, `tools/` |

---

## Reference: setup, commands, tests, runtimes
**Setup**
```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Python 3.10–3.12
pip install -r requirements-dev.txt                       # requirements.txt = runtime only
```
The dataset loads from the Hugging Face Hub (`eriktks/conll2003`, parquet conversion; see PLAN.md for why).
No HF token is needed.

**Commands**
```bash
bash run_lab.sh                    # Virtual Lab: env evidence + full run + tests + results_lab.zip
python run_all.py --case-aug       # full: 3 seeds x 2 models + case-aug extra -> predictions -> analysis -> results/
python run_all.py --seeds 42       # one seed only
python run_all.py --quick          # smoke run: 500 sentences, 1 epoch -> results_quick/ (add --cpu to force CPU)
python run_all.py --stage analyze  # rebuild results.json + figures from saved predictions (no GPU)
python tools/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace NER_Cased_vs_Uncased.ipynb
python report/build_report.py      # -> report/Report.pdf
python tools/update_readme.py      # README results table
bash make_zip.sh                   # -> dist/G101_NLP_Assignment2.zip
```
In the notebook, `TRAIN = False` (default) reads `results/`; `TRAIN = True` fine-tunes both models inside it.
Fine-tuned checkpoints go to `outputs/` (about 400 MB each, git-ignored).

**Tests**
```bash
pytest -m "not slow"   # unit + compliance tests (seconds)
pytest                 # + end-to-end smoke test (run_all.py --quick --cpu, about 5–10 min)
```

**Runtimes**
| Hardware | One training run (3 epochs) | Full `run_all.py --case-aug` (7 runs + analysis) |
|---|---|---|
| Apple M4 (MPS), measured | ≈ 19 min | ≈ 2.5 h |
| CUDA GPU (fp16) | ≈ 3–8 min (estimate) | ≈ 30–60 min (estimate) |
| CPU only | hours; use `--quick` | not recommended |

**Alternative GPU: RunPod.** Start a PyTorch pod and open JupyterLab → Terminal. Then run the same commands as
Part 1 (`git clone …`, `bash run_lab.sh`, or `KEEP_TORCH=1 bash run_lab.sh` to keep the pod's torch). Download
`results_lab.zip`.

---

## BEFORE SUBMISSION
- **CHECK THE DEADLINE TIME AND THE EXACT SUBMISSION NAMING ON THE PORTAL.**
- **CONFIRM THE CONTRIBUTION PERCENTAGES AND HEADER WORDING IN `G101_Contribution.xlsx` (DEFAULT: 100 EACH).**
- **RUN THE FULL TRAINING ON THE VIRTUAL LAB (PART 1) AND TAKE SCREENSHOTS S1–S8.**
- **COPY `results/` BACK, ADD THE SCREENSHOTS AND `screenshots/VIRTUAL_LAB.txt` (PART 2).**
- **REBUILD THE NOTEBOOK, REPORT, README TABLE AND ZIP FROM THE NEW RESULTS (PARTS 3–4).**
- **OPEN THE NOTEBOOK AND THE REPORT ONCE AND SKIM THEM.**
- **UPLOAD `dist/G101_NLP_Assignment2.zip` TO THE PORTAL BEFORE 16 NOVEMBER.**
