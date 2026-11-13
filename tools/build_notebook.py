"""Generate NER_Cased_vs_Uncased.ipynb (the executed copy is produced with nbconvert --execute)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# NLP - Assignment-2, Set-11 | Group 101

## Token Classification (NER): `bert-base-cased` vs `bert-base-uncased` on CoNLL-2003

| Name | Registration number |
|---|---|
| SHREYANSH SHARMA | 2025AG05772 |
| VINAY PRATAP SINGH BHADAURIA | 2025AG05689 |
| LAKSHYA KANCHAN | 2025AG05062 |
| VINAYA RAGHAVA R | 2025AG05187 |

**Task.** Fine-tune and compare a cased and an uncased BERT model for Named Entity Recognition on the
Hugging Face dataset `conll2003`, report Accuracy, Precision, Recall and F1, and explain the effect of
capitalization on names (PER) and organizations (ORG).

**How this notebook works.** All heavy work (3 seeds × 2 models of fine-tuning, predictions on 4 case
variants of the test set) is done by the same library code (`ner/`) that `run_all.py` uses, and every number
is read from `results/results.json`. With `TRAIN = False` (default) the notebook re-uses the saved results and
runs in a couple of minutes; with `TRAIN = True` it fine-tunes both models itself (≈20 min per model on an
Apple M4 GPU, a few minutes on a CUDA GPU).

Sections follow the 7 assignment requirements:
1. Dataset loading · 2. Preprocessing & tokenization · 3. Model loading · 4. Fine-tuning ·
5. Evaluation (+ 5b capitalization experiments) · 6. Custom predictions · 7. Error analysis ·
Results summary · Conclusion
""")

code("""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
pd.set_option("display.max_colwidth", 120); pd.set_option("display.width", 200)

TRAIN = False            # True -> fine-tune both models here (seed 42) instead of loading saved results
RESULTS = os.environ.get("NER_RESULTS", "results")
OUTPUTS = os.environ.get("NER_OUTPUTS", "outputs")
FIG = f"{RESULTS}/figures"
MODELS = ["bert-base-cased", "bert-base-uncased"]
SHORT = {"bert-base-cased": "cased", "bert-base-uncased": "uncased"}
def fig(name, width=700): display(Image(filename=f"{FIG}/{name}", width=width))
def pct(x): return f"{100*x:.2f}"
""")

md("## 1. Dataset loading (Hugging Face `datasets`)")
md("""
`load_dataset("conll2003")` fails with recent `datasets` (≥4) because the repository still ships a loading
script (`Dataset scripts are no longer supported`). `ner.data.load_conll()` tries the official name first and then
falls back to the Hub's automatic **parquet conversion of the same repository**
(`eriktks/conll2003`, revision `refs/convert/parquet`) — identical sentences and label ids, no script needed.
""")
code("""
from ner.data import load_conll, label_maps, capitalization_profile, case_transform, CASE_VARIANTS
ds = load_conll()
names, id2label, label2id = label_maps(ds)
print(ds)
print("label names:", names)
""")
code("""
# Check that label ids map to the names the dataset declares, on known sentences
ex = ds["train"][0]
display(pd.DataFrame({"token": ex["tokens"], "id": ex["ner_tags"], "label": [id2label[i] for i in ex["ner_tags"]]}).T)
ex = ds["train"][1]
print(list(zip(ex["tokens"], [id2label[i] for i in ex["ner_tags"]])))
assert id2label[3] == "B-ORG" and ds["train"][0]["tokens"][0] == "EU" and ds["train"][0]["ner_tags"][0] == 3
""")
code("""
sizes = {k: len(v) for k, v in ds.items()}
print("split sizes (sentences):", sizes)
ent_counts = {}
for sp in ds:
    c = {t: 0 for t in ["PER","ORG","LOC","MISC"]}
    for tags in ds[sp]["ner_tags"]:
        for t in tags:
            if names[t].startswith("B-"): c[names[t][2:]] += 1
    ent_counts[sp] = c
display(pd.DataFrame(ent_counts).rename_axis("entities (B- count)"))
""")
md("""
### Capitalization profile of the data
Share of tokens of each entity type by word shape (Title = first letter upper; ALLCAPS = ≥2 letters, all upper).
This tells us how much signal capitalization *could* carry for each type.
""")
code("""
R = json.load(open(f"{RESULTS}/results.json"))
prof = R["capitalization_profile"]
tab = pd.DataFrame(prof["train"]["shape_share"]).T.reindex(["PER","ORG","LOC","MISC","O"])[["Title","ALLCAPS","lower","mixed","nonalpha"]].fillna(0)
display((tab*100).round(1).rename_axis("train, % of tokens"))
print({sp: {"ALL-CAPS sentences": prof[sp]["n_allcaps_sentences"], "-DOCSTART- rows": prof[sp]["n_docstart"],
            "sentence-initial capitalised (entity/O)": prof[sp]["sentence_initial_capitalised"]} for sp in prof})
fig("cap_profile.png")
""")

md("## 2. Data preprocessing and tokenization")
md("""
* Each model uses **its own tokenizer** (`AutoTokenizer`), with `is_split_into_words=True` because CoNLL is
  already split into words.
* **Label alignment with `word_ids()`**: the first sub-token of every word receives the word's label; the remaining
  sub-tokens and the special tokens (`[CLS]`, `[SEP]`, padding) receive `-100`, so they are ignored by the loss and by
  the metrics. (Alternative: label every sub-token, turning `B-X` into `I-X` on continuation pieces; first-sub-token
  labelling keeps evaluation exactly word-level.)
* Sentences are truncated at `max_length=128` sub-tokens for training; padding is dynamic (`DataCollatorForTokenClassification`).

**The core mechanism.** `bert-base-uncased` lower-cases (and strips accents from) every input *before* WordPiece, so
the model never sees capitalization. `bert-base-cased` keeps it, and has a different vocabulary.
""")
code("""
from transformers import AutoTokenizer
from ner.tokenize import show_tokenization, tokenize_and_align
toks = {m: AutoTokenizer.from_pretrained(m) for m in MODELS}
s = "Apple CEO Tim Cook visited Washington .".split()
for m in MODELS:
    print(f"{m:18s}", toks[m].convert_ids_to_tokens(toks[m](s, is_split_into_words=True)["input_ids"]))
""")
code("""
# Label alignment on a real training sentence with a split word
sent = ds["train"][12]
for m in MODELS:
    enc = tokenize_and_align({"tokens": [sent["tokens"]], "ner_tags": [sent["ner_tags"]]}, toks[m])
    sub = toks[m].convert_ids_to_tokens(enc["input_ids"][0])
    lab = [id2label[l] if l != -100 else "-100" for l in enc["labels"][0]]
    print(m); display(pd.DataFrame({"sub-token": sub, "word_id": enc.word_ids(0), "label": lab}).T)
""")
code("""
st = R["tokenization"]["subword_stats"]
rows = []
for m in MODELS:
    for sp in ["train", "test"]:
        d = st[m][sp]; rows.append({"model": m, "split": sp, **{k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}})
display(pd.DataFrame(rows).set_index(["model", "split"]))
inv = R["tokenization"]["uncased_ids_identical_after_lowercasing"]
for m in MODELS:
    print(f"{m}: test sentences whose input_ids are identical after lower-casing the text = {inv[m]['identical_sentences']}/{inv[m]['total']}")
""")
md("""
For the cased model, lower-casing changes the input ids of almost every sentence (its vocabulary is case-sensitive);
for the uncased model the ids are identical in all sentences, i.e. lower-cased text is literally the same input.
""")
code("""
# Case variants used later for the robustness experiment (gold labels unchanged, word count preserved)
ex = ds["test"][3]["tokens"]
display(pd.DataFrame({v: case_transform(ex, v) for v in CASE_VARIANTS}).T)
""")

md("## 3. Model loading (Hugging Face `transformers`)")
md("""
`AutoModelForTokenClassification` puts a randomly initialised linear layer (768 → 9 labels) on top of the
pre-trained BERT encoder; `id2label`/`label2id` are stored in the config so predictions come back as tag names.
""")
code("""
from ner.train import build_model, TrainConfig
for m in MODELS:
    tok, model = build_model(m, id2label, label2id)
    n = sum(p.numel() for p in model.parameters())
    print(f"{m}: {n/1e6:.1f}M parameters, vocab={tok.vocab_size}, do_lower_case={getattr(tok, 'do_lower_case', None)}, head={model.classifier}")
    del model
""")

md("## 4. Fine-tuning pipeline")
md("""
Both models are trained by the **same function** (`ner.train.train_model`) with the **same** hyper-parameters,
data, seed and evaluation code — the only difference is the checkpoint name. The best epoch is chosen on the
**validation** entity F1; the test set is used only for final numbers. fp16 is used only on CUDA (training here ran
on one Apple-silicon GPU (MPS) for both models, so both used fp32). We repeat the run with three seeds (13, 42, 1234);
seed 42 is the primary run used for the detailed analysis.

*Implementation note:* in transformers 5.x, `Trainer(load_best_model_at_end=True)` reloads checkpoints without the
legacy `gamma/beta → weight/bias` key conversion and silently keeps the last epoch's LayerNorm weights, so we track
the best checkpoint ourselves and reload it with `from_pretrained` (and assert its validation F1 equals the logged best).
""")
code("""
from dataclasses import asdict
display(pd.DataFrame([asdict(TrainConfig())]).T.rename(columns={0: "both models"}))
if TRAIN:
    from ner.train import train_model
    for m in MODELS:
        s = train_model(m, ds, id2label, label2id, OUTPUTS, f"{RESULTS}/logs", TrainConfig(seed=42), force=True)
        print(m, "best val F1", s["best_metric_val_f1"], f"{s['train_time_sec']/60:.1f} min")
    print("Re-run `python run_all.py --stage predict --stage analyze` to refresh results.json")
""")
code("""
tr = R["training"]
display(pd.DataFrame({m: {"device": tr[m]["device"], "fp16": tr[m]["fp16"],
                          "train time seed 42 (min)": round(tr[m]["train_time_sec"]/60, 1),
                          "best val F1 (seed 42)": round(tr[m]["best_metric_val_f1"], 4),
                          "best checkpoint": os.path.basename(tr[m]["best_checkpoint"])} for m in MODELS}))
ev = {m: [(e["epoch"], round(e["eval_f1"], 4), round(e["eval_loss"], 4)) for e in tr[m]["log_history"] if "eval_f1" in e] for m in MODELS}
display(pd.DataFrame({m: {f"epoch {int(ep)}": f"F1={f} loss={l}" for ep, f, l in ev[m]} for m in MODELS}))
fig("training_curves.png", 850)
""")

md("## 5. Evaluation (Accuracy, Precision, Recall, F1)")
md("""
* **Precision, Recall, F1 — entity level**, micro-averaged with `seqeval` (an entity is correct only if both its
  span and its type match exactly; conlleval-compatible mode).
* **Accuracy — token level**: share of words whose tag is correct (`-100` positions ignored). It includes the
  `O` tokens, which is why it is much higher than F1.
* Macro F1 = unweighted mean of PER/ORG/LOC/MISC F1.
""")
code("""
def mtab(split):
    return pd.DataFrame({m: {"Accuracy (token)": R[split][m]["accuracy"], "Precision (entity)": R[split][m]["precision"],
                             "Recall (entity)": R[split][m]["recall"], "F1 (entity)": R[split][m]["f1"],
                             "Macro F1": R[split][m]["macro_f1"], "F1 strict IOB2": R[split][m]["f1_strict_iob2"]}
                         for m in MODELS}).T.round(4)
print("TEST set (seed 42)"); display(mtab("test"))
print("VALIDATION set (seed 42)"); display(mtab("validation"))
fig("metrics_bar.png", 600)
""")
code("""
per = pd.concat({m: pd.DataFrame(R["test"][m]["per_type"]).T for m in MODELS}, axis=1).round(4)
display(per)
fig("per_type_f1.png", 600)
for m in MODELS:
    print(m); print(R["test"][m]["report"])
""")
md("### Seed variation (3 seeds) and statistical significance")
code("""
ss = R["seed_summary"]
display(pd.DataFrame({m: {k: f"{100*v['mean']:.2f} ± {100*v['std']:.2f}" for k, v in ss[m].items()} for m in MODELS})
        .rename_axis(f"test, mean ± std over seeds {R['seeds']} (%)"))
display(pd.concat({m: pd.DataFrame(R["seed_runs"][m]).T for m in MODELS}).round(4))
b = R["bootstrap_cased_minus_uncased"]
display(pd.DataFrame(b).T.rename(columns={"f1_a": "F1 cased", "f1_b": "F1 uncased", "delta": "Δ (cased-uncased)",
        "ci_low": "95% CI low", "ci_high": "95% CI high", "p_two_sided": "bootstrap p"}).round(4))
print("McNemar on word-level correctness:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in R["mcnemar_token"].items()})
if os.path.exists(f"{FIG}/seed_variance.png"): fig("seed_variance.png", 600)
""")
code("""
fig("confusion_cased.png", 480); fig("confusion_uncased.png", 480)
""")

md("""
## 5b. Capitalization experiments — the effect of capitalization

### (1) Case-perturbed test sets
Both fine-tuned models are evaluated on the test set rewritten four ways — original, all lower-case,
all UPPER-case and Title Case — with the gold labels unchanged.
""")
code("""
cv = R["case_variants"]
rows = []
for m in MODELS:
    for v in ["original", "lower", "upper", "title"]:
        rows.append({"model": m, "input": v, **{k: cv[m][v][k] for k in ["f1", "PER", "ORG", "LOC", "MISC", "accuracy"]}})
cvt = pd.DataFrame(rows).set_index(["model", "input"]).round(4)
display(cvt)
fig("case_variants.png", 950)
inv = R["uncased_invariance"]
for v in inv:
    print(f"uncased model, original vs {v}: {inv[v]['n_words_different']} of {inv[v]['n_words']} word predictions differ "
          f"({inv[v]['n_sentences_different']} sentences)")
    if inv[v]["examples"]: display(pd.DataFrame(inv[v]["examples"]))
""")
md("""
**Extra experiment — can the cased model be made robust?** `bert-base-cased` fine-tuned with *case augmentation*
(25% of training sentences lower-cased, 10% upper-cased; everything else identical, seed 42), evaluated on the same
four test variants.
""")
code("""
if "case_augmentation" in R:
    ca = R["case_augmentation"]["variants"]
    rows = {"bert-base-cased": cv["bert-base-cased"], "bert-base-cased + case aug.": ca, "bert-base-uncased": cv["bert-base-uncased"]}
    display(pd.concat({k: pd.DataFrame(v).T[["f1", "PER", "ORG", "LOC", "MISC"]] for k, v in rows.items()}).round(4))
else:
    print("case-augmentation run not present (python run_all.py --case-aug)")
""")
md("### (2) Slices of the original test set by capitalization pattern")
code("""
sl = R["slices"]
def slice_tab(key, val="recall"):
    return pd.DataFrame({m: {k: f"{sl[key][m][k][val]:.3f} (n={sl[key][m][k]['n']})" for k in sorted(sl[key][m])} for m in MODELS})
print("Entity-token recall by token shape (word-level: predicted tag == gold tag)")
t = slice_tab("token_shape"); display(t[t.index.str.split("|").str[0].isin(["ALL", "PER", "ORG", "LOC", "MISC"])])
fig("slices_recall.png", 950)
print("Entity-token recall: sentence-initial vs mid-sentence"); display(slice_tab("position"))
print("Where capitals mislead: share of gold-O tokens predicted as an entity, by shape and position")
o = slice_tab("o_false_positive", "fp_rate"); display(o[o.index.str.split("|").str[0].isin(["Title", "ALLCAPS", "lower"])])
print("ALL-CAPS headline sentences vs normal sentences (entity level)")
display(pd.concat({m: pd.DataFrame({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} | v["per_type_f1"]
                                    for k, v in sl["headline"][m].items()}).T for m in MODELS}).round(4))
""")
md("### (3) Case-ambiguous words")
md("""
Words whose lower-case form occurs **both** as an entity and as a non-entity in the **training** data
(≥2 times each), e.g. *may/May*, *us/US*, *turkey/Turkey*. The vocabulary is built from train only (no test leakage).
""")
code("""
aw = R["ambiguous_words"]
print("ambiguous word types in train:", aw["n_ambiguous_types_train"])
display(pd.DataFrame({m: {k: f"{v['accuracy']:.4f} (n={v['n']})" for k, v in aw["per_model"][m].items()} for m in MODELS}))
amb = pd.read_csv(f"{RESULTS}/ambiguous_words.csv")
display(amb.head(20).round(3))
""")
md("### (4) PER vs ORG")
code("""
rows = []
for ty in ["PER", "ORG"]:
    for m in MODELS:
        ts = sl["token_shape"][m]
        rows.append({"type": ty, "model": m,
            "test share Title": prof["test"]["shape_share"][ty].get("Title", 0),
            "test share ALLCAPS": prof["test"]["shape_share"][ty].get("ALLCAPS", 0),
            "recall Title tokens": ts.get(f"{ty}|Title", {}).get("recall"),
            "recall ALLCAPS tokens": ts.get(f"{ty}|ALLCAPS", {}).get("recall"),
            "recall lower tokens": ts.get(f"{ty}|lower", {}).get("recall"),
            "F1 original": cv[m]["original"][ty], "F1 lower input": cv[m]["lower"][ty], "F1 UPPER input": cv[m]["upper"][ty],
            "F1 in ALL-CAPS sentences": sl["headline"][m]["allcaps_sentences"]["per_type_f1"].get(ty),
            "F1 in normal sentences": sl["headline"][m]["normal_sentences"]["per_type_f1"].get(ty)})
display(pd.DataFrame(rows).set_index(["type", "model"]).round(3).T)
""")

md("## 6. Prediction on custom examples")
md("""
18 hand-written sentences, mostly **minimal pairs that differ only in capitalization**. Expected entities are our
own annotation (CoNLL conventions: sports clubs = ORG, countries = LOC). Predictions are word-level (first sub-token
of each word); a sentence is counted correct only if the predicted set of (span, type) equals the expected set.
If the fine-tuned checkpoints exist they are run live; otherwise the saved `results/custom_predictions.csv` is shown.
""")
code("""
from ner.analysis import CUSTOM_SENTENCES, custom_predictions
have_ckpt = all(os.path.exists(f"{OUTPUTS}/{m}_s42/best/config.json") for m in MODELS)
cp = custom_predictions(OUTPUTS, RESULTS, 42) if have_ckpt else pd.read_csv(f"{RESULTS}/custom_predictions.csv")
print("live predictions" if have_ckpt else "loaded saved predictions")
display(cp[["id", "sentence", "expected", "cased", "uncased", "correct", "tests"]])
print("correct sentences: cased", int(cp.cased_correct.sum()), "/", len(cp), "| uncased", int(cp.uncased_correct.sum()), "/", len(cp))
""")
code("""
# Ad-hoc predictions on any raw text (needs the checkpoints)
if have_ckpt:
    from ner.predict import NERPredictor, spans_to_text
    preds = {m: NERPredictor(f"{OUTPUTS}/{m}_s42/best") for m in MODELS}
    for text in ["Tim Cook said Apple will open a store in Mumbai .", "tim cook said apple will open a store in mumbai ."]:
        for m in MODELS:
            w, t, _ = preds[m].predict_text(text); print(f"{SHORT[m]:8s} | {text} -> {spans_to_text(w, t)}")
""")

md("## 7. Error analysis")
md("""
Span-level categories (per model, seed 42, test): **missed** (gold entity with no overlapping prediction, FN),
**spurious** (prediction with no overlapping gold entity, FP), **wrong type** (exact span, wrong type),
**boundary** (overlapping span, right type, wrong boundaries), **type & boundary** (both wrong).
""")
code("""
er = R["errors"]
cat = pd.DataFrame({m: er[m]["counts"] for m in MODELS}).fillna(0).astype(int)
cat.loc["errors total"] = cat.drop("correct", errors="ignore").sum()
display(cat)
fig("error_categories.png", 700)
for m in MODELS:
    print(m, "- errors by entity type"); display(pd.DataFrame(er[m]["by_type"]).fillna(0).astype(int))
""")
code("""
for m in MODELS:
    print(m, "- most confused (gold -> predicted) type pairs"); display(pd.DataFrame(er[m]["confused_pairs"]).head(8))
    print(m, "- most frequently mis-tagged tokens"); display(pd.DataFrame(er[m]["top_mistagged_tokens"]).head(12).T)
""")
md("### Disagreements between the models")
code("""
d = R["disagreements"]
print("sentences where only the cased model is fully correct:", d["n_only_cased_correct"])
print("sentences where only the uncased model is fully correct:", d["n_only_uncased_correct"])
print("\\nOnly CASED correct (10 examples)"); display(pd.DataFrame(d["only_cased_correct_examples"]).drop(columns="idx", errors="ignore"))
print("Only UNCASED correct (10 examples)"); display(pd.DataFrame(d["only_uncased_correct_examples"]).drop(columns="idx", errors="ignore"))
""")
md("### Seen vs unseen entities and annotation noise")
code("""
su = R["seen_unseen"]
keys = ["seen", "seen_other_case", "unseen"] + [f"{g}|{t}" for g in ["seen", "unseen"] for t in ["PER", "ORG", "LOC", "MISC"]]
display(pd.DataFrame({m: {k: f"{su[m][k]['recall']:.3f} (n={su[m][k]['n']})" for k in keys if k in su[m]} for m in MODELS})
        .rename_axis("entity recall; seen = exact string appears as an entity in train"))
an = R["annotation_noise_candidates"]
print("sentences where all 6 runs agree with each other but not with gold:", an["n_sentences_all_runs_agree_but_differ_from_gold"])
display(pd.DataFrame(an["examples"]).drop(columns="idx", errors="ignore").head(15))
""")

md("## Results summary")
code("""
summary = pd.DataFrame({m: {"Accuracy (token)": R["test"][m]["accuracy"], "Precision (entity)": R["test"][m]["precision"],
                            "Recall (entity)": R["test"][m]["recall"], "F1 (entity)": R["test"][m]["f1"],
                            f"F1 mean ± std, {len(R['seeds'])} seeds": f"{R['seed_summary'][m]['f1']['mean']:.4f} ± {R['seed_summary'][m]['f1']['std']:.4f}",
                            "F1 PER": R["test"][m]["per_type"]["PER"]["f1"], "F1 ORG": R["test"][m]["per_type"]["ORG"]["f1"],
                            "F1 on lower-cased input": cv[m]["lower"]["f1"], "F1 on UPPER-cased input": cv[m]["upper"]["f1"]}
                        for m in MODELS})
display(summary)
b = R["bootstrap_cased_minus_uncased"]["overall"]
print(f"ΔF1 (cased − uncased) = {b['delta']:+.4f}, 95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] (paired bootstrap, 1000 resamples)")
""")
md("""## Conclusion: effect of capitalization on names and organizations

CONCLUSION_PLACEHOLDER
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
import sys
out = sys.argv[1] if len(sys.argv) > 1 else "NER_Cased_vs_Uncased.ipynb"
concl = open(sys.argv[2]).read() if len(sys.argv) > 2 else "_(written after the full run)_"
for c in nb["cells"]:
    if "CONCLUSION_PLACEHOLDER" in c["source"]:
        c["source"] = c["source"].replace("CONCLUSION_PLACEHOLDER", concl)
nbf.write(nb, out)
print("wrote", out)
