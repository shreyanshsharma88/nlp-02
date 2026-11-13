"""Build report/Report.pdf from results/results.json and results/figures/*.png. No number is typed by hand.

    python report/build_report.py [--results results] [--out report/Report.pdf]
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["bert-base-cased", "bert-base-uncased"]
SH = {"bert-base-cased": "cased", "bert-base-uncased": "uncased"}
TYPES = ["PER", "ORG", "LOC", "MISC"]
MEMBERS = [("SHREYANSH SHARMA", "2025AG05772"), ("VINAY PRATAP SINGH BHADAURIA", "2025AG05689"),
           ("LAKSHYA KANCHAN", "2025AG05062"), ("VINAYA RAGHAVA R", "2025AG05187")]

ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.2, spaceAfter=4)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.6, leading=9.4, textColor=colors.HexColor("#52514e"))
CELL = ParagraphStyle("c", parent=BODY, fontSize=7.4, leading=8.8, spaceAfter=0)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=12.5, spaceBefore=8,
                    spaceAfter=4, textColor=colors.HexColor("#0d366b"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10, spaceBefore=5, spaceAfter=2,
                    textColor=colors.HexColor("#184f95"))
TITLE = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=16, leading=19, spaceAfter=2)
SUB = ParagraphStyle("st", parent=BODY, alignment=TA_CENTER, fontSize=10.5, leading=13)
CAP = ParagraphStyle("cap", parent=SMALL, alignment=TA_CENTER, spaceAfter=6)


def P(t, st=BODY):
    return Paragraph(t, st)


def pc(x, d=2):
    return f"{100 * x:.{d}f}"


def table(rows, col_widths=None, header=True, font=7.6, zebra=True):
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="CENTER")
    st = [("FONTSIZE", (0, 0), (-1, -1), font), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#184f95")),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#c3c2b7")),
          ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0fb")))
    if zebra:
        for i in range(2, len(rows), 2):
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f6f4")))
    t.setStyle(TableStyle(st))
    return t


def fig(path, width_cm, caption=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    return [img, P(caption, CAP)] if caption else [img]


def side_by_side(paths, width_cm, caption):
    from PIL import Image as PILImage
    cells = []
    for p in paths:
        w, h = PILImage.open(p).size
        cells.append(Image(p, width=width_cm * cm, height=width_cm * cm * h / w))
    t = Table([cells], hAlign="CENTER")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return [t, P(caption, CAP)]


def sig_text(b):
    if b["ci_low"] > 0:
        return "significantly higher for the cased model"
    if b["ci_high"] < 0:
        return "significantly higher for the uncased model"
    return "not significantly different (the 95% CI contains 0)"


def build(res_dir, out):
    R = json.load(open(os.path.join(res_dir, "results.json")))
    F = os.path.join(res_dir, "figures")
    T, V, CV, SL = R["test"], R["validation"], R["case_variants"], R["slices"]
    BS = R["bootstrap_cased_minus_uncased"]
    prof = R["capitalization_profile"]
    from report.narrative import narrative
    N = narrative(R)
    story = []

    # ---------------------------------------------------------------- title block
    story += [P("NLP - Assignment-2, Set-11", TITLE), P("Group 101", SUB),
              P("Token Classification (NER) on CoNLL-2003: <b>bert-base-cased</b> vs <b>bert-base-uncased</b> "
                "and the effect of capitalization on names and organizations", SUB), Spacer(1, 4)]
    story.append(table([["Name", "Registration number"]] + [list(m) for m in MEMBERS], [7.5 * cm, 4.5 * cm]))
    story.append(Spacer(1, 6))
    c, u = MODELS
    story.append(P("<b>Abstract.</b> " + N["abstract"]))

    # ---------------------------------------------------------------- 1 intro
    story.append(P("1. Introduction and task", H1))
    story.append(P(N["intro"]))

    # ---------------------------------------------------------------- 2 dataset
    story.append(P("2. Dataset", H1))
    sz = R["meta"]["split_sizes"]
    story.append(P(
        f"CoNLL-2003 (English, Reuters news) was loaded with Hugging Face <i>datasets</i>. "
        f"<font face='Courier'>load_dataset(\"conll2003\")</font> fails with datasets ≥4 because the repository still "
        f"contains a loading script, so we use the Hub's automatic parquet conversion of the same repository "
        f"(<font face='Courier'>{R['meta']['dataset_source']}</font>). Splits: train {sz['train']:,}, validation "
        f"{sz['validation']:,}, test {sz['test']:,} sentences; 9 BIO labels "
        f"({', '.join(R['meta']['label_names'])}); label ids were checked against the dataset's ClassLabel names. "
        f"No <font face='Courier'>-DOCSTART-</font> rows remain in this version."))
    rows = [["Type", "Title", "ALL-CAPS", "lower", "mixed", "non-alpha", "tokens (train)"]]
    for t in TYPES + ["O"]:
        sh = prof["train"]["shape_share"][t]
        rows.append([t] + [pc(sh.get(k, 0), 1) for k in ["Title", "ALLCAPS", "lower", "mixed", "nonalpha"]]
                    + [f"{sum(prof['train']['shape_counts'][t].values()):,}"])
    story.append(KeepTogether([P("Capitalization profile (share of tokens, %, train split)", H2),
                               table(rows, [2 * cm] + [1.8 * cm] * 5 + [2.4 * cm])]))
    story.append(P(N["dataset"]))

    # ---------------------------------------------------------------- 3 models & tokenization
    story.append(P("3. Models and tokenization", H1))
    story.append(P(N["models"]))
    ex = R["tokenization"]["example"]
    rows = [["Model", "WordPiece tokens for \"Apple CEO Tim Cook visited Washington .\""]]
    for m in MODELS:
        rows.append([m, " ".join(t for t, _ in ex[m])])
    story.append(table(rows, [3.2 * cm, 13.5 * cm]))
    st_ = R["tokenization"]["subword_stats"]
    rows = [["Model (test split)", "pieces / word", "words split", "entity pieces / word", "entity words split",
             "[UNK] words", "entity [UNK]", "ids unchanged by lower-casing"]]
    for m in MODELS:
        d = st_[m]["test"]
        inv = R["tokenization"]["uncased_ids_identical_after_lowercasing"][m]
        rows.append([m, f"{d['avg_pieces_per_word']:.3f}", pc(d["share_words_split"], 1) + "%",
                     f"{d['entity_avg_pieces']:.3f}", pc(d["entity_share_split"], 1) + "%", d["n_unk_words"],
                     d["n_entity_unk_words"], f"{inv['identical_sentences']}/{inv['total']}"])
    story.append(Spacer(1, 3))
    story.append(table(rows, [3.0 * cm, 1.7 * cm, 1.7 * cm, 2.1 * cm, 2.0 * cm, 1.6 * cm, 1.7 * cm, 2.9 * cm]))
    story.append(P(N["tokenization"]))

    # ---------------------------------------------------------------- 4 method
    story.append(P("4. Method", H1))
    tr = R["training"]
    cfg = tr[c]["config"]
    story.append(P(N["method"]))
    rows = [["Setting", "Value (identical for both models)"],
            ["Learning rate / schedule", f"{cfg['learning_rate']} linear decay, warm-up ratio {cfg['warmup_ratio']}"],
            ["Epochs / batch size", f"{cfg['num_train_epochs']} / {cfg['per_device_train_batch_size']} (eval {cfg['per_device_eval_batch_size']})"],
            ["Weight decay / max length", f"{cfg['weight_decay']} / {cfg['max_length']} sub-tokens"],
            ["Seeds", ", ".join(str(s) for s in R["seeds"]) + f" (primary {R['meta']['primary_seed']})"],
            ["Checkpoint selection", "best epoch by validation entity F1"],
            ["Hardware / precision", f"{tr[c]['device']}, fp16={tr[c]['fp16']}"],
            ["Training time (seed 42)", "; ".join(f"{SH[m]} {tr[m]['train_time_sec'] / 60:.1f} min" for m in MODELS)],
            ["Software", f"torch {R['meta']['torch']}, transformers {R['meta']['transformers']}, Python {R['meta']['python']}"]]
    story.append(table(rows, [4.2 * cm, 12.3 * cm]))

    # ---------------------------------------------------------------- 5 results
    story.append(P("5. Results", H1))
    story.append(P("Precision, recall and F1 are <b>entity-level</b> (seqeval, micro-average, exact span and type); "
                   "accuracy is <b>token-level</b> (word-level, includes O). Test numbers are from the seed-42 run; "
                   "mean ± std over seeds is given in the last columns.", BODY))
    rows = [["Model", "Accuracy (token)", "Precision", "Recall", "F1", "Macro F1", "Val F1",
             f"F1 mean ± std ({len(R['seeds'])} seeds)", "Accuracy mean ± std"]]
    for m in MODELS:
        s = R["seed_summary"][m]
        rows.append([m, pc(T[m]["accuracy"]), pc(T[m]["precision"]), pc(T[m]["recall"]), pc(T[m]["f1"]),
                     pc(T[m]["macro_f1"]), pc(V[m]["f1"]), f"{pc(s['f1']['mean'])} ± {pc(s['f1']['std'])}",
                     f"{pc(s['accuracy']['mean'])} ± {pc(s['accuracy']['std'])}"])
    story.append(table(rows, [2.9 * cm, 1.8 * cm, 1.5 * cm, 1.4 * cm, 1.3 * cm, 1.5 * cm, 1.3 * cm, 2.6 * cm, 2.6 * cm]))
    story.append(Spacer(1, 4))
    rows = [["Type", "support"] + [f"{SH[m]} {k}" for m in MODELS for k in ("P", "R", "F1")] + ["ΔF1 95% CI (c−u)"]]
    for t in TYPES:
        b = BS[t]
        rows.append([t, T[c]["per_type"][t]["support"]] +
                    [pc(T[m]["per_type"][t][k]) for m in MODELS for k in ("precision", "recall", "f1")] +
                    [f"{100 * b['delta']:+.2f} [{100 * b['ci_low']:+.2f}, {100 * b['ci_high']:+.2f}]"])
    b = BS["overall"]
    rows.append(["overall", sum(T[c]["per_type"][t]["support"] for t in TYPES)] +
                [pc(T[m][k]) for m in MODELS for k in ("precision", "recall", "f1")] +
                [f"{100 * b['delta']:+.2f} [{100 * b['ci_low']:+.2f}, {100 * b['ci_high']:+.2f}]"])
    story.append(table(rows, [1.6 * cm, 1.4 * cm] + [1.45 * cm] * 6 + [3.6 * cm]))
    story.append(P(N["results"]))
    story += side_by_side([f"{F}/metrics_bar.png", f"{F}/per_type_f1.png"], 8.3,
                          "Figure 1. Test metrics (left) and per-type F1 (right), seed 42.")
    story += side_by_side([f"{F}/confusion_cased.png", f"{F}/confusion_uncased.png"], 8.0,
                          "Figure 2. Row-normalised token-level confusion matrices on test (left cased, right uncased).")
    story += fig(f"{F}/training_curves.png", 15, "Figure 3. Training loss and validation entity F1 per epoch (seed 42).")
    if os.path.exists(f"{F}/seed_variance.png"):
        story += fig(f"{F}/seed_variance.png", 9.5, "Figure 4. Test F1 for each seed, original and lower-cased input.")

    # ---------------------------------------------------------------- 6 capitalization
    story.append(P("6. Capitalization analysis", H1))
    story.append(P("6.1 Case-perturbed test sets", H2))
    story.append(P("The test sentences are rewritten in four casings while the gold labels stay unchanged; both "
                   "fine-tuned models (seed 42) are evaluated on each version."))
    rows = [["Model", "Input", "F1", "PER", "ORG", "LOC", "MISC", "Accuracy"]]
    for m in MODELS:
        for v in ["original", "lower", "upper", "title"]:
            d = CV[m][v]
            rows.append([m if v == "original" else "", v] + [pc(d[k]) for k in ["f1"] + TYPES + ["accuracy"]])
    story.append(table(rows, [3.0 * cm, 1.6 * cm] + [1.55 * cm] * 6))
    story += fig(f"{F}/case_variants.png", 16.5, "Figure 5. Entity F1 by input casing: overall, PER and ORG.")
    story.append(P(N["case_variants"]))
    if "case_augmentation" in R:
        CA = R["case_augmentation"]["variants"]
        rows = [["F1 by input casing", "original", "lower", "upper", "title", "PER lower", "ORG lower"]]
        for lab, d in [("bert-base-cased", CV[c]), ("bert-base-cased + case augmentation", CA), ("bert-base-uncased", CV[u])]:
            rows.append([lab] + [pc(d[v]["f1"]) for v in ["original", "lower", "upper", "title"]] +
                        [pc(d["lower"]["PER"]), pc(d["lower"]["ORG"])])
        story.append(KeepTogether([P("6.1b Extra: case augmentation for the cased model", H2), P(N["case_aug"]),
                                   table(rows, [5.4 * cm] + [1.7 * cm] * 6)]))
    story.append(P("6.2 Slices by capitalization pattern (original test set)", H2))
    rows = [["Slice (word-level recall on gold entity tokens)"] + [SH[m] for m in MODELS] + ["n"]]
    for key in ["ALL|Title", "ALL|ALLCAPS", "ALL|lower", "ALL|mixed", "PER|Title", "PER|ALLCAPS", "ORG|Title",
                "ORG|ALLCAPS", "ORG|lower", "ORG|mixed", "LOC|ALLCAPS", "MISC|ALLCAPS"]:
        if key in SL["token_shape"][c] and SL["token_shape"][c][key]["n"] >= 5:
            rows.append([key.replace("|", " · ").replace("ALL ·", "all entities ·")] +
                        [f"{SL['token_shape'][m][key]['recall']:.3f}" for m in MODELS] + [SL["token_shape"][c][key]["n"]])
    for key in ["ALL|sentence_initial", "ALL|mid_sentence", "PER|sentence_initial", "ORG|sentence_initial"]:
        if key in SL["position"][c]:
            rows.append([key.replace("|", " · ").replace("ALL ·", "all entities ·").replace("_", "-")] +
                        [f"{SL['position'][m][key]['recall']:.3f}" for m in MODELS] + [SL["position"][c][key]["n"]])
    t1 = table(rows, [5.2 * cm, 1.5 * cm, 1.5 * cm, 1.2 * cm])
    rows = [["Gold-O tokens predicted as entity (false-positive rate)"] + [SH[m] for m in MODELS] + ["n"]]
    for key in ["Title|sentence_initial", "Title|mid_sentence", "ALLCAPS|sentence_initial", "ALLCAPS|mid_sentence",
                "lower|mid_sentence"]:
        if key in SL["o_false_positive"][c]:
            rows.append([key.replace("|", " · ").replace("_", "-")] +
                        [f"{SL['o_false_positive'][m][key]['fp_rate']:.4f}" for m in MODELS] +
                        [SL["o_false_positive"][c][key]["n"]])
    rows.append([P("<b>Entity F1 by sentence type</b>", CELL), "", "", ""])
    for k, lab in [("allcaps_sentences", "ALL-CAPS headline sentences"), ("normal_sentences", "normal sentences")]:
        rows.append([lab] + [pc(SL["headline"][m][k]["f1"]) for m in MODELS] + [SL["headline"][c][k]["n_sentences"]])
    t2 = table(rows, [5.2 * cm, 1.5 * cm, 1.5 * cm, 1.2 * cm])
    g = Table([[t1, t2]], hAlign="CENTER")
    g.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(g)
    story += fig(f"{F}/slices_recall.png", 16.5, "Figure 6. Word-level recall of entity tokens by token shape.")
    story.append(P(N["slices"]))
    story.append(P("6.3 Case-ambiguous words", H2))
    aw = R["ambiguous_words"]
    rows = [["Accuracy on test tokens whose lower-case form is both entity and non-entity in train"] +
            [SH[m] for m in MODELS] + ["n"]]
    for k, lab in [("all", "all such tokens"), ("gold_entity", "… gold entity"), ("gold_O", "… gold O")]:
        rows.append([lab] + [f"{aw['per_model'][m][k]['accuracy']:.4f}" for m in MODELS] + [aw["per_model"][c][k]["n"]])
    story.append(table(rows, [9.5 * cm, 1.8 * cm, 1.8 * cm, 1.4 * cm]))
    amb = pd.read_csv(os.path.join(res_dir, "ambiguous_words.csv"))
    sel = amb[amb.n_test >= 3].head(14)
    rows = [["word", "gold", "n test", "train ent/O", "acc cased", "acc uncased"]]
    for _, r in sel.iterrows():
        rows.append([r.word, r.gold.replace("gold_", ""), r.n_test, f"{r.train_as_entity}/{r.train_as_O}",
                     f"{r.acc_cased:.2f}", f"{r.acc_uncased:.2f}"])
    story.append(Spacer(1, 3))
    story.append(table(rows, [2.6 * cm, 2.0 * cm, 1.5 * cm, 2.2 * cm, 1.8 * cm, 1.9 * cm]))
    story.append(P(N["ambiguous"]))
    story.append(P("6.4 Names (PER) versus organizations (ORG)", H2))
    story.append(P(N["per_org"]))

    # ---------------------------------------------------------------- 7 custom
    story.append(P("7. Custom examples", H1))
    story.append(P(N["custom_intro"]))
    cp = R["custom_predictions"]["rows"]
    rows = [["#", "Sentence", "Expected", "Cased output", "Uncased output", "Correct"]]
    for r in cp:
        rows.append([r["id"], r["sentence"], r["expected"], r["cased"], r["uncased"], r["correct"]])
    story.append(table(rows, [0.8 * cm, 4.6 * cm, 3.2 * cm, 3.4 * cm, 3.4 * cm, 1.6 * cm], font=7))
    story.append(P(N["custom"]))

    # ---------------------------------------------------------------- 8 errors
    story.append(P("8. Error analysis", H1))
    E = R["errors"]
    cats = ["correct", "missed", "spurious", "wrong_type", "boundary", "type_and_boundary"]
    rows = [["Category (span level, test)"] + [SH[m] for m in MODELS]]
    for k in cats:
        rows.append([k.replace("_", " ")] + [E[m]["counts"].get(k, 0) for m in MODELS])
    rows.append(["total errors"] + [sum(v for kk, v in E[m]["counts"].items() if kk != "correct") for m in MODELS])
    t1 = table(rows, [4.0 * cm, 1.6 * cm, 1.6 * cm])
    rows = [["Error type by gold type"] + [f"{t} {SH[m]}" for t in ["PER", "ORG"] for m in MODELS]]
    for k in cats[1:]:
        rows.append([k.replace("_", " ")] + [E[m]["by_type"].get(t, {}).get(k, 0) for t in ["PER", "ORG"] for m in MODELS])
    t2 = table(rows, [3.0 * cm] + [1.6 * cm] * 4)
    g = Table([[t1, t2]], hAlign="CENTER")
    g.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(g)
    story += fig(f"{F}/error_categories.png", 13, "Figure 7. Span-level error categories on test.")
    rows = [["Most confused types (gold→pred)"] + [SH[m] for m in MODELS]]
    pairs = sorted({(p["gold"], p["pred"]) for m in MODELS for p in E[m]["confused_pairs"]},
                   key=lambda x: -sum(p["count"] for m in MODELS for p in E[m]["confused_pairs"]
                                      if (p["gold"], p["pred"]) == x))[:6]
    for a, b_ in pairs:
        rows.append([f"{a} → {b_}"] + [next((p["count"] for p in E[m]["confused_pairs"]
                                              if (p["gold"], p["pred"]) == (a, b_)), 0) for m in MODELS])
    t1 = table(rows, [4.0 * cm, 1.6 * cm, 1.6 * cm])
    rows = [["Top mis-tagged tokens: cased", "uncased"]]
    for x, y in zip(E[c]["top_mistagged_tokens"][:8], E[u]["top_mistagged_tokens"][:8]):
        rows.append([f"{x['token']} ({x['errors']}, {x['most_common']})", f"{y['token']} ({y['errors']}, {y['most_common']})"])
    t2 = table(rows, [4.6 * cm, 4.6 * cm])
    g = Table([[t1, t2]], hAlign="CENTER")
    g.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(g)
    story.append(P(N["errors"]))
    D = R["disagreements"]
    story.append(P(f"8.1 Disagreements ({D['n_only_cased_correct']} sentences only cased fully correct, "
                   f"{D['n_only_uncased_correct']} only uncased)", H2))
    for key, lab in [("only_cased_correct_examples", "Only the cased model correct"),
                     ("only_uncased_correct_examples", "Only the uncased model correct")]:
        rows = [["Sentence", "Gold", "Cased", "Uncased", "Capitalization situation (computed)"]]
        for r in D[key][:10]:
            s = r["sentence"] if len(r["sentence"]) < 150 else r["sentence"][:147] + "…"
            rows.append([s, r["gold"], r["cased"], r["uncased"], r["capitalization_role"]])
        story.append(P(lab, SMALL))
        story.append(table(rows, [5.4 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm, 3.0 * cm], font=6.6))
        story.append(Spacer(1, 4))
    story.append(P(N["disagreements"]))
    story.append(P("8.2 Seen vs unseen entities and annotation noise", H2))
    SU = R["seen_unseen"]
    rows = [["Entity recall (test)"] + [SH[m] for m in MODELS] + ["n"]]
    for k in ["seen", "seen_other_case", "unseen", "unseen|PER", "unseen|ORG", "unseen|LOC", "unseen|MISC"]:
        if k in SU[c]:
            rows.append([k.replace("_", " ").replace("|", " · ")] + [f"{SU[m][k]['recall']:.3f}" for m in MODELS] +
                        [SU[c][k]["n"]])
    story.append(table(rows, [5 * cm, 1.8 * cm, 1.8 * cm, 1.4 * cm]))
    story.append(P(N["seen"]))
    story.append(P(N["noise"]))

    # ---------------------------------------------------------------- 9-11
    story.append(P("9. Discussion: when to prefer cased or uncased", H1))
    story.append(P(N["discussion"]))
    story.append(P("10. Limitations and future work", H1))
    story.append(P(N["limitations"]))
    story.append(P("11. Conclusion", H1))
    story.append(P(N["conclusion"]))

    # ---------------------------------------------------------------- appendix
    story.append(P("Appendix A. How to run", H1))
    story.append(P("<font face='Courier'>python -m venv .venv &amp;&amp; source .venv/bin/activate &amp;&amp; "
                   "pip install -r requirements-dev.txt</font><br/>"
                   "<font face='Courier'>python run_all.py</font> (train 3 seeds × 2 models → predict → analyse → "
                   "results/), <font face='Courier'>python run_all.py --quick</font> (smoke run), "
                   "<font face='Courier'>pytest</font>, <font face='Courier'>python report/build_report.py</font>, "
                   "<font face='Courier'>bash make_zip.sh</font>. The notebook "
                   "<font face='Courier'>NER_Cased_vs_Uncased.ipynb</font> rebuilds every table and figure from "
                   "results/ (set <font face='Courier'>TRAIN=True</font> to fine-tune inside it). "
                   "Code: https://github.com/shreyanshsharma88/nlp-02"))
    story.append(P("Appendix B. Environment", H1))
    story.append(P(f"{R['meta']['platform']}; Python {R['meta']['python']}; torch {R['meta']['torch']}; "
                   f"transformers {R['meta']['transformers']}; training device: {tr[c]['device']}."))
    shots_dir = os.path.join(ROOT, "screenshots")
    shots = sorted(f for f in os.listdir(shots_dir) if f.lower().endswith(".png")) if os.path.isdir(shots_dir) else []
    if os.path.exists(os.path.join(shots_dir, "VIRTUAL_LAB.txt")):
        story.append(P("Appendix D. Evidence of execution (Virtual Lab)", H1))
        story.append(P(open(os.path.join(shots_dir, "VIRTUAL_LAB.txt")).read().strip()))
        env_txt = os.path.join(res_dir, "logs", "lab_environment.txt")
        if os.path.exists(env_txt):
            story.append(P("<font face='Courier' size='6.5'>" + open(env_txt).read().replace("&", "&amp;")
                           .replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>").replace(" ", "&nbsp;")
                           + "</font>", SMALL))
        for f in shots:
            story += fig(os.path.join(shots_dir, f), 15, os.path.splitext(f)[0].replace("_", " "))
    elif shots:
        story.append(P("Appendix D. Screenshots", H1))
        for f in shots:
            story += fig(os.path.join(shots_dir, f), 15, os.path.splitext(f)[0].replace("_", " "))
    story.append(P("Appendix C. References", H1))
    for ref in [
        "J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for "
        "Language Understanding. NAACL 2019.",
        "E. F. Tjong Kim Sang, F. De Meulder. Introduction to the CoNLL-2003 Shared Task: Language-Independent "
        "Named Entity Recognition. CoNLL 2003.",
        "H. Nakayama. seqeval: A Python framework for sequence labeling evaluation. 2018. "
        "https://github.com/chakki-works/seqeval",
        "T. Wolf et al. Transformers: State-of-the-Art Natural Language Processing. EMNLP 2020 (System Demos); "
        "Hugging Face documentation, Token classification guide.",
        "Q. Lhoest et al. Datasets: A Community Library for Natural Language Processing. EMNLP 2021 (System Demos).",
        "B. Efron, R. Tibshirani. An Introduction to the Bootstrap. 1993 (paired bootstrap); Q. McNemar, 1947."]:
        story.append(P("• " + ref, SMALL))

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#52514e"))
        canvas.drawString(1.6 * cm, 1.0 * cm, "NLP - Assignment-2, Set-11 | Group 101 | NER: cased vs uncased BERT")
        canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.6 * cm, title="NLP Assignment-2 Set-11 Group 101 - NER cased vs uncased",
                            author="Group 101")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "Report.pdf"))
    a = ap.parse_args()
    build(a.results, a.out)
