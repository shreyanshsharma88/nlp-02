"""Capitalization experiments, error analysis (requirement 7), custom predictions (requirement 6) and figures.

Everything here works from the saved word-level predictions in results/preds/, so the analysis (and the
notebook / report built from it) can be regenerated without re-training. Only the custom-sentence step needs
the fine-tuned checkpoints; if they are missing the previously saved CSV is reused.
"""
from __future__ import annotations

import collections
import json
import os
import platform
from typing import Dict, List

import numpy as np
import pandas as pd

from ner.data import (CASE_VARIANTS, ENTITY_TYPES, capitalization_profile, entity_spans, is_allcaps_sentence,
                      word_shape)
from ner.evaluate import compute_all_metrics, confusion, mcnemar, paired_bootstrap
from ner.train import MODELS, run_name

SHORT = {"bert-base-cased": "cased", "bert-base-uncased": "uncased"}
COLORS = {"bert-base-cased": "#2a78d6", "bert-base-uncased": "#eb6834"}
PRIMARY_SEED = 42

# ------------------------------------------------------------------ custom sentences (requirement 6)
# Minimal pairs that differ only in capitalization. `expected` is our own annotation following CoNLL-2003
# conventions (sports clubs = ORG, countries = LOC). Correctness is computed by code against it.
CUSTOM_SENTENCES = [
    {"id": "1a", "text": "Apple released a new phone .", "expected": [("Apple", "ORG")],
     "tests": "Company name, Title case"},
    {"id": "1b", "text": "apple released a new phone .", "expected": [("apple", "ORG")],
     "tests": "Same sentence lower-cased: only context signals the company"},
    {"id": "2a", "text": "I will meet Bill in May .", "expected": [("Bill", "PER")],
     "tests": "Person name vs common noun 'bill'; month 'May' is not an entity"},
    {"id": "2b", "text": "i will meet bill in may .", "expected": [("bill", "PER")],
     "tests": "Lower-cased: 'bill' and 'may' look like common words"},
    {"id": "3a", "text": "The WHO warned about the outbreak .", "expected": [("WHO", "ORG")],
     "tests": "Acronym ORG that collides with the pronoun 'who'"},
    {"id": "3b", "text": "who warned about the outbreak ?", "expected": [],
     "tests": "Pronoun 'who' – should not be an entity"},
    {"id": "4", "text": "Jordan scored for Chicago .", "expected": [("Jordan", "PER"), ("Chicago", "ORG")],
     "tests": "Jordan = person (not country); Chicago = sports club (ORG in CoNLL)"},
    {"id": "5", "text": "Turkey exports turkey to Germany .", "expected": [("Turkey", "LOC"), ("Germany", "LOC")],
     "tests": "Country vs the bird; only capitalization separates them"},
    {"id": "6a", "text": "Amazon shipped my order yesterday .", "expected": [("Amazon", "ORG")],
     "tests": "Company"},
    {"id": "6b", "text": "We flew over the amazon rainforest .", "expected": [("amazon", "LOC")],
     "tests": "Lower-case geographic name"},
    {"id": "6c", "text": "AMAZON STOCK SURGES AFTER EARNINGS", "expected": [("AMAZON", "ORG")],
     "tests": "ALL-CAPS headline: every word is capitalised"},
    {"id": "7a", "text": "US officials met in Geneva .", "expected": [("US", "LOC"), ("Geneva", "LOC")],
     "tests": "Acronym 'US' (country)"},
    {"id": "7b", "text": "let us go home .", "expected": [], "tests": "Pronoun 'us'"},
    {"id": "8a", "text": "Paris Hilton visited Paris .", "expected": [("Paris Hilton", "PER"), ("Paris", "LOC")],
     "tests": "Same word as person and city"},
    {"id": "8b", "text": "paris hilton visited paris .", "expected": [("paris hilton", "PER"), ("paris", "LOC")],
     "tests": "Lower-cased version"},
    {"id": "9", "text": "Washington signed the bill in Washington .",
     "expected": [("Washington", "PER"), ("Washington", "LOC")], "tests": "Person vs place, same surface form"},
    {"id": "10", "text": "Angela Merkel met Emmanuel Macron and Joe Biden in Berlin .",
     "expected": [("Angela Merkel", "PER"), ("Emmanuel Macron", "PER"), ("Joe Biden", "PER"), ("Berlin", "LOC")],
     "tests": "Name-heavy sentence"},
    {"id": "11", "text": "Microsoft , Google and the United Nations criticised NATO and UEFA .",
     "expected": [("Microsoft", "ORG"), ("Google", "ORG"), ("United Nations", "ORG"), ("NATO", "ORG"),
                  ("UEFA", "ORG")], "tests": "Organization-heavy: Title case, multi-word and acronyms"},
]


# ------------------------------------------------------------------ helpers
def _load(path):
    with open(path) as f:
        return json.load(f)


def _save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def _span_set(tags):
    return {(t, s, e) for t, s, e in entity_spans(tags)}


def _r(x, n=4):
    return float(round(x, n))


# ------------------------------------------------------------------ slices
def case_slices(tokens, gold, preds: Dict[str, List[List[str]]]):
    """Recall-style slices on the original test set.

    token_shape: word-level recall on gold entity tokens, by the token's shape (Title / ALLCAPS / lower / mixed).
    position:    same, sentence-initial vs mid-sentence entity tokens.
    o_false_positive: share of gold-O tokens predicted as an entity, by shape and position (where capitals mislead).
    headline:    entity-level P/R/F1 on ALL-CAPS sentences vs normal sentences.
    """
    out = {"token_shape": {}, "position": {}, "o_false_positive": {}, "headline": {}}
    for m, pred in preds.items():
        shape_c = collections.defaultdict(lambda: [0, 0])
        pos_c = collections.defaultdict(lambda: [0, 0])
        ofp = collections.defaultdict(lambda: [0, 0])
        for toks, g, p in zip(tokens, gold, pred):
            for i, (w, gt, pt) in enumerate(zip(toks, g, p)):
                shp = word_shape(w)
                pos = "sentence_initial" if i == 0 else "mid_sentence"
                if gt != "O":
                    ty = gt[2:]
                    ok = int(gt == pt)
                    for key in (ty, "ALL"):
                        shape_c[(key, shp)][0] += ok
                        shape_c[(key, shp)][1] += 1
                        pos_c[(key, pos)][0] += ok
                        pos_c[(key, pos)][1] += 1
                else:
                    fp = int(pt != "O")
                    ofp[(shp, pos)][0] += fp
                    ofp[(shp, pos)][1] += 1
        out["token_shape"][m] = {f"{k[0]}|{k[1]}": {"recall": v[0] / v[1], "n": v[1]} for k, v in shape_c.items()}
        out["position"][m] = {f"{k[0]}|{k[1]}": {"recall": v[0] / v[1], "n": v[1]} for k, v in pos_c.items()}
        out["o_false_positive"][m] = {f"{k[0]}|{k[1]}": {"fp_rate": v[0] / v[1], "n": v[1]} for k, v in ofp.items()}
        caps_idx = [i for i, t in enumerate(tokens) if is_allcaps_sentence(t)]
        norm_idx = [i for i, t in enumerate(tokens) if not is_allcaps_sentence(t)]
        res = {}
        for nm, idx in [("allcaps_sentences", caps_idx), ("normal_sentences", norm_idx)]:
            mm = compute_all_metrics([gold[i] for i in idx], [pred[i] for i in idx])
            res[nm] = {"n_sentences": len(idx), "precision": mm["precision"], "recall": mm["recall"], "f1": mm["f1"],
                       "per_type_f1": {t: v["f1"] for t, v in mm["per_type"].items()},
                       "per_type_recall": {t: v["recall"] for t, v in mm["per_type"].items()}}
        out["headline"][m] = res
    return out


# ------------------------------------------------------------------ ambiguous words
def ambiguous_words(train_tokens, train_tags, tokens, gold, preds, min_each: int = 2):
    """Words whose lower-case form occurs both as an entity and as a non-entity in TRAIN (no test leakage)."""
    ent = collections.Counter()
    non = collections.Counter()
    for toks, tags in zip(train_tokens, train_tags):
        for w, t in zip(toks, tags):
            if not any(c.isalpha() for c in w):
                continue
            (ent if t != "O" else non)[w.lower()] += 1
    amb = {w for w in ent if ent[w] >= min_each and non[w] >= min_each}
    per_model = {}
    word_rows = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for m, pred in preds.items():
        c = collections.defaultdict(lambda: [0, 0])
        for toks, g, p in zip(tokens, gold, pred):
            for w, gt, pt in zip(toks, g, p):
                lw = w.lower()
                if lw not in amb:
                    continue
                grp = "gold_entity" if gt != "O" else "gold_O"
                ok = int(gt == pt)
                for k in (grp, "all"):
                    c[k][0] += ok
                    c[k][1] += 1
                word_rows[(lw, grp)][m][0] += ok
                word_rows[(lw, grp)][m][1] += 1
        per_model[m] = {k: {"accuracy": v[0] / v[1], "n": v[1]} for k, v in c.items()}
    rows = []
    for (w, grp), d in word_rows.items():
        n = next(iter(d.values()))[1]
        row = {"word": w, "gold": grp, "n_test": n, "train_as_entity": ent[w], "train_as_O": non[w]}
        for m in preds:
            row[f"acc_{SHORT.get(m, m)}"] = d[m][0] / d[m][1]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("n_test", ascending=False)
    return {"n_ambiguous_types_train": len(amb), "per_model": per_model}, df


# ------------------------------------------------------------------ error analysis
def categorize_errors(gold, pred):
    """Span-level error categories.

    correct            exact span and type
    wrong_type         exact span, wrong type
    boundary           overlapping span, same type, different boundaries
    type_and_boundary  overlapping span, different type and boundaries
    missed (FN)        gold span overlaps no predicted span
    spurious (FP)      predicted span overlaps no gold span
    """
    cats = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    confused = collections.Counter()
    for g, p in zip(gold, pred):
        gs, ps = entity_spans(g), entity_spans(p)
        for gt, s, e in gs:
            if (gt, s, e) in ps:
                cats["correct"] += 1
                by_type[gt]["correct"] += 1
                continue
            over = [(pt, ps_, pe) for pt, ps_, pe in ps if ps_ < e and s < pe]
            if not over:
                c = "missed"
            elif any(ps_ == s and pe == e for _, ps_, pe in over):
                c = "wrong_type"
                pt = next(pt for pt, ps_, pe in over if ps_ == s and pe == e)
                confused[(gt, pt)] += 1
            elif any(pt == gt for pt, _, _ in over):
                c = "boundary"
            else:
                c = "type_and_boundary"
                confused[(gt, over[0][0])] += 1
            cats[c] += 1
            by_type[gt][c] += 1
        for pt, s, e in ps:
            if not any(s < ge and gs_ < e for _, gs_, ge in gs):
                cats["spurious"] += 1
                by_type[pt]["spurious"] += 1
    errors = {k: v for k, v in cats.items() if k != "correct"}
    tot = sum(errors.values())
    return {"counts": dict(cats), "error_share": {k: v / tot for k, v in errors.items()} if tot else {},
            "by_type": {t: dict(v) for t, v in by_type.items()},
            "confused_pairs": [{"gold": a, "pred": b, "count": n} for (a, b), n in confused.most_common(10)]}


def mistagged_tokens(tokens, gold, pred, k=15):
    c = collections.Counter()
    detail = collections.defaultdict(collections.Counter)
    for toks, g, p in zip(tokens, gold, pred):
        for w, gt, pt in zip(toks, g, p):
            if gt != pt:
                c[w] += 1
                detail[w][f"{gt}->{pt}"] += 1
    return [{"token": w, "errors": n, "most_common": detail[w].most_common(1)[0][0]} for w, n in c.most_common(k)]


def _cap_role(toks, g, pa, pb):
    """Code-derived description of the capitalization situation in a sentence (used to annotate examples)."""
    notes = []
    if is_allcaps_sentence(toks):
        notes.append("ALL-CAPS headline")
    diff = [i for i in range(len(toks)) if pa[i] != pb[i]]
    shapes = collections.Counter(word_shape(toks[i]) for i in diff)
    if diff and diff[0] == 0:
        notes.append("disagreement on sentence-initial word")
    for shp, n in shapes.items():
        notes.append(f"{n} disputed {shp} token(s)")
    ent_lower = [toks[i] for i, t in enumerate(g) if t != "O" and word_shape(toks[i]) == "lower"]
    if ent_lower:
        notes.append("lower-case entity: " + ", ".join(ent_lower))
    o_title = [toks[i] for i in diff if g[i] == "O" and word_shape(toks[i]) in ("Title", "ALLCAPS") and i > 0]
    if o_title:
        notes.append("capitalised non-entity: " + ", ".join(o_title))
    return "; ".join(notes)


def _fmt_spans(toks, tags):
    return "; ".join(f"{' '.join(toks[s:e])}[{t}]" for t, s, e in entity_spans(tags)) or "-"


def disagreements(tokens, gold, pred_c, pred_u, k=10):
    only_c, only_u = [], []
    for i, (toks, g, pc, pu) in enumerate(zip(tokens, gold, pred_c, pred_u)):
        gset = _span_set(g)
        ok_c, ok_u = _span_set(pc) == gset, _span_set(pu) == gset
        if ok_c == ok_u:
            continue
        row = {"idx": i, "sentence": " ".join(toks), "gold": _fmt_spans(toks, g), "cased": _fmt_spans(toks, pc),
               "uncased": _fmt_spans(toks, pu), "capitalization_role": _cap_role(toks, g, pc, pu)}
        (only_c if ok_c else only_u).append(row)

    def pick(rows):  # prefer short, informative sentences; deterministic
        return sorted(rows, key=lambda r: (len(r["sentence"]) > 140, -("ALL-CAPS" in r["capitalization_role"]),
                                           r["idx"]))[:k]
    return {"n_only_cased_correct": len(only_c), "n_only_uncased_correct": len(only_u),
            "only_cased_correct_examples": pick(only_c), "only_uncased_correct_examples": pick(only_u),
            "all_only_cased": only_c, "all_only_uncased": only_u}


def seen_unseen(train_tokens, train_tags, tokens, gold, preds):
    seen_exact, seen_lower = set(), set()
    for toks, tags in zip(train_tokens, train_tags):
        for t, s, e in entity_spans(tags):
            txt = " ".join(toks[s:e])
            seen_exact.add(txt)
            seen_lower.add(txt.lower())
    res = {}
    for m, pred in preds.items():
        c = collections.defaultdict(lambda: [0, 0])
        for toks, g, p in zip(tokens, gold, pred):
            ps = _span_set(p)
            for t, s, e in entity_spans(g):
                txt = " ".join(toks[s:e])
                grp = "seen" if txt in seen_exact else ("seen_other_case" if txt.lower() in seen_lower else "unseen")
                ok = int((t, s, e) in ps)
                for key in (grp, f"{grp}|{t}"):
                    c[key][0] += ok
                    c[key][1] += 1
        res[m] = {k: {"recall": v[0] / v[1], "n": v[1]} for k, v in c.items()}
    return res


def noise_candidates(tokens, gold, pred_lists: List[List[List[str]]], k=25):
    """Sentences where ALL models/seeds agree with each other but not with gold — candidates for annotation noise."""
    rows = []
    for i, toks in enumerate(tokens):
        first = _span_set(pred_lists[0][i])
        if all(_span_set(p[i]) == first for p in pred_lists[1:]) and first != _span_set(gold[i]):
            rows.append({"idx": i, "sentence": " ".join(toks), "gold": _fmt_spans(toks, gold[i]),
                         "all_models_predict": _fmt_spans(toks, pred_lists[0][i])})
    return len(rows), sorted(rows, key=lambda r: (len(r["sentence"]), r["idx"]))[:k]


# ------------------------------------------------------------------ figures
def _figs(R, res_dir, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "figure.dpi": 150, "savefig.bbox": "tight", "axes.edgecolor": "#52514e",
                         "text.color": "#0b0b0b", "axes.labelcolor": "#52514e"})
    fdir = os.path.join(res_dir, "figures")
    os.makedirs(fdir, exist_ok=True)
    models = MODELS
    w = 0.38

    def grouped(ax, cats, vals_by_model, ylim=None, fmt="{:.3f}", label_vals=True):
        x = np.arange(len(cats))
        for j, m in enumerate(models):
            v = vals_by_model[m]
            bars = ax.bar(x + (j - 0.5) * w, v, w * 0.92, color=COLORS[m], label=m)
            if label_vals:
                for b, val in zip(bars, v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(val), ha="center",
                            va="bottom", fontsize=7, color="#52514e")
        ax.set_xticks(x, cats)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False, fontsize=8)

    # 1 metrics bar
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    keys = ["accuracy", "precision", "recall", "f1"]
    grouped(ax, ["Accuracy\n(token)", "Precision\n(entity)", "Recall\n(entity)", "F1\n(entity)"],
            {m: [R["test"][m][k] for k in keys] for m in models}, ylim=(0.85, 1.0))
    ax.set_title("Test-set metrics (seed 42)", loc="left", fontsize=11)
    fig.savefig(f"{fdir}/metrics_bar.png"); plt.close(fig)

    # 2 per-type F1
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    grouped(ax, ENTITY_TYPES, {m: [R["test"][m]["per_type"][t]["f1"] for t in ENTITY_TYPES] for m in models},
            ylim=(0.7, 1.0))
    ax.set_title("Per-entity-type F1 on test", loc="left", fontsize=11)
    fig.savefig(f"{fdir}/per_type_f1.png"); plt.close(fig)

    # 3 confusion matrices
    import seaborn as sns
    for m in models:
        cm = np.array(R["confusion"][m]["normalized"])
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names, ax=ax,
                    cbar=False, annot_kws={"fontsize": 7}, vmin=0, vmax=1, linewidths=0.5, linecolor="#fcfcfb")
        ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.grid(False)
        ax.set_title(f"{m}: row-normalised token confusion", loc="left", fontsize=10)
        fig.savefig(f"{fdir}/confusion_{SHORT[m]}.png"); plt.close(fig)

    # 4 training curves
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for m in models:
        lh = R["training"][m]["log_history"]
        st = [(e["step"], e["loss"]) for e in lh if "loss" in e and "eval_loss" not in e]
        ev = [(e["epoch"], e["eval_f1"]) for e in lh if "eval_f1" in e]
        if st:
            axes[0].plot(*zip(*st), color=COLORS[m], lw=1.6, label=m)
        if ev:
            axes[1].plot(*zip(*ev), color=COLORS[m], lw=2, marker="o", ms=6, label=m)
    axes[0].set_title("Training loss", loc="left", fontsize=11); axes[0].set_xlabel("step")
    axes[1].set_title("Validation entity F1 per epoch", loc="left", fontsize=11); axes[1].set_xlabel("epoch")
    for a in axes:
        a.legend(frameon=False, fontsize=8)
    fig.savefig(f"{fdir}/training_curves.png"); plt.close(fig)

    # 5 case variants
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, key, title in zip(axes, ["overall", "PER", "ORG"], ["Overall F1", "PER F1", "ORG F1"]):
        grouped(ax, CASE_VARIANTS, {m: [R["case_variants"][m][v]["f1" if key == "overall" else key] for v in CASE_VARIANTS]
                                    for m in models}, ylim=(0, 1.05), fmt="{:.2f}")
        ax.set_title(title + " by input casing", loc="left", fontsize=10)
    fig.savefig(f"{fdir}/case_variants.png"); plt.close(fig)

    # 6 slices: recall by entity-token shape for PER / ORG / ALL
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, ty in zip(axes, ["ALL", "PER", "ORG"]):
        shapes = [s for s in ["Title", "ALLCAPS", "lower", "mixed"]
                  if f"{ty}|{s}" in R["slices"]["token_shape"][models[0]]
                  and R["slices"]["token_shape"][models[0]][f"{ty}|{s}"]["n"] >= 5]
        grouped(ax, [f"{s}\n(n={R['slices']['token_shape'][models[0]][f'{ty}|{s}']['n']})" for s in shapes],
                {m: [R["slices"]["token_shape"][m][f"{ty}|{s}"]["recall"] for s in shapes] for m in models},
                ylim=(0, 1.05), fmt="{:.2f}")
        ax.set_title(f"{'All entity' if ty == 'ALL' else ty} tokens: recall by shape", loc="left", fontsize=10)
    fig.savefig(f"{fdir}/slices_recall.png"); plt.close(fig)

    # 7 error categories (stacked horizontal, counts)
    cats = ["missed", "spurious", "wrong_type", "boundary", "type_and_boundary"]
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    for yi, m in enumerate(models):
        left = 0
        for c, col in zip(cats, pal):
            v = R["errors"][m]["counts"].get(c, 0)
            ax.barh(yi, v, left=left, color=col, edgecolor="#fcfcfb", linewidth=2, label=c if yi == 0 else None)
            if v >= 25:
                ax.text(left + v / 2, yi, str(v), ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks(range(len(models)), models); ax.grid(axis="y", visible=False)
    ax.set_xlabel("number of span-level errors (test)")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.32))
    fig.savefig(f"{fdir}/error_categories.png"); plt.close(fig)

    # 8 capitalization profile of training data
    prof = R["capitalization_profile"]["train"]["shape_share"]
    shapes = ["Title", "ALLCAPS", "lower", "mixed", "nonalpha"]
    shp_col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c3c2b7"]
    rows = ENTITY_TYPES + ["O"]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for yi, t in enumerate(rows):
        left = 0
        for s, col in zip(shapes, shp_col):
            v = prof[t].get(s, 0)
            ax.barh(yi, v, left=left, color=col, edgecolor="#fcfcfb", linewidth=1.5, label=s if yi == 0 else None)
            if v >= 0.07:
                ax.text(left + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks(range(len(rows)), rows); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1); ax.set_xlabel("share of tokens (train)")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.savefig(f"{fdir}/cap_profile.png"); plt.close(fig)

    # 9 seed variance
    if R.get("seed_summary") and len(R["seeds"]) > 1:
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        for j, m in enumerate(models):
            per = R["seed_runs"][m]
            for k, key in enumerate(["f1", "lower_f1"]):
                vals = [per[str(s)][key] for s in R["seeds"]]
                xs = np.full(len(vals), k) + (j - 0.5) * 0.3
                ax.scatter(xs, vals, color=COLORS[m], s=40, label=m if k == 0 else None, zorder=3)
        ax.set_xticks([0, 1], ["test F1 (original)", "test F1 (lower-cased input)"])
        ax.set_title("Per-seed test F1 (each dot = one seed)", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(f"{fdir}/seed_variance.png"); plt.close(fig)


# ------------------------------------------------------------------ custom predictions
def custom_predictions(out_root, res_dir, seed, use_cpu=False):
    from ner.predict import NERPredictor, simple_tokenize
    path = os.path.join(res_dir, "custom_predictions.csv")
    dirs = {m: os.path.join(out_root, run_name(m, seed), "best") for m in MODELS}
    if not all(os.path.exists(os.path.join(d, "config.json")) for d in dirs.values()):
        print("[custom] checkpoints not found; keeping existing", path)
        return pd.read_csv(path) if os.path.exists(path) else None
    rows = [{"id": s["id"], "sentence": s["text"], "tests": s["tests"],
             "expected": "; ".join(f"{t}[{y}]" for t, y in s["expected"]) or "-"} for s in CUSTOM_SENTENCES]
    for m, d in dirs.items():
        pr = NERPredictor(d, device="cpu" if use_cpu else None)
        for row, s in zip(rows, CUSTOM_SENTENCES):
            words = simple_tokenize(s["text"])
            tags = pr.predict_words([words])[0]
            got = {(" ".join(words[a:b]), t) for t, a, b in entity_spans(tags)}
            row[SHORT[m]] = "; ".join(f"{' '.join(words[a:b])}[{t}]" for t, a, b in entity_spans(tags)) or "-"
            row[f"{SHORT[m]}_tags"] = " ".join(tags)
            row[f"{SHORT[m]}_correct"] = got == set(s["expected"])
    df = pd.DataFrame(rows)

    def verdict(r):
        if r.cased_correct and r.uncased_correct:
            return "both"
        if r.cased_correct:
            return "cased only"
        if r.uncased_correct:
            return "uncased only"
        return "neither"
    df["correct"] = df.apply(verdict, axis=1)
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------------ main entry
def run_analysis(ds, names, seeds, res_dir, out_root, quick=False, skip_custom=False, use_cpu=False):
    import torch
    import transformers
    from transformers import AutoTokenizer

    from ner.tokenize import show_tokenization, subword_stats

    P = os.path.join(res_dir, "preds")
    primary = PRIMARY_SEED if PRIMARY_SEED in seeds else seeds[0]
    gold_test = _load(f"{P}/gold_test.json")
    gold_val = _load(f"{P}/gold_validation.json")
    tokens, gold = gold_test["tokens"], gold_test["tags"]
    train_tokens = [list(t) for t in ds["train"]["tokens"]]
    train_tags = [[names[i] for i in s] for s in ds["train"]["ner_tags"]]

    _cache: Dict = {}

    def pred(m, seed, split="test", variant="original"):
        key = f"{P}/{run_name(m, seed)}_{split}_{variant}.json"
        if key not in _cache:
            _cache[key] = _load(key)
        return _cache[key]

    R: Dict = {"meta": {
        "title": "NLP - Assignment-2, Set-11 | Group 101",
        "dataset_source": getattr(ds, "source", "?"), "quick": quick, "seeds": seeds, "primary_seed": primary,
        "split_sizes": {k: len(v) for k, v in ds.items()}, "label_names": names,
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "metric_definitions": {
            "accuracy": "token-level (word-level; first sub-token per word, -100 ignored), includes O",
            "precision/recall/f1": "entity-level micro average, seqeval default (conlleval) mode",
            "macro_f1": "unweighted mean of PER/ORG/LOC/MISC entity F1",
            "f1_strict_iob2": "entity-level F1, seqeval strict IOB2 mode"}},
        "seeds": seeds}

    # training logs
    R["training"] = {}
    for m in MODELS:
        lg = _load(os.path.join(res_dir, "logs", f"{run_name(m, primary)}.json"))
        R["training"][m] = {k: lg[k] for k in ["device", "fp16", "train_time_sec", "best_metric_val_f1",
                                                "best_checkpoint", "config", "log_history"]}
        R["training"][m]["train_time_all_seeds_sec"] = {
            str(s): _load(os.path.join(res_dir, "logs", f"{run_name(m, s)}.json"))["train_time_sec"] for s in seeds}

    # main metrics (primary seed)
    R["test"], R["validation"] = {}, {}
    for m in MODELS:
        R["test"][m] = compute_all_metrics(gold, pred(m, primary))
        R["validation"][m] = compute_all_metrics(gold_val["tags"], pred(m, primary, "validation"))

    # seeds
    R["seed_runs"], R["seed_summary"] = {}, {}
    for m in MODELS:
        R["seed_runs"][m] = {}
        for s in seeds:
            mm = compute_all_metrics(gold, pred(m, s))
            low = compute_all_metrics(gold, pred(m, s, variant="lower"))
            vv = compute_all_metrics(gold_val["tags"], pred(m, s, "validation"))
            R["seed_runs"][m][str(s)] = {"accuracy": mm["accuracy"], "precision": mm["precision"],
                                         "recall": mm["recall"], "f1": mm["f1"], "macro_f1": mm["macro_f1"],
                                         "PER_f1": mm["per_type"]["PER"]["f1"], "ORG_f1": mm["per_type"]["ORG"]["f1"],
                                         "lower_f1": low["f1"], "val_f1": vv["f1"]}
        keys = list(next(iter(R["seed_runs"][m].values())).keys())
        R["seed_summary"][m] = {k: {"mean": float(np.mean([R["seed_runs"][m][str(s)][k] for s in seeds])),
                                    "std": float(np.std([R["seed_runs"][m][str(s)][k] for s in seeds], ddof=1))
                                    if len(seeds) > 1 else 0.0} for k in keys}

    # significance (primary seed)
    pc, pu = pred(MODELS[0], primary), pred(MODELS[1], primary)
    R["bootstrap_cased_minus_uncased"] = paired_bootstrap(gold, pc, pu, n_resamples=200 if quick else 1000)
    R["mcnemar_token"] = mcnemar(gold, pc, pu)

    # confusion
    R["confusion"] = {}
    for m in MODELS:
        cm, norm = confusion(gold, pred(m, primary), names)
        R["confusion"][m] = {"labels": names, "counts": cm.tolist(), "normalized": norm.tolist()}

    # case variants
    R["case_variants"] = {}
    for m in MODELS:
        R["case_variants"][m] = {}
        for v in CASE_VARIANTS:
            mm = compute_all_metrics(gold, pred(m, primary, variant=v))
            R["case_variants"][m][v] = {"f1": mm["f1"], "precision": mm["precision"], "recall": mm["recall"],
                                        "accuracy": mm["accuracy"],
                                        **{t: mm["per_type"][t]["f1"] for t in ENTITY_TYPES}}
    # uncased invariance check: original vs lower (and upper)
    inv = {}
    for v in ["lower", "upper", "title"]:
        a, b = pred(MODELS[1], primary), pred(MODELS[1], primary, variant=v)
        diff_words = [(i, j) for i in range(len(a)) for j in range(len(a[i])) if a[i][j] != b[i][j]]
        ex = []
        for i, j in diff_words[:10]:
            ex.append({"sentence_idx": i, "word": tokens[i][j], "transformed": {"lower": tokens[i][j].lower(),
                       "upper": tokens[i][j].upper(), "title": tokens[i][j][:1].upper() + tokens[i][j][1:].lower()}[v],
                       "pred_original": a[i][j], "pred_variant": b[i][j]})
        inv[v] = {"n_words_different": len(diff_words), "n_words": sum(len(x) for x in a),
                  "n_sentences_different": len({i for i, _ in diff_words}), "examples": ex}
    R["uncased_invariance"] = inv

    # tokenizer evidence
    R["tokenization"] = {"example_sentence": "Apple CEO Tim Cook visited Washington .", "example": {}, "subword_stats": {},
                         "uncased_ids_identical_after_lowercasing": {}}
    ex_words = R["tokenization"]["example_sentence"].split()
    for m in MODELS:
        tok = AutoTokenizer.from_pretrained(m)
        R["tokenization"]["example"][m] = show_tokenization(tok, ex_words)
        R["tokenization"]["subword_stats"][m] = {sp: subword_stats(tok, ds[sp], names) for sp in ["train", "test"]}
        same = 0
        for t in tokens:
            same += tok(t, is_split_into_words=True)["input_ids"] == tok([w.lower() for w in t],
                                                                        is_split_into_words=True)["input_ids"]
        R["tokenization"]["uncased_ids_identical_after_lowercasing"][m] = {"identical_sentences": same,
                                                                            "total": len(tokens)}

    R["capitalization_profile"] = capitalization_profile(ds, names)

    preds_primary = {m: pred(m, primary) for m in MODELS}
    R["slices"] = case_slices(tokens, gold, preds_primary)
    amb, amb_df = ambiguous_words(train_tokens, train_tags, tokens, gold, preds_primary)
    R["ambiguous_words"] = amb
    amb_df.to_csv(os.path.join(res_dir, "ambiguous_words.csv"), index=False)

    # errors
    R["errors"] = {m: categorize_errors(gold, preds_primary[m]) for m in MODELS}
    for m in MODELS:
        R["errors"][m]["top_mistagged_tokens"] = mistagged_tokens(tokens, gold, preds_primary[m])
    dis = disagreements(tokens, gold, preds_primary[MODELS[0]], preds_primary[MODELS[1]])
    pd.DataFrame(dis.pop("all_only_cased")).to_csv(os.path.join(res_dir, "disagreements_only_cased_correct.csv"),
                                                   index=False)
    pd.DataFrame(dis.pop("all_only_uncased")).to_csv(os.path.join(res_dir, "disagreements_only_uncased_correct.csv"),
                                                     index=False)
    R["disagreements"] = dis
    R["seen_unseen"] = seen_unseen(train_tokens, train_tags, tokens, gold, preds_primary)
    all_preds = [pred(m, s) for m in MODELS for s in seeds]
    n_noise, noise = noise_candidates(tokens, gold, all_preds)
    R["annotation_noise_candidates"] = {"n_sentences_all_runs_agree_but_differ_from_gold": n_noise,
                                        "examples": noise}

    # predictions CSV (primary seed, word level)
    rows = []
    for i, (toks, g) in enumerate(zip(tokens, gold)):
        for j, w in enumerate(toks):
            rows.append({"sent": i, "pos": j, "word": w, "gold": g[j], "cased": preds_primary[MODELS[0]][i][j],
                         "uncased": preds_primary[MODELS[1]][i][j],
                         "cased_lower_input": pred(MODELS[0], primary, variant="lower")[i][j]})
    pd.DataFrame(rows).to_csv(os.path.join(res_dir, "test_predictions_word_level.csv"), index=False)

    # custom sentences
    if not skip_custom:
        df = custom_predictions(out_root, res_dir, primary, use_cpu)
    else:
        p = os.path.join(res_dir, "custom_predictions.csv")
        df = pd.read_csv(p) if os.path.exists(p) else None
    if df is not None:
        R["custom_predictions"] = {"n": len(df), "cased_correct": int(df["cased_correct"].sum()),
                                   "uncased_correct": int(df["uncased_correct"].sum()),
                                   "rows": df.drop(columns=[c for c in df.columns if c.endswith("_tags")]).to_dict(
                                       "records")}

    # extra experiment: cased model trained with case augmentation (if it was run)
    aug = run_name("bert-base-cased-caseaug", primary)
    if os.path.exists(f"{P}/{aug}_test_original.json"):
        lg = _load(os.path.join(res_dir, "logs", f"{aug}.json"))
        R["case_augmentation"] = {"description": "bert-base-cased fine-tuned with 25% of training sentences "
                                                 "lower-cased and 10% upper-cased (same config, seed)",
                                  "val_f1": lg["best_metric_val_f1"], "train_time_sec": lg["train_time_sec"],
                                  "variants": {}}
        for v in CASE_VARIANTS:
            mm = compute_all_metrics(gold, _load(f"{P}/{aug}_test_{v}.json"))
            R["case_augmentation"]["variants"][v] = {"f1": mm["f1"], "precision": mm["precision"],
                                                     "recall": mm["recall"], "accuracy": mm["accuracy"],
                                                     **{t: mm["per_type"][t]["f1"] for t in ENTITY_TYPES}}
    _figs(R, res_dir, names)
    for m in MODELS:  # keep results.json readable: drop the long log from the copy, kept in logs/
        R["training"][m]["log_history"] = [e for e in R["training"][m]["log_history"]
                                           if "eval_f1" in e or "loss" in e]
    _save(R, os.path.join(res_dir, "results.json"))
    print("analysis written to", os.path.join(res_dir, "results.json"))
    return R
