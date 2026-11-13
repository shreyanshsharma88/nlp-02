"""Evaluation (requirement 5).

Precision / Recall / F1  -> ENTITY level, micro-averaged (seqeval, default conlleval-compatible mode).
Accuracy                 -> TOKEN level (= word level: one label per word, -100 positions ignored).
Per-type P/R/F1 for PER, ORG, LOC, MISC; macro F1 = unweighted mean of the four type F1s.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.scheme import IOB2

from ner.data import ENTITY_TYPES

IGNORE = -100


def token_accuracy(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> float:
    n = c = 0
    for g, p in zip(gold, pred):
        assert len(g) == len(p), "gold/pred length mismatch"
        n += len(g)
        c += sum(a == b for a, b in zip(g, p))
    return c / n if n else 0.0


def _counts(gold_seq, pred_seq):
    """Per-type TP/FP/FN for one sentence using seqeval's entity extraction (exact span + type match)."""
    g = set(get_entities(list(gold_seq)))
    p = set(get_entities(list(pred_seq)))
    out = {}
    for t in ENTITY_TYPES:
        gt = {e for e in g if e[0] == t}
        pt = {e for e in p if e[0] == t}
        out[t] = (len(gt & pt), len(pt - gt), len(gt - pt))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def compute_all_metrics(gold: List[List[str]], pred: List[List[str]]) -> Dict:
    rep = classification_report(gold, pred, output_dict=True, zero_division=0)
    per_type = {t: {"precision": rep[t]["precision"], "recall": rep[t]["recall"], "f1": rep[t]["f1-score"],
                    "support": int(rep[t]["support"])} for t in ENTITY_TYPES if t in rep}
    return {
        "accuracy": token_accuracy(gold, pred),
        "precision": precision_score(gold, pred, zero_division=0),
        "recall": recall_score(gold, pred, zero_division=0),
        "f1": f1_score(gold, pred, zero_division=0),
        "macro_f1": float(np.mean([per_type[t]["f1"] for t in per_type])) if per_type else 0.0,
        "f1_strict_iob2": f1_score(gold, pred, mode="strict", scheme=IOB2, zero_division=0),
        "per_type": per_type,
        "report": classification_report(gold, pred, digits=4, zero_division=0),
    }


def make_trainer_compute_metrics(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        gold, pred = [], []
        for p_row, l_row in zip(preds, labels):
            keep = l_row != IGNORE
            gold.append([id2label[int(x)] for x in l_row[keep]])
            pred.append([id2label[int(x)] for x in p_row[keep]])
        m = compute_all_metrics(gold, pred)
        return {k: m[k] for k in ["accuracy", "precision", "recall", "f1", "macro_f1"]}
    return compute_metrics


def paired_bootstrap(gold, pred_a, pred_b, n_resamples: int = 1000, seed: int = 0) -> Dict:
    """Paired bootstrap over sentences of F1(a) - F1(b), overall and per entity type."""
    ca = [_counts(g, p) for g, p in zip(gold, pred_a)]
    cb = [_counts(g, p) for g, p in zip(gold, pred_b)]
    A = np.array([[c[t] for t in ENTITY_TYPES] for c in ca])  # (n, 4, 3)
    B = np.array([[c[t] for t in ENTITY_TYPES] for c in cb])
    rng = np.random.default_rng(seed)
    n = len(gold)
    keys = ["overall"] + ENTITY_TYPES

    def f1s(M):
        s = M.sum(0)  # (4,3)
        res = {"overall": prf(*s.sum(0))[2]}
        for i, t in enumerate(ENTITY_TYPES):
            res[t] = prf(*s[i])[2]
        return res

    obs_a, obs_b = f1s(A), f1s(B)
    diffs = {k: [] for k in keys}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        fa, fb = f1s(A[idx]), f1s(B[idx])
        for k in keys:
            diffs[k].append(fa[k] - fb[k])
    out = {}
    for k in keys:
        d = np.array(diffs[k])
        out[k] = {"f1_a": obs_a[k], "f1_b": obs_b[k], "delta": obs_a[k] - obs_b[k],
                  "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                  "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    return out


def mcnemar(gold, pred_a, pred_b) -> Dict:
    """McNemar test (continuity-corrected chi-square, 1 df) on word-level correctness."""
    b = c = 0
    for g, pa, pb in zip(gold, pred_a, pred_b):
        for x, ya, yb in zip(g, pa, pb):
            ok_a, ok_b = ya == x, yb == x
            b += ok_a and not ok_b
            c += ok_b and not ok_a
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    p = math.erfc(math.sqrt(stat / 2))
    return {"only_a_correct": b, "only_b_correct": c, "chi2": stat, "p_value": p}


def confusion(gold, pred, names: List[str]):
    from sklearn.metrics import confusion_matrix
    g = [x for s in gold for x in s]
    p = [x for s in pred for x in s]
    cm = confusion_matrix(g, p, labels=names)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    return cm, norm
