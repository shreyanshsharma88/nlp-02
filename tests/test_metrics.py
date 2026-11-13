import pytest

from ner.evaluate import compute_all_metrics, mcnemar, paired_bootstrap, token_accuracy

GOLD = [["B-PER", "I-PER", "O", "B-LOC"], ["B-ORG", "O", "B-MISC", "I-MISC"]]
PRED = [["B-PER", "I-PER", "O", "B-ORG"], ["B-ORG", "O", "B-MISC", "O"]]
# gold entities: PER(0-1) LOC(3) ORG(0) MISC(2-3) = 4 ; pred: PER ok, ORG(3) wrong, ORG(0) ok, MISC(2) boundary
# TP=2, FP=2, FN=2 -> P=R=F1=0.5 ; token acc = 6/8


def test_entity_prf_and_accuracy():
    m = compute_all_metrics(GOLD, PRED)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert token_accuracy(GOLD, PRED) == pytest.approx(0.75)
    assert m["per_type"]["PER"]["f1"] == pytest.approx(1.0)
    assert m["per_type"]["LOC"]["recall"] == pytest.approx(0.0)
    assert m["macro_f1"] == pytest.approx((1.0 + 2 / 3 + 0 + 0) / 4)


def test_perfect():
    m = compute_all_metrics(GOLD, GOLD)
    assert m["f1"] == 1.0 and m["accuracy"] == 1.0


def test_bootstrap_matches_seqeval_and_identical_models():
    b = paired_bootstrap(GOLD * 5, PRED * 5, PRED * 5, n_resamples=50)
    assert b["overall"]["f1_a"] == pytest.approx(compute_all_metrics(GOLD * 5, PRED * 5)["f1"])
    assert b["overall"]["delta"] == 0 and b["overall"]["ci_low"] == 0 and b["overall"]["ci_high"] == 0


def test_mcnemar_counts():
    r = mcnemar(GOLD, GOLD, PRED)
    assert r["only_a_correct"] == 2 and r["only_b_correct"] == 0
