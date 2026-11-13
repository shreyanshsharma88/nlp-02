import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "results.json")
MODELS = ["bert-base-cased", "bert-base-uncased"]


def _walk_metrics(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _walk_metrics(v)
        elif k in ("accuracy", "precision", "recall", "f1", "macro_f1"):
            yield k, v


@pytest.mark.skipif(not os.path.exists(RES), reason="full results not produced yet")
def test_results_json_sane():
    R = json.load(open(RES))
    for m in MODELS:
        assert m in R["test"] and m in R["validation"]
        for k, v in _walk_metrics(R["test"][m]):
            assert 0.0 <= v <= 1.0, (m, k, v)
        assert set(R["case_variants"][m]) == {"original", "lower", "upper", "title"}
        for v in R["case_variants"][m].values():
            for key in ["f1", "PER", "ORG", "LOC", "MISC"]:
                assert 0 <= v[key] <= 1
    assert R["meta"]["quick"] is False
    assert R["test"]["bert-base-cased"]["f1"] > 0.85, "cased F1 far below published ~0.91"
    assert len(R["custom_predictions"]["rows"]) >= 5


def test_deliverables_exist():
    for f in ["README.md", "PLAN.md", "requirements.txt", "run_all.py", "NER_Cased_vs_Uncased.ipynb",
              "G101_Contribution.xlsx", "make_zip.sh", "report/build_report.py", "ner/data.py", "ner/tokenize.py",
              "ner/train.py", "ner/evaluate.py", "ner/predict.py", "ner/analysis.py"]:
        assert os.path.exists(os.path.join(ROOT, f)), f


def test_contribution_sheet():
    from openpyxl import load_workbook
    ws = load_workbook(os.path.join(ROOT, "G101_Contribution.xlsx")).active
    rows = list(ws.iter_rows(values_only=True))
    assert list(rows[0][:3]) == ["Student Registration Number", "Name", "Percentage of Contribution out of 100%"]
    body = [r for r in rows[1:] if r[0]]
    assert {r[0] for r in body} == {"2025AG05772", "2025AG05689", "2025AG05062", "2025AG05187"}
    assert all(r[2] == 100 for r in body)
