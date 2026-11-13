"""End-to-end smoke test: `python run_all.py --quick --cpu` (500 train sentences, 1 epoch) into a temp dir."""
import json
import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
def test_quick_pipeline(tmp_path):
    t0 = time.time()
    r = subprocess.run([sys.executable, "run_all.py", "--quick", "--cpu", "--results", str(tmp_path / "res"),
                        "--outputs", str(tmp_path / "out")], cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-3000:]
    assert time.time() - t0 < 900
    R = json.load(open(tmp_path / "res" / "results.json"))
    for key in ["meta", "training", "test", "validation", "seed_summary", "bootstrap_cased_minus_uncased",
                "mcnemar_token", "confusion", "case_variants", "uncased_invariance", "tokenization",
                "capitalization_profile", "slices", "ambiguous_words", "errors", "disagreements", "seen_unseen",
                "annotation_noise_candidates", "custom_predictions"]:
        assert key in R, key
    for m in ["bert-base-cased", "bert-base-uncased"]:
        assert 0 <= R["test"][m]["accuracy"] <= 1
    assert os.path.exists(tmp_path / "res" / "figures" / "case_variants.png")
