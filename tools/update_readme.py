"""Insert the headline results table (from results/results.json) into README.md between the markers."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "results.json")))
M = ["bert-base-cased", "bert-base-uncased"]
p = lambda x: f"{100 * x:.2f}"  # noqa: E731
lines = ["| Model | Accuracy (token) | Precision (entity) | Recall (entity) | F1 (entity) | "
         f"F1 mean ± std ({len(R['seeds'])} seeds) | F1, lower-cased input | F1, UPPER-cased input |",
         "|---|---|---|---|---|---|---|---|"]
for m in M:
    t, s, cv = R["test"][m], R["seed_summary"][m]["f1"], R["case_variants"][m]
    lines.append(f"| {m} | {p(t['accuracy'])} | {p(t['precision'])} | {p(t['recall'])} | {p(t['f1'])} | "
                 f"{p(s['mean'])} ± {p(s['std'])} | {p(cv['lower']['f1'])} | {p(cv['upper']['f1'])} |")
b = R["bootstrap_cased_minus_uncased"]["overall"]
lines.append("")
lines.append(f"ΔF1 (cased − uncased) = {100 * b['delta']:+.2f} points, 95% paired-bootstrap CI "
             f"[{100 * b['ci_low']:+.2f}, {100 * b['ci_high']:+.2f}].")
path = os.path.join(ROOT, "README.md")
s = open(path).read()
s = re.sub(r"(<!-- RESULTS_TABLE_START -->\n).*?(\n<!-- RESULTS_TABLE_END -->)",
           lambda mm: mm.group(1) + "\n".join(lines) + mm.group(2), s, flags=re.S)
open(path, "w").write(s)
print("\n".join(lines))
