#!/usr/bin/env bash
# Build the submission zip:  bash make_zip.sh [G101]  ->  dist/G101_NLP_Assignment2.zip
# Includes code, executed notebook (+HTML/PDF), results/, report PDF, README, requirements, contribution sheet,
# screenshots. Excludes checkpoints (outputs/), the venv, caches and quick-run artefacts.
set -euo pipefail
GROUP="${1:-G101}"
cd "$(dirname "$0")"
NAME="${GROUP}_NLP_Assignment2"
STAGE="dist/${NAME}"
rm -rf "$STAGE" "dist/${NAME}.zip"
mkdir -p "$STAGE"
FILES=(README.md PLAN.md requirements.txt requirements-dev.txt pytest.ini .gitignore run_all.py run_lab.sh make_zip.sh
       NER_Cased_vs_Uncased.ipynb NER_Cased_vs_Uncased.html NER_Cased_vs_Uncased.pdf
       "${GROUP}_Contribution.xlsx" report/Report.pdf report/build_report.py report/narrative.py report/__init__.py)
for f in "${FILES[@]}"; do
  if [ -e "$f" ]; then mkdir -p "$STAGE/$(dirname "$f")"; cp "$f" "$STAGE/$f"; else echo "WARNING: missing $f"; fi
done
for d in ner tests tools results screenshots; do
  [ -d "$d" ] && rsync -a --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.DS_Store' "$d" "$STAGE/"
done
(cd dist && zip -qr "${NAME}.zip" "${NAME}")
rm -rf "$STAGE"
echo "built dist/${NAME}.zip ($(du -h "dist/${NAME}.zip" | cut -f1))"
unzip -l "dist/${NAME}.zip" | tail -1
