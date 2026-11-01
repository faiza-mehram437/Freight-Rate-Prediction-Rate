#!/usr/bin/env sh
# Full pipeline: exploration -> CV + final fit -> predictions -> report.
set -e
cd "$(dirname "$0")"
python src/eda.py
python src/train.py
python src/predict.py
python src/make_report.py
python src/check_submission.py
