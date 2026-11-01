# Freight Rate Prediction

Predicts `posted_rate` for truckloads. Trains on `data/train_test.csv` (Jan–Oct 2025) and scores the
12,000 loads in `data/validation.csv` (Nov–Dec 2025).

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
sh run_all.sh
```

On Windows PowerShell (the default VS Code terminal), use `.\run_all.ps1` instead. Both scripts finish by
running `src/check_submission.py`, which prints PASS/FAIL for the submission format.

Or run it step by step:

| Step | Command | Output |
|---|---|---|
| Exploration + data-quality summary | `python src/eda.py` | `outputs/eda.png`, `outputs/data_quality.json` |
| Time-based CV, model comparison, final fit | `python src/train.py` | `outputs/cv_results.csv`, `outputs/cv_summary.csv`, `outputs/holdout_daily.png`, `outputs/feature_importance.png`, `outputs/model.joblib` |
| Predictions | `python src/predict.py` | `validation_predictions.csv`, `outputs/december_chart_predictions.csv`, `outputs/december_chart.png` |
| Report | `python src/make_report.py` | `report/report.docx` |

Training plus CV takes a few minutes on a laptop.

### December chart via `score.py`

`outputs/december_chart_predictions.csv` holds `data/december-chart-inputs.csv` with `predicted_rate`
filled in. Run the provided `score.py` on it and save its chart as `outputs/score_december_chart.png`.
`make_report.py` uses that file when it exists, and falls back to `outputs/december_chart.png` otherwise.

## Approach in brief

- **Cleaning** (`src/pipeline.py: clean`):
  - Negative weights are sign errors, so they are flipped.
  - Missing weight is kept with a flag.
  - Missing `market_index` is filled with that day's median, since it's a daily market-wide value.
  - 677 labels with rate per mile outside $1.4–$4.0 form separate error clusters and are dropped from training.
- **Split:** time-based only.
  - Six expanding-window folds, each trained on all prior months and tested on the next two months.
  - Sep–Oct is the primary holdout.
  - Each fold also hides 8 random cities from training, because the validation set has 8 cities never seen in the labelled data.
- **Model** (`RateModel(mode="hybrid")`):
  - Target is log(rate per mile).
  - A Huber linear stage covers the parts that must extrapolate: the `market_index` slope, the quarter-on-quarter rise in base rates, and the ramp into quarter end.
  - LightGBM on the residuals learns lane geometry, equipment, weight and calendar effects.
- **Result:** Sep–Oct holdout MAE is about $29, vs $83 for a rate-per-mile lookup baseline. Full tables are in the report.

## Layout

```
data/            input CSVs
src/pipeline.py  cleaning, features, models, metrics
src/eda.py       exploration figures
src/train.py     cross-validation + final model
src/predict.py   validation predictions + December chart
src/make_report.py
outputs/         figures, CV tables, model
report/          report.docx
validation_predictions.csv
```
