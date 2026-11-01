"""Build report/report.docx from the files written by eda.py, train.py and predict.py."""
import json

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

from pipeline import OUT, ROOT, RPM_MAX, RPM_MIN

MODEL_LABELS = {
    "baseline_rpm_table": "Baseline: median $/mile by equipment x distance band",
    "linear_log_rpm": "Linear (Huber) regression only",
    "lgbm_only": "LightGBM only (all features)",
    "hybrid_linear_plus_lgbm": "Hybrid: linear stage + LightGBM on residuals (chosen)",
}
SCENARIO_LABELS = {
    "all_loads": "All test loads",
    "new_cities": "Loads touching 8 cities hidden from training",
    "incl_outlier_labels": "All test loads incl. bad labels",
}
CHOSEN = "hybrid_linear_plus_lgbm"
FIRST_FOLD = "2025-04..2025-05"


def table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for cell, h in zip(t.rows[0].cells, header):
        cell.text = str(h)
    for r in rows:
        for cell, v in zip(t.add_row().cells, r):
            cell.text = str(v)
    doc.add_paragraph()


def bullets(doc, items):
    for text in items:
        doc.add_paragraph(text, style="List Bullet")


def main():
    q = json.load(open(OUT / "data_quality.json"))
    info = json.load(open(OUT / "train_info.json"))
    coef = info["linear_coefficients"]
    cv = pd.read_csv(OUT / "cv_results.csv")
    s = cv.groupby(["scenario", "model"])[["MAE", "MAPE"]].agg(["mean", "std"])
    s_late = cv[cv["fold"] != FIRST_FOLD].groupby(["scenario", "model"])["MAE"].mean()
    holdout = cv[cv["fold"] == cv["fold"].iloc[-1]].set_index(["scenario", "model"])
    fold_tbl = cv[cv["scenario"] == "all_loads"].pivot(index="fold", columns="model", values="MAE")
    dec = pd.read_csv(OUT / "december_chart_predictions.csv")
    late = fold_tbl.drop(index=FIRST_FOLD)
    n_best, n_late = int((late.idxmin(axis=1) == CHOSEN).sum()), len(late)
    lgbm_vs_hybrid = holdout.loc[("all_loads", "lgbm_only"), "MAE"] / holdout.loc[("all_loads", CHOSEN), "MAE"]

    doc = Document()
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.add_heading("Freight Rate Prediction: Approach and Validation", 0)

    doc.add_heading("1. Summary", 1)
    h, b = holdout.loc[("all_loads", CHOSEN)], holdout.loc[("all_loads", "baseline_rpm_table")]
    doc.add_paragraph(
        "A two-stage model predicts log(rate per mile). A robust linear stage captures the effects that "
        "must extrapolate into Nov-Dec (the market_index slope, a quarter-on-quarter rise in base rates, "
        "and a ramp into quarter end); LightGBM then learns lane geometry, equipment, weight and "
        "calendar effects from the linear stage's residuals. On the primary holdout (train Jan-Aug, "
        f"predict Sep-Oct) the mean absolute error is ${h['MAE']:.0f} ({h['MAPE']:.1f}% MAPE) versus "
        f"${b['MAE']:.0f} ({b['MAPE']:.1f}%) for a rate-per-mile lookup baseline. Predictions for all "
        f"{q['validation_rows']:,} validation loads are in validation_predictions.csv."
    )

    doc.add_heading("2. Key findings from exploration", 1)
    bullets(doc, [
        f"Distance drives the rate (correlation {q['corr_rate_distance']}); median rate is "
        f"${q['median_rate_per_mile']}/mile, with Reefer and Flatbed above Dry Van "
        f"({', '.join(f'{k} ${v}' for k, v in q['median_rpm_by_equipment'].items())}). Rate per mile "
        "falls with distance (short-haul premium).",
        "market_index is a daily, market-wide value with a strong weekly cycle. Day-to-day it moves rates "
        f"almost linearly (about {coef['market_index'] * 100:.0f}% per 1.0 of index), but its monthly "
        "level is not a stable guide: Jan and Sep have similar index levels and clearly different rates.",
        "After removing distance, equipment and market effects, the remaining base rate follows a "
        "quarterly cycle: flat for ~7 weeks, then a ~5% ramp over the last ~5 weeks of each quarter, "
        f"resetting at the next quarter; each quarter also starts ~{coef['quarter_index'] * 100:.1f}% higher "
        "than the last. This pattern is consistent across all four quarters. December is the end of Q4, "
        "so its rates should climb through the month.",
        f"The validation set includes {len(q['cities_only_in_validation'])} cities that never appear in "
        f"the labelled data ({', '.join(q['cities_only_in_validation'])}), affecting "
        f"{q['validation_rows_touching_new_city']:,} of {q['validation_rows']:,} loads.",
    ])
    doc.add_picture(str(OUT / "eda.png"), width=Inches(6.5))

    doc.add_heading("3. Data-quality issues and handling", 1)
    table(doc, ["Issue", "Labelled", "Validation", "Handling"], [
        ["Missing weight", q["missing_weight"]["train"], q["missing_weight"]["validation"],
         "Kept missing with a flag; LightGBM learns which branch missing values take; the linear stage "
         "uses the median"],
        ["Negative weight", q["negative_weight"]["train"], q["negative_weight"]["validation"],
         "Sign error (negative range mirrors the positive range exactly): absolute value + flag"],
        ["Missing market_index", q["missing_market_index"]["train"], q["missing_market_index"]["validation"],
         f"Daily market-wide value (within-day std {q['within_day_std_market_index']} vs "
         f"{q['overall_std_market_index']} overall): filled with that day's median"],
        [f"Rate per mile < ${RPM_MIN} or > ${RPM_MAX}",
         q["rate_per_mile_below_min"] + q["rate_per_mile_above_max"], "-",
         "Two separate clusters (~0.3x and 2-6x normal) with empty gaps around them: treated as label "
         "errors, removed from training and from the main score"],
        ["Cities never seen in labelled data", "-",
         f"{len(q['cities_only_in_validation'])} cities / {q['validation_rows_touching_new_city']:,} loads",
         "No city-ID features; the model uses coordinates and distance; tested explicitly in CV"],
        ["Duplicate rows", q["duplicate_rows"], "-", "None found"],
    ])
    doc.add_paragraph(
        "City coordinates are consistent per city but do not match real-world locations, so they are used "
        "as relative position features, and the provided distance is used as the main length signal."
    )

    doc.add_heading("4. Train/test split and validation approach", 1)
    doc.add_paragraph(
        "The validation set lies in the future (Nov-Dec) relative to all labelled data (Jan-Oct). A random "
        "split would let the model see each month's market level and quarter-cycle position during "
        "training and would overstate accuracy, so every split follows time:"
    )
    bullets(doc, [
        "Expanding-window folds: for each two-month window (Apr-May, May-Jun, ..., Sep-Oct) the model is "
        "trained on every earlier month and tested on that window. This matches the real 1-2 month "
        "forecast horizon and shows how stable the error is across market conditions (6 folds).",
        "Primary holdout: the last fold (train Jan-Aug, test Sep-Oct), the closest copy of the real task. "
        "It also crosses a quarter boundary, like Nov-Dec.",
        "New-city check: in every fold, 8 random cities are removed from training and the model is scored "
        "only on test loads touching them, which reproduces the 8 unseen cities in the validation set.",
        "Leakage controls: cleaning uses features only, never the target; bad labels are removed only from "
        "training and scoring rows; every model is refit from scratch inside each fold on that fold's "
        "training months only.",
        "Metrics: MAE in dollars (primary), RMSE and MAPE. Test loads with bad labels are excluded from the "
        "main score and reported separately.",
        f"Final model: refit on all {info['n_train']:,} clean labelled loads (Jan-Oct) with the settings "
        "chosen in CV, then applied to validation.csv.",
    ])

    doc.add_heading("5. Results", 1)
    rows = []
    for scn in ["all_loads", "new_cities", "incl_outlier_labels"]:
        for m in MODEL_LABELS:
            r = s.loc[(scn, m)]
            rows.append([SCENARIO_LABELS[scn], MODEL_LABELS[m],
                         f"${holdout.loc[(scn, m), 'MAE']:.0f}",
                         f"${r[('MAE', 'mean')]:.0f}", f"${s_late.loc[(scn, m)]:.0f}",
                         f"{r[('MAPE', 'mean')]:.1f}%"])
    table(doc, ["Scenario", "Model", "Sep-Oct holdout MAE", "CV MAE (6 folds)",
                "CV MAE (excl. Apr-May)", "CV MAPE"], rows)
    table(doc, ["Fold (test window)"] + [MODEL_LABELS[m].split(":")[0].split(" (")[0] for m in MODEL_LABELS],
          [[f] + [f"${fold_tbl.loc[f, m]:.0f}" for m in MODEL_LABELS] for f in fold_tbl.index])
    doc.add_paragraph(
        "The Apr-May fold is the one place every learned model loses to the baseline. It trains on Jan-Mar "
        "only: a single quarter (so the quarter step-up cannot be learned) during which market_index and the "
        "base rate rose together, so their effects cannot be separated. When market_index jumped to record "
        "levels in April the models over-attributed it. From the May-Jun fold onward, with more than one "
        f"quarter of history, the hybrid is the best model in {n_best} of {n_late} folds. The final model trains on four quarters, and Nov-Dec market_index "
        "values (0.72-1.10) lie inside the training range."
    )
    doc.add_picture(str(OUT / "holdout_daily.png"), width=Inches(6.5))

    doc.add_heading("6. Model choice and features", 1)
    doc.add_paragraph(
        "Target: log(rate / distance). Rate scales almost linearly with distance, so modelling rate per mile "
        "in log space makes errors comparable across short and long loads; predictions are multiplied back "
        "by distance."
    )
    doc.add_paragraph(
        "Why a hybrid: plain LightGBM cannot extrapolate. It cannot represent a level that keeps rising "
        f"quarter after quarter (its Sep-Oct holdout error is {lgbm_vs_hybrid:.1f}x the hybrid's). A purely "
        "linear model cannot learn lane geometry and interactions. So stage 1 is a Huber (outlier-robust) linear "
        "regression on market_index, quarter index, a quarter-end ramp (0 until 6 weeks before quarter "
        "end, rising to 1 on the last day), log distance (+ squared), equipment, weight and quote_signal. "
        f"Fitted coefficients include market_index {coef['market_index']:+.3f}, quarter index "
        f"{coef['quarter_index']:+.4f} and quarter-end ramp {coef['quarter_end_ramp']:+.3f} (log points). "
        "Stage 2 is LightGBM (Huber loss) on stage-1 residuals using pickup/delivery coordinates and their "
        "difference, route circuity (distance / straight-line distance), distance, equipment, weight (+ "
        "missing flag), quote_signal, day of week, day of month and days to quarter end. market_index and "
        "quarter index are deliberately left out of stage 2 so that only the linear stage extrapolates them; "
        "every stage-2 feature's range is fully covered by the training data."
    )
    doc.add_paragraph(
        "Also deliberately excluded: month and city IDs (Nov-Dec and 8 cities are unseen). Smoothed "
        "lane/city target encodings were tried and gave no CV improvement, so they were removed."
    )
    doc.add_picture(str(OUT / "feature_importance.png"), width=Inches(5))

    doc.add_heading("7. December prediction chart", 1)
    doc.add_paragraph(
        f"Lane {dec['pickup'][0]} to {dec['delivery'][0]}, {dec['distance'][0]} miles, {dec['equipment'][0]}, "
        f"{dec['weight'][0]:,} lb, each day of December 2025. The chart inputs carry no market signals, so "
        "each day's market_index and quote_signal are the median of that day's validation loads, and city "
        f"coordinates come from the labelled data. Predictions range from ${dec['predicted_rate'].min():,.0f} "
        f"to ${dec['predicted_rate'].max():,.0f}: they rise through the month as Q4 ends, with a small "
        "weekly ripple from the daily market_index."
    )
    chart = OUT / "score_december_chart.png"
    if not chart.exists():
        chart = OUT / "december_chart.png"
    doc.add_picture(str(chart), width=Inches(6.5))

    doc.add_heading("8. Limitations and next steps", 1)
    bullets(doc, [
        "The quarter step-up is extrapolated one quarter ahead from four observed quarters; a regime change "
        "would break it. Retraining monthly as new labels arrive would limit that risk.",
        "Loads touching the 8 new cities rely on coordinates only; their CV error is somewhat higher.",
        "Prediction intervals (quantile LightGBM) would show how uncertain each quote is.",
    ])

    (ROOT / "report").mkdir(exist_ok=True)
    doc.save(ROOT / "report" / "report.docx")
    print("wrote report/report.docx")


if __name__ == "__main__":
    main()
