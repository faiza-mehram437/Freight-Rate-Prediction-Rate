"""Time-based cross-validation, model comparison, and final model fit.

Validation set = Nov-Dec 2025, i.e. 1-2 months after the last labelled date, and it
contains 8 cities that never appear in the labelled data. The CV below mimics both:
  * expanding-window folds: train on every month before the window, test on the next
    two months (the same forecast horizon as the real task);
  * within each fold, a "new-city" check: 8 random cities are removed from training and
    the model is scored on test loads that touch them.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import OUT, TARGET, BaselineModel, RateModel, clean, label_outliers, load, metrics

N_NEW_CITIES = 8
TEST_STARTS = ["2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]

MODELS = {
    "baseline_rpm_table": lambda: BaselineModel(),
    "linear_log_rpm": lambda: RateModel(mode="linear"),
    "lgbm_only": lambda: RateModel(mode="lgbm"),
    "hybrid_linear_plus_lgbm": lambda: RateModel(mode="hybrid"),
}


def folds(df):
    month = df["date"].dt.to_period("M")
    for start in TEST_STARTS:
        p = pd.Period(start, "M")
        yield f"{p}..{p + 1}", month < p, (month == p) | (month == p + 1)


def touches(df, cities):
    return df["pickup"].isin(cities) | df["delivery"].isin(cities)


def run_cv(df):
    rows = []
    cities = np.array(sorted(df["pickup"].unique()))
    for i, (name, tr_mask, te_mask) in enumerate(folds(df)):
        train = df[tr_mask & ~df["is_outlier"]]
        test = df[te_mask & ~df["is_outlier"]]  # outlier labels are errors; don't score on them
        test_all = df[te_mask]
        new_cities = np.random.default_rng(i).choice(cities, N_NEW_CITIES, replace=False)
        train_nc = train[~touches(train, new_cities)]
        test_nc = test[touches(test, new_cities)]
        for model_name, make in MODELS.items():
            m = make().fit(train)
            for scenario, data, model in [
                ("all_loads", test, m),
                ("incl_outlier_labels", test_all, m),
                ("new_cities", test_nc, make().fit(train_nc)),
            ]:
                rows.append({"fold": name, "model": model_name, "scenario": scenario,
                             **metrics(data[TARGET], model.predict(data))})
            print(f"{name} {model_name:20s} MAE={rows[-3]['MAE']:.1f} new-city MAE={rows[-1]['MAE']:.1f}")
    return pd.DataFrame(rows)


def plot_holdout(df):
    """Primary holdout: train Jan-Aug, predict Sep-Oct; compare daily median $/mile."""
    clean_rows = df[~df["is_outlier"]]
    in_test = clean_rows["date"] >= "2025-09-01"
    train, test = clean_rows[~in_test], clean_rows[in_test].copy()
    test["pred"] = RateModel(mode="hybrid").fit(train).predict(test)
    test["actual_rpm"] = test[TARGET] / test["distance"]
    test["pred_rpm"] = test["pred"] / test["distance"]
    daily = test.groupby("date")[["actual_rpm", "pred_rpm"]].median()
    daily.columns = ["actual", "predicted"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(daily.index, daily["actual"], label="actual", color="#555")
    ax.plot(daily.index, daily["predicted"], label="predicted (hybrid)", color="#3b6ea5")
    ax.set(title="Holdout Sep-Oct (trained on Jan-Aug): daily median rate per mile", ylabel="$/mile")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "holdout_daily.png", dpi=150)


def main():
    OUT.mkdir(exist_ok=True)
    df = clean(load("train_test.csv"))
    df["is_outlier"] = label_outliers(df)
    print(f"{len(df)} labelled loads, {df['is_outlier'].sum()} outlier labels removed from training")

    cv = run_cv(df)
    cv.to_csv(OUT / "cv_results.csv", index=False)
    summary = cv.groupby(["scenario", "model"])[["MAE", "RMSE", "MAPE"]].agg(["mean", "std"]).round(2)
    print(summary.to_string())
    summary.to_csv(OUT / "cv_summary.csv")

    plot_holdout(df)

    # Final model: every clean labelled load (Jan-Oct).
    final = RateModel(mode="hybrid").fit(df[~df["is_outlier"]])
    joblib.dump(final, OUT / "model.joblib")

    imp = pd.Series(final.model.booster_.feature_importance("gain"), index=final.feature_names)
    imp = (imp / imp.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot.barh(ax=ax, color="#3b6ea5")
    ax.set_title("Residual LightGBM feature importance (share of gain)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)

    json.dump(
        {"n_train": int((~df["is_outlier"]).sum()), "n_outliers_dropped": int(df["is_outlier"].sum()),
         "linear_coefficients": dict(zip(final.linear.feature_names_in_, final.linear.coef_.round(4))),
         "top_features": imp.sort_values(ascending=False).head(8).round(4).to_dict()},
        open(OUT / "train_info.json", "w"), indent=2,
    )
    print("saved", OUT / "model.joblib")


if __name__ == "__main__":
    main()
