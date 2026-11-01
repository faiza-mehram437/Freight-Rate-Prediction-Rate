"""Exploration figures and a data-quality summary used in the report."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import OUT, RPM_MAX, RPM_MIN, TARGET, load


def main():
    OUT.mkdir(exist_ok=True)
    tr, val = load("train_test.csv"), load("validation.csv")
    tr["rpm"] = tr[TARGET] / tr["distance"]
    tr_cities = set(tr["pickup"]) | set(tr["delivery"])
    new_cities = sorted((set(val["pickup"]) | set(val["delivery"])) - tr_cities)
    val_new_city = val["pickup"].isin(new_cities) | val["delivery"].isin(new_cities)
    tr_lanes = set(zip(tr["pickup"], tr["delivery"]))
    val_new_lane = ~pd.Series(list(zip(val["pickup"], val["delivery"]))).isin(tr_lanes)

    quality = {
        "train_rows": len(tr), "validation_rows": len(val),
        "train_dates": [str(tr["date"].min().date()), str(tr["date"].max().date())],
        "validation_dates": [str(val["date"].min().date()), str(val["date"].max().date())],
        "missing_weight": {"train": int(tr["weight"].isna().sum()), "validation": int(val["weight"].isna().sum())},
        "missing_market_index": {"train": int(tr["market_index"].isna().sum()),
                                 "validation": int(val["market_index"].isna().sum())},
        "negative_weight": {"train": int((tr["weight"] < 0).sum()), "validation": int((val["weight"] < 0).sum())},
        "rate_per_mile_above_max": int((tr["rpm"] > RPM_MAX).sum()),
        "rate_per_mile_below_min": int((tr["rpm"] < RPM_MIN).sum()),
        "duplicate_rows": int(tr.drop(columns="load_id").duplicated().sum()),
        "cities_train": len(tr_cities),
        "cities_only_in_validation": new_cities,
        "validation_rows_touching_new_city": int(val_new_city.sum()),
        "validation_rows_on_unseen_lane": int(val_new_lane.sum()),
        "median_rate_per_mile": round(float(tr["rpm"].median()), 3),
        "corr_rate_distance": round(float(tr[[TARGET, "distance"]].corr().iloc[0, 1]), 3),
        "median_rpm_by_equipment": tr.groupby("equipment")["rpm"].median().round(3).to_dict(),
        "within_day_std_market_index": round(float(tr.groupby("date")["market_index"].std().mean()), 4),
        "overall_std_market_index": round(float(tr["market_index"].std()), 4),
    }
    json.dump(quality, open(OUT / "data_quality.json", "w"), indent=2)
    print(json.dumps(quality, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ok = tr["rpm"].between(RPM_MIN, RPM_MAX)
    ax.scatter(tr.loc[ok, "distance"], tr.loc[ok, TARGET], s=2, alpha=0.15, color="#3b6ea5", label="kept")
    ax.scatter(tr.loc[~ok, "distance"], tr.loc[~ok, TARGET], s=8, color="#c0392b", label="outlier label")
    ax.set(xlabel="Distance (mi)", ylabel="Posted rate ($)", title="Rate vs distance")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(tr["rpm"].clip(0, 10), bins=120, color="#3b6ea5")
    for b in (RPM_MIN, RPM_MAX):
        ax.axvline(b, color="#c0392b", ls="--")
    ax.set(xlabel="Rate per mile ($, clipped at 10)", ylabel="Loads", title="Rate per mile with outlier bounds")
    ax.set_yscale("log")

    ax = axes[1, 0]
    daily = tr.groupby("date").agg(rpm=("rpm", "median"), mi=("market_index", "median"))
    vdaily = val.groupby("date")["market_index"].median()
    ax.plot(daily.index, daily["rpm"].rolling(7).mean(), color="#3b6ea5", label="median $/mile (7d avg)")
    ax.set_ylabel("$/mile")
    ax2 = ax.twinx()
    ax2.plot(daily.index, daily["mi"], color="#e67e22", alpha=0.7, label="market_index (train)")
    ax2.plot(vdaily.index, vdaily, color="#e67e22", ls=":", label="market_index (validation)")
    ax2.set_ylabel("market_index")
    ax.set_title("Median rate per mile vs market_index over time")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)

    ax = axes[1, 1]
    for eq, g in tr[ok].groupby("equipment"):
        ax.hist(g["rpm"], bins=80, alpha=0.5, label=eq, density=True)
    ax.set(xlabel="Rate per mile ($)", title="Rate per mile by equipment", xlim=(1, 4))
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "eda.png", dpi=150)


if __name__ == "__main__":
    main()
