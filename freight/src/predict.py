"""Score every validation load and fill the December chart inputs with the trained model."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import DATA, OUT, ROOT, clean, load


def predict_validation(model):
    val = clean(load("validation.csv"))
    pred = pd.Series(model.predict(val), index=val["load_id"])

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    template["predicted_rate"] = template["load_id"].map(pred).round(2)

    assert len(template) == 12000, len(template)
    assert template["load_id"].is_unique
    assert template["predicted_rate"].notna().all(), "load_id in template missing from validation.csv"
    assert (template["predicted_rate"] > 0).all()
    template[["load_id", "predicted_rate"]].to_csv(ROOT / "validation_predictions.csv", index=False)
    print("wrote validation_predictions.csv", template["predicted_rate"].describe().round(1).to_dict())
    return val


def december_chart(model, val):
    """The chart inputs only carry lane/equipment/weight/date. City coordinates come from
    the labelled data, and the daily market signals come from that day's validation loads
    (market_index is market-wide per day; quote_signal uses the daily median)."""
    chart = pd.read_csv(DATA / "december-chart-inputs.csv", parse_dates=["date"])
    labelled = load("train_test.csv")
    coords = pd.concat([
        labelled[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        val[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).drop_duplicates("city").set_index("city")
    daily = val.groupby("date")[["market_index", "quote_signal"]].median()

    X = chart.copy()
    X["pickup_lat"], X["pickup_lon"] = X["pickup"].map(coords["lat"]), X["pickup"].map(coords["lon"])
    X["delivery_lat"], X["delivery_lon"] = X["delivery"].map(coords["lat"]), X["delivery"].map(coords["lon"])
    X = X.join(daily, on="date")
    X = clean(X)
    chart["predicted_rate"] = np.round(model.predict(X), 2)
    chart.to_csv(OUT / "december_chart_predictions.csv", index=False)

    lane = f"{chart['pickup'][0]} -> {chart['delivery'][0]}, {chart['distance'][0]} mi, {chart['equipment'][0]}"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(chart["date"], chart["predicted_rate"], marker="o", color="#3b6ea5")
    hist = labelled[(labelled["pickup"] == chart["pickup"][0]) & (labelled["delivery"] == chart["delivery"][0])
                    & (labelled["equipment"] == chart["equipment"][0])]
    if len(hist):
        ax.axhline(hist["posted_rate"].median(), ls="--", color="grey",
                   label=f"Jan-Oct median posted rate on this lane (n={len(hist)})")
        ax.legend()
    ax.set_title(f"December 2025 predicted rate: {lane}")
    ax.set_ylabel("Predicted rate ($)")
    lo, hi = chart["predicted_rate"].min(), chart["predicted_rate"].max()
    ax.set_ylim(lo * 0.95, hi * 1.05)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "december_chart.png", dpi=150)
    print("December predictions", chart["predicted_rate"].describe().round(1).to_dict())


def main():
    model = joblib.load(OUT / "model.joblib")
    val = predict_validation(model)
    december_chart(model, val)


if __name__ == "__main__":
    main()
