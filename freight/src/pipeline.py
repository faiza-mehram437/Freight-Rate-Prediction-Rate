"""Shared data loading, cleaning, feature engineering and model code."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

ROOT = Path(__file__).absolute().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

TARGET = "posted_rate"

# Rate-per-mile bounds outside which a training label is treated as a data error.
# 98.6% of loads sit in 1.6-3.5 $/mile; the rest form two separate clusters
# (~0.3x and 2-6x the typical rate) with empty gaps at 1.3-1.5 and 3.6-4.0.
RPM_MIN, RPM_MAX = 1.4, 4.0

def load(name):
    return pd.read_csv(DATA / name, parse_dates=["date"])


def clean(df):
    """Fix data-quality issues that exist in both labelled and unlabelled data.

    Uses only the features (never the target), so it is safe to apply to any split.
    """
    df = df.copy()
    df["equipment"] = df["equipment"].str.strip()

    # Negative weights mirror the positive range exactly (min -47,500 vs max 47,500),
    # so they are sign errors rather than garbage: flip them and keep a flag.
    df["weight_was_negative"] = (df["weight"] < 0).astype(int)
    df["weight"] = df["weight"].abs()
    df["weight_missing"] = df["weight"].isna().astype(int)  # left NaN; LightGBM routes it

    # market_index is a daily market-wide signal (within-day std 0.025 vs 0.17 overall),
    # so a missing value is best recovered from other loads on the same date.
    df["market_index_missing"] = df["market_index"].isna().astype(int)
    daily = df.groupby("date")["market_index"].transform("median")
    df["market_index"] = df["market_index"].fillna(daily)
    df = df.sort_values("date", kind="stable")
    df["market_index"] = df["market_index"].ffill().bfill()
    return df.sort_index()


def label_outliers(df):
    rpm = df[TARGET] / df["distance"]
    return (rpm < RPM_MIN) | (rpm > RPM_MAX)


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def base_features(df):
    X = pd.DataFrame(index=df.index)
    X["distance"] = df["distance"]
    X["log_distance"] = np.log(df["distance"])
    X["weight"] = df["weight"]
    X["weight_missing"] = df["weight_missing"]
    X["market_index"] = df["market_index"]
    X["quote_signal"] = df["quote_signal"]
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = df[c]
    X["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
    X["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]
    X["circuity"] = df["distance"] / _haversine(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    ).clip(lower=1)
    X["dow"] = df["date"].dt.dayofweek
    X["day_of_month"] = df["date"].dt.day
    # Rates follow a quarterly cycle: flat for ~7 weeks, then a ~5% ramp into quarter end,
    # resetting at the next quarter. Days-to-quarter-end covers its full 0-91 range in every
    # quarter of training data, so trees can learn the shape without extrapolating.
    X["days_to_quarter_end"] = days_to_quarter_end(df["date"])
    X["equipment"] = pd.Categorical(df["equipment"], categories=["Dry Van", "Flatbed", "Reefer"])
    # Deliberately no month / day-of-year / time index or city IDs: validation is Nov-Dec
    # (never seen) and includes 8 cities absent from the labelled data. market_index
    # carries the market level and coordinates carry location.
    return X


def days_to_quarter_end(dates):
    return (dates.dt.to_period("Q").dt.end_time.dt.normalize() - dates).dt.days


def quarter_index(dates):
    """0 for 2025-Q1, 1 for 2025-Q2, ...: the base rate steps up ~2% per quarter."""
    return (dates.dt.year - 2025) * 4 + dates.dt.quarter - 1


LGB_PARAMS = dict(
    objective="huber",
    alpha=0.3,  # Huber threshold on log(rpm): residuals beyond ~35% are down-weighted
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
    random_state=42,
)


LINEAR_COLS = ["market_index", "quarter_index", "quarter_end_ramp", "log_distance",
               "log_distance_sq", "weight_10k", "quote_signal", "is_flatbed", "is_reefer"]


def linear_features(df):
    X = pd.DataFrame(index=df.index)
    X["market_index"] = df["market_index"]
    X["quarter_index"] = quarter_index(df["date"])
    # 0 until 6 weeks before quarter end, then rising linearly to 1 on the last day.
    X["quarter_end_ramp"] = (1 - days_to_quarter_end(df["date"]) / 42).clip(lower=0)
    X["log_distance"] = np.log(df["distance"])
    X["log_distance_sq"] = X["log_distance"] ** 2
    X["weight_10k"] = df["weight"].fillna(31500) / 1e4  # median weight
    X["quote_signal"] = df["quote_signal"]
    X["is_flatbed"] = (df["equipment"] == "Flatbed").astype(float)
    X["is_reefer"] = (df["equipment"] == "Reefer").astype(float)
    return X[LINEAR_COLS]


class RateModel:
    """Predicts log(rate per mile); predictions are converted back to dollars.

    mode="lgbm":   LightGBM on all features.
    mode="linear": linear regression on LINEAR_COLS only.
    mode="hybrid": linear regression first, then LightGBM on its residuals using every
                   feature except market_index. The linear stage owns the effects that
                   must extrapolate: the market_index slope and the quarter-on-quarter
                   step-up in base rates (Nov-Dec sit in a quarter never seen in training).
                   The trees then learn lane/geometry/equipment and within-quarter effects,
                   whose feature ranges are fully covered by the training data.
    """

    def __init__(self, mode="hybrid", params=None):
        self.mode = mode
        self.params = {**LGB_PARAMS, **(params or {})}

    def _tree_features(self, df):
        X = base_features(df)
        return X.drop(columns="market_index") if self.mode == "hybrid" else X

    def fit(self, df):
        y = np.log(df[TARGET] / df["distance"]).to_numpy()
        self.linear = None
        if self.mode in ("linear", "hybrid"):
            self.linear = HuberRegressor(epsilon=1.35, max_iter=500).fit(linear_features(df), y)
            y = y - self.linear.predict(linear_features(df))
        self.model = None
        if self.mode in ("lgbm", "hybrid"):
            X = self._tree_features(df)
            self.feature_names = list(X.columns)
            self.model = lgb.LGBMRegressor(**self.params).fit(X, y)
        return self

    def predict(self, df):
        log_rpm = np.zeros(len(df))
        if self.linear is not None:
            log_rpm += self.linear.predict(linear_features(df))
        if self.model is not None:
            log_rpm += self.model.predict(self._tree_features(df)[self.feature_names])
        return np.exp(log_rpm) * df["distance"].to_numpy()


class BaselineModel:
    """Median rate-per-mile by equipment x distance band, times distance."""

    BANDS = [0, 250, 500, 1000, 1500, 2000, np.inf]

    def fit(self, df):
        rpm = df[TARGET] / df["distance"]
        self.table = rpm.groupby([df["equipment"], pd.cut(df["distance"], self.BANDS)], observed=True).median()
        self.global_ = float(rpm.median())
        return self

    def predict(self, df):
        idx = pd.MultiIndex.from_arrays([df["equipment"], pd.cut(df["distance"], self.BANDS)])
        rpm = self.table.reindex(idx).fillna(self.global_).to_numpy()
        return rpm * df["distance"].to_numpy()


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err) / y_true) * 100),
        "n": int(len(y_true)),
    }
