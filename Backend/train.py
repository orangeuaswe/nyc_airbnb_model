import json
import os

import numpy as np
import tensorflow as tf
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import layers, models

from calibration import (
    conformal_quantile,
    empirical_coverage,
    log_ratio_scores,
    prediction_interval,
)
from clean import cleanFull
from config import (
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    CONFORMAL_ALPHA,
    CONFORMAL_PATH,
    DATA,
    FEATURES,
    KMEANS_PATH,
    LINEARMODEL_PATH,
    NEURAL_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
    RANDFOREST_PATH,
    RANDOM_STATE,
)
from metrics import regression_metrics

METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def split_by_listing(df):
    """Split into fit / calibration / test index arrays (64/16/20), grouped
    by listing id.

    ~5.7k listings appear in both the 2019 and 2024 snapshots. A plain random
    split would put the same listing (same location, room type, minimum
    nights) on both sides and inflate every metric. The calibration slice is
    used only to size the prediction interval, never to fit a model.
    """
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    rest_idx, test_idx = next(outer.split(df, groups=df["id"]))
    rest = df.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    fit_rel, cal_rel = next(inner.split(rest, groups=rest["id"]))
    return rest_idx[fit_rel], rest_idx[cal_rel], test_idx


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    df, km = cleanFull(DATA)
    dump(km, KMEANS_PATH)

    X = df[FEATURES]
    y = df["price"]

    train_idx, cal_idx, test_idx = split_by_listing(df)
    X_train, X_cal, X_test = X.iloc[train_idx], X.iloc[cal_idx], X.iloc[test_idx]
    y_train, y_cal, y_test = y.iloc[train_idx], y.iloc[cal_idx], y.iloc[test_idx]

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    X_train_p = pre.fit_transform(X_train)
    X_cal_p = pre.transform(X_cal)
    X_test_p = pre.transform(X_test)
    dump(pre, PREPROCESSOR_PATH)

    lr = LinearRegression().fit(X_train_p, y_train)
    dump(lr, LINEARMODEL_PATH)

    rf = RandomForestRegressor(
        n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X_train_p, y_train)
    dump(rf, RANDFOREST_PATH)

    nn = models.Sequential([
        layers.Input(shape=(X_train_p.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    nn.compile(optimizer="adam", loss="mse", metrics=["mae"])
    nn.fit(_dense(X_train_p), y_train.values, epochs=20, batch_size=256, verbose=1)
    nn.save(str(NEURAL_PATH))

    # Evaluate every model on the same held-out listings. The baseline
    # predicts the training median, so a model only counts if it beats that.
    test_preds = {
        "baseline_median": np.full(len(y_test), float(np.median(y_train))),
        "linear_regression": lr.predict(X_test_p),
        "random_forest": rf.predict(X_test_p),
        "neural_network": nn.predict(_dense(X_test_p), verbose=0).ravel(),
    }
    results = {name: regression_metrics(y_test, p) for name, p in test_preds.items()}

    # Split-conformal interval around the random forest (the model the API
    # serves): size it on the calibration slice, then check coverage on test.
    cal_scores = log_ratio_scores(y_cal, rf.predict(X_cal_p))
    q = conformal_quantile(cal_scores, CONFORMAL_ALPHA)
    low, high = prediction_interval(test_preds["random_forest"], q)
    coverage = empirical_coverage(y_test, low, high)
    coverage_by_borough = {
        borough: empirical_coverage(
            y_test[mask.values], low[mask.values], high[mask.values]
        )
        for borough, mask in (
            (b, X_test["neighbourhood_group"] == b)
            for b in sorted(X_test["neighbourhood_group"].unique())
        )
    }
    conformal = {
        "model": "random_forest",
        "alpha": CONFORMAL_ALPHA,
        "target_coverage": 1 - CONFORMAL_ALPHA,
        "log_ratio_quantile": q,
        "n_calibration": int(len(cal_idx)),
        "test_coverage": coverage,
        "test_coverage_by_borough": coverage_by_borough,
    }
    with open(CONFORMAL_PATH, "w") as f:
        json.dump(conformal, f, indent=2)

    report = {
        "n_train": int(len(train_idx)),
        "n_calibration": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "split": "64/16/20 fit/calibration/test, grouped by listing id",
        "metrics": results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nHeld-out test set: {len(test_idx)} listings "
          f"(fit on {len(train_idx)}, calibrated on {len(cal_idx)})")
    print(f"{'model':<20}{'MAE':>10}{'RMSE':>10}{'MedAE':>10}{'R2':>8}")
    for name, m in results.items():
        print(f"{name:<20}{m['mae']:>10.2f}{m['rmse']:>10.2f}"
              f"{m['median_ae']:>10.2f}{m['r2']:>8.3f}")
    print(f"\n{(1 - CONFORMAL_ALPHA):.0%} prediction interval: x/ {np.exp(q):.2f} around the "
          f"random forest estimate; coverage on test = {coverage:.1%}")
    for borough, c in coverage_by_borough.items():
        print(f"  {borough:<15}{c:.1%}")
    print(f"\nSaved {METRICS_PATH} and {CONFORMAL_PATH}")


if __name__ == "__main__":
    main()
