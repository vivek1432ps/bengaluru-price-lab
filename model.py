"""Train, compare and calibrate Bengaluru housing regressors."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "listings.csv.gz"
MODEL = ROOT / "model.joblib"
REPORT = ROOT / "evaluation.json"
FEATURES = ["locality", "sqft", "bhk", "bath"]
NUM = FEATURES[1:]


def prepare(source):
    """Exclude ambiguous units/ranges and nonphysical records, not large prices."""
    df = pd.read_csv(source)
    df = pd.DataFrame({
        "locality": df.location.str.strip().str.replace(r"\s+", " ", regex=True).str.casefold(),
        "sqft": pd.to_numeric(df.total_sqft, errors="coerce"),
        "bhk": pd.to_numeric(df["size"].str.extract(r"^(\d+)", expand=False), errors="coerce"),
        "bath": pd.to_numeric(df.bath, errors="coerce"),
        "price": pd.to_numeric(df.price, errors="coerce"),
    }).dropna()
    df = df[np.isfinite(df[["sqft", "bhk", "bath", "price"]]).all(axis=1)]
    df = df[df.locality.ne("") & df.sqft.between(250, 15000) &
            df.bhk.between(1, 10) & df.bath.between(1, 12) &
            (df.sqft >= 150 * df.bhk) & df.price.gt(0)]
    df = df.drop_duplicates().sort_values(FEATURES + ["price"])
    df.to_csv(DATA, index=False, compression={"method": "gzip", "mtime": 0})
    return len(df)


def split(df, seed=42):
    groups = pd.util.hash_pandas_object(df[FEATURES], index=False)
    rest, test = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed).split(df, groups=groups))
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed+1).split(rest, groups=groups.iloc[rest]))
    train, select = next(GroupShuffleSplit(n_splits=1, test_size=1/6, random_state=seed+2).split(rest[fit], groups=groups.iloc[rest[fit]]))
    return rest[fit][train], rest[fit][select], rest[val], test


def candidates():
    ridge = make_pipeline(ColumnTransformer([
        ("loc", OneHotEncoder(handle_unknown="ignore", min_frequency=8, sparse_output=False), ["locality"]),
        ("num", StandardScaler(), NUM),
    ]), Ridge(alpha=5))
    hgb = make_pipeline(ColumnTransformer([
        ("loc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, max_categories=180), ["locality"]),
        ("num", "passthrough", NUM),
    ]), HistGradientBoostingRegressor(categorical_features=[0], max_iter=230,
                                     max_leaf_nodes=25, learning_rate=.08, random_state=42))
    wrap = lambda estimator: TransformedTargetRegressor(regressor=estimator, func=np.log, inverse_func=np.exp)
    return {"Global median": DummyRegressor(strategy="median"), "Ridge": wrap(ridge),
            "Gradient boosting": wrap(hgb)}


def metrics(y, yhat):
    return {"mae_lakh": round(mean_absolute_error(y, yhat), 2),
            "rmse_lakh": round(mean_squared_error(y, yhat)**.5, 2),
            "r2": round(r2_score(y, yhat), 3)}


def train():
    df = pd.read_csv(DATA)
    train_idx, select_idx, cal_idx, test_idx = split(df)
    x, y = df[FEATURES], df.price
    models = candidates()
    selection = {}
    for name, estimator in models.items():
        estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
        selection[name] = mean_absolute_error(y.iloc[select_idx], estimator.predict(x.iloc[select_idx]))
    winner = min(selection, key=selection.get)
    fit_idx = np.r_[train_idx, select_idx]
    for clf in models.values():
        clf.fit(x.iloc[fit_idx], y.iloc[fit_idx])
    model = models[winner]
    scores = np.sort(np.abs(np.log(y.iloc[cal_idx].to_numpy() / model.predict(x.iloc[cal_idx]))))
    q = float(scores[min(len(scores), int(np.ceil((len(scores)+1)*.8)))-1])
    test_pred = model.predict(x.iloc[test_idx])
    fit = df.iloc[fit_idx]
    localities = fit.groupby("locality").agg(count=("price", "size"),
                  lo=("sqft", lambda s: s.quantile(.05)), hi=("sqft", lambda s: s.quantile(.95)),
                  median=("price", "median")).round(2)
    localities = localities[localities["count"] >= 12].to_dict("index")
    report = {"model": winner, "clean_rows": len(df), "fit_rows": len(fit_idx),
              "selection_rows": len(select_idx), "calibration_rows": len(cal_idx), "test_rows": len(test_idx),
              "localities": len(localities), "selection_mae_lakh": {k: round(v, 2) for k,v in selection.items()},
              "test_metrics": {k: metrics(y.iloc[test_idx], a.predict(x.iloc[test_idx])) for k,a in models.items()},
              "test_interval_coverage": round(float(np.mean(np.abs(np.log(y.iloc[test_idx]/test_pred)) <= q)), 3),
              "interval_target": .8}
    REPORT.write_text(json.dumps(report, indent=2)+"\n")
    joblib.dump((model, q, localities, report), MODEL)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, metavar="RAW_CSV")
    args = parser.parse_args()
    if args.prepare:
        print(f"Prepared {prepare(args.prepare):,} rows")
    print(json.dumps(train(), indent=2))
