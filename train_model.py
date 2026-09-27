import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import FEATURES_FILE, MARKET_VALUES_FILE, TRAINING_FILE, MODEL_FILE, DATA_DIR

NUMERIC_FEATURES = [
    "age", "goals", "assists", "penalties", "matches_played",
    "team_win_rate", "team_goals_for_avg", "team_goals_against_avg",
]
CATEGORICAL_FEATURES = ["position_group", "nationality", "team"]
TARGET = "market_value_eur"


def _normalize_name(name: str) -> str:
    import unicodedata
    stripped = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return "".join(ch.lower() for ch in stripped if ch.isalnum())


def merge_features_and_values() -> pd.DataFrame:
    features = pd.read_csv(FEATURES_FILE, encoding="utf-8-sig")
    values = pd.read_csv(MARKET_VALUES_FILE, encoding="utf-8-sig")

    if "player_id" in values.columns:
        merged = features.merge(values[["player_id", TARGET]], on="player_id", how="inner")
    else:
        features["_norm_name"] = features["player_name"].map(_normalize_name)
        values["_norm_name"] = values["player_name"].map(_normalize_name)
        merged = features.merge(values[["_norm_name", TARGET]], on="_norm_name", how="inner")
        merged = merged.drop(columns=["_norm_name"])

    print(f"Matched {len(merged)} / {len(features)} players to a market value.")
    os.makedirs(DATA_DIR, exist_ok=True)
    merged.to_csv(TRAINING_FILE, index=False, encoding="utf-8-sig")
    return merged


def train(model_type: str = "random_forest") -> None:
    df = merge_features_and_values()
    df = df.dropna(subset=[TARGET])

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")

    if model_type == "gradient_boosting":
        regressor = GradientBoostingRegressor(random_state=42)
    else:
        regressor = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", regressor),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE: €{mae:,.0f}")
    print(f"R^2: {r2:.3f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Saved trained model to {MODEL_FILE}")


if __name__ == "__main__":
    train()
