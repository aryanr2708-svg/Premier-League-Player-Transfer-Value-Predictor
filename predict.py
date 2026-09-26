"""
Loads the trained model and predicts market values for every player in
data/player_features.csv (i.e. the full current PL squads), including those
with no ground-truth label. Saves results to data/predicted_values.csv.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from config import FEATURES_FILE, MODEL_FILE, DATA_DIR
from train_model import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def predict_all() -> pd.DataFrame:
    pipeline = joblib.load(MODEL_FILE)
    df = pd.read_csv(FEATURES_FILE, encoding="utf-8-sig")

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    df["predicted_market_value_eur"] = pipeline.predict(X)

    out_path = f"{DATA_DIR}/predicted_values.csv"
    cols = ["player_name", "team", "position_group", "age", "goals", "assists",
            "predicted_market_value_eur"]
    ranked = df[cols].sort_values("predicted_market_value_eur", ascending=False)
    ranked.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions for {len(df)} players to {out_path}")

    def _abbreviate(v: float) -> str:
        if v >= 1_000_000:
            return f"€{v / 1_000_000:.1f}M"
        if v >= 1_000:
            return f"€{v / 1_000:.0f}K"
        return f"€{v:.0f}"

    top_n = 20
    print(f"\nTop {top_n} predicted market values:\n")
    display = ranked.head(top_n).copy()
    display["predicted_market_value_eur"] = display["predicted_market_value_eur"].map(_abbreviate)
    print(display.to_string(index=False))

    _show_chart(ranked.head(top_n))

    return df


def _show_chart(top_df: pd.DataFrame) -> None:
    """Pops up a horizontal bar chart of the top predicted players in a window."""
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = top_df["player_name"] + " (" + top_df["team"] + ")"
    values_m = top_df["predicted_market_value_eur"] / 1_000_000

    ax.barh(labels[::-1], values_m[::-1], color="#3d5afe")
    ax.set_xlabel("Predicted market value (€M)")
    ax.set_title("Top predicted Premier League player market values")

    for i, v in enumerate(values_m[::-1]):
        ax.text(v, i, f" €{v:.1f}M", va="center")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    predict_all()
