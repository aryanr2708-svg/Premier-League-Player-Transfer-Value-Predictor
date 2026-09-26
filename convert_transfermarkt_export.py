"""
One-off converter for a Transfermarkt-style bulk export (the kind of CSV
commonly found on Kaggle, with columns like player_id, name, last_season,
current_club_domestic_competition_id, market_value_in_eur, etc.)

It filters down to players active in the most recent season present in the
file and with a non-null market value, then writes out the two-column
format train_model.py expects: player_name, market_value_eur.

Usage:
    python convert_transfermarkt_export.py /path/to/raw_export.csv
"""

import sys
import pandas as pd

from config import MARKET_VALUES_FILE


def convert(raw_path: str) -> None:
    df = pd.read_csv(raw_path, encoding="utf-8-sig")

    required = {"name", "market_value_in_eur", "last_season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns not found in {raw_path}: {missing}")

    latest_season = df["last_season"].max()
    print(f"Most recent season in file: {latest_season}")

    current = df[df["last_season"] == latest_season].copy()
    before = len(current)
    current = current.dropna(subset=["market_value_in_eur"])
    print(f"Filtered to {latest_season}: {before} players, {len(current)} with a market value")

    out = current[["name", "market_value_in_eur"]].rename(
        columns={"name": "player_name", "market_value_in_eur": "market_value_eur"}
    )

    out.to_csv(MARKET_VALUES_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out)} rows to {MARKET_VALUES_FILE}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_transfermarkt_export.py /path/to/raw_export.csv")
        sys.exit(1)
    convert(sys.argv[1])
