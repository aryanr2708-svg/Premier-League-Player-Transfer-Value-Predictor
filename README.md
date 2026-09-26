<<<<<<< HEAD
# Premier League Player Market Value Predictor

Predicts market values for current Premier League players using performance
stats pulled live from [football-data.org](https://www.football-data.org/)
and a scikit-learn regression model.

## Important caveat

**football-data.org does not provide market values.** It's a fixtures/stats
API (squads, positions, goals, assists, standings), not a valuation source.
No free API reliably publishes market values — the standard approach is to
use **Transfermarkt** data as the training label:

- Easiest: search Kaggle for "Transfermarkt player market value dataset" —
  several are kept reasonably up to date and are free to download.
- Alternative: scrape Transfermarkt squad value pages yourself (check their
  terms of use first).

Either way, export it to `data/market_values.csv` with at least these columns:

```
player_name,market_value_eur
Bukayo Saka,120000000
Cole Palmer,110000000
...
```

If your source has a `player_id` that happens to align with football-data.org
IDs, include it too — the script will prefer an exact ID match over fuzzy
name matching. (Note: raw Transfermarkt player IDs will *not* align with
football-data.org IDs — leave `player_id` out and let it fuzzy-match on name.)

### If you have a raw Transfermarkt bulk export

If your dataset looks like the common Kaggle Transfermarkt export — one row
per player with columns like `name`, `last_season`, `market_value_in_eur`,
`current_club_domestic_competition_id` — it covers many seasons at once,
including retired/inactive players, so it needs filtering before use. Run:

```bash
python convert_transfermarkt_export.py /path/to/raw_export.csv
```

This filters to the most recent season in the file, drops players with no
market value, and writes `data/market_values.csv` in the expected format.

## Setup

```bash
pip install requests pandas scikit-learn joblib

export FOOTBALL_DATA_API_KEY="your_key_here"   # free tier at football-data.org
```

## Pipeline

```bash
# 1. Pull squads, scorers, and match results from football-data.org
python data_fetcher.py

# 2. Turn raw JSON into a per-player feature table (data/player_features.csv)
python build_dataset.py

# 3. Put your market value labels at data/market_values.csv (see above),
#    then train the model
python train_model.py

# 4. Predict market values for every current PL player
python predict.py
```

Output: `data/predicted_values.csv`, ranked by predicted value.

## Features used

- Age (derived from date of birth)
- Position group (Goalkeeper / Defender / Midfielder / Forward)
- Nationality
- Team
- Goals, assists, penalties, matches played (from the scorers endpoint —
  note this endpoint only includes players with 1+ goal, so non-scorers get
  zeros; see "Known limitations")
- Team win rate, goals-for average, goals-against average (proxy for
  "plays for a strong team", which correlates with value)

## Known limitations

1. **Non-scoring players are underrepresented.** football-data.org's
   `/scorers` endpoint only returns players with at least one goal, so
   defenders and backup players will often show 0 goals/assists/appearances
   even if they played plenty of minutes. For a more accurate model, swap
   in a stats source with full appearance/minutes data per player (e.g.
   a paid football-data.org tier, or a stats API like API-Football).
2. **No minutes-played feature.** `playedMatches` from the scorers endpoint
   is a rough proxy, not actual minutes.
3. **Small training set.** Only ~500-600 PL players exist at a time, and
   you'll only be able to train on however many you can match to a market
   value source — a few hundred rows for a Random Forest is workable but
   not huge. Consider adding prior seasons' data to grow the dataset.
4. **Name matching is fragile.** The fuzzy match in `train_model.py` is a
   simple normalized-string match. For messier name formats (accents,
   suffixes, nicknames), consider adding `rapidfuzz` for proper fuzzy
   matching, or manually reconcile a mapping table for edge cases.

## Files

| File | Purpose |
|---|---|
| `config.py` | API key, endpoints, file paths |
| `data_fetcher.py` | Pulls raw JSON from football-data.org via `requests` |
| `build_dataset.py` | Builds `data/player_features.csv` |
| `train_model.py` | Merges features + your market values, trains + evaluates the model |
| `predict.py` | Predicts values for every current player, saves ranked CSV |
=======
# Premier-League-Player-Transfer-Value-Predictor
# Football Transfer Value Predictor

## About
Machine learning project that predicts football player
market values for Premier League and La Liga players.

## Dataset
Explain where the player statistics and market values came from.

## Features
- Age
- Position
- Goals
- Assists
- Appearances
- Minutes
- etc.

## Models
- Linear Regression
- Random Forest
- XGBoost (if you used it)

## Evaluation
Explain MAE / RMSE / R² results.

## How to Run
Explain how someone can install dependencies
and run the prediction script.

## Future Improvements
- More leagues
- More seasons
- Better features
- Web interface
>>>>>>> 023469f7963cc94f72ea39b4a5fb5c6e36b9f2ea
