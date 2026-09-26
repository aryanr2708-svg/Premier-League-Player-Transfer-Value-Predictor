"""
Configuration for the Premier League Market Value Predictor.

Get a free API key at https://www.football-data.org/client/register
Free tier limits: 10 requests/minute, access to Premier League (competition code "PL").
"""

import os

# Prefer environment variable; fall back to placeholder.
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "b3d2e492ff58412cb0a3dd5cfc82b2a6")

BASE_URL = "https://api.football-data.org/v4"

HEADERS = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}

# Premier League competition code on football-data.org
COMPETITION_CODE = "PL"

# Season start year, e.g. 2026 means the 2026/27 season
SEASON = 2026

# Where raw/processed data gets cached locally
DATA_DIR = "data"
RAW_SQUADS_FILE = f"{DATA_DIR}/raw_squads.json"
RAW_MATCHES_FILE = f"{DATA_DIR}/raw_matches.json"
FEATURES_FILE = f"{DATA_DIR}/player_features.csv"
MARKET_VALUES_FILE = f"{DATA_DIR}/market_values.csv"  # YOU must supply this (see README)
TRAINING_FILE = f"{DATA_DIR}/training_data.csv"
MODEL_FILE = f"{DATA_DIR}/value_model.joblib"
