
import os

FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "b3d2e492ff58412cb0a3dd5cfc82b2a6")

BASE_URL = "https://api.football-data.org/v4"

HEADERS = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}


COMPETITION_CODE = "PL"

SEASON = 2026

DATA_DIR = "data"
RAW_SQUADS_FILE = f"{DATA_DIR}/raw_squads.json"
RAW_MATCHES_FILE = f"{DATA_DIR}/raw_matches.json"
FEATURES_FILE = f"{DATA_DIR}/player_features.csv"
MARKET_VALUES_FILE = f"{DATA_DIR}/market_values.csv"  
TRAINING_FILE = f"{DATA_DIR}/training_data.csv"
MODEL_FILE = f"{DATA_DIR}/value_model.joblib"
