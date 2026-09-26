"""
Turns the raw cached JSON (squads, scorers, matches) into one row-per-player
feature table, saved as data/player_features.csv.

Features engineered:
  - age (from dateOfBirth)
  - position (raw string from API, e.g. "Centre-Forward")
  - position_group (Goalkeeper / Defender / Midfielder / Forward, simplified)
  - nationality
  - team, team_id
  - goals, assists, penalties, matches_played  (from scorers endpoint; 0 if not a scorer)
  - team_win_rate, team_goals_for_avg, team_goals_against_avg (from matches, as a proxy
    for "plays for a strong/weak team", which correlates with market value)
"""

import json
import os
from datetime import date

import pandas as pd

from config import DATA_DIR, RAW_SQUADS_FILE, RAW_MATCHES_FILE, FEATURES_FILE

POSITION_GROUP_MAP = {
    "Goalkeeper": "Goalkeeper",
    "Centre-Back": "Defender",
    "Left-Back": "Defender",
    "Right-Back": "Defender",
    "Defence": "Defender",
    "Defender": "Defender",
    "Defensive Midfield": "Midfielder",
    "Central Midfield": "Midfielder",
    "Attacking Midfield": "Midfielder",
    "Left Midfield": "Midfielder",
    "Right Midfield": "Midfielder",
    "Midfield": "Midfielder",
    "Midfielder": "Midfielder",
    "Left Winger": "Forward",
    "Right Winger": "Forward",
    "Centre-Forward": "Forward",
    "Offence": "Forward",
    "Forward": "Forward",
    "Attacker": "Forward",
}


def _age_from_dob(dob_str: str | None) -> float | None:
    if not dob_str:
        return None
    try:
        dob = date.fromisoformat(dob_str[:10])
    except ValueError:
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _team_strength_table(matches: list[dict]) -> pd.DataFrame:
    """Aggregates finished matches into per-team win rate and goal averages."""
    rows = []
    for m in matches:
        home = m["homeTeam"]
        away = m["awayTeam"]
        score = m.get("score", {}).get("fullTime", {})
        hg, ag = score.get("home"), score.get("away")
        if hg is None or ag is None:
            continue
        rows.append({"team_id": home["id"], "team_name": home["name"], "gf": hg, "ga": ag,
                      "win": int(hg > ag), "draw": int(hg == ag)})
        rows.append({"team_id": away["id"], "team_name": away["name"], "gf": ag, "ga": hg,
                      "win": int(ag > hg), "draw": int(hg == ag)})

    if not rows:
        return pd.DataFrame(columns=["team_id", "team_win_rate", "team_goals_for_avg", "team_goals_against_avg"])

    df = pd.DataFrame(rows)
    agg = df.groupby("team_id").agg(
        matches=("win", "count"),
        wins=("win", "sum"),
        draws=("draw", "sum"),
        goals_for=("gf", "sum"),
        goals_against=("ga", "sum"),
    ).reset_index()
    agg["team_win_rate"] = (agg["wins"] + 0.5 * agg["draws"]) / agg["matches"]
    agg["team_goals_for_avg"] = agg["goals_for"] / agg["matches"]
    agg["team_goals_against_avg"] = agg["goals_against"] / agg["matches"]
    return agg[["team_id", "team_win_rate", "team_goals_for_avg", "team_goals_against_avg"]]


def build_player_features() -> pd.DataFrame:
    with open(RAW_SQUADS_FILE) as f:
        teams = json.load(f)

    scorers_path = f"{DATA_DIR}/raw_scorers.json"
    scorers_by_player_id = {}
    if os.path.exists(scorers_path):
        with open(scorers_path) as f:
            scorers = json.load(f)
        for s in scorers:
            pid = s["player"]["id"]
            scorers_by_player_id[pid] = {
                "goals": s.get("goals", 0) or 0,
                "assists": s.get("assists", 0) or 0,
                "penalties": s.get("penalties", 0) or 0,
                "matches_played": s.get("playedMatches", 0) or 0,
            }

    matches = []
    if os.path.exists(RAW_MATCHES_FILE):
        with open(RAW_MATCHES_FILE) as f:
            matches = json.load(f)
    strength = _team_strength_table(matches)

    rows = []
    for team in teams:
        team_id = team["id"]
        team_name = team["name"]
        for p in team.get("squad", []):
            pid = p["id"]
            perf = scorers_by_player_id.get(pid, {"goals": 0, "assists": 0, "penalties": 0, "matches_played": 0})
            raw_pos = p.get("position") or "Unknown"
            rows.append({
                "player_id": pid,
                "player_name": p.get("name"),
                "team_id": team_id,
                "team": team_name,
                "nationality": p.get("nationality"),
                "date_of_birth": p.get("dateOfBirth"),
                "age": _age_from_dob(p.get("dateOfBirth")),
                "position": raw_pos,
                "position_group": POSITION_GROUP_MAP.get(raw_pos, "Unknown"),
                **perf,
            })

    df = pd.DataFrame(rows)
    if not strength.empty:
        df = df.merge(strength, on="team_id", how="left")
    else:
        df["team_win_rate"] = None
        df["team_goals_for_avg"] = None
        df["team_goals_against_avg"] = None

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(FEATURES_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} player rows to {FEATURES_FILE}")
    return df


if __name__ == "__main__":
    build_player_features()
