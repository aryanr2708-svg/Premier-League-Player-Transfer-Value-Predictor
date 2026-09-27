
import json
import os
import time
import requests

from config import BASE_URL, HEADERS, COMPETITION_CODE, DATA_DIR, RAW_SQUADS_FILE, RAW_MATCHES_FILE


def _get(endpoint: str, params: dict | None = None) -> dict:
    """GET a football-data.org endpoint with basic error handling and rate-limit backoff."""
    url = f"{BASE_URL}{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)

    if resp.status_code == 429:
        # Free tier rate limit hit; wait and retry once.
        print("Rate limited, waiting 60s...")
        time.sleep(60)
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)

    if not resp.ok:
        print(f"API error {resp.status_code} for {url}")
        print(f"Response body: {resp.text}")
    resp.raise_for_status()
    return resp.json()


def fetch_teams_and_squads() -> list[dict]:
    """
    Returns a list of team dicts, each containing a 'squad' list of players with
    fields like id, name, position, dateOfBirth, nationality. Relies on the API's
    "current season" default — this account's tier rejects an explicit season param.
    """
    data = _get(f"/competitions/{COMPETITION_CODE}/teams")
    teams = data.get("teams", [])
    return teams


def fetch_scorers(limit: int = 100) -> list[dict]:
    """
    Returns top scorers for the competition: player, team, goals, assists, penalties,
    playedMatches. Note: football-data.org only returns players with at least 1 goal
    on this endpoint, so it won't cover the full squad.
    """
    data = _get(f"/competitions/{COMPETITION_CODE}/scorers", params={"limit": limit})
    return data.get("scorers", [])


def fetch_matches(status: str = "FINISHED") -> list[dict]:
    """Returns matches for the competition (used to derive team strength)."""
    data = _get(f"/competitions/{COMPETITION_CODE}/matches", params={"status": status})
    return data.get("matches", [])


def fetch_all_and_cache() -> None:
    """Fetches teams/squads and matches, caching them to disk as JSON so you don't
    burn API calls re-running the pipeline while iterating on features."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching teams and squads...")
    teams = fetch_teams_and_squads()
    with open(RAW_SQUADS_FILE, "w") as f:
        json.dump(teams, f)
    print(f"  Saved {len(teams)} teams to {RAW_SQUADS_FILE}")

    time.sleep(6)  # stay under 10 req/min

    print("Fetching top scorers...")
    scorers = fetch_scorers()
    with open(f"{DATA_DIR}/raw_scorers.json", "w") as f:
        json.dump(scorers, f)
    print(f"  Saved {len(scorers)} scorer entries")

    time.sleep(6)

    print("Fetching finished matches...")
    matches = fetch_matches()
    with open(RAW_MATCHES_FILE, "w") as f:
        json.dump(matches, f)
    print(f"  Saved {len(matches)} matches to {RAW_MATCHES_FILE}")


if __name__ == "__main__":
    fetch_all_and_cache()
