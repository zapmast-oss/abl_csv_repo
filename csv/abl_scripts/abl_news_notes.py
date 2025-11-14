import pandas as pd
from pathlib import Path

# === Config ===
MAX_TRADES = 5
MAX_INJURIES = 5


def safe_load_csv(name: str) -> pd.DataFrame | None:
    path = Path(name)
    if not path.exists():
        print(f"[File not found: {name}]")
        return None
    try:
        return pd.read_csv(path, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"[Failed to load {name}: {e}]")
        return None


def get_recent_trades() -> list[str]:
    df = safe_load_csv("messages.csv")
    if df is None or df.empty:
        return ["[No recent trades found.]"]

    if "date" not in df.columns or "subject" not in df.columns:
        return ["[messages.csv missing 'date' or 'subject']"]

    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["parsed_date"])
    df = df[df["subject"].astype(str).str.contains("trade", case=False, na=False)]
    df = df.sort_values("parsed_date", ascending=False)

    if df.empty:
        return ["[No recent trades found.]"]

    return [
        f"{row['parsed_date'].date()}: {row['subject']}"
        for _, row in df.head(MAX_TRADES).iterrows()
    ]


def get_team_lookup() -> dict[int, str]:
    df = safe_load_csv("teams.csv")
    if df is None or df.empty:
        return {}

    required = {"team_id", "name", "nickname"}
    if not required.issubset(df.columns):
        return {}

    return {
        int(row["team_id"]): f"{row['name']} {row['nickname']}"
        for _, row in df.iterrows()
        if pd.notna(row["team_id"])
    }


def get_player_lookup() -> dict[int, dict]:
    df = safe_load_csv("players.csv")
    if df is None or df.empty:
        return {}

    required = {"player_id", "first_name", "last_name", "team_id"}
    if not required.issubset(df.columns):
        return {}

    team_map = get_team_lookup()

    return {
        int(row["player_id"]): {
            "name": f"{row['first_name']} {row['last_name']}",
            "team": team_map.get(int(row["team_id"]), "Unknown Team"),
        }
        for _, row in df.iterrows()
        if pd.notna(row["player_id"])
    }


def get_recent_injuries() -> list[str]:
    df = safe_load_csv("injuries.csv")
    if df is None or df.empty:
        return ["[No recent injuries found.]"]

    required = {"player_id", "date", "days"}
    if not required.issubset(df.columns):
        return ["[injuries.csv missing required columns]"]

    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["parsed_date"])
    df = df.sort_values("parsed_date", ascending=False)

    players = get_player_lookup()

    results = []
    for _, row in df.head(MAX_INJURIES).iterrows():
        pid = int(row["player_id"])
        date = row["parsed_date"].strftime("%Y-%m-%d")
        days = int(row["days"]) if pd.notna(row["days"]) else "?"

        player = players.get(pid, {"name": f"Unknown #{pid}", "team": "Unknown Team"})
        results.append(f"{date}: {player['team']} – {player['name']} (out {days} days)")

    return results if results else ["[No recent injuries found.]"]


def main():
    print("=== ABL News & Notes ===\n")

    print("== Recent Trades ==\n")
    for line in get_recent_trades():
        print(line)

    print("\n== Recent Injuries ==\n")
    for line in get_recent_injuries():
        print(line)


if __name__ == "__main__":
    main()
