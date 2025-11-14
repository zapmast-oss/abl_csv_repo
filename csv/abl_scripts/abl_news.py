import pandas as pd
import re

LEAGUE_ID = 200
YEAR = 1981

PLAYERS_CSV = "players.csv"
TEAMS_CSV = "teams.csv"
INJURIES_CSV = "players_injury_history.csv"
TRADES_CSV = "trade_history.csv"

# Match things like <Portland Lumberjacks:team#18> or <Noah White:player#8837>
TAG_RE = re.compile(r"<([^:>]+):[^>]+>")


def clean_summary(summary: str) -> str:
    """Remove OOTP markup tags like <Name:type#id> -> Name, and trim any broken tag at the end."""
    if not isinstance(summary, str):
        return ""

    # First: replace full tags like <Portland Lumberjacks:team#18> -> Portland Lumberjacks
    text = TAG_RE.sub(r"\1", summary)

    # If there's still a bare "<" (probably a truncated tag), cut the text at that point
    if "<" in text:
        text = text.split("<", 1)[0].rstrip()

    return text



def load_players_with_teams():
    """Return a DataFrame with player_id, player_name, team_display (ABL only)."""
    players = pd.read_csv(PLAYERS_CSV)
    teams = pd.read_csv(TEAMS_CSV)

    # Only ABL teams
    teams = teams[teams["league_id"] == LEAGUE_ID][
        ["team_id", "name", "nickname"]
    ].copy()
    teams["team_display"] = teams["name"] + " " + teams["nickname"]

    players = players.merge(teams[["team_id", "team_display"]], on="team_id", how="left")

    players["player_name"] = players["first_name"].fillna("") + " " + players[
        "last_name"
    ].fillna("")

    return players[["player_id", "player_name", "team_display"]]


def recent_trades(limit=5):
    """Load the most recent trades, returning a small DataFrame."""
    try:
        trades = pd.read_csv(TRADES_CSV)
    except FileNotFoundError:
        print("No trade_history.csv found.")
        return pd.DataFrame()

    if "date" not in trades.columns or "summary" not in trades.columns:
        print("trade_history.csv does not have expected columns.")
        return pd.DataFrame()

    # Parse dates and sort newest first
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce")
    trades = trades.dropna(subset=["date"])
    trades = trades.sort_values("date", ascending=False)

    # Keep just the most recent ones
    trades = trades.head(limit).copy()

    # Clean markup in summaries
    trades["summary_clean"] = trades["summary"].apply(clean_summary)

    return trades[["date", "summary_clean"]]


def recent_injuries(players_df, limit=5):
    """Load the most recent injuries with player names and teams (ABL only)."""
    try:
        inj = pd.read_csv(INJURIES_CSV)
    except FileNotFoundError:
        print("No players_injury_history.csv found.")
        return pd.DataFrame()

    if "date" not in inj.columns:
        print("players_injury_history.csv does not have a 'date' column.")
        return pd.DataFrame()

    inj["date"] = pd.to_datetime(inj["date"], errors="coerce")
    inj = inj.dropna(subset=["date"])

    # Merge with players to get names and ABL teams
    inj = inj.merge(players_df, on="player_id", how="left")

    # Keep only players who are on an ABL team (drop NaN team_display)
    inj = inj[~inj["team_display"].isna()].copy()

    # Sort newest first
    inj = inj.sort_values("date", ascending=False)

    # Keep most recent injuries
    inj = inj.head(limit).copy()

    # Clean length column (days)
    if "length" in inj.columns:
        inj["length"] = pd.to_numeric(inj["length"], errors="coerce")
    else:
        inj["length"] = pd.NA

    return inj


def print_news():
    players_df = load_players_with_teams()

    print("=== ABL News & Notes ===\n")

    # --- Recent Trades ---
    trades = recent_trades(limit=5)
    if trades.empty:
        print("No recent trades found.\n")
    else:
        print("== Recent Trades ==\n")
        for _, row in trades.iterrows():
            date_str = row["date"].date()
            summary = str(row["summary_clean"])
            print(f"{date_str}: {summary}")
        print()

    # --- Recent Injuries ---
    injuries = recent_injuries(players_df, limit=5)
    if injuries.empty:
        print("No recent injuries found.\n")
    else:
        print("== Recent Injuries ==\n")
        for _, row in injuries.iterrows():
            date_str = row["date"].date()
            name = row.get("player_name", "Unknown Player")
            team = row.get("team_display", "Unknown Team")
            length = row.get("length", pd.NA)

            if pd.isna(length):
                extra = "injury – duration unknown"
            else:
                extra = f"out {int(length)} days"

            print(f"{date_str}: {team} – {name} ({extra})")
        print()


if __name__ == "__main__":
    print_news()
