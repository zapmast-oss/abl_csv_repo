import pandas as pd

LEAGUE_ID = 200

TEAMS_CSV = "teams.csv"
TEAM_RECORD_CSV = "team_record.csv"

# Load data
teams = pd.read_csv(TEAMS_CSV)
team_record = pd.read_csv(TEAM_RECORD_CSV)

# ABL teams only
abl_teams = teams[teams["league_id"] == LEAGUE_ID][["team_id", "name", "nickname", "abbr"]].copy()
abl_team_ids = set(abl_teams["team_id"].tolist())

# Join team_record with team info
rec = team_record[team_record["team_id"].isin(abl_team_ids)].copy()

rec = rec.merge(
    abl_teams,
    on="team_id",
    how="left"
)

# Build a display name
rec["team_display"] = rec["abbr"].fillna("") + " " + rec["nickname"].fillna("")

# Parse streak: 'W5', 'L3', etc.
def parse_streak(val):
    """
    Interpret streak as an integer:
      >0 = winning streak of that length
      <0 = losing streak of that length (absolute value)
       0 = no streak
    """
    try:
        n = int(val)
    except (ValueError, TypeError):
        return None, 0

    if n > 0:
        return "W", n
    elif n < 0:
        return "L", -n
    else:
        return None, 0


rec["streak_type"], rec["streak_len"] = zip(*rec["streak"].map(parse_streak))

def format_streak(row):
    if row["streak_type"] == "W":
        return f"W{row['streak_len']}"
    elif row["streak_type"] == "L":
        return f"L{row['streak_len']}"
    else:
        return ""

rec["streak_display"] = rec.apply(format_streak, axis=1)


# Separate winners and losers
win_streaks = rec[rec["streak_type"] == "W"].copy()
lose_streaks = rec[rec["streak_type"] == "L"].copy()

# Sort: longest streak first
win_streaks = win_streaks.sort_values("streak_len", ascending=False)
lose_streaks = lose_streaks.sort_values("streak_len", ascending=False)

# Take top N
TOP_N = 5
top_win = win_streaks.head(TOP_N)
top_lose = lose_streaks.head(TOP_N)

# Table output
print("=== Top ABL Winning Streaks ===")
if top_win.empty:
    print("No winning streaks found.")
else:
    print(
    top_win[["team_display", "w", "l", "pct", "streak_display"]]
    .assign(pct=lambda df: df["pct"].round(3))
    .to_string(index=False)
)


print("\n=== Top ABL Losing Streaks ===")
if top_lose.empty:
    print("No losing streaks found.")
else:
    print(
    top_lose[["team_display", "w", "l", "pct", "streak_display"]]
    .assign(pct=lambda df: df["pct"].round(3))
    .to_string(index=False)
)



# Broadcast lines
print("\n--- Streaks (Broadcast lines) ---")
for _, row in top_win.iterrows():
    print(
        f"{row['team_display']}: {int(row['w'])}-{int(row['l'])} "
        f"({row['pct']:.3f}), on a {int(row['streak_len'])}-game winning streak"
    )

for _, row in top_lose.iterrows():
    print(
        f"{row['team_display']}: {int(row['w'])}-{int(row['l'])} "
        f"({row['pct']:.3f}), on a {int(row['streak_len'])}-game losing streak"
    )

