import pandas as pd

LEAGUE_ID = 200

TEAMS_CSV = "teams.csv"
TEAM_RECORD_CSV = "team_record.csv"
LAST10_CSV = "abl_last10.csv"


def compute_streak_display(streak_value: int) -> str:
    """Return something like 'W5', 'L3', or '—' for 0."""
    if streak_value > 0:
        return f"W{int(streak_value)}"
    elif streak_value < 0:
        return f"L{abs(int(streak_value))}"
    else:
        return "—"


def compute_D_forum(streak_value: int) -> int:
    """
    Forum D term:

    D = (+1 if on winning streak OR -1 if on losing streak) *
        ROUND( ( (Streak length) - 1 ) / 2 )
    """
    if streak_value == 0:
        return 0

    sign = 1 if streak_value > 0 else -1
    length = abs(int(streak_value))
    # Python's round matches Excel's banker's rounding, which is fine here.
    return int(round((length - 1) / 2))


def main():
    # --- Load teams and limit to ABL (league_id = 200) ---
    teams = pd.read_csv(TEAMS_CSV)
    abl_teams = teams[teams["league_id"] == LEAGUE_ID][["team_id", "name", "nickname"]].copy()
    abl_teams["team_display"] = abl_teams["name"] + " " + abl_teams["nickname"]

    # --- Load team records (W, L, pct, streak, etc.) ---
    rec = pd.read_csv(TEAM_RECORD_CSV)

    # Keep only teams that are in our ABL team list
    rec = rec.merge(abl_teams[["team_id", "team_display"]], on="team_id", how="inner")

    # --- Load last 10 data ---
    last10 = pd.read_csv(LAST10_CSV)

    # Merge last10 onto records
    rec = rec.merge(last10[["team_id", "last10_w", "last10_l", "last10_diff"]], on="team_id", how="left")

    # If for some reason any team is missing last10 info, treat as 0
    rec[["last10_w", "last10_l", "last10_diff"]] = rec[["last10_w", "last10_l", "last10_diff"]].fillna(0)

    # Ensure pct is float
    rec["pct"] = rec["pct"].astype(float)

    # --- Compute forum power terms A, B, C, D ---

    # A term
    rec["A"] = 90

    # B term: ROUND(((Winning Pct) - 0.500) * 162 * 10/9)
    rec["B_raw"] = (rec["pct"] - 0.5) * 162 * (10.0 / 9.0)
    rec["B"] = rec["B_raw"].round().astype(int)

    # C term: (Wins in Last10 - Losses in Last10)
    rec["C"] = rec["last10_w"] - rec["last10_l"]

    # D term: streak-based
    rec["D"] = rec["streak"].apply(compute_D_forum).astype(int)

    # Total power score
    rec["power_total"] = rec["A"] + rec["B"] + rec["C"] + rec["D"]

    # Streak display like W5 / L3 / —
    rec["streak_display"] = rec["streak"].apply(compute_streak_display)

    # For convenience, build a nice Last10 string
    rec["last10_str"] = rec["last10_w"].astype(int).astype(str) + "-" + rec["last10_l"].astype(int).astype(str)

    # --- Sort by total power descending ---
    rec = rec.sort_values("power_total", ascending=False).reset_index(drop=True)

    # Add rank column
    rec["rank"] = rec.index + 1

    # Order columns for table output
    output_cols = [
        "rank",
        "team_display",
        "w",
        "l",
        "pct",
        "last10_str",
        "streak_display",
        "A",
        "B",
        "C",
        "D",
        "power_total",
    ]

    print("=== ABL Power Rankings (Forum A+B+C+D Formula) ===")
    print(rec[output_cols].to_string(index=False))

    # --- Broadcast-style lines for the top 10 teams ---
    print("\n--- Power Rankings (Broadcast lines) ---")
    top_n = rec.head(10)

    for _, row in top_n.iterrows():
        rank = int(row["rank"])
        name = row["team_display"]
        w = int(row["w"])
        l = int(row["l"])
        pct = float(row["pct"])
        last10 = row["last10_str"]
        streak_disp = row["streak_display"]
        A = int(row["A"])
        B = int(row["B"])
        C = int(row["C"])
        D = int(row["D"])
        total = int(row["power_total"])

        line = (
            f"#{rank} power: {name}: {w}-{l} ({pct:.3f}), last 10 {last10}, "
            f"streak {streak_disp}, A={A}, B={B}, C={C}, D={D}, total {total}."
        )
        print(line)


if __name__ == "__main__":
    main()
