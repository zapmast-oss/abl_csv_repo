import pandas as pd
from pathlib import Path

ABL_LEAGUE_ID = 200
SEASON_YEAR = 1981
TEAM_ID = 12  # Chicago by default

def load_csv(name: str) -> pd.DataFrame:
    path = Path(name)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {name}")
    return pd.read_csv(path)

def format_ip(outs: int) -> str:
    innings = outs // 3
    remainder = outs % 3
    return f"{innings}.{remainder}"

def get_team_info(team_id: int) -> dict:
    teams = load_csv("teams.csv")
    t = teams[teams["team_id"] == team_id]
    if t.empty:
        raise ValueError(f"No team with team_id={team_id}")
    row = t.iloc[0]
    return {
        "team_id": team_id,
        "full_name": f"{row['name']} {row['nickname']}",
        "abbr": str(row["abbr"]),
    }

def get_team_record(team_id: int) -> dict:
    rec = load_csv("team_record.csv")
    t = rec[rec["team_id"] == team_id]
    if t.empty:
        return {"w": 0, "l": 0, "pct": 0, "gb": 0, "streak_phrase": "no record"}
    row = t.iloc[0]
    streak_raw = int(row["streak"])
    if streak_raw > 0:
        phrase = f"on a {streak_raw}-game winning streak"
    elif streak_raw < 0:
        phrase = f"on a {abs(streak_raw)}-game losing streak"
    else:
        phrase = "no current streak"
    return {
        "w": int(row["w"]),
        "l": int(row["l"]),
        "pct": float(row["pct"]),
        "gb": float(row["gb"]),
        "streak_phrase": phrase,
    }

def get_runs_for_team(team_id: int) -> dict:
    bat = load_csv("team_batting_stats.csv")
    pit = load_csv("team_pitching_stats.csv")
    bat = bat[(bat["league_id"] == ABL_LEAGUE_ID) & (bat["year"] == SEASON_YEAR)]
    pit = pit[(pit["league_id"] == ABL_LEAGUE_ID) & (pit["year"] == SEASON_YEAR)]
    b = bat[bat["team_id"] == team_id]
    p = pit[pit["team_id"] == team_id]
    if b.empty or p.empty:
        return {"rs": None, "ra": None, "rd": None}
    rs = int(b.iloc[0]["r"])
    ra = int(p.iloc[0]["r"])
    return {"rs": rs, "ra": ra, "rd": rs - ra}

def get_last10_for_team(team_id: int) -> dict:
    path = Path("abl_last10.csv")
    if not path.exists():
        return {"w": None, "l": None, "diff": None}
    df = pd.read_csv(path)
    t = df[df["team_id"] == team_id]
    if t.empty:
        return {"w": None, "l": None, "diff": None}
    row = t.iloc[0]
    return {"w": int(row["last10_w"]), "l": int(row["last10_l"]), "diff": int(row["last10_diff"])}

def get_top_hitters(team_id: int, limit: int = 3) -> pd.DataFrame:
    try:
        bat_games = load_csv("players_game_batting.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    bat = bat_games[
        (bat_games["league_id"] == ABL_LEAGUE_ID)
        & (bat_games["year"] == SEASON_YEAR)
        & (bat_games["team_id"] == team_id)
        & (bat_games["split_id"] == 0)
    ].copy()
    if bat.empty:
        return pd.DataFrame()

    bat_agg = bat.groupby(["player_id"], as_index=False).agg(
        pa=("pa", "sum"), ab=("ab", "sum"), h=("h", "sum"),
        d=("d", "sum"), t=("t", "sum"), hr=("hr", "sum"),
        rbi=("rbi", "sum"), bb=("bb", "sum"), sf=("sf", "sum"), hp=("hp", "sum")
    )
    bat_agg = bat_agg[bat_agg["pa"] >= 50]
    if bat_agg.empty:
        return pd.DataFrame()

    bat_agg["avg"] = (bat_agg["h"] / bat_agg["ab"]).round(3)
    obp = (bat_agg["h"] + bat_agg["bb"] + bat_agg["hp"]) / (
        bat_agg["ab"] + bat_agg["bb"] + bat_agg["hp"] + bat_agg["sf"]
    )
    bat_agg["obp"] = obp.round(3)
    singles = bat_agg["h"] - bat_agg["d"] - bat_agg["t"] - bat_agg["hr"]
    tb = singles + 2 * bat_agg["d"] + 3 * bat_agg["t"] + 4 * bat_agg["hr"]
    bat_agg["slg"] = (tb / bat_agg["ab"]).round(3)
    bat_agg["ops"] = (bat_agg["obp"] + bat_agg["slg"]).round(3)

    players = load_csv("players.csv")[["player_id", "first_name", "last_name"]]
    bat_agg = bat_agg.merge(players, on="player_id", how="left")
    bat_agg["player_name"] = (bat_agg["first_name"].fillna("") + " " + bat_agg["last_name"].fillna("")).str.strip()

    return bat_agg.sort_values(by=["ops", "hr", "rbi"], ascending=[False, False, False]).head(limit)

def get_top_pitchers(team_id: int, limit: int = 3) -> pd.DataFrame:
    try:
        pit_games = load_csv("players_game_pitching_stats.csv")
        games = load_csv("games.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    valid_ids = set(
        games[
            (games["league_id"] == ABL_LEAGUE_ID)
            & (games["game_type"] == 0)
            & (games["played"] == 1)
        ]["game_id"]
    )

    pit = pit_games[
        (pit_games["team_id"] == team_id)
        & (pit_games["league_id"] == ABL_LEAGUE_ID)
        & (pit_games["year"] == SEASON_YEAR)
        & (pit_games["game_id"].isin(valid_ids))
    ].copy()
    if pit.empty:
        return pd.DataFrame()

    pit_agg = pit.groupby("player_id", as_index=False).agg(
        outs=("outs", "sum"), w=("w", "sum"), l=("l", "sum"),
        s=("s", "sum"), er=("er", "sum"), ha=("ha", "sum"),
        bb=("bb", "sum"), k=("k", "sum")
    )
    pit_agg["ip"] = pit_agg["outs"] / 3.0
    pit_agg = pit_agg[pit_agg["ip"] >= 10.0]
    if pit_agg.empty:
        return pd.DataFrame()

    pit_agg["era"] = (pit_agg["er"] * 9) / pit_agg["ip"]
    pit_agg["whip"] = (pit_agg["bb"] + pit_agg["ha"]) / pit_agg["ip"]

    players = load_csv("players.csv")[["player_id", "first_name", "last_name"]]
    pit_agg = pit_agg.merge(players, on="player_id", how="left")
    pit_agg["player_name"] = (pit_agg["first_name"].fillna("") + " " + pit_agg["last_name"].fillna("")).str.strip()

    return pit_agg.sort_values(by=["era", "whip", "ip"], ascending=[True, True, False]).head(limit)

def main() -> None:
    info = get_team_info(TEAM_ID)
    rec = get_team_record(TEAM_ID)
    runs = get_runs_for_team(TEAM_ID)
    last10 = get_last10_for_team(TEAM_ID)
    hitters = get_top_hitters(TEAM_ID)
    pitchers = get_top_pitchers(TEAM_ID)

    print(f"=== ABL Team Report – {info['full_name']} ({info['abbr']}) ===\n")
    rd_text = "N/A" if runs["rd"] is None else f"{runs['rd']:+}"
    gb_text = "leading their division" if rec["gb"] == 0 else f"{rec['gb']:.1f} games back"
    last10_text = "N/A" if last10["w"] is None else f"{last10['w']}-{last10['l']}"
    print(f"Record: {rec['w']}-{rec['l']} ({rec['pct']:.3f}), {gb_text}, run diff {rd_text}, last 10 {last10_text}, {rec['streak_phrase']}.\n")

    print("Top bats:")
    if hitters.empty:
        print("  (No hitter stats available.)")
    else:
        for _, row in hitters.iterrows():
            avg = row.get("avg", 0)
            hr = int(row.get("hr", 0))
            rbi = int(row.get("rbi", 0))
            ops = row.get("ops", 0)
            pa = int(row.get("pa", 0))
            print(f"  {row['player_name']}: AVG {avg:.3f}, {hr} HR, {rbi} RBI, OPS {ops:.3f} (PA {pa}).")
    print()

    print("Top arms:")
    if pitchers.empty:
        print("  (No pitcher stats available.)")
    else:
        for _, row in pitchers.iterrows():
            ip_outs = int(row["ip"] * 3)
            ip_text = format_ip(ip_outs)
            save_text = f", {int(row['s'])} SV" if row["s"] > 0 else ""
            print(f"  {row['player_name']}: {int(row['w'])}-{int(row['l'])}{save_text}, {ip_text} IP, ERA {row['era']:.2f}, {int(row['k'])} K, WHIP {row['whip']:.2f}.")
    print()

if __name__ == "__main__":
    main()
