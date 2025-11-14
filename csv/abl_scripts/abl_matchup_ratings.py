"""Simple conference matchup ranking for the ABL based on available team stats."""

from datetime import timedelta
from pathlib import Path

import pandas as pd

from abl_config import RAW_CSV_ROOT

LEAGUE_ID = 200
TEAM_MIN, TEAM_MAX = 1, 24
DATA_DIR = RAW_CSV_ROOT
SAY_NOTES = False
DEFAULT_HR_PA = 0.04


def note(msg: str):
    if SAY_NOTES:
        print(msg)


def pick(df: pd.DataFrame, *names: str):
    """Return the first available column name among the candidates (case-insensitive)."""
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def filter_abl(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ABL league rows and the 24 ABL teams if those columns exist."""
    result = df
    league_col = pick(df, "league_id", "league", "lg_id")
    if league_col:
        result = result[result[league_col] == LEAGUE_ID]
    team_col = pick(df, "team_id", "teamid", "tid")
    if team_col:
        result = result[result[team_col].between(TEAM_MIN, TEAM_MAX)]
    return result


def read_csv_smart(*names: str):
    """Search DATA_DIR for a CSV by trying common naming variants."""
    cands = []
    for n in names:
        cands += [
            n,
            n.lower(),
            n.upper(),
            n.replace(" ", "_"),
            n.replace("_", " "),
            n.title(),
            n.capitalize(),
        ]
    seen = set()
    for c in cands:
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        p = DATA_DIR / c
        if p.exists():
            return pd.read_csv(p)
    if SAY_NOTES:
        print(f"[Note] Missing: {names}")
    return None


def print_table(df: pd.DataFrame):
    """Print the DataFrame in fixed-width monospace."""
    print(df.to_string(index=False))


def save(df: pd.DataFrame, csv_name: str, txt_name: str = None, text_block: str = None):
    """Save a CSV and optional text block."""
    df.to_csv(csv_name, index=False)
    if txt_name and text_block:
        Path(txt_name).write_text(text_block, encoding="utf-8")


def load_team_data():
    records = read_csv_smart(
        "team_record.csv",
        "team_records.csv",
        "Team_Record.csv",
        "Team Records.csv",
    )
    if records is None:
        raise FileNotFoundError("team_record.csv is required for standings.")
    records = filter_abl(records)
    team_col = pick(records, "team_id", "teamid", "tid")
    if not team_col:
        raise KeyError("No team_id column found in team_record.csv.")
    name_col = pick(records, "team_display", "team_name", "name", "nickname")
    teams_lookup: dict[int, dict[str, str]] = {}
    teams_csv = read_csv_smart("teams.csv", "Teams.csv")
    if teams_csv is not None:
        teams_csv = filter_abl(teams_csv)
        team_name_col = pick(teams_csv, "team_display", "team_name", "name", "nickname")
        id_col = pick(teams_csv, "team_id", "teamid", "tid")
        sub_col = pick(teams_csv, "sub_league_id", "sub_league", "sub_leag", "sub_leagueid")
        division_col = pick(
            teams_csv, "division_id", "division", "divisionid", "div_id"
        )
        if id_col:
            for row in teams_csv.itertuples():
                row_d = row._asdict()
                tid_val = int(row_d[id_col])
                entry: dict[str, str] = {}
                if team_name_col and row_d.get(team_name_col):
                    entry["name"] = row_d[team_name_col]
                if sub_col in row_d and row_d[sub_col] is not None:
                    entry["conference"] = "NBC" if int(row_d[sub_col]) == 0 else "ABC"
                if division_col in row_d and row_d[division_col] is not None:
                    entry["division"] = str(row_d[division_col])
                if entry:
                    teams_lookup[tid_val] = entry
    conf_col = pick(records, "conference", "conf")
    div_col = pick(records, "division", "div")
    fallback_conf = False
    stats = []
    note_once = False
    for row in records.itertuples():
        row_dict = row._asdict()
        tid = int(row_dict[team_col])
        raw_name = row_dict.get(name_col) if name_col else None
        team_name = teams_lookup.get(tid, {}).get("name") or raw_name or f"Team {tid}"
        record_conf = row_dict.get(conf_col, "")
        record_div = row_dict.get(div_col, "")
        conf = (
            record_conf.strip() if isinstance(record_conf, str) else record_conf
        ) or ""
        div = (
            record_div.strip() if isinstance(record_div, str) else record_div
        ) or ""
        entry = teams_lookup.get(tid, {})
        conf = conf or entry.get("conference") or ("NBC" if tid <= 12 else "ABC")
        div = div or entry.get("division") or (chr(65 + ((tid - 1) % 12) // 4))
        if (not conf or not div) and not fallback_conf and not note_once:
            fallback_conf = True
            note_once = True
            note("[Note] Fallback conference/division mapping used.")
        w_col = pick(records, "w", "wins")
        l_col = pick(records, "l", "losses")
        if not w_col or not l_col:
            raise KeyError("Win/loss columns missing from team_record.csv.")
        w = float(row_dict[w_col])
        l = float(row_dict[l_col])
        act_wpct = w / (w + l) if (w + l) > 0 else 0.0
        rs_col = pick(records, "runs_scored", "rs", "r")
        ra_col = pick(records, "runs_against", "ra")
        rs = float(row_dict[rs_col]) if rs_col and row_dict.get(rs_col) is not None else None
        ra = float(row_dict[ra_col]) if ra_col and row_dict.get(ra_col) is not None else None
        py_wpct = act_wpct
        if rs is not None and ra is not None and (rs + ra) > 0:
            denom = rs**2 + ra**2
            if denom > 0:
                py_wpct = (rs**2) / denom
        else:
            note("[Note] Pythag fallback to actual wpct.")
        l10_wpct = act_wpct
        l10w_col = pick(records, "last10_w", "l10_w", "l10w")
        l10l_col = pick(records, "last10_l", "l10_l", "l10l")
        l10_col = pick(records, "last10", "last_10", "l10")
        if l10w_col and l10l_col:
            lw = float(row_dict[l10w_col]) if row_dict.get(l10w_col) is not None else 0.0
            ll = float(row_dict[l10l_col]) if row_dict.get(l10l_col) is not None else 0.0
            total = lw + ll
            l10_wpct = lw / total if total > 0 else act_wpct
        elif l10_col:
            val = str(row_dict[l10_col])
            if "-" in val:
                parts = val.split("-")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    lw = float(parts[0])
                    ll = float(parts[1])
                    total = lw + ll
                    l10_wpct = lw / total if total > 0 else act_wpct
                else:
                    note("[Note] l10 format unexpected; using season wpct.")
            else:
                note("[Note] l10 format missing dash; using season wpct.")
        else:
            note("[Note] l10 not available; using season wpct.")
        stats.append(
            {
                "team_id": tid,
                "team_name": team_name,
                "conference": conf,
                "division": div,
                "act_wpct": act_wpct,
                "py_wpct": py_wpct,
                "l10_wpct": l10_wpct,
                "rs": rs,
                "ra": ra,
            }
        )
    return pd.DataFrame(stats)


def load_batting_data():
    return read_csv_smart(
        "team_batting.csv", "Team_Batting.csv", "Team Batting.csv"
    )


def calculate_hr_pa(teams_df: pd.DataFrame, batting_df: pd.DataFrame):
    if batting_df is None:
        teams_df["hr_pa"] = DEFAULT_HR_PA
        return DEFAULT_HR_PA
    hr_col = pick(batting_df, "hr", "home_runs")
    components = {
        "ab": pick(batting_df, "ab", "at_bats"),
        "bb": pick(batting_df, "bb", "bases_on_balls"),
        "hbp": pick(batting_df, "hbp", "hit_by_pitch"),
        "sf": pick(batting_df, "sf", "sac_fly"),
        "sh": pick(batting_df, "sh", "sac_bunt"),
    }
    hr_pa_map = {}
    for row in batting_df.itertuples():
        row_dict = row._asdict()
        tid = int(row_dict[pick(batting_df, "team_id", "teamid", "tid")])
        hr = float(row_dict[hr_col]) if hr_col and row_dict.get(hr_col) is not None else 0.0
        pa = sum(
            float(row_dict[components[c]])
            if components[c] and row_dict.get(components[c]) is not None
            else 0.0
            for c in components
        )
        hr_pa_map[tid] = hr / max(pa, 1.0)
    league_mean = sum(hr_pa_map.values()) / max(len(hr_pa_map), 1)
    league_mean = max(league_mean, DEFAULT_HR_PA)
    teams_df["hr_pa"] = teams_df["team_id"].map(lambda tid: hr_pa_map.get(tid, league_mean))
    return league_mean


def load_schedule():
    schedule = read_csv_smart(
        "schedule.csv",
        "Schedule.csv",
        "games.csv",
        "Games.csv",
    )
    if schedule is None:
        return None
    schedule = filter_abl(schedule)
    for col in ("date", "game_date"):
        if col in schedule.columns:
            schedule["date"] = pd.to_datetime(schedule[col], errors="coerce", format="%Y-%m-%d")
            break
    else:
        schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce", format="%Y-%m-%d")
    return schedule


def find_next_unplayed_week(sched: pd.DataFrame):
    if sched is None or sched.empty:
        return None
    played_col = pick(sched, "played", "completed")
    if played_col:
        played_mask = sched[played_col].astype(int) != 0
    else:
        played_mask = pd.Series(False, index=sched.index)
        handled = False
        score_pairs = [
            ("runs0", "runs1"),
            ("score0", "score1"),
            ("away_score", "home_score"),
            ("home_score", "away_score"),
        ]
        for a_col, b_col in score_pairs:
            if a_col in sched.columns and b_col in sched.columns:
                handled = True
                a_vals = pd.to_numeric(sched[a_col], errors="coerce")
                b_vals = pd.to_numeric(sched[b_col], errors="coerce")
                pair_played = (
                    a_vals.notna()
                    & b_vals.notna()
                    & ((a_vals + b_vals) > 0)
                )
                played_mask = played_mask | pair_played
        if not handled:
            score_cols = [c for c in sched.columns if "score" in c.lower()]
            if score_cols:
                temp_mask = pd.Series(True, index=sched.index)
                for col in score_cols:
                    vals = pd.to_numeric(sched[col], errors="coerce")
                    temp_mask &= vals.notna()
                played_mask = temp_mask
    unplayed = sched[~played_mask].copy()
    if unplayed.empty:
        return None
    next_date = unplayed["date"].min()
    if pd.isna(next_date):
        return None
    monday = next_date - timedelta(days=next_date.weekday())
    return monday + timedelta(days=6)


def build_matchups(teams_df: pd.DataFrame, pairings: pd.DataFrame, league_hr_pa: float):
    rows = []
    columns = [
        "game_date",
        "conference",
        "same_div",
        "team_id_A",
        "team_name_A",
        "team_id_B",
        "team_name_B",
        "act_wpct_A",
        "act_wpct_B",
        "py_wpct_A",
        "py_wpct_B",
        "l10_wpct_A",
        "l10_wpct_B",
        "hr_pa_A",
        "hr_pa_B",
        "matchup_score",
    ]
    if pairings is None or pairings.empty:
        return pd.DataFrame(columns=columns)
    for _, game in pairings.iterrows():
        a_id = int(game["away_team"])
        h_id = int(game["home_team"])
        a = teams_df[teams_df["team_id"] == a_id]
        b = teams_df[teams_df["team_id"] == h_id]
        if a.empty or b.empty:
            continue
        row_a = a.iloc[0]
        row_b = b.iloc[0]
        if row_a["conference"] != row_b["conference"]:
            continue
        wpct_diff = abs(row_a["act_wpct"] - row_b["act_wpct"])
        closeness = 1 - min(wpct_diff, 0.300) / 0.300
        same_div = row_a["division"] == row_b["division"]
        where = 1.0 if same_div else 0.4
        div_leverage = 20 * where * closeness
        l10_diff = abs(row_a["l10_wpct"] - row_b["l10_wpct"])
        streak = 10 * (1 - min(l10_diff, 0.700) / 0.700)
        stand_swing = 10 * (1 - min(wpct_diff, 0.300) / 0.300)
        py_diff = abs(row_a["py_wpct"] - row_b["py_wpct"])
        true_parity = 20 * (1 - min(py_diff, 0.300) / 0.300)
        env_raw = ((row_a["hr_pa"] + row_b["hr_pa"]) / 2) / max(league_hr_pa, 1e-9)
        env = 15 * min(env_raw, 1.4) / 1.4
        recent_parity = 15 * (1 - min(l10_diff, 0.700) / 0.700)
        practical = 10
        score = div_leverage + streak + stand_swing + true_parity + env + recent_parity + practical
        rows.append(
            {
                "game_date": game["date"].strftime("%Y-%m-%d"),
                "conference": row_a["conference"],
                "same_div": same_div,
                "team_id_A": a_id,
                "team_name_A": row_a["team_name"],
                "team_id_B": h_id,
                "team_name_B": row_b["team_name"],
                "act_wpct_A": row_a["act_wpct"],
                "act_wpct_B": row_b["act_wpct"],
                "py_wpct_A": row_a["py_wpct"],
                "py_wpct_B": row_b["py_wpct"],
                "l10_wpct_A": row_a["l10_wpct"],
                "l10_wpct_B": row_b["l10_wpct"],
                "hr_pa_A": row_a["hr_pa"],
                "hr_pa_B": row_b["hr_pa"],
                "matchup_score": max(0.0, min(score, 100.0)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def select_sunday_games(schedule: pd.DataFrame, sunday_date):
    if schedule is None or sunday_date is None:
        return pd.DataFrame()
    return schedule[
        (schedule["date"] == sunday_date)
        & schedule["away_team"].between(TEAM_MIN, TEAM_MAX)
        & schedule["home_team"].between(TEAM_MIN, TEAM_MAX)
    ].copy()


def pick_top_unique(matchups: pd.DataFrame, conference: str, top_n: int = 3):
    if matchups.empty or "conference" not in matchups.columns:
        return pd.DataFrame()
    subset = matchups[matchups["conference"] == conference]
    if subset.empty:
        return subset
    subset = subset.sort_values("matchup_score", ascending=False)
    rows = []
    seen = set()
    for _, row in subset.iterrows():
        key = tuple(sorted((int(row["team_id_A"]), int(row["team_id_B"]))))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        if len(rows) == top_n:
            break
    return pd.DataFrame(rows)


def main():
    team_df = load_team_data()
    batting_df = load_batting_data()
    league_hr_pa = calculate_hr_pa(team_df, batting_df)
    schedule = load_schedule()
    sunday_date = find_next_unplayed_week(schedule)
    sunday_games = select_sunday_games(schedule, sunday_date)
    matchup_df = build_matchups(team_df, sunday_games, league_hr_pa)
    if "conference" in matchup_df.columns and not matchup_df.empty:
        matchup_df = matchup_df.sort_values(
            ["conference", "matchup_score"], ascending=[True, False]
        )
    summary_lines = []
    for conference, title in (
        ("NBC", "Sunday NBC Feature Options"),
        ("ABC", "Sunday ABC Feature Options"),
    ):
        subset = pick_top_unique(matchup_df, conference, top_n=3)
        if not subset.empty:
            print(f"=== {title} ===")
            print_table(subset)
            summary_lines.append(f"=== {title} ===\n{subset.to_string(index=False)}")
    summary_text = "\n\n".join(summary_lines) if summary_lines else "No Sunday feature options available."
    save(
        matchup_df,
        "abl_sunday_matchups.csv",
        "abl_sunday_matchups.txt",
        summary_text,
    )

    print("abl_sunday_matchups.csv written; check console for feature options.")


if __name__ == "__main__":
    main()
