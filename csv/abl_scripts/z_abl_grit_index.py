"""ABL Grit Index: measure late-game resilience and walk-off heroics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24

LINESCORE_CANDIDATES = [
    "game_linescore.csv",
    "linescores.csv",
    "games_linescore.csv",
    "games_score.csv",
]
PBP_CANDIDATES = [
    "play_by_play.csv",
    "pbp.csv",
    "game_events.csv",
]
TEAM_LOG_CANDIDATES = [
    "team_game_log.csv",
    "teams_game_log.csv",
    "game_log_team.csv",
    "team_log.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]
GAMES_FILE = "games.csv"


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def read_first(base: Path, override: Optional[Path], candidates: Sequence[str]) -> Optional[pd.DataFrame]:
    if override:
        if not override.exists():
            raise FileNotFoundError(f"Specified file not found: {override}")
        return pd.read_csv(override)
    for name in candidates:
        path = base / name
        if path.exists():
            return pd.read_csv(path)
    return None


def load_team_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    display_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "conference_id")
    div_col = pick_column(df, "division_id", "division")
    display_map: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    if not team_col:
        return display_map, conf_map
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    for _, row in df.iterrows():
        tid = row.get(team_col)
        if pd.isna(tid):
            continue
        try:
            tid_int = int(tid)
        except (TypeError, ValueError):
            continue
        if not (TEAM_MIN <= tid_int <= TEAM_MAX):
            continue
        if display_col and pd.notna(row.get(display_col)):
            display_map[tid_int] = str(row.get(display_col))
        if tid_int in conf_map or not sub_col or not div_col:
            continue
        sub_val = row.get(sub_col)
        div_val = row.get(div_col)
        if pd.isna(sub_val) or pd.isna(div_val):
            continue
        try:
            sub_key = int(sub_val)
        except (TypeError, ValueError):
            sub_key = None
        try:
            div_key = int(div_val)
        except (TypeError, ValueError):
            div_key = None
        sub = conf_lookup.get(sub_key, str(sub_val)[0].upper())
        div = div_lookup.get(div_key, str(div_val)[0].upper())
        conf_map[tid_int] = f"{sub}-{div}"
    return display_map, conf_map


def load_games(base: Path) -> pd.DataFrame:
    path = base / GAMES_FILE
    if not path.exists():
        raise FileNotFoundError("games.csv is required for home/away mappings.")
    df = pd.read_csv(path)
    gid_col = pick_column(df, "game_id", "GameID")
    home_col = pick_column(df, "home_team", "home_team_id")
    away_col = pick_column(df, "away_team", "away_team_id")
    date_col = pick_column(df, "date", "game_date")
    played_col = pick_column(df, "played")
    if not gid_col or not home_col or not away_col:
        raise ValueError("games.csv must include game_id, home_team, and away_team columns.")
    data = df.copy()
    data = data[pd.notna(data[gid_col])]
    if played_col and played_col in data.columns:
        data = data[data[played_col] == 1]
    out = pd.DataFrame()
    out["game_id"] = pd.to_numeric(data[gid_col], errors="coerce").astype("Int64")
    out["home_team"] = pd.to_numeric(data[home_col], errors="coerce").astype("Int64")
    out["away_team"] = pd.to_numeric(data[away_col], errors="coerce").astype("Int64")
    if date_col:
        out["game_date"] = pd.to_datetime(data[date_col], errors="coerce")
    else:
        out["game_date"] = pd.NaT
    out = out.dropna(subset=["game_id"])
    return out


def normalize_linescore(df: pd.DataFrame) -> pd.DataFrame:
    gid_col = pick_column(df, "game_id", "GameID")
    if not gid_col:
        raise ValueError("Linescore data requires a game_id column.")
    team_col = pick_column(df, "team", "side", "team_flag")
    inning_col = pick_column(df, "inning", "inn", "inning_num")
    score_col = pick_column(df, "score", "runs", "run")
    if team_col and inning_col and score_col:
        data = df[[gid_col, team_col, inning_col, score_col]].copy()
        data.columns = ["game_id", "team_raw", "inning", "runs"]
        data["team_flag"] = data["team_raw"].apply(_normalize_team_flag)
        data["game_id"] = pd.to_numeric(data["game_id"], errors="coerce").astype("Int64")
        data["inning"] = pd.to_numeric(data["inning"], errors="coerce").astype("Int64")
        data["runs"] = pd.to_numeric(data["runs"], errors="coerce").fillna(0.0)
        data = data.dropna(subset=["team_flag", "inning"])
        data["team_flag"] = data["team_flag"].astype(int)
        return data[["game_id", "team_flag", "inning", "runs"]]
    return _normalize_wide_linescore(df, gid_col)


def _normalize_team_flag(value) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"home", "1", "h", "bottom"}:
            return 1
        if val in {"away", "0", "a", "top", "visitor", "vis"}:
            return 0
    try:
        num = int(value)
        if num in (0, 1):
            return num
    except (TypeError, ValueError):
        return None
    return None


def _normalize_wide_linescore(df: pd.DataFrame, gid_col: str) -> pd.DataFrame:
    pattern_home = [
        re.compile(r"h(\d+)$", re.IGNORECASE),
        re.compile(r"home(\d+)$", re.IGNORECASE),
        re.compile(r"inn(\d+)_home", re.IGNORECASE),
        re.compile(r"home_inn(\d+)", re.IGNORECASE),
    ]
    pattern_away = [
        re.compile(r"a(\d+)$", re.IGNORECASE),
        re.compile(r"away(\d+)$", re.IGNORECASE),
        re.compile(r"inn(\d+)_away", re.IGNORECASE),
        re.compile(r"away_inn(\d+)", re.IGNORECASE),
    ]
    records = []
    for _, row in df.iterrows():
        gid = row.get(gid_col)
        if pd.isna(gid):
            continue
        try:
            gid_int = int(gid)
        except (TypeError, ValueError):
            continue
        for col in df.columns:
            if col == gid_col:
                continue
            value = row.get(col)
            if pd.isna(value):
                continue
            inning = None
            flag = None
            col_lower = col.lower()
            for pattern in pattern_away:
                match = pattern.match(col_lower)
                if match:
                    inning = int(match.group(1))
                    flag = 0
                    break
            if inning is None:
                for pattern in pattern_home:
                    match = pattern.match(col_lower)
                    if match:
                        inning = int(match.group(1))
                        flag = 1
                        break
            if inning is None or flag is None:
                continue
            run_val = pd.to_numeric(value, errors="coerce")
            if pd.isna(run_val):
                run_val = 0.0
            records.append({"game_id": gid_int, "team_flag": flag, "inning": inning, "runs": float(run_val)})
    if not records:
        raise ValueError("Unable to parse linescore format.")
    return pd.DataFrame(records)


def linescore_from_pbp(df: pd.DataFrame, games: pd.DataFrame) -> Optional[pd.DataFrame]:
    gid_col = pick_column(df, "game_id", "GameID")
    inning_col = pick_column(df, "inning", "inn")
    half_col = pick_column(df, "half", "inning_half", "half_inning")
    team_col = pick_column(df, "batting_team_id", "team_id", "offense_team_id")
    runs_col = pick_column(df, "runs_scored", "runs", "result_runs")
    if not all([gid_col, inning_col, half_col, team_col, runs_col]):
        return None
    pbp = df[[gid_col, inning_col, half_col, team_col, runs_col]].copy()
    pbp.columns = ["game_id", "inning", "half", "bat_team_id", "runs"]
    pbp["game_id"] = pd.to_numeric(pbp["game_id"], errors="coerce").astype("Int64")
    pbp["inning"] = pd.to_numeric(pbp["inning"], errors="coerce").astype("Int64")
    pbp["bat_team_id"] = pd.to_numeric(pbp["bat_team_id"], errors="coerce").astype("Int64")
    pbp["runs"] = pd.to_numeric(pbp["runs"], errors="coerce").fillna(0.0)
    pbp = pbp.dropna(subset=["game_id", "inning", "bat_team_id"])
    home_map = games.set_index("game_id")["home_team"].to_dict()
    away_map = games.set_index("game_id")["away_team"].to_dict()
    def to_flag(row):
        gid = row["game_id"]
        tid = row["bat_team_id"]
        home = home_map.get(gid)
        away = away_map.get(gid)
        if tid == home:
            return 1
        if tid == away:
            return 0
        # Fall back to half indicator
        half_val = str(row["half"]).strip().lower() if pd.notna(row["half"]) else ""
        if half_val.startswith("top"):
            return 0
        if half_val.startswith("bot"):
            return 1
        return np.nan
    pbp["team_flag"] = pbp.apply(to_flag, axis=1)
    pbp = pbp.dropna(subset=["team_flag"])
    pbp["team_flag"] = pbp["team_flag"].astype(int)
    grouped = (
        pbp.groupby(["game_id", "team_flag", "inning"], as_index=False)["runs"].sum()
    )
    return grouped if not grouped.empty else None


def load_linescore(base: Path, override: Optional[Path], pbp_override: Optional[Path], games: pd.DataFrame) -> pd.DataFrame:
    df = read_first(base, override, LINESCORE_CANDIDATES)
    if df is not None:
        normalized = normalize_linescore(df)
        if not normalized.empty:
            return normalized
    pbp_df = read_first(base, pbp_override, PBP_CANDIDATES)
    if pbp_df is not None:
        derived = linescore_from_pbp(pbp_df, games)
        if derived is not None and not derived.empty:
            return derived
    raise FileNotFoundError("Unable to locate usable linescore or play-by-play data.")


def summarize_game(segment: pd.DataFrame) -> Optional[Dict]:
    if segment.empty:
        return None
    max_inning = int(segment["inning"].max())
    summary: Dict[int, Dict[str, float]] = {}
    for flag in (0, 1):
        subset = segment[segment["team_flag"] == flag]
        if subset.empty:
            return None
        total_runs = subset["runs"].sum()
        runs_thru6 = subset.loc[subset["inning"] <= 6, "runs"].sum()
        pre_final = subset.loc[subset["inning"] < max_inning, "runs"].sum()
        final_inning_runs = subset.loc[subset["inning"] == max_inning, "runs"].sum()
        summary[flag] = {
            "total_runs": total_runs,
            "runs_thru6": runs_thru6,
            "pre_final_runs": pre_final,
            "final_inning_runs": final_inning_runs,
        }
    home = summary[1]
    away = summary[0]
    home_won = home["total_runs"] > away["total_runs"]
    away_won = away["total_runs"] > home["total_runs"]
    home_walkoff = bool(
        home_won
        and home["final_inning_runs"] > 0
        and home["pre_final_runs"] <= away["total_runs"]
    )
    return {
        "max_inning": max_inning,
        "teams": summary,
        "home_walkoff": home_walkoff,
        "home_won": home_won,
        "away_won": away_won,
    }


def build_team_game_rows(games: pd.DataFrame, linescore: pd.DataFrame) -> pd.DataFrame:
    if linescore.empty:
        raise RuntimeError("No linescore data available.")
    game_groups = {gid: grp for gid, grp in linescore.groupby("game_id")}
    records = []
    for _, game in games.iterrows():
        gid = int(game["game_id"])
        segment = game_groups.get(gid)
        if segment is None:
            continue
        summary = summarize_game(segment)
        if summary is None:
            continue
        max_inning = summary["max_inning"]
        is_extras = max_inning > 9
        for flag, team_id_raw in ((0, game["away_team"]), (1, game["home_team"])):
            if pd.isna(team_id_raw):
                continue
            team_id = int(team_id_raw)
            if not (TEAM_MIN <= team_id <= TEAM_MAX):
                continue
            team_stats = summary["teams"][flag]
            opp_stats = summary["teams"][1 - flag]
            runs6_team = team_stats["runs_thru6"]
            runs6_opp = opp_stats["runs_thru6"]
            if pd.isna(runs6_team) or pd.isna(runs6_opp):
                trailing = np.nan
            else:
                trailing = runs6_team < runs6_opp
            team_won = team_stats["total_runs"] > opp_stats["total_runs"]
            walkoff = bool(summary["home_walkoff"]) if flag == 1 else False
            records.append(
                {
                    "team_id": team_id,
                    "game_id": gid,
                    "game_date": game.get("game_date"),
                    "won": team_won,
                    "trailing_after6": trailing,
                    "is_extras": is_extras,
                    "walkoff": walkoff,
                }
            )
    if not records:
        raise RuntimeError("Unable to derive any team-game rows from input data.")
    return pd.DataFrame(records)


def compute_grit_index(trail_pct: float, extras_pct: float, walkoff_wins: int, total_games: int) -> float:
    components = []
    if not pd.isna(trail_pct):
        components.append((0.5, trail_pct))
    if not pd.isna(extras_pct):
        components.append((0.3, extras_pct))
    walk_rate = (walkoff_wins / total_games) if total_games > 0 else np.nan
    if not pd.isna(walk_rate):
        components.append((0.2, walk_rate))
    if not components:
        return np.nan
    weight_sum = sum(weight for weight, _ in components)
    return sum(weight * value for weight, value in components) / weight_sum


def aggregate_team_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    data = []
    for tid in range(TEAM_MIN, TEAM_MAX + 1):
        team_rows = rows[rows["team_id"] == tid]
        overall_g = len(team_rows)
        overall_w = int(team_rows["won"].sum()) if overall_g else 0
        overall_l = overall_g - overall_w
        overall_winpct = overall_w / overall_g if overall_g else np.nan
        trail_subset = team_rows[team_rows["trailing_after6"].fillna(False)]
        trail6_g = len(trail_subset)
        trail6_w = int(trail_subset["won"].sum()) if trail6_g else 0
        trail6_l = trail6_g - trail6_w
        trail6_winpct = trail6_w / trail6_g if trail6_g else np.nan
        extras_subset = team_rows[team_rows["is_extras"].fillna(False)]
        extras_g = len(extras_subset)
        extras_w = int(extras_subset["won"].sum()) if extras_g else 0
        extras_l = extras_g - extras_w
        extras_winpct = extras_w / extras_g if extras_g else np.nan
        walkoff_wins = int(team_rows["walkoff"].sum()) if overall_g else 0
        grit_index = compute_grit_index(trail6_winpct, extras_winpct, walkoff_wins, overall_g)
        data.append(
            {
                "team_id": tid,
                "overall_g": overall_g,
                "overall_w": overall_w,
                "overall_l": overall_l,
                "overall_winpct": overall_winpct,
                "trail6_g": trail6_g,
                "trail6_w": trail6_w,
                "trail6_l": trail6_l,
                "trail6_winpct": trail6_winpct,
                "extras_g": extras_g,
                "extras_w": extras_w,
                "extras_l": extras_l,
                "extras_winpct": extras_winpct,
                "walkoff_wins": walkoff_wins,
                "grit_index": grit_index,
            }
        )
    return pd.DataFrame(data)


def rate_grit(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value >= 0.60:
        return "Ironclad"
    if value >= 0.50:
        return "Steely"
    if value >= 0.40:
        return "Scrappy"
    return "Searching"


def build_text_report(df: pd.DataFrame) -> str:
    lines = [
        "ABL Grit Index",
        "=" * 21,
        "Measures which clubs claw back from late deficits, thrive in extras, and deliver walk-off blows.",
        "Valuable for spotting resilient teams that can steal playoff-style games even when the bats go cold.",
        "",
    ]
    header = f"{'Team':<24} {'CD':<4} {'Rating':<12} {'Grit':>7} {'Trail6%':>10} {'Extras%':>9} {'Walk-off':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.iterrows():
        conf = row["conf_div"] if row["conf_div"] else "--"
        grit_txt = f"{row['grit_index']:.3f}" if not pd.isna(row["grit_index"]) else "NA "
        trail_txt = f"{row['trail6_winpct']:.3f}" if not pd.isna(row["trail6_winpct"]) else "NA "
        extras_txt = f"{row['extras_winpct']:.3f}" if not pd.isna(row["extras_winpct"]) else "NA "
        walkoffs = int(row["walkoff_wins"]) if pd.notna(row["walkoff_wins"]) else 0
        lines.append(
            f"{row['team_display']:<24} {conf:<4} {row['grit_rating']:<12} {grit_txt:>7} {trail_txt:>10} {extras_txt:>9} {walkoffs:>9}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  Ironclad >= 0.60, Steely 0.50-0.59, Scrappy 0.40-0.49, Searching < 0.40.")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  Trail6 Win% = record in games where the club trailed entering the 7th inning.")
    lines.append("  Extras Win% = record in extra-inning games (max inning > 9).")
    lines.append("  Walk-off Rate = walk-off wins / games; home teams only.")
    lines.append("  Grit Index = weighted blend: 50% Trail6 Win%, 30% Extras Win%, 20% Walk-off Rate.")
    return "\n".join(lines)


def format_record(wins: int, losses: int, pct: float) -> str:
    if wins + losses == 0 or pd.isna(pct):
        return f"{wins}-{losses} (NA )"
    return f"{wins}-{losses} ({pct:.3f})"


def print_top_table(df: pd.DataFrame) -> None:
    subset = df[
        [
            "team_display",
            "conf_div",
            "grit_rating",
            "grit_index",
            "trail6_winpct",
            "extras_winpct",
            "walkoff_wins",
        ]
    ].head(12)
    display_df = subset.copy()
    display_df["grit_index"] = display_df["grit_index"].map(lambda v: f"{v:.3f}" if not pd.isna(v) else "NA ")
    display_df["trail6_winpct"] = display_df["trail6_winpct"].map(
        lambda v: f"{v:.3f}" if not pd.isna(v) else "NA "
    )
    display_df["extras_winpct"] = display_df["extras_winpct"].map(
        lambda v: f"{v:.3f}" if not pd.isna(v) else "NA "
    )
    display_df = display_df.rename(
        columns={
            "team_display": "Team",
            "conf_div": "ConfDiv",
            "grit_rating": "Rating",
            "grit_index": "Grit",
            "trail6_winpct": "Trail6%",
            "extras_winpct": "Extras%",
            "walkoff_wins": "Walk-off",
        }
    )
    print(display_df.to_string(index=False))


def add_team_labels(df: pd.DataFrame, display_map: Dict[int, str], conf_map: Dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    df["team_display"] = df["team_id"].apply(lambda tid: display_map.get(tid, f"Team {tid}"))
    df["conf_div"] = df["team_id"].apply(lambda tid: conf_map.get(tid, ""))
    df["grit_rating"] = df["grit_index"].apply(rate_grit)
    return df


def resolve_optional_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Grit Index report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSV files.")
    parser.add_argument("--linescore", type=str, help="Override linescore file.")
    parser.add_argument("--pbp", type=str, help="Override play-by-play file for fallback.")
    parser.add_argument("--teamlogs", type=str, help="Override team log file (optional fallback).")
    parser.add_argument("--teams", type=str, help="Override team info file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Grit_Index.csv",
        help="Output CSV path (default inside out/csv_out).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base).resolve()
    teams_override = resolve_optional_path(base_dir, args.teams)
    linescore_override = resolve_optional_path(base_dir, args.linescore)
    pbp_override = resolve_optional_path(base_dir, args.pbp)

    games = load_games(base_dir)
    linescore = load_linescore(base_dir, linescore_override, pbp_override, games)
    linescore = linescore[linescore["game_id"].isin(games["game_id"].unique())]
    team_rows = build_team_game_rows(games, linescore)
    aggregates = aggregate_team_metrics(team_rows)
    display_map, conf_map = load_team_info(base_dir, teams_override)
    aggregates = add_team_labels(aggregates, display_map, conf_map)
    aggregates = aggregates.sort_values(
        by=["grit_index", "trail6_winpct", "extras_winpct"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_df = aggregates.copy()
    for col in [
        "overall_winpct",
        "trail6_winpct",
        "extras_winpct",
        "grit_index",
    ]:
        csv_df[col] = csv_df[col].round(3)
    csv_columns = [
        "team_id",
        "team_display",
        "conf_div",
        "overall_g",
        "overall_w",
        "overall_l",
        "overall_winpct",
        "trail6_g",
        "trail6_w",
        "trail6_l",
        "trail6_winpct",
        "extras_g",
        "extras_w",
        "extras_l",
        "extras_winpct",
        "walkoff_wins",
        "grit_index",
        "grit_rating",
    ]
    csv_df[csv_columns].to_csv(out_path, index=False)

    text_report = build_text_report(aggregates)
    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() in {'csv_out'}:
        text_dir = out_path.parent.parent / "txt_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / text_filename).write_text(text_report, encoding="utf-8")

    print_top_table(aggregates)


if __name__ == "__main__":
    main()

