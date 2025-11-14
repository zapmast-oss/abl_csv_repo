"""ABL Travel Fatigue report."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24

GAMELOG_CANDIDATES = [
    "team_game_log.csv",
    "schedule_team.csv",
    "game_results_by_team.csv",
    "games.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]
PARK_CANDIDATES = [
    "parks.csv",
    "park_info.csv",
]

EARTH_RADIUS_MILES = 3958.8


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def read_first(base: Path, override: Optional[Path], candidates: Sequence[str]) -> Optional[pd.DataFrame]:
    if override:
        path = override
        if not path.exists():
            raise FileNotFoundError(f"Specified file not found: {path}")
        return pd.read_csv(path)
    for name in candidates:
        path = base / name
        if path.exists():
            return pd.read_csv(path)
    return None


def resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_MILES * c


def safe_div(numer: float, denom: float) -> float:
    if pd.isna(numer) or pd.isna(denom) or denom == 0:
        return np.nan
    return numer / denom


def load_games(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, GAMELOG_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate team game logs/schedule.")
    team_col = pick_column(df, "team_id", "teamid")
    date_col = pick_column(df, "game_date", "date")
    home_team_col = pick_column(df, "home_team_id", "home_team", "home")
    away_team_col = pick_column(df, "away_team_id", "away_team", "away")
    park_col = pick_column(df, "park_id", "park")
    result_col = pick_column(df, "result", "wl")
    runs_for_col = pick_column(df, "runs_for", "runs0")
    runs_against_col = pick_column(df, "runs_against", "runs1")
    game_id_col = pick_column(df, "game_id", "game_key")
    if not date_col:
        raise ValueError("Game log missing date.")
    data = df.copy()
    if team_col:
        data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    else:
        if not home_team_col or not away_team_col:
            raise ValueError("Game log missing team identifiers.")
        home_rows = data.copy()
        home_rows["team_id"] = pd.to_numeric(home_rows[home_team_col], errors="coerce")
        away_rows = data.copy()
        away_rows["team_id"] = pd.to_numeric(away_rows[away_team_col], errors="coerce")
        away_rows["home_team_id"], away_rows["away_team_id"] = (
            pd.to_numeric(away_rows[home_team_col], errors="coerce"),
            pd.to_numeric(away_rows[away_team_col], errors="coerce"),
        )
        home_rows["home_team_id"] = pd.to_numeric(home_rows[home_team_col], errors="coerce")
        home_rows["away_team_id"] = pd.to_numeric(home_rows[away_team_col], errors="coerce")
        data = pd.concat([home_rows, away_rows], ignore_index=True)
        data["team_id"] = pd.to_numeric(data["team_id"], errors="coerce").astype("Int64")
    data = data.dropna(subset=["team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    data["game_date"] = pd.to_datetime(data[date_col])
    data["game_id"] = data[game_id_col] if game_id_col else np.nan
    data["home_team_id"] = pd.to_numeric(data[home_team_col], errors="coerce").astype("Int64") if home_team_col else np.nan
    data["away_team_id"] = pd.to_numeric(data[away_team_col], errors="coerce").astype("Int64") if away_team_col else np.nan
    data["park_id"] = data[park_col] if park_col else np.nan
    if result_col:
        data["WL"] = data[result_col].str.upper().str[0]
    elif runs_for_col and runs_against_col:
        rf = pd.to_numeric(data[runs_for_col], errors="coerce")
        ra = pd.to_numeric(data[runs_against_col], errors="coerce")
        data["WL"] = np.where(rf > ra, "W", "L")
    else:
        data["WL"] = np.nan
    data = data.dropna(subset=["game_date"])
    data = data.sort_values(by=["team_id", "game_date", "game_id"]).reset_index(drop=True)
    return data


def load_team_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}, {}
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    name_col = pick_column(df, "abbr", "team_abbr", "team_display", "team_name", "name")
    park_col = pick_column(df, "park_id", "home_park_id")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "conference_id")
    div_col = pick_column(df, "division_id", "division")
    names: Dict[int, str] = {}
    home_parks: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return names, home_parks, conf_map
    for _, row in df.iterrows():
        tid_val = row.get(team_col)
        if pd.isna(tid_val):
            continue
        try:
            tid = int(tid_val)
        except (TypeError, ValueError):
            continue
        if not (TEAM_MIN <= tid <= TEAM_MAX):
            continue
        if name_col and pd.notna(row.get(name_col)):
            names[tid] = str(row.get(name_col))
        if park_col and pd.notna(row.get(park_col)):
            home_parks[tid] = str(row.get(park_col))
        if tid in conf_map or not sub_col or not div_col:
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
        conf_map[tid] = f"{conf_lookup.get(sub_key, str(sub_val)[0].upper())}-{div_lookup.get(div_key, str(div_val)[0].upper())}"
    return names, home_parks, conf_map


def load_parks(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, PARK_CANDIDATES)
    if df is None:
        return pd.DataFrame()
    park_col = pick_column(df, "park_id", "ParkID", "park")
    lat_col = pick_column(df, "latitude", "lat")
    lon_col = pick_column(df, "longitude", "lon")
    if not park_col or not lat_col or not lon_col:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["park_id"] = df[park_col].astype(str)
    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    return out


def compute_team_rows(games: pd.DataFrame, team_home_parks: Dict[int, str]) -> pd.DataFrame:
    rows = []
    for team_id, team_df in games.groupby("team_id"):
        team_df = team_df.sort_values(by=["game_date", "game_id"]).reset_index(drop=True)
        home_park = team_home_parks.get(team_id)
        for _, row in team_df.iterrows():
            is_home = np.nan
            if pd.notna(row.get("home_team_id")):
                is_home = row["team_id"] == row["home_team_id"]
            venue = row["team_id"] if is_home else row.get("home_team_id")
            city_key = str(row["park_id"]) if pd.notna(row["park_id"]) else (str(venue) if pd.notna(venue) else str(home_park))
            rows.append(
                {
                    "team_id": row["team_id"],
                    "game_date": row["game_date"],
                    "game_id": row.get("game_id"),
                    "is_home": bool(is_home) if isinstance(is_home, bool) else False,
                    "opponent_id": row["away_team_id"] if bool(is_home) else row["home_team_id"],
                    "city_key": city_key,
                    "WL": row.get("WL"),
                }
            )
    team_games = pd.DataFrame(rows)
    team_games = team_games.sort_values(by=["team_id", "game_date", "game_id"]).reset_index(drop=True)
    return team_games


def annotate_travel(team_games: pd.DataFrame, parks_df: pd.DataFrame, short_miles: float, long_miles: float) -> pd.DataFrame:
    park_coords = {row["park_id"]: (row["lat"], row["lon"]) for _, row in parks_df.iterrows()}
    legs = []
    for team_id, group in team_games.groupby("team_id"):
        group = group.sort_values(by=["game_date", "game_id"]).reset_index(drop=True)
        prev_city = None
        prev_date = None
        road_trip_id = 0
        road_game_no = 0
        for idx, row in group.iterrows():
            travel_flag = False
            distance = np.nan
            dist_band = ""
            days_rest = np.nan
            if prev_city is not None:
                travel_flag = row["city_key"] != prev_city
            if prev_date is not None:
                days_rest = (row["game_date"] - prev_date).days
            if travel_flag and row["city_key"] in park_coords and prev_city in park_coords:
                lat1, lon1 = park_coords[prev_city]
                lat2, lon2 = park_coords[row["city_key"]]
                distance = haversine_miles(lat1, lon1, lat2, lon2)
                if distance < short_miles:
                    dist_band = "SHORT"
                elif distance < long_miles:
                    dist_band = "MEDIUM"
                else:
                    dist_band = "LONG"
            first_after = "Y" if travel_flag else ""
            no_off = "Y" if travel_flag and days_rest == 1 else ""
            with_off = "Y" if travel_flag and days_rest and days_rest > 1 else ""
            if not row["is_home"]:
                if idx == 0 or group.loc[idx - 1, "is_home"]:
                    road_trip_id += 1
                    road_game_no = 1
                else:
                    road_game_no += 1
            else:
                road_game_no = 0
            legs.append(
                {
                    "team_id": team_id,
                    "game_date": row["game_date"],
                    "prev_game_date": prev_date,
                    "city_key": row["city_key"],
                    "prev_city_key": prev_city,
                    "travel_flag": travel_flag,
                    "days_rest": days_rest,
                    "distance_miles": distance,
                    "dist_band": dist_band,
                    "first_after_travel": first_after,
                    "no_offday_travel": no_off,
                    "with_offday_travel": with_off,
                    "is_home": row["is_home"],
                    "opponent_id": row["opponent_id"],
                    "WL": row["WL"],
                    "road_trip_id": road_trip_id if road_trip_id > 0 else np.nan,
                    "road_trip_game_no": road_game_no if road_game_no > 0 else np.nan,
                }
            )
            prev_city = row["city_key"]
            prev_date = row["game_date"]
    legs_df = pd.DataFrame(legs)
    return legs_df


def compute_road_trips(legs_df: pd.DataFrame) -> pd.DataFrame:
    trips = []
    for team_id, group in legs_df.groupby("team_id"):
        road_games = group[group["is_home"] == False]
        if road_games.empty:
            continue
        road_games = road_games.sort_values(by=["game_date"]).reset_index(drop=True)
        start_idx = 0
        while start_idx < len(road_games):
            trip_start_date = road_games.loc[start_idx, "game_date"]
            end_idx = start_idx
            road_wins = 1 if road_games.loc[end_idx, "WL"] == "W" else 0
            road_losses = 1 if road_games.loc[end_idx, "WL"] == "L" else 0
            while end_idx + 1 < len(road_games) and (road_games.loc[end_idx + 1, "game_date"] - road_games.loc[end_idx, "game_date"]).days <= 1:
                end_idx += 1
                road_wins += 1 if road_games.loc[end_idx, "WL"] == "W" else 0
                road_losses += 1 if road_games.loc[end_idx, "WL"] == "L" else 0
            trip_end_date = road_games.loc[end_idx, "game_date"]
            trips.append(
                {
                    "team_id": team_id,
                    "trip_len_days": (trip_end_date - trip_start_date).days + 1,
                    "games_in_trip": end_idx - start_idx + 1,
                    "road_wins": road_wins,
                    "road_losses": road_losses,
                }
            )
            start_idx = end_idx + 1
    return pd.DataFrame(trips)


def summarize_team(legs_df: pd.DataFrame, trips_df: pd.DataFrame, min_inn: float, min_attempts: int) -> pd.DataFrame:
    summary_rows = []
    for team_id, group in legs_df.groupby("team_id"):
        road_games = group[group["is_home"] == False]
        road_w = (road_games["WL"] == "W").sum()
        road_l = (road_games["WL"] == "L").sum()
        road_games_count = len(road_games)
        road_w_pct = safe_div(road_w, road_w + road_l)

        trips_team = trips_df[trips_df["team_id"] == team_id]
        road_trip_count = len(trips_team)
        avg_trip_len = trips_team["trip_len_days"].mean() if not trips_team.empty else np.nan
        max_trip_len = trips_team["trip_len_days"].max() if not trips_team.empty else np.nan

        fat_games = group[group["first_after_travel"] == "Y"]
        fat_w = (fat_games["WL"] == "W").sum()
        fat_l = (fat_games["WL"] == "L").sum()
        fat_w_pct = safe_div(fat_w, fat_w + fat_l)

        fat0 = fat_games[fat_games["no_offday_travel"] == "Y"]
        fat0_w = (fat0["WL"] == "W").sum()
        fat0_l = (fat0["WL"] == "L").sum()
        fat0_w_pct = safe_div(fat0_w, fat0_w + fat0_l)

        fat1 = fat_games[fat_games["with_offday_travel"] == "Y"]
        fat1_w = (fat1["WL"] == "W").sum()
        fat1_l = (fat1["WL"] == "L").sum()
        fat1_w_pct = safe_div(fat1_w, fat1_w + fat1_l)

        dist_summary = {}
        for band in ["SHORT", "MEDIUM", "LONG"]:
            band_games = fat_games[fat_games["dist_band"] == band]
            w = (band_games["WL"] == "W").sum()
            l = (band_games["WL"] == "L").sum()
            dist_summary[band] = {
                "games": len(band_games) if not band_games.empty else np.nan,
                "W": w if len(band_games) else np.nan,
                "L": l if len(band_games) else np.nan,
                "pct": safe_div(w, w + l) if w + l > 0 else np.nan,
            }

        summary_rows.append(
            {
                "team_id": team_id,
                "road_trip_count": road_trip_count,
                "avg_trip_len_days": avg_trip_len,
                "max_trip_len_days": max_trip_len,
                "road_games": road_games_count,
                "road_W": road_w,
                "road_L": road_l,
                "road_W_pct": road_w_pct,
                "FAT_games": len(fat_games),
                "FAT_W": fat_w,
                "FAT_L": fat_l,
                "FAT_W_pct": fat_w_pct,
                "FAT0_games": len(fat0),
                "FAT0_W": fat0_w,
                "FAT0_L": fat0_l,
                "FAT0_W_pct": fat0_w_pct,
                "FAT1p_games": len(fat1),
                "FAT1p_W": fat1_w,
                "FAT1p_L": fat1_l,
                "FAT1p_W_pct": fat1_w_pct,
                "SHORT_games": dist_summary["SHORT"]["games"],
                "SHORT_W": dist_summary["SHORT"]["W"],
                "SHORT_L": dist_summary["SHORT"]["L"],
                "SHORT_W_pct": dist_summary["SHORT"]["pct"],
                "MEDIUM_games": dist_summary["MEDIUM"]["games"],
                "MEDIUM_W": dist_summary["MEDIUM"]["W"],
                "MEDIUM_L": dist_summary["MEDIUM"]["L"],
                "MEDIUM_W_pct": dist_summary["MEDIUM"]["pct"],
                "LONG_games": dist_summary["LONG"]["games"],
                "LONG_W": dist_summary["LONG"]["W"],
                "LONG_L": dist_summary["LONG"]["L"],
                "LONG_W_pct": dist_summary["LONG"]["pct"],
            }
        )
    return pd.DataFrame(summary_rows)


def text_table(
    df: pd.DataFrame,
    columns: Sequence[Tuple[str, str, int, bool, str]],
    title: str,
    subtitle: str,
    key_lines: Sequence[str],
    def_lines: Sequence[str],
) -> str:
    lines = [title, "=" * len(title), subtitle, ""]
    header = " ".join(
        f"{label:<{width}}" if not align_right else f"{label:>{width}}"
        for label, _, width, align_right, _ in columns
    )
    lines.append(header)
    lines.append("-" * len(header))
    if df.empty:
        lines.append("(No qualifying teams.)")
    else:
        for _, row in df.iterrows():
            parts = []
            for _, col_name, width, align_right, fmt in columns:
                value = row.get(col_name, "")
                if isinstance(value, (int, float, np.number)):
                    if pd.isna(value):
                        display = "NA"
                    else:
                        display = format(value, fmt) if fmt else str(value)
                else:
                    display = str(value)
                fmt_str = f"{{:>{width}}}" if align_right else f"{{:<{width}}}"
                parts.append(fmt_str.format(display[:width]))
            lines.append(" ".join(parts))
    lines.append("")
    lines.append("Key:")
    for line in key_lines:
        lines.append(f"  {line}")
    lines.append("")
    lines.append("Definitions:")
    for line in def_lines:
        lines.append(f"  {line}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="ABL Travel Fatigue report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory.")
    parser.add_argument("--gamelogs", type=str, help="Override game log path.")
    parser.add_argument("--teams", type=str, help="Override team info path.")
    parser.add_argument("--parks", type=str, help="Override park info path.")
    parser.add_argument("--out_summary", type=str, default="out/z_ABL_Travel_Fatigue_Summary.csv", help="Summary CSV path.")
    parser.add_argument("--out_legs", type=str, default="out/z_ABL_Travel_Fatigue_Legs.csv", help="Legs CSV path.")
    parser.add_argument("--short_miles", type=float, default=300.0, help="Short-haul threshold in miles.")
    parser.add_argument("--long_miles", type=float, default=800.0, help="Long-haul threshold.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    games = load_games(base_dir, resolve_path(base_dir, args.gamelogs))
    team_names, team_home_parks, conf_map = load_team_info(base_dir, resolve_path(base_dir, args.teams))
    parks_df = load_parks(base_dir, resolve_path(base_dir, args.parks))

    team_games = compute_team_rows(games, team_home_parks)
    legs_df = annotate_travel(team_games, parks_df, args.short_miles, args.long_miles)

    trips_df = compute_road_trips(legs_df)
    summary_df = summarize_team(legs_df, trips_df, args.min_inn if hasattr(args, "min_inn") else 0, args.min_attempts if hasattr(args, "min_attempts") else 0)

    summary_df["team_display"] = summary_df["team_id"].map(team_names)
    summary_df["team_display"] = summary_df.apply(
        lambda r: r["team_display"] if pd.notna(r["team_display"]) else f"T{int(r['team_id'])}",
        axis=1,
    )
    summary_df["conf_div"] = summary_df["team_id"].map(conf_map).fillna("")

    csv_summary = summary_df.copy()
    for col in [
        "avg_trip_len_days",
        "road_W_pct",
        "FAT_W_pct",
        "FAT0_W_pct",
        "FAT1p_W_pct",
        "SHORT_W_pct",
        "MEDIUM_W_pct",
        "LONG_W_pct",
    ]:
        csv_summary[col] = csv_summary[col].round(3)
    out_summary = Path(args.out_summary)
    if not out_summary.is_absolute():
        out_summary = base_dir / out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    csv_summary.to_csv(out_summary, index=False)

    legs_out = legs_df.copy()
    legs_out["team_display"] = legs_out["team_id"].map(team_names).fillna("")
    legs_out["conf_div"] = legs_out["team_id"].map(conf_map).fillna("")
    legs_out["distance_miles"] = legs_out["distance_miles"].round(1)
    out_legs = Path(args.out_legs)
    if not out_legs.is_absolute():
        out_legs = base_dir / out_legs
    out_legs.parent.mkdir(parents=True, exist_ok=True)
    legs_out.to_csv(out_legs, index=False)

    display_df = summary_df[summary_df["FAT0_games"] >= 5].copy()
    display_df = display_df.sort_values(by=["FAT0_W_pct", "FAT_W_pct"], ascending=[True, True]).head(10)
    text_columns = [
        ("Team", "team_display", 10, False, ""),
        ("Conf", "conf_div", 6, False, ""),
        ("Road%", "road_W_pct", 7, True, ".3f"),
        ("Trips", "road_trip_count", 6, True, ".0f"),
        ("AvgTrip", "avg_trip_len_days", 7, True, ".2f"),
        ("FAT%", "FAT_W_pct", 6, True, ".3f"),
        ("NoRest%", "FAT0_W_pct", 8, True, ".3f"),
    ]
    text_output = text_table(
        display_df,
        text_columns,
        "ABL Travel Fatigue",
        "Teams struggling most after no-rest travel (>=5 such games)",
        [
            "Ratings: <=0.350 Grounded Flyers, 0.351-0.450 Weary Wings, 0.451-0.550 Jet-Lag Neutral, >0.550 Road-Ready.",
        ],
        [
            "Road% uses away games only; FAT metrics cover first games after venue changes.",
            "NoRest% isolates games with zero off-days between cities.",
            "Distance splits available in CSV when park coordinates exist.",
        ],
    )
    text_path = out_summary.with_suffix(".txt")
    text_path.write_text(text_output, encoding="utf-8")
    print(text_output)


if __name__ == "__main__":
    main()
