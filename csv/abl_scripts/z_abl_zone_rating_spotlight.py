"""ABL Zone Rating Spotlight report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24
POSITION_ORDER = ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
POS_MAP_DIRECT = {"1": "P", "2": "C", "3": "1B", "4": "2B", "5": "3B", "6": "SS", "7": "LF", "8": "CF", "9": "RF"}
POSITION_MAP = {
    "SP": "P",
    "RP": "P",
    "PITCHER": "P",
    "LF ": "LF",
    "RF ": "RF",
    "CF ": "CF",
    "1B ": "1B",
    "2B ": "2B",
    "3B ": "3B",
    "SS ": "SS",
    "C ": "C",
    "DH ": "DH",
}

FIELDING_CANDIDATES = [
    "players_career_fielding_stats.csv",
    "players_fielding.csv",
    "fielding.csv",
]
ROSTER_CANDIDATES = [
    "players.csv",
    "player_register.csv",
    "rosters.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]


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


def load_fielding(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, FIELDING_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate fielding data.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    pos_col = pick_column(df, "position", "pos")
    inn_col = pick_column(df, "inn", "ip", "innings")
    zr_col = pick_column(df, "zr", "zone_rating")
    year_col = pick_column(df, "year", "season")
    if not id_col or not team_col or not pos_col or not inn_col or not zr_col:
        raise ValueError("Fielding file missing key columns.")
    data = df.copy()
    if year_col:
        max_year = pd.to_numeric(data[year_col], errors="coerce").max()
        data = data[pd.to_numeric(data[year_col], errors="coerce") == max_year]
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    data["pos"] = data[pos_col].astype(str).str.strip().str.upper()
    data["pos"] = data["pos"].replace(POSITION_MAP)
    data["pos"] = data["pos"].replace(POS_MAP_DIRECT)
    data = data[data["pos"].isin(POSITION_ORDER)]
    out = pd.DataFrame()
    out["player_id"] = data["player_id"]
    out["team_id"] = data["team_id"]
    out["pos"] = data["pos"]
    out["INN"] = pd.to_numeric(data[inn_col], errors="coerce")
    out["ZR"] = pd.to_numeric(data[zr_col], errors="coerce")
    out = out.dropna(subset=["INN", "ZR"])
    return out


def load_roster(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, ROSTER_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["player_id", "player_name"])
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    first_col = pick_column(df, "first_name", "firstname")
    last_col = pick_column(df, "last_name", "lastname")
    full_col = pick_column(df, "name_full", "name", "player_name")
    if not id_col:
        return pd.DataFrame(columns=["player_id", "player_name"])
    data = df.copy()
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id"])
    out = pd.DataFrame()
    out["player_id"] = data["player_id"]
    if first_col and last_col:
        out["player_name"] = (
            data[first_col].fillna("").astype(str).str.strip()
            + " "
            + data[last_col].fillna("").astype(str).str.strip()
        ).str.strip()
    elif full_col:
        out["player_name"] = data[full_col].fillna("").astype(str)
    else:
        out["player_name"] = out["player_id"].astype(str)
    return out


def load_team_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    name_col = pick_column(df, "abbr", "team_abbr", "team_display", "team_name", "name")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "conference_id")
    div_col = pick_column(df, "division_id", "division")
    names: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return names, conf_map
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
    return names, conf_map


def aggregate_team_position(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["team_id", "pos"]
    agg = df.groupby(group_cols, as_index=False).agg(team_pos_inn=("INN", "sum"), team_pos_zr=("ZR", "sum"))
    return agg


def aggregate_player_position(df: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["team_id", "pos", "player_id"]
    agg = df.groupby(group_cols, as_index=False).agg(player_inn=("INN", "sum"), player_zr=("ZR", "sum"))
    if not roster.empty:
        agg = agg.merge(roster, on="player_id", how="left")
    else:
        agg["player_name"] = agg["player_id"].astype(str)
    agg["player_name"] = agg["player_name"].fillna("")
    return agg


def compute_team_if_zr(team_pos: pd.DataFrame) -> pd.DataFrame:
    infield = team_pos[team_pos["pos"].isin(["1B", "2B", "3B", "SS"])]
    agg = infield.groupby("team_id", as_index=False)["team_pos_zr"].sum()
    agg.rename(columns={"team_pos_zr": "team_if_zr_total"}, inplace=True)
    lg_mean = agg["team_if_zr_total"].mean()
    agg["team_if_zr_plus"] = agg["team_if_zr_total"] / lg_mean if lg_mean and not np.isnan(lg_mean) else np.nan
    return agg


def attach_top_players(team_pos: pd.DataFrame, player_pos: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
    top_rows = []
    for (team_id, pos), group in player_pos.groupby(["team_id", "pos"]):
        ordered = group.sort_values(
            by=["player_inn", "player_zr"],
            ascending=[False, False],
        ).head(top_n)
        row = {"team_id": team_id, "pos": pos}
        for idx, (_, player_row) in enumerate(ordered.iterrows(), start=1):
            row[f"player{idx}_id"] = player_row["player_id"]
            row[f"player{idx}_name"] = player_row["player_name"]
            row[f"player{idx}_inn"] = player_row["player_inn"]
            row[f"player{idx}_zr"] = player_row["player_zr"]
        top_rows.append(row)
    top_df = pd.DataFrame(top_rows)
    if top_df.empty:
        for idx in range(1, top_n + 1):
            team_pos[f"player{idx}_id"] = np.nan
            team_pos[f"player{idx}_name"] = ""
            team_pos[f"player{idx}_inn"] = np.nan
            team_pos[f"player{idx}_zr"] = np.nan
        return team_pos
    return team_pos.merge(top_df, on=["team_id", "pos"], how="left")


def compute_position_means(team_pos: pd.DataFrame) -> pd.DataFrame:
    pos_means = (
        team_pos.assign(abs_zr=team_pos["team_pos_zr"].abs())
        .groupby("pos", as_index=False)["abs_zr"]
        .mean()
        .rename(columns={"abs_zr": "lg_pos_mean_zr"})
    )
    return team_pos.merge(pos_means, on="pos", how="left")


def text_table(df: pd.DataFrame, columns: Sequence[Tuple[str, str, int, bool, str]], title: str, subtitle: str, key_lines: Sequence[str], def_lines: Sequence[str]) -> str:
    lines = [title, "=" * len(title), subtitle, ""]
    header = " ".join(
        f"{label:<{width}}" if not align_right else f"{label:>{width}}"
        for label, _, width, align_right, _ in columns
    )
    lines.append(header)
    lines.append("-" * len(header))
    if df.empty:
        lines.append("(No data available.)")
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
    parser = argparse.ArgumentParser(description="ABL Zone Rating Spotlight.")
    parser.add_argument("--base", type=str, default=".", help="Base directory.")
    parser.add_argument("--fielding", type=str, help="Override fielding path.")
    parser.add_argument("--teams", type=str, help="Override team info path.")
    parser.add_argument("--roster", type=str, help="Override roster path.")
    parser.add_argument("--out", type=str, default="out/z_ABL_Zone_Rating_Spotlight.csv", help="Output CSV path.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    fielding = load_fielding(base_dir, resolve_path(base_dir, args.fielding))
    roster = load_roster(base_dir, resolve_path(base_dir, args.roster))
    team_display, conf_map = load_team_info(base_dir, resolve_path(base_dir, args.teams))

    player_pos = aggregate_player_position(fielding, roster)
    team_pos = aggregate_team_position(fielding)

    team_pos = attach_top_players(team_pos, player_pos, top_n=2)
    team_pos = compute_position_means(team_pos)
    team_pos["team_pos_zr_plus"] = np.where(
        team_pos["lg_pos_mean_zr"].abs() > 1e-6,
        team_pos["team_pos_zr"] / team_pos["lg_pos_mean_zr"],
        np.nan,
    )

    team_if = compute_team_if_zr(team_pos)
    team_pos = team_pos.merge(team_if, on="team_id", how="left")

    team_pos["team_display"] = team_pos["team_id"].map(team_display)
    team_pos["team_display"] = team_pos.apply(
        lambda r: r["team_display"]
        if pd.notna(r["team_display"])
        else (f"T{int(r['team_id'])}" if pd.notna(r["team_id"]) else ""),
        axis=1,
    )
    team_pos["conf_div"] = team_pos["team_id"].map(conf_map).fillna("")

    pos_rank = {p: i for i, p in enumerate(POSITION_ORDER)}
    team_pos["pos_order"] = team_pos["pos"].map(pos_rank)
    team_pos = team_pos.sort_values(
        by=["pos_order", "team_pos_zr"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    csv_columns = [
        "team_id",
        "team_display",
        "conf_div",
        "pos",
        "team_pos_inn",
        "team_pos_zr",
        "team_pos_zr_plus",
        "player1_id",
        "player1_name",
        "player1_inn",
        "player1_zr",
        "player2_id",
        "player2_name",
        "player2_inn",
        "player2_zr",
        "team_if_zr_total",
        "team_if_zr_plus",
    ]
    csv_df = team_pos[csv_columns].copy()
    csv_df["team_pos_inn"] = csv_df["team_pos_inn"].round(1)
    csv_df["team_pos_zr"] = csv_df["team_pos_zr"].round(2)
    csv_df["team_pos_zr_plus"] = csv_df["team_pos_zr_plus"].round(3)
    csv_df["player1_inn"] = csv_df["player1_inn"].round(1)
    csv_df["player1_zr"] = csv_df["player1_zr"].round(2)
    csv_df["player2_inn"] = csv_df["player2_inn"].round(1)
    csv_df["player2_zr"] = csv_df["player2_zr"].round(2)
    csv_df["team_if_zr_total"] = csv_df["team_if_zr_total"].round(2)
    csv_df["team_if_zr_plus"] = csv_df["team_if_zr_plus"].round(3)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_path, index=False)

    display_df = team_pos.sort_values(
        by=["team_pos_zr_plus", "team_pos_zr"],
        ascending=[False, False],
        na_position="last",
    ).head(10)
    text_columns = [
        ("Team", "team_display", 8, False, ""),
        ("Conf", "conf_div", 6, False, ""),
        ("Pos", "pos", 3, False, ""),
        ("ZR", "team_pos_zr", 6, True, ".2f"),
        ("ZR+", "team_pos_zr_plus", 6, True, ".3f"),
        ("Player1", "player1_name", 16, False, ""),
        ("P1 INN", "player1_inn", 6, True, ".1f"),
        ("P1 ZR", "player1_zr", 6, True, ".2f"),
        ("Player2", "player2_name", 16, False, ""),
        ("P2 INN", "player2_inn", 6, True, ".1f"),
        ("P2 ZR", "player2_zr", 6, True, ".2f"),
    ]
    text_output = text_table(
        display_df,
        text_columns,
        "ABL Zone Rating Spotlight",
        "Top team-position ZR vs league averages",
        [
            "Ratings: ZR+ >=1.25 (Elite Wall), 1.10-1.24 (Gold Glove Tier), 0.95-1.09 (Solid), <0.95 (Needs Support).",
        ],
        [
            "ZR aggregates team defense at each position (higher better).",
            "ZR+ compares team ZR to league positional mean (1.00 = league average).",
            "Top two players listed by innings share with ZR for quick reference.",
        ],
    )
    text_path = out_path.with_suffix(".txt")
    text_path.write_text(text_output, encoding="utf-8")
    print(text_output)


if __name__ == "__main__":
    main()
