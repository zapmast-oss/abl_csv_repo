"""ABL Table Setter vs Clearer report."""

from __future__ import annotations

import argparse
import numbers
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24

TOTAL_CANDIDATES = [
    "players_career_batting_stats.csv",
    "player_batting_totals.csv",
    "players_batting.csv",
    "batting_players.csv",
]
SPLIT_CANDIDATES = [
    "batting_splits_situational.csv",
    "batting_splits.csv",
    "players_batting_splits.csv",
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
PBP_CANDIDATES = [
    "players_at_bat_batting_stats.csv",
    "play_by_play.csv",
    "game_events.csv",
]


def empty_opportunity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": pd.Series(dtype="Int64"),
            "team_id": pd.Series(dtype="Int64"),
            "PA_leadoff": pd.Series(dtype="float"),
            "OBP_leadoff": pd.Series(dtype="float"),
            "PA_men_on": pd.Series(dtype="float"),
            "proxy_source": pd.Series(dtype="object"),
        }
    )


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
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


def load_roster(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, ROSTER_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate roster/player info.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    first_col = pick_column(df, "first_name", "firstname")
    last_col = pick_column(df, "last_name", "lastname")
    full_col = pick_column(df, "name_full", "name", "player_name")
    if not id_col:
        raise ValueError("Roster file missing player_id column.")
    df = df.copy()
    df["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64") if team_col else pd.NA
    df = df.dropna(subset=["player_id"])
    out = df[["player_id", "team_id"]].copy()
    if first_col and last_col:
        out["player_name"] = (
            df[first_col].fillna("").astype(str).str.strip()
            + " "
            + df[last_col].fillna("").astype(str).str.strip()
        ).str.strip()
    elif full_col:
        out["player_name"] = df[full_col].fillna("").astype(str)
    else:
        out["player_name"] = out["player_id"].astype(str)
    return out


def load_team_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}, {}
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    abbr_col = pick_column(df, "abbr", "team_abbr", "short_name")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "conference_id")
    div_col = pick_column(df, "division_id", "division")
    names: Dict[int, str] = {}
    abbrs: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return names, abbrs, conf_map
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
        if abbr_col and pd.notna(row.get(abbr_col)):
            abbrs[tid] = str(row.get(abbr_col))
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
    return names, abbrs, conf_map


def load_totals(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, TOTAL_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate batting totals.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    pa_col = pick_column(df, "pa", "PA")
    ab_col = pick_column(df, "ab", "AB")
    h_col = pick_column(df, "h", "H")
    bb_col = pick_column(df, "bb", "BB")
    hbp_col = pick_column(df, "hbp", "HBP")
    sf_col = pick_column(df, "sf", "SF")
    rbi_col = pick_column(df, "rbi", "RBI")
    obp_col = pick_column(df, "obp", "OBP")
    year_col = pick_column(df, "year", "season")
    split_col = pick_column(df, "split_id", "split")
    if not id_col or not team_col or not pa_col:
        raise ValueError("Totals file missing key columns.")
    data = df.copy()
    if split_col:
        data = data[data[split_col] == data[split_col].min()]
    if year_col:
        data = data[data[year_col] == data[year_col].max()]
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    out = data[["player_id", "team_id"]].copy()
    out["PA"] = pd.to_numeric(data[pa_col], errors="coerce")
    out["AB"] = pd.to_numeric(data[ab_col], errors="coerce") if ab_col else np.nan
    out["H"] = pd.to_numeric(data[h_col], errors="coerce") if h_col else np.nan
    out["BB"] = pd.to_numeric(data[bb_col], errors="coerce") if bb_col else np.nan
    out["HBP"] = pd.to_numeric(data[hbp_col], errors="coerce") if hbp_col else 0.0
    out["SF"] = pd.to_numeric(data[sf_col], errors="coerce") if sf_col else 0.0
    out["RBI"] = pd.to_numeric(data[rbi_col], errors="coerce") if rbi_col else np.nan
    out["OBP_raw"] = pd.to_numeric(data[obp_col], errors="coerce") if obp_col else np.nan
    return out


def load_splits(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, SPLIT_CANDIDATES)
    if df is None:
        return empty_opportunity_frame()
    player_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    leadoff_col = pick_column(df, "pa_leadoff", "PA_leadoff", "pa_start_inning")
    leadoff_obp_col = pick_column(df, "obp_leadoff", "obp_start_inning", "OBP_leadoff")
    men_on_col = pick_column(df, "pa_men_on", "PA_men_on")
    risp_col = pick_column(df, "pa_risp", "PA_RISP")
    split_col = pick_column(df, "split_id", "split")
    year_col = pick_column(df, "year", "season")
    if not player_col:
        return empty_opportunity_frame()
    data = df.copy()
    if split_col:
        data = data[data[split_col] == data[split_col].min()]
    if year_col:
        data = data[data[year_col] == data[year_col].max()]
    data["player_id"] = pd.to_numeric(data[player_col], errors="coerce").astype("Int64")
    if team_col:
        data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    else:
        data["team_id"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    data = data.dropna(subset=["player_id"])
    result = pd.DataFrame()
    result["player_id"] = data["player_id"]
    result["team_id"] = data["team_id"]
    result["PA_leadoff"] = pd.to_numeric(data[leadoff_col], errors="coerce") if leadoff_col else np.nan
    result["OBP_leadoff"] = pd.to_numeric(data[leadoff_obp_col], errors="coerce") if leadoff_obp_col else np.nan
    if men_on_col:
        result["PA_men_on"] = pd.to_numeric(data[men_on_col], errors="coerce")
        result["proxy_source"] = "men_on"
    elif risp_col:
        result["PA_men_on"] = pd.to_numeric(data[risp_col], errors="coerce")
        result["proxy_source"] = "risp"
    else:
        result["PA_men_on"] = np.nan
        result["proxy_source"] = ""
    return result


def load_pbp_opps(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, PBP_CANDIDATES)
    if df is None:
        return empty_opportunity_frame()
    player_col = pick_column(df, "player_id", "batter_id", "PlayerID")
    team_col = pick_column(df, "team_id", "bat_team_id", "TeamID")
    outs_col = pick_column(df, "outs", "outs_before", "outs_start")
    base1_col = pick_column(df, "base1", "runner_on_first", "start_base_1")
    base2_col = pick_column(df, "base2", "runner_on_second", "start_base_2")
    base3_col = pick_column(df, "base3", "runner_on_third", "start_base_3")
    if not player_col or not team_col:
        return empty_opportunity_frame()
    data = df.copy()
    data["player_id"] = pd.to_numeric(data[player_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    if data.empty:
        return empty_opportunity_frame()

    def base_flag(col: Optional[str]) -> Optional[pd.Series]:
        if not col or col not in data:
            return None
        return pd.to_numeric(data[col], errors="coerce").fillna(0).ne(0)

    runners_flag = None
    for col_name in (base1_col, base2_col, base3_col):
        series = base_flag(col_name)
        if series is None:
            continue
        runners_flag = series if runners_flag is None else (runners_flag | series)

    outs_series = (
        pd.to_numeric(data[outs_col], errors="coerce")
        if outs_col and outs_col in data
        else None
    )

    agg_kwargs = {"PA_total": ("player_id", "size")}
    if runners_flag is not None and outs_series is not None:
        data["leadoff_val"] = ((outs_series == 0) & (~runners_flag)).astype(int)
        agg_kwargs["PA_leadoff"] = ("leadoff_val", "sum")
    if runners_flag is not None:
        data["men_on_val"] = runners_flag.astype(int)
        agg_kwargs["PA_men_on"] = ("men_on_val", "sum")

    grouped = (
        data.groupby(["player_id", "team_id"], as_index=False).agg(**agg_kwargs)
        if agg_kwargs
        else empty_opportunity_frame()
    )
    if grouped.empty:
        return empty_opportunity_frame()
    if "PA_leadoff" not in grouped:
        grouped["PA_leadoff"] = np.nan
    if "PA_men_on" not in grouped:
        grouped["PA_men_on"] = np.nan
    grouped["OBP_leadoff"] = np.nan
    grouped["proxy_source"] = np.where(
        grouped[["PA_leadoff", "PA_men_on"]].notna().any(axis=1), "pbp", ""
    )
    return grouped[["player_id", "team_id", "PA_leadoff", "OBP_leadoff", "PA_men_on", "proxy_source"]]


def combine_opportunity_sources(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty and secondary.empty:
        return empty_opportunity_frame()
    if primary.empty:
        return secondary
    if secondary.empty:
        return primary
    merged = primary.merge(
        secondary,
        on=["player_id", "team_id"],
        how="outer",
        suffixes=("", "_sec"),
    )
    for col in ["PA_leadoff", "OBP_leadoff", "PA_men_on"]:
        sec_col = f"{col}_sec"
        if sec_col in merged:
            merged[col] = merged[col].combine_first(merged[sec_col])
    merged["proxy_source"] = merged["proxy_source"].replace("", np.nan)
    if "proxy_source_sec" in merged:
        merged["proxy_source_sec"] = merged["proxy_source_sec"].replace("", np.nan)
    merged["proxy_source"] = merged["proxy_source"].combine_first(merged.get("proxy_source_sec"))
    merged["proxy_source"] = merged["proxy_source"].fillna("")
    cols = ["player_id", "team_id", "PA_leadoff", "OBP_leadoff", "PA_men_on", "proxy_source"]
    return merged[cols]


def compute_obp(row: pd.Series) -> float:
    if not pd.isna(row.get("OBP_raw")):
        return row["OBP_raw"]
    den = row["AB"] + row["BB"] + row["HBP"] + row["SF"]
    if den > 0:
        return (row["H"] + row["BB"] + row["HBP"]) / den
    return np.nan


def safe_div(numer: float, denom: float) -> float:
    if pd.isna(numer) or pd.isna(denom) or denom <= 0:
        return np.nan
    return numer / denom


def compute_avg(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(valid.mean())


def text_table(df: pd.DataFrame, columns: Sequence[Tuple[str, str, int, bool, str]], title: str, threshold_line: str, key_lines: Sequence[str], def_lines: Sequence[str]) -> str:
    lines = [title, "=" * len(title), ""]
    header = " ".join(f"{label:<{width}}" if not align_right else f"{label:>{width}}" for label, _, width, align_right, _ in columns)
    lines.append(header)
    lines.append("-" * len(header))
    if df.empty:
        lines.append("(No hitters met the qualification thresholds.)")
    for _, row in df.iterrows():
        parts = []
        for _, col_name, width, align_right, fmt in columns:
            value = row.get(col_name, "")
            if isinstance(value, numbers.Number):
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
    lines.append(threshold_line)
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
    parser = argparse.ArgumentParser(description="ABL Table Setter vs Clearer report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--totals", type=str, help="Override path for batting totals.")
    parser.add_argument("--splits", type=str, help="Override path for situational splits.")
    parser.add_argument("--pbp", type=str, help="Override path for play-by-play / at-bat data.")
    parser.add_argument("--roster", type=str, help="Override path for roster file.")
    parser.add_argument("--teams", type=str, help="Override path for team info file.")
    parser.add_argument("--min_pa", type=int, default=100, help="Minimum PA to qualify.")
    parser.add_argument("--min_pa_leadoff", type=int, default=20, help="Minimum leadoff PA to compute share.")
    parser.add_argument("--min_pa_menon", type=int, default=30, help="Minimum men-on PA to compute share.")
    parser.add_argument("--show_all", action="store_true", help="Include players below thresholds.")
    parser.add_argument("--out", type=str, default="out/z_ABL_Table_Setter_Clearer.csv", help="Output CSV path.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    totals = load_totals(base_dir, resolve_path(base_dir, args.totals))
    splits = load_splits(base_dir, resolve_path(base_dir, args.splits))
    pbp = load_pbp_opps(base_dir, resolve_path(base_dir, args.pbp))
    opps = combine_opportunity_sources(splits, pbp)
    roster = load_roster(base_dir, resolve_path(base_dir, args.roster)).rename(
        columns={"team_id": "team_id_roster"}
    )
    team_names, team_abbrs, conf_map = load_team_info(base_dir, resolve_path(base_dir, args.teams))

    df = totals.merge(roster, on="player_id", how="left")
    if "team_id_roster" in df.columns:
        df["team_id"] = df["team_id"].combine_first(df["team_id_roster"])
        df = df.drop(columns=["team_id_roster"])
    df = df.merge(opps, on=["player_id", "team_id"], how="left", suffixes=("", "_opp"))

    df["player_name"] = df["player_name"].fillna("")
    fallback = df["player_name"] == ""
    df.loc[fallback, "player_name"] = df.loc[fallback, "player_id"].apply(
        lambda pid: str(int(pid)) if pd.notna(pid) else ""
    )
    df["OBP"] = df.apply(compute_obp, axis=1)
    df["PA_leadoff"] = pd.to_numeric(df["PA_leadoff"], errors="coerce")
    df["OBP_leadoff"] = pd.to_numeric(df["OBP_leadoff"], errors="coerce")
    df["PA_men_on"] = pd.to_numeric(df["PA_men_on"], errors="coerce")

    df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
    df["RBI"] = pd.to_numeric(df["RBI"], errors="coerce")

    df = df[df["team_id"].between(TEAM_MIN, TEAM_MAX)]

    df["leadoff_share"] = df.apply(lambda r: safe_div(r["PA_leadoff"], r["PA"]), axis=1)
    df["rbi_opp_share"] = df.apply(lambda r: safe_div(r["PA_men_on"], r["PA"]), axis=1)
    df["RBI_per_PA"] = df.apply(lambda r: safe_div(r["RBI"], r["PA"]), axis=1)
    mask_leadoff = df["PA_leadoff"] < args.min_pa_leadoff
    df.loc[mask_leadoff, ["leadoff_share", "OBP_leadoff"]] = np.nan
    mask_menon = df["PA_men_on"] < args.min_pa_menon
    df.loc[mask_menon, ["rbi_opp_share"]] = np.nan

    lg_leadoff_share = compute_avg(df.loc[df["PA_leadoff"] >= args.min_pa_leadoff, "leadoff_share"])
    lg_rbi_share = compute_avg(df.loc[df["PA_men_on"] >= args.min_pa_menon, "rbi_opp_share"])
    lg_rbi_eff = compute_avg(df["RBI_per_PA"])
    lg_obp = compute_avg(df["OBP"])
    lg_obp_leadoff = compute_avg(df["OBP_leadoff"])

    df["leadoff_plus"] = df["leadoff_share"] / lg_leadoff_share if lg_leadoff_share else np.nan
    df["rbi_opp_plus"] = df["rbi_opp_share"] / lg_rbi_share if lg_rbi_share else np.nan
    df["rbi_eff_plus"] = df["RBI_per_PA"] / lg_rbi_eff if lg_rbi_eff else np.nan
    df["onbase_plus"] = df["OBP"] / lg_obp if lg_obp else np.nan
    df["onbase_leadoff_plus"] = df["OBP_leadoff"] / lg_obp_leadoff if lg_obp_leadoff else np.nan

    df["table_setter_index"] = 0.6 * df["leadoff_plus"] + 0.4 * df["onbase_leadoff_plus"].combine_first(df["onbase_plus"])
    df["table_clearer_index"] = 0.6 * df["rbi_opp_plus"] + 0.4 * df["rbi_eff_plus"]
    df["role_score"] = df["table_clearer_index"] - df["table_setter_index"]

    def classify_role(row: pd.Series) -> str:
        setter = row["table_setter_index"]
        clearer = row["table_clearer_index"]
        score = row["role_score"]
        if pd.isna(score):
            if pd.notna(clearer) and pd.isna(setter):
                return "Table-Clearer"
            if pd.notna(setter) and pd.isna(clearer):
                return "Table-Setter"
            return ""
        if score >= 0.15:
            return "Table-Clearer"
        if score <= -0.15:
            return "Table-Setter"
        return "Balanced"

    df["role"] = df.apply(classify_role, axis=1)

    def assign_rating(score: float) -> str:
        if pd.isna(score):
            return "Unknown"
        if score >= 0.35:
            return "Cleanup Cannon"
        if score >= 0.15:
            return "Run Driver"
        if score > -0.15:
            return "Dual Threat"
        if score > -0.35:
            return "Spark Plug"
        return "Table Igniter"

    df["rating"] = df["role_score"].apply(assign_rating)
    df["team_display"] = df["team_id"].map(team_abbrs)
    fallback_names = df["team_id"].map(team_names)
    df["team_display"] = df["team_display"].fillna(fallback_names)
    df["team_display"] = df.apply(
        lambda r: r["team_display"]
        if pd.notna(r["team_display"])
        else (f"T{int(r['team_id'])}" if pd.notna(r["team_id"]) else ""),
        axis=1,
    )
    df["conf_div"] = df["team_id"].map(conf_map).fillna("")

    df = df[df["PA"] >= args.min_pa] if not args.show_all else df

    df = df.sort_values(
        by=["role_score", "table_clearer_index", "PA"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(args.base).resolve() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_df = df[
        [
            "team_id",
            "team_display",
            "player_id",
            "player_name",
            "PA",
            "PA_leadoff",
            "leadoff_share",
            "OBP_leadoff",
            "OBP",
            "PA_men_on",
            "rbi_opp_share",
            "RBI",
            "RBI_per_PA",
            "leadoff_plus",
            "onbase_plus",
            "onbase_leadoff_plus",
            "rbi_opp_plus",
            "rbi_eff_plus",
            "table_setter_index",
            "table_clearer_index",
            "role_score",
            "role",
            "rating",
            "conf_div",
        ]
    ].copy()
    for col in [
        "leadoff_share",
        "OBP_leadoff",
        "OBP",
        "rbi_opp_share",
        "RBI_per_PA",
        "leadoff_plus",
        "onbase_plus",
        "onbase_leadoff_plus",
        "rbi_opp_plus",
        "rbi_eff_plus",
        "table_setter_index",
        "table_clearer_index",
        "role_score",
    ]:
        csv_df[col] = csv_df[col].round(3)
    csv_df["PA"] = csv_df["PA"].round(0)
    csv_df["PA_leadoff"] = csv_df["PA_leadoff"].round(0)
    csv_df["PA_men_on"] = csv_df["PA_men_on"].round(0)
    csv_df.to_csv(out_path, index=False)

    text_columns = [
        ("Player", "player_name", 24, False, ""),
        ("Tm", "team_display", 6, False, ""),
        ("Conf", "conf_div", 6, False, ""),
        ("Role", "role", 13, False, ""),
        ("Rating", "rating", 14, False, ""),
        ("Setter+", "table_setter_index", 8, True, ".3f"),
        ("Clearer+", "table_clearer_index", 9, True, ".3f"),
        ("Score", "role_score", 7, True, ".3f"),
        ("PA", "PA", 5, True, ".0f"),
    ]
    text_output = text_table(
        df.head(25),
        text_columns,
        "ABL Table Setter vs Clearer",
        f"Thresholds: PA >= {args.min_pa} (leadoff share requires {args.min_pa_leadoff}+ PA; men-on share requires {args.min_pa_menon}+ PA).",
        [
            "Role labels: role_score >= +0.15 = Table-Clearer, <= -0.15 = Table-Setter, otherwise Balanced.",
            "Setter+ highlights leadoff/on-base prowess; Clearer+ highlights RBI opportunity and conversion.",
            "Ratings: Cleanup Cannon (>= +0.35), Run Driver (+0.15 to +0.34), Dual Threat (-0.14 to +0.14), Spark Plug (-0.34 to -0.15), Table Igniter (<= -0.35).",
        ],
        [
            "Leadoff share = PA_leadoff / PA; RBI opp share = PA_men_on / PA (uses PA_RISP if PA_men_on missing).",
            "Plus metrics compare player rates to league averages (1.00 = league average).",
            "Indices blend opportunity (60%) with efficiency (40%) to profile each role; PBP fallbacks estimate opportunities when splits are unavailable.",
        ],
    )
    text_path = out_path.with_suffix(".txt")
    text_path.write_text(text_output, encoding="utf-8")
    print(text_output)


if __name__ == "__main__":
    main()
