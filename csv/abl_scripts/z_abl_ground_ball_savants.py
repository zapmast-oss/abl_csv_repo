"""ABL Ground Ball Savants report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24

BATTED_BALL_CANDIDATES = [
    "pitchers_batted_ball_against.csv",
    "pitching_batted_ball.csv",
    "pitcher_contact_profile.csv",
    "players_career_pitching_stats.csv",
]
PITCHING_CANDIDATES = [
    "player_pitching_totals.csv",
    "players_career_pitching_stats.csv",
    "pitching_players.csv",
    "players_pitching.csv",
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
FIELDING_CANDIDATES = [
    "players_career_fielding_stats.csv",
    "team_fielding_stats.csv",
]
INFIELD_POSITIONS = {"1B", "2B", "3B", "SS", "3", "4", "5", "6"}


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


def load_pitching_totals(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, PITCHING_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate pitching totals.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    ip_col = pick_column(df, "ip", "IP")
    ip_outs_col = pick_column(df, "ip_outs", "outs")
    gidp_col = pick_column(df, "gidp", "dp", "DP_induced")
    g_col = pick_column(df, "g", "G")
    gs_col = pick_column(df, "gs", "GS")
    gf_col = pick_column(df, "gf", "GF")
    sv_col = pick_column(df, "sv", "SV")
    bf_col = pick_column(df, "bf", "BF")
    year_col = pick_column(df, "year", "season")
    split_col = pick_column(df, "split_id", "split")
    if not id_col or not team_col:
        raise ValueError("Pitching totals missing key columns.")
    data = df.copy()
    if year_col:
        max_year = pd.to_numeric(data[year_col], errors="coerce").max()
        data = data[pd.to_numeric(data[year_col], errors="coerce") == max_year]
    if split_col:
        min_split = pd.to_numeric(data[split_col], errors="coerce").min()
        data = data[pd.to_numeric(data[split_col], errors="coerce") == min_split]
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    out = pd.DataFrame()
    out["player_id"] = data["player_id"]
    out["team_id"] = data["team_id"]
    if ip_outs_col and ip_outs_col in data:
        out["IP"] = pd.to_numeric(data[ip_outs_col], errors="coerce") / 3.0
    else:
        out["IP"] = pd.to_numeric(data[ip_col], errors="coerce") if ip_col else np.nan
    out["gidp"] = pd.to_numeric(data[gidp_col], errors="coerce") if gidp_col else np.nan
    out["G"] = pd.to_numeric(data[g_col], errors="coerce") if g_col else np.nan
    out["GS"] = pd.to_numeric(data[gs_col], errors="coerce") if gs_col else np.nan
    out["GF"] = pd.to_numeric(data[gf_col], errors="coerce") if gf_col else np.nan
    out["SV"] = pd.to_numeric(data[sv_col], errors="coerce") if sv_col else np.nan
    out["BF"] = pd.to_numeric(data[bf_col], errors="coerce") if bf_col else np.nan
    return out


def load_batted_ball(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, BATTED_BALL_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate pitcher batted-ball data.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    gb_col = pick_column(df, "gb_allowed", "gb", "groundballs")
    fb_col = pick_column(df, "fb_allowed", "fb", "flyballs")
    ld_col = pick_column(df, "ld_allowed", "ld", "linedrives")
    iffb_col = pick_column(df, "iffb_allowed", "iffb", "pu", "popup")
    hr_col = pick_column(df, "hr", "HR", "hra")
    goao_col = pick_column(df, "go_ao", "GO_AO_ratio")
    year_col = pick_column(df, "year", "season")
    split_col = pick_column(df, "split_id", "split")
    if not id_col or not team_col:
        raise ValueError("Batted-ball file missing player/team identifiers.")
    data = df.copy()
    if year_col:
        max_year = pd.to_numeric(data[year_col], errors="coerce").max()
        data = data[pd.to_numeric(data[year_col], errors="coerce") == max_year]
    if split_col:
        min_split = pd.to_numeric(data[split_col], errors="coerce").min()
        data = data[pd.to_numeric(data[split_col], errors="coerce") == min_split]
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    out = pd.DataFrame()
    out["player_id"] = data["player_id"]
    out["team_id"] = data["team_id"]
    out["GB_allowed"] = pd.to_numeric(data[gb_col], errors="coerce") if gb_col else np.nan
    out["FB_allowed"] = pd.to_numeric(data[fb_col], errors="coerce") if fb_col else np.nan
    out["LD_allowed"] = pd.to_numeric(data[ld_col], errors="coerce") if ld_col else np.nan
    out["IFFB_allowed"] = pd.to_numeric(data[iffb_col], errors="coerce") if iffb_col else np.nan
    out["HR_allowed"] = pd.to_numeric(data[hr_col], errors="coerce") if hr_col else np.nan
    out["GO_AO_ratio"] = pd.to_numeric(data[goao_col], errors="coerce") if goao_col else np.nan
    return out


def load_team_infield_zr(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, FIELDING_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["team_id", "team_if_zr", "team_if_zr_plus"])
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    pos_col = pick_column(df, "position", "pos")
    zr_col = pick_column(df, "zr", "zone_rating")
    year_col = pick_column(df, "year", "season")
    if not team_col or not zr_col:
        return pd.DataFrame(columns=["team_id", "team_if_zr", "team_if_zr_plus"])
    data = df.copy()
    if year_col:
        max_year = pd.to_numeric(data[year_col], errors="coerce").max()
        data = data[pd.to_numeric(data[year_col], errors="coerce") == max_year]
    if pos_col:
        data = data[data[pos_col].astype(str).isin(INFIELD_POSITIONS)]
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    data["ZR"] = pd.to_numeric(data[zr_col], errors="coerce")
    agg = data.groupby("team_id", as_index=False)["ZR"].sum()
    agg.rename(columns={"ZR": "team_if_zr"}, inplace=True)
    league_mean = agg["team_if_zr"].mean()
    if league_mean and not np.isnan(league_mean):
        agg["team_if_zr_plus"] = agg["team_if_zr"] / league_mean
    else:
        agg["team_if_zr_plus"] = np.nan
    return agg


def safe_div(numer: float, denom: float) -> float:
    if pd.isna(numer) or pd.isna(denom) or denom == 0:
        return np.nan
    return numer / denom


def compute_role(row: pd.Series) -> str:
    g = row.get("G", 0) or 0
    gs = row.get("GS", 0) or 0
    gf = row.get("GF", 0) or 0
    sv = row.get("SV", 0) or 0
    if gs >= 10 or (gs >= 5 and safe_div(gs, g) >= 0.4):
        return "SP"
    if gs == 0 or ((gf + sv) >= 10 and safe_div(gs, g) < 0.2):
        return "RP"
    return "Swing"


def reweighted_average(values: Dict[str, float], weights: Dict[str, float]) -> float:
    valid = {k: v for k, v in values.items() if pd.notna(v)}
    if not valid:
        return np.nan
    total_weight = sum(weights[k] for k in valid)
    if total_weight == 0:
        return np.nan
    return sum(values[k] * (weights[k] / total_weight) for k in valid)


def classify_rating(index: float) -> str:
    if pd.isna(index):
        return "Unknown"
    if index >= 1.30:
        return "Wormburner Supreme"
    if index >= 1.15:
        return "Dirt Artist"
    if index >= 1.00:
        return "Ground Boss"
    if index >= 0.85:
        return "Steady Roller"
    return "Flyer Risk"


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
        lines.append("(No pitchers met the qualification thresholds.)")
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
    parser = argparse.ArgumentParser(description="ABL Ground Ball Savants report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--battedball", type=str, help="Override path for pitcher batted-ball data.")
    parser.add_argument("--pitching", type=str, help="Override path for pitching totals.")
    parser.add_argument("--fielding", type=str, help="Override path for fielding/ZR data.")
    parser.add_argument("--teams", type=str, help="Override path for team info.")
    parser.add_argument("--roster", type=str, help="Override path for roster file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Ground_Ball_Savants.csv",
        help="Output CSV path (default inside out/csv_out).",
    )
    parser.add_argument("--min_ip_sp", type=float, default=30.0, help="Minimum IP for SP/Swing qualification.")
    parser.add_argument("--min_ip_rp", type=float, default=15.0, help="Minimum IP for RP qualification.")
    parser.add_argument("--min_bip_allowed", type=int, default=120, help="Minimum opponent BIP to stabilize GB%.")
    parser.add_argument("--min_fb_allowed", type=int, default=30, help="Minimum fly balls for HR/FB.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    batted = load_batted_ball(base_dir, resolve_path(base_dir, args.battedball))
    pitching = load_pitching_totals(base_dir, resolve_path(base_dir, args.pitching))
    roster = load_roster(base_dir, resolve_path(base_dir, args.roster))
    team_display, conf_map = load_team_info(base_dir, resolve_path(base_dir, args.teams))
    infield_zr = load_team_infield_zr(base_dir, resolve_path(base_dir, args.fielding))

    df = batted.merge(pitching, on=["player_id", "team_id"], how="inner", suffixes=("", "_tot"))
    if "player_name" not in df.columns:
        df["player_name"] = np.nan
    if not roster.empty:
        df = df.merge(roster, on="player_id", how="left", suffixes=("", "_ros"))
        if "player_name_ros" in df.columns:
            df["player_name"] = df["player_name"].combine_first(df["player_name_ros"])
            df = df.drop(columns=[c for c in df.columns if c.endswith("_ros")])
    else:
        df["player_name"] = df["player_name"].fillna(df["player_id"].astype(str))

    df["player_name"] = df["player_name"].fillna("")
    mask_blank = df["player_name"].str.strip() == ""
    df.loc[mask_blank, "player_name"] = df.loc[mask_blank, "player_id"].astype("Int64").astype(str)

    gb_calc = df["GB_allowed"].fillna(0)
    fb_calc = df["FB_allowed"].fillna(0)
    ld_calc = df["LD_allowed"].fillna(0)
    iffb_calc = df["IFFB_allowed"].fillna(0)
    hr_calc = df["HR_allowed"].fillna(0)

    df["BIP_allowed"] = gb_calc + fb_calc + ld_calc + iffb_calc
    df["GB_pct"] = np.where(df["BIP_allowed"] > 0, gb_calc / df["BIP_allowed"], np.nan)
    df["OF_FB"] = np.where(
        df["FB_allowed"].notna(),
        np.maximum(df["FB_allowed"].fillna(0) - df["IFFB_allowed"].fillna(0), 0),
        np.nan,
    )
    df["HR_per_FB"] = np.where(
        (df["OF_FB"] >= args.min_fb_allowed) & df["OF_FB"].notna(),
        hr_calc / df["OF_FB"],
        np.nan,
    )
    df["DP_per9"] = df.apply(lambda r: safe_div(r["gidp"] * 9, r["IP"]), axis=1)
    df["role"] = df.apply(compute_role, axis=1)
    df["rank_flag"] = ""
    qual_mask = (
        ((df["role"].isin(["SP", "Swing"])) & (df["IP"] >= args.min_ip_sp))
        | ((df["role"] == "RP") & (df["IP"] >= args.min_ip_rp))
    )
    df.loc[qual_mask, "rank_flag"] = "QUAL"

    df["team_display"] = df["team_id"].map(team_display)
    df["team_display"] = df.apply(
        lambda r: r["team_display"]
        if pd.notna(r["team_display"])
        else (f"T{int(r['team_id'])}" if pd.notna(r["team_id"]) else ""),
        axis=1,
    )
    df["conf_div"] = df["team_id"].map(conf_map).fillna("")

    if not infield_zr.empty:
        df = df.merge(infield_zr, on="team_id", how="left")
    else:
        df["team_if_zr"] = np.nan
        df["team_if_zr_plus"] = np.nan

    gb_mask = df["BIP_allowed"] >= args.min_bip_allowed
    lg_GB_pct = df.loc[gb_mask, "GB_pct"].mean(skipna=True)
    lg_DP_per9 = df["DP_per9"].mean(skipna=True)
    hr_mask = df["HR_per_FB"].notna()
    lg_HR_per_FB = df.loc[hr_mask, "HR_per_FB"].mean(skipna=True)

    df["GB_plus"] = df["GB_pct"] / lg_GB_pct if lg_GB_pct and not np.isnan(lg_GB_pct) else np.nan
    df["DP9_plus"] = df["DP_per9"] / lg_DP_per9 if lg_DP_per9 and not np.isnan(lg_DP_per9) else np.nan
    df["HRFB_plus"] = df["HR_per_FB"] / lg_HR_per_FB if lg_HR_per_FB and not np.isnan(lg_HR_per_FB) else np.nan

    weight_map = {"GB_plus": 0.6, "DP9_plus": 0.3, "HR_component": 0.1}

    def compute_savant(row: pd.Series) -> float:
        components = {
            "GB_plus": row["GB_plus"],
            "DP9_plus": row["DP9_plus"],
            "HR_component": 1 / row["HRFB_plus"] if pd.notna(row["HRFB_plus"]) and row["HRFB_plus"] != 0 else np.nan,
        }
        return reweighted_average(components, weight_map)

    df["savant_index"] = df.apply(compute_savant, axis=1)
    df["rating"] = df["savant_index"].apply(classify_rating)

    df["badge_sinkerballer"] = np.where(
        (df["GB_plus"] >= 1.20)
        & pd.notna(df["HR_per_FB"])
        & pd.notna(lg_HR_per_FB)
        & (df["HR_per_FB"] <= 0.95 * lg_HR_per_FB),
        "Y",
        "",
    )
    df["badge_dp_machine"] = np.where(df["DP9_plus"] >= 1.35, "Y", "")

    csv_columns = [
        "team_id",
        "team_display",
        "player_id",
        "player_name",
        "role",
        "IP",
        "BIP_allowed",
        "GB_allowed",
        "FB_allowed",
        "IFFB_allowed",
        "GB_pct",
        "OF_FB",
        "HR_allowed",
        "HR_per_FB",
        "gidp",
        "DP_per9",
        "GO_AO_ratio",
        "GB_plus",
        "DP9_plus",
        "HRFB_plus",
        "savant_index",
        "badge_sinkerballer",
        "badge_dp_machine",
        "team_if_zr",
        "team_if_zr_plus",
        "rank_flag",
        "conf_div",
        "rating",
    ]
    csv_df = df[csv_columns].copy()
    csv_df["IP"] = csv_df["IP"].round(1)
    for col in ["GB_pct", "HR_per_FB", "DP_per9", "GO_AO_ratio", "GB_plus", "DP9_plus", "HRFB_plus", "savant_index", "team_if_zr_plus"]:
        csv_df[col] = csv_df[col].round(3)
    csv_df["team_if_zr"] = csv_df["team_if_zr"].round(1)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_path, index=False)

    display_df = df[df["rank_flag"] == "QUAL"].copy()
    display_df = display_df.sort_values(
        by=["savant_index", "GB_pct", "DP_per9"],
        ascending=[False, False, False],
        na_position="last",
    ).head(25)

    gb_str = f"{lg_GB_pct:.3f}" if pd.notna(lg_GB_pct) else "NA"
    dp_str = f"{lg_DP_per9:.3f}" if pd.notna(lg_DP_per9) else "NA"
    text_columns = [
        ("Player", "player_name", 24, False, ""),
        ("Team", "team_display", 10, False, ""),
        ("CD", "conf_div", 4, False, ""),
        ("Role", "role", 6, False, ""),
        ("Rating", "rating", 18, False, ""),
        ("IP", "IP", 7, True, ".1f"),
        ("GB%", "GB_pct", 7, True, ".3f"),
        ("DP/9", "DP_per9", 7, True, ".3f"),
        ("HR/FB", "HR_per_FB", 7, True, ".3f"),
        ("Savant+", "savant_index", 8, True, ".3f"),
        ("Team IF ZR", "team_if_zr", 11, True, ".1f"),
    ]
    subtitle_line = (
        "Spotlights sinker specialists who keep contact on the ground, turn double plays, and choke off HR damage."
    )
    context_line = f"League GB% {gb_str}, DP/9 {dp_str}"
    subtitle_text = f"{subtitle_line} {context_line}".strip()
    text_output = text_table(
        display_df,
        text_columns,
        "ABL Ground Ball Savants",
        subtitle_text,
        [
            "Ratings: Wormburner Supreme (>=1.30), Dirt Artist (1.15-1.29), Ground Boss (1.00-1.14), Steady Roller (0.85-0.99), Flyer Risk (<0.85).",
            "Badges: Sinkerballer (GB+ >=1.20 & HR/FB <= 95% of league), DP Machine (DP9+ >=1.35).",
        ],
        [
            "GB% = GB / opponent BIP; DP/9 from GIDP induced. HR/FB uses OF-only flyballs when available.",
            "Savant Index blends GB dominance (60%), DP knack (30%), and HR avoidance (10% inverse).",
            f"Infield synergy uses summed ZR at 1B-3B-SS scaled vs league (team_if_zr_plus).",
        ],
    )
    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() in {'csv_out'}:
        text_dir = out_path.parent.parent / "text_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / text_filename).write_text(stamp_text_block(text_output), encoding="utf-8")
    print(text_output)


if __name__ == "__main__":
    main()

