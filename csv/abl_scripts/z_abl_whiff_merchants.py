"""ABL Whiff Merchants report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24

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
PITCH_DETAIL_CANDIDATES = [
    "pitch_results.csv",
    "pitch_type_summary.csv",
    "pitcher_pitch_detail.csv",
    "pfx.csv",
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


def load_team_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    name_col = pick_column(df, "abbr", "team_abbr", "short_name", "team_display", "team_name", "name")
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


def load_pitching(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, PITCHING_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate pitching totals.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    first_col = pick_column(df, "first_name", "firstname")
    last_col = pick_column(df, "last_name", "lastname")
    full_col = pick_column(df, "name_full", "name", "player_name")
    g_col = pick_column(df, "g", "G")
    gs_col = pick_column(df, "gs", "GS")
    gf_col = pick_column(df, "gf", "GF")
    sv_col = pick_column(df, "sv", "SV")
    ip_col = pick_column(df, "ip", "IP")
    ip_outs_col = pick_column(df, "ip_outs", "outs")
    bf_col = pick_column(df, "bf", "BF")
    so_col = pick_column(df, "so", "SO", "k", "K")
    bb_col = pick_column(df, "bb", "BB")
    year_col = pick_column(df, "year", "season")
    split_col = pick_column(df, "split_id", "split")
    if not id_col or not team_col:
        raise ValueError("Pitching file missing key columns.")
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
    out["G"] = pd.to_numeric(data[g_col], errors="coerce") if g_col else np.nan
    out["GS"] = pd.to_numeric(data[gs_col], errors="coerce") if gs_col else np.nan
    out["GF"] = pd.to_numeric(data[gf_col], errors="coerce") if gf_col else np.nan
    out["SV"] = pd.to_numeric(data[sv_col], errors="coerce") if sv_col else np.nan
    out["BF"] = pd.to_numeric(data[bf_col], errors="coerce") if bf_col else np.nan
    out["SO"] = pd.to_numeric(data[so_col], errors="coerce") if so_col else np.nan
    out["BB"] = pd.to_numeric(data[bb_col], errors="coerce") if bb_col else np.nan
    if ip_outs_col and ip_outs_col in data:
        ip = pd.to_numeric(data[ip_outs_col], errors="coerce") / 3.0
    else:
        ip = pd.to_numeric(data[ip_col], errors="coerce") if ip_col else np.nan
    out["IP"] = ip
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


def load_pitch_detail(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, PITCH_DETAIL_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["player_id", "pitch_type", "pitches", "cs", "swstr", "ball", "foul", "swings"])
    player_col = pick_column(df, "player_id", "playerid", "batter_pitcher_id")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    type_col = pick_column(df, "pitch_type", "pitch", "type", "pitch_code")
    pitches_col = pick_column(df, "pitches", "total_pitches", "pitch_count")
    called_col = pick_column(df, "cs", "called_strike", "called")
    whiff_col = pick_column(df, "swstr", "swinging_strike", "whiff")
    foul_col = pick_column(df, "foul", "fouls")
    ball_col = pick_column(df, "ball", "balls")
    ip_col = pick_column(df, "in_play", "ip")
    swings_col = pick_column(df, "swings", "swing")
    if not player_col or not type_col or not pitches_col:
        return pd.DataFrame(columns=["player_id", "pitch_type", "pitches", "cs", "swstr", "ball", "foul", "swings", "team_id"])
    data = df.copy()
    data["player_id"] = pd.to_numeric(data[player_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id"])
    if team_col:
        data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    else:
        data["team_id"] = pd.NA
    out = pd.DataFrame()
    out["player_id"] = data["player_id"]
    out["team_id"] = data["team_id"]
    out["pitch_type"] = data[type_col].fillna("").astype(str)
    out["pitches"] = pd.to_numeric(data[pitches_col], errors="coerce")
    out["cs"] = pd.to_numeric(data[called_col], errors="coerce") if called_col else np.nan
    out["swstr"] = pd.to_numeric(data[whiff_col], errors="coerce") if whiff_col else np.nan
    out["foul"] = pd.to_numeric(data[foul_col], errors="coerce") if foul_col else np.nan
    out["ball"] = pd.to_numeric(data[ball_col], errors="coerce") if ball_col else np.nan
    out["in_play"] = pd.to_numeric(data[ip_col], errors="coerce") if ip_col else np.nan
    out["swings"] = pd.to_numeric(data[swings_col], errors="coerce") if swings_col else np.nan
    return out


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
        return "Whiff Wizard"
    if index >= 1.15:
        return "K Maestro"
    if index >= 1.00:
        return "Miss Maker"
    if index >= 0.85:
        return "Steady"
    return "Needs Bite"


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
    parser = argparse.ArgumentParser(description="ABL Whiff Merchants report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--pitching", type=str, help="Override path for pitching totals.")
    parser.add_argument("--pitchdetail", type=str, help="Override path for pitch-type summary.")
    parser.add_argument("--teams", type=str, help="Override path for team info.")
    parser.add_argument("--roster", type=str, help="Override path for roster file.")
    parser.add_argument("--out", type=str, default="out/csv_out/z_ABL_Whiff_Merchants.csv", help="Output CSV path.")
    parser.add_argument("--min_ip_sp", type=float, default=30.0, help="Minimum IP for SP/Swing qualification.")
    parser.add_argument("--min_ip_rp", type=float, default=15.0, help="Minimum IP for RP qualification.")
    parser.add_argument("--min_pitches_total", type=int, default=300, help="Minimum total pitches for CSW stability.")
    parser.add_argument("--min_pitches_type", type=int, default=100, help="Minimum pitches for top pitch consideration.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    pitching = load_pitching(base_dir, resolve_path(base_dir, args.pitching))
    roster = load_roster(base_dir, resolve_path(base_dir, args.roster))
    team_display, conf_map = load_team_info(base_dir, resolve_path(base_dir, args.teams))
    pitch_detail = load_pitch_detail(base_dir, resolve_path(base_dir, args.pitchdetail))

    df = pitching.copy()
    if not roster.empty:
        df = df.merge(roster, on="player_id", how="left", suffixes=("", "_ros"))
        df["player_name"] = df["player_name_ros"].combine_first(df["player_name"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_ros")])

    df["player_name"] = df["player_name"].fillna("")
    blank_mask = df["player_name"].str.strip() == ""
    df.loc[blank_mask, "player_name"] = df.loc[blank_mask, "player_id"].astype("Int64").astype(str)
    df = df[df["IP"].notna() & (df["IP"] > 0)]

    df["K_pct"] = df.apply(lambda r: safe_div(r["SO"], r["BF"]), axis=1)
    df["K_per9"] = df.apply(lambda r: safe_div(r["SO"] * 9, r["IP"]), axis=1)
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

    pitches_totals = pd.DataFrame()
    pitch_types = pd.DataFrame()
    if not pitch_detail.empty:
        pitch_detail = pitch_detail[pitch_detail["pitches"].notna()]
        pitch_detail = pitch_detail[pitch_detail["pitches"] > 0]
        if not pitch_detail.empty:
            pitch_types = pitch_detail.copy()
            pitches_totals = pitch_detail.groupby("player_id", as_index=False).agg(
                pitches_total=("pitches", "sum"),
                cs_total=("cs", "sum"),
                swstr_total=("swstr", "sum"),
            )
            df = df.merge(pitches_totals, on="player_id", how="left")
            df["CSW_pct"] = (df["cs_total"] + df["swstr_total"]) / df["pitches_total"]
        else:
            df["pitches_total"] = np.nan
            df["CSW_pct"] = np.nan
    else:
        df["pitches_total"] = np.nan
        df["CSW_pct"] = np.nan

    # Top pitch selection
    top_pitch_columns = {
        "top_pitch_type": "",
        "top_pitch_usage_pct": np.nan,
        "top_pitch_csw_pct": np.nan,
        "top_pitch_whiff_pct": np.nan,
    }
    top_pitch_df = pd.DataFrame()
    if not pitch_types.empty:
        pitch_types = pitch_types.merge(df[["player_id", "pitches_total"]], on="player_id", how="left")
        pitch_types["usage_pct"] = pitch_types.apply(
            lambda r: safe_div(r["pitches"], r["pitches_total"]), axis=1
        )
        pitch_types["csw_pct"] = (pitch_types["cs"].fillna(0) + pitch_types["swstr"].fillna(0)) / pitch_types["pitches"]
        pitch_types["swings_calc"] = pitch_types["swings"]
        missing_swings = pitch_types["swings_calc"].isna()
        if "ball" in pitch_types.columns and "cs" in pitch_types.columns:
            derived_swings = pitch_types["pitches"] - (
                pitch_types["ball"].fillna(0)
                + pitch_types["cs"].fillna(0)
                + pitch_types["in_play"].fillna(0)
                + pitch_types["foul"].fillna(0)
            )
            pitch_types.loc[missing_swings, "swings_calc"] = derived_swings[missing_swings]
        pitch_types["whiff_pct"] = pitch_types.apply(
            lambda r: safe_div(r["swstr"], r["swings_calc"]), axis=1
        )
        eligible = pitch_types[pitch_types["pitches"] >= args.min_pitches_type].copy()
        if not eligible.empty:
            def pick_top(group: pd.DataFrame) -> pd.Series:
                ordered = group.sort_values(
                    by=["csw_pct", "whiff_pct", "usage_pct"],
                    ascending=[False, False, False],
                )
                best = ordered.iloc[0]
                return pd.Series(
                    {
                        "top_pitch_type": best["pitch_type"],
                        "top_pitch_usage_pct": best["usage_pct"],
                        "top_pitch_csw_pct": best["csw_pct"],
                        "top_pitch_whiff_pct": best["whiff_pct"],
                    }
                )

            top_pitch_df = eligible.groupby("player_id").apply(pick_top).reset_index()
            df = df.merge(top_pitch_df, on="player_id", how="left")
        else:
            for col, val in top_pitch_columns.items():
                df[col] = val
    else:
        for col, val in top_pitch_columns.items():
            df[col] = val

    lg_K_pct = df["K_pct"].mean(skipna=True)
    lg_K_per9 = df["K_per9"].mean(skipna=True)
    stable_csw = df["pitches_total"] >= args.min_pitches_total
    lg_CSW_pct = df.loc[stable_csw, "CSW_pct"].mean(skipna=True)

    df["Kpct_plus"] = df["K_pct"] / lg_K_pct if lg_K_pct and not np.isnan(lg_K_pct) else np.nan
    df["K9_plus"] = df["K_per9"] / lg_K_per9 if lg_K_per9 and not np.isnan(lg_K_per9) else np.nan
    df["CSW_plus"] = np.where(
        stable_csw & pd.notna(df["CSW_pct"]) & pd.notna(lg_CSW_pct) & (lg_CSW_pct != 0),
        df["CSW_pct"] / lg_CSW_pct,
        np.nan,
    )

    weight_map = {"Kpct_plus": 0.5, "CSW_plus": 0.3, "K9_plus": 0.2}
    df["whiff_index"] = df.apply(
        lambda r: reweighted_average(
            {"Kpct_plus": r["Kpct_plus"], "CSW_plus": r["CSW_plus"], "K9_plus": r["K9_plus"]},
            weight_map,
        ),
        axis=1,
    )
    df["rating"] = df["whiff_index"].apply(classify_rating)

    csv_columns = [
        "team_id",
        "team_display",
        "conf_div",
        "player_id",
        "player_name",
        "role",
        "IP",
        "BF",
        "SO",
        "K_pct",
        "K_per9",
        "pitches_total",
        "CSW_pct",
        "Kpct_plus",
        "K9_plus",
        "CSW_plus",
        "whiff_index",
        "top_pitch_type",
        "top_pitch_usage_pct",
        "top_pitch_csw_pct",
        "top_pitch_whiff_pct",
        "perf_flag",
        "rank_flag",
        "rating",
    ]
    df["perf_flag"] = ""
    df.loc[df["whiff_index"] >= 1.15, "perf_flag"] = "ELITE"
    df.loc[(df["whiff_index"] >= 1.00) & (df["whiff_index"] < 1.15), "perf_flag"] = "IMPACT"
    df.loc[(df["whiff_index"] >= 0.85) & (df["whiff_index"] < 1.00), "perf_flag"] = "STEADY"

    csv_df = df[csv_columns].copy()
    csv_df["IP"] = csv_df["IP"].round(1)
    csv_df["K_pct"] = csv_df["K_pct"].round(3)
    csv_df["K_per9"] = csv_df["K_per9"].round(2)
    csv_df["CSW_pct"] = csv_df["CSW_pct"].round(3)
    csv_df["Kpct_plus"] = csv_df["Kpct_plus"].round(3)
    csv_df["K9_plus"] = csv_df["K9_plus"].round(3)
    csv_df["CSW_plus"] = csv_df["CSW_plus"].round(3)
    csv_df["whiff_index"] = csv_df["whiff_index"].round(3)
    csv_df["top_pitch_usage_pct"] = csv_df["top_pitch_usage_pct"].round(3)
    csv_df["top_pitch_csw_pct"] = csv_df["top_pitch_csw_pct"].round(3)
    csv_df["top_pitch_whiff_pct"] = csv_df["top_pitch_whiff_pct"].round(3)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_path, index=False)

    display_df = df[df["rank_flag"] == "QUAL"].copy()
    display_df = display_df.sort_values(
        by=["whiff_index", "K_pct", "CSW_pct"],
        ascending=[False, False, False],
        na_position="last",
    ).head(25)

    text_columns = [
        ("Player", "player_name", 24, False, ""),
        ("Tm", "team_display", 5, False, ""),
        ("Conf", "conf_div", 6, False, ""),
        ("Role", "role", 6, False, ""),
        ("Rating", "rating", 13, False, ""),
        ("IP", "IP", 6, True, ".1f"),
        ("K%", "K_pct", 6, True, ".3f"),
        ("K/9", "K_per9", 6, True, ".2f"),
        ("CSW%", "CSW_pct", 6, True, ".3f"),
        ("Whiff+", "whiff_index", 7, True, ".3f"),
        ("Top Pitch", "top_pitch_type", 10, False, ""),
    ]
    lg_k_pct_str = f"{lg_K_pct:.3f}" if pd.notna(lg_K_pct) else "NA"
    lg_csw_str = f"{lg_CSW_pct:.3f}" if pd.notna(lg_CSW_pct) else "NA"
    text_output = text_table(
        display_df,
        text_columns,
        "ABL Whiff Merchants",
        f"League K% {lg_k_pct_str}, CSW% stable base {lg_csw_str}",
        [
            "Ratings: Whiff Wizard (>=1.30), K Maestro (1.15-1.29), Miss Maker (1.00-1.14), Steady (0.85-0.99), Needs Bite (<0.85).",
            "perf_flag: ELITE (>=1.15), IMPACT (1.00-1.14), STEADY (0.85-0.99).",
        ],
        [
            "K% = strikeouts per batter faced; K/9 = strikeouts per 9 IP.",
            "CSW% = (Called Strikes + Whiffs) / total pitches (needs >= min_pitches_total).",
            "Whiff Index blends K%+, CSW%+, and K/9+ (weights 50/30/20, reweighted when missing).",
            f"Top pitch requires >= {args.min_pitches_type} pitches; usage/CSW/whiff derived per type.",
        ],
    )
    txt_dir = base_dir / "out" / "text_out"
    txt_dir.mkdir(parents=True, exist_ok=True)
    text_path = txt_dir / out_path.with_suffix(".txt").name
    text_path.write_text(stamp_text_block(text_output), encoding="utf-8")
    print(text_output)


if __name__ == "__main__":
    main()
