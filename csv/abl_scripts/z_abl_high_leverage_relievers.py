"""ABL High Leverage Relievers report."""

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
RELIEF_CANDIDATES = [
    "relief_splits.csv",
    "pitching_relief_splits.csv",
    "bullpen_splits.csv",
]
APP_LOG_CANDIDATES = [
    "pitching_gamelog.csv",
    "players_game_pitching_stats.csv",
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
    g_col = pick_column(df, "g", "G")
    gs_col = pick_column(df, "gs", "GS")
    gf_col = pick_column(df, "gf", "GF")
    sv_col = pick_column(df, "sv", "SV")
    bs_col = pick_column(df, "bs", "BS")
    hld_col = pick_column(df, "hld", "holds", "H", "HLD")
    er_col = pick_column(df, "er", "ER")
    so_col = pick_column(df, "so", "SO", "k", "K")
    bb_col = pick_column(df, "bb", "BB")
    hr_col = pick_column(df, "hr", "HR", "hra")
    wpa_col = pick_column(df, "wpa", "WPA")
    li_col = pick_column(df, "li", "avg_li", "pLI")
    year_col = pick_column(df, "year", "season")
    split_col = pick_column(df, "split_id", "split")
    if not id_col or not team_col:
        raise ValueError("Pitching totals missing player/team.")
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
    out["G"] = pd.to_numeric(data[g_col], errors="coerce") if g_col else np.nan
    out["GS"] = pd.to_numeric(data[gs_col], errors="coerce") if gs_col else np.nan
    out["GF"] = pd.to_numeric(data[gf_col], errors="coerce") if gf_col else np.nan
    out["SV"] = pd.to_numeric(data[sv_col], errors="coerce") if sv_col else np.nan
    out["BS"] = pd.to_numeric(data[bs_col], errors="coerce") if bs_col else np.nan
    out["HLD"] = pd.to_numeric(data[hld_col], errors="coerce") if hld_col else np.nan
    out["ER"] = pd.to_numeric(data[er_col], errors="coerce") if er_col else np.nan
    out["SO"] = pd.to_numeric(data[so_col], errors="coerce") if so_col else np.nan
    out["BB"] = pd.to_numeric(data[bb_col], errors="coerce") if bb_col else np.nan
    out["HR"] = pd.to_numeric(data[hr_col], errors="coerce") if hr_col else np.nan
    out["WPA"] = pd.to_numeric(data[wpa_col], errors="coerce") if wpa_col else np.nan
    out["avg_LI"] = pd.to_numeric(data[li_col], errors="coerce") if li_col else np.nan
    return out


def load_relief_splits(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, RELIEF_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["player_id", "team_id", "IR", "IRS", "avg_LI", "G_high_lev"])
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    ir_col = pick_column(df, "ir", "IR", "inherited_runners")
    irs_col = pick_column(df, "irs", "IRS", "inherited_runners_scored")
    li_col = pick_column(df, "li", "avg_li", "pLI")
    g_high_col = pick_column(df, "g_high_lev", "G_hiLev", "G_high")
    if not id_col or not team_col:
        return pd.DataFrame(columns=["player_id", "team_id", "IR", "IRS", "avg_LI", "G_high_lev"])
    data = df.copy()
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    out = pd.DataFrame()
    out["player_id"] = data["player_id"]
    out["team_id"] = data["team_id"]
    out["IR"] = pd.to_numeric(data[ir_col], errors="coerce") if ir_col else np.nan
    out["IRS"] = pd.to_numeric(data[irs_col], errors="coerce") if irs_col else np.nan
    out["avg_LI_relief"] = pd.to_numeric(data[li_col], errors="coerce") if li_col else np.nan
    out["G_high_lev"] = pd.to_numeric(data[g_high_col], errors="coerce") if g_high_col else np.nan
    return out


def load_app_logs(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, APP_LOG_CANDIDATES)
    if df is None:
        return pd.DataFrame()
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    li_col = pick_column(df, "li", "LI", "entry_li")
    save_flag = pick_column(df, "is_save_situation", "save_situation")
    hold_flag = pick_column(df, "is_hold_situation", "hold_situation")
    entering_lead_col = pick_column(df, "entering_lead", "entering_run_diff")
    if not id_col:
        return pd.DataFrame()
    data = df.copy()
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    if team_col:
        data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    else:
        data["team_id"] = pd.NA
    data = data.dropna(subset=["player_id"])
    data = data[(data["team_id"].isna()) | ((data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX))]
    data["entry_li"] = pd.to_numeric(data[li_col], errors="coerce") if li_col else np.nan
    for flag_col in [save_flag, hold_flag]:
        if flag_col and flag_col in data:
            data[flag_col] = pd.to_numeric(data[flag_col], errors="coerce")
    if entering_lead_col:
        data["entering_lead"] = pd.to_numeric(data[entering_lead_col], errors="coerce")
    return data


def safe_div(numer: float, denom: float) -> float:
    if pd.isna(numer) or pd.isna(denom) or denom == 0:
        return np.nan
    return numer / denom


def compute_role(row: pd.Series) -> str:
    g = row.get("G", 0) or 0
    gs = row.get("GS", 0) or 0
    gf = row.get("GF", 0) or 0
    sv = row.get("SV", 0) or 0
    if gs == 0 or ((gf + sv) >= 10) or (safe_div(gs, g) < 0.2 and gf >= 5):
        return "RP"
    if gs >= 10 or (gs >= 5 and safe_div(gs, g) >= 0.4):
        return "SP"
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
    if index >= 1.35:
        return "Fireman Supreme"
    if index >= 1.15:
        return "Door Slammer"
    if index >= 1.00:
        return "Reliable Stopper"
    if index >= 0.85:
        return "Steady Hand"
    return "Volatile"


def estimate_lev_apps(row: pd.Series, logs: pd.DataFrame, li_high: float) -> float:
    if pd.isna(row["player_id"]) or logs.empty:
        return np.nan
    pid = row["player_id"]
    player_logs = logs[logs["player_id"] == pid]
    if player_logs.empty:
        return np.nan
    if player_logs["entry_li"].notna().any():
        return (player_logs["entry_li"] >= li_high).sum()
    save_col = pick_column(player_logs, "is_save_situation", "save_situation")
    hold_col = pick_column(player_logs, "is_hold_situation", "hold_situation")
    entering_lead_col = "entering_lead" if "entering_lead" in player_logs else None
    if save_col or hold_col or entering_lead_col:
        mask = pd.Series(False, index=player_logs.index)
        if save_col:
            mask = mask | (player_logs[save_col] == 1)
        if hold_col:
            mask = mask | (player_logs[hold_col] == 1)
        if entering_lead_col:
            mask = mask | player_logs[entering_lead_col].isin([0, 1])
        return int(mask.sum())
    return np.nan


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
        lines.append("(No relievers met the qualification thresholds.)")
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
    parser = argparse.ArgumentParser(description="ABL High-Leverage Relievers report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--pitching", type=str, help="Override path for pitching totals.")
    parser.add_argument("--relief", type=str, help="Override path for relief splits.")
    parser.add_argument("--applogs", type=str, help="Override path for appearance logs.")
    parser.add_argument("--teams", type=str, help="Override path for team info.")
    parser.add_argument("--roster", type=str, help="Override path for roster file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_High_Leverage_Relievers.csv",
        help="Output CSV path (default inside out/csv_out).",
    )
    parser.add_argument("--min_ip_rp", type=float, default=15.0, help="Minimum IP for RP qualification.")
    parser.add_argument("--min_app", type=int, default=20, help="Minimum appearances for qualification.")
    parser.add_argument("--li_high", type=float, default=1.5, help="Threshold to define high-leverage entry LI.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    pitching = load_pitching_totals(base_dir, resolve_path(base_dir, args.pitching))
    roster = load_roster(base_dir, resolve_path(base_dir, args.roster))
    team_display, conf_map = load_team_info(base_dir, resolve_path(base_dir, args.teams))
    relief = load_relief_splits(base_dir, resolve_path(base_dir, args.relief))
    app_logs = load_app_logs(base_dir, resolve_path(base_dir, args.applogs))

    df = pitching.copy()
    if not roster.empty:
        df = df.merge(roster, on="player_id", how="left", suffixes=("", "_ros"))
        if "player_name_ros" in df.columns:
            df["player_name"] = df["player_name"].combine_first(df["player_name_ros"])
            df = df.drop(columns=[c for c in df.columns if c.endswith("_ros")])
    df["player_name"] = df["player_name"].fillna(df["player_id"].astype("Int64").astype(str))
    if not relief.empty:
        df = df.merge(relief, on=["player_id", "team_id"], how="left", suffixes=("", "_relief"))
        df["avg_LI"] = df["avg_LI"].combine_first(df["avg_LI_relief"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_relief")])
    else:
        df["IR"] = np.nan
        df["IRS"] = np.nan
        if "avg_LI" not in df:
            df["avg_LI"] = np.nan
    df["IR"] = df["IR"].fillna(0)
    df["IRS"] = df["IRS"].fillna(0)

    df["role"] = df.apply(compute_role, axis=1)
    df["SV_HLD"] = df[["SV", "HLD"]].sum(axis=1, min_count=1)
    df["SV_HLD_opp"] = df["SV_HLD"] + df["BS"].fillna(0)
    df["SV_HLD_rate"] = df.apply(lambda r: safe_div(r["SV_HLD"], r["SV_HLD_opp"]), axis=1)
    df["IRS_pct"] = df.apply(lambda r: safe_div(r["IRS"], r["IR"]) if r["IR"] > 0 else np.nan, axis=1)
    df["WPA_per9"] = df.apply(lambda r: safe_div(r["WPA"] * 9, r["IP"]), axis=1)
    df["WPA_per_LI"] = df.apply(lambda r: safe_div(r["WPA"], r["avg_LI"]) if r["avg_LI"] not in (0, np.nan) else np.nan, axis=1)

    if "G_high_lev" in df and df["G_high_lev"].notna().any():
        df["lev_apps"] = df["G_high_lev"]
    else:
        df["lev_apps"] = np.nan
        if not app_logs.empty:
            lev_counts = {}
            for pid in df["player_id"]:
                lev_counts[pid] = estimate_lev_apps(df[df["player_id"] == pid].iloc[0], app_logs, args.li_high)
            df["lev_apps"] = df["player_id"].map(lev_counts)

    df["qual_apps"] = df["G"].fillna(0)
    df["rank_flag"] = np.where(
        (df["role"] == "RP")
        & (
            (df["IP"].fillna(0) >= args.min_ip_rp)
            | (df["qual_apps"] >= args.min_app)
        ),
        "QUAL",
        "",
    )

    df["team_display"] = df["team_id"].map(team_display)
    df["team_display"] = df.apply(
        lambda r: r["team_display"]
        if pd.notna(r["team_display"])
        else (f"T{int(r['team_id'])}" if pd.notna(r["team_id"]) else ""),
        axis=1,
    )
    df["conf_div"] = df["team_id"].map(conf_map).fillna("")

    lg_IRS_pct = df["IRS_pct"].mean(skipna=True)
    lg_SV_HLD_rate = df["SV_HLD_rate"].mean(skipna=True)
    lg_WPA_per9 = df["WPA_per9"].mean(skipna=True)

    if pd.notna(lg_IRS_pct) and lg_IRS_pct != 1:
        df["IRS_plus"] = (1 - df["IRS_pct"]) / (1 - lg_IRS_pct)
    else:
        df["IRS_plus"] = np.nan
    df["Conv_plus"] = df["SV_HLD_rate"] / lg_SV_HLD_rate if lg_SV_HLD_rate and not np.isnan(lg_SV_HLD_rate) else np.nan
    df["WPA9_plus"] = df["WPA_per9"] / lg_WPA_per9 if lg_WPA_per9 and not np.isnan(lg_WPA_per9) else np.nan

    weight_map = {"IRS_plus": 0.45, "Conv_plus": 0.35, "WPA9_plus": 0.20}
    df["lev_eff_index"] = df.apply(
        lambda r: reweighted_average(
            {"IRS_plus": r["IRS_plus"], "Conv_plus": r["Conv_plus"], "WPA9_plus": r["WPA9_plus"]},
            weight_map,
        ),
        axis=1,
    )
    df["rating"] = df["lev_eff_index"].apply(classify_rating)

    csv_columns = [
        "team_id",
        "team_display",
        "player_id",
        "player_name",
        "role",
        "IP",
        "G",
        "GF",
        "SV",
        "HLD",
        "BS",
        "SV_HLD",
        "SV_HLD_opp",
        "SV_HLD_rate",
        "IR",
        "IRS",
        "IRS_pct",
        "avg_LI",
        "lev_apps",
        "WPA",
        "WPA_per9",
        "WPA_per_LI",
        "IRS_plus",
        "Conv_plus",
        "WPA9_plus",
        "lev_eff_index",
        "rank_flag",
        "conf_div",
        "rating",
    ]
    csv_df = df[csv_columns].copy()
    csv_df["IP"] = csv_df["IP"].round(1)
    csv_df["SV_HLD_rate"] = csv_df["SV_HLD_rate"].round(3)
    csv_df["IRS_pct"] = csv_df["IRS_pct"].round(3)
    csv_df["avg_LI"] = csv_df["avg_LI"].round(2)
    csv_df["lev_apps"] = csv_df["lev_apps"].round(0)
    csv_df["WPA"] = csv_df["WPA"].round(2)
    for col in ["WPA_per9", "WPA_per_LI", "IRS_plus", "Conv_plus", "WPA9_plus", "lev_eff_index"]:
        csv_df[col] = csv_df[col].round(3)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_path, index=False)

    display_df = df[df["rank_flag"] == "QUAL"].copy()
    display_df = display_df.sort_values(
        by=["lev_eff_index", "IRS_pct", "SV_HLD_rate"],
        ascending=[False, True, False],
        na_position="last",
    ).head(25)
    irs_str = f"{lg_IRS_pct:.3f}" if pd.notna(lg_IRS_pct) else "NA"
    conv_str = f"{lg_SV_HLD_rate:.3f}" if pd.notna(lg_SV_HLD_rate) else "NA"
    text_columns = [
        ("Player", "player_name", 24, False, ""),
        ("Team", "team_display", 10, False, ""),
        ("CD", "conf_div", 4, False, ""),
        ("Role", "role", 6, False, ""),
        ("Rating", "rating", 20, False, ""),
        ("IP", "IP", 7, True, ".1f"),
        ("IRS%", "IRS_pct", 6, True, ".3f"),
        ("Conv%", "SV_HLD_rate", 7, True, ".3f"),
        ("WPA/9", "WPA_per9", 7, True, ".3f"),
        ("Lev+", "lev_eff_index", 7, True, ".3f"),
        ("Lev Apps", "lev_apps", 9, True, ".0f"),
    ]
    subtitle_line = (
        "Grades relievers trusted with traffic and leverage: stranding inherited runners, sealing saves, and adding WPA."
    )
    context_line = f"League IRS% {irs_str}, SV/HLD% {conv_str}"
    subtitle_text = f"{subtitle_line} {context_line}".strip()
    text_output = text_table(
        display_df,
        text_columns,
        "ABL High-Leverage Relievers",
        subtitle_text,
        [
            "Ratings: Fireman Supreme (>=1.35), Door Slammer (1.15-1.34), Reliable Stopper (1.00-1.14), Steady Hand (0.85-0.99), Volatile (<0.85).",
            "Qualification: RP role with IP >= min_ip_rp or appearances >= min_app.",
        ],
        [
            "IRS% = inherited runners scored rate. Conv% = (SV+HLD)/(SV+HLD+BS).",
            "Lev Apps uses high-leverage counts when available, else proxies from save/hold situations.",
            "Leverage Effectiveness Index blends IRS+, Conv+, and WPA/9+ with weights 45/35/20.",
        ],
    )
    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() in {'csv_out'}:
        text_dir = out_path.parent.parent / "txt_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / text_filename).write_text(stamp_text_block(text_output), encoding="utf-8")
    print(text_output)


if __name__ == "__main__":
    main()

