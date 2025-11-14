
"""ABL FIP vs ERA gap report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24

PITCH_CANDIDATES = [
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
    abbr_col = pick_column(df, "abbr", "team_abbr", "short_name")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
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
        if abbr_col and pd.notna(row.get(abbr_col)):
            names[tid] = str(row.get(abbr_col))
        elif name_col and pd.notna(row.get(name_col)):
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
    df = read_first(base, override, PITCH_CANDIDATES)
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
    ip_outs_col = pick_column(df, "ip_outs", "IP_outs")
    era_col = pick_column(df, "era", "ERA")
    er_col = pick_column(df, "er", "ER")
    hr_col = pick_column(df, "hr", "HR", "hra")
    so_col = pick_column(df, "so", "SO", "k", "K")
    bb_col = pick_column(df, "bb", "BB")
    ibb_col = pick_column(df, "ibb", "IBB", "iw")
    hbp_col = pick_column(df, "hbp", "HBP", "hp")
    bf_col = pick_column(df, "bf", "BF")
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
    out["ER"] = pd.to_numeric(data[er_col], errors="coerce") if er_col else np.nan
    out["HR"] = pd.to_numeric(data[hr_col], errors="coerce") if hr_col else np.nan
    out["SO"] = pd.to_numeric(data[so_col], errors="coerce") if so_col else np.nan
    out["BB"] = pd.to_numeric(data[bb_col], errors="coerce") if bb_col else np.nan
    out["IBB"] = pd.to_numeric(data[ibb_col], errors="coerce") if ibb_col else np.nan
    out["HBP"] = pd.to_numeric(data[hbp_col], errors="coerce") if hbp_col else np.nan
    out["BF"] = pd.to_numeric(data[bf_col], errors="coerce") if bf_col else np.nan
    if ip_outs_col and ip_outs_col in data:
        ip = pd.to_numeric(data[ip_outs_col], errors="coerce") / 3.0
    else:
        ip = pd.to_numeric(data[ip_col], errors="coerce") if ip_col else np.nan
    out["IP"] = ip
    if era_col:
        out["ERA_raw"] = pd.to_numeric(data[era_col], errors="coerce")
    else:
        out["ERA_raw"] = np.nan
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


def classify_rating(delta: float) -> str:
    if pd.isna(delta):
        return "Unknown"
    if delta >= 1.00:
        return "Tough Luck"
    if delta >= 0.50:
        return "Headwind"
    if delta > -0.50:
        return "Neutral"
    if delta > -1.00:
        return "Tailwind"
    return "Magic Touch"


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
    parser = argparse.ArgumentParser(description="ABL FIP vs ERA gap report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--pitching", type=str, help="Override path for pitching totals.")
    parser.add_argument("--teams", type=str, help="Override path for team info.")
    parser.add_argument("--roster", type=str, help="Override path for roster file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_FIP_vs_ERA_Gap.csv",
        help="Output CSV path (default inside out/csv_out).",
    )
    parser.add_argument("--min_ip_sp", type=float, default=30.0, help="Minimum IP for SP/Swing qualification.")
    parser.add_argument("--min_ip_rp", type=float, default=15.0, help="Minimum IP for RP qualification.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    pitching = load_pitching(base_dir, resolve_path(base_dir, args.pitching))
    roster = load_roster(base_dir, resolve_path(base_dir, args.roster))
    team_display, conf_map = load_team_info(base_dir, resolve_path(base_dir, args.teams))

    pitching = pitching.dropna(subset=["IP"])
    pitching = pitching[pitching["IP"] > 0]

    df = pitching.copy()
    if not roster.empty:
        df = df.merge(roster, on="player_id", how="left", suffixes=("", "_roster"))
        df["player_name"] = df["player_name_roster"].combine_first(df.get("player_name"))
        df = df.drop(columns=[col for col in df.columns if col.endswith("_roster")])
    df["player_name"] = df["player_name"].fillna("")
    blank_mask = df["player_name"].str.strip() == ""
    df.loc[blank_mask, "player_name"] = df.loc[blank_mask, "player_id"].astype("Int64").astype(str)
    df["ERA"] = df["ERA_raw"]
    need_era = df["ERA"].isna() & df["ER"].notna() & df["IP"].notna()
    df.loc[need_era, "ERA"] = df.loc[need_era, :].apply(lambda r: safe_div(r["ER"] * 9, r["IP"]), axis=1)
    bb_used = df["BB"]
    if "IBB" in df:
        bb_used = (df["BB"] - df["IBB"]).clip(lower=0)
    df["BB_used"] = bb_used
    df["role"] = df.apply(compute_role, axis=1)
    df["team_display"] = df["team_id"].map(team_display)
    df["team_display"] = df.apply(
        lambda r: r["team_display"]
        if pd.notna(r["team_display"])
        else (f"T{int(r['team_id'])}" if pd.notna(r["team_id"]) else ""),
        axis=1,
    )
    df["conf_div"] = df["team_id"].map(conf_map).fillna("")

    # League totals for FIP constant.
    lg_df = df.copy()
    lg_IP = lg_df["IP"].sum(min_count=1)
    lg_ER = lg_df["ER"].sum(min_count=1)
    lg_HR = lg_df["HR"].sum(min_count=1)
    lg_BB = lg_df["BB_used"].sum(min_count=1)
    lg_SO = lg_df["SO"].sum(min_count=1)
    lg_ERA = safe_div(lg_ER * 9, lg_IP) if pd.notna(lg_ER) else np.nan
    if pd.isna(lg_ERA):
        lg_ERA = safe_div((df["ERA"] * df["IP"]).sum(min_count=1), lg_IP)
    if any(pd.isna(x) for x in [lg_IP, lg_ERA, lg_HR, lg_BB, lg_SO]) or lg_IP == 0:
        c = 3.10
        c_note = "(fallback constant 3.10 used; incomplete league data)"
    else:
        c = lg_ERA - safe_div(13 * lg_HR + 3 * lg_BB - 2 * lg_SO, lg_IP)
        c_note = ""

    df["FIP"] = (13 * df["HR"] + 3 * df["BB_used"] - 2 * df["SO"]) / df["IP"] + c
    df.loc[df["IP"] <= 0, "FIP"] = np.nan
    df["delta"] = df["ERA"] - df["FIP"]
    df["K_pct"] = df.apply(lambda r: safe_div(r["SO"], r["BF"]), axis=1)
    df["BB_pct"] = df.apply(lambda r: safe_div(r["BB"], r["BF"]), axis=1)
    df["perf_flag"] = df["delta"].apply(
        lambda d: "OVERPERFORM" if pd.notna(d) and d <= -0.50 else ("UNDERPERFORM" if pd.notna(d) and d >= 0.50 else "")
    )
    df["rating"] = df["delta"].apply(classify_rating)
    df["rank_flag"] = ""
    qual_mask = (
        ((df["role"].isin(["SP", "Swing"])) & (df["IP"] >= args.min_ip_sp))
        | ((df["role"] == "RP") & (df["IP"] >= args.min_ip_rp))
    )
    df.loc[qual_mask, "rank_flag"] = "QUAL"
    df["FIP_constant_c"] = c

    csv_columns = [
        "team_id",
        "team_display",
        "conf_div",
        "player_id",
        "player_name",
        "role",
        "G",
        "GS",
        "IP",
        "ERA",
        "FIP",
        "delta",
        "SO",
        "BB",
        "HR",
        "BF",
        "K_pct",
        "BB_pct",
        "perf_flag",
        "rank_flag",
        "rating",
        "FIP_constant_c",
    ]
    csv_df = df[csv_columns].copy()
    csv_df["IP"] = csv_df["IP"].round(1)
    csv_df["ERA"] = csv_df["ERA"].round(2)
    csv_df["FIP"] = csv_df["FIP"].round(2)
    csv_df["delta"] = csv_df["delta"].round(2)
    csv_df["K_pct"] = csv_df["K_pct"].round(3)
    csv_df["BB_pct"] = csv_df["BB_pct"].round(3)
    if pd.notna(c):
        csv_df["FIP_constant_c"] = round(float(c), 3)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_path, index=False)

    display_df = df[df["rank_flag"] == "QUAL"].copy()
    display_df = display_df.sort_values(
        by=["delta", "IP", "ERA"],
        ascending=[False, False, False],
        na_position="last",
    ).head(25)

    text_columns = [
        ("Player", "player_name", 24, False, ""),
        ("Team", "team_display", 10, False, ""),
        ("CD", "conf_div", 4, False, ""),
        ("Role", "role", 6, False, ""),
        ("Rating", "rating", 14, False, ""),
        ("IP", "IP", 7, True, ".1f"),
        ("ERA", "ERA", 7, True, ".2f"),
        ("FIP", "FIP", 7, True, ".2f"),
        ("Delta", "delta", 8, True, ".2f"),
        ("K%", "K_pct", 6, True, ".3f"),
        ("BB%", "BB_pct", 6, True, ".3f"),
    ]
    subtitle_line = (
        "Highlights pitchers whose ERA deviates from FIP to flag luck-driven results and possible regression paths."
    )
    const_line = f"FIP constant c = {c:.3f}"
    if c_note:
        const_line = f"{const_line} {c_note}"
    subtitle_text = f"{subtitle_line} {const_line}"

    text_output = text_table(
        display_df,
        text_columns,
        "ABL FIP vs ERA Gap",
        subtitle_text.strip(),
        [
            "perf_flag: UNDERPERFORM >= +0.50 (ERA >> FIP), OVERPERFORM <= -0.50 (ERA << FIP).",
            "Rating tiers: Tough Luck (>= +1.00), Headwind (+0.50 to +0.99), Neutral (-0.49 to +0.49), Tailwind (-0.99 to -0.50), Magic Touch (<= -1.00).",
            f"QUAL requires IP >= {args.min_ip_sp} for SP/Swing, >= {args.min_ip_rp} for RP.",
        ],
        [
            "Delta = ERA - FIP (positive suggests rougher results than peripherals).",
            "FIP uses league-wide constant derived from HR, BB, SO, IP population.",
            "K% and BB% use BF when available.",
        ],
    )
    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() in {'csv_out'}:
        text_dir = out_path.parent.parent / "txt_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / text_filename
    text_path.write_text(text_output, encoding="utf-8")
    c_line = f"FIP constant c = {c:.3f}"
    if c_note:
        c_line = f"{c_line} {c_note}"
    print(c_line)
    print(text_output)


if __name__ == "__main__":
    main()

