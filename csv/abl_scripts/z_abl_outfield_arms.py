"""ABL Outfield Arms report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24
OF_POSITIONS = {"LF", "CF", "RF", "7", "8", "9"}

FIELDING_CANDIDATES = [
    "players_career_fielding_stats.csv",
    "players_fielding.csv",
    "fielding.csv",
]
OPP_CANDIDATES = [
    "outfield_throws.csv",
    "baserunning_advances.csv",
    "runners_on_hits.csv",
    "throw_attempts.csv",
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


def normalize_pos(value: str) -> Optional[str]:
    val = str(value).strip().upper()
    if val in {"LF", "CF", "RF"}:
        return val
    if val in {"7", "8", "9"}:
        return {"7": "LF", "8": "CF", "9": "RF"}[val]
    return None


def load_fielding(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, FIELDING_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate fielding totals.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    pos_col = pick_column(df, "position", "pos")
    inn_col = pick_column(df, "inn", "ip", "innings")
    assists_col = pick_column(df, "of_a", "a", "assists")
    err_col = pick_column(df, "e", "errors")
    dp_col = pick_column(df, "dp", "double_plays")
    year_col = pick_column(df, "year", "season")
    if not id_col or not team_col or not pos_col or not inn_col or not assists_col:
        raise ValueError("Fielding file missing key columns.")
    data = df.copy()
    if year_col:
        max_year = pd.to_numeric(data[year_col], errors="coerce").max()
        data = data[pd.to_numeric(data[year_col], errors="coerce") == max_year]
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    data["pos"] = data[pos_col].apply(normalize_pos)
    data = data[data["pos"].notna()]
    out = pd.DataFrame()
    out["player_id"] = data["player_id"]
    out["team_id"] = data["team_id"]
    out["pos"] = data["pos"]
    out["OF_INN"] = pd.to_numeric(data[inn_col], errors="coerce")
    out["OF_A"] = pd.to_numeric(data[assists_col], errors="coerce")
    out["OF_E"] = pd.to_numeric(data[err_col], errors="coerce") if err_col else np.nan
    out["OF_DP"] = pd.to_numeric(data[dp_col], errors="coerce") if dp_col else np.nan
    out = out.groupby(["player_id", "team_id", "pos"], as_index=False).sum(min_count=1)
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


def load_opportunities(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, OPP_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["player_id", "team_id", "pos", "adv_attempts", "advances", "holds"])
    player_col = pick_column(df, "fielder_id", "player_id", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    pos_col = pick_column(df, "position", "pos")
    attempts_col = pick_column(df, "adv_attempts", "attempts")
    advances_col = pick_column(df, "advances", "successful_adv", "successes")
    holds_col = pick_column(df, "holds", "runners_held")
    if not player_col or not team_col or not attempts_col:
        return pd.DataFrame(columns=["player_id", "team_id", "pos", "adv_attempts", "advances", "holds"])
    data = df.copy()
    data["player_id"] = pd.to_numeric(data[player_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    if pos_col:
        data["pos"] = data[pos_col].apply(normalize_pos)
    else:
        data["pos"] = np.nan
    data["adv_attempts"] = pd.to_numeric(data[attempts_col], errors="coerce")
    data["advances"] = pd.to_numeric(data[advances_col], errors="coerce") if advances_col else np.nan
    data["holds"] = pd.to_numeric(data[holds_col], errors="coerce") if holds_col else np.nan
    if holds_col is None and advances_col:
        derived = data["adv_attempts"] - data["advances"]
        data["holds"] = np.where(derived >= 0, derived, np.nan)
    out = data.groupby(["player_id", "team_id", "pos"], as_index=False).sum(min_count=1)
    return out


def safe_div(numer: float, denom: float) -> float:
    if pd.isna(numer) or pd.isna(denom) or denom == 0:
        return np.nan
    return numer / denom


def classify_rating(arma_plus: float, nogo_plus: float) -> str:
    if pd.notna(arma_plus) and arma_plus >= 1.3:
        return "Sniper Arm"
    if pd.notna(nogo_plus) and nogo_plus >= 1.25:
        return "Respected Cannon"
    if pd.notna(arma_plus) and arma_plus >= 1.1:
        return "Strong Arm"
    if pd.notna(arma_plus) and arma_plus >= 0.9:
        return "Steady Arm"
    return "Tested Often"


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
        lines.append("(No outfielders met the qualification thresholds.)")
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
    parser = argparse.ArgumentParser(description="ABL Outfield Arms report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory.")
    parser.add_argument("--fielding", type=str, help="Override fielding path.")
    parser.add_argument("--opps", type=str, help="Override opportunity file.")
    parser.add_argument("--teams", type=str, help="Override team info.")
    parser.add_argument("--roster", type=str, help="Override roster file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Outfield_Arms.csv",
        help="Output CSV path (default inside out/csv_out).",
    )
    parser.add_argument("--min_inn", type=float, default=200.0, help="Minimum OF innings to rank.")
    parser.add_argument("--min_attempts", type=int, default=15, help="Minimum attempts to rank no-go rate.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    fielding = load_fielding(base_dir, resolve_path(base_dir, args.fielding))
    roster = load_roster(base_dir, resolve_path(base_dir, args.roster))
    team_display, conf_map = load_team_info(base_dir, resolve_path(base_dir, args.teams))
    opps = load_opportunities(base_dir, resolve_path(base_dir, args.opps))

    df = fielding.copy()
    if not roster.empty:
        df = df.merge(roster, on="player_id", how="left")
    else:
        df["player_name"] = df["player_id"].astype(str)
    df["player_name"] = df["player_name"].fillna(df["player_id"].astype(str))
    df["OF_E"] = df["OF_E"].fillna(0)
    df["OF_DP"] = df["OF_DP"].fillna(0)
    df["team_display"] = df["team_id"].map(team_display)
    df["team_display"] = df.apply(
        lambda r: r["team_display"]
        if pd.notna(r["team_display"])
        else (f"T{int(r['team_id'])}" if pd.notna(r["team_id"]) else ""),
        axis=1,
    )
    df["conf_div"] = df["team_id"].map(conf_map).fillna("")

    df["A_per_1000"] = df.apply(lambda r: safe_div(r["OF_A"] * 1000, r["OF_INN"]), axis=1)
    if not opps.empty:
        opps["pos"] = opps["pos"].fillna("ALL")
        merged_opps = opps.merge(df[["player_id", "team_id", "pos", "OF_INN"]], on=["player_id", "team_id"], how="left", suffixes=("", "_field"))
        same_pos = merged_opps[merged_opps["pos_field"] == merged_opps["pos"]]
        fallback_all = merged_opps[(merged_opps["pos"] == "ALL") & merged_opps["pos_field"].notna()]
        opps_final = pd.concat([same_pos, fallback_all], ignore_index=True)
        agg = opps_final.groupby(["player_id", "team_id", "pos_field"], as_index=False).agg(
            adv_attempts=("adv_attempts", "sum"),
            advances=("advances", "sum"),
            holds=("holds", "sum"),
        ).rename(columns={"pos_field": "pos"})
        df = df.merge(agg, on=["player_id", "team_id", "pos"], how="left")
    else:
        df["adv_attempts"] = np.nan
        df["advances"] = np.nan
        df["holds"] = np.nan

    df["no_go_rate"] = df.apply(lambda r: safe_div(r["holds"], r["adv_attempts"]), axis=1)

    team_agg = df.groupby("team_id", as_index=False).agg(
        team_OF_INN=("OF_INN", "sum"),
        team_OF_A=("OF_A", "sum"),
    )
    team_agg["team_A_per_1000"] = team_agg.apply(
        lambda r: safe_div(r["team_OF_A"] * 1000, r["team_OF_INN"]), axis=1
    )
    df = df.merge(team_agg, on="team_id", how="left")

    lg_A_per_1000 = df.loc[df["OF_INN"] >= args.min_inn, "A_per_1000"].mean(skipna=True)
    lg_no_go = df.loc[df["adv_attempts"] >= args.min_attempts, "no_go_rate"].mean(skipna=True)

    df["ArmA_plus"] = df["A_per_1000"] / lg_A_per_1000 if lg_A_per_1000 and not np.isnan(lg_A_per_1000) else np.nan
    df["NoGo_plus"] = df["no_go_rate"] / lg_no_go if lg_no_go and not np.isnan(lg_no_go) else np.nan
    df["badge_sniper"] = np.where((df["ArmA_plus"] >= 1.30) & (df["A_per_1000"] >= 6.0), "Y", "")
    df["badge_respect"] = np.where((df["NoGo_plus"] >= 1.25) & (df["adv_attempts"] >= args.min_attempts), "Y", "")
    df["rank_flag"] = np.where(df["OF_INN"] >= args.min_inn, "QUAL", "")
    df["rating"] = df.apply(lambda r: classify_rating(r["ArmA_plus"], r["NoGo_plus"]), axis=1)

    csv_columns = [
        "team_id",
        "team_display",
        "conf_div",
        "player_id",
        "player_name",
        "pos",
        "OF_INN",
        "OF_A",
        "OF_E",
        "OF_DP",
        "A_per_1000",
        "adv_attempts",
        "advances",
        "holds",
        "no_go_rate",
        "ArmA_plus",
        "NoGo_plus",
        "badge_sniper",
        "badge_respect",
        "team_OF_INN",
        "team_OF_A",
        "team_A_per_1000",
        "rank_flag",
        "rating",
    ]
    csv_df = df[csv_columns].copy()
    csv_df["OF_INN"] = csv_df["OF_INN"].round(1)
    csv_df["A_per_1000"] = csv_df["A_per_1000"].round(2)
    csv_df["no_go_rate"] = csv_df["no_go_rate"].round(3)
    csv_df["ArmA_plus"] = csv_df["ArmA_plus"].round(3)
    csv_df["NoGo_plus"] = csv_df["NoGo_plus"].round(3)
    csv_df["team_OF_INN"] = csv_df["team_OF_INN"].round(1)
    csv_df["team_A_per_1000"] = csv_df["team_A_per_1000"].round(2)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_path, index=False)

    display_df = df[df["rank_flag"] == "QUAL"].copy()
    display_df = display_df.sort_values(
        by=["A_per_1000", "no_go_rate", "OF_INN"],
        ascending=[False, False, False],
        na_position="last",
    ).head(25)
    text_columns = [
        ("Player", "player_name", 22, False, ""),
        ("Tm", "team_display", 5, False, ""),
        ("Conf", "conf_div", 6, False, ""),
        ("Pos", "pos", 3, False, ""),
        ("INN", "OF_INN", 6, True, ".1f"),
        ("A/1000", "A_per_1000", 7, True, ".2f"),
        ("CS%", "no_go_rate", 6, True, ".3f"),
        ("Team A/1000", "team_A_per_1000", 11, True, ".2f"),
        ("Rating", "rating", 16, False, ""),
    ]
    mean_str = f"{lg_A_per_1000:.2f}" if pd.notna(lg_A_per_1000) else "NA"
    nogo_str = f"{lg_no_go:.3f}" if pd.notna(lg_no_go) else "NA"
    subtitle_line = (
        "Highlights the outfielders who gun down runners (assists/1000 INN) or freeze them (no-go rate)."
    )
    league_line = f"League A/1000 {mean_str}, League No-Go {nogo_str}"
    text_output = text_table(
        display_df,
        text_columns,
        "ABL Outfield Arms",
        f"{subtitle_line} {league_line}",
        [
            "Ratings: Sniper Arm (ArmA+ >=1.30), Respected Cannon (NoGo+ >=1.25), Strong Arm (ArmA+ >=1.10), Steady Arm (>=0.90), Tested Often (<0.90).",
            "Badges: Sniper (ArmA+>=1.30 & A/1000>=6.0), Respect (NoGo+>=1.25 & attempts>=threshold).",
        ],
        [
            "A/1000 = outfield assists per 1000 innings.",
            "No-go rate = holds / advance attempts when opportunity data exists.",
            "Team context shows total OF assists per 1000 innings for comparison.",
        ],
    )
    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() in {'csv_out'}:
        text_dir = out_path.parent.parent / "text_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / text_filename
    text_path.write_text(stamp_text_block(text_output), encoding="utf-8")
    print(text_output)


if __name__ == "__main__":
    main()

