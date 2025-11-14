"""ABL Team BABIP Luck: compare batting/pitching BABIP vs league averages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_RANGE = (1, 24)

TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]
BAT_CANDIDATES = [
    "team_batting.csv",
    "teams_batting.csv",
    "batting_team_totals.csv",
    "team_batting_stats.csv",
]
PITCH_CANDIDATES = [
    "team_pitching.csv",
    "teams_pitching.csv",
    "pitching_team_totals.csv",
    "team_pitching_stats.csv",
]

BAT_FLAG_THRESHOLD = 0.015


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
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return display_map, conf_map
    for _, row in df.iterrows():
        tid = row.get(team_col)
        if pd.isna(tid):
            continue
        tid = int(tid)
        if not (TEAM_RANGE[0] <= tid <= TEAM_RANGE[1]):
            continue
        if display_col and pd.notna(row.get(display_col)):
            display_map[tid] = str(row.get(display_col))
        if tid not in conf_map and sub_col and div_col:
            sub_val = row.get(sub_col)
            div_val = row.get(div_col)
            if pd.notna(sub_val) and pd.notna(div_val):
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
                conf_map[tid] = f"{sub}-{div}"
    return display_map, conf_map


def load_batting(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, BAT_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate team batting totals.")
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    if not team_col:
        raise ValueError("team_id column required in batting totals.")
    df = df.copy()
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_RANGE[0]) & (df["team_id"] <= TEAM_RANGE[1])]
    def num(col_names: Sequence[str]) -> pd.Series:
        col = pick_column(df, *col_names)
        return pd.to_numeric(df[col], errors="coerce") if col else np.nan
    data = pd.DataFrame()
    data["team_id"] = df["team_id"]
    disp_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    if disp_col:
        data["team_display"] = df[disp_col].fillna("")
    else:
        data["team_display"] = pd.Series([""] * len(df))
    data["H_bat"] = num(["h", "H"])
    data["HR_bat"] = num(["hr", "HR"])
    data["AB_bat"] = num(["ab", "AB"])
    data["SO_bat"] = num(["so", "SO", "k", "K"])
    data["SF_bat"] = num(["sf", "SF"]).fillna(0)
    return data


def load_pitching(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, PITCH_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate team pitching totals.")
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    if not team_col:
        raise ValueError("team_id column required in pitching totals.")
    df = df.copy()
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_RANGE[0]) & (df["team_id"] <= TEAM_RANGE[1])]
    def num(col_names: Sequence[str]) -> pd.Series:
        col = pick_column(df, *col_names)
        return pd.to_numeric(df[col], errors="coerce") if col else np.nan
    data = pd.DataFrame()
    data["team_id"] = df["team_id"]
    data["H_allowed"] = num(["h_allowed", "ha", "h", "H"])
    data["HR_allowed"] = num(["hr_allowed", "hra", "hr", "HR"])
    data["AB_against"] = num(["ab_against", "aba", "ab", "AB"])
    data["SO_pitch"] = num(["so", "SO", "k", "K"])
    data["SF_against"] = num(["sf_against", "sfa", "sf", "SF"]).fillna(0)
    return data


def compute_babip(h: float, hr: float, ab: float, so: float, sf: float) -> float:
    if any(pd.isna(val) for val in [h, hr, ab, so]):
        return np.nan
    numerator = h - hr
    denominator = ab - so - hr + sf
    if denominator <= 0:
        return np.nan
    return numerator / denominator


def assign_rating(b_flag: str, p_flag: str) -> str:
    if b_flag == "LUCK" and p_flag == "UNLUCKY":
        return "Hot & leaking"
    if b_flag == "LUCK" and p_flag == "STINGY":
        return "Riding high"
    if b_flag == "DRAG" and p_flag == "UNLUCKY":
        return "Snakebit"
    if b_flag == "DRAG" and p_flag == "STINGY":
        return "Pitch saves"
    if b_flag:
        return f"Bat {b_flag}"
    if p_flag:
        return f"Pitch {p_flag}"
    return "Neutral"


def build_table(
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    team_names: Dict[int, str],
    conf_map: Dict[int, str],
) -> pd.DataFrame:
    merged = batting.merge(pitching, on="team_id", how="outer")
    merged["team_display"] = merged["team_id"].map(team_names).fillna(merged.get("team_display", ""))
    merged["conf_div"] = merged["team_id"].map(conf_map).fillna("")
    merged["SF_bat"] = merged["SF_bat"].fillna(0)
    merged["SF_against"] = merged["SF_against"].fillna(0)
    merged["babip_bat"] = merged.apply(
        lambda row: compute_babip(row["H_bat"], row["HR_bat"], row["AB_bat"], row["SO_bat"], row["SF_bat"]),
        axis=1,
    )
    merged["babip_pitch"] = merged.apply(
        lambda row: compute_babip(
            row["H_allowed"],
            row["HR_allowed"],
            row["AB_against"],
            row["SO_pitch"],
            row["SF_against"],
        ),
        axis=1,
    )
    lg_bat = merged["babip_bat"].mean(skipna=True)
    lg_pitch = merged["babip_pitch"].mean(skipna=True)
    merged["lg_babip_bat"] = lg_bat
    merged["lg_babip_pitch"] = lg_pitch
    merged["babip_bat_diff"] = merged["babip_bat"] - lg_bat
    merged["babip_pitch_diff"] = merged["babip_pitch"] - lg_pitch
    merged["bat_flag"] = merged["babip_bat_diff"].apply(
        lambda diff: "LUCK" if diff >= BAT_FLAG_THRESHOLD else ("DRAG" if diff <= -BAT_FLAG_THRESHOLD else "")
    )
    merged["pitch_flag"] = merged["babip_pitch_diff"].apply(
        lambda diff: "UNLUCKY" if diff >= BAT_FLAG_THRESHOLD else ("STINGY" if diff <= -BAT_FLAG_THRESHOLD else "")
    )
    merged["rating"] = merged.apply(lambda row: assign_rating(row["bat_flag"], row["pitch_flag"]), axis=1)
    merged["babip_bat"] = merged["babip_bat"].round(3)
    merged["lg_babip_bat"] = np.round(merged["lg_babip_bat"], 3)
    merged["babip_bat_diff"] = merged["babip_bat_diff"].round(3)
    merged["babip_pitch"] = merged["babip_pitch"].round(3)
    merged["lg_babip_pitch"] = np.round(merged["lg_babip_pitch"], 3)
    merged["babip_pitch_diff"] = merged["babip_pitch_diff"].round(3)
    merged = merged[
        [
            "team_id",
            "team_display",
            "conf_div",
            "H_bat",
            "HR_bat",
            "AB_bat",
            "SO_bat",
            "SF_bat",
            "babip_bat",
            "lg_babip_bat",
            "babip_bat_diff",
            "bat_flag",
            "H_allowed",
            "HR_allowed",
            "AB_against",
            "SO_pitch",
            "SF_against",
            "babip_pitch",
            "lg_babip_pitch",
            "babip_pitch_diff",
            "pitch_flag",
            "rating",
        ]
    ]
    merged = merged.sort_values(
        by=["babip_bat_diff", "babip_pitch_diff"],
        ascending=[False, True],
        na_position="last",
    )
    return merged.reset_index(drop=True)


def build_text_report(df: pd.DataFrame, limit: Optional[int] = None) -> str:
    lines = ["ABL Team BABIP Luck", "=" * 24, ""]
    subset = df if limit is None else df.head(limit)
    for _, row in subset.iterrows():
        team_str = f"{row['team_display']} ({row['conf_div']})"
        bat_flag = row["bat_flag"] or "EVEN"
        pitch_flag = row["pitch_flag"] or "EVEN"
        lines.append(
            f"{team_str:<24} | Bat {row['babip_bat']:.3f} ({bat_flag:<7}) | "
            f"Pitch {row['babip_pitch']:.3f} ({pitch_flag:<7}) | {row['rating']}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  Bat LUCK/DRAG -> batting BABIP +/-0.015 vs league.")
    lines.append("  Pitch UNLUCKY/STINGY -> pitching BABIP allowed +/-0.015 vs league.")
    lines.append("  Rating summarizes combined batting & pitching luck.")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  BABIP batting = (H-HR)/(AB-SO-HR+SF).")
    lines.append("  BABIP allowed = (Hits allowed - HR allowed)/(AB against - SO - HR allowed + SF against).")
    lines.append("  Diff columns measure deviation from league BABIP.")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute team BABIP luck metrics.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--batting", type=str, help="Override team batting file.")
    parser.add_argument("--pitching", type=str, help="Override team pitching file.")
    parser.add_argument("--teams", type=str, help="Override team info file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/z_ABL_Team_BABIP_Luck.csv",
        help="Output CSV path.",
    )
    return parser.parse_args(argv if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base).resolve()
    batting = load_batting(base_dir, Path(args.batting) if args.batting else None)
    pitching = load_pitching(base_dir, Path(args.pitching) if args.pitching else None)
    team_names, conf_map = load_team_info(base_dir, Path(args.teams) if args.teams else None)
    final_df = build_table(batting, pitching, team_names, conf_map)

    out_path = (base_dir / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    text_path = out_path.with_suffix(".txt")
    text_path.write_text(build_text_report(final_df), encoding="utf-8")

    print("Team BABIP luck (top 12):")
    print(final_df.head(12).to_string(index=False))
    print(f"\nWrote {len(final_df)} rows to {out_path}.")


if __name__ == "__main__":
    main()
