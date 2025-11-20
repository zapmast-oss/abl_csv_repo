"""ABL Run Prevention DNA: summarize DER, ZR, and error/DP rates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24
FIELDING_CANDIDATES = [
    "team_fielding_stats_stats.csv",
    "team_fielding.csv",
    "fielding_totals.csv",
    "fielding.csv",
    "players_fielding.csv",
]
PITCHING_CANDIDATES = [
    "team_pitching.csv",
    "team_pitching_stats.csv",
    "teams_pitching.csv",
    "teams_pitching_stats.csv",
    "team_defense.csv",
]
TEAM_META_CANDIDATES = [
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


def read_first_available(base: Path, override: Optional[Path], candidates: Sequence[str]) -> Optional[pd.DataFrame]:
    if override:
        override_path = Path(override)
        if not override_path.exists():
            raise FileNotFoundError(f"Specified file not found: {override_path}")
        return pd.read_csv(override_path)
    for name in candidates:
        path = base / name
        if path.exists():
            return pd.read_csv(path)
    return None


def load_fielding(base: Path, override: Optional[Path]) -> Tuple[pd.DataFrame, bool]:
    df = read_first_available(base, override, FIELDING_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to find a fielding source.")
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        raise ValueError("Fielding data missing team_id column.")
    pos_col = pick_column(df, "pos", "position")
    inn_col = pick_column(df, "inn", "innings", "ip", "outs")
    zr_col = pick_column(df, "zr", "zone_rating")
    zr_proxy_col = None
    zr_is_proxy = False
    if not zr_col:
        zr_proxy_col = pick_column(df, "range", "range_factor", "rf")
        if zr_proxy_col:
            zr_col = zr_proxy_col
            zr_is_proxy = True
    err_col = pick_column(df, "e", "errors")
    dp_col = pick_column(df, "dp", "double_plays")

    df = df.copy()
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[df["team_id"].between(TEAM_MIN, TEAM_MAX)]

    if pos_col:
        df = df[~df[pos_col].astype(str).str.upper().isin({"P", "DH"})]

    def numeric_series(col: Optional[str], default_zero: bool = True) -> pd.Series:
        if not col or col not in df.columns:
            fill_value = 0.0 if default_zero else np.nan
            return pd.Series(fill_value, index=df.index, dtype="float64")
        return pd.to_numeric(df[col], errors="coerce").astype("float64")

    df["inn_def"] = numeric_series(inn_col, default_zero=True).fillna(0.0)
    if inn_col and inn_col.lower() in {"ip", "ip_outs"}:
        df["inn_def"] = df["inn_def"] / 3.0
    df["zr_sum"] = numeric_series(zr_col, default_zero=False)
    df["errors"] = numeric_series(err_col, default_zero=True).fillna(0.0)
    df["dp"] = numeric_series(dp_col, default_zero=True).fillna(0.0)

    grouped = (
        df.groupby("team_id")[["inn_def", "zr_sum", "errors", "dp"]]
        .sum(min_count=1)
        .reset_index()
    )
    return grouped, zr_is_proxy


def load_pitching(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first_available(base, override, PITCHING_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to find a pitching/defense source.")
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        raise ValueError("Pitching data missing team_id column.")

    out = pd.DataFrame()
    out["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    out = out[out["team_id"].between(TEAM_MIN, TEAM_MAX)]

    der_col = pick_column(df, "der", "def_eff")
    if der_col:
        out["der"] = pd.to_numeric(df.loc[out.index, der_col], errors="coerce")
    else:
        out["der"] = pd.NA

    for source_col, target in [
        (pick_column(df, "h", "hits", "ha", "H"), "h"),
        (pick_column(df, "hr", "home_runs", "hra", "HR"), "hr"),
        (pick_column(df, "ab", "AB"), "ab"),
        (pick_column(df, "so", "k", "strikeouts", "SO"), "so"),
    ]:
        if source_col:
            out[target] = pd.to_numeric(df.loc[out.index, source_col], errors="coerce")
        else:
            out[target] = pd.NA

    # Compute proxy DER where needed.
    needs_proxy = out["der"].isna()
    valid_proxy = needs_proxy & out[["ab", "h", "hr", "so"]].notna().all(axis=1)
    denom = out.loc[valid_proxy, "ab"] - out.loc[valid_proxy, "so"] - out.loc[valid_proxy, "hr"]
    with np.errstate(divide="ignore", invalid="ignore"):
        proxy = 1 - (out.loc[valid_proxy, "h"] - out.loc[valid_proxy, "hr"]) / denom
    proxy = proxy.where(denom > 0)
    out.loc[valid_proxy, "der"] = proxy
    return out[["team_id", "der"]]


def load_team_meta(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first_available(base, override, TEAM_META_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["team_id", "team_display", "division_id", "sub_league_id"])
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    div_col = pick_column(df, "division_id", "divisionid", "div_id")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "sub_id", "subleague")
    if not team_col:
        return pd.DataFrame(columns=["team_id", "team_display", "division_id", "sub_league_id"])
    meta = pd.DataFrame()
    meta["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    if name_col:
        meta["team_display"] = df[name_col].fillna("")
    else:
        meta["team_display"] = ""
    if div_col:
        meta["division_id"] = pd.to_numeric(df[div_col], errors="coerce").astype("Int64")
    else:
        meta["division_id"] = pd.NA
    if sub_col:
        meta["sub_league_id"] = pd.to_numeric(df[sub_col], errors="coerce").astype("Int64")
    else:
        meta["sub_league_id"] = pd.NA
    return meta[(meta["team_id"] >= TEAM_MIN) & (meta["team_id"] <= TEAM_MAX)]


def build_conf_div_map(meta: pd.DataFrame) -> Dict[int, str]:
    conf_map = {0: "N", 1: "A"}
    div_map = {0: "E", 1: "C", 2: "W"}
    mapping: Dict[int, str] = {}
    for _, row in meta.iterrows():
        tid = row.get("team_id")
        if pd.isna(tid):
            continue
        sub = row.get("sub_league_id")
        div = row.get("division_id")
        conf = conf_map.get(int(sub)) if pd.notna(sub) else ""
        division = div_map.get(int(div)) if pd.notna(div) else ""
        label = "-".join(filter(None, [conf, division]))
        if label:
            mapping[int(tid)] = label
    return mapping


def classify_profile(der: float, err9: float, dp9: float) -> str:
    if pd.isna(der) or pd.isna(err9):
        return ""
    if der >= 0.715 and err9 <= 0.18:
        return "Elite"
    if der >= 0.700 and err9 <= 0.20:
        return "Clean"
    if der >= 0.700 and (err9 > 0.20 or (pd.notna(dp9) and dp9 >= 0.40)):
        return "Playmaking"
    if 0.680 <= der < 0.700:
        return "Middle"
    if der < 0.680:
        return "Leaky"
    return ""


def build_output_text(df: pd.DataFrame, limit: int = 24, zr_proxy: bool = False) -> str:
    lines = [
        "ABL Run Prevention DNA",
        "=" * 25,
        "Defense efficiency, range, and mistake rates blended to show who prevents runs.",
        "Use it to spot elite or leaky gloves before setting lineups or pitching plans.",
        "",
        "Team               Div Profile        DER     ZR  Err/9  DP/9",
        "-----------------------------------------------------------------------",
    ]

    def fmt_num(value: float, width: int, decimals: int) -> str:
        if pd.notna(value):
            return f"{value:.{decimals}f}".rjust(width)
        return "NA".rjust(width)

    for _, row in df.head(limit).iterrows():
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        conf = (row.get("conf_div") or "--").strip()[:3]
        profile = (row.get("profile") or "").strip()[:12]
        der = fmt_num(row["der"], 6, 3)
        zr = fmt_num(row["zr_sum"], 7, 2)
        err9 = fmt_num(row["errors_per_9"], 7, 3)
        dp9 = fmt_num(row["dp_per_9"], 7, 3)
        lines.append(
            f"{name:<18} {conf:>3} {profile:<12} {der} {zr} {err9} {dp9}"
        )
    lines.append("")
    lines.append(
        "Key:",
    )
    lines.append("  DER     -> Share of balls in play turned into outs (higher = cleaner fielding).")
    lines.append("  ZR      -> Summed zone rating contributions; positive values mean above-average range.")
    lines.append("  Errors/9-> Errors committed per nine defensive innings (lower = steadier hands).")
    lines.append("  DP/9    -> Double plays turned per nine defensive innings (higher = slick infield).")
    lines.append("  Profiles: Elite (DER >= .715 & Err/9 <= .18), Clean (0.700-0.714 with low errors), Playmaking (high DER but messy), Middle (0.680-0.699), Leaky (<0.680).")
    lines.append("")
    lines.append(
        "Definition: Run Prevention DNA blends DER, range (ZR), and mistake/situational rates to show how each defense prevents runs."
    )
    if zr_proxy:
        lines.append("Note: ZR column uses team range factor because no zone-rating stats were exported.")
    return "\n".join(lines)

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Run Prevention DNA.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--fielding", type=str, help="Override fielding CSV.")
    parser.add_argument("--pitching", type=str, help="Override team pitching/defense CSV.")
    parser.add_argument("--teams", type=str, help="Override team info CSV.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Run_Prevention_DNA.csv",
        help="Output CSV path (relative to --base).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()

    fielding, zr_proxy = load_fielding(base_dir, Path(args.fielding) if args.fielding else None)
    pitching = load_pitching(base_dir, Path(args.pitching) if args.pitching else None)
    meta = load_team_meta(base_dir, Path(args.teams) if args.teams else None)
    conf_div_map = build_conf_div_map(meta)

    df = fielding.merge(pitching, on="team_id", how="left")
    df = df.merge(meta[["team_id", "team_display"]], on="team_id", how="left")
    df["team_display"] = df["team_display"].fillna("")

    df["errors_per_9"] = np.where(
        df["inn_def"] > 0, 9 * df["errors"] / df["inn_def"], np.nan
    )
    df["dp_per_9"] = np.where(
        df["inn_def"] > 0, 9 * df["dp"] / df["inn_def"], np.nan
    )

    for col in ["zr_sum", "errors_per_9", "dp_per_9", "der"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["profile"] = df.apply(
        lambda row: classify_profile(row["der"], row["errors_per_9"], row["dp_per_9"]),
        axis=1,
    )

    df["zr_sum"] = df["zr_sum"].round(2)
    df["errors_per_9"] = df["errors_per_9"].round(3)
    df["dp_per_9"] = df["dp_per_9"].round(3)
    df["der"] = df["der"].round(3)

    column_order = [
        "team_id",
        "team_display",
        "inn_def",
        "zr_sum",
        "errors",
        "errors_per_9",
        "dp",
        "dp_per_9",
        "der",
        "profile",
    ]
    df = df[column_order]

    df = df.sort_values(
        by=["der", "zr_sum", "errors_per_9"],
        ascending=[False, False, True],
        na_position="last",
    )

    out_path = (base_dir / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    txt_dir = base_dir / "out" / "text_out"
    txt_dir.mkdir(parents=True, exist_ok=True)
    text_path = (txt_dir / out_path.stem).with_suffix(".txt")
    text_df = df.copy()
    text_df["conf_div"] = text_df["team_id"].map(conf_div_map).fillna("")
    text_path.write_text(stamp_text_block(build_output_text(text_df, zr_proxy=zr_proxy)), encoding="utf-8")

    preview = df.head(12)
    print("Run Prevention DNA (top 12):")
    print(preview.to_string(index=False))
    print(f"\nWrote {len(df)} rows to {out_path} and summary to {text_path}.")


if __name__ == "__main__":
    main()


