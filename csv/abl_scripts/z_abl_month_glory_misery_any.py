#!/usr/bin/env python
"""
Build calendar-month performance splits and Month of Glory/Misery markdown fragment.
"""
from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

TOP_N = 10


def log(msg: str) -> None:
    print(msg, flush=True)


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_games(path: Path) -> pd.DataFrame:
    if not path.exists():
        log(f"[ERROR] games CSV not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, encoding="utf-8", na_filter=False)
    log(f"[INFO] Loaded games CSV: {len(df)} rows from {path}")
    log(f"[DEBUG] games columns: {list(df.columns)}")
    return df


def parse_win_flag(series: pd.Series) -> pd.Series:
    """Generic helper for cases where a result column already exists."""
    vals = series.astype(str)
    if set(vals.unique()) <= {"W", "L", "w", "l"}:
        return vals.str.upper().eq("W").astype(int)
    try:
        nums = pd.to_numeric(series, errors="coerce")
        if nums.notna().any():
            return nums.fillna(0).astype(int)
    except Exception:
        pass
    return vals.str.lower().isin({"w", "win", "true", "1"}).astype(int)


def filter_regular_season(df: pd.DataFrame) -> pd.DataFrame:
    for cand in ["subseason", "game_type", "season_type"]:
        if cand in df.columns:
            mask = df[cand].astype(str).str.lower().isin(
                {"regular season", "regular", "reg", "season"}
            )
            kept = df[mask].copy()
            log(f"[INFO] Filtered to regular season via {cand}: {len(df)} -> {len(kept)}")
            return kept
    return df


def parse_schedule_months(path: Path) -> list[int]:
    """Right now we only use this to confirm which months exist in the grid."""
    if not path.exists():
        log(f"[WARN] schedule grid HTML not found: {path}; proceeding without it.")
        return []
    try:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        table = soup.find("table")
        if not table:
            log("[WARN] schedule grid has no table; continuing without it.")
            return []
        months_seen: set[int] = set()
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            for cell in cells:
                for m_idx, m_name in enumerate(calendar.month_name):
                    if m_name and m_name.lower() in cell.lower():
                        months_seen.add(m_idx)
        log(f"[DEBUG] Months detected in schedule grid: {sorted(months_seen)}")
        return sorted(months_seen)
    except Exception as exc:  # noqa: BLE001
        log(f"[WARN] unable to parse schedule grid; continuing without it. Error: {exc}")
        return []


def compute_win_flag_from_runs(games: pd.DataFrame) -> pd.DataFrame:
    """
    For ABL games_by_team CSV:
    - One row per team per game, with 'game_id' and 'runs'.
    - Winner is the row with the highest runs within each game_id.
    """
    required = ["game_id", "runs"]
    missing = [c for c in required if c not in games.columns]
    if missing:
        log(f"[ERROR] Cannot compute wins; missing columns: {missing}")
        sys.exit(1)

    games = games.copy()
    games["runs"] = pd.to_numeric(games["runs"], errors="coerce")
    if games["runs"].isna().any():
        log("[WARN] Some runs could not be parsed as numeric; treating them as 0.")
        games["runs"] = games["runs"].fillna(0)

    games["win_flag"] = 0
    idx = games.groupby("game_id")["runs"].idxmax()
    games.loc[idx, "win_flag"] = 1

    too_small = games.groupby("game_id")["game_id"].transform("size") < 2
    if too_small.any():
        bad_games = games.loc[too_small, "game_id"].unique()
        log(f"[WARN] Some game_ids have fewer than 2 team rows: {bad_games[:10]} ...")

    return games


def build_monthly(
    games: pd.DataFrame, season: int, team_col: str, date_col: str, win_col: str, min_games: int
) -> pd.DataFrame:
    games = games.copy()
    games["parsed_date"] = pd.to_datetime(games[date_col], errors="coerce")
    games = games.dropna(subset=["parsed_date"])
    games["year"] = games["parsed_date"].dt.year
    games["month"] = games["parsed_date"].dt.month
    games = games[games["year"] == season]

    games["win_flag"] = parse_win_flag(games[win_col])

    season_group = games.groupby(team_col, as_index=False).agg(G=("win_flag", "size"), W=("win_flag", "sum"))
    season_group["L"] = season_group["G"] - season_group["W"]
    season_group["season_pct"] = season_group["W"] / season_group["G"]

    monthly = (
        games.groupby([team_col, "month"], as_index=False)
        .agg(G=("win_flag", "size"), W=("win_flag", "sum"))
    )
    monthly["L"] = monthly["G"] - monthly["W"]
    monthly["month_pct"] = monthly["W"] / monthly["G"]
    monthly = monthly[monthly["G"] >= min_games]

    monthly = monthly.merge(season_group[[team_col, "season_pct"]], on=team_col, how="left")
    monthly["delta"] = monthly["month_pct"] - monthly["season_pct"]
    return monthly


def attach_team_meta(monthly: pd.DataFrame, team_col: str, dim_path: Path) -> pd.DataFrame:
    """
    Attach team_id / abbr / name / conference / division using dim_team_park.
    For this script, team_col will be a team NAME (e.g., 'Detroit Dukes'),
    so we prioritize joining by name.
    """
    if not dim_path.exists():
        log(f"[ERROR] dim_team_park not found: {dim_path}")
        sys.exit(1)
    dim = pd.read_csv(dim_path, encoding="utf-8")
    id_col = find_column(dim, ["ID", "team_id"])
    abbr_col = find_column(dim, ["Abbr", "team_abbr"])
    name_col = find_column(dim, ["Team Name", "team_name", "Name"])
    conf_col = find_column(dim, ["SL", "conf", "conference"])
    div_col = find_column(dim, ["DIV", "division"])
    if not abbr_col or not name_col:
        log(f"[ERROR] dim_team_park missing abbr or team name columns. cols={list(dim.columns)}")
        sys.exit(1)

    dim_small = dim[[c for c in [id_col, abbr_col, name_col, conf_col, div_col] if c]].copy()
    dim_small = dim_small.rename(
        columns={
            id_col or "": "team_id",
            abbr_col: "team_abbr",
            name_col: "team_name",
            conf_col or "": "conference",
            div_col or "": "division",
        }
    )

    dim_small["join_name"] = dim_small["team_name"].astype(str).str.upper()
    monthly = monthly.copy()
    monthly["join_name"] = monthly[team_col].astype(str).str.upper()
    monthly["_raw_team_name"] = monthly[team_col].astype(str)

    merged = monthly.merge(dim_small, on="join_name", how="left")

    dim_name_col = "team_name_y" if "team_name_y" in merged.columns else "team_name"
    dim_abbr_col = "team_abbr_y" if "team_abbr_y" in merged.columns else "team_abbr"

    merged["team_name"] = merged.get(dim_name_col, pd.Series(dtype=object)).combine_first(merged["_raw_team_name"])
    merged["team_abbr"] = merged.get(dim_abbr_col, pd.Series(dtype=object)).combine_first(merged["_raw_team_name"])

    missing = merged["team_abbr"].isna().sum()
    if missing:
        log(f"[WARN] {missing} monthly rows missing team metadata after name join.")

    drop_cols = [c for c in ["team_name_x", "team_name_y", "team_abbr_x", "team_abbr_y", "_raw_team_name", "join_name"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)
    return merged


def top_lists(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    glory = monthly.sort_values(["delta", "G"], ascending=[False, False]).head(TOP_N)
    misery = monthly.sort_values(["delta", "G"], ascending=[True, False]).head(TOP_N)
    return glory, misery


def render_md(glory: pd.DataFrame, misery: pd.DataFrame, min_games: int, season: int) -> str:
    lines: list[str] = []
    lines.append("<!-- EB Month-of-Glory/Misery fragment, generated by z_abl_month_glory_misery_any.py -->")
    lines.append(f"## Month of Glory - Overachievers (calendar months, G >= {min_games})")
    for _, r in glory.iterrows():
        month_name = calendar.month_name[int(r['month'])]
        lines.append(
            f"- {r['team_name']} ({r['team_abbr']}) - in {month_name} went {int(r['W'])}-{int(r['L'])} "
            f"({r['month_pct']:.3f}), delta vs season={r['delta']:+.3f}"
        )
    lines.append("")
    lines.append(f"## Month of Misery - Slumps (calendar months, G >= {min_games})")
    for _, r in misery.iterrows():
        month_name = calendar.month_name[int(r['month'])]
        lines.append(
            f"- {r['team_name']} ({r['team_abbr']}) - in {month_name} went {int(r['W'])}-{int(r['L'])} "
            f"({r['month_pct']:.3f}), delta vs season={r['delta']:+.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Month of Glory/Misery from games CSV.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    parser.add_argument(
        "--games-csv",
        type=Path,
        default=None,
        help="Path to games_{season}_league{league_id}_by_team.csv",
    )
    parser.add_argument(
        "--schedule-grid-html",
        type=Path,
        default=None,
        help="Path to schedule grid HTML",
    )
    parser.add_argument(
        "--dim-team-park",
        type=Path,
        default=Path("csv/out/star_schema/dim_team_park.csv"),
        help="Path to dim_team_park.csv",
    )
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output markdown fragment path",
    )
    args = parser.parse_args()

    season = args.season
    league_id = args.league_id
    games_csv = args.games_csv or Path(
        f"csv/out/almanac/{season}/games_{season}_league{league_id}_by_team.csv"
    )
    sched_html = args.schedule_grid_html or Path(
        f"csv/in/almanac_core/{season}/leagues/league_{league_id}_schedule_grid.html"
    )
    out_md = args.out_md or Path(
        f"csv/out/eb/eb_month_glory_misery_{season}_league{league_id}.md"
    )

    games = load_games(games_csv)

    date_col = find_column(games, ["game_date", "date"])
    if not date_col:
        log(f"[ERROR] Could not find a date column in games CSV. cols={list(games.columns)}")
        return 1

    if "team_name" not in games.columns:
        log(f"[ERROR] Expected 'team_name' in games CSV, found: {list(games.columns)}")
        return 1
    team_col = "team_name"

    games = filter_regular_season(games)
    games = compute_win_flag_from_runs(games)
    win_col = "win_flag"

    schedule_months = parse_schedule_months(sched_html)
    if schedule_months:
        log(f"[INFO] Schedule grid months: {schedule_months}")

    monthly = build_monthly(games, season, team_col, date_col, win_col, args.min_games)
    monthly = attach_team_meta(monthly, team_col, args.dim_team_park)
    glory, misery = top_lists(monthly)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(glory, misery, args.min_games, season), encoding="utf-8")
    log(f"[OK] Wrote Month of Glory/Misery fragment to {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
