#!/usr/bin/env python
"""
Build league-season summary for a given season/league_id.

Inputs (expected to exist before running):

- dim_team_park: csv/out/star_schema/dim_team_park.csv
- league standings HTML:
    csv/in/almanac_core/<season>/leagues/league_<league_id>_standings.html
- league stats HTML (team batting + team pitching):
    csv/in/almanac_core/<season>/leagues/league_<league_id>_stats.html

Output:

csv/out/almanac/<season>/league_season_summary_<season>_league<league_id>.csv

Columns:

season, league_id, team_id, team_abbr, team_name, conf, division,
wins, losses, pct, runs_for, runs_against, run_diff
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, file=sys.stdout)


def norm_name(value: str) -> str:
    """Normalize a team name for joining: lowercase, strip, remove non-alnum."""
    if pd.isna(value):
        return ""
    s = str(value).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def find_first_column(df: pd.DataFrame, candidates) -> str:
    """Return the first column name in df whose lowercase matches any candidate."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"None of {candidates} found in columns={list(df.columns)}")


def load_dim_team_park(dim_path: Path, league_id: int) -> pd.DataFrame:
    if not dim_path.exists():
        raise FileNotFoundError(f"dim_team_park not found: {dim_path}")

    dim = pd.read_csv(dim_path)
    has_league = "league_id" in dim.columns
    if has_league:
        dim = dim[dim["league_id"] == league_id].copy()
        if dim.empty:
            raise ValueError(f"dim_team_park has no rows for league_id={league_id}")
    else:
        log(f"[WARN] dim_team_park missing league_id column; assuming it is already filtered for league_id={league_id}.")
        dim["league_id"] = league_id

    team_id_col = find_first_column(dim, ["team_id", "ID"])
    name_col = find_first_column(dim, ["Team Name", "Name", "team_name", "City"])
    abbr_col = find_first_column(dim, ["Abbr", "team_abbr"])
    conf_col = find_first_column(dim, ["conf", "conference", "Conference", "SL"])
    div_col = find_first_column(dim, ["division", "Division", "div", "DIV", "DIV.1"])

    dim_use = dim[[team_id_col, "league_id", name_col, abbr_col, conf_col, div_col]].copy()
    dim_use = dim_use.rename(
        columns={
            team_id_col: "team_id",
            name_col: "team_name",
            abbr_col: "team_abbr",
            conf_col: "conf",
            div_col: "division",
        }
    )

    dim_use["name_key"] = dim_use["team_name"].map(norm_name)

    if dim_use["name_key"].duplicated().any():
        dups = dim_use[dim_use["name_key"].duplicated(keep=False)]
        raise ValueError(
            f"dim_team_park has duplicate name_key values; cannot join safely:\n{dups}"
        )

    log(f"[INFO] Loaded {len(dim_use)} teams from dim_team_park (league_id={league_id})")
    return dim_use


# ---------------------------------------------------------------------------
# Parse stats HTML for RS / RA
# ---------------------------------------------------------------------------

def load_team_batting_pitching(stats_html: Path, expected_teams: int) -> pd.DataFrame:
    if not stats_html.exists():
        raise FileNotFoundError(f"Stats HTML not found: {stats_html}")

    log(f"[INFO] Reading stats HTML from {stats_html}")
    tables = pd.read_html(stats_html)
    log(f"[DEBUG] stats_html: parsed {len(tables)} tables")

    batting_candidates = []
    pitching_candidates = []

    for i, t in enumerate(tables):
        cols = [str(c).strip() for c in t.columns]
        lower = [c.lower() for c in cols]

        if "team" in lower and "r" in lower:
            has_avg_like = any(k in lower for k in ["avg", "ab", "obp", "slg"])
            has_era_like = any(k in lower for k in ["era", "ip"])

            if has_avg_like:
                batting_candidates.append((i, t))
            if has_era_like:
                pitching_candidates.append((i, t))

    if not batting_candidates:
        raise ValueError("Could not find a team batting table with columns [Team, R, AVG/AB] in stats_html.")
    if not pitching_candidates:
        raise ValueError("Could not find a team pitching table with columns [Team, R, ERA/IP] in stats_html.")

    # Extract team + R columns
    def extract_team_r(df, label):
        cols = [str(c).strip() for c in df.columns]
        lower = [c.lower() for c in cols]
        team_col = cols[lower.index("team")]
        r_col = cols[lower.index("r")]

        out = df[[team_col, r_col]].copy()
        out = out.rename(columns={team_col: "team_name_src", r_col: label})
        out["name_key"] = out["team_name_src"].map(norm_name)
        return out

    bat_frames = [extract_team_r(tbl, "runs_for") for _, tbl in batting_candidates]
    pit_frames = [extract_team_r(tbl, "runs_against") for _, tbl in pitching_candidates]
    bat_df = pd.concat(bat_frames, ignore_index=True)
    pit_df = pd.concat(pit_frames, ignore_index=True)

    # Aggregate in case there are duplicates / multiple tables
    bat_agg = bat_df.groupby("name_key", as_index=False)["runs_for"].sum()
    pit_agg = pit_df.groupby("name_key", as_index=False)["runs_against"].sum()

    merged = pd.merge(bat_agg, pit_agg, on="name_key", how="outer", validate="1:1")

    if merged.shape[0] != expected_teams:
        log(
            f"[WARN] Combined batting/pitching rows={merged.shape[0]} (expected {expected_teams}); proceeding with aggregated set."
        )

    merged["runs_for"] = pd.to_numeric(merged["runs_for"], errors="coerce")
    merged["runs_against"] = pd.to_numeric(merged["runs_against"], errors="coerce")
    merged["run_diff"] = merged["runs_for"] - merged["runs_against"]

    log(f"[INFO] Derived RS/RA from stats HTML for {len(merged)} keys")
    return merged


# ---------------------------------------------------------------------------
# Parse standings HTML for W / L / PCT
# ---------------------------------------------------------------------------

def load_team_standings(standings_html: Path, expected_teams: int) -> pd.DataFrame:
    if not standings_html.exists():
        raise FileNotFoundError(f"Standings HTML not found: {standings_html}")

    log(f"[INFO] Reading standings HTML from {standings_html}")
    tables = pd.read_html(standings_html)
    log(f"[DEBUG] standings_html: parsed {len(tables)} tables")

    pieces = []

    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        lower = [c.lower() for c in cols]

        # We want tables that look like: Team, W, L, PCT, ...
        if "team" in lower and "w" in lower and "pct" in lower:
            team_col = cols[lower.index("team")]
            w_col = cols[lower.index("w")]
            l_col = cols[lower.index("l")]
            pct_col = cols[lower.index("pct")]

            sub = t[[team_col, w_col, l_col, pct_col]].copy()
            sub = sub.rename(
                columns={
                    team_col: "team_name_src",
                    w_col: "wins",
                    l_col: "losses",
                    pct_col: "pct",
                }
            )
            pieces.append(sub)

    if not pieces:
        raise ValueError("Could not find any standings tables with [Team, W, L, PCT] in standings_html.")

    standings = pd.concat(pieces, ignore_index=True)

    # Normalize team name and drop duplicates across divisions/wildcards
    standings["name_key"] = standings["team_name_src"].map(norm_name)
    standings["wins"] = pd.to_numeric(standings["wins"], errors="coerce")
    standings["losses"] = pd.to_numeric(standings["losses"], errors="coerce")
    standings["pct"] = pd.to_numeric(standings["pct"], errors="coerce")

    standings = standings.drop_duplicates(subset=["name_key"], keep="first")

    if standings.shape[0] != expected_teams:
        log(
            f"[WARN] Standings have {standings.shape[0]} unique teams (by name_key), "
            f"expected {expected_teams}."
        )

    log(f"[INFO] Derived W/L/PCT from standings HTML for {standings.shape[0]} teams")
    return standings


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_league_season_summary(season: int, league_id: int) -> pd.DataFrame:
    # Paths
    dim_path = Path("csv/out/star_schema/dim_team_park.csv")
    core_root = Path("csv/in/almanac_core") / str(season) / "leagues"
    stats_html = core_root / f"league_{league_id}_stats.html"
    standings_html = core_root / f"league_{league_id}_standings.html"

    log(f"[DEBUG] season={season}, league_id={league_id}")
    log(f"[DEBUG] dim_team_park={dim_path}")
    log(f"[DEBUG] stats_html={stats_html}")
    log(f"[DEBUG] standings_html={standings_html}")

    # Load dimension
    dim = load_dim_team_park(dim_path, league_id)
    expected_teams = len(dim)

    # Load RS/RA and standings
    rsra = load_team_batting_pitching(stats_html, expected_teams)
    st = load_team_standings(standings_html, expected_teams)

    # Join RS/RA + dimension
    teams = pd.merge(
        dim,
        rsra,
        on="name_key",
        how="left",
        validate="1:1",
    )

    # Join W/L/PCT
    teams = pd.merge(
        teams,
        st[["name_key", "wins", "losses", "pct"]],
        on="name_key",
        how="left",
        validate="1:1",
    )

    teams["season"] = season
    # league_id already present and filtered; keep as-is

    # Reorder columns
    cols = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
        "wins",
        "losses",
        "pct",
        "runs_for",
        "runs_against",
        "run_diff",
    ]
    teams = teams[cols].copy()

    # Type coercion and validation
    for c in ["wins", "losses", "pct", "runs_for", "runs_against", "run_diff"]:
        teams[c] = pd.to_numeric(teams[c], errors="coerce")

    # Check for NaNs in headline metrics
    bad = teams[
        teams[["wins", "losses", "pct", "runs_for", "runs_against", "run_diff"]].isna().any(axis=1)
    ]
    if not bad.empty:
        raise ValueError(
            "league_season_summary has NaNs in headline metric columns for these teams:\n"
            f"{bad[['team_name', 'team_abbr', 'wins', 'losses', 'pct', 'runs_for', 'runs_against', 'run_diff']]}"
        )

    # Sort in a stable way: conf/division, then wins desc, pct desc, team_name
    teams = teams.sort_values(
        by=["conf", "division", "wins", "pct", "team_name"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)

    # Helpful debug line for ATL if present
    atl = teams[teams["team_abbr"] == "ATL"]
    if not atl.empty:
        log("[DEBUG] ATL row (sanity check):")
        log(atl.to_string(index=False))

    return teams


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build league-season summary from almanac core + dim_team_park.")
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g. 1972")
    parser.add_argument("--league-id", type=int, required=True, help="League ID, e.g. 200")
    args = parser.parse_args(argv)

    df = build_league_season_summary(args.season, args.league_id)

    out_root = Path("csv/out/almanac") / str(args.season)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"league_season_summary_{args.season}_league{args.league_id}.csv"

    df.to_csv(out_path, index=False)
    log(f"[OK] Wrote league season summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
