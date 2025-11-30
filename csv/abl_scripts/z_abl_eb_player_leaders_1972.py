#!/usr/bin/env python
"""EB player leaders brief for a season/league (team-aware)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from eb_text_utils import normalize_eb_text


def log(msg: str) -> None:
    print(msg, flush=True)


def norm_key(val: str) -> str:
    return "".join(ch for ch in str(val).strip().lower() if ch.isalnum())


def load_team_lookup(team_path: Path) -> pd.DataFrame:
    team_df = pd.read_csv(team_path)
    team_id_col = "team_id" if "team_id" in team_df.columns else "ID"
    abbr_col = "team_abbr" if "team_abbr" in team_df.columns else "Abbr"
    name_col = "team_name" if "team_name" in team_df.columns else ("Team Name" if "Team Name" in team_df.columns else "Name")
    conf_col = "conf" if "conf" in team_df.columns else ("SL" if "SL" in team_df.columns else None)
    div_col = "division" if "division" in team_df.columns else ("DIV" if "DIV" in team_df.columns else None)
    use_cols = [team_id_col, abbr_col, name_col]
    if conf_col:
        use_cols.append(conf_col)
    if div_col:
        use_cols.append(div_col)
    df = team_df[use_cols].rename(
        columns={team_id_col: "team_id", abbr_col: "team_abbr", name_col: "team_name", conf_col: "conf", div_col: "division"}
    )
    df["team_key"] = df["team_name"].map(norm_key)
    return df


def load_player_lookup(profile_path: Path, team_lookup: pd.DataFrame) -> pd.DataFrame:
    prof = pd.read_csv(profile_path)
    name_col = "player_name" if "player_name" in prof.columns else ("Name" if "Name" in prof.columns else None)
    team_col = None
    for cand in ["TM", "Team", "ORG"]:
        if cand in prof.columns:
            team_col = cand
            break
    id_col = "player_id" if "player_id" in prof.columns else ("ID" if "ID" in prof.columns else None)
    if name_col is None or team_col is None:
        return pd.DataFrame(columns=["player_name", "team_id", "team_abbr", "team_name", "conf", "division"])
    pf = prof[[c for c in [name_col, team_col, id_col] if c]].rename(
        columns={name_col: "player_name", team_col: "team_name", id_col: "player_id"}
    )
    pf["player_key"] = pf["player_name"].map(norm_key)
    pf["team_key"] = pf["team_name"].map(norm_key)
    lookup = pf.merge(team_lookup.drop(columns=["team_name"]), on="team_key", how="left")
    lookup = lookup.drop_duplicates(subset=["player_key"])
    return lookup


def team_label(row: pd.Series) -> str:
    abbr = row.get("team_abbr")
    name = row.get("team_name")
    if pd.notna(name) and pd.notna(abbr):
        return f"{name} ({abbr})"
    if pd.notna(name):
        return str(name)
    if pd.notna(abbr):
        return str(abbr)
    return "Free agent"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    log(f"[INFO] Loaded {len(df)} rows from {path}")
    log(f"[DEBUG] Columns in {path.name}: {df.columns.tolist()}")
    return df


def find_best(df: pd.DataFrame, stat: str, ascending: bool, qualifier: Optional[pd.Series] = None) -> Optional[pd.Series]:
    if stat not in df.columns:
        log(f"[WARN] Missing stat {stat}; skipping leader.")
        return None
    working = df.copy()
    working[stat] = pd.to_numeric(working[stat], errors="coerce")
    if qualifier is not None:
        working = working[qualifier]
    working = working.dropna(subset=[stat])
    if working.empty:
        log(f"[WARN] No rows after filtering for {stat}.")
        return None
    idx = working[stat].idxmin() if ascending else working[stat].idxmax()
    return working.loc[idx]


def enrich_with_team(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    if df.empty or lookup.empty:
        return df
    df = df.copy()
    df["player_key"] = df["player_name"].map(norm_key)
    merged = df.merge(lookup, on="player_key", how="left", suffixes=("", "_lkp"))
    for col in ["team_id", "team_abbr", "team_name", "conf", "division"]:
        lkp = f"{col}_lkp"
        if lkp in merged.columns:
            merged[col] = merged[col].combine_first(merged[lkp])
            merged = merged.drop(columns=[lkp])
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Build EB player leaders brief for a season/league.")
    parser.add_argument("--season", type=int, default=1972)
    parser.add_argument("--league-id", type=int, default=200)
    parser.add_argument("--player-profile", type=Path, default=Path("csv/out/star_schema/dim_player_profile.csv"))
    parser.add_argument("--team-dim", type=Path, default=Path("csv/out/star_schema/dim_team_park.csv"))
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    base = Path("csv/out/almanac") / str(season)
    batting_full = load_csv(base / f"player_batting_{season}_league{league_id}.csv")
    pitching_full = load_csv(base / f"player_pitching_{season}_league{league_id}.csv")

    team_lookup = load_team_lookup(args.team_dim)
    player_lookup = load_player_lookup(args.player_profile, team_lookup)
    batting_full = enrich_with_team(batting_full, player_lookup)
    pitching_full = enrich_with_team(pitching_full, player_lookup)

    pa_col = "PA" if "PA" in batting_full.columns else None
    ab_col = "AB" if "AB" in batting_full.columns else None
    if pa_col:
        batting_full[pa_col] = pd.to_numeric(batting_full[pa_col], errors="coerce")
        qualifier = batting_full[pa_col] >= 400
    elif ab_col:
        batting_full[ab_col] = pd.to_numeric(batting_full[ab_col], errors="coerce")
        qualifier = batting_full[ab_col] >= 350
    else:
        qualifier = pd.Series([True] * len(batting_full))

    leaders = []
    for stat, asc in [("HR", False), ("SB", False), ("AVG", False), ("OPS", False)]:
        row = find_best(batting_full, stat, ascending=asc, qualifier=qualifier)
        if row is None:
            continue
        leaders.append(
            {
                "stat_name": stat,
                "stat_value": row[stat],
                "player_name": row["player_name"],
                "team_abbr": row.get("team_abbr"),
                "team_name": row.get("team_name"),
            }
        )

    ip_col = "IP" if "IP" in pitching_full.columns else None
    if ip_col:

        def ip_to_float(val):
            try:
                if isinstance(val, str) and "." in val:
                    whole, frac = val.split(".")
                    return float(whole) + float(frac) / 10
                return float(val)
            except Exception:
                return pd.NA

        pitching_full[ip_col] = pitching_full[ip_col].map(ip_to_float)
        pitcher_qual = pitching_full[ip_col] >= 162
    else:
        pitcher_qual = pd.Series([True] * len(pitching_full))

    for stat, asc in [("ERA", True), ("SO", False), ("SV", False)]:
        row = find_best(pitching_full, stat, ascending=asc, qualifier=pitcher_qual)
        if row is None:
            continue
        leaders.append(
            {
                "stat_name": stat,
                "stat_value": row[stat],
                "player_name": row["player_name"],
                "team_abbr": row.get("team_abbr"),
                "team_name": row.get("team_name"),
            }
        )

    if not leaders:
        raise RuntimeError("No leaders found to report.")

    md_lines: list[str] = []
    md_lines.append(f"# EB Player Leaders {season} — Career Context (DO NOT PUBLISH)")
    md_lines.append(f"_League ID {league_id}_")
    md_lines.append("")
    md_lines.append(f"## Career batting leaders through {season}")
    for leader in leaders:
        if leader["stat_name"] in {"HR", "SB", "AVG", "OPS"}:
            md_lines.append(
                f"- {leader['stat_name']} leader: {leader['player_name']} — {team_label(pd.Series(leader))} — {leader['stat_value']}"
            )
    md_lines.append("")
    md_lines.append(f"## Career pitching leaders through {season}")
    for leader in leaders:
        if leader["stat_name"] in {"ERA", "SO", "SV"}:
            md_lines.append(
                f"- {leader['stat_name']} leader: {leader['player_name']} — {team_label(pd.Series(leader))} — {leader['stat_value']}"
            )
    md_lines.append("")

    out_path = base / f"eb_player_leaders_{season}_league{league_id}.md"
    full_text = "\n".join(md_lines)
    full_text = normalize_eb_text(full_text)
    out_path.write_text(full_text, encoding="utf-8")
    log(f"[OK] Wrote player leaders brief to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
