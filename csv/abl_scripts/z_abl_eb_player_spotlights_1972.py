#!/usr/bin/env python
"""EB player spotlight brief for any season/league (team-aware, CLI-driven)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from eb_text_utils import canonicalize_team_city, normalize_eb_text


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
    id_col = "player_id" if "player_id" in prof.columns else ("ID" if "ID" in prof.columns else None)
    team_col = None
    for cand in ["TM", "Team", "ORG"]:
        if cand in prof.columns:
            team_col = cand
            break
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
    name = canonicalize_team_city(row.get("team_name"))
    abbr = row.get("team_abbr")
    if pd.notna(name) and pd.notna(abbr):
        return f"{name} ({abbr})"
    if pd.notna(name):
        return str(name)
    if pd.notna(abbr):
        return str(abbr)
    return "Free agent"


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    log(f"[INFO] Loaded {len(df)} rows from {path}")
    log(f"[DEBUG] Columns in {path.name}: {df.columns.tolist()}")
    return df


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


def fmt_slash(row: pd.Series) -> str:
    parts: List[str] = []
    for k in ("AVG", "OBP", "SLG"):
        if k in row and pd.notna(row[k]):
            try:
                parts.append(f"{float(row[k]):.3f}")
            except Exception:
                parts.append(str(row[k]))
    return "/".join(parts) if parts else ""


def build_hits_section(df: pd.DataFrame, season: int) -> list[str]:
    lines = [f"## Top bats — career through {season}"]
    for _, r in df.head(15).iterrows():
        bits = []
        slash = fmt_slash(r)
        for col in ("HR", "RBI", "OPS", "WAR"):
            if col in r and pd.notna(r[col]):
                try:
                    val = float(r[col])
                    bits.append(f"{col}: {val:.3f}" if col == "OPS" else f"{col}: {val:.1f}")
                except Exception:
                    bits.append(f"{col}: {r[col]}")
        tlabel = team_label(r)
        parts = [f"- {r['player_name']}"]
        if tlabel:
            parts.append(tlabel)
        if slash:
            parts.append(slash)
        if bits:
            parts.append(", ".join(bits))
        lines.append(" — ".join(parts))
    lines.append("")
    return lines


def build_pitch_section(df: pd.DataFrame, season: int) -> list[str]:
    lines = [f"## Aces on the hill — career through {season}"]
    for _, r in df.head(15).iterrows():
        bits = []
        for col in ("ERA", "IP", "SO", "WHIP", "WAR", "SV"):
            if col in r and pd.notna(r[col]):
                try:
                    val = float(r[col])
                    bits.append(f"{col}: {val:.3f}" if col in {"ERA", "WHIP"} else f"{col}: {val:.1f}")
                except Exception:
                    bits.append(f"{col}: {r[col]}")
        tlabel = team_label(r)
        parts = [f"- {r['player_name']}"]
        if tlabel:
            parts.append(tlabel)
        if bits:
            parts.append(", ".join(bits))
        lines.append(" — ".join(parts))
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="EB player spotlight brief for a season/league.")
    parser.add_argument("--season", type=int, default=1972)
    parser.add_argument("--league-id", type=int, default=200)
    parser.add_argument("--player-profile", type=Path, default=Path("csv/out/star_schema/dim_player_profile.csv"))
    parser.add_argument("--team-dim", type=Path, default=Path("csv/out/star_schema/dim_team_park.csv"))
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    base = Path("csv/out/almanac") / str(season)
    hitters = load_df(base / f"player_hitting_leaders_{season}_league{league_id}.csv")
    pitchers = load_df(base / f"player_pitching_leaders_{season}_league{league_id}.csv")

    team_lookup = load_team_lookup(args.team_dim)
    player_lookup = load_player_lookup(args.player_profile, team_lookup)
    hitters = enrich_with_team(hitters, player_lookup)
    pitchers = enrich_with_team(pitchers, player_lookup)

    lines: list[str] = []
    lines.append(f"# EB Player Spotlights {season} — Career Context (DO NOT PUBLISH)")
    lines.append(f"_League ID {league_id}_")
    lines.append("")
    lines.append("Internal data brief for EB: player performance highlights for the season.")
    lines.append("")
    lines.extend(build_hits_section(hitters, season))
    lines.extend(build_pitch_section(pitchers, season))

    out_path = base / f"eb_player_spotlights_{season}_league{league_id}.md"
    full_text = "\n".join(lines)
    full_text = normalize_eb_text(full_text)
    out_path.write_text(full_text, encoding="utf-8")
    log(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
