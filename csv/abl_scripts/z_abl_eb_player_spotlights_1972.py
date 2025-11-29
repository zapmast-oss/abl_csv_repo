#!/usr/bin/env python
"""EB player spotlight brief for a season/league (defaults to 1972/200)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    log(f"[INFO] Loaded {len(df)} rows from {path}")
    log(f"[DEBUG] Columns in {path.name}: {df.columns.tolist()}")
    return df


def fmt_slash(row: pd.Series) -> str:
    parts: List[str] = []
    for k in ("AVG", "OBP", "SLG"):
        if k in row and pd.notna(row[k]):
            try:
                parts.append(f"{float(row[k]):.3f}")
            except Exception:
                parts.append(str(row[k]))
    return "/".join(parts) if parts else ""


def attach_team(df: pd.DataFrame, stats: pd.DataFrame, league: pd.DataFrame) -> pd.DataFrame:
    merged = df.copy()
    # ensure columns exist
    for col in ["team_abbr", "team_name", "team_id", "conf", "division"]:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged["team_abbr"] = merged["team_abbr"].astype(str)
    merged["team_name"] = merged["team_name"].astype(str)
    stats = stats.copy()
    stats["team_abbr"] = stats["team_abbr"].astype(str)
    stats["team_name"] = stats["team_name"].astype(str)
    subset = stats[["player_name", "team_abbr", "team_name", "team_id", "conf", "division"]].drop_duplicates()
    merged = merged.merge(subset, on="player_name", how="left", suffixes=("", "_stat"))
    for col in ["team_abbr", "team_name", "team_id", "conf", "division"]:
        merged[col] = merged[col].fillna(merged.get(f"{col}_stat"))
        stat_col = f"{col}_stat"
        if stat_col in merged.columns:
            merged = merged.drop(columns=[stat_col])
    league_key = league[["team_abbr", "team_name", "team_id", "conf", "division"]].drop_duplicates()
    league_key["team_abbr"] = league_key["team_abbr"].astype(str)
    league_key["team_name"] = league_key["team_name"].astype(str)
    merged = merged.merge(league_key, on=["team_abbr", "team_name"], how="left", suffixes=("", "_lg"))
    for col in ["team_id", "conf", "division"]:
        merged[col] = merged[col].fillna(merged.get(f"{col}_lg"))
        lg_col = f"{col}_lg"
        if lg_col in merged.columns:
            merged = merged.drop(columns=[lg_col])
    missing = merged["team_abbr"].isna().sum()
    if missing:
        log(f"[WARN] {missing} spotlight rows missing team_abbr after joins.")
    return merged


def build_hits_section(df: pd.DataFrame) -> list[str]:
    lines = ["## Top Bats of ’72"]
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
        team = (r.get("team_abbr") or "").strip()
        team_name = (r.get("team_name") or "").strip()
        parts = [f"- {r['player_name']}"]
        if team or team_name:
            parts.append(f"{team} ({team_name})" if team_name else team)
        if slash:
            parts.append(slash)
        if bits:
            parts.append(", ".join(bits))
        lines.append(" — ".join(parts))
    lines.append("")
    return lines


def build_pitch_section(df: pd.DataFrame) -> list[str]:
    lines = ["## Aces on the Hill"]
    for _, r in df.head(15).iterrows():
        bits = []
        for col in ("ERA", "IP", "SO", "WHIP", "WAR", "SV"):
            if col in r and pd.notna(r[col]):
                try:
                    val = float(r[col])
                    fmt = f"{val:.3f}" if col in {"ERA", "WHIP"} else f"{val:.1f}"
                    bits.append(f"{col}: {fmt}")
                except Exception:
                    bits.append(f"{col}: {r[col]}")
        team = (r.get("team_abbr") or "").strip()
        team_name = (r.get("team_name") or "").strip()
        parts = [f"- {r['player_name']}"]
        if team or team_name:
            parts.append(f"{team} ({team_name})" if team_name else team)
        if bits:
            parts.append(", ".join(bits))
        lines.append(" — ".join(parts))
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="EB player spotlight brief for a season/league.")
    parser.add_argument("--season", type=int, default=1972)
    parser.add_argument("--league-id", type=int, default=200)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    base = Path("csv/out/almanac") / str(season)
    league = load_csv(base / f"league_season_summary_{season}_league{league_id}.csv")
    hitters = load_csv(base / f"player_hitting_leaders_{season}_league{league_id}.csv")
    pitchers = load_csv(base / f"player_pitching_leaders_{season}_league{league_id}.csv")
    batting_full = load_csv(base / f"player_batting_{season}_league{league_id}.csv")
    pitching_full = load_csv(base / f"player_pitching_{season}_league{league_id}.csv")

    hitters = attach_team(hitters, batting_full, league)
    pitchers = attach_team(pitchers, pitching_full, league)

    lines: list[str] = []
    lines.append(f"# EB Player Spotlights {season} – Data Brief (DO NOT PUBLISH)")
    lines.append(f"_League ID {league_id}_")
    lines.append("")
    lines.append("Internal data brief for EB: player performance highlights for the season.")
    lines.append("")
    lines.extend(build_hits_section(hitters))
    lines.extend(build_pitch_section(pitchers))

    out_path = base / f"eb_player_spotlights_{season}_league{league_id}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[OK] Wrote player spotlight brief to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
