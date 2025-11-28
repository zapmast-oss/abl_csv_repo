#!/usr/bin/env python
"""EB player spotlight brief for 1972 ABL (league_id=200)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def fmt_slash(row: pd.Series) -> str:
    parts = []
    for k in ["AVG", "OBP", "SLG"]:
        if k in row and pd.notna(row[k]):
            try:
                parts.append(f"{float(row[k]):.3f}")
            except Exception:
                parts.append(str(row[k]))
    return "/".join(parts) if parts else ""


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_hits_section(df: pd.DataFrame) -> list[str]:
    lines = ["## Top Bats of ’72"]
    df = df.head(15)
    for _, r in df.iterrows():
        slash = fmt_slash(r)
        extras = []
        for col in ["HR", "RBI", "PA", "AB", "OPS", "WAR"]:
            if col in r and pd.notna(r[col]):
                try:
                    val = float(r[col])
                    extras.append(f"{col}:{val:.3f}" if col == "OPS" else f"{col}:{val:.1f}")
                except Exception:
                    extras.append(f"{col}:{r[col]}")
        extras_str = ", ".join(extras)
        lines.append(
            f"- {r['player_name']}, {r['team_abbr']} ({r['team_name']}) — "
            f"{slash} {extras_str}"
        )
    lines.append("")
    return lines


def build_pitch_section(df: pd.DataFrame) -> list[str]:
    lines = ["## Aces on the Hill"]
    df = df.head(15)
    for _, r in df.iterrows():
        extras = []
        for col in ["ERA", "IP", "SO", "WHIP", "WAR", "SV"]:
            if col in r and pd.notna(r[col]):
                try:
                    val = float(r[col])
                    fmt = f"{val:.3f}" if col in {"ERA", "WHIP"} else f"{val:.1f}"
                    extras.append(f"{col}:{fmt}")
                except Exception:
                    extras.append(f"{col}:{r[col]}")
        extras_str = ", ".join(extras)
        lines.append(
            f"- {r['player_name']}, {r['team_abbr']} ({r['team_name']}) — {extras_str}"
        )
    lines.append("")
    return lines


def main() -> int:
    season = 1972
    league_id = 200
    base = Path("csv/out/almanac") / str(season)
    league = load_csv(base / f"league_season_summary_{season}_league{league_id}.csv")
    hitters = load_csv(base / f"player_hitting_leaders_{season}_league{league_id}.csv")
    pitchers = load_csv(base / f"player_pitching_leaders_{season}_league{league_id}.csv")

    # ensure team identity present
    key_cols = ["team_id", "team_abbr", "team_name", "conf", "division"]
    league_key = league[["team_id", "team_abbr", "team_name", "conf", "division"]].drop_duplicates()
    for df in (hitters, pitchers):
        for col in ["team_id", "team_abbr", "team_name"]:
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = df[col].astype(str)
        df[:] = df  # type: ignore
    hitters = hitters.merge(league_key, on=["team_abbr", "team_name"], how="left")
    pitchers = pitchers.merge(league_key, on=["team_abbr", "team_name"], how="left")

    lines: list[str] = []
    lines.append(f"# EB Player Spotlights {season} – Data Brief (DO NOT PUBLISH)")
    lines.append(f"_League ID {league_id}_")
    lines.append("")
    lines.append("Internal data brief for EB: player performance highlights for the inaugural season.")
    lines.append("")
    lines.extend(build_hits_section(hitters))
    lines.extend(build_pitch_section(pitchers))

    out_path = base / f"eb_player_spotlights_{season}_league{league_id}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[OK] Wrote player spotlight brief to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
