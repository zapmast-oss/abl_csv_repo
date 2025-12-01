#!/usr/bin/env python
"""
Build preseason hype brief for EB based on hyped players' season WAR.

- Parse the preseason prediction HTML to get distinct player IDs and display names.
- Look up full names from dim_player_profile.
- Look up season WAR and team abbreviation from star-schema batting and pitching facts.
- Bucket players into Over-delivered / Delivered / Under-delivered based on season WAR.
- Render markdown with lines like: Full Name (CIN) - WAR: 3.45

Rules:
- Use season WAR only (from fact tables), not career WAR.
- No "Free agent" labels. If team is missing, omit the parentheses.
- Put team abbreviation after the player name in parentheses, not after the team name.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from bs4 import BeautifulSoup


def _get_repo_root() -> Path:
    """
    Return the repo root (abl_csv_repo) based on this script's location.

    Assumes this file lives at: <repo_root>/csv/abl_scripts/z_abl_preseason_hype_any.py
    """
    return Path(__file__).resolve().parents[2]


def _parse_preseason_players(html_path: Path) -> pd.DataFrame:
    """
    Parse preseason prediction HTML and return a DataFrame with:

      player_id      int
      display_name   str  (as shown in the link, e.g. "M. Morales, CF")
      team_abbr_html str  (team abbreviation from the same row, e.g. "CHI")
    """
    if not html_path.exists():
        raise FileNotFoundError(f"Preseason HTML not found: {html_path}")

    html_text = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")

    records = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "player_" not in href:
            continue

        m = re.search(r"player_(\d+)\.html", href)
        if not m:
            continue
        player_id = int(m.group(1))

        tr = a.find_parent("tr")
        if tr is None:
            continue

        display_name = a.get_text(strip=True)

        # Try to find the first team link in this row
        team_abbr = ""
        for link in tr.find_all("a", href=True):
            if "teams/team_" in link["href"]:
                team_abbr = link.get_text(strip=True)
                break

        records.append(
            {
                "player_id": player_id,
                "display_name": display_name,
                "team_abbr_html": team_abbr,
            }
        )

    if not records:
        logging.warning("No preseason players found in HTML: %s", html_path)
        return pd.DataFrame(columns=["player_id", "display_name", "team_abbr_html"])

    df = pd.DataFrame.from_records(records)

    # Deduplicate by player_id to avoid multiple rows for the same player
    df = df.drop_duplicates(subset=["player_id"]).reset_index(drop=True)

    logging.info("Parsed %d preseason hype rows from HTML", len(df))
    return df


def _load_dim_player_profile(root: Path) -> pd.DataFrame:
    """
    Load dim_player_profile and return columns needed to build full names.

    Expected columns:
      - ID
      - First Name
      - Last Name
    """
    path = root / "csv" / "out" / "star_schema" / "dim_player_profile.csv"
    if not path.exists():
        raise FileNotFoundError(f"dim_player_profile not found at {path}")

    df = pd.read_csv(path)
    for col in ["ID", "First Name", "Last Name"]:
        if col not in df.columns:
            raise ValueError(f"dim_player_profile is missing required column '{col}'")

    return df[["ID", "First Name", "Last Name"]]


def _load_season_war(root: Path) -> pd.DataFrame:
    """
    Load season WAR and team abbreviation from star-schema batting and pitching facts.

    Returns a DataFrame with:
      ID           int
      TM           str  (team abbreviation, from batting if available, else pitching)
      WAR_season   float (WAR_bat + WAR_pit; missing treated as 0.0)
    """
    bat_path = root / "csv" / "out" / "star_schema" / "fact_player_batting.csv"
    pit_path = root / "csv" / "out" / "star_schema" / "fact_player_pitching.csv"

    if not bat_path.exists() and not pit_path.exists():
        raise FileNotFoundError("Neither batting nor pitching fact table found for players.")

    war_frames = []

    if bat_path.exists():
        bat = pd.read_csv(bat_path)
        required = {"ID", "TM", "WAR"}
        missing = required - set(bat.columns)
        if missing:
            raise ValueError(
                f"fact_player_batting missing columns {sorted(missing)} at {bat_path}"
            )
        bat = bat[["ID", "TM", "WAR"]].copy()
        bat.rename(columns={"TM": "TM_bat", "WAR": "WAR_bat"}, inplace=True)
        war_frames.append(bat)
    else:
        logging.warning("fact_player_batting not found at %s", bat_path)

    if pit_path.exists():
        pit = pd.read_csv(pit_path)
        required = {"ID", "TM_p1", "WAR"}
        missing = required - set(pit.columns)
        if missing:
            raise ValueError(
                f"fact_player_pitching missing columns {sorted(missing)} at {pit_path}"
            )
        pit = pit[["ID", "TM_p1", "WAR"]].copy()
        pit.rename(columns={"TM_p1": "TM_pit", "WAR": "WAR_pit"}, inplace=True)
        if war_frames:
            stats = war_frames[0].merge(pit, on="ID", how="outer")
        else:
            stats = pit
    else:
        logging.warning("fact_player_pitching not found at %s", pit_path)
        stats = war_frames[0]

    # Ensure WAR_bat and WAR_pit exist and are numeric
    if "WAR_bat" not in stats.columns:
        stats["WAR_bat"] = 0.0
    if "WAR_pit" not in stats.columns:
        stats["WAR_pit"] = 0.0

    stats["WAR_bat"] = pd.to_numeric(stats["WAR_bat"], errors="coerce").fillna(0.0)
    stats["WAR_pit"] = pd.to_numeric(stats["WAR_pit"], errors="coerce").fillna(0.0)

    stats["WAR_season"] = stats["WAR_bat"] + stats["WAR_pit"]

    # Team abbreviation: batting team preferred, else pitching team
    stats["TM_bat"] = stats.get("TM_bat", "")
    stats["TM_pit"] = stats.get("TM_pit", "")
    stats["TM"] = stats["TM_bat"]
    stats.loc[stats["TM"].isna() | (stats["TM"] == ""), "TM"] = stats["TM_pit"]

    out = stats[["ID", "TM", "WAR_season"]].copy()
    logging.info("Loaded season WAR for %d players", len(out))
    return out


def _build_full_name(row: pd.Series) -> str:
    """
    Build full name from dim_player_profile when possible,
    otherwise fall back to the display name from the HTML.
    """
    fn = row.get("First Name")
    ln = row.get("Last Name")
    if isinstance(fn, str) and isinstance(ln, str) and fn and ln:
        return f"{fn} {ln}"

    disp = str(row.get("display_name", "")).strip()
    if "," in disp:
        disp = disp.split(",", 1)[0].strip()
    return disp


def _bucket_war(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign players to Over-delivered / Delivered / Under-delivered buckets
    based on season WAR.

    - If at least 3 players, use 1/3 and 2/3 quantiles.
    - If fewer, fall back to min/max logic without crashing.
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    war_series = df["WAR_season"].astype(float).sort_values()

    if len(war_series) >= 3:
        low_cut = float(war_series.quantile(1.0 / 3.0))
        high_cut = float(war_series.quantile(2.0 / 3.0))
    else:
        low_cut = float(war_series.min())
        high_cut = float(war_series.max())

    def assign_bucket(w: float) -> str:
        if w >= high_cut:
            return "over"
        if w <= low_cut:
            return "under"
        return "delivered"

    df = df.copy()
    df["bucket"] = df["WAR_season"].astype(float).apply(assign_bucket)

    over = df[df["bucket"] == "over"].sort_values("WAR_season", ascending=False)
    delivered = df[df["bucket"] == "delivered"].sort_values(
        "WAR_season", ascending=False
    )
    under = df[df["bucket"] == "under"].sort_values("WAR_season", ascending=True)

    return over, delivered, under


def _render_markdown(df: pd.DataFrame, season: int) -> str:
    """
    Render the preseason hype markdown fragment.

    All lines are of the form:
      Full Name (TM) - WAR: 3.45

    If team abbreviation is unavailable for some player, omit the parentheses:
      Full Name - WAR: 3.45

    No "Free agent" strings are ever emitted here.
    """
    lines = []
    lines.append("## Preseason hype - who delivered?")
    lines.append(
        f"_Based on preseason predictions and {season} season WAR among hyped players._"
    )
    lines.append("")

    over, delivered, under = _bucket_war(df)

    def emit_bucket(title: str, bucket_df: pd.DataFrame) -> None:
        lines.append(f"**{title}**")
        if bucket_df.empty:
            lines.append("- (none)")
            lines.append("")
            return
        for _, row in bucket_df.iterrows():
            name = str(row["full_name"])
            team = str(row.get("team_abbr", "") or "").strip()
            war_val = float(row["WAR_season"])
            if team:
                lines.append(f"- {name} ({team}) - WAR: {war_val:.2f}")
            else:
                lines.append(f"- {name} - WAR: {war_val:.2f}")
        lines.append("")

    emit_bucket("Over-delivered", over)
    emit_bucket("Delivered", delivered)
    emit_bucket("Under-delivered", under)

    return "\n".join(lines).rstrip() + "\n"


def build_preseason_hype_fragment(
    season: int, league_id: int, preseason_html: Path | None
) -> Path:
    """
    Orchestrate the preseason hype brief construction for one season/league.

    Uses:
      - Preseason HTML (predictions) to identify the hyped players.
      - dim_player_profile for full names.
      - fact_player_batting + fact_player_pitching for season WAR and team abbreviations.
    """
    root = _get_repo_root()

    if preseason_html is None:
        preseason_html = (
            root
            / "csv"
            / "in"
            / "almanac_core"
            / str(season)
            / "leagues"
            / f"league_{league_id}_preseason_prediction_report.html"
        )

    logging.info("Using preseason HTML: %s", preseason_html)
    hype_df = _parse_preseason_players(preseason_html)

    if hype_df.empty:
        logging.warning("No hype players parsed; writing a minimal placeholder fragment.")
        out_dir = root / "csv" / "out" / "eb"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eb_preseason_hype_{season}_league{league_id}.md"
        out_path.write_text(
            "## Preseason hype - who delivered?\n"
            f"_Based on preseason predictions and {season} season WAR among hyped players._\n\n"
            "- (no hyped players parsed)\n",
            encoding="utf-8",
        )
        return out_path

    profiles = _load_dim_player_profile(root)
    stats = _load_season_war(root)

    merged = hype_df.merge(
        profiles, left_on="player_id", right_on="ID", how="left", suffixes=("", "_prof")
    )
    logging.info(
        "Hype rows whose names were enriched from dim_player_profile: %d",
        merged["First Name"].notna().sum(),
    )

    merged = merged.merge(
        stats, left_on="player_id", right_on="ID", how="left", suffixes=("", "_stat")
    )
    logging.info(
        "Hype rows matched to season stats: %d", merged["WAR_season"].notna().sum()
    )

    merged["full_name"] = merged.apply(_build_full_name, axis=1)
    merged["team_abbr"] = merged["TM"].fillna("").astype(str)
    merged["WAR_season"] = pd.to_numeric(
        merged["WAR_season"], errors="coerce"
    ).fillna(0.0)

    md = _render_markdown(merged, season=season)

    out_dir = root / "csv" / "out" / "eb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eb_preseason_hype_{season}_league{league_id}.md"
    out_path.write_text(md, encoding="utf-8")

    logging.info("Wrote preseason hype fragment to %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build EB preseason hype fragment from almanac preseason predictions."
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year, e.g. 1972 or 1980",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        default=200,
        help="League ID (default 200 for ABL).",
    )
    parser.add_argument(
        "--preseason-html",
        type=str,
        default=None,
        help=(
            "Optional explicit path to preseason prediction HTML. "
            "If omitted, defaults to "
            "csv/in/almanac_core/{season}/leagues/league_{league_id}_preseason_prediction_report.html"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    html_path = Path(args.preseason_html) if args.preseason_html else None
    out_path = build_preseason_hype_fragment(
        season=args.season, league_id=args.league_id, preseason_html=html_path
    )
    print(f"[OK] Wrote preseason hype brief to {out_path}")


if __name__ == "__main__":
    main()
