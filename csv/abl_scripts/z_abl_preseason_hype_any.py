#!/usr/bin/env python3
"""
Build EB preseason hype brief for a given ABL season.

Key rules:
- Parse preseason prediction HTML directly, pull player_id from links like player_8025.html.
- Map player_id to full name via dim_player_profile.
- Pull SEASON WAR and team abbreviation (TM) from star_schema/fact_player_batting.csv
  for the requested season only.
- Rank hyped players by SEASON WAR (not career).
- Split into three buckets by rank:
    Over-delivered   = top third
    Delivered        = middle third
    Under-delivered  = bottom third
- Render markdown like:
    - Scott Reis (CIN) - WAR: 0.80

No "Free agent" labels, no "(nan)" team names, no team names spelled out.
Only "Full Name (ABBR) - WAR: X.XX".
"""

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EB preseason hype brief for a season."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument(
        "--preseason-html",
        type=str,
        required=True,
        help="Path to league_XXX_preseason_prediction_report.html",
    )
    return parser.parse_args()


def parse_preseason_html(html_path: Path) -> pd.DataFrame:
    """
    Extract unique player_ids from preseason prediction HTML.

    We scan <tr> rows where the first <td> contains a link to ../players/player_XXXX.html.
    The XXXX portion is the numeric player_id. We keep one row per player_id.
    """
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(text, "html.parser")

    rows: List[Dict[str, Any]] = []
    seen_ids = set()

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 1:
            continue

        a = tds[0].find("a", href=True)
        if not a:
            continue

        href = a["href"]
        m = re.search(r"player_(\d+)\.html", href)
        if not m:
            continue

        player_id = int(m.group(1))
        if player_id in seen_ids:
            continue
        seen_ids.add(player_id)

        raw_name = a.get_text(strip=True)
        rows.append(
            {
                "player_id": player_id,
                "raw_name": raw_name,
            }
        )

    df = pd.DataFrame(rows)
    log.info("Parsed %d unique preseason hype players from HTML", len(df))
    return df


def load_dim_player_profile(repo_root: Path) -> pd.DataFrame:
    """
    Load dim_player_profile for name resolution.

    We rely on:
    - ID
    - Name
    - First Name
    - Last Name
    """
    path = repo_root / "csv" / "out" / "star_schema" / "dim_player_profile.csv"
    df = pd.read_csv(path)
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    return df


def load_fact_player_batting(repo_root: Path, season: int) -> pd.DataFrame:
    """
    Load fact_player_batting for SEASON stats.

    Columns used:
    - ID       (player id)
    - season   (season year)
    - TM       (team abbreviation)
    - WAR      (season WAR)

    We filter to the requested season and then collapse to one row per ID:
    keep the row with the highest WAR for that season (in case of trades).
    """
    path = repo_root / "csv" / "out" / "star_schema" / "fact_player_batting.csv"
    df = pd.read_csv(path)

    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df = df[df["season"] == season].copy()

    # Normalize WAR and TM
    df["WAR"] = pd.to_numeric(df.get("WAR"), errors="coerce").fillna(0.0)
    df["TM"] = df.get("TM", "").fillna("").astype(str).str.strip()

    if df.empty:
        return df

    # For players with multiple rows in a season (trades), keep row with max WAR.
    df = df.sort_values(["ID", "WAR"], ascending=[True, False])
    df = df.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)
    return df


def build_hype_table(
    hype_df: pd.DataFrame,
    dim_profile: pd.DataFrame,
    fact_batting: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join hype players to dim + fact tables to get:
    - full_name
    - team_abbr
    - WAR (season)
    """
    if hype_df.empty:
        return hype_df

    # Merge hype -> dim_player_profile on ID
    df = hype_df.merge(
        dim_profile[["ID", "Name", "First Name", "Last Name"]],
        left_on="player_id",
        right_on="ID",
        how="left",
    )

    # Merge -> fact_player_batting for season WAR + team abbreviation
    df = df.merge(
        fact_batting[["ID", "TM", "WAR"]],
        on="ID",
        how="left",
        suffixes=("", "_bat"),
    )

    def pick_name(row: pd.Series) -> str:
        name = row.get("Name")
        if isinstance(name, str) and name.strip():
            return name.strip()

        fn = str(row.get("First Name") or "").strip()
        ln = str(row.get("Last Name") or "").strip()
        combined = (fn + " " + ln).strip()
        if combined:
            return combined

        raw = row.get("raw_name")
        return str(raw or "Unknown Player").strip()

    df["full_name"] = df.apply(pick_name, axis=1)

    # Team abbreviation from fact table (season-specific)
    df["team_abbr"] = df.get("TM", "").fillna("").astype(str).str.strip()

    # Ensure WAR is numeric and default 0.0
    df["WAR"] = pd.to_numeric(df.get("WAR"), errors="coerce").fillna(0.0)

    # Drop rows where name really could not be resolved
    df = df[~df["full_name"].eq("Unknown Player")].copy()

    # Sort by WAR desc, then name asc for stable ordering
    df = df.sort_values(["WAR", "full_name"], ascending=[False, True]).reset_index(
        drop=True
    )

    return df


def bucketize_by_war(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split into three buckets by WAR rank:

    - Over-delivered   = top third
    - Delivered        = middle third
    - Under-delivered  = bottom third
    """
    if df.empty:
        return df

    n = len(df)
    if n <= 3:
        labels: List[str] = []
        for idx in range(n):
            if idx == 0:
                labels.append("over")
            elif idx == n - 1:
                labels.append("under")
            else:
                labels.append("delivered")
        df["bucket"] = labels
        return df

    top_cut = int(math.ceil(n / 3.0))
    mid_cut = int(math.ceil(2.0 * n / 3.0))

    bucket_labels: List[str] = []
    for idx in range(n):
        if idx < top_cut:
            bucket_labels.append("over")
        elif idx < mid_cut:
            bucket_labels.append("delivered")
        else:
            bucket_labels.append("under")

    df["bucket"] = bucket_labels
    return df


def render_markdown(df: pd.DataFrame, season: int) -> str:
    """
    Render the preseason hype section as markdown.

    Format:

    ## Preseason hype - who delivered?
    _Based on preseason predictions and 1980 season WAR among hyped players._

    **Over-delivered**
    - Scott Reis (CIN) - WAR: 0.80
    """
    lines: List[str] = []
    lines.append("## Preseason hype - who delivered?")
    lines.append(
        "_Based on preseason predictions and {} season WAR among hyped players._".format(
            season
        )
    )
    lines.append("")

    def emit_bucket(label: str, heading: str) -> None:
        bucket = df[df["bucket"] == label]
        if bucket.empty:
            return
        lines.append(heading)
        for _, row in bucket.iterrows():
            name = row["full_name"]
            abbr = row.get("team_abbr") or ""
            war_val = float(row.get("WAR") or 0.0)
            if abbr:
                lines.append("- {} ({}) - WAR: {:.2f}".format(name, abbr, war_val))
            else:
                lines.append("- {} - WAR: {:.2f}".format(name, war_val))
        lines.append("")

    emit_bucket("over", "**Over-delivered**")
    emit_bucket("delivered", "**Delivered**")
    emit_bucket("under", "**Under-delivered**")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    setup_logging()
    args = parse_args()
    season = args.season
    league_id = args.league_id

    # repo_root: .../abl_csv_repo
    repo_root = Path(__file__).resolve().parents[3]
    log.info("Repo root resolved to %s", repo_root)

    preseason_html_path = Path(args.preseason_html)
    if not preseason_html_path.is_file():
        log.error("Preseason HTML not found: %s", preseason_html_path)
        return 1

    hype_df = parse_preseason_html(preseason_html_path)

    if hype_df.empty:
        log.warning("No hype players parsed; writing empty brief.")
        md_text = (
            "## Preseason hype - who delivered?\n"
            "_No preseason hype data found in the predictions report._\n"
        )
    else:
        dim_profile = load_dim_player_profile(repo_root)
        fact_batting = load_fact_player_batting(repo_root, season)
        hype_table = build_hype_table(hype_df, dim_profile, fact_batting)
        hype_table = bucketize_by_war(hype_table)
        md_text = render_markdown(hype_table, season)

    out_dir = repo_root / "csv" / "out" / "eb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eb_preseason_hype_{}_league{}.md".format(
        season, league_id
    )
    out_path.write_text(md_text, encoding="utf-8")
    log.info("Wrote preseason hype brief to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
