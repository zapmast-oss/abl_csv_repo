#!/usr/bin/env python
"""
Build preseason hype brief for EB based on hyped players' season WAR.

Rules:
- Parse preseason prediction HTML by player_id.
- Resolve full name from dim_player_profile (First Name + Last Name).
- Resolve team abbreviation from season stats, then HTML, then profile team/org fields.
- Never emit "Free agent" or "nan"; hard fail if any player lacks name or team.
- Emit markdown: Full Name (TM) - WAR: X.XX.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and CLI
# ---------------------------------------------------------------------------

def get_repo_root() -> Path:
    """Return repo root assuming this file lives under csv/abl_scripts."""
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EB preseason hype fragment.")
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g. 1980")
    parser.add_argument("--league-id", type=int, default=200, help="League ID (default 200)")
    parser.add_argument(
        "--preseason-html",
        type=str,
        default=None,
        help=(
            "Optional explicit path to preseason prediction HTML. "
            "Defaults to csv/in/almanac_core/{season}/leagues/league_{league_id}_preseason_prediction_report.html"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional output markdown path override.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def parse_preseason_players(html_path: Path) -> pd.DataFrame:
    """Parse preseason prediction HTML and return hype players.

    Columns:
      player_id      int
      raw_name       str
      raw_team_abbr  str
    """
    if not html_path.exists():
        raise FileNotFoundError(f"Preseason HTML not found: {html_path}")

    html_text = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")

    records: List[dict] = []
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
        raw_name = a.get_text(strip=True)
        team_abbr = ""
        for link in tr.find_all("a", href=True):
            if "team_" in link["href"]:
                team_abbr = link.get_text(strip=True)
                break
        records.append({"player_id": player_id, "raw_name": raw_name, "raw_team_abbr": team_abbr})

    df = pd.DataFrame.from_records(records)
    df = df.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    logger.info("Parsed %d preseason hype players from HTML", len(df))
    return df


# ---------------------------------------------------------------------------
# Load reference data
# ---------------------------------------------------------------------------

def load_dim_player_profile(root: Path) -> pd.DataFrame:
    """Load player profile with ID, names, and optional team/org fields."""
    path = root / "csv" / "out" / "star_schema" / "dim_player_profile.csv"
    if not path.exists():
        raise FileNotFoundError(f"dim_player_profile not found at {path}")
    df = pd.read_csv(path)
    required = {"ID", "First Name", "Last Name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dim_player_profile missing columns: {sorted(missing)}")
    keep = ["ID", "First Name", "Last Name"]
    for opt in ["TM", "TM.1", "ORG", "ORG.1"]:
        if opt in df.columns:
            keep.append(opt)
    df = df[keep].copy()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    df["full_name"] = (
        df["First Name"].fillna("").astype(str).str.strip()
        + " "
        + df["Last Name"].fillna("").astype(str).str.strip()
    ).str.strip()
    return df


def load_season_war(root: Path, season: int, league_id: int) -> pd.DataFrame:
    """Load season WAR and team abbreviation from batting/pitching fact tables.

    Tries season-specific files first, then generic. Sums batting + pitching WAR.
    """
    base = root / "csv" / "out" / "star_schema"

    def fact_candidates(prefix: str) -> List[Path]:
        return [
            base / f"{prefix}_{season}_league{league_id}.csv",
            base / f"{prefix}_{season}.csv",
            base / f"{prefix}.csv",
        ]

    def load_fact(paths: List[Path], team_cols: List[str], war_col: str, team_label: str) -> pd.DataFrame:
        for p in paths:
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "season" in df.columns:
                df = df[df["season"] == season]
            team_col = next((c for c in team_cols if c in df.columns), None)
            if team_col is None or war_col not in df.columns or "ID" not in df.columns:
                continue
            out = df[["ID", team_col, war_col]].copy()
            out.rename(columns={team_col: f"team_abbr_{team_label}", war_col: f"WAR_{team_label}"}, inplace=True)
            out["ID"] = pd.to_numeric(out["ID"], errors="coerce").astype("Int64")
            out[f"WAR_{team_label}"] = pd.to_numeric(out[f"WAR_{team_label}"], errors="coerce").fillna(0.0)
            return out
        return pd.DataFrame()

    bat = load_fact(fact_candidates("fact_player_batting"), ["TM", "team_abbr", "Tm", "tm"], "WAR", "bat")
    pit = load_fact(fact_candidates("fact_player_pitching"), ["TM_p1", "TM", "team_abbr", "Tm", "tm"], "WAR", "pit")

    if bat.empty and pit.empty:
        raise FileNotFoundError("No season WAR found in batting or pitching fact tables.")

    if bat.empty:
        stats = pit
    elif pit.empty:
        stats = bat
    else:
        stats = bat.merge(pit, on="ID", how="outer")

    if "WAR_bat" not in stats.columns:
        stats["WAR_bat"] = 0.0
    if "WAR_pit" not in stats.columns:
        stats["WAR_pit"] = 0.0

    stats["WAR_season"] = stats["WAR_bat"] + stats["WAR_pit"]

    stats["team_abbr_stats"] = stats.get("team_abbr_bat", "")
    mask = stats["team_abbr_stats"].isna() | (stats["team_abbr_stats"] == "")
    stats.loc[mask, "team_abbr_stats"] = stats.get("team_abbr_pit", "")

    return stats[["ID", "team_abbr_stats", "WAR_season"]]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_team_abbr(row: pd.Series) -> str:
    """Choose first non-empty, non-free-agent team abbreviation."""
    candidates = [
        str(row.get("team_abbr_stats", "")),
        str(row.get("raw_team_abbr", "")),
        str(row.get("TM", "")),
        str(row.get("TM.1", "")),
        str(row.get("ORG", "")),
        str(row.get("ORG.1", "")),
    ]
    for cand in candidates:
        val = cand.strip()
        if not val:
            continue
        if val.lower() in {"free agent", "fa", "nan"}:
            continue
        return val
    return ""


def resolve_players(hype: pd.DataFrame, profiles: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Join hype with profiles and stats; enforce full_name/team completeness."""
    merged = hype.merge(profiles, left_on="player_id", right_on="ID", how="left", suffixes=("", "_prof"))
    logging.info("Hype rows matched to profiles: %d of %d", merged["First Name"].notna().sum(), len(merged))

    merged = merged.merge(stats, left_on="player_id", right_on="ID", how="left", suffixes=("", "_stat"))
    logging.info("Hype rows matched to season stats: %d of %d", merged["WAR_season"].notna().sum(), len(merged))

    merged["full_name"] = merged.apply(lambda r: f"{r.get('First Name', '').strip()} {r.get('Last Name', '').strip()}".strip(), axis=1)
    merged["team_abbr"] = merged.apply(resolve_team_abbr, axis=1)
    merged["war_value"] = pd.to_numeric(merged.get("WAR_season", 0), errors="coerce").fillna(0.0)

    bad_name = merged[merged["full_name"].isna() | (merged["full_name"].str.strip() == "")]
    bad_team = merged[merged["team_abbr"].isna() | (merged["team_abbr"].str.strip() == "")]
    if not bad_name.empty or not bad_team.empty:
        logging.error("Preseason hype mapping failed. Offending rows:")
        logging.error(pd.concat([bad_name, bad_team], ignore_index=True)[["player_id", "raw_name", "full_name", "team_abbr"]])
        raise RuntimeError("Missing full_name or team_abbr for some hype players.")

    return merged[["player_id", "full_name", "team_abbr", "war_value"]]


# ---------------------------------------------------------------------------
# Bucketing and rendering
# ---------------------------------------------------------------------------

def bucket_war(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df, df
    war = df["war_value"].astype(float).sort_values()
    if len(war) >= 3:
        low_cut = float(war.quantile(1.0 / 3.0))
        high_cut = float(war.quantile(2.0 / 3.0))
    else:
        low_cut = float(war.min())
        high_cut = float(war.max())

    def assign(val: float) -> str:
        if val >= high_cut:
            return "over"
        if val <= low_cut:
            return "under"
        return "delivered"

    df = df.copy()
    df["bucket"] = df["war_value"].apply(assign)
    over = df[df["bucket"] == "over"].sort_values("war_value", ascending=False)
    delivered = df[df["bucket"] == "delivered"].sort_values("war_value", ascending=False)
    under = df[df["bucket"] == "under"].sort_values("war_value", ascending=True)
    return over, delivered, under


def render_markdown(df: pd.DataFrame, season: int) -> str:
    lines: List[str] = []
    lines.append("## Preseason hype - who delivered?")
    lines.append(f"_Based on preseason predictions and {season} season WAR among hyped players._")
    lines.append("")

    over, delivered, under = bucket_war(df)

    def emit(title: str, bucket: pd.DataFrame) -> None:
        lines.append(f"**{title}**")
        if bucket.empty:
            lines.append("- (none)")
            lines.append("")
            return
        for _, row in bucket.iterrows():
            lines.append(f"- {row['full_name']} ({row['team_abbr']}) - WAR: {row['war_value']:.2f}")
        lines.append("")

    emit("Over-delivered", over)
    emit("Delivered", delivered)
    emit("Under-delivered", under)

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def build_preseason_hype_fragment(season: int, league_id: int, preseason_html: Path | None, output_md: Path | None) -> Path:
    root = get_repo_root()
    if preseason_html is None:
        preseason_html = root / "csv" / "in" / "almanac_core" / str(season) / "leagues" / f"league_{league_id}_preseason_prediction_report.html"

    logging.info("Using preseason HTML: %s", preseason_html)
    hype = parse_preseason_players(preseason_html)
    if hype.empty:
        raise RuntimeError("No hype players parsed from preseason HTML; aborting.")

    profiles = load_dim_player_profile(root)
    stats = load_season_war(root, season, league_id)
    resolved = resolve_players(hype, profiles, stats)

    md_text = render_markdown(resolved, season)

    out_dir = root / "csv" / "out" / "eb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_md if output_md else out_dir / f"eb_preseason_hype_{season}_league{league_id}.md"
    out_path.write_text(md_text, encoding="utf-8")
    logging.info("Wrote preseason hype fragment to %s", out_path)
    return out_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    html_path = Path(args.preseason_html) if args.preseason_html else None
    output_md = Path(args.output_md) if args.output_md else None

    out_path = build_preseason_hype_fragment(args.season, args.league_id, html_path, output_md)
    print(f"[OK] Wrote preseason hype brief to {out_path}")


if __name__ == "__main__":
    main()
