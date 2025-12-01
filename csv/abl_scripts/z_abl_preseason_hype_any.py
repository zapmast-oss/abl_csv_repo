# -*- coding: utf-8 -*-
"""Build a preseason hype vs performance markdown for any season/league.

Behavior (simple and seasonal):
- Parse preseason predictions HTML (via parse_preseason_predictions) to get player_id,
  short name, and hype metadata.
- Use dim_player_profile to turn player_id into a full player name.
- Load per-season batting and pitching stats for that year from the almanac:
    csv/out/almanac/{season}/player_batting_{season}_league{league_id}.csv
    csv/out/almanac/{season}/player_pitching_{season}_league{league_id}.csv
- Use the WAR column in those files as the season performance score.
- Use the stats to get the player's team abbreviation from a team column.
- Match each hyped player to his season stats by name (last name + first initial).
- Deduplicate players, keeping the highest season WAR row.
- Bucket players into over-delivered, delivered, under-delivered by season WAR.
- Render markdown lines like:
    - Scott Reis (CIN) - WAR: 7.80
  No full team names in this section and never print "Free agent".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from eb_text_utils import normalize_eb_text
from z_abl_almanac_html_helpers import parse_preseason_predictions


def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------
# Name helpers
# -----------------------------

def _normalize_name(name: str) -> str:
    if not name:
        return ""
    return str(name).strip().lower()


def _last_name(name: str) -> str:
    parts = _normalize_name(name).split()
    return parts[-1] if parts else ""


def _first_initial(name: str) -> str:
    parts = _normalize_name(name).split()
    return parts[0][0] if parts else ""


def _attach_name_norms(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df["player_name_norm"] = df[col].apply(_normalize_name)
    df["player_last"] = df[col].apply(_last_name)
    df["player_initial"] = df[col].apply(_first_initial)
    return df


# -----------------------------
# Score: season WAR
# -----------------------------

def _compute_season_war(row: pd.Series) -> float:
    """Return a numeric performance score for a player.

    Preference:
    1) WAR (or war) in the per-season almanac stats file.
    2) If no WAR found, fall back to 0.0.
    """
    for key in ("WAR", "war"):
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                pass
    return 0.0


# -----------------------------
# Player profile lookups
# -----------------------------

def _build_profile_lookup(repo_root: Path) -> Dict[int, str]:
    """Load dim_player_profile and build an ID -> full_name lookup."""
    path = repo_root / "csv" / "out" / "star_schema" / "dim_player_profile.csv"
    if not path.exists():
        log("[WARN] dim_player_profile not found; cannot enrich hype names")
        return {}

    df = pd.read_csv(path)
    cols = list(df.columns)
    log("[INFO] dim_player_profile columns: %s" % cols)

    id_col_candidates = ["ID", "Id", "id", "Player ID", "player_id"]
    id_col = next((c for c in id_col_candidates if c in df.columns), None)
    if not id_col:
        log("[WARN] No usable ID column found in dim_player_profile")
        return {}

    first_candidates = ["First Name", "First_Name", "first_name"]
    last_candidates = ["Last Name", "Last_Name", "last_name"]

    first_col = next((c for c in first_candidates if c in df.columns), None)
    last_col = next((c for c in last_candidates if c in df.columns), None)

    full_col = "Name" if "Name" in df.columns else None

    id_lookup: Dict[int, str] = {}

    for _, row in df.iterrows():
        try:
            pid = int(row[id_col])
        except Exception:
            continue

        first = str(row[first_col]).strip() if first_col and pd.notna(row.get(first_col)) else ""
        last = str(row[last_col]).strip() if last_col and pd.notna(row.get(last_col)) else ""

        parts: List[str] = []
        if first:
            parts.append(first)
        if last:
            parts.append(last)
        full_name = " ".join(parts).strip()

        if not full_name and full_col and pd.notna(row.get(full_col)):
            full_name = str(row.get(full_col)).strip()

        if not full_name:
            continue

        id_lookup[pid] = full_name

    log("[INFO] Built profile ID lookup for %d players" % len(id_lookup))
    return id_lookup


# -----------------------------
# Stats loading (season-only, with team abbr)
# -----------------------------

def _attach_team_abbr(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize a team abbreviation column named 'team_abbr_std'."""
    df = df.copy()

    candidates = [
        "team_abbr",
        "Tm",
        "TM",
        "Team",
        "team",
        "Team Abbr",
        "Team_Name",
    ]
    team_col = next((c for c in candidates if c in df.columns), None)

    if team_col is None:
        df["team_abbr_std"] = pd.NA
        log("[WARN] No recognizable team column in season stats; team abbreviations may be missing")
        return df

    df["team_abbr_std"] = df[team_col].astype(str)
    log("[INFO] Using '%s' as team abbreviation source for season stats" % team_col)
    return df


def _load_season_stats(season: int, league_id: int, repo_root: Path) -> pd.DataFrame:
    """Load batting and pitching stats and attach name norms, season WAR, and team_abbr_std.

    Uses per-season almanac exports:
      csv/out/almanac/{season}/player_batting_{season}_league{league_id}.csv
      csv/out/almanac/{season}/player_pitching_{season}_league{league_id}.csv
    """
    base = repo_root / "csv" / "out" / "almanac" / str(season)
    bat_path = base / ("player_batting_%d_league%d.csv" % (season, league_id))
    pit_path = base / ("player_pitching_%d_league%d.csv" % (season, league_id))

    frames: List[pd.DataFrame] = []

    if bat_path.exists():
        bat = pd.read_csv(bat_path)
        if "player_name" not in bat.columns:
            log("[WARN] Batting stats missing 'player_name' column: %s" % bat_path)
        else:
            bat = _attach_name_norms(bat, "player_name")
            bat["role_type"] = "bat"
            bat["score"] = bat.apply(_compute_season_war, axis=1)
            bat = _attach_team_abbr(bat)
            frames.append(bat)
            log("[INFO] Loaded batting season stats from %s" % bat_path)
    else:
        log("[WARN] Missing batting season stats: %s" % bat_path)

    if pit_path.exists():
        pit = pd.read_csv(pit_path)
        if "player_name" not in pit.columns:
            log("[WARN] Pitching stats missing 'player_name' column: %s" % pit_path)
        else:
            pit = _attach_name_norms(pit, "player_name")
            pit["role_type"] = "pit"
            pit["score"] = pit.apply(_compute_season_war, axis=1)
            pit = _attach_team_abbr(pit)
            frames.append(pit)
            log("[INFO] Loaded pitching season stats from %s" % pit_path)
    else:
        log("[WARN] Missing pitching season stats: %s" % pit_path)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


# -----------------------------
# Matching hype entries to stats
# -----------------------------

def match_hype_entry(entry: Dict[str, Any], stats: pd.DataFrame) -> Optional[pd.Series]:
    """Match a preseason hype entry to a row in the combined stats dataframe."""
    if stats.empty:
        return None

    name = entry.get("player_name", "") or ""
    last = _last_name(name)
    initial = _first_initial(name)

    if not last:
        return None

    pool = stats[stats["player_last"] == last]
    if pool.empty:
        return None

    if initial:
        narrowed = pool[pool["player_initial"] == initial]
        if not narrowed.empty:
            pool = narrowed

    best_idx = pool["score"].idxmax()
    if pd.isna(best_idx):
        return None
    return pool.loc[best_idx]


# -----------------------------
# Bucketing and markdown
# -----------------------------

def bucketize(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split matched hype records into three buckets by season WAR terciles."""
    if not records:
        return {"over-delivered": [], "delivered": [], "under-delivered": []}

    df = pd.DataFrame(records)
    if "score" not in df.columns or df["score"].isna().all():
        return {"over-delivered": records, "delivered": [], "under-delivered": []}

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    n = len(df)
    top_cut = max(1, n // 3)
    mid_cut = max(1, 2 * n // 3)

    over = df.iloc[:top_cut].to_dict(orient="records")
    mid = df.iloc[top_cut:mid_cut].to_dict(orient="records")
    under = df.iloc[mid_cut:].to_dict(orient="records")

    return {"over-delivered": over, "delivered": mid, "under-delivered": under}


def build_md(buckets: Dict[str, List[Dict[str, Any]]], season: int) -> str:
    """Render the preseason hype vs WAR performance brief as markdown.

    Line format:
      - Player Name (ABBR) - WAR: X.XX

    Rules:
      - If team_abbr is missing, omit the (ABBR) part.
      - Never print "Free agent" anywhere.
    """
    lines: List[str] = []
    lines.append("## Preseason hype - who delivered?")
    lines.append("_Based on preseason predictions and %d season WAR among hyped players._" % season)
    lines.append("")

    order = [
        ("over-delivered", "Over-delivered"),
        ("delivered", "Delivered"),
        ("under-delivered", "Under-delivered"),
    ]

    for key, title in order:
        lines.append("**" + title + "**")
        bucket = buckets.get(key, [])
        if not bucket:
            lines.append("- None")
        else:
            for rec in bucket:
                player_name = rec.get("player_name", "Unknown")
                team_abbr = rec.get("team_abbr")
                score = rec.get("score")
                score_str = "n/a"
                if isinstance(score, (int, float)):
                    score_str = "%.2f" % float(score)

                if team_abbr and isinstance(team_abbr, str) and team_abbr.strip().lower() != "free agent":
                    line = "- %s (%s) - WAR: %s" % (player_name, team_abbr, score_str)
                else:
                    line = "- %s - WAR: %s" % (player_name, score_str)

                lines.append(line)
        lines.append("")

    return normalize_eb_text("\n".join(lines).strip() + "\n")


# -----------------------------
# Main orchestration
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Build preseason hype vs performance brief.")
    parser.add_argument("--season", type=int, required=True, help="Season year, for example 1980")
    parser.add_argument(
        "--league-id",
        type=int,
        default=200,
        help="League ID (default 200 for ABL)",
    )
    parser.add_argument(
        "--preseason-html",
        type=Path,
        required=True,
        help="Path to league_{league_id}_preseason_prediction_report.html",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    season = args.season
    league_id = args.league_id

    # Parse preseason predictions HTML
    hype_rows = parse_preseason_predictions(args.preseason_html)
    if not hype_rows:
        log("[WARN] No preseason predictions parsed from %s" % args.preseason_html)
        return 0

    log("[INFO] Parsed %d preseason hype rows from HTML" % len(hype_rows))

    # Build player_id -> full_name lookup
    id_lookup = _build_profile_lookup(repo_root)

    # Enrich hype entries with full names via player_id
    if id_lookup:
        total_with_pid = 0
        enriched = 0
        new_rows: List[Dict[str, Any]] = []

        for entry in hype_rows:
            e = dict(entry)
            pid_raw = e.get("player_id")
            full_name_to_use: Optional[str] = None

            pid_int: Optional[int]
            try:
                pid_int = int(pid_raw) if pid_raw is not None else None
            except Exception:
                pid_int = None

            if pid_int is not None:
                total_with_pid += 1
                full_name_to_use = id_lookup.get(pid_int)

            if full_name_to_use:
                e["player_name"] = full_name_to_use
                enriched += 1

            new_rows.append(e)

        hype_rows = new_rows
        log("[INFO] Hype rows with player_id: %d" % total_with_pid)
        log("[INFO] Hype rows enriched via player_id: %d" % enriched)
    else:
        log("[WARN] No ID lookup available; hype names will remain as in HTML")

    # Load season stats (WAR and team abbreviations)
    stats = _load_season_stats(season, league_id, repo_root)
    if stats.empty:
        log("[WARN] No season stats available; preseason hype brief will be empty.")
        return 0

    # Match hype entries to stats and build records
    matched_records: List[Dict[str, Any]] = []
    for entry in hype_rows:
        matched = match_hype_entry(entry, stats)
        if matched is None:
            continue

        team_abbr = matched.get("team_abbr_std")

        rec: Dict[str, Any] = {
            "player_id": entry.get("player_id"),
            "player_name": entry.get("player_name") or matched.get("player_name") or "Unknown",
            "team_abbr": team_abbr if pd.notna(team_abbr) else None,
            "role_type": matched.get("role_type"),
            "hype_role": entry.get("hype_role"),
            "source": entry.get("source"),
            "rank": entry.get("rank"),
            "score": matched.get("score"),
        }
        matched_records.append(rec)

    log("[INFO] Matched %d hype entries to season stats before dedupe" % len(matched_records))

    if not matched_records:
        log("[WARN] No preseason hype entries matched to season stats; nothing to write.")
        return 0

    # Deduplicate by player_id when available, else by normalized name + team_abbr
    df_match = pd.DataFrame(matched_records)
    df_match = df_match.sort_values("score", ascending=False).reset_index(drop=True)

    if "player_id" not in df_match.columns:
        df_match["player_id"] = None

    def _dedupe_key(row: pd.Series) -> str:
        if pd.notna(row.get("player_id")):
            return "id_%s" % str(row["player_id"])
        name_key = _normalize_name(str(row.get("player_name", "")))
        team_key = str(row.get("team_abbr", ""))
        return "name_%s|%s" % (name_key, team_key)

    df_match["dedupe_key"] = df_match.apply(_dedupe_key, axis=1)
    df_unique = df_match.drop_duplicates(subset=["dedupe_key"], keep="first").drop(columns=["dedupe_key"])
    matched_records = df_unique.to_dict(orient="records")
    log("[INFO] Unique matched hype players after dedupe: %d" % len(matched_records))

    buckets = bucketize(matched_records)
    md_text = build_md(buckets, season)

    out_dir = repo_root / "csv" / "out" / "eb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("eb_preseason_hype_%d_league%d.md" % (season, league_id))
    out_path.write_text(md_text, encoding="utf-8")
    log("[OK] Wrote preseason hype brief to %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
