# -*- coding: utf-8 -*-
"""Build a preseason hype vs performance markdown for any season/league.

Behavior:
- Parse preseason predictions HTML (via parse_preseason_predictions) to get player_id,
  short name, and any other hype metadata.
- Use dim_player_profile to turn player_id into a full player name. Fall back to
  matching the short name if needed.
- Load batting and pitching stats for the season and compute a WAR-based score.
- Use dim_team_park to attach real ABL teams to the stat rows.
- Match each hyped player to his season stats by name (last name + first initial).
- Deduplicate players, keeping the highest WAR match.
- Render markdown lines like:
    - Scott Reis (CIN) - Cincinnati Cougars - WAR: 78.40
  Never print "Free agent" as a team.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from eb_text_utils import canonicalize_team_city, normalize_eb_text
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
# Score: WAR first
# -----------------------------

def _compute_score(row: pd.Series) -> float:
    """Return a numeric performance score for a player.

    Preference:
    1) WAR (or war)
    2) OPS (higher is better)
    3) -ERA (lower ERA is better)
    4) SO (strikeouts)
    """
    for key in ("WAR", "war"):
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                pass
    if "OPS" in row and pd.notna(row["OPS"]):
        try:
            return float(row["OPS"])
        except Exception:
            pass
    if "ERA" in row and pd.notna(row["ERA"]):
        try:
            return -float(row["ERA"])
        except Exception:
            pass
    if "SO" in row and pd.notna(row["SO"]):
        try:
            return float(row["SO"])
        except Exception:
            pass
    return 0.0


# -----------------------------
# Team helpers from dim_team_park
# -----------------------------

def _norm_team_key(value: str) -> str:
    """Normalize a team string for fuzzy matching against dim_team_park."""
    import re

    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return text


def _build_team_lookup(dim_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build lookup dicts from dim_team_park.

    Returns:
      abbr_map:  normalized key -> team_abbr
      label_map: normalized key -> pretty label ("City Team Name")
    """
    if not dim_path.exists():
        log("[WARN] dim_team_park not found; team mapping will be limited")
        return {}, {}

    dim = pd.read_csv(dim_path)
    required = {"Team Name", "Abbr", "City"}
    if not required.issubset(set(dim.columns)):
        log("[WARN] dim_team_park missing expected columns")
        return {}, {}

    abbr_map: Dict[str, str] = {}
    label_map: Dict[str, str] = {}

    for _, row in dim.iterrows():
        team_name = str(row["Team Name"])
        abbr = str(row["Abbr"])
        city_full = "" if pd.isna(row["City"]) else str(row["City"])
        city_short = city_full.split("(")[0].strip() if city_full else ""
        label = (city_short + " " + team_name).strip() if city_short else team_name

        candidates = {
            team_name,
            abbr,
            city_full,
            city_short,
            (city_short + " " + team_name).strip(),
            team_name + " (" + abbr + ")",
        }

        for cand in candidates:
            if not cand:
                continue
            key = _norm_team_key(cand)
            if not key:
                continue
            if key not in abbr_map:
                abbr_map[key] = abbr
            if key not in label_map:
                label_map[key] = label

    log("[INFO] Built team lookup for %d keys from dim_team_park" % len(abbr_map))
    return abbr_map, label_map


def _attach_team_metadata_for_stats(df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """Attach canonical team_abbr / team_label columns to the stats dataframe."""
    dim_path = repo_root / "csv" / "out" / "star_schema" / "dim_team_park.csv"
    abbr_map, label_map = _build_team_lookup(dim_path)
    if not abbr_map:
        return df

    df = df.copy()
    if "team_abbr" not in df.columns:
        df["team_abbr"] = pd.NA
    if "team_label" not in df.columns:
        df["team_label"] = pd.NA

    team_candidates = ["team_name", "Team", "team", "Tm", "Team Name", "TM", "TM.1"]
    team_raw_col = next((c for c in team_candidates if c in df.columns), None)

    if not team_raw_col:
        if "team_abbr" in df.columns:
            for idx, abbr in df["team_abbr"].items():
                if pd.isna(abbr):
                    continue
                key = _norm_team_key(str(abbr))
                label = label_map.get(key)
                if label:
                    df.at[idx, "team_label"] = label
        return df

    for idx, raw in df[team_raw_col].items():
        if pd.isna(raw):
            continue
        key = _norm_team_key(str(raw))
        if not key:
            continue
        abbr = abbr_map.get(key)
        label = label_map.get(key)
        if abbr and pd.isna(df.at[idx, "team_abbr"]):
            df.at[idx, "team_abbr"] = abbr
        if label and pd.isna(df.at[idx, "team_label"]):
            df.at[idx, "team_label"] = label

    return df


def _pick_team_fields(row: pd.Series) -> Dict[str, Optional[str]]:
    """Return printable team_name / team_abbr for markdown.

    Rules:
    - Prefer team_label and team_abbr from dim_team_park.
    - Fallback to canonicalized team_name from stats.
    - Never treat any name that looks like "free agent" as a team.
    """
    team_name: Optional[str] = None
    team_abbr: Optional[str] = None

    label = row.get("team_label")
    abbr = row.get("team_abbr")

    if pd.notna(label):
        team_name = str(label)
    if pd.notna(abbr):
        team_abbr = str(abbr)

    if not team_name:
        name_raw = row.get("team_name")
        if pd.notna(name_raw):
            cand = canonicalize_team_city(str(name_raw))
            if cand and cand.strip().lower() != "free agent":
                team_name = cand

    if team_name and team_name.strip().lower() == "free agent":
        team_name = None

    return {"team_name": team_name, "team_abbr": team_abbr}


# -----------------------------
# Player profile lookups
# -----------------------------

def _build_profile_lookups(repo_root: Path) -> Tuple[
    Dict[int, Dict[str, str]],
    Dict[str, Dict[str, str]],
]:
    """Load dim_player_profile and build two lookups:

      id_lookup:    player_id -> {full_name, short_name}
      short_lookup: normalized short or full name -> {full_name, short_name}
    """
    path = repo_root / "csv" / "out" / "star_schema" / "dim_player_profile.csv"
    if not path.exists():
        log("[WARN] dim_player_profile not found; cannot enrich hype names")
        return {}, {}

    df = pd.read_csv(path)
    cols = list(df.columns)
    log("[INFO] dim_player_profile columns: %s" % cols)

    id_candidates = ["ID", "Id", "id", "Player ID", "player_id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if not id_col:
        log("[WARN] No usable ID column found in dim_player_profile")
        return {}, {}

    first_candidates = ["First Name", "First_Name", "first_name"]
    last_candidates = ["Last Name", "Last_Name", "last_name"]

    first_col = next((c for c in first_candidates if c in df.columns), None)
    last_col = next((c for c in last_candidates if c in df.columns), None)

    full_col = "Name" if "Name" in df.columns else None
    short_col = "Name.1" if "Name.1" in df.columns else None

    id_lookup: Dict[int, Dict[str, str]] = {}
    short_lookup: Dict[str, Dict[str, str]] = {}

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
        if not full_name and short_col and pd.notna(row.get(short_col)):
            full_name = str(row.get(short_col)).strip()

        short_name = ""
        if short_col and pd.notna(row.get(short_col)):
            short_name = str(row.get(short_col)).strip()
        if not short_name:
            short_name = full_name

        if not full_name and not short_name:
            continue

        rec = {
            "full_name": full_name if full_name else short_name,
            "short_name": short_name if short_name else full_name,
        }

        id_lookup[pid] = rec

        for cand in (short_name, full_name):
            if not cand:
                continue
            key = _normalize_name(cand)
            if key and key not in short_lookup:
                short_lookup[key] = rec

    log("[INFO] Built profile ID lookup for %d players" % len(id_lookup))
    log("[INFO] Built profile short-name lookup for %d keys" % len(short_lookup))
    return id_lookup, short_lookup


# -----------------------------
# Stats loading
# -----------------------------

def _load_stats(season: int, league_id: int, repo_root: Path) -> pd.DataFrame:
    """Load batting and pitching stats and attach name norms, WAR score, and team metadata."""
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
            bat["score"] = bat.apply(_compute_score, axis=1)
            frames.append(bat)
    else:
        log("[WARN] Missing batting stats: %s" % bat_path)

    if pit_path.exists():
        pit = pd.read_csv(pit_path)
        if "player_name" not in pit.columns:
            log("[WARN] Pitching stats missing 'player_name' column: %s" % pit_path)
        else:
            pit = _attach_name_norms(pit, "player_name")
            pit["role_type"] = "pit"
            pit["score"] = pit.apply(_compute_score, axis=1)
            frames.append(pit)
    else:
        log("[WARN] Missing pitching stats: %s" % pit_path)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _attach_team_metadata_for_stats(combined, repo_root)
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
    """Split matched hype records into three buckets by score terciles."""
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
      - Player Name (ABBR) - Team Name - WAR: X.XX

    Rules:
      - Never print "Free agent" as a team name.
      - If team_name is missing or looked like "free agent", omit it.
      - If team_abbr is missing, omit the (ABBR) part.
    """
    lines: List[str] = []
    lines.append("## Preseason hype - who delivered?")
    lines.append("_Based on preseason predictions and %d WAR among hyped players._" % season)
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
                team_name = rec.get("team_name")
                team_abbr = rec.get("team_abbr")
                score = rec.get("score")
                score_str = "n/a"
                if isinstance(score, (int, float)):
                    score_str = "%.2f" % float(score)

                if isinstance(team_name, str) and team_name.strip().lower() == "free agent":
                    team_name = None

                if team_abbr and team_name:
                    line = "- %s (%s) - %s - WAR: %s" % (player_name, team_abbr, team_name, score_str)
                elif team_abbr and not team_name:
                    line = "- %s (%s) - WAR: %s" % (player_name, team_abbr, score_str)
                elif team_name and not team_abbr:
                    line = "- %s - %s - WAR: %s" % (player_name, team_name, score_str)
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

    hype_rows = parse_preseason_predictions(args.preseason_html)
    if not hype_rows:
        log("[WARN] No preseason predictions parsed from %s" % args.preseason_html)
        return 0

    log("[INFO] Parsed %d preseason hype rows from HTML" % len(hype_rows))

    id_lookup, short_lookup = _build_profile_lookups(repo_root)

    if id_lookup or short_lookup:
        total_with_pid = 0
        enriched_by_id = 0
        enriched_by_short = 0
        enriched: List[Dict[str, Any]] = []

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
                profile = id_lookup.get(pid_int)
                if profile and profile.get("full_name"):
                    full_name_to_use = profile["full_name"]
                    enriched_by_id += 1

            if full_name_to_use is None:
                raw_short = e.get("player_name") or ""
                key = _normalize_name(raw_short)
                if key and key in short_lookup:
                    profile = short_lookup[key]
                    if profile.get("full_name"):
                        full_name_to_use = profile["full_name"]
                        enriched_by_short += 1

            if full_name_to_use:
                e["player_name"] = full_name_to_use

            enriched.append(e)

        hype_rows = enriched
        log("[INFO] Hype rows with player_id: %d" % total_with_pid)
        log("[INFO] Hype rows enriched via player_id: %d" % enriched_by_id)
        log("[INFO] Hype rows enriched via short-name fallback: %d" % enriched_by_short)
    else:
        log("[WARN] No profile lookup available; hype names will remain as in HTML")

    stats = _load_stats(season, league_id, repo_root)
    if stats.empty:
        log("[WARN] No stats available; preseason hype brief will be empty.")
        return 0

    matched_records: List[Dict[str, Any]] = []
    for entry in hype_rows:
        matched = match_hype_entry(entry, stats)
        if matched is None:
            continue

        team_fields = _pick_team_fields(matched)

        rec: Dict[str, Any] = {
            "player_id": entry.get("player_id"),
            "player_name": entry.get("player_name") or matched.get("player_name") or "Unknown",
            "team_name": team_fields.get("team_name"),
            "team_abbr": team_fields.get("team_abbr"),
            "role_type": matched.get("role_type"),
            "hype_role": entry.get("hype_role"),
            "source": entry.get("source"),
            "rank": entry.get("rank"),
            "score": matched.get("score"),
        }
        matched_records.append(rec)

    log("[INFO] Matched %d hype entries to season stats before dedupe" % len(matched_records))

    if not matched_records:
        log("[WARN] No preseason hype entries matched to stats; nothing to write.")
        return 0

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
