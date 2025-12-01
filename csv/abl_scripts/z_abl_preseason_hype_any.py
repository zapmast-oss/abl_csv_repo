#!/usr/bin/env python
"""Build a preseason hype vs performance markdown for any season/league.

This version:
- Uses player_id parsed from the preseason HTML (via parse_preseason_predictions)
  and dim_player_profile to recover full player names.
- Falls back to short-name matching (Name.1) when player_id lookup fails, so we
  still get full names for any player present in the profile table.
- Uses dim_team_park to attach real ABL teams to matched stats where possible.
- Uses WAR as the primary performance metric and labels it clearly.
- Deduplicates players so we don't print the same player/team combo multiple times.
- Formats lines as: "Player Name (ABBR) — Team Name — WAR: X.XX" with no label like "Free agent".
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


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Score: WAR-first
# ---------------------------------------------------------------------------

def _compute_score(row: pd.Series) -> float:
    """Return a numeric performance score for a player.

    Primary intent: use WAR where available. Fall back to OPS / -ERA / SO if needed.
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
            # Lower ERA is better => invert sign
            return -float(row["ERA"])
        except Exception:
            pass
    if "SO" in row and pd.notna(row["SO"]):
        try:
            return float(row["SO"])
        except Exception:
            pass
    return 0.0


# ---------------------------------------------------------------------------
# Team helpers (dim_team_park mapping)
# ---------------------------------------------------------------------------

def _norm_team_key(value: str) -> str:
    """Normalize a team string for fuzzy matching against dim_team_park."""
    import re

    if value is None:
        return ""
    text = str(value)
    # drop anything in parentheses
    text = re.sub(r"\(.*?\)", "", text)
    # strip to alphanumerics
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return text


def _build_team_lookup(dim_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build lookup dicts from dim_team_park:

    Returns:
      abbr_map:  normalized key -> team_abbr
      label_map: normalized key -> pretty label ("City Team Name")

    Keys include combinations of Team Name / City / Abbr so that raw stats
    headers like "Cincinnati", "Cincinnati Cougars", "CIN" all map cleanly.
    """
    if not dim_path.exists():
        log(f"[WARN] dim_team_park not found at {dim_path}; team mapping will be limited")
        return {}, {}

    dim = pd.read_csv(dim_path)
    required = {"Team Name", "Abbr", "City"}
    if not required.issubset(set(dim.columns)):
        log(f"[WARN] dim_team_park missing expected columns {sorted(required)}; got {list(dim.columns)}")
        return {}, {}

    abbr_map: Dict[str, str] = {}
    label_map: Dict[str, str] = {}

    for _, row in dim.iterrows():
        team_name = str(row["Team Name"])
        abbr = str(row["Abbr"])
        city_full = "" if pd.isna(row["City"]) else str(row["City"])
        city_short = city_full.split("(")[0].strip() if city_full else ""
        label = f"{city_short} {team_name}".strip() if city_short else team_name

        candidates = {
            team_name,
            abbr,
            city_full,
            city_short,
            f"{city_short} {team_name}".strip(),
            f"{team_name} ({abbr})",
        }

        for cand in candidates:
            if not cand:
                continue
            key = _norm_team_key(cand)
            if not key:
                continue
            # First come, first served to avoid noisy overwrites
            abbr_map.setdefault(key, abbr)
            label_map.setdefault(key, label)

    log(f"[INFO] Built team lookup for {len(abbr_map)} keys from dim_team_park")
    return abbr_map, label_map


def _attach_team_metadata_for_stats(df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """Attach canonical team_abbr / team_label columns to the stats dataframe.

    Uses dim_team_park and any usable team-like column in the stats CSVs.
    If mapping fails, leaves fields null so downstream can decide how to display.
    """
    dim_path = repo_root / "csv" / "out" / "star_schema" / "dim_team_park.csv"
    abbr_map, label_map = _build_team_lookup(dim_path)
    if not abbr_map:
        return df

    df = df.copy()
    if "team_abbr" not in df.columns:
        df["team_abbr"] = pd.NA
    if "team_label" not in df.columns:
        df["team_label"] = pd.NA

    # Candidates for the raw team column in the stats CSVs
    team_candidates = ["team_name", "Team", "team", "Tm", "Team Name", "TM", "TM.1"]
    team_raw_col = next((c for c in team_candidates if c in df.columns), None)

    if not team_raw_col:
        # If we already have team_abbr in the stats, at least try to map label from that.
        if "team_abbr" in df.columns:
            for idx, abbr in df["team_abbr"].items():
                if pd.isna(abbr):
                    continue
                key = _norm_team_key(str(abbr))
                label = label_map.get(key)
                if label:
                    df.at[idx, "team_label"] = label
        return df

    # Fill in team_abbr / team_label where missing
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
    """Return a dict with printable team_name / team_abbr fields for markdown.

    Rules:
    - Prefer team_label/team_abbr attached from dim_team_park.
    - Fallback to canonicalized team_name from stats.
    - Never propagate a label that is literally "free agent" (case-insensitive).
    """
    team_name: Optional[str] = None
    team_abbr: Optional[str] = None

    # Prefer mapped label/abbr from dim_team_park
    label = row.get("team_label")
    abbr = row.get("team_abbr")

    if pd.notna(label):
        team_name = str(label)
    if pd.notna(abbr):
        team_abbr = str(abbr)

    # Fallback to canonicalized raw team_name
    if not team_name:
        name_raw = row.get("team_name")
        if pd.notna(name_raw):
            cand = canonicalize_team_city(str(name_raw))
            # Do not treat any canonicalization result that looks like "free agent" as a real team
            if cand and cand.strip().lower() != "free agent":
                team_name = cand

    # Final guard: if team_name somehow equals "free agent", drop it
    if team_name and team_name.strip().lower() == "free agent":
        team_name = None

    return {
        "team_name": team_name,
        "team_abbr": team_abbr,
    }


# ---------------------------------------------------------------------------
# Player profile lookups (ID + short-name)
# ---------------------------------------------------------------------------

def _build_profile_lookups(repo_root: Path) -> Tuple[
    Dict[int, Dict[str, str]],
    Dict[str, Dict[str, str]],
]:
    """Load dim_player_profile and build:

      id_lookup:     player_id (int) -> {full_name, short_name}
      short_lookup:  normalized short/full name -> {full_name, short_name}
    """
    path = repo_root / "csv" / "out" / "star_schema" / "dim_player_profile.csv"
    if not path.exists():
        log(f"[WARN] dim_player_profile not found at {path}; cannot enrich hype names")
        return {}, {}

    df = pd.read_csv(path)
    cols = list(df.columns)
    log(f"[INFO] dim_player_profile columns: {cols}")

    # Determine ID column
    id_candidates = ["ID", "Id", "id", "Player ID", "player_id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if not id_col:
        log("[WARN] No usable ID column found in dim_player_profile; cannot build profile lookup")
        return {}, {}

    # Determine name columns (may or may not exist)
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

        full_name_parts: List[str] = []
        if first:
            full_name_parts.append(first)
        if last:
            full_name_parts.append(last)
        full_name = " ".join(full_name_parts).strip()

        # Fallbacks for full_name if first/last aren't usable
        if not full_name and full_col and pd.notna(row.get(full_col)):
            full_name = str(row.get(full_col)).strip()
        if not full_name and short_col and pd.notna(row.get(short_col)):
            full_name = str(row.get(short_col)).strip()

        # Short display name
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

        # ID-based lookup
        id_lookup[pid] = rec

        # Short-name & full-name based lookup (normalized)
        for cand in (short_name, full_name):
            if not cand:
                continue
            key = _normalize_name(cand)
            if key and key not in short_lookup:
                short_lookup[key] = rec

    log(f"[INFO] Built profile ID lookup for {len(id_lookup)} players")
    log(f"[INFO] Built profile short-name lookup for {len(short_lookup)} keys")
    return id_lookup, short_lookup


# ---------------------------------------------------------------------------
# Stats loading
# ---------------------------------------------------------------------------

def _load_stats(season: int, league_id: int, repo_root: Path) -> pd.DataFrame:
    """Load batting and pitching stats for one season/league and attach name norms, score, team."""
    base = repo_root / "csv" / "out" / "almanac" / str(season)
    bat_path = base / f"player_batting_{season}_league{league_id}.csv"
    pit_path = base / f"player_pitching_{season}_league{league_id}.csv"

    frames: List[pd.DataFrame] = []

    if bat_path.exists():
        bat = pd.read_csv(bat_path)
        if "player_name" not in bat.columns:
            log(f"[WARN] Batting stats missing 'player_name' column: {bat_path}")
        else:
            bat = _attach_name_norms(bat, "player_name")
            bat["role_type"] = "bat"
            bat["score"] = bat.apply(_compute_score, axis=1)
            frames.append(bat)
    else:
        log(f"[WARN] Missing batting stats: {bat_path}")

    if pit_path.exists():
        pit = pd.read_csv(pit_path)
        if "player_name" not in pit.columns:
            log(f"[WARN] Pitching stats missing 'player_name' column: {pit_path}")
        else:
            pit = _attach_name_norms(pit, "player_name")
            pit["role_type"] = "pit"
            pit["score"] = pit.apply(_compute_score, axis=1)
            frames.append(pit)
    else:
        log(f"[WARN] Missing pitching stats: {pit_path}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _attach_team_metadata_for_stats(combined, repo_root)
    return combined


# ---------------------------------------------------------------------------
# Matching hype entries to stats
# ---------------------------------------------------------------------------

def match_hype_entry(entry: Dict[str, Any], stats: pd.DataFrame) -> Optional[pd.Series]:
    """Match a preseason hype entry to a row in the combined stats dataframe."""
    if stats.empty:
        return None

    name = entry.get("player_name", "") or ""
    last = _last_name(name)
    initial = _first_initial(name)

    if not last:
        return None

    # Filter by last name first
    pool = stats[stats["player_last"] == last]
    if pool.empty:
        return None

    # If we have a first initial, narrow further
    if initial:
        narrowed = pool[pool["player_initial"] == initial]
        if not narrowed.empty:
            pool = narrowed

    # Pick the row with the highest score
    best_idx = pool["score"].idxmax()
    if pd.isna(best_idx):
        return None
    return pool.loc[best_idx]


# ---------------------------------------------------------------------------
# Bucketing + markdown
# ---------------------------------------------------------------------------

def bucketize(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split matched hype records into over/neutral/under buckets by WAR/score terciles."""
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
    """Render the preseason hype vs WAR performance brief as Markdown.

    Output line format:
      - Player Name (ABBR) — Team Name — WAR: X.XX

    Rules:
      - If team_name is missing or looks like "free agent", we leave it blank and NEVER print that label.
      - If team_abbr is missing, we just omit the (ABBR) part.
    """
    lines: List[str] = []
    lines.append("## Preseason hype — who delivered?")
    lines.append(f"_Based on preseason predictions and {season} WAR among hyped players._")
    lines.append("")

    order = [
        ("over-delivered", "Over-delivered"),
        ("delivered", "Delivered"),
        ("under-delivered", "Under-delivered"),
    ]

    for key, title in order:
        lines.append(f"**{title}**")
        bucket = buckets.get(key, [])
        if not bucket:
            lines.append("- None")
        else:
            for rec in bucket:
                player_name = rec.get("player_name", "Unknown")
                team_name = rec.get("team_name")
                team_abbr = rec.get("team_abbr")
                score = rec.get("score")
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"

                # Guard against any residual "free agent" string sneaking in
                if isinstance(team_name, str) and team_name.strip().lower() == "free agent":
                    team_name = None

                # Build the display line: Player (ABBR) — Team Name — WAR
                if team_abbr and team_name:
                    line = f"- {player_name} ({team_abbr}) — {team_name} — WAR: {score_str}"
                elif team_abbr and not team_name:
                    line = f"- {player_name} ({team_abbr}) — WAR: {score_str}"
                elif team_name and not team_abbr:
                    line = f"- {player_name} — {team_name} — WAR: {score_str}"
                else:
                    line = f"- {player_name} — WAR: {score_str}"

                lines.append(line)
        lines.append("")

    return normalize_eb_text("\n".join(lines).strip() + "\n")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Build preseason hype vs performance brief.")
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g. 1980")
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

    # Parse preseason predictions HTML (includes player_id and short names)
    hype_rows = parse_preseason_predictions(args.preseason_html)
    if not hype_rows:
        log(f"[WARN] No preseason predictions parsed from {args.preseason_html}")
        return 0

    log(f"[INFO] Parsed {len(hype_rows)} preseason hype rows from HTML")

    # Build player profile lookups (ID + short/full names)
    id_lookup, short_lookup = _build_profile_lookups(repo_root)

    # Enrich hype entries with full names via player_id first, then short-name match
    if id_lookup or short_lookup:
        total_with_pid = 0
        enriched_by_id = 0
        enriched_by_short = 0
        enriched: List[Dict[str, Any]] = []

        for entry in hype_rows:
            e = dict(entry)
            pid_raw = e.get("player_id")
            full_name_to_use: Optional[str] = None

            # 1) Try player_id-based lookup
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

            # 2) If that failed, try short-name-based lookup (Name.1 style)
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
        log(f"[INFO] Hype rows with player_id: {total_with_pid}")
        log(f"[INFO] Hype rows enriched via player_id: {enriched_by_id}")
        log(f"[INFO] Hype rows enriched via short-name fallback: {enriched_by_short}")
    else:
        log("[WARN] No profile lookup available; hype names will remain as in HTML")

    # Load season stats (batting + pitching)
    stats = _load_stats(season, league_id, repo_root)
    if stats.empty:
        log("[WARN] No stats available; preseason hype brief will be empty.")
        return 0

    # Match hype entries to stats and build records
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

    log(f"[INFO] Matched {len(matched_records)} hype entries to season stats before dedupe")

    if not matched_records:
        log("[WARN] No preseason hype entries matched to stats; nothing to write.")
        return 0

    # Deduplicate by (player_id, team_abbr, player_name), keeping highest WAR
    df_match = pd.DataFrame(matched_records)
    df_match = df_match.sort_values("score", ascending=False).reset_index(drop=True)

    # Use a synthetic key when player_id is missing
    if "player_id" not in df_match.columns:
        df_match["player_id"] = None

    df_match["dedupe_key"] = df_match.apply(
        lambda r: (
            r["player_id"]
            if pd.notna(r["player_id"])
            else f"{_normalize_name(str(r.get('player_name', '')))}|{str(r.get('team_abbr', ''))}"
        ),
        axis=1,
    )

    df_unique = df_match.drop_duplicates(subset=["dedupe_key"], keep="first").drop(columns=["dedupe_key"])
    matched_records = df_unique.to_dict(orient="records")
    log(f"[INFO] Unique matched hype players after dedupe: {len(matched_records)}")

    buckets = bucketize(matched_records)
    md_text = build_md(buckets, season)

    out_dir = repo_root / "csv" / "out" / "eb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eb_preseason_hype_{season}_league{league_id}.md"
    out_path.write_text(md_text, encoding="utf-8")
    log(f"[OK] Wrote preseason hype brief to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
