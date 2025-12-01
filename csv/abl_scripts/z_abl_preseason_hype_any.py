#!/usr/bin/env python
"""Build a preseason hype vs performance markdown for any season/league.

This version:
- Uses player_id parsed from the preseason HTML (via parse_preseason_predictions)
  and dim_player_profile to recover full player names.
- Uses WAR as the primary performance metric when available and labels it clearly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from eb_text_utils import canonicalize_team_city, format_team_label, normalize_eb_text
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


def _load_stats(season: int, league_id: int, repo_root: Path) -> pd.DataFrame:
    """Load batting and pitching stats for one season/league and attach name norms + score."""
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
    return combined


def _pick_team_fields(row: pd.Series) -> Dict[str, Optional[str]]:
    """Return a dict with printable team_name / team_abbr fields."""
    name = row.get("team_name")
    abbr = row.get("team_abbr")

    if pd.notna(name):
        name = canonicalize_team_city(str(name))
    if pd.isna(abbr):
        abbr = None

    return {
        "team_name": name if name else None,
        "team_abbr": abbr if abbr else None,
    }


# ---------------------------------------------------------------------------
# Player profile lookup (robust column handling)
# ---------------------------------------------------------------------------

def _build_profile_lookup(repo_root: Path) -> Dict[int, Dict[str, str]]:
    """Load dim_player_profile and build a lookup: player_id -> names.

    We try to be tolerant of different column names:

      ID column candidates:
        - "ID", "Id", "id", "Player ID", "player_id"

      Name column candidates:
        - First Name: "First Name", "First_Name", "first_name"
        - Last Name:  "Last Name", "Last_Name", "last_name"
        - Full/short display names: "Name", "Name.1"

    The resulting mapping for each player_id is:
      {
        "full_name": "Miguel Morales",
        "short_name": "M. Morales"
      }
    """
    path = repo_root / "csv" / "out" / "star_schema" / "dim_player_profile.csv"
    if not path.exists():
        log(f"[WARN] dim_player_profile not found at {path}; cannot enrich hype names via player_id")
        return {}

    df = pd.read_csv(path)
    cols = list(df.columns)
    log(f"[INFO] dim_player_profile columns: {cols}")

    # Determine ID column
    id_candidates = ["ID", "Id", "id", "Player ID", "player_id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if not id_col:
        log("[WARN] No usable ID column found in dim_player_profile; cannot build profile lookup")
        return {}

    # Determine name columns (may or may not exist)
    first_candidates = ["First Name", "First_Name", "first_name"]
    last_candidates = ["Last Name", "Last_Name", "last_name"]

    first_col = next((c for c in first_candidates if c in df.columns), None)
    last_col = next((c for c in last_candidates if c in df.columns), None)

    full_col = "Name" if "Name" in df.columns else None
    short_col = "Name.1" if "Name.1" in df.columns else None

    lookup: Dict[int, Dict[str, str]] = {}

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

        lookup[pid] = {
            "full_name": full_name if full_name else short_name,
            "short_name": short_name if short_name else full_name,
        }

    log(f"[INFO] Built profile lookup for {len(lookup)} players from dim_player_profile")
    return lookup


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
    """Render the preseason hype vs WAR performance brief as Markdown."""
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
                team_label = format_team_label(rec.get("team_name"), rec.get("team_abbr"))
                score = rec.get("score")
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
                player_name = rec.get("player_name", "Unknown")
                lines.append(f"- {player_name} — {team_label} — WAR: {score_str}")
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

    # Parse preseason predictions HTML (includes player_id)
    hype_rows = parse_preseason_predictions(args.preseason_html)
    if not hype_rows:
        log(f"[WARN] No preseason predictions parsed from {args.preseason_html}")
        return 0

    log(f"[INFO] Parsed {len(hype_rows)} preseason hype rows from HTML")

    # Enrich hype entries with full names via player_id + dim_player_profile
    profile_lookup = _build_profile_lookup(repo_root)
    if profile_lookup:
        total_with_pid = 0
        enriched_count = 0
        enriched: List[Dict[str, Any]] = []
        for entry in hype_rows:
            e = dict(entry)
            pid_raw = e.get("player_id")
            try:
                pid_int = int(pid_raw) if pid_raw is not None else None
            except Exception:
                pid_int = None

            if pid_int is not None:
                total_with_pid += 1
                profile = profile_lookup.get(pid_int)
                if profile and profile.get("full_name"):
                    e["player_name"] = profile["full_name"]
                    enriched_count += 1
            enriched.append(e)
        hype_rows = enriched
        log(f"[INFO] Hype rows with player_id: {total_with_pid}")
        log(f"[INFO] Hype rows whose names were enriched from dim_player_profile: {enriched_count}")
    else:
        log("[WARN] No profile lookup available; hype names will remain as in HTML")

    # Load season stats (batting + pitching)
    stats = _load_stats(season, league_id, repo_root)
    if stats.empty:
        log("[WARN] No stats available; preseason hype brief will be empty.")
        return 0

    # Match hype entries to stats
    matched_records: List[Dict[str, Any]] = []
    for entry in hype_rows:
        matched = match_hype_entry(entry, stats)
        if matched is None:
            continue

        team_fields = _pick_team_fields(matched)

        rec: Dict[str, Any] = {
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

    log(f"[INFO] Matched {len(matched_records)} hype entries to season stats")

    if not matched_records:
        log("[WARN] No preseason hype entries matched to stats; nothing to write.")
        return 0

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
