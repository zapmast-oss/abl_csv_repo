#!/usr/bin/env python
"""Build a preseason hype vs performance markdown for any season/league.

This version is designed to:
- Parse preseason predictions from the almanac HTML.
- Join them to season stats (batting + pitching) exported from the almanac.
- Attach each hyped player to their actual ABL club when possible, using:
    * player name matching vs the stats
    * team_abbr/team_name from the stats rows
    * dim_team_park as a backup lookup
- Only fall back to "Free agent" when we truly cannot locate a team.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from eb_text_utils import canonicalize_team_city, format_team_label, normalize_eb_text
from z_abl_almanac_html_helpers import parse_preseason_predictions


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    txt = str(name or "").replace(".", " ").replace(",", " ")
    return " ".join(txt.split()).lower()


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
# Scoring and stats loading
# ---------------------------------------------------------------------------

def _compute_score(row: pd.Series) -> float:
    """Compute a single scalar "score" from a batting or pitching row.

    Priority:
    - WAR if available
    - OPS for hitters
    - negative ERA for pitchers (lower ERA => higher score)
    - SO as a last resort
    """
    # WAR is ideal when present
    for key in ("WAR", "war"):
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                pass

    # OPS is a decent single-number proxy for hitters
    if "OPS" in row and pd.notna(row["OPS"]):
        try:
            return float(row["OPS"])
        except Exception:
            pass

    # ERA (invert: lower ERA => higher score)
    if "ERA" in row and pd.notna(row["ERA"]):
        try:
            era_val = float(row["ERA"])
            if era_val > 0:
                return -era_val
        except Exception:
            pass

    # Raw strikeouts as a fallback
    if "SO" in row and pd.notna(row["SO"]):
        try:
            return float(row["SO"])
        except Exception:
            pass

    return 0.0


def _load_stats(repo_root: Path, season: int, league_id: int) -> pd.DataFrame:
    """Load combined batting + pitching stats for the season/league.

    Expects almanac CSVs:
      csv/out/almanac/{season}/player_batting_{season}_league{league_id}.csv
      csv/out/almanac/{season}/player_pitching_{season}_league{league_id}.csv

    Attaches:
      - player_name_norm, player_last, player_initial
      - role_type ("bat" / "pit")
      - score (via _compute_score)
    """
    base = repo_root / "csv" / "out" / "almanac" / str(season)
    bat_path = base / f"player_batting_{season}_league{league_id}.csv"
    pit_path = base / f"player_pitching_{season}_league{league_id}.csv"

    frames: List[pd.DataFrame] = []

    if bat_path.exists():
        bat = pd.read_csv(bat_path)
        if "player_name" in bat.columns:
            bat = _attach_name_norms(bat, "player_name")
            bat["role_type"] = "bat"
            bat["score"] = bat.apply(_compute_score, axis=1)
            frames.append(bat)
        else:
            log(f"[WARN] Batting stats at {bat_path} missing 'player_name' column; skipping.")
    else:
        log(f"[WARN] Missing batting stats: {bat_path}")

    if pit_path.exists():
        pit = pd.read_csv(pit_path)
        if "player_name" in pit.columns:
            pit = _attach_name_norms(pit, "player_name")
            pit["role_type"] = "pit"
            pit["score"] = pit.apply(_compute_score, axis=1)
            frames.append(pit)
        else:
            log(f"[WARN] Pitching stats at {pit_path} missing 'player_name' column; skipping.")
    else:
        log(f"[WARN] Missing pitching stats: {pit_path}")

    if not frames:
        log("[WARN] No stats frames loaded; preseason hype will have no matches.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


# ---------------------------------------------------------------------------
# Team lookup helpers (dim_team_park + canonicalize_team_city)
# ---------------------------------------------------------------------------

def _norm_key(value: str) -> str:
    """Normalize arbitrary text to a key suitable for fuzzy team lookup."""
    txt = str(value or "").lower()
    return re.sub(r"[^a-z0-9]+", "", txt)


def _build_team_lookup(repo_root: Path) -> Dict[str, Tuple[str, str]]:
    """Build a lookup of various team strings -> (team_abbr, team_name).

    Uses dim_team_park.csv and canonicalize_team_city for robust matching.
    """
    dim_path = (
        repo_root
        / "csv"
        / "out"
        / "star_schema"
        / "dim_team_park.csv"
    )
    if not dim_path.exists():
        log(f"[WARN] dim_team_park not found at {dim_path}; team lookup will be limited.")
        return {}

    dim = pd.read_csv(dim_path)
    required = {"Team Name", "Abbr", "City"}
    if not required.issubset(set(dim.columns)):
        log(
            f"[WARN] dim_team_park at {dim_path} missing expected columns "
            f"{sorted(required)}; got {list(dim.columns)}"
        )
        return {}

    lookup: Dict[str, Tuple[str, str]] = {}
    for _, row in dim.iterrows():
        team_name = str(row["Team Name"])
        abbr = str(row["Abbr"])
        city_raw = "" if pd.isna(row["City"]) else str(row["City"])
        city_short = city_raw.split("(")[0].strip() if city_raw else ""
        city_canon = canonicalize_team_city(city_raw) if city_raw else ""
        label = team_name

        candidates = {
            team_name,
            abbr,
            city_raw,
            city_short,
            city_canon,
            f"{city_short} {team_name}".strip(),
            f"{team_name} ({abbr})",
        }

        for cand in candidates:
            if not cand:
                continue
            key = _norm_key(cand)
            if not key:
                continue
            # First come, first served to avoid noisy overwrites.
            lookup.setdefault(key, (abbr, label))

    return lookup


def _lookup_team_from_text(raw: Optional[str], lookup: Dict[str, Tuple[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """Attempt to map a raw team string to (team_abbr, team_name) via lookup."""
    if not raw or not lookup:
        return None, None
    variants = [raw, canonicalize_team_city(raw)]
    for txt in variants:
        if not txt:
            continue
        key = _norm_key(txt)
        if key in lookup:
            abbr, name = lookup[key]
            return abbr, name
    return None, None


# ---------------------------------------------------------------------------
# Matching hype entries to stats
# ---------------------------------------------------------------------------

def match_hype_entry(entry: Dict[str, Any], stats: pd.DataFrame) -> Optional[pd.Series]:
    """Pick the best stats row for a hyped player using name matching and score."""
    if stats.empty:
        return None

    hyp_name = entry.get("player_name", "") or ""
    hyp_last = _last_name(hyp_name)
    hyp_init = _first_initial(hyp_name)

    if not hyp_last:
        return None

    candidates = stats[stats["player_last"] == hyp_last]
    if hyp_init:
        narrowed = candidates[candidates["player_initial"] == hyp_init]
        if not narrowed.empty:
            candidates = narrowed

    if candidates.empty:
        return None

    candidates = candidates.sort_values("score", ascending=False)
    return candidates.iloc[0]


def _resolve_team_for_entry(
    entry: Dict[str, Any],
    matched_row: pd.Series,
    team_lookup: Dict[str, Tuple[str, str]],
) -> Tuple[Optional[str], Optional[str]]:
    """Determine the best (team_abbr, team_name) for a hype/season match.

    Priority:
    1. Use team_abbr/team_name from the matched stats row if present.
    2. If only one of those is present, fill the other via dim_team_park lookup.
    3. If the stats row has no team info, try the hype entry's team fields
       via canonicalize_team_city + dim_team_park.
    4. If everything fails, return (None, None) and let format_team_label
       fall back to "Free agent".
    """
    team_abbr: Optional[str] = None
    team_name: Optional[str] = None

    # 1) From stats row, if available
    stats_team_cols_abbr = ["team_abbr", "Team Abbr", "team", "Team"]
    stats_team_cols_name = ["team_name", "Team Name"]

    for col in stats_team_cols_abbr:
        if col in matched_row.index and pd.notna(matched_row[col]):
            team_abbr = str(matched_row[col])
            break

    for col in stats_team_cols_name:
        if col in matched_row.index and pd.notna(matched_row[col]):
            team_name = str(matched_row[col])
            break

    # 2) Use dim_team_park to fill missing piece if we have at least one.
    if team_lookup:
        if team_abbr and not team_name:
            key = _norm_key(team_abbr)
            if key in team_lookup:
                _, name = team_lookup[key]
                team_name = name
        elif team_name and not team_abbr:
            abbr, name = _lookup_team_from_text(team_name, team_lookup)
            if abbr and name:
                team_abbr, team_name = abbr, name

    # 3) If we still have nothing, fall back to hype entry's team text.
    if not team_abbr and not team_name:
        hype_team_text: Optional[str] = None
        for key in ("team_name", "team", "team_city", "team_name_x", "team_name_y"):
            val = entry.get(key)
            if val:
                hype_team_text = str(val)
                break

        if hype_team_text:
            abbr, name = _lookup_team_from_text(hype_team_text, team_lookup)
            if abbr and name:
                team_abbr, team_name = abbr, name

    return team_abbr, team_name


# ---------------------------------------------------------------------------
# Bucketing + markdown
# ---------------------------------------------------------------------------

def bucketize(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split records into over-delivered / delivered / under-delivered thirds."""
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "over-delivered": [],
        "delivered": [],
        "under-delivered": [],
    }
    if not records:
        return buckets

    sorted_recs = sorted(records, key=lambda r: r.get("score", 0.0), reverse=True)
    n = len(sorted_recs)
    third = max(n // 3, 1)
    for idx, rec in enumerate(sorted_recs):
        if idx < third:
            buckets["over-delivered"].append(rec)
        elif idx < third * 2:
            buckets["delivered"].append(rec)
        else:
            buckets["under-delivered"].append(rec)
    return buckets


def build_md(buckets: Dict[str, List[Dict[str, Any]]], season: int) -> str:
    """Render the hype vs performance buckets to markdown."""
    lines: List[str] = []
    lines.append("## Preseason hype — who delivered?")
    lines.append(f"_Based on preseason predictions and actual {season} performance._")
    lines.append("")

    order = [
        ("over-delivered", "Over-delivered"),
        ("delivered", "Delivered"),
        ("under-delivered", "Under-delivered"),
    ]
    for key, title in order:
        lines.append(f"**{title}**")
        if not buckets[key]:
            lines.append("- None")
        else:
            for rec in buckets[key]:
                team_label = format_team_label(rec.get("team_name"), rec.get("team_abbr"))
                score = rec.get("score")
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
                player_name = rec.get("player_name", "Unknown")
                lines.append(f"- {player_name} — {team_label} — score: {score_str}")
        lines.append("")

    return normalize_eb_text("\n".join(lines).strip() + "\n")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build preseason hype vs performance markdown for any season/league."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
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

    log(f"[INFO] Building preseason hype brief for season={season}, league={league_id}")

    # Parse hype from HTML (or create a minimal placeholder if missing).
    if not args.preseason_html.exists():
        log(
            f"[WARN] Preseason HTML not found at {args.preseason_html}; "
            "output will contain placeholder data."
        )
        hype_rows: List[Dict[str, Any]] = [
            {
                "player_name": "Unknown",
                "team_name": "",
                "hype_role": "preseason",
                "rank": None,
                "source": "missing_html",
            }
        ]
    else:
        raw_hype = parse_preseason_predictions(args.preseason_html)
        if isinstance(raw_hype, pd.DataFrame):
            hype_df = raw_hype.copy()
        else:
            hype_df = pd.DataFrame(raw_hype)
        log(f"[INFO] Parsed {len(hype_df)} hype rows from {args.preseason_html}")
        hype_rows = hype_df.to_dict(orient="records")

    # Persist a CSV version for verification (used by the runner).
    out_almanac_dir = repo_root / "csv" / "out" / "almanac" / str(season)
    out_almanac_dir.mkdir(parents=True, exist_ok=True)
    preseason_csv_path = out_almanac_dir / f"preseason_player_predictions_{season}_league{league_id}.csv"
    pd.DataFrame(hype_rows).to_csv(preseason_csv_path, index=False)
    log(f"[INFO] Wrote preseason predictions CSV to {preseason_csv_path}")

    # Load stats and team lookup.
    stats = _load_stats(repo_root, season, league_id)
    team_lookup = _build_team_lookup(repo_root)

    matched_records: List[Dict[str, Any]] = []

    for entry in hype_rows:
        matched = match_hype_entry(entry, stats)
        if matched is None:
            continue

        player_name = entry.get("player_name") or matched.get("player_name") or "Unknown"
        try:
            score_val = float(matched.get("score", 0.0))
        except Exception:
            score_val = 0.0

        team_abbr, team_name = _resolve_team_for_entry(entry, matched, team_lookup)

        rec: Dict[str, Any] = {
            "player_name": player_name,
            "team_abbr": team_abbr,
            "team_name": team_name,
            "score": score_val,
        }
        matched_records.append(rec)

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
