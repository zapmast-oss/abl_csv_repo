#!/usr/bin/env python
"""Build a preseason hype vs performance markdown for any season/league."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from eb_text_utils import canonicalize_team_city, format_team_label, normalize_eb_text
from z_abl_almanac_html_helpers import parse_preseason_predictions


def log(msg: str) -> None:
    print(msg, flush=True)


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


def _compute_score(row: pd.Series) -> float:
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


def _load_stats(repo_root: Path, season: int, league_id: int) -> pd.DataFrame:
    base = repo_root / "csv" / "out" / "almanac" / str(season)
    bat_path = base / f"player_batting_{season}_league{league_id}.csv"
    pit_path = base / f"player_pitching_{season}_league{league_id}.csv"
    frames: List[pd.DataFrame] = []
    if bat_path.exists():
        bat = pd.read_csv(bat_path)
        bat = _attach_name_norms(bat, "player_name")
        bat["role_type"] = "bat"
        bat["score"] = bat.apply(_compute_score, axis=1)
        frames.append(bat)
    else:
        log(f"[WARN] Missing batting stats: {bat_path}")
    if pit_path.exists():
        pit = pd.read_csv(pit_path)
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
    name = row.get("team_name")
    abbr = row.get("team_abbr")
    if pd.notna(name):
        name = canonicalize_team_city(name)
    return {
        "team_name": name if pd.notna(name) else None,
        "team_abbr": abbr if pd.notna(abbr) else None,
    }


def match_hype_entry(entry: Dict[str, Any], stats: pd.DataFrame) -> Optional[pd.Series]:
    if stats.empty:
        return None
    hyp_last = _last_name(entry.get("player_name", ""))
    hyp_init = _first_initial(entry.get("player_name", ""))
    candidates = stats[stats["player_last"] == hyp_last]
    if hyp_init:
        narrowed = candidates[candidates["player_initial"] == hyp_init]
        if not narrowed.empty:
            candidates = narrowed
    if candidates.empty:
        return None
    candidates = candidates.sort_values("score", ascending=False)
    return candidates.iloc[0]


def bucketize(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {"over-delivered": [], "delivered": [], "under-delivered": []}
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
    lines: List[str] = []
    lines.append("## Preseason hype — who delivered?")
    lines.append(f"_Based on preseason predictions and actual {season} performance._")
    lines.append("")
    order = [("over-delivered", "Over-delivered"), ("delivered", "Delivered"), ("under-delivered", "Under-delivered")]
    for key, title in order:
        lines.append(f"**{title}**")
        if not buckets[key]:
            lines.append("- None")
        for rec in buckets[key]:
            team_label = format_team_label(rec.get("team_name"), rec.get("team_abbr"))
            score = rec.get("score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
            lines.append(f"- {rec.get('player_name','Unknown')} — {team_label} — score: {score_str}")
        lines.append("")
    return normalize_eb_text("\n".join(lines).strip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build preseason hype vs performance brief.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    parser.add_argument("--preseason-html", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    season = args.season
    league_id = args.league_id

    log(f"[INFO] Building preseason hype brief for season={season}, league={league_id}")
    if not args.preseason_html.exists():
        log(f"[WARN] Preseason HTML not found at {args.preseason_html}; output will contain placeholder data.")
        hype_rows = [{"player_name": "Unknown", "team_name": "", "hype_role": "preseason", "rank": None, "source": "missing_html"}]
    else:
        hype_rows = parse_preseason_predictions(args.preseason_html)
        log(f"[INFO] Parsed {len(hype_rows)} hype rows from {args.preseason_html}")

    stats = _load_stats(repo_root, season, league_id)
    if stats.empty:
        log("[WARN] No stats available; output will be generic.")

    matched_records: List[Dict[str, Any]] = []
    for entry in hype_rows:
        match = match_hype_entry(entry, stats) if not stats.empty else None
        if match is None:
            log(f"[WARN] Could not match hyped player: {entry.get('player_name')}")
            continue
        team_fields = _pick_team_fields(match)
        rec = {
            "player_name": entry.get("player_name"),
            "team_name": team_fields["team_name"],
            "team_abbr": team_fields["team_abbr"],
            "hype_role": entry.get("hype_role"),
            "source": entry.get("source"),
            "score": match.get("score", 0.0),
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
