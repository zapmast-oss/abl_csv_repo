"""Generate pregame pack markdown."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from _abl_pregame_utils import (
    format_arsenal_display,
    find_csv_files,
    load_pitcher_arsenals,
    load_best_csv,
    md_table,
    normalize_team_table,
    resolve_base,
    safe_get,
    write_md,
)

TEAM_PATTERNS = ["dim_team", "teams", "team.csv", "team_list"]
BALLPARK_PATTERNS = ["dim_team_park", "parks", "ballpark", "park_factors", "team_park"]
FIN_PATTERNS = ["financial", "finances", "team_financial", "team_budget", "payroll", "revenue"]
FAN_PATTERNS = ["market", "fan", "attendance", "ticket", "media"]
FEATURED_PATTERNS = ["featured", "matchup"]
SCHEDULE_PATTERNS = ["probable", "probables", "game_preview", "schedule", "games", "matchup", "featured"]
PITCH_RATINGS_PATTERNS = ["pitching_ratings", "pitcher_ratings", "players_pitching", "player_pitching"]
BAT_RATINGS_PATTERNS = ["batting_ratings", "batter_ratings", "players_batting", "player_batting"]

AWAY_COLS = ["away", "away_abbr", "away_team", "away_team_abbr"]
HOME_COLS = ["home", "home_abbr", "home_team", "home_team_abbr"]
AWAY_PITCHER_COLS = ["away_probable", "probable_away", "away_pitcher", "away_sp", "away_starter"]
HOME_PITCHER_COLS = ["home_probable", "probable_home", "home_pitcher", "home_sp", "home_starter"]
AWAY_PITCHER_ID_COLS = ["away_pitcher_id", "away_player_id", "away_sp_id"]
HOME_PITCHER_ID_COLS = ["home_pitcher_id", "home_player_id", "home_sp_id"]

OVERALL_BAT_COLS = ["overall", "overall_bat", "overallbat", "rating_overall", "war", "ops", "wrc_plus", "wrc+"]


def merge_key(df: pd.DataFrame) -> pd.Series:
    if "team_id" in df.columns:
        return df["team_id"].apply(lambda v: f"id:{int(v)}" if pd.notna(v) else "")
    if "ID" in df.columns:
        return pd.to_numeric(df["ID"], errors="coerce").apply(lambda v: f"id:{int(v)}" if pd.notna(v) else "")
    if "team_abbr" in df.columns:
        return df["team_abbr"].fillna("").astype(str).str.upper()
    if "Abbr" in df.columns:
        return df["Abbr"].fillna("").astype(str).str.upper()
    if "team_name" in df.columns:
        return df["team_name"].fillna("").astype(str).str.lower()
    return pd.Series([""] * len(df))


def pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    series = safe_get(df, candidates)
    return series.name if isinstance(series, pd.Series) else None


def load_ballpark_info(base: Path, teams: pd.DataFrame) -> tuple[Dict[str, dict], Optional[Path], List[str]]:
    parks_df, parks_path = load_best_csv(base, BALLPARK_PATTERNS)
    notes: List[str] = []
    park_map: Dict[str, dict] = {}
    if parks_df.empty:
        notes.append("No ballpark data found.")
        return park_map, parks_path, notes

    park_name_col = pick_col(parks_df, ["park_name", "ballpark", "stadium", "park", "Park"])
    capacity_col = pick_col(parks_df, ["capacity", "cap"])
    factor_cols = [
        c
        for c in parks_df.columns
        if any(tok in c.lower() for tok in ["pf_", "pf ", "pf", "park_", "factor", "hr", "runs", "avg l", "avg r", "avg d"])
    ]

    parks_df = parks_df.copy()
    parks_df["__merge_key"] = merge_key(parks_df)
    for _, row in parks_df.iterrows():
        key = row.get("__merge_key", "")
        park_map[key] = {
            "name": row.get(park_name_col) if park_name_col else None,
            "capacity": row.get(capacity_col) if capacity_col else None,
            "factors": {col: row.get(col) for col in factor_cols} if factor_cols else {},
        }
    if park_name_col is None:
        notes.append("Park name column missing; using N/A.")
    if capacity_col is None:
        notes.append("Capacity column missing; using N/A.")
    return park_map, parks_path, notes


def load_finance_info(base: Path, teams: pd.DataFrame) -> tuple[Dict[str, dict], Optional[Path], List[str]]:
    fin_df, fin_path = load_best_csv(base, FIN_PATTERNS)
    notes: List[str] = []
    fin_map: Dict[str, dict] = {}
    if fin_df.empty:
        notes.append("No finance data found.")
        return fin_map, fin_path, notes
    fin_df = fin_df.copy()
    fin_df["__merge_key"] = merge_key(fin_df)
    budget_col = pick_col(fin_df, ["budget", "team_budget", "budget_total", "player_budget"])
    payroll_col = pick_col(fin_df, ["payroll", "team_payroll", "salary", "salary_total"])
    cash_col = pick_col(fin_df, ["cash", "cash_on_hand", "balance"])
    for _, row in fin_df.iterrows():
        key = row.get("__merge_key", "")
        fin_map[key] = {
            "budget": row.get(budget_col) if budget_col else None,
            "payroll": row.get(payroll_col) if payroll_col else None,
            "cash": row.get(cash_col) if cash_col else None,
        }
    return fin_map, fin_path, notes


def load_fan_info(base: Path, teams: pd.DataFrame) -> tuple[Dict[str, dict], List[Path], List[str]]:
    fan_paths = find_csv_files(base, FAN_PATTERNS)
    fan_map: Dict[str, dict] = {}
    notes: List[str] = []
    used_paths: List[Path] = []
    for path in fan_paths[:5]:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df = df.copy()
        df["__merge_key"] = merge_key(df)
        used_paths.append(path)
        for _, row in df.iterrows():
            key = row.get("__merge_key", "")
            current = fan_map.get(key, {})
            current.setdefault("market", row.get(pick_col(df, ["market", "market_size", "market_value"])))
            current.setdefault("interest", row.get(pick_col(df, ["fan_interest", "interest", "fan_interest_current"])))
            current.setdefault("loyalty", row.get(pick_col(df, ["fan_loyalty", "loyalty"])))
            current.setdefault("attendance", row.get(pick_col(df, ["attendance", "att", "home_attendance", "attendance_total", "attendance_ytd"])))
            current.setdefault("ticket", row.get(pick_col(df, ["ticket_price", "avg_ticket_price", "ticket"])))
            current.setdefault("media", row.get(pick_col(df, ["media_revenue", "tv_revenue", "radio_revenue", "local_media"])))
            fan_map[key] = current
    if not used_paths:
        notes.append("No fans/markets data found.")
    return fan_map, used_paths, notes


def parse_matchups_arg(arg: Optional[str]) -> List[Tuple[str, str]]:
    if not arg:
        return []
    pairs: List[Tuple[str, str]] = []
    for chunk in arg.split(","):
        if "@" not in chunk:
            continue
        away, home = chunk.split("@", 1)
        away = away.strip().upper()
        home = home.strip().upper()
        if away and home:
            pairs.append((away, home))
    return pairs


def discover_featured_matchups(base: Path) -> tuple[List[Tuple[str, str]], Optional[pd.DataFrame], Optional[Path]]:
    paths = [p for p in find_csv_files(base, FEATURED_PATTERNS) if "featured" in p.name.lower() and "matchup" in p.name.lower()]
    for path in paths:
        if path.suffix.lower() != ".csv":
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        away_col = pick_col(df, AWAY_COLS)
        home_col = pick_col(df, HOME_COLS)
        if away_col and home_col:
            matchups = []
            for _, row in df.iterrows():
                away = str(row.get(away_col, "")).strip().upper()
                home = str(row.get(home_col, "")).strip().upper()
                if away and home:
                    matchups.append((away, home))
            if matchups:
                return matchups, df, path
    return [], None, None


def choose_probables_source(base: Path, featured_df: Optional[pd.DataFrame], featured_path: Optional[Path]) -> tuple[pd.DataFrame, Optional[Path]]:
    if featured_df is not None:
        prob_cols = any(col.lower().startswith(("away_prob", "home_prob")) for col in featured_df.columns)
        if prob_cols:
            return featured_df, featured_path
    prob_df, prob_path = load_best_csv(base, SCHEDULE_PATTERNS)
    return prob_df, prob_path


def lookup_probable(row: pd.Series, away_abbr: str, home_abbr: str) -> tuple[Optional[dict], Optional[dict]]:
    away_name_col = pick_col(row.to_frame().T, AWAY_PITCHER_COLS)
    home_name_col = pick_col(row.to_frame().T, HOME_PITCHER_COLS)
    away_id_col = pick_col(row.to_frame().T, AWAY_PITCHER_ID_COLS)
    home_id_col = pick_col(row.to_frame().T, HOME_PITCHER_ID_COLS)

    away = None
    home = None
    if away_name_col or away_id_col:
        away = {
            "name": str(row.get(away_name_col)).strip() if away_name_col else None,
            "player_id": row.get(away_id_col),
            "team": away_abbr,
        }
    if home_name_col or home_id_col:
        home = {
            "name": str(row.get(home_name_col)).strip() if home_name_col else None,
            "player_id": row.get(home_id_col),
            "team": home_abbr,
        }
    return away, home


def describe_probables(prob_df: pd.DataFrame, away_abbr: str, home_abbr: str) -> tuple[Optional[dict], Optional[dict]]:
    if prob_df is None or prob_df.empty:
        return None, None
    away_col = pick_col(prob_df, AWAY_COLS)
    home_col = pick_col(prob_df, HOME_COLS)
    for _, row in prob_df.iterrows():
        a_val = str(row.get(away_col, "")).strip().upper() if away_col else ""
        h_val = str(row.get(home_col, "")).strip().upper() if home_col else ""
        if a_val == away_abbr and h_val == home_abbr:
            return lookup_probable(row, away_abbr, home_abbr)
    return None, None


def describe_arsenal(pitch_df: pd.DataFrame, pitcher: dict) -> str:
    if pitch_df is None or pitch_df.empty or not pitcher:
        return "Arsenal: N/A"
    pid_col = pick_col(pitch_df, ["player_id", "id"])
    if not pid_col:
        return "Arsenal: N/A"
    try:
        pid_val = int(pitcher.get("player_id"))
    except Exception:
        return "Arsenal: N/A"
    match = pitch_df[pd.to_numeric(pitch_df[pid_col], errors="coerce") == pid_val]
    if match.empty:
        return "Arsenal: N/A"
    row = match.iloc[0]
    pitch_cols = [c for c in pitch_df.columns if "pitch" in c.lower() and "rating" not in c.lower()]
    if not pitch_cols:
        return "Arsenal: N/A"
    pitches = []
    for col in pitch_cols[:5]:
        name = row.get(col)
        if pd.isna(name) or str(name).strip() == "":
            continue
        rating_col = f"{col}_rating"
        rating = row.get(rating_col) if rating_col in pitch_df.columns else None
        pitches.append((str(name).strip(), rating))
    if not pitches:
        return "Arsenal: N/A"
    pitch_count = len(pitches)
    best_pitch = max([p for p in pitches if p[1] is not None], key=lambda x: x[1], default=None)
    lines = [f"Arsenal ({pitch_count}): " + ", ".join(f"{name}{' '+str(rating) if rating is not None else ''}" for name, rating in pitches[:5])]
    if best_pitch:
        lines.append(f"Best pitch: {best_pitch[0]} ({best_pitch[1]})")
    return " ".join(lines)


def describe_key_bats(bat_df: pd.DataFrame, team_abbr: str) -> str:
    if bat_df is None or bat_df.empty:
        return "Key Bats: N/A"
    team_col = pick_col(bat_df, ["team_abbr", "team", "team_code"])
    if not team_col:
        return "Key Bats: N/A"
    overall_col = pick_col(bat_df, OVERALL_BAT_COLS)
    if not overall_col:
        return "Key Bats: N/A"
    subset = bat_df[bat_df[team_col].astype(str).str.upper() == team_abbr]
    if subset.empty:
        return "Key Bats: N/A"
    subset = subset.copy()
    subset["__overall"] = pd.to_numeric(subset[overall_col], errors="coerce")
    subset = subset.dropna(subset=["__overall"])
    if subset.empty:
        return "Key Bats: N/A"
    name_col = pick_col(subset, ["player_name", "name"])
    if not name_col:
        first = pick_col(subset, ["first_name", "first"])
        last = pick_col(subset, ["last_name", "last"])
        if first and last:
            subset["__name"] = subset[first].fillna("").astype(str) + " " + subset[last].fillna("").astype(str)
            name_col = "__name"
    subset = subset.sort_values("__overall", ascending=False).head(3)
    bats = []
    for _, row in subset.iterrows():
        name = row.get(name_col, "").strip() if name_col else ""
        overall = row.get("__overall")
        bats.append(f"{name} ({overall:.1f})" if name else f"{overall:.1f}")
    if not bats:
        return "Key Bats: N/A"
    return "Key Bats: " + ", ".join(bats)


def describe_team_detail(team_key: str, park_map: Dict[str, dict], fin_map: Dict[str, dict], fan_map: Dict[str, dict]) -> List[str]:
    lines = []
    park = park_map.get(team_key, {})
    park_name = park.get("name") or "N/A"
    capacity = park.get("capacity")
    cap_txt = f"{int(pd.to_numeric(capacity, errors='coerce')):,}" if capacity is not None and pd.notna(capacity) else "N/A"
    lines.append(f"Park: {park_name} (Cap: {cap_txt})")

    fin = fin_map.get(team_key, {})
    budget = fin.get("budget")
    payroll = fin.get("payroll")
    cash = fin.get("cash")
    budget_txt = f"{float(budget):,.0f}" if budget is not None and pd.notna(budget) else "N/A"
    payroll_txt = f"{float(payroll):,.0f}" if payroll is not None and pd.notna(payroll) else "N/A"
    cash_txt = f"{float(cash):,.0f}" if cash is not None and pd.notna(cash) else "N/A"
    parts = [f"Budget ${budget_txt}" if budget_txt != "N/A" else None, f"Payroll ${payroll_txt}" if payroll_txt != "N/A" else None, f"Cash ${cash_txt}" if cash_txt != "N/A" else None, f"Balance {fin.get('profit')}" if fin.get("profit") is not None else None]
    parts = [p for p in parts if p]
    lines.append("Finances: " + (" | ".join(parts) if parts else "N/A"))

    fan = fan_map.get(team_key, {})
    market = fan.get("market")
    interest = fan.get("interest")
    attendance = fan.get("attendance")
    fan_bits = [
        f"Market {market}" if market not in (None, pd.NA) else None,
        f"Interest {interest}" if interest not in (None, pd.NA) else None,
        f"Attendance {attendance}" if attendance not in (None, pd.NA) else None,
    ]
    fan_bits = [b for b in fan_bits if b]
    lines.append("Fans/Market: " + (", ".join(fan_bits) if fan_bits else "N/A"))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABL pregame pack.")
    parser.add_argument("--base", help="Repo root (optional).")
    parser.add_argument("--season", type=int, required=False)
    parser.add_argument("--week", type=int, required=False)
    parser.add_argument("--league_id", type=int, default=200)
    parser.add_argument("--matchups", help="Explicit matchups list, e.g., CHI@MIA,DEN@NAS")
    parser.add_argument("--arsenal-top", type=int, default=3, help="Top N pitches to display for arsenal")
    args = parser.parse_args()

    base = resolve_base(args.base)
    season = args.season
    week = args.week
    league_id = args.league_id
    arsenal_top = max(1, args.arsenal_top if args.arsenal_top else 3)

    teams_raw, team_path = load_best_csv(base, TEAM_PATTERNS)
    teams = normalize_team_table(teams_raw)
    if "league_id" in teams.columns:
        teams = teams[(teams["league_id"].isna()) | (teams["league_id"] == league_id)]
    teams = teams.reset_index(drop=True)
    teams["__merge_key"] = merge_key(teams)
    abbr_to_key = {row["team_abbr"]: row["__merge_key"] for _, row in teams.iterrows() if pd.notna(row.get("team_abbr"))}

    park_map, park_path, park_notes = load_ballpark_info(base, teams)
    fin_map, fin_path, fin_notes = load_finance_info(base, teams)
    fan_map, fan_paths, fan_notes = load_fan_info(base, teams)
    arsenal_df, arsenal_sources, arsenal_notes = load_pitcher_arsenals(base)
    arsenal_by_id: Dict[int, pd.Series] = {}
    arsenal_by_name_team: Dict[tuple[str, str], pd.Series] = {}
    for _, row in arsenal_df.iterrows():
        pid = row.get("player_id")
        try:
            pid_int = int(pid)
            arsenal_by_id[pid_int] = row
        except Exception:
            pass
        raw_name = row.get("player_name")
        name_key = str(raw_name) if pd.notna(raw_name) else ""
        name_key = name_key.strip().upper()
        team_raw = row.get("team_abbr")
        team_key = str(team_raw) if pd.notna(team_raw) else ""
        team_key = team_key.strip().upper()
        if name_key:
            arsenal_by_name_team[(name_key, team_key)] = row

    matchups, featured_df, featured_path = discover_featured_matchups(base)
    explicit = parse_matchups_arg(args.matchups)
    if explicit:
        matchups = explicit

    prob_df, prob_path = choose_probables_source(base, featured_df, featured_path)
    pitch_df, pitch_path = load_best_csv(base, PITCH_RATINGS_PATTERNS)
    bat_df, bat_path = load_best_csv(base, BAT_RATINGS_PATTERNS)

    season_label = season if season is not None else "N/A"
    week_label = f"{week:02d}" if isinstance(week, int) else ("N/A" if week is None else str(week))
    md_lines = [f"# ABL Pregame Pack - Season {season_label} Week {week_label}", ""]

    if not matchups:
        md_lines.append("No featured matchups artifact found; use --matchups to provide pairs like CHI@MIA.")
    else:
        for away_abbr, home_abbr in matchups:
            md_lines.append(f"## {away_abbr} at {home_abbr}")
            away_prob, home_prob = describe_probables(prob_df, away_abbr, home_abbr)
            if away_prob or home_prob:
                away_name = away_prob.get("name") if away_prob else None
                home_name = home_prob.get("name") if home_prob else None
                away_txt = away_name if away_name else "N/A"
                home_txt = home_name if home_name else "N/A"
                md_lines.append(f"Probable Starters: {away_abbr} {away_txt} vs {home_abbr} {home_txt}")
            else:
                md_lines.append("Probable Starters: TBD")

            md_lines.append(describe_arsenal(pitch_df, away_prob) if away_prob else "Arsenal: N/A")
            md_lines.append(describe_arsenal(pitch_df, home_prob) if home_prob else "Arsenal: N/A")

            md_lines.append(describe_key_bats(bat_df, away_abbr))
            md_lines.append(describe_key_bats(bat_df, home_abbr))

            away_key = abbr_to_key.get(away_abbr, "")
            home_key = abbr_to_key.get(home_abbr, "")
            for line in describe_team_detail(away_key, park_map, fin_map, fan_map):
                md_lines.append(f"{away_abbr} {line}")
            for line in describe_team_detail(home_key, park_map, fin_map, fan_map):
                md_lines.append(f"{home_abbr} {line}")
            md_lines.append("### Pitching Snapshot")
            # helpers for arsenal lookup
            def arsenal_line(pitcher: Optional[dict], team_abbr: str) -> str:
                if not pitcher or pitcher.get("name") in (None, "", pd.NA):
                    return "TBD"
                pid = pitcher.get("player_id")
                row = None
                try:
                    if pid is not None and not pd.isna(pid):
                        row = arsenal_by_id.get(int(pid))
                except Exception:
                    row = None
                if row is None:
                    name_key = str(pitcher.get("name") or "").strip().upper()
                    row = arsenal_by_name_team.get((name_key, team_abbr.upper()), None)
                if row is None:
                    return "Arsenal: N/A (no repertoire source found)"
                return format_arsenal_display(row, top_k=arsenal_top)

            away_ars = arsenal_line(away_prob, away_abbr)
            home_ars = arsenal_line(home_prob, home_abbr)
            md_lines.append(f"{away_abbr} starter: {away_ars}")
            md_lines.append(f"{home_abbr} starter: {home_ars}")
            if arsenal_sources:
                md_lines.append("Arsenal sources: " + "; ".join(arsenal_sources))
            else:
                md_lines.append("Arsenal sources: none found")
            if arsenal_notes:
                md_lines.append("Arsenal notes: " + "; ".join(arsenal_notes))
            md_lines.append("")

    data_sources: List[str] = []
    for path in [team_path, park_path, fin_path, featured_path, prob_path, pitch_path, bat_path]:
        if path:
            data_sources.append(str(path))
    data_sources.extend(str(p) for p in fan_paths)
    data_sources.extend(arsenal_sources)
    md_lines.append("## Data Sources")
    if data_sources:
        for src in data_sources:
            md_lines.append(f"- {src}")
    else:
        md_lines.append("- None found")
    md_lines.append("")
    md_lines.append("## Notes")
    notes = park_notes + fin_notes + fan_notes
    if not matchups:
        notes.append("No matchups provided or discovered.")
    if notes:
        md_lines.extend(f"- {n}" for n in notes)
    else:
        md_lines.append("- None")

    out_path = base / "csv" / "out" / "text_out" / "pregame" / "pregame_packs.md"
    write_md(out_path, "\n".join(md_lines).rstrip() + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
