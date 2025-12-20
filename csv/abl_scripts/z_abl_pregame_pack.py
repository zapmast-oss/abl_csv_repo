"""Generate pregame pack markdown."""

from __future__ import annotations

import argparse
from pathlib import Path


from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from _abl_pregame_utils import (
    format_int_commas,
    format_money_short,
    format_manager_tendencies,
    format_arsenal_display,
    find_best_pitch_for_pitcher,
    find_csv_files,
    load_team_fans_markets,
    load_team_financials,
    load_pitcher_arsenals,
    load_best_csv,
    parse_batter_profile_all,
    load_batter_profiles,
    load_players_lookup,
    load_projected_starters,
    load_team_reporting,
    load_manager_tendencies,
    load_lsdl_schedule,
    parse_money_to_float,
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
AWAY_PITCHER_ID_COLS = ["away_pitcher_id", "away_player_id", "away_sp_id", "starter0", "away_starter_id"]
HOME_PITCHER_ID_COLS = ["home_pitcher_id", "home_player_id", "home_sp_id", "starter1", "home_starter_id"]
AWAY_TEAM_ID_COLS = ["away_team", "away_team_id", "away_id"]
HOME_TEAM_ID_COLS = ["home_team", "home_team_id", "home_id"]

OVERALL_BAT_COLS = ["overall", "overall_bat", "overallbat", "rating_overall", "war", "ops", "wrc_plus", "wrc+"]


def format_money(val) -> str:
    if pd.isna(val):
        return "N/A"
    num = parse_numeric(val).iloc[0]
    if pd.notna(num):
        return f"${num:,.0f}"
    return str(val)


def parse_numeric(val) -> pd.Series:
    """Best-effort numeric parser that handles trailing m (millions)."""
    if pd.isna(val):
        return pd.Series([pd.NA])
    if isinstance(val, str):
        txt = val.strip().lower()
        txt = txt.replace("$", "").replace(",", "")
        multiplier = 1
        if txt.endswith("m"):
            multiplier = 1_000_000
            txt = txt[:-1]
        num = pd.to_numeric(pd.Series([txt]), errors="coerce")
        if num.notna().iloc[0]:
            return num * multiplier
    return pd.to_numeric(pd.Series([val]), errors="coerce")


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


def describe_probables(
    prob_df: pd.DataFrame,
    away_abbr: str,
    home_abbr: str,
    abbr_to_id: dict[str, int],
) -> tuple[Optional[dict], Optional[dict]]:
    if prob_df is None or prob_df.empty:
        return None, None
    away_col = pick_col(prob_df, AWAY_COLS)
    home_col = pick_col(prob_df, HOME_COLS)
    away_id_col = pick_col(prob_df, AWAY_TEAM_ID_COLS)
    home_id_col = pick_col(prob_df, HOME_TEAM_ID_COLS)
    away_id = abbr_to_id.get(away_abbr)
    home_id = abbr_to_id.get(home_abbr)
    for _, row in prob_df.iterrows():
        a_val = str(row.get(away_col, "")).strip().upper() if away_col else ""
        h_val = str(row.get(home_col, "")).strip().upper() if home_col else ""
        if away_col and home_col and a_val == away_abbr and h_val == home_abbr:
            return lookup_probable(row, away_abbr, home_abbr)
        if away_id_col and home_id_col and away_id and home_id:
            a_id = pd.to_numeric(pd.Series([row.get(away_id_col)]), errors="coerce").iloc[0]
            h_id = pd.to_numeric(pd.Series([row.get(home_id_col)]), errors="coerce").iloc[0]
            if pd.notna(a_id) and pd.notna(h_id) and int(a_id) == away_id and int(h_id) == home_id:
                return lookup_probable(row, away_abbr, home_abbr)
    return None, None


def matchup_in_schedule(
    prob_df: pd.DataFrame,
    away_abbr: str,
    home_abbr: str,
    abbr_to_id: dict[str, int],
) -> bool:
    if prob_df is None or prob_df.empty:
        return False
    away_col = pick_col(prob_df, AWAY_COLS)
    home_col = pick_col(prob_df, HOME_COLS)
    if away_col and home_col:
        a_vals = prob_df[away_col].astype(str).str.strip().str.upper()
        h_vals = prob_df[home_col].astype(str).str.strip().str.upper()
        return bool(((a_vals == away_abbr) & (h_vals == home_abbr)).any())
    away_id_col = pick_col(prob_df, AWAY_TEAM_ID_COLS)
    home_id_col = pick_col(prob_df, HOME_TEAM_ID_COLS)
    away_id = abbr_to_id.get(away_abbr)
    home_id = abbr_to_id.get(home_abbr)
    if away_id_col and home_id_col and away_id and home_id:
        a_vals = pd.to_numeric(prob_df[away_id_col], errors="coerce")
        h_vals = pd.to_numeric(prob_df[home_id_col], errors="coerce")
        return bool(((a_vals == away_id) & (h_vals == home_id)).any())
    return False


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


def describe_team_detail(
    team_abbr: Optional[str],
    team_id: Optional[int],
    team_key: str,
    park_map: Dict[str, dict],
    fan_by_abbr: Dict[str, pd.Series],
    fan_by_id: Dict[int, pd.Series],
    fin_by_abbr: Dict[str, pd.Series],
    fin_by_id: Dict[int, pd.Series],
) -> List[str]:
    lines = []
    park = park_map.get(team_key, {})
    park_name = park.get("name") or "N/A"
    capacity = park.get("capacity")
    cap_txt = f"{int(pd.to_numeric(capacity, errors='coerce')):,}" if capacity is not None and pd.notna(capacity) else "N/A"
    lines.append(f"Park: {park_name} (Cap: {cap_txt})")

    def lookup_fin_row() -> Optional[pd.Series]:
        abbr_key = str(team_abbr).strip().upper() if team_abbr else None
        if abbr_key and abbr_key in fin_by_abbr:
            return fin_by_abbr[abbr_key]
        if team_id is not None and pd.notna(team_id) and int(team_id) in fin_by_id:
            return fin_by_id[int(team_id)]
        return None

    fin_row = lookup_fin_row()
    if fin_row is None or fin_row.empty:
        lines.append("Finances: N/A (no finance row matched team_abbr/team_id)")
    else:
        budget = fin_row.get("budget")
        payroll = fin_row.get("payroll")
        cash = fin_row.get("cash")
        revenue = fin_row.get("revenue")
        balance = fin_row.get("profit")
        parts = [
            f"Budget {format_money(budget)}" if not pd.isna(budget) else None,
            f"Payroll {format_money(payroll)}" if not pd.isna(payroll) else None,
            f"Cash {format_money(cash)}" if not pd.isna(cash) else None,
            f"Revenue {format_money(revenue)}" if not pd.isna(revenue) else None,
            f"Balance {format_money(balance)}" if not pd.isna(balance) else None,
        ]
        parts = [p for p in parts if p]
        tier = fin_row.get("__tier", "N/A")
        line = "Finances: " + (" | ".join(parts) if parts else "N/A")
        if pd.notna(tier) and tier != "N/A":
            line += f" | Tier: {tier}"
        lines.append(line)

    def lookup_fan_row() -> Optional[pd.Series]:
        abbr_key = str(team_abbr).strip().upper() if team_abbr else None
        if abbr_key and abbr_key in fan_by_abbr:
            return fan_by_abbr[abbr_key]
        if team_id is not None and pd.notna(team_id) and int(team_id) in fan_by_id:
            return fan_by_id[int(team_id)]
        return None

    fan_row = lookup_fan_row()
    if fan_row is None or fan_row.empty:
        lines.append("Fans/Market: N/A")
    else:
        market = fan_row.get("market")
        interest = fan_row.get("fan_interest")
        loyalty = fan_row.get("fan_loyalty")
        att_avg = fan_row.get("attendance_avg")
        att_tot = fan_row.get("attendance_total")
        ticket_price = fan_row.get("ticket_price_avg")
        gate_rev = fan_row.get("gate_revenue")
        bits = []
        if pd.notna(market):
            bits.append(f"Market {market}")
        if pd.notna(interest):
            bits.append(f"Interest {interest}")
        if pd.notna(loyalty):
            bits.append(f"Loyalty {loyalty}")
        if pd.notna(att_avg):
            bits.append(f"Att {format_int_commas(att_avg)}")
        elif pd.notna(att_tot):
            bits.append(f"Att {format_int_commas(att_tot)} (season)")
        if pd.notna(ticket_price):
            num = parse_money_to_float(ticket_price)
            if pd.notna(num):
                bits.append(f"Ticket ${num:,.2f}")
        if pd.notna(gate_rev):
            bits.append(f"Gate {format_money_short(gate_rev)}")
        lines.append("Fans/Market: " + (" | ".join(bits) if bits else "N/A"))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABL pregame pack.")
    parser.add_argument("--base", help="Repo root (optional).")
    parser.add_argument("--season", type=int, required=False)
    parser.add_argument("--week", type=int, required=False)
    parser.add_argument("--league_id", type=int, default=200)
    parser.add_argument("--matchups", help="Explicit matchups list, e.g., CHI@MIA,DEN@NAS")
    parser.add_argument("--arsenal-top", type=int, default=3, help="Top N pitches to display for arsenal")
    parser.add_argument("--show-arsenal-count", action="store_true", default=True, help="Show pitch count when available")
    parser.add_argument("--bats-top", type=int, default=2, help="Top N key bats per team")
    args = parser.parse_args()

    base = resolve_base(args.base)
    season = args.season
    week = args.week
    league_id = args.league_id
    arsenal_top = max(1, args.arsenal_top if args.arsenal_top else 3)
    show_ars_count = bool(args.show_arsenal_count)
    bats_top = max(1, args.bats_top if args.bats_top else 2)

    teams_raw, team_path = load_best_csv(base, TEAM_PATTERNS)
    teams = normalize_team_table(teams_raw)
    if "league_id" in teams.columns:
        teams = teams[(teams["league_id"].isna()) | (teams["league_id"] == league_id)]
    teams = teams.reset_index(drop=True)
    teams["__merge_key"] = merge_key(teams)
    abbr_to_key = {row["team_abbr"]: row["__merge_key"] for _, row in teams.iterrows() if pd.notna(row.get("team_abbr"))}

    park_map, park_path, park_notes = load_ballpark_info(base, teams)
    fin_df, fin_sources, fin_notes = load_team_financials(base, league_id=league_id, season=season)
    fan_df, fan_sources, fan_notes = load_team_fans_markets(base, league_id=league_id, season=season)
    # Finance lookups and tiers
    fin_df = fin_df.copy()
    if "team_abbr" in fin_df.columns:
        fin_df["__abbr"] = fin_df["team_abbr"].astype(str).str.strip().str.upper()
    else:
        fin_df["__abbr"] = pd.NA
    if "team_id" in fin_df.columns:
        fin_df["__team_id"] = pd.to_numeric(fin_df["team_id"], errors="coerce").astype("Int64")
    else:
        fin_df["__team_id"] = pd.Series(dtype="Int64")
    def to_numeric_series(series: pd.Series | None) -> pd.Series:
        if series is None:
            return pd.Series(dtype=float)
        return series.apply(lambda v: parse_numeric(v).iloc[0])

    payroll_raw = to_numeric_series(fin_df.get("payroll") if "payroll" in fin_df.columns else None)
    budget_raw = to_numeric_series(fin_df.get("budget") if "budget" in fin_df.columns else None)
    payroll_count = int(payroll_raw.notna().sum()) if isinstance(payroll_raw, pd.Series) else 0
    metric_raw = payroll_raw if payroll_count > 0 else budget_raw

    def assign_fin_tier(series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        if series.notna().sum() < 8:
            return pd.Series(["N/A"] * len(series), index=series.index)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        tiers = []
        for val in series:
            if pd.isna(val):
                tiers.append("N/A")
            elif val >= q3:
                tiers.append("High")
            elif val <= q1:
                tiers.append("Low")
            else:
                tiers.append("Mid")
        return pd.Series(tiers, index=series.index)

    fin_df["__tier"] = assign_fin_tier(metric_raw) if not fin_df.empty else pd.Series(dtype=object)
    fin_by_abbr: Dict[str, pd.Series] = {}
    fin_by_id: Dict[int, pd.Series] = {}
    for _, row in fin_df.iterrows():
        abbr = row.get("__abbr")
        tid = row.get("__team_id")
        if pd.notna(abbr):
            fin_by_abbr[str(abbr)] = row
        if pd.notna(tid):
            fin_by_id[int(tid)] = row

    fan_by_abbr: Dict[str, pd.Series] = {}
    fan_by_id: Dict[int, pd.Series] = {}
    if not fan_df.empty:
        if "team_abbr" in fan_df.columns:
            fan_df["__abbr"] = fan_df["team_abbr"].astype(str).str.strip().str.upper()
        else:
            fan_df["__abbr"] = pd.NA
        if "team_id" in fan_df.columns:
            fan_df["__team_id"] = pd.to_numeric(fan_df["team_id"], errors="coerce").astype("Int64")
        else:
            fan_df["__team_id"] = pd.Series(dtype="Int64")
        for _, r in fan_df.iterrows():
            abbr = r.get("__abbr")
            tid = r.get("__team_id")
            if pd.notna(abbr):
                fan_by_abbr[str(abbr)] = r
            if pd.notna(tid):
                fan_by_id[int(tid)] = r

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

    proj_df, proj_sources, proj_notes = load_projected_starters(base, league_id=league_id)
    players_df, player_sources, player_notes = load_players_lookup(base)
    player_name_map = {}
    if not players_df.empty:
        for _, r in players_df.iterrows():
            pid = r.get("player_id")
            name = r.get("player_name")
            if pd.notna(pid):
                player_name_map[int(pid)] = str(name)
    batter_df, batter_sources, batter_notes = parse_batter_profile_all(base)
    team_reporting_df, team_reporting_sources, team_reporting_notes = load_team_reporting(base, league_id=league_id)
    mgr_tend_df, mgr_tend_sources, mgr_tend_notes = load_manager_tendencies(base)

    matchups, featured_df, featured_path = discover_featured_matchups(base)
    explicit = parse_matchups_arg(args.matchups)
    if explicit:
        matchups = explicit

    prob_df, prob_path = choose_probables_source(base, featured_df, featured_path)
    prob_label = prob_path.name if prob_path else "schedule"
    lsdl_df, lsdl_sources, lsdl_notes = load_lsdl_schedule(base)
    schedule_df = lsdl_df if lsdl_df is not None and not lsdl_df.empty else prob_df
    schedule_label = lsdl_sources[0] if lsdl_sources else prob_label
    pitch_df, pitch_path = load_best_csv(base, PITCH_RATINGS_PATTERNS)
    bat_df, bat_path = load_best_csv(base, BAT_RATINGS_PATTERNS)
    games_path = base / "csv" / "ootp_csv" / "games.csv"
    games_df = None
    if games_path.exists():
        try:
            games_df = pd.read_csv(games_path)
        except Exception:
            games_df = None

    abbr_to_id = {}
    if "team_abbr" in teams.columns and "team_id" in teams.columns:
        for _, row in teams.iterrows():
            abbr = row.get("team_abbr")
            tid = row.get("team_id")
            if pd.notna(abbr) and pd.notna(tid):
                abbr_to_id[str(abbr).upper()] = int(tid)

    proj_by_teamid = {}
    proj_by_abbr = {}
    if not proj_df.empty:
        for _, row in proj_df.iterrows():
            tid = row.get("team_id")
            pid = row.get("player_id")
            tabbr = row.get("team_abbr")
            if pd.notna(tid):
                proj_by_teamid[int(tid)] = pid
            if pd.notna(tabbr):
                proj_by_abbr[str(tabbr).upper()] = pid

    def resolve_starter(away_abbr: str, home_abbr: str) -> tuple[dict, dict]:
        away = {"id": None, "name": None, "source": None}
        home = {"id": None, "name": None, "source": None}
        away_tid = abbr_to_id.get(away_abbr)
        home_tid = abbr_to_id.get(home_abbr)
        if prob_df is not None and not prob_df.empty:
            sched_away, sched_home = describe_probables(prob_df, away_abbr, home_abbr, abbr_to_id)
            for entry, sched in ((away, sched_away), (home, sched_home)):
                if not sched:
                    continue
                pid = sched.get("player_id")
                name = sched.get("name")
                if pd.notna(pid):
                    try:
                        entry["id"] = int(pid)
                    except Exception:
                        entry["id"] = pid
                if name:
                    entry["name"] = str(name).strip()
                entry["source"] = prob_label
        if games_df is not None and away_tid and home_tid:
            match_rows = games_df[(games_df["away_team"] == away_tid) & (games_df["home_team"] == home_tid)]
            if not match_rows.empty:
                row = match_rows.iloc[-1]
                a_id = pd.to_numeric(pd.Series([row.get("starter0")]), errors="coerce").iloc[0]
                h_id = pd.to_numeric(pd.Series([row.get("starter1")]), errors="coerce").iloc[0]
                if away["id"] is None and pd.notna(a_id) and a_id > 0:
                    away["id"] = int(a_id)
                    if not away["source"]:
                        away["source"] = "games.csv"
                if home["id"] is None and pd.notna(h_id) and h_id > 0:
                    home["id"] = int(h_id)
                    if not home["source"]:
                        home["source"] = "games.csv"
        if away["id"] is None:
            pid = None
            if away_tid and away_tid in proj_by_teamid:
                pid = proj_by_teamid.get(away_tid)
            elif away_abbr in proj_by_abbr:
                pid = proj_by_abbr.get(away_abbr)
            if pid:
                away["id"] = pid
                away["source"] = "projected"
        if home["id"] is None:
            pid = None
            if home_tid and home_tid in proj_by_teamid:
                pid = proj_by_teamid.get(home_tid)
            elif home_abbr in proj_by_abbr:
                pid = proj_by_abbr.get(home_abbr)
            if pid:
                home["id"] = pid
                home["source"] = "projected"
        for entry in (away, home):
            pid = entry.get("id")
            if pid is not None and pid in player_name_map:
                entry["name"] = player_name_map[pid]
            elif pid is not None:
                entry["name"] = f"Player {pid}"
        return away, home

    fan_tiers: Dict[str, str] = {}

    batter_by_team: Dict[str, pd.DataFrame] = {}
    if not batter_df.empty and "team_abbr" in batter_df.columns:
        for team, group in batter_df.groupby(batter_df["team_abbr"].str.upper()):
            batter_by_team[team] = group.copy()

    def ballpark_env(team_key: str) -> str:
        park = park_map.get(team_key, {})
        factors = park.get("factors", {})
        pf_avg = None
        pf_hr = None
        for col, val in factors.items():
            low = str(col).lower().strip()
            norm = low.replace("_", " ")
            num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            if pd.isna(num):
                continue
            if pf_avg is None and (
                "pf avg" in norm or "pf runs" in norm or norm == "pf" or norm == "avg"
            ):
                pf_avg = num
            if pf_hr is None and (
                "pf hr" in norm or norm == "hr"
            ):
                pf_hr = num
        if pf_avg is None or pf_hr is None or pd.isna(pf_avg) or pd.isna(pf_hr):
            return "N/A"
        if pf_avg >= 1.05 or pf_hr >= 1.05:
            return f"Hitter (AVG {pf_avg}, HR {pf_hr})"
        if pf_avg <= 0.95 and pf_hr <= 0.95:
            return f"Pitcher (AVG {pf_avg}, HR {pf_hr})"
        return f"Neutral (AVG {pf_avg}, HR {pf_hr})"

    def lookup_team_reporting(abbr: str) -> dict:
        if team_reporting_df is None or team_reporting_df.empty:
            return {}
        row = team_reporting_df[team_reporting_df["team_abbr"].str.upper() == abbr]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    season_label = season if season is not None else "N/A"
    week_label = f"{week:02d}" if isinstance(week, int) else ("N/A" if week is None else str(week))
    md_lines = [f"# ABL Pregame Pack - Season {season_label} Week {week_label}", ""]

    if not matchups:
        md_lines.append("No featured matchups artifact found; use --matchups to provide pairs like CHI@MIA.")
    else:
        # League dash
        md_lines.insert(0, "")
        md_lines.insert(0, f"Arsenal top: {arsenal_top}, Bats top: {bats_top}")
        if not fin_df.empty:
            md_lines.insert(0, "League Pregame Dash (see boards below)")
        for away_abbr, home_abbr in matchups:
            md_lines.append(f"## {away_abbr} at {home_abbr}")
            away_rep = lookup_team_reporting(away_abbr)
            home_rep = lookup_team_reporting(home_abbr)
            def banner(rep: dict) -> str:
                if not rep:
                    return "N/A"
                rec = f"{int(rep.get('wins',0))}-{int(rep.get('losses',0))}"
                rank = rep.get("division_rank", "N/A")
                gb = rep.get("games_back", "N/A")
                return f"{rep.get('team_abbr','TEAM')} ({rec}, rank {rank}, GB {gb})"
            md_lines.append(f"{banner(away_rep)} @ {banner(home_rep)}")
            away_key = abbr_to_key.get(away_abbr, "")
            home_key = abbr_to_key.get(home_abbr, "")
            park_env = ballpark_env(home_key or away_key)
            park_name = park_map.get(home_key, {}).get("name") or park_map.get(away_key, {}).get("name") or "N/A"
            md_lines.append(f"Ballpark: {park_name} | Park Env: {park_env}")
            if schedule_df is not None and not schedule_df.empty:
                if not matchup_in_schedule(schedule_df, away_abbr, home_abbr, abbr_to_id):
                    md_lines.append(f"Schedule: NOT FOUND in {schedule_label}")

            away_prob, home_prob = resolve_starter(away_abbr, home_abbr)
            away_name = away_prob.get("name") or "TBD"
            home_name = home_prob.get("name") or "TBD"
            away_src = away_prob.get("source") or "unknown"
            home_src = home_prob.get("source") or "unknown"
            md_lines.append(f"Probable Starters: {away_abbr} {away_name} ({away_src}) vs {home_abbr} {home_name} ({home_src})")

            # Managers
            md_lines.append("### Managers")
            def manager_line(rep: dict, abbr: str) -> str:
                name = rep.get("manager_name") if rep else None
                wins = rep.get("manager_career_wins") if rep else None
                losses = rep.get("manager_career_losses") if rep else None
                titles = rep.get("manager_titles") if rep else None
                base = name or "N/A"
                extras = []
                if pd.notna(wins) and pd.notna(losses):
                    try:
                        pct = float(wins) / max(float(wins) + float(losses), 1)
                        extras.append(f"{int(wins)}-{int(losses)} ({pct:.3f})")
                    except Exception:
                        extras.append(f"{wins}-{losses}")
                if pd.notna(titles):
                    extras.append(f"Titles {titles}")
                tend_row = mgr_tend_df[mgr_tend_df["team_abbr"].astype(str).str.upper() == abbr] if not mgr_tend_df.empty else pd.DataFrame()
                if not tend_row.empty:
                    extras.append("Tendencies: " + format_manager_tendencies(tend_row.iloc[0]))
                else:
                    extras.append("Tendencies: N/A")
                return base + (" | " + " | ".join(extras) if extras else "")
            md_lines.append(f"{away_abbr}: {manager_line(away_rep, away_abbr)}")
            md_lines.append(f"{home_abbr}: {manager_line(home_rep, home_abbr)}")

            # Finances/Fans
            away_tid = abbr_to_id.get(away_abbr)
            home_tid = abbr_to_id.get(home_abbr)
            for line in describe_team_detail(away_abbr, away_tid, away_key, park_map, fan_by_abbr, fan_by_id, fin_by_abbr, fin_by_id):
                md_lines.append(f"{away_abbr} {line}")
            for line in describe_team_detail(home_abbr, home_tid, home_key, park_map, fan_by_abbr, fan_by_id, fin_by_abbr, fin_by_id):
                md_lines.append(f"{home_abbr} {line}")

            # Key Bats
            md_lines.append("### Key Bats")
            def bats_for(team_abbr: str) -> List[str]:
                df_team = batter_by_team.get(team_abbr.upper())
                if df_team is None or df_team.empty:
                    return []
                df_team = df_team.copy()
                if "overall_bat" in df_team.columns:
                    df_team["__overall"] = pd.to_numeric(df_team["overall_bat"], errors="coerce")
                    df_team = df_team.sort_values("__overall", ascending=False)
                lines = []
                for _, rec in df_team.head(bats_top).iterrows():
                    name = rec.get("player_name", "N/A")
                    pos = rec.get("pos", "")
                    bats_hand = rec.get("bats", "")
                    best_tool = rec.get("best_tool_name")
                    best_val = rec.get("best_tool_value")
                    hook = f"Best: {best_tool} {best_val}" if pd.notna(best_tool) and pd.notna(best_val) else ""
                    overall = rec.get("overall_bat")
                    parts = [f"{name} ({pos})" if pos else name]
                    if pd.notna(overall):
                        parts.append(f"Bat {overall}")
                    if hook:
                        parts.append(hook)
                    if bats_hand:
                        parts.append(f"Bats: {bats_hand}")
                    lines.append(" — ".join(parts))
                return lines
            def fallback_bats(team_abbr: str) -> str:
                fallback = describe_key_bats(bat_df, team_abbr)
                if fallback == "Key Bats: N/A":
                    return ""
                return fallback.replace("Key Bats: ", "")

            away_bats = bats_for(away_abbr)
            home_bats = bats_for(home_abbr)
            away_line = "; ".join(away_bats) if away_bats else fallback_bats(away_abbr)
            home_line = "; ".join(home_bats) if home_bats else fallback_bats(home_abbr)
            md_lines.append(f"{away_abbr}: " + (away_line if away_line else "N/A (no batter profile or ratings source found)"))
            md_lines.append(f"{home_abbr}: " + (home_line if home_line else "N/A (no batter profile or ratings source found)"))

            # Pitching snapshot
            md_lines.append("### Pitching Snapshot")
            def arsenal_line(row: Optional[pd.Series]) -> str:
                if row is None:
                    return "Arsenal: N/A (no repertoire source found)"
                return format_arsenal_display(row, top_k=arsenal_top, show_count=show_ars_count)
            def arsenal_row_for(pid, name, abbr):
                row = None
                try:
                    if pid is not None and pid in arsenal_by_id:
                        row = arsenal_by_id.get(int(pid))
                except Exception:
                    row = None
                if row is None and name:
                    key = str(name).strip().upper()
                    row = arsenal_by_name_team.get((key, abbr.upper()), None)
                return row
            away_row = arsenal_row_for(away_prob.get("id"), away_name, away_abbr)
            home_row = arsenal_row_for(home_prob.get("id"), home_name, home_abbr)
            md_lines.append(f"{away_abbr} starter: {away_name}")
            md_lines.append(arsenal_line(away_row))
            best, note = find_best_pitch_for_pitcher(arsenal_df, away_abbr, away_name)
            md_lines.append(f"Best pitch: {best if best else 'N/A'}" + (f" ({note})" if note else ""))
            md_lines.append(f"{home_abbr} starter: {home_name}")
            md_lines.append(arsenal_line(home_row))
            best, note = find_best_pitch_for_pitcher(arsenal_df, home_abbr, home_name)
            md_lines.append(f"Best pitch: {best if best else 'N/A'}" + (f" ({note})" if note else ""))
            if arsenal_sources:
                md_lines.append("Arsenal sources: " + "; ".join(arsenal_sources))
            else:
                md_lines.append("Arsenal sources: none found")
            md_lines.append("Starter source: games.csv starters where present; otherwise projected_starting_pitchers.csv")
            md_lines.append("")

    data_sources: List[str] = []
    for path in [team_path, park_path, featured_path, prob_path, pitch_path, bat_path]:
        if path:
            data_sources.append(str(path))
    data_sources.extend(fin_sources)
    data_sources.extend(str(p) for p in fan_sources)
    data_sources.extend(lsdl_sources)
    data_sources.extend(batter_sources)
    data_sources.extend(arsenal_sources)
    md_lines.append("## Data Sources")
    if data_sources:
        for src in data_sources:
            md_lines.append(f"- {src}")
    else:
        md_lines.append("- None found")
    md_lines.append("")
    md_lines.append("## Notes")
    notes = park_notes + fin_notes + fan_notes + lsdl_notes + batter_notes
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
