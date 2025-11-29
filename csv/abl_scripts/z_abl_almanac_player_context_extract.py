#!/usr/bin/env python
"""
Extract player-context tables (top players, top games, preseason, positional strength,
financials, transactions, prospects) from almanac HTML for a season/league.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def norm_key(val: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(val).strip().lower())


def read_tables(html_path: Path) -> List[pd.DataFrame]:
    try:
        return pd.read_html(html_path, flavor="lxml")
    except ValueError:
        return []


def extract_player_ids(html_path: Path, rows: int) -> List[Optional[int]]:
    try:
        text = html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return [None] * rows
    matches = re.findall(r"player_id=(\d+)", text)
    ids: List[Optional[int]] = []
    for i in range(rows):
        ids.append(int(matches[i]) if i < len(matches) else None)
    return ids


def load_dim_team() -> pd.DataFrame:
    dim = pd.read_csv("csv/out/star_schema/dim_team_park.csv")
    team_id_col = "team_id" if "team_id" in dim.columns else "ID"
    abbr_col = "team_abbr" if "team_abbr" in dim.columns else "Abbr"
    name_col = "team_name" if "team_name" in dim.columns else "Team Name"
    city_col = "City" if "City" in dim.columns else name_col
    conf_col = "conf" if "conf" in dim.columns else ("SL" if "SL" in dim.columns else None)
    div_col = "division" if "division" in dim.columns else ("DIV" if "DIV" in dim.columns else None)
    df = dim[[team_id_col, abbr_col, name_col, city_col]].copy()
    df = df.rename(columns={team_id_col: "team_id", abbr_col: "team_abbr", name_col: "team_name", city_col: "team_city"})
    if conf_col:
        df["conf"] = dim[conf_col]
    if div_col:
        df["division"] = dim[div_col]
    df["team_key"] = df["team_city"].str.split("(").str[0].map(norm_key)
    return df


def load_dim_player() -> pd.DataFrame:
    dim = pd.read_csv("csv/out/star_schema/dim_player_profile.csv")
    pid_col = "player_id" if "player_id" in dim.columns else "ID"
    name_col = "player_name" if "player_name" in dim.columns else ("full_name" if "full_name" in dim.columns else "Name")
    cols = [pid_col, name_col]
    for c in ["bats", "throws", "primary_position", "team_id"]:
        if c in dim.columns:
            cols.append(c)
    df = dim[cols].copy()
    df = df.rename(columns={pid_col: "player_id", name_col: "player_name_dim"})
    return df


def join_team(df: pd.DataFrame, team_dim: pd.DataFrame) -> pd.DataFrame:
    if "team_name" in df.columns:
        df["team_key"] = df["team_name"].map(norm_key)
        df = df.merge(team_dim, on="team_key", how="left")
    return df


def join_player(df: pd.DataFrame, player_dim: pd.DataFrame) -> pd.DataFrame:
    if "player_name" not in df.columns:
        df["player_name"] = pd.NA
    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
        player_dim = player_dim.copy()
        player_dim["player_id"] = pd.to_numeric(player_dim["player_id"], errors="coerce").astype("Int64")
        df = df.merge(player_dim, on="player_id", how="left")
        if "player_name_dim" in df.columns:
            df["player_name"] = df["player_name"].fillna(df["player_name_dim"])
            df = df.drop(columns=["player_name_dim"])
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass
    return df


def cols_lower(df: pd.DataFrame) -> List[str]:
    return [str(c).lower() for c in df.columns]


def pick_table(tables: List[pd.DataFrame], predicate: Callable[[pd.DataFrame], bool]) -> Optional[pd.DataFrame]:
    for t in tables:
        if predicate(t):
            return t
    return None


def parse_table_generic(html_path: Path, predicate: Callable[[pd.DataFrame], bool], rename_map=None) -> pd.DataFrame:
    tables = read_tables(html_path)
    target = pick_table(tables, predicate)
    if target is None:
        log(f"[WARN] Could not identify table in {html_path}")
        return pd.DataFrame()
    if rename_map:
        target = target.rename(columns=rename_map)
    target = target.dropna(how="all")
    target = clean_numeric(target)
    if "player_id" not in target.columns:
        target["player_id"] = extract_player_ids(html_path, len(target))
    return target


def parse_top_players(html_path: Path) -> pd.DataFrame:
    return parse_table_generic(
        html_path,
        lambda t: any("name" in s or "player" in s for s in cols_lower(t)) and any(
            k in cols_lower(t) for k in ["avg", "hr", "ops", "rbi", "era"]
        ),
        rename_map={"Player": "player_name", "Team": "team_name"},
    )


def parse_top_games(html_path: Path) -> pd.DataFrame:
    return parse_table_generic(
        html_path,
        lambda t: "player" in " ".join(cols_lower(t)) and "date" in " ".join(cols_lower(t)),
        rename_map={"Player": "player_name", "Team": "team_name"},
    )


def parse_preseason(html_path: Path) -> pd.DataFrame:
    tables = read_tables(html_path)
    frames = []
    section = 1
    for t in tables:
        flat_cols = []
        for c in t.columns:
            if isinstance(c, tuple):
                parts = [str(x) for x in c if str(x) not in ("nan", "None")]
                flat_cols.append(" ".join(parts) if parts else "")
            else:
                flat_cols.append(str(c))
        t.columns = flat_cols
        lower = [c.lower() for c in flat_cols]
        if "team" in lower and "w" in lower and "l" in lower:
            tmp = t.rename(columns={"Team": "team_name"})
            tmp["section"] = section
            section += 1
            frames.append(tmp)
    if not frames:
        log(f"[WARN] Could not identify preseason predictions table in {html_path}")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = clean_numeric(df)
    return df


def parse_pos_strength_teams(html_path: Path) -> pd.DataFrame:
    return parse_table_generic(
        html_path,
        lambda t: any("team" in s for s in cols_lower(t)),
        rename_map={"Team": "team_name"},
    )


def parse_pos_strength_positions(html_path: Path) -> pd.DataFrame:
    tables = read_tables(html_path)
    blocks = [t for t in tables if list(t.columns) == ["Team", "Top player", "Team ranking", "Top prospect", "Organizational ranking", "Overall ranking"]]
    if not blocks:
        log(f"[WARN] Could not identify positional_strength_positions table in {html_path}")
        return pd.DataFrame()
    positions = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "SP", "RP"]
    rows = []
    for pos, tbl in zip(positions, blocks):
        tbl = tbl.rename(columns={"Top player": "player_name", "Team": "team_name"})
        tbl["position"] = pos
        rows.append(tbl)
    df = pd.concat(rows, ignore_index=True)
    df["player_id"] = extract_player_ids(html_path, len(df))
    df = clean_numeric(df)
    return df


def parse_financial(html_path: Path) -> pd.DataFrame:
    return parse_table_generic(
        html_path,
        lambda t: "player" in " ".join(cols_lower(t)) and any(
            k in " ".join(cols_lower(t)) for k in ["salary", "years", "contract"]
        ),
        rename_map={"Player": "player_name", "Team": "team_name"},
    )


def parse_transactions(html_path: Path) -> pd.DataFrame:
    tables = read_tables(html_path)
    if not tables:
        log(f"[WARN] No tables in {html_path}")
        return pd.DataFrame()
    records = []
    current_date = None
    for tbl in tables:
        if tbl.shape[1] != 1:
            continue
        val = str(tbl.iloc[0, 0])
        if "day" in val.lower() and "," in val:
            current_date = val.strip()
            continue
        for item in tbl.iloc[:, 0].dropna():
            desc = str(item).strip()
            if not desc:
                continue
            records.append({"transaction_date": current_date, "description": desc})
    df = pd.DataFrame(records)
    df["player_id"] = extract_player_ids(html_path, len(df))
    return df


def parse_prospects(html_path: Path) -> pd.DataFrame:
    tables = read_tables(html_path)
    targets = [t for t in tables if "#" in t.columns or any("name" in str(c).lower() for c in t.columns)]
    if not targets:
        log(f"[WARN] Could not identify top_prospects table in {html_path}; skipping.")
        return pd.DataFrame()
    frames = []
    for t in targets:
        if "Name" in t.columns:
            t = t.rename(columns={"Name": "player_name", "Team": "team_name"})
        t["player_id"] = extract_player_ids(html_path, len(t))
        frames.append(clean_numeric(t))
    return pd.concat(frames, ignore_index=True)


def write_output(df: pd.DataFrame, out_path: Path, label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log(f"[OK] Wrote {label} to {out_path} ({len(df)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract player context tables from almanac core HTML.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, required=True)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    core_root = Path("csv/in/almanac_core") / str(season) / "leagues"
    if not core_root.exists():
        log(f"[ERROR] Core root not found: {core_root}")
        return 1

    team_dim = load_dim_team()
    player_dim = load_dim_player()

    files = {
        "top_players": core_root / f"league_{league_id}_top_players_page.html",
        "top_games": core_root / f"league_{league_id}_top_game_performances.html",
        "preseason": core_root / f"league_{league_id}_preseason_prediction_report.html",
        "pos_teams": core_root / f"league_{league_id}_positional_strength_overview_teams.html",
        "pos_positions": core_root / f"league_{league_id}_positional_strength_overview_positions.html",
        "financial": core_root / f"league_{league_id}_financial_report.html",
        "transactions": core_root / f"league_{league_id}_transactions_0_0.html",
        "prospects": core_root / f"league_{league_id}_top_prospects.html",
    }

    parsers = {
        "top_players": parse_top_players,
        "top_games": parse_top_games,
        "preseason": parse_preseason,
        "pos_teams": parse_pos_strength_teams,
        "pos_positions": parse_pos_strength_positions,
        "financial": parse_financial,
        "transactions": parse_transactions,
        "prospects": parse_prospects,
    }

    outputs = {
        "top_players": Path(f"csv/out/almanac/{season}/player_top_players_{season}_league{league_id}.csv"),
        "top_games": Path(f"csv/out/almanac/{season}/player_top_game_performances_{season}_league{league_id}.csv"),
        "preseason": Path(f"csv/out/almanac/{season}/preseason_player_predictions_{season}_league{league_id}.csv"),
        "pos_teams": Path(f"csv/out/almanac/{season}/positional_strength_teams_{season}_league{league_id}.csv"),
        "pos_positions": Path(f"csv/out/almanac/{season}/positional_strength_positions_{season}_league{league_id}.csv"),
        "financial": Path(f"csv/out/almanac/{season}/player_financials_{season}_league{league_id}.csv"),
        "transactions": Path(f"csv/out/almanac/{season}/transactions_{season}_league{league_id}.csv"),
        "prospects": Path(f"csv/out/almanac/{season}/player_top_prospects_{season}_league{league_id}.csv"),
    }

    fallback_patterns = {
        "preseason": "preseason",
        "transactions": "transactions",
    }

    for key, html_path in files.items():
        if not html_path.exists() and key in fallback_patterns:
            pattern = fallback_patterns[key]
            matches = list(core_root.glob(f"*{pattern}*{league_id}*.html")) + list(core_root.glob(f"*league_{league_id}*{pattern}*.html"))
            if matches:
                html_path = matches[0]
                files[key] = html_path
        if not html_path.exists():
            log(f"[WARN] Missing HTML for {key}: {html_path}; skipping.")
            write_output(pd.DataFrame(), outputs[key], key)
            continue
        log(f"[INFO] Reading {key} from {html_path}")
        df = parsers[key](html_path)
        if df.empty:
            log(f"[WARN] Parsed zero rows for {key}")
        df["season"] = season
        df["league_id"] = league_id
        df = join_team(df, team_dim)
        df = join_player(df, player_dim)
        write_output(df, outputs[key], key)

    log(f"[INFO] Completed player context extraction for season {season}, league {league_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
