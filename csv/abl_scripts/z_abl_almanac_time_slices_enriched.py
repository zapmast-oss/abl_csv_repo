import argparse
from pathlib import Path

import pandas as pd


def load_dim_team_park(path: Path) -> pd.DataFrame:
    print(f"[DEBUG] dim_team_park={path}")
    if not path.exists():
        raise FileNotFoundError(f"dim_team_park not found: {path}")
    dim = pd.read_csv(path)
    print(f"[INFO] Loaded {len(dim)} rows from dim_team_park")

    # Required core columns; we do NOT demand team_name here.
    required = ["ID", "Abbr", "SL", "DIV"]
    missing = [c for c in required if c not in dim.columns]
    if missing:
        raise KeyError(f"dim_team_park is missing required columns: {missing}")
    # Derive helper columns to improve matching
    dim = dim.copy()
    dim["city_simple"] = dim["City"].astype(str).str.split("(").str[0].str.strip()
    dim["team_city_from_team_name"] = dim["Team Name"].astype(str).apply(
        lambda x: " ".join(x.split(" ")[:-1]).strip() if len(x.split(" ")) > 1 else x
    )
    return dim


def _short(label: str) -> str:
    return (
        label.replace("team_", "")
        .replace("_summary", "")
        .replace("series_", "series")
    )


def _normalize(val: object) -> str:
    if pd.isna(val):
        return ""
    txt = str(val).lower().strip()
    txt = txt.replace("seatlle", "seattle")
    if "(" in txt:
        txt = txt.split("(")[0].strip()
    for ch in [",", ".", "-", "_"]:
        txt = txt.replace(ch, " ")
    txt = " ".join(txt.split())
    return txt


def _find_name_and_abbr_cols(df: pd.DataFrame, label: str, dim: bool = False):
    """
    Inspect df columns and guess the 'name' and 'abbr' columns.
    Returns (name_cols, abbr_cols) as lists of column names.
    """
    lower_map = {c.lower(): c for c in df.columns}

    name_keys = ["team_name", "team", "name", "team name", "city", "team city"]
    if dim:
        name_keys.extend(["team name", "city_simple", "team_city_from_team_name"])

    name_candidates = []
    for key in name_keys:
        if key in lower_map:
            name_candidates.append(lower_map[key])

    abbr_candidates = []
    for key in ("team_abbr", "abbr", "team code", "code"):
        if key in lower_map:
            abbr_candidates.append(lower_map[key])

    print(
        f"[DEBUG] {_short(label)} name_candidates={name_candidates}, "
        f"abbr_candidates={abbr_candidates}"
    )
    return name_candidates, abbr_candidates


def _best_dim_column_by_overlap(
    dim: pd.DataFrame,
    df_values,
    candidate_cols,
    label: str,
):
    """
    Among candidate dim columns, pick the one whose values overlap
    most with df_values. Returns (best_col, best_hits).
    """
    df_set = set([_normalize(v) for v in df_values if pd.notna(v)])
    best_col = None
    best_hits = 0

    for col in candidate_cols:
        dim_vals = set([_normalize(v) for v in dim[col].dropna().unique()])
        hits = len(df_set & dim_vals)
        print(f"[DEBUG] {label}: overlap with dim[{col}] = {hits}")
        if hits > best_hits:
            best_hits = hits
            best_col = col

    return best_col, best_hits


def _choose_join_keys(dim: pd.DataFrame, df: pd.DataFrame, label: str):
    """
    Decide how to join df to dim_team_park based on actual value overlap.

    Priority:
      1) name-based join using the dim name column that overlaps most
      2) abbr-based join using the dim abbr column that overlaps most

    Returns (left_key, right_key, mode) where mode is 'name' or 'abbr'.
    """
    dim_name_cols, dim_abbr_cols = _find_name_and_abbr_cols(dim, "dim_team_park", dim=True)
    df_name_cols, df_abbr_cols = _find_name_and_abbr_cols(df, label)

    # 1) try name-based join with overlap scoring
    if dim_name_cols and df_name_cols:
        left_key = df_name_cols[0]
        df_vals = df[left_key].dropna().unique()
        best_right, best_hits = _best_dim_column_by_overlap(
            dim, df_vals, dim_name_cols, f"{label} NAME-join"
        )
        if best_right is not None and best_hits > 0:
            print(
                f"[INFO] {label}: joining to dim_team_park by NAME "
                f"({left_key} -> {best_right}), hits={best_hits}"
            )
            return left_key, best_right, "name"
        else:
            print(
                f"[WARN] {label}: no meaningful overlap on name columns; "
                f"falling back to ABBR logic if available."
            )

    # 2) abbr-based join with overlap scoring
    if dim_abbr_cols and df_abbr_cols:
        left_key = df_abbr_cols[0]
        df_vals = df[left_key].dropna().unique()
        best_right, best_hits = _best_dim_column_by_overlap(
            dim, df_vals, dim_abbr_cols, f"{label} ABBR-join"
        )
        if best_right is not None and best_hits > 0:
            print(
                f"[INFO] {label}: joining to dim_team_park by ABBR "
                f"({left_key} -> {best_right}), hits={best_hits}"
            )
            return left_key, best_right, "abbr"
        else:
            print(
                f"[WARN] {label}: no meaningful overlap on abbr columns either."
            )

    # if we get here, we failed
    raise RuntimeError(
        f"{label}: could not determine join keys with non-zero overlap. "
        f"df columns={list(df.columns)}, dim columns={list(dim.columns)}"
    )


def load_team_monthly(almanac_root: Path, season: int, league_id: int) -> pd.DataFrame:
    p = almanac_root / str(season) / f"team_monthly_summary_{season}_league{league_id}.csv"
    print(f"[DEBUG] team_monthly_path={p}")
    if not p.exists():
        print(f"[WARN] Monthly summary not found: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p)
    print(f"[INFO] Loaded {len(df)} rows from team_monthly")
    return df


def load_team_weekly(almanac_root: Path, season: int, league_id: int) -> pd.DataFrame:
    p = almanac_root / str(season) / f"team_weekly_summary_{season}_league{league_id}.csv"
    print(f"[DEBUG] team_weekly_path={p}")
    if not p.exists():
        print(f"[WARN] Weekly summary not found: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p)
    print(f"[INFO] Loaded {len(df)} rows from team_weekly")
    return df


def load_series_summary(almanac_root: Path, season: int, league_id: int) -> pd.DataFrame:
    p = almanac_root / str(season) / f"series_summary_{season}_league{league_id}.csv"
    print(f"[DEBUG] series_summary_path={p}")
    if not p.exists():
        print(f"[WARN] Series summary not found: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p)
    print(f"[INFO] Loaded {len(df)} rows from series_summary")
    return df


def enrich_team_table(df: pd.DataFrame, dim: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Enrich a team-based table (monthly or weekly) with ID/abbr/conference/division.

    It will:
      * drop conference/league header rows if they show up as "team_name"
      * pick join keys automatically (name or abbr, based on overlap)
      * merge dim_team_park onto df
      * expose team_id/team_abbr/conference/division
    """
    if df.empty:
        print(f"[WARN] {label} table is empty, skipping enrichment.")
        return df

    # Drop non-team rows that sneak in as "teams"
    if "team_name" in df.columns:
        banned = ["American Baseball Conference", "National Baseball Conference"]
        before_rows = len(df)
        df = df[~df["team_name"].isin(banned)].copy()
        dropped = before_rows - len(df)
        if dropped > 0:
            print(f"[INFO] {label}: dropped {dropped} non-team rows by team_name filter")

    left_key, right_key, mode = _choose_join_keys(dim, df, label)

    # always keep these from dim
    dim_small = dim[[right_key, "ID", "Abbr", "SL", "DIV"]].drop_duplicates()

    left_norm = df[left_key].apply(_normalize)
    right_norm = dim_small[right_key].apply(_normalize)
    dim_small = dim_small.assign(__join_key=right_norm)
    df = df.assign(__join_key=left_norm)

    before = len(df)
    merged = df.merge(
        dim_small,
        on="__join_key",
        how="left",
        validate="many_to_one",
    )
    after = len(merged)

    if before != after:
        raise RuntimeError(f"{label}: row-count changed during join ({before} -> {after}).")

    missing = merged["ID"].isna().sum()
    if missing:
        bad = merged[merged["ID"].isna()][left_key].unique()
        raise RuntimeError(f"{label}: {missing} rows failed to match dim_team_park. Missing keys: {bad}")

    # Rename ID/Abbr/SL/DIV to canonical names
    merged = merged.rename(
        columns={
            "ID": "team_id",
            "Abbr": "team_abbr",
            "SL": "conference",
            "DIV": "division",
        }
    )

    # Drop the right_key helper if it is different from left_key
    merged = merged.drop(columns=["__join_key"])

    print(f"[OK] Enriched {label} table with team_id / conference / division via {mode.upper()} join")
    return merged


def enrich_series_table(series: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich series_summary with home/away team IDs and conference/division.

    We attempt a name-based join using overlap, otherwise fall back to abbr.
    """
    if series.empty:
        print("[WARN] series_summary is empty, skipping enrichment.")
        return series

    banned = {"American Baseball Conference", "National Baseball Conference"}
    before_rows = len(series)
    series = series[
        ~series.get("home_team", series.get("home", pd.Series(dtype=object))).astype(str).isin(banned)
    ].copy()
    if "away_team" in series.columns:
        series = series[~series["away_team"].astype(str).isin(banned)]
    dropped = before_rows - len(series)
    if dropped > 0:
        print(f"[INFO] series_summary: dropped {dropped} conference summary rows")

    # figure out what the series table calls its team columns
    lower = {c.lower(): c for c in series.columns}
    home_key = None
    away_key = None
    for key in ("home_team", "home name", "home"):
        if key in lower:
            home_key = lower[key]
            break
    for key in ("away_team", "away name", "away"):
        if key in lower:
            away_key = lower[key]
            break

    if away_key is None and "pair_key" in series.columns:
        # derive away from pair_key if not present
        series = series.copy()
        series["away_team"] = series["pair_key"].astype(str).apply(
            lambda x: x.split("/")[-1].strip() if "/" in x else ""
        )
        lower = {c.lower(): c for c in series.columns}
        away_key = lower.get("away_team")

    if home_key is None or away_key is None:
        raise KeyError(f"series_summary is missing home/away team columns. columns={list(series.columns)}")

    dim_name_cols, dim_abbr_cols = _find_name_and_abbr_cols(dim, "dim_team_park", dim=True)

    # Combine home + away values for overlap test
    combined_vals = pd.concat(
        [series[home_key].dropna(), series[away_key].dropna()]
    ).unique()

    # 1) try name-based for series
    best_right_name, best_hits_name = (None, 0)
    if dim_name_cols:
        best_right_name, best_hits_name = _best_dim_column_by_overlap(
            dim, combined_vals, dim_name_cols, "series_summary NAME-join"
        )

    # 2) try abbr-based for series
    best_right_abbr, best_hits_abbr = (None, 0)
    if dim_abbr_cols:
        best_right_abbr, best_hits_abbr = _best_dim_column_by_overlap(
            dim, combined_vals, dim_abbr_cols, "series_summary ABBR-join"
        )

    mode = None
    if best_hits_name > 0 or best_hits_abbr > 0:
        if best_hits_name >= best_hits_abbr:
            # name-based
            right_key = best_right_name
            print(
                f"[INFO] series_summary: joining by NAME using dim[{right_key}], hits={best_hits_name}"
            )
            dim_slice = dim[[right_key, "ID", "Abbr", "SL", "DIV"]].drop_duplicates()
            dim_slice["__join_key"] = dim_slice[right_key].apply(_normalize)
            series["__home_key"] = series[home_key].apply(_normalize)
            series["__away_key"] = series[away_key].apply(_normalize)
            home_dim = dim_slice.rename(
                columns={
                    "__join_key": "__home_key",
                    "ID": "home_team_id",
                    "Abbr": "home_team_abbr",
                    "SL": "home_conference",
                    "DIV": "home_division",
                }
            )
            away_dim = dim_slice.rename(
                columns={
                    "__join_key": "__away_key",
                    "ID": "away_team_id",
                    "Abbr": "away_team_abbr",
                    "SL": "away_conference",
                    "DIV": "away_division",
                }
            )
            mode = "name"
        else:
            # abbr-based
            right_key = best_right_abbr
            print(
                f"[INFO] series_summary: joining by ABBR using dim[{right_key}], hits={best_hits_abbr}"
            )
            dim_slice = dim[[right_key, "ID", "Abbr", "SL", "DIV"]].drop_duplicates()
            dim_slice["__join_key"] = dim_slice[right_key].apply(_normalize)
            series["__home_key"] = series[home_key].apply(_normalize)
            series["__away_key"] = series[away_key].apply(_normalize)
            home_dim = dim_slice.rename(
                columns={
                    "__join_key": "__home_key",
                    "ID": "home_team_id",
                    "Abbr": "home_team_abbr",
                    "SL": "home_conference",
                    "DIV": "home_division",
                }
            )
            away_dim = dim_slice.rename(
                columns={
                    "__join_key": "__away_key",
                    "ID": "away_team_id",
                    "Abbr": "away_team_abbr",
                    "SL": "away_conference",
                    "DIV": "away_division",
                }
            )
            mode = "abbr"
    else:
        raise RuntimeError(
            "series_summary: could not find any dim column with non-zero overlap "
            "for home/away teams."
        )

    before = len(series)
    merged = series.merge(home_dim, on="__home_key", how="left", validate="many_to_one")
    merged = merged.merge(away_dim, on="__away_key", how="left", validate="many_to_one")
    merged = merged.drop(columns=["__home_key", "__away_key"])
    after = len(merged)

    if before != after:
        raise RuntimeError(f"series_summary: row-count changed during join ({before} -> {after}).")

    missing_home = merged["home_team_id"].isna().sum()
    missing_away = merged["away_team_id"].isna().sum()
    if missing_home or missing_away:
        bad_home = merged[merged["home_team_id"].isna()][home_key].unique()
        bad_away = merged[merged["away_team_id"].isna()][away_key].unique()
        raise RuntimeError(
            f"series_summary: unmatched teams "
            f"(home missing={missing_home}, away missing={missing_away}). "
            f"Bad home keys: {bad_home}, bad away keys: {bad_away}"
        )

    print(
        f"[OK] Enriched series_summary with home/away team IDs and conference/division "
        f"via {mode.upper()} join"
    )
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Enrich almanac time-slice tables (monthly/weekly/series) with dim_team_park info."
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g. 1972).",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        default=200,
        help="League ID (ABL = 200). Default: 200.",
    )
    parser.add_argument(
        "--almanac-root",
        default="csv/out/almanac",
        help="Root folder where almanac-derived CSVs live (default: csv/out/almanac).",
    )
    parser.add_argument(
        "--dim-team-park",
        default="csv/out/star_schema/dim_team_park.csv",
        help="Path to dim_team_park.csv (default: csv/out/star_schema/dim_team_park.csv).",
    )

    args = parser.parse_args(argv)

    almanac_root = Path(args.almanac_root)
    season_dir = almanac_root / str(args.season)
    season_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] season={args.season}, league_id={args.league_id}")
    print(f"[DEBUG] almanac_root={almanac_root}")
    print(f"[DEBUG] dim_team_park={args.dim_team_park}")

    dim = load_dim_team_park(Path(args.dim_team_park))

    monthly = load_team_monthly(almanac_root, args.season, args.league_id)
    weekly = load_team_weekly(almanac_root, args.season, args.league_id)
    series = load_series_summary(almanac_root, args.season, args.league_id)

    if not monthly.empty:
        monthly_enriched = enrich_team_table(monthly, dim, label="team_monthly")
        out_p = season_dir / f"team_monthly_summary_{args.season}_league{args.league_id}_enriched.csv"
        monthly_enriched.to_csv(out_p, index=False)
        print(f"[OK] Wrote enriched monthly summary to {out_p}")

    if not weekly.empty:
        weekly_enriched = enrich_team_table(weekly, dim, label="team_weekly")
        out_p = season_dir / f"team_weekly_summary_{args.season}_league{args.league_id}_enriched.csv"
        weekly_enriched.to_csv(out_p, index=False)
        print(f"[OK] Wrote enriched weekly summary to {out_p}")

    if not series.empty:
        series_enriched = enrich_series_table(series, dim)
        out_p = season_dir / f"series_summary_{args.season}_league{args.league_id}_enriched.csv"
        series_enriched.to_csv(out_p, index=False)
        print(f"[OK] Wrote enriched series summary to {out_p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
