#!/usr/bin/env python
"""
Extract league champions (division winners + wild cards) from the almanac standings
HTML and map them to canonical team metadata via dim_team_park.

Outputs a champions dict for programmatic use and, optionally, a CSV snapshot for
auditing:
  csv/out/almanac/<season>/league_champions_<season>_league<league_id>.csv
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

# Reuse the dim loading / matching logic from the time-slice enricher to stay
# consistent with other joins.
from z_abl_almanac_time_slices_enriched import _choose_join_keys, _normalize, load_dim_team_park  # type: ignore


def _log(msg: str) -> None:
    print(msg, flush=True)


def _short_conf(conf: str) -> str:
    upper = conf.upper()
    if upper.startswith("N") or "NATIONAL" in upper:
        return "NBC"
    if upper.startswith("A") or "AMERICAN" in upper:
        return "ABC"
    return conf


def _find_standings_html_path(zf: zipfile.ZipFile, league_id: int) -> str:
    candidates = [
        name
        for name in zf.namelist()
        if "standings" in name.lower() and f"league_{league_id}" in name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No standings HTML found in zip containing 'standings' and league_{league_id}"
        )
    # Prefer deterministic ordering
    candidates.sort()
    return candidates[0]


def _parse_standings_tables(html_text: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html_text))
    team_tables: list[pd.DataFrame] = []
    for df in tables:
        cols_lower = [str(c).lower() for c in df.columns]
        if not {"team", "w", "l"}.issubset(set(cols_lower)):
            continue
        rename_map = {}
        for col in df.columns:
            lc = str(col).lower()
            if lc == "team":
                rename_map[col] = "team_name"
            elif lc in {"w", "wins"}:
                rename_map[col] = "wins"
            elif lc in {"l", "losses"}:
                rename_map[col] = "losses"
            elif "pct" in lc:
                rename_map[col] = "pct"
            elif "run" in lc and "diff" in lc:
                rename_map[col] = "run_diff"
            elif lc in {"rs", "runs_for"}:
                rename_map[col] = "runs_for"
            elif lc in {"ra", "runs_against"}:
                rename_map[col] = "runs_against"
            elif "m#" in lc or "magic" in lc:
                rename_map[col] = "magic_num"
        df = df.rename(columns=rename_map)
        team_tables.append(df)

    if not team_tables:
        raise RuntimeError("No standings tables with Team/W/L columns were detected.")

    combined = pd.concat(team_tables, ignore_index=True)
    # Keep rows that look like real teams (numeric wins/losses)
    combined["wins"] = pd.to_numeric(combined["wins"], errors="coerce")
    combined["losses"] = pd.to_numeric(combined["losses"], errors="coerce")
    combined = combined.dropna(subset=["wins", "losses"])
    combined = combined.reset_index(drop=True)
    return combined


def _dedupe_by_team(df: pd.DataFrame) -> pd.DataFrame:
    if "team_name" not in df.columns:
        raise KeyError("Expected 'team_name' column after parsing standings tables.")

    def pick(group: pd.DataFrame) -> pd.Series:
        if "magic_num" in group.columns:
            clinched = group["magic_num"].astype(str).str.contains(
                "clinch", case=False, na=False
            )
            if clinched.any():
                return group[clinched].iloc[0]
        return group.iloc[0]

    return df.groupby("team_name", group_keys=False).apply(pick).reset_index(drop=True)


def _attach_dim(
    df: pd.DataFrame, dim: pd.DataFrame, label: str = "standings"
) -> pd.DataFrame:
    if df.empty:
        return df
    left_key, right_key, mode = _choose_join_keys(dim, df, label)
    dim_small = dim[[right_key, "ID", "Abbr", "SL", "DIV"]].drop_duplicates()
    dim_small = dim_small.assign(__join_key=dim_small[right_key].apply(_normalize))
    df = df.assign(__join_key=df[left_key].apply(_normalize))

    merged = df.merge(
        dim_small,
        on="__join_key",
        how="left",
        validate="many_to_one",
    ).drop(columns="__join_key")

    missing = merged["ID"].isna().sum()
    if missing:
        missing_keys = merged[merged["ID"].isna()][left_key].unique()
        raise RuntimeError(
            f"{label}: failed to match dim_team_park for {missing} rows. Missing keys: {missing_keys}"
        )

    merged = merged.rename(
        columns={
            "ID": "team_id",
            "Abbr": "team_abbr",
            "SL": "conference",
            "DIV": "division",
        }
    )
    merged["conference"] = merged["conference"].astype(str).apply(_short_conf)
    merged["division"] = merged["division"].astype(str)
    _log(
        f"[OK] {label}: attached team_id/abbr/conference/division via {mode.upper()} join"
    )
    return merged


def _attach_run_diff_from_summary(
    champions: pd.DataFrame, season: int, league_id: int, repo_root: Path
) -> pd.DataFrame:
    summary_path = (
        repo_root
        / "csv"
        / "out"
        / "almanac"
        / str(season)
        / f"league_season_summary_{season}_league{league_id}.csv"
    )
    if not summary_path.exists():
        _log(f"[WARN] league_season_summary not found at {summary_path}; skipping run_diff enrichment")
        return champions

    summary = pd.read_csv(summary_path)
    if "team_id" not in summary.columns:
        _log("[WARN] league_season_summary missing team_id; cannot attach run_diff")
        return champions

    fields = ["team_id", "runs_for", "runs_against", "run_diff"]
    missing = [f for f in fields if f not in summary.columns]
    if missing:
        _log(f"[WARN] league_season_summary missing columns {missing}; run_diff enrichment limited.")
    summary_small = summary[[c for c in fields if c in summary.columns]].drop_duplicates()

    merged = champions.merge(summary_small, on="team_id", how="left", validate="one_to_one")

    if "run_diff_x" in merged.columns or "run_diff_y" in merged.columns:
        merged["run_diff"] = merged.get("run_diff_x", merged.get("run_diff_y"))
        merged = merged.drop(columns=[c for c in ["run_diff_x", "run_diff_y"] if c in merged.columns])
    if "run_diff" not in merged.columns or merged["run_diff"].isna().all():
        if "runs_for" in merged.columns and "runs_against" in merged.columns:
            merged["run_diff"] = merged["runs_for"] - merged["runs_against"]
    return merged


def _select_division_champs(champs: pd.DataFrame) -> pd.DataFrame:
    required = {"conference", "division"}
    if not required.issubset(set(champs.columns)):
        missing = required - set(champs.columns)
        raise KeyError(f"Champions frame missing required columns: {missing}")

    div_champs: list[pd.Series] = []
    for (conf, div), group in champs.groupby(["conference", "division"]):
        clinched = group
        if clinched.empty:
            raise RuntimeError(f"No clinched team found for {conf} / {div}")
        # Highest wins -> pct as tiebreak
        clinched = clinched.sort_values(
            by=["wins", "pct"], ascending=[False, False]
        ).reset_index(drop=True)
        div_champs.append(clinched.iloc[0])

    return pd.DataFrame(div_champs)


def _select_wildcards(
    champs: pd.DataFrame, div_champs: pd.DataFrame
) -> pd.DataFrame:
    taken_ids = set(div_champs["team_id"].tolist())
    wildcards: list[pd.Series] = []
    for conf, group in champs.groupby("conference"):
        eligible = group[~group["team_id"].isin(taken_ids)].copy()
        if eligible.empty:
            raise RuntimeError(f"No clinched wildcard found for {conf}")
        eligible = eligible.sort_values(by=["wins", "pct"], ascending=[False, False])
        wildcards.append(eligible.iloc[0])
    return pd.DataFrame(wildcards)


def get_league_champions_from_standings(
    season: int,
    league_id: int = 200,
    almanac_zip: Path | None = None,
    dim_team_park_path: Path | None = None,
) -> dict:
    """
    Return champions as:
    {
        "ABC": {"East": {...}, "Central": {...}, "West": {...}, "Wild Card": {...}},
        "NBC": {"East": {...}, "Central": {...}, "West": {...}, "Wild Card": {...}},
    }

    Each champion dict contains:
        {
            "team_id": int,
            "team_name": str,
            "team_abbr": str,
            "conference": str,
            "division": str,
            "wins": int,
            "losses": int,
            "run_diff": int,
        }
    """
    repo_root = Path(__file__).resolve().parents[2]
    if almanac_zip is None:
        almanac_zip = repo_root / "data_raw" / "ootp_html" / f"almanac_{season}.zip"
    if dim_team_park_path is None:
        dim_team_park_path = repo_root / "csv" / "out" / "star_schema" / "dim_team_park.csv"

    if not almanac_zip.exists():
        raise FileNotFoundError(f"Almanac zip not found: {almanac_zip}")
    dim = load_dim_team_park(Path(dim_team_park_path))

    with zipfile.ZipFile(almanac_zip, "r") as zf:
        html_path = _find_standings_html_path(zf, league_id)
        _log(f"[INFO] Using standings HTML: {html_path}")
        html_text = zf.read(html_path).decode("utf-8", errors="ignore")

    parsed = _parse_standings_tables(html_text)
    parsed = _dedupe_by_team(parsed)
    parsed["season"] = season
    parsed["league_id"] = league_id

    enriched = _attach_dim(parsed, dim, label="standings")

    if "magic_num" not in enriched.columns:
        raise RuntimeError("Standings tables missing magic/clinch column; cannot detect champions.")
    clinched_mask = enriched["magic_num"].astype(str).str.contains(
        "clinch", case=False, na=False
    )
    champions = enriched[clinched_mask].copy()
    if champions.empty:
        raise RuntimeError("No clinched teams detected; champion extraction failed.")

    champions = _attach_run_diff_from_summary(champions, season, league_id, repo_root)
    if "run_diff" not in champions.columns:
        champions["run_diff"] = pd.NA

    div_champs = _select_division_champs(champions)
    wildcards = _select_wildcards(champions, div_champs)

    def row_to_dict(row: pd.Series) -> dict[str, Any]:
        rd = row.get("run_diff")
        if pd.isna(rd):
            rd = None
        div_label = str(row["division"]).replace("Division", "")
        div_label = div_label.replace("/", " ").strip()
        div_parts = div_label.split()
        div_short = div_parts[-1] if div_parts else div_label
        div_short = div_short.title()
        if div_short.startswith("East"):
            div_short = "East"
        elif div_short.startswith("West"):
            div_short = "West"
        elif div_short.startswith("Central"):
            div_short = "Central"
        return {
            "team_id": int(row["team_id"]),
            "team_name": str(row["team_name"]),
            "team_abbr": str(row["team_abbr"]),
            "conference": _short_conf(str(row["conference"])),
            "division": div_short,
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "run_diff": int(rd) if rd is not None else None,
        }

    champions_dict: dict[str, dict[str, Any]] = {"ABC": {}, "NBC": {}}
    for _, row in div_champs.iterrows():
        payload = row_to_dict(row)
        conf = payload["conference"]
        champions_dict.setdefault(conf, {})[payload["division"]] = payload
    for _, row in wildcards.iterrows():
        payload = row_to_dict(row)
        conf = payload["conference"]
        champions_dict.setdefault(conf, {})["Wild Card"] = payload

    # Validate completeness
    for conf in ["ABC", "NBC"]:
        expected = {"East", "Central", "West", "Wild Card"}
        missing = expected - set(champions_dict.get(conf, {}).keys())
        if missing:
            raise RuntimeError(f"{conf} champions incomplete; missing {missing}")

    return champions_dict


def _champions_to_frame(champions: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for conf, slots in champions.items():
        for slot, payload in slots.items():
            row = {"conference": conf, "slot": slot}
            row.update(payload)
            rows.append(row)
    return pd.DataFrame(rows)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Extract league champions from almanac standings HTML.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    parser.add_argument("--almanac-zip", type=Path, default=None)
    parser.add_argument("--dim-team-park", type=Path, default=None)
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write champions CSV to csv/out/almanac/<season>/league_champions_<season>_league<league_id>.csv",
    )
    args = parser.parse_args()

    champions = get_league_champions_from_standings(
        args.season,
        league_id=args.league_id,
        almanac_zip=args.almanac_zip,
        dim_team_park_path=args.dim_team_park,
    )

    df = _champions_to_frame(champions)
    if df.empty:
        _log("[WARN] Champions dataframe is empty.")
    else:
        _log("[OK] Detected champions:")
        for _, row in df.sort_values(["conference", "slot"]).iterrows():
            _log(
                f"- {row['conference']} {row['slot']}: "
                f"{row['team_name']} ({row['team_abbr']}) — "
                f"{row['wins']}-{row['losses']}, run_diff={row.get('run_diff')}"
            )

    if args.write_csv:
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = repo_root / "csv" / "out" / "almanac" / str(args.season)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"league_champions_{args.season}_league{args.league_id}.csv"
        df.to_csv(out_path, index=False)
        _log(f"[OK] Wrote champions CSV to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
