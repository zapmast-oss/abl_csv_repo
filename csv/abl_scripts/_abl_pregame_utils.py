"""Shared helpers for ABL pregame reports."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def resolve_base(args_base: str | None) -> Path:
    """Return the repo root; default to repo root if --base is omitted."""
    if args_base:
        return Path(args_base).resolve()
    return Path(__file__).resolve().parents[2]


def _ranking_key(path: Path, base: Path) -> Tuple[int, int, str]:
    """Prefer csv/out first, then csv, then anything else; shorter paths win."""
    try:
        rel = path.relative_to(base)
    except Exception:
        rel = path
    parts = [p.lower() for p in rel.parts]
    if "csv" in parts and "out" in parts:
        rank = 0
    elif "csv" in parts:
        rank = 1
    else:
        rank = 2
    return (rank, len(parts), rel.as_posix())


def find_csv_files(base: Path, patterns: Sequence[str]) -> List[Path]:
    """Recursively find CSVs under base/csv whose names match any pattern."""
    csv_root = base / "csv"
    if not csv_root.exists():
        return []

    patterns_lower = [p.lower() for p in patterns]
    matches: List[Path] = []
    for path in csv_root.rglob("*.csv"):
        name_lower = path.name.lower()
        if any(p in name_lower for p in patterns_lower):
            matches.append(path)
            continue
        if any(fnmatch.fnmatch(name_lower, p) for p in patterns_lower):
            matches.append(path)
    matches.sort(key=lambda p: _ranking_key(p, base))
    return matches


def list_all_csv_paths(base: Path) -> List[Path]:
    """Return all CSV paths under base/csv, ranked deterministically."""
    csv_root = base / "csv"
    if not csv_root.exists():
        return []
    paths = list(csv_root.rglob("*.csv"))
    paths.sort(key=lambda p: _ranking_key(p, base))
    return paths


def scan_csv_headers_for_columns(csv_paths: Sequence[Path], needle_substrings: Sequence[str]) -> List[tuple[Path, List[str]]]:
    """Scan headers only; return paths with columns containing any needle substrings."""
    needles = [n.lower() for n in needle_substrings]
    hits: List[tuple[Path, List[str]]] = []
    for path in csv_paths:
        try:
            header_df = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        cols = list(header_df.columns)
        matches = [c for c in cols if any(n in c.lower() for n in needles)]
        if matches:
            hits.append((path, matches))
    return hits


def load_best_csv(
    base: Path, patterns: Sequence[str], required_cols: Optional[Iterable[str]] = None
) -> tuple[pd.DataFrame, Optional[Path]]:
    """Load the first CSV that matches patterns and (optionally) has required cols."""
    for path in find_csv_files(base, patterns):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if required_cols:
            if not all(col in df.columns for col in required_cols):
                continue
        return df, path
    return pd.DataFrame(), None


def safe_get(df: pd.DataFrame, col_candidates: Sequence[str]) -> Optional[pd.Series]:
    """Return the first matching column (case-insensitive)."""
    if df is None or df.empty:
        return None
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in col_candidates:
        col = lower_map.get(candidate.lower())
        if col:
            return df[col]
    return None


def normalize_team_table(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to normalize a team dimension table."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["team_id", "team_abbr", "team_name", "league_id"])

    def pick(cols: Sequence[str]) -> Optional[pd.Series]:
        return safe_get(df, cols)

    team_id = pick(["team_id", "teamid", "id"])
    abbr = pick(["team_abbr", "abbr", "team_code", "code"])
    name = pick(["team_name", "name", "team"])
    city = pick(["city"])
    nickname = pick(["nickname"])
    league = pick(["league_id", "lg_id"])

    norm = pd.DataFrame()
    if team_id is not None:
        norm["team_id"] = pd.to_numeric(team_id, errors="coerce").astype("Int64")
    if abbr is not None:
        norm["team_abbr"] = abbr.astype(str).str.strip().str.upper()
    else:
        norm["team_abbr"] = pd.NA

    if name is not None:
        norm["team_name"] = name.astype(str).str.strip()
    else:
        combo = ""
        if city is not None and nickname is not None:
            combo = (city.fillna("").astype(str) + " " + nickname.fillna("").astype(str)).str.strip()
        norm["team_name"] = combo if isinstance(combo, pd.Series) else pd.NA

    norm["league_id"] = pd.to_numeric(league, errors="coerce").astype("Int64") if league is not None else pd.NA
    norm = norm.drop_duplicates(subset=["team_id", "team_abbr", "team_name"], keep="first")
    return norm.reset_index(drop=True)


def _merge_key_from_columns(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        col = lower_cols.get(cand.lower())
        if not col:
            continue
        series = df[col]
        if "id" in cand.lower():
            return pd.to_numeric(series, errors="coerce").apply(lambda v: f"id:{int(v)}" if pd.notna(v) else "")
        if "abbr" in cand.lower():
            return series.astype(str).str.strip().str.upper()
        if "name" in cand.lower():
            return series.astype(str).str.strip().str.lower()
    return pd.Series([""] * len(df))


def load_team_keyed_frame(path: Path, team_key_candidates: Sequence[str], keep_cols: Sequence[str]) -> tuple[pd.DataFrame, List[str]]:
    """Load CSV with only needed columns and normalize team keys."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(), []
    cols_to_keep = [c for c in keep_cols if c in df.columns]
    key_cols = [c for c in team_key_candidates if c in df.columns]
    if not key_cols:
        return pd.DataFrame(), []
    cols = list(dict.fromkeys(cols_to_keep + key_cols))
    df = df[cols].copy()
    used = cols.copy()
    # Normalize keys if present
    if "team_abbr" in df.columns:
        df["team_abbr"] = df["team_abbr"].astype(str).str.strip().str.upper()
    if "team_name" in df.columns:
        df["team_name"] = df["team_name"].astype(str).str.strip().str.lower()
    if "league_id" in df.columns:
        df["league_id"] = pd.to_numeric(df["league_id"], errors="coerce").astype("Int64")
    df["__merge_key"] = _merge_key_from_columns(df, ["team_id", "team_abbr", "team_name"])
    return df, used


def coalesce_team_values(team_df: pd.DataFrame, frames: List[tuple[pd.DataFrame, Path]], value_cols: Sequence[str]) -> tuple[pd.DataFrame, List[Path]]:
    """Merge multiple frames into team_df and coalesce value cols in order."""
    result = team_df.copy()
    if "__merge_key" not in result.columns:
        result["__merge_key"] = _merge_key_from_columns(result, ["team_id", "team_abbr", "team_name"])
    for col in value_cols:
        result[col] = pd.NA
    used_paths: List[Path] = []
    for frame, path in frames:
        if frame is None or frame.empty:
            continue
        if "__merge_key" not in frame.columns:
            frame["__merge_key"] = _merge_key_from_columns(frame, ["team_id", "team_abbr", "team_name"])
        used_in_frame = False
        frame_dedup = frame.drop_duplicates(subset="__merge_key", keep="first")
        for col in value_cols:
            if col not in frame.columns:
                continue
            mask = result[col].isna()
            if not mask.any():
                continue
            matched = frame_dedup.set_index("__merge_key")
            result.loc[mask, col] = result.loc[mask, "__merge_key"].map(matched[col])
            used_in_frame = True
        if used_in_frame:
            used_paths.append(path)
    return result, used_paths


def md_table(df: pd.DataFrame, columns: Sequence[str], header_map: Optional[dict[str, str]] = None) -> str:
    """Render a Markdown table from df with the requested columns."""
    header_map = header_map or {}
    if df is None or df.empty:
        return "No data found.\n"

    present_cols = [col for col in columns if col in df.columns]
    if not present_cols:
        return "No data found.\n"

    headers = [header_map.get(col, col) for col in present_cols]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df[present_cols].iterrows():
        values = []
        for col in present_cols:
            val = row.get(col)
            if pd.isna(val):
                values.append("N/A")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_md(path: Path, text: str) -> None:
    """Write markdown to disk, ensuring parent directories exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
