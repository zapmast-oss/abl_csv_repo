"""Shared helpers for ABL pregame reports."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from pandas import DataFrame


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


def scan_csv_headers_for_columns(csv_paths: Sequence[Path], needle_substrings: Sequence[str]) -> tuple[List[tuple[Path, List[str]]], List[Path]]:
    """Scan headers only; return (hits, scan_errors)."""
    needles = [n.lower() for n in needle_substrings]
    hits: List[tuple[Path, List[str]]] = []
    errors: List[Path] = []
    for path in csv_paths:
        try:
            header_df = pd.read_csv(path, nrows=0)
        except Exception:
            errors.append(path)
            continue
        cols = list(header_df.columns)
        matches = [c for c in cols if any(n in c.lower() for n in needles)]
        if matches:
            hits.append((path, matches))
    return hits, errors


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


def load_team_financials(base: Path, league_id: int = 200, season: int | None = None) -> tuple[pd.DataFrame, List[str], List[str]]:
    """Load team financials from known sources in priority order."""
    sources_used: List[str] = []
    notes: List[str] = []

    def detect_cols(df: DataFrame, wanted: dict[str, list[str]]) -> dict[str, Optional[str]]:
        found: dict[str, Optional[str]] = {k: None for k in wanted}
        headers = list(df.columns)
        lower_map = {h.lower(): h for h in headers}

        for canon, synonyms in wanted.items():
            # exact match
            for syn in synonyms:
                col = lower_map.get(syn.lower())
                if col:
                    found[canon] = col
                    break
            if found[canon]:
                continue
            # substring contains
            for h in headers:
                low = h.lower()
                if any(syn.lower() in low for syn in synonyms):
                    found[canon] = h
                    break
            if found[canon]:
                continue
            # word boundary
            for h in headers:
                low = h.lower()
                if any(f" {syn.lower()} " in f" {low} " for syn in synonyms):
                    found[canon] = h
                    break
        return found

    team_key_candidates = ["team_id", "ID", "team_abbr", "Abbr", "team_name", "Team Name"]
    season_keys = ["season", "year"]
    league_keys = ["league_id", "LG", "league"]
    synonym_map = {
        "budget": ["Bgt", "budget", "team_budget", "player_budget", "budget_total", "budget (", "budget$", "BgtSpc"],
        "payroll": ["Pay", "payroll", "player_payroll", "team_payroll", "salary", "salaries", "total_salary", "player_payroll", "player_payroll"],
        "cash": ["cash", "cash_on_hand", "balance", "bank", "cash balance", "cash_owner", "cash_trades"],
        "revenue": ["Revenue", "revenue", "total_revenue", "income", "total_income"],
        "profit": ["profit", "net", "net_income", "operating_profit", "net profit", "financial_balance"],
    }

    def normalize_keys(df: DataFrame) -> DataFrame:
        df = df.copy()
        if "team_abbr" in df.columns:
            df["team_abbr"] = df["team_abbr"].astype(str).str.strip().str.upper()
        if "team_name" in df.columns:
            df["team_name"] = df["team_name"].astype(str).str.strip()
        return df

    def apply_filters(df: DataFrame, key_map: dict[str, Optional[str]]) -> DataFrame:
        filt = df
        if key_map.get("league_id"):
            filt = filt[pd.to_numeric(filt[key_map["league_id"]], errors="coerce") == league_id]
        season_col = next((c for c in season_keys if key_map.get(c)), None)
        if season_col:
            season_vals = pd.to_numeric(filt[key_map[season_col]], errors="coerce")
            filt = filt.assign(_season=season_vals)
            chosen = season if season is not None else season_vals.max()
            filt = filt[filt["_season"] == chosen]
            notes.append(f"Selected season: {int(chosen) if pd.notna(chosen) else 'N/A'} from {season_col}")
        return filt

    def pick_keys(df: DataFrame) -> dict[str, Optional[str]]:
        key_map: dict[str, Optional[str]] = {"team_id": None, "team_abbr": None, "team_name": None, "league_id": None, "season": None}
        lower_map = {c.lower(): c for c in df.columns}
        for cand in ["team_id", "id"]:
            if cand in lower_map:
                key_map["team_id"] = lower_map[cand]
                break
        for cand in ["team_abbr", "abbr"]:
            if cand in lower_map:
                key_map["team_abbr"] = lower_map[cand]
                break
        for cand in ["team_name", "team", "name"]:
            if cand in lower_map:
                key_map["team_name"] = lower_map[cand]
                break
        for cand in ["league_id", "lg", "league"]:
            if cand in lower_map:
                key_map["league_id"] = lower_map[cand]
                break
        for cand in season_keys:
            if cand in lower_map:
                key_map["season"] = lower_map[cand]
                break
        return key_map

    def load_source(path: Path) -> tuple[DataFrame, dict[str, Optional[str]], dict[str, Optional[str]]]:
        df = pd.read_csv(path)
        key_map = pick_keys(df)
        col_map = detect_cols(df, synonym_map)
        used_cols = [c for c in col_map.values() if c] + [v for v in key_map.values() if v]
        df = df.loc[:, [c for c in used_cols if c in df.columns]].copy()
        df = df.rename(columns={col_map[k]: k for k in col_map if col_map[k]})
        df = df.rename(columns={key_map.get("team_id", ""): "team_id", key_map.get("team_abbr", ""): "team_abbr", key_map.get("team_name", ""): "team_name"})
        if key_map.get("league_id"):
            df = df.rename(columns={key_map["league_id"]: "league_id"})
        if key_map.get("season"):
            df = df.rename(columns={key_map["season"]: "season"})
        df = apply_filters(df, key_map)
        df = normalize_keys(df)
        return df, key_map, col_map

    frames: list[tuple[DataFrame, Path]] = []

    # Priority 1: fact_team_financials
    fact_path = base / "csv" / "out" / "star_schema" / "fact_team_financials.csv"
    if fact_path.exists():
        try:
            df, key_map, col_map = load_source(fact_path)
            if not df.empty:
                frames.append((df, fact_path))
                sources_used.append(str(fact_path))
                notes.append(f"fact_team_financials columns: {col_map}")
        except Exception as exc:
            notes.append(f"fact_team_financials load error: {exc}")

    # Priority 2: team_last_financials, team_financials
    for name in ["team_last_financials.csv", "team_financials.csv"]:
        path = base / "csv" / "ootp_csv" / name
        if not path.exists():
            continue
        try:
            df, key_map, col_map = load_source(path)
            if not df.empty:
                frames.append((df, path))
                sources_used.append(str(path))
                notes.append(f"{name} columns: {col_map}")
                break
        except Exception as exc:
            notes.append(f"{name} load error: {exc}")

    # Priority 3: team_history_financials
    history_path = base / "csv" / "ootp_csv" / "team_history_financials.csv"
    if history_path.exists():
        try:
            df, key_map, col_map = load_source(history_path)
            if not df.empty:
                frames.append((df, history_path))
                sources_used.append(str(history_path))
                notes.append(f"team_history_financials columns: {col_map}")
        except Exception as exc:
            notes.append(f"team_history_financials load error: {exc}")

    if not frames:
        result = pd.DataFrame(columns=["team_id", "team_abbr", "team_name", "budget", "payroll", "cash", "revenue", "profit"])
        return result, sources_used, notes

    # Start with an empty scaffold of teams from the first frame
    # Build base scaffold with whatever team keys are present
    scaffold_cols = [c for c in ["team_id", "team_abbr", "team_name"] if c in frames[0][0].columns]
    base_df = frames[0][0][scaffold_cols].copy()
    for col in ["team_id", "team_abbr", "team_name"]:
        if col not in base_df.columns:
            base_df[col] = pd.NA
    base_df["__merge_key"] = _merge_key_from_columns(base_df, ["team_id", "team_abbr", "team_name"])
    coalesced, used_paths = coalesce_team_values(base_df, frames, ["budget", "payroll", "cash", "revenue", "profit"])
    sources_used = sources_used  # already ordered
    return coalesced, sources_used, notes


def find_pitcher_arsenal_sources(base: Path) -> List[Path]:
    """Find pitcher_arsenal_all.txt files; most recent first."""
    candidates = list((base).rglob("pitcher_arsenal_all.txt"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def _parse_arsenal_text(path: Path) -> tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    records = []
    current_team = None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    name_pattern = re.compile(r"^(?P<name>.+?)\s*\(.*\).*?\|\s*Pitches\s*(?P<count>\d+)\s*\|\s*Best\s+(?P<best>[^\d]+?)\s*(?P<best_val>\d+)")
    arsenal_pattern = re.compile(r"^Arsenal:\s*(.+)$")
    for idx, line in enumerate(lines):
        team_match = re.match(r"^TEAM\s+(?P<abbr>\w+)", line.strip())
        if team_match:
            current_team = team_match.group("abbr").strip().upper()
            continue
        m = name_pattern.match(line.strip())
        if not m:
            continue
        name = m.group("name").strip()
        count = pd.to_numeric(m.group("count"), errors="coerce")
        best_name = m.group("best").strip()
        best_val = pd.to_numeric(m.group("best_val"), errors="coerce")
        pitches_list = []
        if idx + 1 < len(lines):
            arm = arsenal_pattern.match(lines[idx + 1].strip())
            if arm:
                parts = [p.strip() for p in arm.group(1).split(",") if p.strip()]
                for part in parts:
                    toks = part.rsplit(" ", 1)
                    if len(toks) == 2:
                        pname, pval = toks
                        pval_num = pd.to_numeric(pval, errors="coerce")
                        pitches_list.append((pname, pval_num if pd.notna(pval_num) else pval))
        records.append(
            {
                "player_id": pd.NA,
                "player_name": name,
                "team_abbr": current_team,
                "pitch_count": count,
                "best_pitch": best_name,
                "best_pitch_value": best_val,
                "pitches": pitches_list,
            }
        )
    return pd.DataFrame(records), notes


def _format_arsenal_line(row: pd.Series, top_k: int = 3) -> str:
    count = row.get("pitch_count")
    count_txt = f"{int(count)}" if pd.notna(count) else "N/A"
    pitches = row.get("pitches") or []
    best_pitch = str(row.get("best_pitch")) if pd.notna(row.get("best_pitch")) else None
    best_val = row.get("best_pitch_value")
    rendered: List[str] = []
    for name, val in pitches:
        label = name
        if best_pitch and name == best_pitch:
            label += " (best)"
        if pd.notna(val):
            label += f" {int(val) if float(val).is_integer() else val}"
        rendered.append(label)
    if top_k > 0:
        rendered = rendered[:top_k]
    if not rendered and best_pitch:
        rendered = [f"{best_pitch} {best_val if best_val is not None else ''} (best)".strip()]
    if not rendered:
        return "Arsenal: N/A (no repertoire source found)"
    return f"Arsenal ({count_txt}): " + ", ".join(rendered)


def load_pitcher_arsenals(base: Path) -> tuple[pd.DataFrame, List[str], List[str]]:
    """Load pitcher arsenal data from text or CSV repertoires."""
    sources: List[str] = []
    notes: List[str] = []
    text_sources = find_pitcher_arsenal_sources(base)
    if text_sources:
        src = text_sources[0]
        try:
            df, parse_notes = _parse_arsenal_text(src)
            sources.append(str(src))
            notes.extend(parse_notes)
            if not df.empty:
                return df, sources, notes
        except Exception as exc:
            notes.append(f"Text arsenal parse error: {exc}")

    # CSV fallback
    csv_candidates = list((base / "csv").rglob("*.csv"))
    csv_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in csv_candidates:
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        cols = list(header.columns)
        pitch_cols = [c for c in cols if any(tok in str(c).lower() for tok in ["pitch", "repertoire", "arsenal", "p1", "p2", "p3"])]
        if not pitch_cols:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        name_col = next((c for c in df.columns if str(c).lower() in {"player_name", "name"}), None)
        pid_col = next((c for c in df.columns if "player_id" in str(c).lower() or str(c).lower() == "id"), None)
        team_col = next((c for c in df.columns if "team_abbr" in str(c).lower() or str(c).lower() == "team"), None)
        if not name_col and not pid_col:
            continue
        records = []
        for _, row in df.iterrows():
            pitches = []
            for col in pitch_cols:
                val = row.get(col)
                if pd.isna(val):
                    continue
                if isinstance(val, str) and val.strip() == "":
                    continue
                pitches.append((str(col), val))
            if not pitches:
                continue
            best = None
            best_val = None
            for name, val in pitches:
                num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
                if pd.notna(num):
                    if best_val is None or num > best_val:
                        best = name
                        best_val = num
            records.append(
                {
                    "player_id": row.get(pid_col) if pid_col else pd.NA,
                    "player_name": row.get(name_col) if name_col else pd.NA,
                    "team_abbr": str(row.get(team_col)).strip().upper() if team_col else pd.NA,
                    "pitch_count": len(pitches),
                    "best_pitch": best,
                    "best_pitch_value": best_val,
                    "pitches": pitches,
                }
            )
        if records:
            sources.append(str(path))
            notes.append(f"CSV arsenal from {path} using columns {pitch_cols}")
            return pd.DataFrame(records), sources, notes

    return pd.DataFrame(columns=["player_id", "player_name", "team_abbr", "pitch_count", "best_pitch", "best_pitch_value", "pitches"]), sources, notes


def format_arsenal_display(row: pd.Series, top_k: int = 3) -> str:
    try:
        return _format_arsenal_line(row, top_k)
    except Exception:
        return "Arsenal: N/A (format error)"


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
