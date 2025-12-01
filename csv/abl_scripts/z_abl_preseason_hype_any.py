#!/usr/bin/env python
"""
Build preseason hype brief for EB based on hyped players' season WAR.

Rules:
- Parse preseason prediction HTML by player_id.
- Resolve full name from dim_player_profile (First Name + Last Name).
- Resolve team abbreviation from season stats, then HTML, then profile team/org fields.
- Never emit "Free agent" or "nan"; hard fail if any player lacks name or team.
- Emit markdown: Full Name (TM) - WAR: X.XX.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from bs4 import BeautifulSoup
import zipfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and CLI
# ---------------------------------------------------------------------------

def get_repo_root() -> Path:
    """Return repo root assuming this file lives under csv/abl_scripts."""
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EB preseason hype fragment.")
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g. 1980")
    parser.add_argument("--league-id", type=int, default=200, help="League ID (default 200)")
    parser.add_argument(
        "--preseason-html",
        type=str,
        default=None,
        help=(
            "Optional explicit path to preseason prediction HTML. "
            "Defaults to csv/in/almanac_core/{season}/leagues/league_{league_id}_preseason_prediction_report.html"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional output markdown path override.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def parse_preseason_players(html_path: Path) -> pd.DataFrame:
    """Parse preseason prediction HTML and return hype players.

    Columns:
      player_id      int
      raw_name       str
      raw_team_abbr  str
    """
    if not html_path.exists():
        raise FileNotFoundError(f"Preseason HTML not found: {html_path}")

    html_text = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")

    records: List[dict] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "player_" not in href:
            continue
        m = re.search(r"player_(\d+)\.html", href)
        if not m:
            continue
        player_id = int(m.group(1))
        tr = a.find_parent("tr")
        if tr is None:
            continue
        raw_name = a.get_text(strip=True)
        team_abbr = ""
        for link in tr.find_all("a", href=True):
            if "team_" in link["href"]:
                team_abbr = link.get_text(strip=True)
                break
        records.append({"player_id": player_id, "raw_name": raw_name, "raw_team_abbr": team_abbr})

    df = pd.DataFrame.from_records(records)
    df = df.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    logger.info("Parsed %d preseason hype players from HTML", len(df))
    return df


# ---------------------------------------------------------------------------
# Load reference data
# ---------------------------------------------------------------------------

def load_dim_player_profile(root: Path) -> pd.DataFrame:
    """Load player profile with ID, names, and optional team/org fields."""
    path = root / "csv" / "out" / "star_schema" / "dim_player_profile.csv"
    if not path.exists():
        raise FileNotFoundError(f"dim_player_profile not found at {path}")
    df = pd.read_csv(path)
    required = {"ID", "First Name", "Last Name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dim_player_profile missing columns: {sorted(missing)}")
    keep = ["ID", "First Name", "Last Name"]
    for opt in ["Name", "Name.1", "TM", "TM.1", "ORG", "ORG.1"]:
        if opt in df.columns:
            keep.append(opt)
    df = df[keep].copy()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

    # Build full_name with fallbacks
    df["First Name"] = df["First Name"].fillna("").astype(str).str.strip()
    df["Last Name"] = df["Last Name"].fillna("").astype(str).str.strip()
    df["full_name"] = (df["First Name"] + " " + df["Last Name"]).str.strip()
    if "Name" in df.columns:
        df.loc[df["full_name"] == "", "full_name"] = df.loc[df["full_name"] == "", "Name"].fillna("").astype(str).str.strip()
    if "Name.1" in df.columns:
        df.loc[df["full_name"] == "", "full_name"] = df.loc[df["full_name"] == "", "Name.1"].fillna("").astype(str).str.strip()
    return df


def load_players_csv(root: Path) -> pd.DataFrame:
    """Load ootp players.csv for additional name/position resolution."""
    path = root / "csv" / "ootp_csv" / "players.csv"
    if not path.exists():
        logging.warning("players.csv not found at %s; skipping extra name resolution", path)
        return pd.DataFrame(columns=["ID", "players_first", "players_last", "players_full", "players_position"])
    df = pd.read_csv(path)
    id_col = None
    for cand in ["player_id", "id", "ID"]:
        if cand in df.columns:
            id_col = cand
            break
    first_col = next((c for c in ["first_name", "First Name", "first name"] if c in df.columns), None)
    last_col = next((c for c in ["last_name", "Last Name", "last name"] if c in df.columns), None)
    pos_col = "position" if "position" in df.columns else None
    if id_col is None or first_col is None or last_col is None:
        logging.warning("players.csv missing id/first/last columns; skipping extra names")
        return pd.DataFrame(columns=["ID", "players_first", "players_last", "players_full", "players_position"])
    keep = [id_col, first_col, last_col]
    if pos_col:
        keep.append(pos_col)
    out = df[keep].rename(
        columns={id_col: "ID", first_col: "players_first", last_col: "players_last", pos_col: "players_position" if pos_col else pos_col}
    )
    out["ID"] = pd.to_numeric(out["ID"], errors="coerce").astype("Int64")
    out["players_first"] = out["players_first"].fillna("").astype(str).str.strip()
    out["players_last"] = out["players_last"].fillna("").astype(str).str.strip()
    out["players_full"] = (out["players_first"] + " " + out["players_last"]).str.strip()
    if pos_col:
        out["players_position"] = out["players_position"].fillna("").astype(str).str.strip()
    else:
        out["players_position"] = ""
    return out


def load_season_war(root: Path, season: int, league_id: int) -> pd.DataFrame:
    """Load season WAR and team abbreviation from batting/pitching fact tables.

    Tries season-specific files first, then generic. Sums batting + pitching WAR.
    """
    base = root / "csv" / "out" / "star_schema"

    def fact_candidates(prefix: str) -> List[Path]:
        return [
            base / f"{prefix}_{season}_league{league_id}.csv",
            base / f"{prefix}_{season}.csv",
            base / f"{prefix}.csv",
        ]

    def load_fact(paths: List[Path], team_cols: List[str], war_col: str, team_label: str) -> pd.DataFrame:
        for p in paths:
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "season" in df.columns:
                df = df[df["season"] == season]
            team_col = next((c for c in team_cols if c in df.columns), None)
            if team_col is None or war_col not in df.columns or "ID" not in df.columns:
                continue
            out = df[["ID", team_col, war_col]].copy()
            out.rename(columns={team_col: f"team_abbr_{team_label}", war_col: f"WAR_{team_label}"}, inplace=True)
            out["ID"] = pd.to_numeric(out["ID"], errors="coerce").astype("Int64")
            out[f"WAR_{team_label}"] = pd.to_numeric(out[f"WAR_{team_label}"], errors="coerce").fillna(0.0)
            return out
        return pd.DataFrame()

    bat = load_fact(fact_candidates("fact_player_batting"), ["TM", "team_abbr", "Tm", "tm"], "WAR", "bat")
    pit = load_fact(fact_candidates("fact_player_pitching"), ["TM_p1", "TM", "team_abbr", "Tm", "tm"], "WAR", "pit")

    if bat.empty and pit.empty:
        raise FileNotFoundError("No season WAR found in batting or pitching fact tables.")

    if bat.empty:
        stats = pit
    elif pit.empty:
        stats = bat
    else:
        stats = bat.merge(pit, on="ID", how="outer")

    if "WAR_bat" not in stats.columns:
        stats["WAR_bat"] = 0.0
    if "WAR_pit" not in stats.columns:
        stats["WAR_pit"] = 0.0

    stats["WAR_season"] = stats["WAR_bat"] + stats["WAR_pit"]

    if "team_abbr_bat" not in stats.columns:
        stats["team_abbr_bat"] = ""
    if "team_abbr_pit" not in stats.columns:
        stats["team_abbr_pit"] = ""

    stats["team_abbr_stats"] = stats["team_abbr_bat"].fillna("")
    mask = stats["team_abbr_stats"].isna() | (stats["team_abbr_stats"] == "")
    stats.loc[mask, "team_abbr_stats"] = stats.loc[mask, "team_abbr_pit"].fillna("")

    return stats[["ID", "team_abbr_stats", "team_abbr_bat", "team_abbr_pit", "WAR_season"]]


def load_html_war(root: Path, season: int, player_ids: List[int]) -> Dict[int, float]:
    """Fallback: read per-player HTML pages from almanac zip to extract season WAR."""
    almanac_zip = root / "data_raw" / "ootp_html" / f"almanac_{season}.zip"
    if not almanac_zip.exists():
        logging.warning("Almanac zip not found for HTML WAR fallback: %s", almanac_zip)
        return {}
    wars: Dict[int, float] = {}
    with zipfile.ZipFile(almanac_zip, "r") as zf:
        for pid in player_ids:
            candidates = [
                f"players/player_{pid}.html",
                f"almanac_{season}/players/player_{pid}.html",
            ]
            try:
                for html_name in candidates:
                    try:
                        with zf.open(html_name) as fp:
                            html_text = fp.read().decode("utf-8", errors="ignore")
                        break
                    except KeyError:
                        html_text = None
                if html_text is None:
                    continue
            except KeyError:
                continue
            try:
                tables = pd.read_html(html_text)
            except Exception:
                continue
            found = False
            for tbl in tables:
                # Normalize column labels
                tbl.columns = [c[1] if isinstance(c, tuple) else c for c in tbl.columns]
                # Preferred: explicit WAR column
                if "WAR" in tbl.columns:
                    for col in ["Year", "Season", "year", "season"]:
                        if col in tbl.columns:
                            sub = tbl[tbl[col].astype(str) == str(season)]
                            if not sub.empty and pd.notna(sub["WAR"]).any():
                                try:
                                    wars[pid] = float(sub["WAR"].iloc[0])
                                    found = True
                                    break
                                except Exception:
                                    pass
                    if found:
                        break
                    # fallback: first column may be year
                    first_col = tbl.columns[0]
                    sub = tbl[tbl[first_col].astype(str) == str(season)]
                    if not sub.empty and pd.notna(sub["WAR"]).any():
                        try:
                            wars[pid] = float(sub["WAR"].iloc[0])
                            found = True
                            break
                        except Exception:
                            pass
                else:
                    # No explicit WAR column; skip to avoid bogus values
                    continue
            if found:
                continue
    logging.info("HTML WAR fallback found values for %d players", len(wars))
    return wars


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_team_abbr(row: pd.Series) -> str:
    """Choose first non-empty, non-free-agent team abbreviation."""
    candidates = [
        str(row.get("team_abbr_stats", "")),
        str(row.get("team_abbr_bat", "")),
        str(row.get("team_abbr_pit", "")),
        str(row.get("raw_team_abbr", "")),
        str(row.get("TM", "")),
        str(row.get("TM.1", "")),
        str(row.get("ORG", "")),
        str(row.get("ORG.1", "")),
    ]
    for cand in candidates:
        val = cand.strip()
        if not val:
            continue
        if val.lower() in {"free agent", "fa", "nan"}:
            continue
        return val
    return ""


def resolve_players(hype: pd.DataFrame, profiles: pd.DataFrame, players: pd.DataFrame, stats: pd.DataFrame, html_war: Dict[int, float]) -> pd.DataFrame:
    """Join hype with profiles/players/stats; enforce full_name/team completeness."""
    merged = hype.merge(profiles, left_on="player_id", right_on="ID", how="left", suffixes=("", "_prof"))
    logging.info("Hype rows matched to profiles: %d of %d", merged["full_name"].notna().sum(), len(merged))

    if not players.empty:
        merged = merged.merge(players, left_on="player_id", right_on="ID", how="left", suffixes=("", "_ply"))

    merged = merged.merge(stats, left_on="player_id", right_on="ID", how="left", suffixes=("", "_stat"))
    logging.info("Hype rows matched to season stats: %d of %d", merged["WAR_season"].notna().sum(), len(merged))

    # Build safe full_name with fallbacks (First+Last, Name/Name.1, players.csv, raw_name trimmed)
    merged["First Name"] = merged["First Name"].fillna("").astype(str).str.strip()
    merged["Last Name"] = merged["Last Name"].fillna("").astype(str).str.strip()

    def build_full_name(row: pd.Series) -> str:
        fn = row.get("First Name", "")
        ln = row.get("Last Name", "")
        if isinstance(fn, str) and fn.strip() and isinstance(ln, str) and ln.strip():
            return f"{fn.strip()} {ln.strip()}"
        for alt in ["Name", "Name.1", "players_full"]:
            if alt in row and isinstance(row[alt], str) and row[alt].strip():
                return row[alt].strip()
        raw = str(row.get("raw_name", "")).strip()
        if "," in raw:
            raw = raw.split(",", 1)[0].strip()
        return raw

    merged["full_name"] = merged.apply(build_full_name, axis=1)
    merged["team_abbr"] = merged.apply(resolve_team_abbr, axis=1)

    # Position: prefer players.csv, else parse from raw_name after comma
    def pick_position(row: pd.Series) -> str:
        pos = row.get("players_position", "")
        if isinstance(pos, str) and pos.strip():
            return pos.strip()
        raw = str(row.get("raw_name", "")).strip()
        if "," in raw:
            return raw.split(",", 1)[1].strip()
        return ""
    merged["position"] = merged.apply(pick_position, axis=1)

    # war from stats, then fallback to HTML per-player pages if still zero/NA
    merged["war_value"] = pd.to_numeric(merged.get("WAR_season", 0), errors="coerce").fillna(0.0)
    def apply_html_war(row: pd.Series) -> float:
        val = float(row.get("war_value", 0.0))
        pid = row.get("player_id")
        if (pd.isna(val) or val == 0.0) and isinstance(pid, (int, float)) and not pd.isna(pid):
            pid_int = int(pid)
            if pid_int in html_war:
                return float(html_war[pid_int])
        return val
    merged["war_value"] = merged.apply(apply_html_war, axis=1)

    bad_name = merged[merged["full_name"].isna() | (merged["full_name"].str.strip() == "")]
    bad_team = merged[
        merged["team_abbr"].isna()
        | (merged["team_abbr"].str.strip() == "")
        | (merged["team_abbr"].str.lower().isin(["free agent", "fa", "nan"]))
    ]
    if not bad_name.empty or not bad_team.empty:
        logging.error("Preseason hype mapping failed. Offending rows:")
        logging.error(pd.concat([bad_name, bad_team], ignore_index=True)[["player_id", "raw_name", "full_name", "team_abbr"]])
        raise RuntimeError("Missing full_name or team_abbr for some hype players.")

    return merged[["player_id", "full_name", "team_abbr", "position", "war_value"]]


# ---------------------------------------------------------------------------
# Bucketing and rendering
# ---------------------------------------------------------------------------

def bucket_war(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df, df
    war = df["war_value"].astype(float).sort_values()
    if len(war) >= 3:
        low_cut = float(war.quantile(1.0 / 3.0))
        high_cut = float(war.quantile(2.0 / 3.0))
    else:
        low_cut = float(war.min())
        high_cut = float(war.max())

    def assign(val: float) -> str:
        if val >= high_cut:
            return "over"
        if val <= low_cut:
            return "under"
        return "delivered"

    df = df.copy()
    df["bucket"] = df["war_value"].apply(assign)
    over = df[df["bucket"] == "over"].sort_values("war_value", ascending=False)
    delivered = df[df["bucket"] == "delivered"].sort_values("war_value", ascending=False)
    under = df[df["bucket"] == "under"].sort_values("war_value", ascending=True)
    return over, delivered, under


def render_markdown(df: pd.DataFrame, season: int) -> str:
    lines: List[str] = []
    lines.append("## Preseason hype - who delivered?")
    lines.append(f"_Based on preseason predictions and {season} season WAR among hyped players._")
    lines.append("")

    over, delivered, under = bucket_war(df)

    pos_map = {
        "1": "P",
        "2": "C",
        "3": "1B",
        "4": "2B",
        "5": "3B",
        "6": "SS",
        "7": "LF",
        "8": "CF",
        "9": "RF",
    }

    def norm_pos(val: str) -> str:
        v = (val or "").strip()
        if not v:
            return ""
        if v in pos_map:
            return pos_map[v]
        return v

    def emit(title: str, bucket: pd.DataFrame) -> None:
        lines.append(f"**{title}**")
        if bucket.empty:
            lines.append("- (none)")
            lines.append("")
            return
        for _, row in bucket.iterrows():
            pos_raw = str(row.get("position", "") or "").strip()
            pos = norm_pos(pos_raw)
            team = row["team_abbr"]
            if pos:
                label = f"{row['full_name']} ({pos}, {team})"
            else:
                label = f"{row['full_name']} ({team})"
            lines.append(f"- {label} - WAR: {row['war_value']:.2f}")
        lines.append("")

    emit("Over-delivered", over)
    emit("Delivered", delivered)
    emit("Under-delivered", under)

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def build_preseason_hype_fragment(season: int, league_id: int, preseason_html: Path | None, output_md: Path | None) -> Path:
    root = get_repo_root()
    if preseason_html is None:
        preseason_html = root / "csv" / "in" / "almanac_core" / str(season) / "leagues" / f"league_{league_id}_preseason_prediction_report.html"

    logging.info("Using preseason HTML: %s", preseason_html)
    hype = parse_preseason_players(preseason_html)
    if hype.empty:
        raise RuntimeError("No hype players parsed from preseason HTML; aborting.")

    profiles = load_dim_player_profile(root)
    players = load_players_csv(root)
    stats = load_season_war(root, season, league_id)
    html_war = load_html_war(root, season, hype["player_id"].tolist())
    resolved = resolve_players(hype, profiles, players, stats, html_war)

    md_text = render_markdown(resolved, season)

    out_dir = root / "csv" / "out" / "eb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_md if output_md else out_dir / f"eb_preseason_hype_{season}_league{league_id}.md"
    out_path.write_text(md_text, encoding="utf-8")
    logging.info("Wrote preseason hype fragment to %s", out_path)
    return out_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    html_path = Path(args.preseason_html) if args.preseason_html else None
    output_md = Path(args.output_md) if args.output_md else None

    out_path = build_preseason_hype_fragment(args.season, args.league_id, html_path, output_md)
    print(f"[OK] Wrote preseason hype brief to {out_path}")


if __name__ == "__main__":
    main()
