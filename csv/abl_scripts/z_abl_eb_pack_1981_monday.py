from pathlib import Path
import sys
import pandas as pd
from textwrap import dedent


def log_err(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        log_err(f"[ERROR] Missing {label} file: {path}")
        sys.exit(1)


def pick_column(df: pd.DataFrame, keywords, numeric=False):
    """
    Pick a column whose name contains ALL of the given keywords (case-insensitive).
    If numeric=True, restrict to numeric dtypes.
    """
    cols = list(df.columns)
    lowered = [c.lower() for c in cols]
    candidates = []

    for col, low in zip(cols, lowered):
        if all(k.lower() in low for k in keywords):
            candidates.append(col)

    if numeric:
        candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]

    if not candidates:
        raise ValueError(
            f"Could not find column matching keywords {keywords} in {list(df.columns)}"
        )

    candidates.sort()
    chosen = candidates[0]
    print(f"[INFO] Using column '{chosen}' for keywords {keywords}")
    return chosen


def maybe_filter_league(df: pd.DataFrame) -> pd.DataFrame:
    if "league_id" in df.columns:
        before = len(df)
        df = df[df["league_id"] == 200].copy()
        after = len(df)
        print(f"[INFO] Filtered league_id==200: {before} -> {after} rows")
    return df


def enforce_team_ids(df: pd.DataFrame) -> None:
    if "team_id" in df.columns:
        bad = df.loc[~df["team_id"].between(1, 24)]
        if not bad.empty:
            raise ValueError(
                "Found team_id outside 1..24:\n"
                + str(bad[['team_id']].drop_duplicates())
            )


def load_df(path: Path, label: str) -> pd.DataFrame:
    ensure_exists(path, label)
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {label}: {path} ({len(df)} rows)")
    df = maybe_filter_league(df)
    enforce_team_ids(df)
    return df


# ---------- SECTION BUILDERS ----------


def build_standings_section(standings: pd.DataFrame) -> str:
    # Division column
    div_col = pick_column(standings, ["div"])

    # Team name column
    team_name_col = None
    for cand in ["team_name", "name", "team"]:
        matches = [c for c in standings.columns if c.lower() == cand]
        if matches:
            team_name_col = matches[0]
            print(f"[INFO] Using '{team_name_col}' as team name column in standings")
            break
    if team_name_col is None:
        non_id_cols = [
            c for c in standings.columns if not any(x in c.lower() for x in ["id"])
        ]
        if not non_id_cols:
            raise ValueError("Could not determine team name column in standings")
        team_name_col = non_id_cols[0]
        print(f"[WARN] Fallback team name column in standings: '{team_name_col}'")

    # Wins / losses / pct
    wins_col = "wins" if "wins" in standings.columns else pick_column(
        standings, ["wins"]
    )
    print(f"[INFO] Using '{wins_col}' as wins column in standings")

    losses_candidates = [c for c in standings.columns if "loss" in c.lower()]
    if losses_candidates:
        losses_candidates.sort()
        losses_col = losses_candidates[0]
        print(f"[INFO] Using '{losses_col}' as losses column in standings")
    else:
        losses_col = None
        print("[WARN] No losses column found; will omit L in W-L if needed")

    try:
        pct_col = pick_column(standings, ["pct"], numeric=True)
    except ValueError:
        pct_col = None
        print("[WARN] No win_pct column found; sorting by wins only")

    lines = ["Division leaders (top 2 per division):"]

    for div_name, grp in standings.groupby(div_col):
        grp = grp.copy()
        if pct_col is not None:
            grp = grp.sort_values(by=pct_col, ascending=False)
        else:
            grp = grp.sort_values(by=wins_col, ascending=False)

        top_two = grp.head(2)
        if top_two.empty:
            continue

        records = []
        for _, row in top_two.iterrows():
            team = str(row[team_name_col])
            w = int(row[wins_col]) if pd.notna(row[wins_col]) else 0
            if losses_col and losses_col in row:
                l = int(row[losses_col]) if pd.notna(row[losses_col]) else 0
                rec = f"{team} ({w}-{l})"
            else:
                rec = f"{team} ({w} W)"
            records.append(rec)

        line = f"- {div_name}: " + ", ".join(records)
        lines.append(line)

    return "\n".join(lines)


def build_power_section(
    power: pd.DataFrame,
    standings: pd.DataFrame | None = None,
) -> str:
    """
    Build the power board section for EB using ONLY csv/out/abl_power_rankings.csv.

    Expected columns in `power`:
      - rank (int)
      - team_name (str, full name)
      - points (numeric or string)
      - tendency (optional, e.g. '++', '+', 'o', '-', '--')
    """
    print(
        "[INFO] Building power board from abl_power_rankings.csv using rank/team_name/points/tendency"
    )

    df = power.copy()

    # Normalize headers
    rename_map = {}
    if "Rank" in df.columns:
        rename_map["Rank"] = "rank"
    if "Team Name" in df.columns:
        rename_map["Team Name"] = "team_name"
    elif "Team" in df.columns:
        rename_map["Team"] = "team_name"
    if "Points" in df.columns:
        rename_map["Points"] = "points"
    tendency_col = None
    for col in df.columns:
        if col in rename_map:
            continue
        if "tend" in col.lower() or "trend" in col.lower():
            rename_map[col] = "tendency"
            tendency_col = "tendency"
            break
    df = df.rename(columns=rename_map)
    if tendency_col is None and "tendency" in df.columns:
        tendency_col = "tendency"

    # Validate required columns
    required = {"rank", "team_name", "points"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(
            "Power ranking CSV missing required columns: rank/team_name/points"
        )

    # Sort by rank and take top 9
    df = df.sort_values("rank", ascending=True)
    top = df.head(9)

    lines: list[str] = []
    lines.append("Power board (top 9 by power rank):")

    for _, row in top.iterrows():
        rank_val = int(row["rank"])
        name = str(row["team_name"]).strip()

        pts = row["points"]
        if pd.api.types.is_number(pts):
            points_str = f"{float(pts):.1f}"
        else:
            points_str = str(pts)

        tendency_str = ""
        if tendency_col and tendency_col in row.index:
            val = str(row[tendency_col]).strip()
            if val and val.lower() != "nan":
                tendency_str = f" [{val}]"

        lines.append(f"{rank_val}) {name} {points_str}{tendency_str}")

    return "\n".join(lines)


def _pick_change_col(df: pd.DataFrame, label: str) -> str:
    # Prefer delta_win_pct, then delta_run_diff, then any 'delta'
    for key in ["delta_win_pct", "delta_run_diff"]:
        matches = [c for c in df.columns if c.lower() == key]
        if matches:
            print(f"[INFO] Using '{matches[0]}' as primary change column in {label}")
            return matches[0]

    delta_candidates = [c for c in df.columns if "delta" in c.lower()]
    if not delta_candidates:
        raise ValueError(f"No delta/change column found in {label}: {df.columns}")
    delta_candidates.sort()
    print(
        f"[WARN] Fallback change column in {label}: '{delta_candidates[0]}' "
        f"from candidates {delta_candidates}"
    )
    return delta_candidates[0]


def build_risers_fallers_section(
    risers: pd.DataFrame, fallers: pd.DataFrame, standings: pd.DataFrame
) -> str:
    # Team name columns for risers/fallers
    def team_name_col_for(df: pd.DataFrame, label: str) -> str:
        for cand in ["team_name", "name", "team"]:
            matches = [c for c in df.columns if c.lower() == cand]
            if matches:
                print(f"[INFO] Using '{matches[0]}' as team name column in {label}")
                return matches[0]
        non_id = [
            c for c in df.columns if not any(x in c.lower() for x in ["id"])
        ]
        if not non_id:
            raise ValueError(f"Could not determine team name column in {label}")
        print(f"[WARN] Fallback team name column in {label}: '{non_id[0]}'")
        return non_id[0]

    riser_team_col = team_name_col_for(risers, "risers")
    faller_team_col = team_name_col_for(fallers, "fallers")

    change_col_r = _pick_change_col(risers, "risers")
    change_col_f = _pick_change_col(fallers, "fallers")

    # Join back to standings on team_abbr if possible
    merge_key = None
    for key in ["team_abbr", "team_id", "teamid"]:
        if key in risers.columns and key in standings.columns:
            merge_key = key
            break

    if merge_key:
        risers_merged = risers.merge(
            standings, on=merge_key, how="left", suffixes=("", "_stand")
        )
        fallers_merged = fallers.merge(
            standings, on=merge_key, how="left", suffixes=("", "_stand")
        )
        print(f"[INFO] Joined risers/fallers to standings on {merge_key}")
    else:
        risers_merged = risers
        fallers_merged = fallers
        print(
            "[WARN] Could not join risers/fallers to standings; will omit records if needed"
        )

    # Wins / losses
    wins_col = None
    losses_col = None
    try:
        wins_col = "wins" if "wins" in risers_merged.columns else pick_column(
            risers_merged, ["wins"]
        )
        loss_candidates = [
            c for c in risers_merged.columns if "loss" in c.lower()
        ]
        if loss_candidates:
            loss_candidates.sort()
            losses_col = loss_candidates[0]
    except ValueError:
        print("[WARN] Could not identify W/L columns for risers/fallers records")

    lines = []

    # Risers: biggest positive change
    r_sorted = risers_merged.sort_values(by=change_col_r, ascending=False).head(3)
    lines.append("Risers (top 3 by change):")
    for _, row in r_sorted.iterrows():
        team = str(row[riser_team_col])
        delta_raw = pd.to_numeric(row[change_col_r], errors="coerce")
        change_str = f"{float(delta_raw) * 100:+.1f}%" if pd.notna(delta_raw) else "n/a"
        if wins_col and losses_col and wins_col in row and losses_col in row:
            try:
                w = int(row[wins_col])
                l = int(row[losses_col])
                rec = f"{team} (change={change_str}, record={w}-{l})"
            except Exception:
                rec = f"{team} (change={change_str})"
        else:
            rec = f"{team} (change={change_str})"
        lines.append(f"- {rec}")

    # Fallers: biggest negative change
    f_sorted = fallers_merged.sort_values(by=change_col_f, ascending=True).head(3)
    lines.append("\nFallers (top 3 by negative change):")
    for _, row in f_sorted.iterrows():
        team = str(row[faller_team_col])
        delta_raw = pd.to_numeric(row[change_col_f], errors="coerce")
        change_str = f"{float(delta_raw) * 100:+.1f}%" if pd.notna(delta_raw) else "n/a"
        if wins_col and losses_col and wins_col in row and losses_col in row:
            try:
                w = int(row[wins_col])
                l = int(row[losses_col])
                rec = f"{team} (change={change_str}, record={w}-{l})"
            except Exception:
                rec = f"{team} (change={change_str})"
        else:
            rec = f"{team} (change={change_str})"
        lines.append(f"- {rec}")

    return "\n".join(lines)


def build_manager_section(
    mgr_score: pd.DataFrame, mgr_dim: pd.DataFrame
) -> str | None:
    # Manager id column
    mgr_id_col = None
    for cand in ["manager_id", "person_id"]:
        if cand in mgr_score.columns:
            mgr_id_col = cand
            print(f"[INFO] Using '{mgr_id_col}' as manager id column in scorecard")
            break
    if mgr_id_col is None:
        print("[WARN] No manager_id/person_id column found; skipping manager section")
        return None

    # Align DIM id column
    if mgr_id_col not in mgr_dim.columns:
        possible = [
            c for c in mgr_dim.columns
            if "manager" in c.lower() and "id" in c.lower()
        ]
        if not possible:
            print(
                "[WARN] Manager DIM has no matching id column; skipping manager section"
            )
            return None
        mgr_dim_id_col = possible[0]
        mgr_dim = mgr_dim.rename(columns={mgr_dim_id_col: mgr_id_col})
        print(f"[INFO] Renamed DIM manager id column '{mgr_dim_id_col}' -> '{mgr_id_col}'")
    else:
        print("[INFO] DIM manager id column matches scorecard")

    merged = mgr_score.merge(mgr_dim, on=mgr_id_col, how="left", suffixes=("", "_dim"))

    # Find a score/rating column if one exists
    score_cols = [
        c
        for c in merged.columns
        if (("score" in c.lower()) or ("rating" in c.lower()))
        and pd.api.types.is_numeric_dtype(merged[c])
    ]
    if not score_cols:
        print(
            "[WARN] No numeric 'score' or 'rating' column found; skipping manager section"
        )
        return None

    score_cols.sort()
    score_col = score_cols[0]
    print(f"[INFO] Using '{score_col}' as manager score column")

    # Manager full name
    first_col = None
    last_col = None
    for c in merged.columns:
        lc = c.lower()
        if "first" in lc and "name" in lc:
            first_col = c
        if "last" in lc and "name" in lc:
            last_col = c
    if not first_col or not last_col:
        print(
            "[WARN] Manager DIM missing clear first/last name; will use fallback name field"
        )
        name_col_candidates = [
            c
            for c in merged.columns
            if "name" in c.lower() and not any(
                x in c.lower() for x in ["first", "last"]
            )
        ]
        name_col_candidates.sort()
        name_col = name_col_candidates[0] if name_col_candidates else mgr_id_col
        merged["mgr_full_name"] = merged[name_col].astype(str)
    else:
        merged["mgr_full_name"] = (
            merged[first_col].astype(str).str.strip()
            + " "
            + merged[last_col].astype(str).str.strip()
        )

    # Team name, if available
    team_name_col = None
    for cand in ["team_name", "name", "team"]:
        matches = [c for c in merged.columns if c.lower() == cand]
        if matches:
            team_name_col = matches[0]
            print(f"[INFO] Using '{team_name_col}' as manager team name column")
            break

    up = merged.sort_values(by=score_col, ascending=False).head(3)
    down = merged.sort_values(by=score_col, ascending=True).head(3)

    lines = []
    lines.append("Managers – stock up (top 3 by manager score):")
    for _, row in up.iterrows():
        name = str(row["mgr_full_name"])
        score = row[score_col]
        if team_name_col and team_name_col in row:
            team = str(row[team_name_col])
            lines.append(f"- {name} ({team}, score={score})")
        else:
            lines.append(f"- {name} (score={score})")

    lines.append("\nManagers – stock down (bottom 3 by manager score):")
    for _, row in down.iterrows():
        name = str(row["mgr_full_name"])
        score = row[score_col]
        if team_name_col and team_name_col in row:
            team = str(row[team_name_col])
            lines.append(f"- {name} ({team}, score={score})")
        else:
            lines.append(f"- {name} (score={score})")

    return "\n".join(lines)


# ---------- MAIN ----------


def main():
    # repo root = two levels up from this script: .../abl_csv_repo
    root = Path(__file__).resolve().parents[2]
    star_schema_dir = root / "csv" / "out" / "star_schema"
    csv_out_dir = root / "csv" / "out" / "csv_out"
    text_out_dir = root / "csv" / "out" / "text_out"

    text_out_dir.mkdir(parents=True, exist_ok=True)

    standings_path = star_schema_dir / "monday_1981_standings_by_division.csv"
    power_path = root / "csv" / "out" / "abl_power_rankings.csv"
    risers_path = star_schema_dir / "monday_1981_risers.csv"
    fallers_path = star_schema_dir / "monday_1981_fallers.csv"
    mgr_score_path = star_schema_dir / "fact_manager_scorecard_1981_current.csv"
    mgr_dim_path = csv_out_dir / "z_ABL_DIM_Managers.csv"

    # Load data
    standings = load_df(standings_path, "Monday standings by division")
    power = load_df(power_path, "ABL power ranking")
    risers = load_df(risers_path, "Monday risers")
    fallers = load_df(fallers_path, "Monday fallers")
    mgr_score = load_df(mgr_score_path, "Manager scorecard current")
    mgr_dim = load_df(mgr_dim_path, "DIM managers")

    # Build sections
    standings_section = build_standings_section(standings)
    power_section = build_power_section(power, standings)
    rf_section = build_risers_fallers_section(risers, fallers, standings)
    mgr_section = build_manager_section(mgr_score, mgr_dim)

    text_parts = [
        "EB DATA PACK – MONDAY 1981 SNAPSHOT",
        "",
        standings_section,
        "",
        power_section,
        "",
        rf_section,
    ]

    if mgr_section:
        text_parts.extend(["", mgr_section])

    full_text = dedent("\n".join(text_parts)).strip() + "\n"

    eb_pack_path = text_out_dir / "eb_data_pack_1981_monday.txt"
    eb_pack_path.write_text(full_text, encoding="utf-8")

    print("[INFO] EB Data Pack written to:", eb_pack_path)
    print("\n" + full_text)


if __name__ == "__main__":
    main()
