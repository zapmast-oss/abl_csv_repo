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
    Raise ValueError with clear message if not found.
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

    # deterministic: sort alphabetically
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
                f"Found team_id outside 1..24:\n{bad[['team_id']].drop_duplicates()}"
            )


def load_df(path: Path, label: str) -> pd.DataFrame:
    ensure_exists(path, label)
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {label}: {path} ({len(df)} rows)")
    df = maybe_filter_league(df)
    enforce_team_ids(df)
    return df


def build_standings_section(standings: pd.DataFrame) -> str:
    # Identify basic columns
    div_col = pick_column(standings, ["div"])
    team_name_col = None
    for cand in ["team_name", "name", "team"]:
        matches = [c for c in standings.columns if c.lower() == cand]
        if matches:
            team_name_col = matches[0]
            print(f"[INFO] Using '{team_name_col}' as team name column in standings")
            break
    if team_name_col is None:
        # Fallback: first non-id column
        non_id_cols = [
            c for c in standings.columns if not any(x in c.lower() for x in ["id"])
        ]
        if not non_id_cols:
            raise ValueError("Could not determine team name column in standings")
        team_name_col = non_id_cols[0]
        print(f"[WARN] Fallback team name column in standings: '{team_name_col}'")

    wins_col = pick_column(standings, ["win"])
    try:
        pct_col = pick_column(standings, ["pct"], numeric=True)
    except ValueError:
        pct_col = None
        print("[WARN] No win_pct column found; will sort by wins only")
    losses_col_candidates = [c for c in standings.columns if "loss" in c.lower()]
    if losses_col_candidates:
        losses_col_candidates.sort()
        losses_col = losses_col_candidates[0]
        print(f"[INFO] Using '{losses_col}' as losses column in standings")
    else:
        losses_col = None
        print("[WARN] No losses column found; will omit L in W-L if needed")

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


def build_power_section(power: pd.DataFrame, standings: pd.DataFrame) -> str:
    rank_col = pick_column(power, ["rank"])
    # team name column
    team_name_col = None
    for cand in ["team_name", "name", "team"]:
        matches = [c for c in power.columns if c.lower() == cand]
        if matches:
            team_name_col = matches[0]
            print(f"[INFO] Using '{team_name_col}' as team name column in power")
            break
    if team_name_col is None:
        non_id_cols = [
            c for c in power.columns if not any(x in c.lower() for x in ["id"])
        ]
        if not non_id_cols:
            raise ValueError("Could not determine team name column in power rankings")
        team_name_col = non_id_cols[0]
        print(f"[WARN] Fallback team name column in power: '{team_name_col}'")

    # attempt to join with standings for W-L
    merge_key = None
    for key in ["team_id", "teamid"]:
        if key in power.columns and key in standings.columns:
            merge_key = key
            break

    if merge_key:
        merged = power.merge(
            standings, on=merge_key, how="left", suffixes=("", "_stand")
        )
        print(f"[INFO] Joined power to standings on {merge_key}")
    else:
        merged = power
        print("[WARN] Could not join power to standings; will omit W-L in this section")

    # try to pick wins/losses for merged
    wins_col = None
    losses_col = None
    if merge_key:
        try:
            wins_col = pick_column(merged, ["win"])
            loss_candidates = [c for c in merged.columns if "loss" in c.lower()]
            if loss_candidates:
                loss_candidates.sort()
                losses_col = loss_candidates[0]
        except ValueError:
            print("[WARN] Could not identify W/L columns for power section")

    top = merged.sort_values(by=rank_col, ascending=True).head(5)

    lines = ["Power board (top 5 by power rank):"]
    for _, row in top.iterrows():
        rank = int(row[rank_col]) if pd.notna(row[rank_col]) else row[rank_col]
        team = str(row[team_name_col])
        if wins_col and losses_col and wins_col in row and losses_col in row:
            try:
                w = int(row[wins_col])
                l = int(row[losses_col])
                rec = f"{team} ({w}-{l})"
            except Exception:
                rec = team
        else:
            rec = team
        lines.append(f"{rank}) {rec}")

    return "\n".join(lines)


def build_risers_fallers_section(
    risers: pd.DataFrame, fallers: pd.DataFrame, standings: pd.DataFrame
) -> str:
    # identify team name col for risers/fallers
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

    # change/delta column
    change_col_r = pick_column(risers, ["change"]) if any(
        "change" in c.lower() for c in risers.columns
    ) else pick_column(risers, ["delta"])
    change_col_f = pick_column(fallers, ["change"]) if any(
        "change" in c.lower() for c in fallers.columns
    ) else pick_column(fallers, ["delta"])

    # join with standings for record if possible
    merge_key = None
    for key in ["team_id", "teamid"]:
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

    # try to get wins/losses from merged if available
    wins_col = None
    losses_col = None
    try:
        wins_col = pick_column(risers_merged, ["win"])
        loss_candidates = [
            c for c in risers_merged.columns if "loss" in c.lower()
        ]
        if loss_candidates:
            loss_candidates.sort()
            losses_col = loss_candidates[0]
    except ValueError:
        print("[WARN] Could not identify W/L columns for risers/fallers records")

    lines = []

    # Risers
    r_sorted = risers_merged.sort_values(by=change_col_r, ascending=False).head(3)
    lines.append("Risers (top 3 by change):")
    for _, row in r_sorted.iterrows():
        team = str(row[riser_team_col])
        delta = row[change_col_r]
        if wins_col and losses_col and wins_col in row and losses_col in row:
            try:
                w = int(row[wins_col])
                l = int(row[losses_col])
                rec = f"{team} (change={delta}, record={w}-{l})"
            except Exception:
                rec = f"{team} (change={delta})"
        else:
            rec = f"{team} (change={delta})"
        lines.append(f"- {rec}")

    # Fallers
    f_sorted = fallers_merged.sort_values(by=change_col_f, ascending=True).head(3)
    lines.append("\nFallers (top 3 by negative change):")
    for _, row in f_sorted.iterrows():
        team = str(row[faller_team_col])
        delta = row[change_col_f]
        if wins_col and losses_col and wins_col in row and losses_col in row:
            try:
                w = int(row[wins_col])
                l = int(row[losses_col])
                rec = f"{team} (change={delta}, record={w}-{l})"
            except Exception:
                rec = f"{team} (change={delta})"
        else:
            rec = f"{team} (change={delta})"
        lines.append(f"- {rec}")

    return "\n".join(lines)


def build_manager_section(
    mgr_score: pd.DataFrame, mgr_dim: pd.DataFrame
) -> str | None:
    # Need manager_id
    mgr_id_col = None
    for cand in ["manager_id", "person_id"]:
        if cand in mgr_score.columns:
            mgr_id_col = cand
            print(f"[INFO] Using '{mgr_id_col}' as manager id column in scorecard")
            break
    if mgr_id_col is None:
        print("[WARN] No manager_id/person_id column found; skipping manager section")
        return None

    # join with dim for names
    if mgr_id_col not in mgr_dim.columns:
        possible = [c for c in mgr_dim.columns if "manager" in c.lower() and "id" in c.lower()]
        if not possible:
            print("[WARN] Manager DIM has no matching id column; skipping manager section")
            return None
        mgr_dim_id_col = possible[0]
        mgr_dim = mgr_dim.rename(columns={mgr_dim_id_col: mgr_id_col})
        print(f"[INFO] Renamed DIM manager id column '{mgr_dim_id_col}' -> '{mgr_id_col}'")
    else:
        print("[INFO] DIM manager id column matches scorecard")

    merged = mgr_score.merge(mgr_dim, on=mgr_id_col, how="left", suffixes=("", "_dim"))

    # find score column
    score_cols = [
        c
        for c in merged.columns
        if (("score" in c.lower()) or ("rating" in c.lower()))
        and pd.api.types.is_numeric_dtype(merged[c])
    ]
    if not score_cols:
        print("[WARN] No numeric 'score' or 'rating' column found; skipping manager section")
        return None

    score_cols.sort()
    score_col = score_cols[0]
    print(f"[INFO] Using '{score_col}' as manager score column")

    # pick name columns
    first_col = None
    last_col = None
    for c in merged.columns:
        lc = c.lower()
        if "first" in lc and "name" in lc:
            first_col = c
        if "last" in lc and "name" in lc:
            last_col = c
    if not first_col or not last_col:
        print("[WARN] Manager DIM missing clear first/last name; will use fallback name field")
        name_col_candidates = [
            c
            for c in merged.columns
            if "name" in c.lower() and not any(x in c.lower() for x in ["first", "last"])
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

    # try to identify team name for context
    team_name_col = None
    for cand in ["team_name", "name", "team"]:
        matches = [c for c in merged.columns if c.lower() == cand]
        if matches:
            team_name_col = matches[0]
            print(f"[INFO] Using '{team_name_col}' as manager team name column")
            break

    # Stock up: top 3 by score
    up = merged.sort_values(by=score_col, ascending=False).head(3)
    # Stock down: bottom 3 by score
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


def main():
    # repo root = two levels up from this script: .../abl_csv_repo
    root = Path(__file__).resolve().parents[2]
    star_schema_dir = root / "csv" / "out" / "star_schema"
    csv_out_dir = root / "csv" / "out" / "csv_out"
    text_out_dir = root / "csv" / "out" / "text_out"

    text_out_dir.mkdir(parents=True, exist_ok=True)

    standings_path = star_schema_dir / "monday_1981_standings_by_division.csv"
    power_path = star_schema_dir / "monday_1981_power_ranking.csv"
    risers_path = star_schema_dir / "monday_1981_risers.csv"
    fallers_path = star_schema_dir / "monday_1981_fallers.csv"
    mgr_score_path = star_schema_dir / "fact_manager_scorecard_1981_current.csv"
    mgr_dim_path = csv_out_dir / "z_ABL_DIM_Managers.csv"

    # Load data
    standings = load_df(standings_path, "Monday standings by division")
    power = load_df(power_path, "Monday power ranking")
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
