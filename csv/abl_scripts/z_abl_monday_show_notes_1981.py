from pathlib import Path
import argparse
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]  # abl_csv_repo/csv
STAR_DIR = CSV_ROOT / "out" / "star_schema"


def ensure_file(path: Path) -> bool:
    if not path.exists():
        print(f"ERROR: required file missing: {path}")
        return False
    return True


def _blank_series(length: int):
    return pd.Series(["" for _ in range(length)])


def _repeat(value, length: int):
    return [value] * length


def main(dry_run: bool = False):
    standings_path = STAR_DIR / "monday_1981_standings_by_division.csv"
    power_path = STAR_DIR / "monday_1981_power_ranking.csv"
    mgr_path = STAR_DIR / "fact_manager_scorecard_1981_current.csv"
    risers_path = STAR_DIR / "monday_1981_risers.csv"
    fallers_path = STAR_DIR / "monday_1981_fallers.csv"

    required = [standings_path, power_path, mgr_path]
    if not all(ensure_file(p) for p in required):
        return

    standings = pd.read_csv(standings_path)
    power = pd.read_csv(power_path)
    mgr = pd.read_csv(mgr_path)

    risers = pd.read_csv(risers_path) if risers_path.exists() else None
    fallers = pd.read_csv(fallers_path) if fallers_path.exists() else None

    print("Standings columns:", list(standings.columns))
    print("Power columns:", list(power.columns))
    print("Manager scorecard columns:", list(mgr.columns))
    if risers is not None:
        print("Risers columns:", list(risers.columns))
    if fallers is not None:
        print("Fallers columns:", list(fallers.columns))

    sections = []

    standings = standings.copy()
    if "games" not in standings.columns:
        standings["games"] = standings["wins"] + standings["losses"]

    div_rows = []
    for (sl, div), group in standings.groupby(["sub_league", "division"]):
        top = group.sort_values(by=["win_pct", "run_diff"], ascending=[False, False]).head(1)
        div_rows.append(top.iloc[0])
    if div_rows:
        div_leaders = pd.DataFrame(div_rows).reset_index(drop=True)
        gb_col = "GB" if "GB" in div_leaders.columns else None
        note_metric_2 = (
            div_leaders[gb_col]
            if gb_col
            else _blank_series(len(div_leaders))
        )
        sections.append(
            pd.DataFrame(
                {
                    "section": _repeat("division_leader", len(div_leaders)),
                    "subsection": div_leaders["division"],
                    "team_abbr": div_leaders["team_abbr"],
                    "team_name": div_leaders["team_name"],
                    "sub_league": div_leaders["sub_league"],
                    "division": div_leaders["division"],
                    "wins": div_leaders["wins"],
                    "losses": div_leaders["losses"],
                    "win_pct": div_leaders["win_pct"],
                    "run_diff": div_leaders["run_diff"],
                    "note_metric_1": div_leaders["games"],
                    "note_metric_2": note_metric_2.reset_index(drop=True),
                    "manager_name": _blank_series(len(div_leaders)),
                }
            )
        )

    power = power.copy()
    if "games" not in power.columns:
        power["games"] = power["wins"] + power["losses"]
    if {"team_abbr", "power_rank"}.issubset(power.columns):
        top_power = power.sort_values("power_rank").head(5).reset_index(drop=True)
        sections.append(
            pd.DataFrame(
                {
                    "section": _repeat("power_top5", len(top_power)),
                    "subsection": _blank_series(len(top_power)),
                    "team_abbr": top_power["team_abbr"],
                    "team_name": top_power["team_name"],
                    "sub_league": top_power["sub_league"],
                    "division": top_power["division"],
                    "wins": top_power["wins"],
                    "losses": top_power["losses"],
                    "win_pct": top_power["win_pct"],
                    "run_diff": top_power["run_diff"],
                    "note_metric_1": top_power["power_rank"],
                    "note_metric_2": top_power["games"],
                    "manager_name": _blank_series(len(top_power)),
                }
            )
        )

    def build_change_section(df: pd.DataFrame, name: str):
        if df is None or df.empty or "team_abbr" not in df.columns:
            return None
        joined = df.merge(
            standings,
            on="team_abbr",
            how="left",
            suffixes=("", "_standings"),
        ).reset_index(drop=True)
        if joined["team_name"].isna().any():
            missing = joined[joined["team_name"].isna()]["team_abbr"].unique()
            raise ValueError(f"{name}: missing standings info for teams {missing}")
        note_metric_1 = (
            joined["delta_wins"] if "delta_wins" in joined.columns else pd.Series([0] * len(joined))
        )
        note_metric_2 = (
            joined["delta_run_diff"]
            if "delta_run_diff" in joined.columns
            else pd.Series([0] * len(joined))
        )
        return pd.DataFrame(
            {
                "section": _repeat(name, len(joined)),
                "subsection": _blank_series(len(joined)),
                "team_abbr": joined["team_abbr"],
                "team_name": joined["team_name"],
                "sub_league": joined["sub_league"],
                "division": joined["division"],
                "wins": joined["wins"],
                "losses": joined["losses"],
                "win_pct": joined["win_pct"],
                "run_diff": joined["run_diff"],
                "note_metric_1": note_metric_1.reset_index(drop=True),
                "note_metric_2": note_metric_2.reset_index(drop=True),
                "manager_name": _blank_series(len(joined)),
            }
        )

    if risers is not None:
        risers_sorted = risers.sort_values(
            by=["delta_wins", "delta_run_diff", "delta_win_pct"],
            ascending=[False, False, False],
        ).head(5)
        riser_section = build_change_section(risers_sorted, "riser")
        if riser_section is not None:
            sections.append(riser_section)

    if fallers is not None:
        fallers_sorted = fallers.sort_values(
            by=["delta_wins", "delta_run_diff", "delta_win_pct"],
            ascending=[True, True, True],
        ).head(5)
        faller_section = build_change_section(fallers_sorted, "faller")
        if faller_section is not None:
            sections.append(faller_section)

    mgr_sorted = mgr.sort_values(by=["win_pct", "run_diff"], ascending=[False, False]).head(3).reset_index(drop=True)
    note_metric_1_mgr = (
        mgr_sorted["manager_career_win_pct"]
        if "manager_career_win_pct" in mgr_sorted.columns
        else _blank_series(len(mgr_sorted))
    )
    note_metric_2_mgr = (
        mgr_sorted["manager_total_seasons"]
        if "manager_total_seasons" in mgr_sorted.columns
        else _blank_series(len(mgr_sorted))
    )
    sections.append(
        pd.DataFrame(
            {
                "section": _repeat("manager_spotlight", len(mgr_sorted)),
                "subsection": _blank_series(len(mgr_sorted)),
                "team_abbr": mgr_sorted["team_abbr"],
                "team_name": mgr_sorted["team_name"],
                "sub_league": mgr_sorted["sub_league"],
                "division": mgr_sorted["division"],
                "wins": mgr_sorted["wins"],
                "losses": mgr_sorted["losses"],
                "win_pct": mgr_sorted["win_pct"],
                "run_diff": mgr_sorted["run_diff"],
                "note_metric_1": note_metric_1_mgr.reset_index(drop=True),
                "note_metric_2": note_metric_2_mgr.reset_index(drop=True),
                "manager_name": mgr_sorted["manager_name"],
            }
        )
    )

    if not sections:
        print("No sections built; check inputs.")
        return

    show_notes = pd.concat(sections, ignore_index=True)
    out_path = STAR_DIR / "monday_1981_show_notes.csv"
    if not dry_run:
        show_notes.to_csv(out_path, index=False)
        print("MONDAY SHOW NOTES: wrote", out_path)
    print(show_notes.head(15))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output CSV")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
