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


def load_optional(path: Path, label: str):
    if path.exists():
        df = pd.read_csv(path)
        print(f"{label} columns:", list(df.columns))
        return df
    print(f"{label} not found at: {path}")
    return None


def build_division_leaders(standings: pd.DataFrame) -> pd.DataFrame:
    leader_rows = []
    for (sub_league, division), grp in standings.groupby(["sub_league", "division"], dropna=False):
        grp_sorted = grp.sort_values(by=["win_pct", "run_diff"], ascending=[False, False])
        leader_rows.append(grp_sorted.iloc[0])
    leaders = pd.DataFrame(leader_rows).reset_index(drop=True)
    return pd.DataFrame(
        {
            "section": "division_leader",
            "subsection": leaders["division"],
            "team_abbr": leaders["team_abbr"],
            "team_name": leaders["team_name"],
            "sub_league": leaders["sub_league"],
            "division": leaders["division"],
            "wins": leaders["wins"],
            "losses": leaders["losses"],
            "win_pct": leaders["win_pct"],
            "run_diff": leaders["run_diff"],
            "note_metric_1": leaders["games"],
            "note_metric_2": "",
            "manager_name": "",
        }
    )


def build_power_top(power: pd.DataFrame) -> pd.DataFrame:
    top = power.sort_values(by="power_rank").head(5)
    return pd.DataFrame(
        {
            "section": "power_top5",
            "subsection": "",
            "team_abbr": top["team_abbr"],
            "team_name": top["team_name"],
            "sub_league": top["sub_league"],
            "division": top["division"],
            "wins": top["wins"],
            "losses": top["losses"],
            "win_pct": top["win_pct"],
            "run_diff": top["run_diff"],
            "note_metric_1": top["power_rank"],
            "note_metric_2": top["games"],
            "manager_name": "",
        }
    )


def build_change_section(
    change_df: pd.DataFrame, standings_lookup: pd.DataFrame, section: str
) -> pd.DataFrame:
    if change_df.empty:
        return pd.DataFrame(
            columns=[
                "section",
                "subsection",
                "team_abbr",
                "team_name",
                "sub_league",
                "division",
                "wins",
                "losses",
                "win_pct",
                "run_diff",
                "note_metric_1",
                "note_metric_2",
                "manager_name",
            ]
        )
    joined = change_df.merge(standings_lookup, on="team_abbr", how="left", validate="one_to_one")
    if joined["team_name"].isna().any():
        missing = joined[joined["team_name"].isna()]["team_abbr"].unique()
        raise ValueError(f"{section}: Missing standings info for teams: {missing}")
    return pd.DataFrame(
        {
            "section": section,
            "subsection": "",
            "team_abbr": joined["team_abbr"],
            "team_name": joined["team_name"],
            "sub_league": joined["sub_league"],
            "division": joined["division"],
            "wins": joined["wins"],
            "losses": joined["losses"],
            "win_pct": joined["win_pct"],
            "run_diff": joined["run_diff"],
            "note_metric_1": joined.get("delta_wins", 0),
            "note_metric_2": joined.get("delta_run_diff", 0),
            "manager_name": "",
        }
    )


def build_manager_spotlight(managers: pd.DataFrame) -> pd.DataFrame:
    spotlight = managers.sort_values(by=["win_pct", "run_diff"], ascending=[False, False]).head(3)
    return pd.DataFrame(
        {
            "section": "manager_spotlight",
            "subsection": "",
            "team_abbr": spotlight["team_abbr"],
            "team_name": spotlight["team_name"],
            "sub_league": spotlight["sub_league"],
            "division": spotlight["division"],
            "wins": spotlight["wins"],
            "losses": spotlight["losses"],
            "win_pct": spotlight["win_pct"],
            "run_diff": spotlight["run_diff"],
            "note_metric_1": spotlight["manager_career_win_pct"],
            "note_metric_2": spotlight["manager_total_seasons"],
            "manager_name": spotlight["manager_name"],
        }
    )


def main(dry_run: bool = False):
    standings_path = STAR_DIR / "monday_1981_standings_by_division.csv"
    power_path = STAR_DIR / "monday_1981_power_ranking.csv"
    manager_path = STAR_DIR / "fact_manager_scorecard_1981_current.csv"
    risers_path = STAR_DIR / "monday_1981_risers.csv"
    fallers_path = STAR_DIR / "monday_1981_fallers.csv"

    if not all(ensure_file(p) for p in [standings_path, power_path, manager_path]):
        return

    standings = pd.read_csv(standings_path)
    power = pd.read_csv(power_path)
    managers = pd.read_csv(manager_path)
    print("Standings columns:", list(standings.columns))
    print("Power columns:", list(power.columns))
    print("Manager scorecard columns:", list(managers.columns))

    risers = load_optional(risers_path, "Risers file")
    fallers = load_optional(fallers_path, "Fallers file")

    sections = []
    sections.append(build_division_leaders(standings))
    sections.append(build_power_top(power))

    standings_lookup = standings[
        ["team_abbr", "team_name", "sub_league", "division", "wins", "losses", "win_pct", "run_diff"]
    ].copy()

    if risers is not None:
        risers_sorted = risers.sort_values(
            by=["delta_wins", "delta_run_diff", "delta_win_pct"], ascending=[False, False, False]
        ).head(5)
        sections.append(build_change_section(risers_sorted, standings_lookup, "riser"))
    else:
        print("No risers file; skipping riser section.")

    if fallers is not None:
        fallers_sorted = fallers.sort_values(
            by=["delta_wins", "delta_run_diff", "delta_win_pct"], ascending=[True, True, True]
        ).head(5)
        sections.append(build_change_section(fallers_sorted, standings_lookup, "faller"))
    else:
        print("No fallers file; skipping faller section.")

    sections.append(build_manager_spotlight(managers))

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
