from pathlib import Path
import subprocess
import sys

SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"


def run_script(rel_path: str, args=None):
    if args is None:
        args = []
    script_path = CSV_ROOT / "abl_scripts" / rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Child script not found: {script_path}")
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n=== RUNNING: {cmd} ===")
    subprocess.run(cmd, check=True)
    print(f"=== COMPLETED: {rel_path} ===")


def main():
    prev_path = STAR_DIR / "fact_team_reporting_1981_prev.csv"
    curr_path = STAR_DIR / "fact_team_reporting_1981_current.csv"

    if curr_path.exists():
        print(f"Found existing current snapshot: {curr_path}")
        print(f"Rotating current -> prev at: {prev_path}")
        if prev_path.exists():
            prev_path.unlink()
        curr_path.rename(prev_path)
        print("Rotation complete.")
    else:
        print("No existing current snapshot found.")
        print("This appears to be a first-run scenario; skipping rotation.")

    run_script("z_abl_current_team_snapshot.py")
    run_script("z_abl_weekly_change_1981.py")
    run_script("z_abl_monday_packet_1981.py")
    run_script("z_abl_manager_scorecard_1981.py")
    run_script("z_abl_monday_show_notes_1981.py")

    print("\n=== WEEKLY 1981 PIPELINE COMPLETE ===")
    print("Updated artifacts:")
    print(" - fact_team_reporting_1981_prev.csv")
    print(" - fact_team_reporting_1981_current.csv")
    print(" - fact_team_reporting_1981_weekly_change.csv")
    print(" - monday_1981_standings_by_division.csv")
    print(" - monday_1981_power_ranking.csv")
    print(" - monday_1981_risers.csv")
    print(" - monday_1981_fallers.csv")
    print(" - fact_manager_scorecard_1981_current.csv")
    print(" - monday_1981_show_notes.csv")


if __name__ == "__main__":
    main()
