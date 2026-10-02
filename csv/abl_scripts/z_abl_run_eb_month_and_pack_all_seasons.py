import sys
import subprocess
from pathlib import Path


def run_cmd(args, cwd: Path) -> None:
    """
    Run a subprocess command, echoing it first.
    Raise SystemExit if it fails.
    """
    print(">>> Running:", " ".join(str(a) for a in args))
    result = subprocess.run(args, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {result.returncode}: "
            + " ".join(str(a) for a in args)
        )


def main() -> None:
    """
    For all ABL seasons 1972–1980 (inclusive), league_id=200:

    1) Build the calendar-month Month of Glory / Month of Misery fragment via
       z_abl_month_glory_misery_any.py.

    2) Rebuild the EB regular-season pack via _run_eb_regular_season_any.py,
       which will splice in the month fragment.

    Assumes this file lives at:
        csv/abl_scripts/z_abl_run_eb_month_and_pack_all_seasons.py

    Repo root is two levels up from this file (abl_csv_repo).
    """
    # Resolve repo root relative to this script location
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    python_exe = sys.executable
    league_id = 200

    # All nine ABL seasons currently in scope
    seasons = list(range(1972, 1981))

    for season in seasons:
        print("\n" + "=" * 72)
        print(f"Processing ABL season {season} (league_id={league_id})")
        print("=" * 72)

        # 1) Build Month of Glory / Month of Misery fragment (calendar-month engine)
        run_cmd(
            [
                python_exe,
                "csv/abl_scripts/z_abl_month_glory_misery_any.py",
                "--season",
                str(season),
                "--league-id",
                str(league_id),
            ],
            cwd=repo_root,
        )

        # 2) Rebuild EB regular-season pack (which will splice in the month fragment)
        run_cmd(
            [
                python_exe,
                "csv/abl_scripts/_run_eb_regular_season_any.py",
                "--season",
                str(season),
                "--league-id",
                str(league_id),
            ],
            cwd=repo_root,
        )

    print("\nAll seasons completed successfully.")


if __name__ == "__main__":
    main()
