from pathlib import Path
import shutil
import json
from datetime import datetime


SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"
CSV_OUT_DIR = CSV_ROOT / "out" / "csv_out"
ARCHIVE_ROOT = CSV_ROOT / "out" / "archive"
SEASON_ARCHIVE = ARCHIVE_ROOT / "season_1981"


def main():
    SEASON_ARCHIVE.mkdir(parents=True, exist_ok=True)

    required_star = [
        "fact_team_reporting_1981_current.csv",
        "fact_team_reporting_1981_prev.csv",
        "fact_team_reporting_1981_weekly_change.csv",
        "monday_1981_standings_by_division.csv",
        "monday_1981_power_ranking.csv",
        "fact_manager_scorecard_1981_current.csv",
        "monday_1981_show_notes.csv",
    ]

    optional_star = [
        "monday_1981_risers.csv",
        "monday_1981_fallers.csv",
    ]

    manager_dim_name = "z_ABL_DIM_Managers.csv"

    archived = []

    def copy_file(src: Path):
        if not src.exists():
            return False
        dst = SEASON_ARCHIVE / src.name
        shutil.copy2(src, dst)
        archived.append(
            {
                "file": src.name,
                "source": str(src),
                "target": str(dst),
            }
        )
        print(f"Archived: {src} -> {dst}")
        return True

    missing_required = []
    for fname in required_star:
        src = STAR_DIR / fname
        if not src.exists():
            missing_required.append(str(src))

    if missing_required:
        print("ERROR: Missing required 1981 season files:")
        for path in missing_required:
            print("  -", path)
        print("Season freeze aborted. Run the full 1981 pipeline before freezing.")
        return

    for fname in required_star:
        copy_file(STAR_DIR / fname)

    for fname in optional_star:
        if not copy_file(STAR_DIR / fname):
            print(f"Optional file not found (skipped): {STAR_DIR / fname}")

    mgr_src = CSV_OUT_DIR / manager_dim_name
    if copy_file(mgr_src):
        print("Manager dimension snapshot archived.")
    else:
        print(f"Manager dimension file not found (skipped): {mgr_src}")

    manifest = {
        "season": 1981,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "archive_dir": str(SEASON_ARCHIVE),
        "files": archived,
    }

    manifest_path = SEASON_ARCHIVE / "season_1981_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_path}")
    print("SEASON 1981 FREEZE COMPLETE.")


if __name__ == "__main__":
    main()
