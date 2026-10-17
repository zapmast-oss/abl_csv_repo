from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).resolve().parent / "z_abl_almanac_time_slices_enriched.py"
    almanac_root = Path(__file__).resolve().parents[2] / "csv" / "out" / "almanac"

    seasons = []
    for child in almanac_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            if any(child.glob("team_monthly_summary_*_league*.csv")):
                seasons.append(child.name)

    if not seasons:
        print("No almanac seasons found with monthly summaries.", file=sys.stderr)
        return 1

    for season in sorted(seasons):
        print(f"[INFO] Running enrichment for season {season}")
        cmd = [
            sys.executable,
            str(script),
            "--season",
            season,
            "--league-id",
            "200",
            "--almanac-root",
            str(almanac_root),
        ]
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"[ERROR] Enrichment failed for season {season}", file=sys.stderr)
            return result.returncode

    print("[OK] Enrichment completed for seasons:", ", ".join(sorted(seasons)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
