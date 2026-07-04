from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sprint 1 weekly Story Signal to Baseball Observer pipeline.")
    parser.add_argument("--week-label", default="1981_week_15")
    parser.add_argument("--as-of", default="1981-07-12")
    parser.add_argument("--run-id", default="sprint1_1981_week_15")
    return parser.parse_args()


def run(script: str, args: list[str]) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script), *args]
    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    common = ["--week-label", args.week_label, "--run-id", args.run_id]
    run("z_abl_story_signal_weekly_1981.py", [*common, "--as-of", args.as_of])
    run("z_abl_baseball_observer_packet_weekly_1981.py", common)
    print(f"[OK] Sprint 1 pipeline complete for {args.week_label} as of {args.as_of}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
