from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

CSV_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = CSV_ROOT / "out" / "csv_out" / "abl_managers_summary.csv"


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def validate(path: Path) -> tuple[bool, list[str], int]:
    errors: list[str] = []
    seen: set[str] = set()
    count = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                errors.append("Missing manager name")
                continue
            key = name.lower()
            if key in seen:
                errors.append(f"Duplicate manager name: {name}")
            else:
                seen.add(key)
            wins = parse_int(row.get("wins"))
            losses = parse_int(row.get("losses"))
            win_pct = parse_float(row.get("win_pct"))
            titles = parse_int(row.get("championships"))
            if wins is None or losses is None:
                errors.append(f"Missing wins/losses for {name}")
            if win_pct is None:
                errors.append(f"Missing win_pct for {name}")
            if titles is None or titles < 0:
                errors.append(f"Invalid titles for {name}")
            if wins is not None and losses is not None and win_pct is not None:
                total = wins + losses
                if total > 0:
                    calc = wins / total
                    if math.fabs(calc - win_pct) >= 1e-4:
                        errors.append(f"Win pct mismatch for {name}: calc={calc:.6f} file={win_pct:.6f}")
            count += 1
    return (len(errors) == 0, errors, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manager summary CSV")
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="Path to abl_managers_summary.csv")
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        raise FileNotFoundError(f"Manager summary not found: {src_path}")

    ok, errors, count = validate(src_path)
    if ok:
        print(f"Managers validation: PASS ({count} managers)")
    else:
        print("Managers validation: FAIL")
        for err in errors:
            print(f" - {err}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
