from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an almanac manifest for a given season/league."
    )
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 1972)")
    parser.add_argument("--league-id", type=int, required=True, help="League ID (e.g. 200)")
    parser.add_argument(
        "--almanac-root",
        type=str,
        default=None,
        help="Path to almanac root (default: almanac_<season> relative to repo root)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    season = args.season
    league_id = args.league_id

    almanac_root = Path(args.almanac_root) if args.almanac_root else Path(f"almanac_{season}")
    if not almanac_root.exists() or not almanac_root.is_dir():
        print(f"Error: almanac root not found: {almanac_root}", file=sys.stderr)
        return 1

    rows = []
    for path in almanac_root.rglob("*"):
        if path.is_dir():
            continue
        rel_path = path.relative_to(almanac_root).as_posix()
        size_bytes = path.stat().st_size
        name = path.name
        if f"league_{league_id}" in name:
            category = "league_core"
        elif rel_path.startswith("leagues/"):
            category = "leagues_other"
        else:
            category = "other"
        rows.append(
            {
                "season": season,
                "league_id": league_id,
                "rel_path": rel_path,
                "size_bytes": size_bytes,
                "category": category,
            }
        )

    if not rows:
        print(f"Warning: no files found under {almanac_root}", file=sys.stderr)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["category", "rel_path"]).reset_index(drop=True)

    out_dir = Path("csv/out/almanac") / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"almanac_manifest_{season}_league{league_id}.csv"
    df.to_csv(out_path, index=False)

    print(f"[INFO] Total files scanned: {len(df)}")
    if not df.empty:
        print("[INFO] Count by category:")
        print(df["category"].value_counts())
        print(f"[INFO] league_core files: {len(df[df['category'] == 'league_core'])}")
    print(f"[OK] Wrote manifest to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
