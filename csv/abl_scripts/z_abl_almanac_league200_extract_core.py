from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd


REQUIRED_MANIFEST_COLS = {"rel_path", "category"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract league_core files from an almanac zip/dir.")
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g. 1972")
    parser.add_argument("--league-id", type=int, required=True, help="League ID, e.g. 200")
    parser.add_argument(
        "--almanac-root",
        type=str,
        default=None,
        help="Path to almanac root (default: almanac_<season> relative to repo root)",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Path to manifest CSV (default: csv/out/almanac/<season>/almanac_manifest_<season>_league<league_id>.csv)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="csv/in/almanac_core",
        help="Destination root for copied core files (default: csv/in/almanac_core)",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    df = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return df


def select_core_files(df: pd.DataFrame, league_id: int) -> pd.DataFrame:
    df = df[df["category"] == "league_core"].copy()
    wanted = {
        f"leagues/league_{league_id}_stats.html",
        f"leagues/league_{league_id}_standings.html",
    }
    df = df[df["rel_path"].isin(wanted)]
    return df


def resolve_source(almanac_root: Path, season: int) -> tuple[str, Path | zipfile.ZipFile]:
    """
    Return (mode, handle) where mode is 'dir' or 'zip'.
    Tries directory first, then zip candidates.
    """
    if almanac_root.exists() and almanac_root.is_dir():
        return "dir", almanac_root
    zip_candidates = [
        almanac_root.with_suffix(".zip"),
        Path("data_raw/ootp_html") / f"almanac_{season}.zip",
    ]
    for zpath in zip_candidates:
        if zpath.exists():
            return "zip", zipfile.ZipFile(zpath, "r")
    raise FileNotFoundError(
        f"almanac root not found as directory ({almanac_root}) or zip ({zip_candidates})"
    )


def main() -> int:
    args = parse_args()
    season = args.season
    league_id = args.league_id

    almanac_root = Path(args.almanac_root) if args.almanac_root else Path(f"almanac_{season}")
    try:
        mode, handle = resolve_source(almanac_root, season)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
    else:
        manifest_path = Path("csv/out/almanac") / str(season) / f"almanac_manifest_{season}_league{league_id}.csv"

    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading manifest: {exc}", file=sys.stderr)
        return 1

    core = select_core_files(manifest, league_id)
    if core.empty:
        print(
            f"Error: No league_core files (stats/standings) found for league_{league_id} in manifest {manifest_path}",
            file=sys.stderr,
        )
        # Try a direct scan fallback
        fallback_rows = []
        candidates = {
            f"leagues/league_{league_id}_stats.html",
            f"leagues/league_{league_id}_standings.html",
        }
        if mode == "dir":
            root: Path = handle  # type: ignore[assignment]
            for p in root.rglob("*"):
                if p.is_dir():
                    continue
                rel = p.relative_to(root).as_posix()
                if rel in candidates or any(rel.endswith(cand) for cand in candidates):
                    fallback_rows.append({"rel_path": rel})
        else:
            zf: zipfile.ZipFile = handle  # type: ignore[assignment]
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                rel = name
                if rel.startswith(f"almanac_{season}/"):
                    rel = rel[len(f"almanac_{season}/") :]
                if rel in candidates or any(rel.endswith(cand) for cand in candidates):
                    fallback_rows.append({"rel_path": rel})
        if fallback_rows:
            core = pd.DataFrame(fallback_rows)
        else:
            return 1

    output_root = Path(args.output_root)
    copied = 0
    for _, row in core.iterrows():
        rel_path = Path(row["rel_path"])
        dst = output_root / str(season) / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "dir":
            src = handle / rel_path  # type: ignore[operator]
            if not src.exists():
                print(f"[WARN] Missing source file in dir mode: {src}", file=sys.stderr)
                continue
            shutil.copy2(src, dst)
            copied += 1
            print(f"[COPIED] {src} -> {dst}")
        else:
            zf: zipfile.ZipFile = handle  # type: ignore[assignment]
            rel_posix = rel_path.as_posix()
            candidates = [rel_posix, f"almanac_{season}/{rel_posix}"]
            data = None
            chosen = None
            for cand in candidates:
                try:
                    data = zf.read(cand)
                    chosen = cand
                    break
                except KeyError:
                    continue
            if data is None:
                print(f"[WARN] Missing source file in zip mode: {rel_posix}", file=sys.stderr)
                continue
            with dst.open("wb") as f:
                f.write(data)
            copied += 1
            print(f"[COPIED] zip:{chosen} -> {dst}")

    if mode == "zip":
        handle.close()  # type: ignore[operator]

    print(f"[OK] Copied {copied} files to {output_root / str(season)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
