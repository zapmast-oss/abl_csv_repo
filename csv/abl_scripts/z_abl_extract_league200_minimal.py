import argparse
import os
from pathlib import Path
import shutil
import sys


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree_if_exists(root_dir: Path, output_dir: Path, subdir_name: str) -> int:
    """
    Copy a top-level subdirectory (e.g. 'css', 'js', 'images') from root_dir
    to output_dir if it exists. Returns number of files copied.
    """
    src = root_dir / subdir_name
    if not src.is_dir():
        return 0

    dst = output_dir / subdir_name
    count = 0

    for dirpath, dirnames, filenames in os.walk(src):
        dirpath = Path(dirpath)
        rel = dirpath.relative_to(root_dir)
        out_dir = output_dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname in filenames:
            src_file = dirpath / fname
            dst_file = out_dir / fname
            shutil.copy2(src_file, dst_file)
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract minimal ABL (League 200) site: league_200_home.html + css/js/images."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Path to the root of the full OOTP HTML site (the 'html' folder).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the output folder for the minimal League 200 site.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Basic validation
    if not root_dir.is_dir():
        print(f"ERROR: root-dir does not exist or is not a directory: {root_dir}", file=sys.stderr)
        sys.exit(1)

    league200_rel = Path("leagues") / "league_200_home.html"
    league200_src = root_dir / league200_rel

    if not league200_src.is_file():
        print(f"ERROR: League 200 home file not found at expected location: {league200_src}", file=sys.stderr)
        sys.exit(1)

    # Create output directory (do not delete if it exists).
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Extracting minimal League 200 site ===")
    print(f"Root dir:   {root_dir}")
    print(f"Output dir: {output_dir}")
    print(f"League200:  {league200_src}")

    # Copy the League 200 home page
    league200_dst = output_dir / league200_rel
    copy_file(league200_src, league200_dst)
    print(f"Copied League 200 home page to: {league200_dst}")

    # Copy core asset folders if present
    total_assets = 0
    for subdir in ("css", "js", "images"):
        copied = copy_tree_if_exists(root_dir, output_dir, subdir)
        if copied > 0:
            print(f"Copied {copied} files from '{subdir}/'")
            total_assets += copied
        else:
            print(f"No '{subdir}/' directory found under root; skipped.")

    print("=== Summary ===")
    print("HTML files copied: 1 (league_200_home.html)")
    print(f"Asset files copied: {total_assets}")
    print("Done. Open the League 200 home page from the output dir to verify:")

    print(output_dir / league200_rel)


if __name__ == "__main__":
    main()
