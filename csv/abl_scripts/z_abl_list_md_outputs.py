"""List Markdown outputs under csv/out and write inventories."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


EXCLUDE_SUBSTRINGS = ["big50", "big_50", "big-50", "47", "reports", "report_pack"]


def find_csv_root(start: Path) -> Path:
    current = start
    while True:
        if any(current.rglob("teams*.csv")):
            return current
        if current.parent == current:
            break
        current = current.parent
    raise RuntimeError("Missing teams CSV in csv/")


def format_entry(base: Path, path: Path) -> str:
    rel = path.relative_to(base).as_posix()
    stat = path.stat()
    ts = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"- {rel} (modified: {ts}, size: {stat.st_size} bytes)"


def write_inventory(path: Path, csv_root: Path, out_dir: Path, entries: list[Path]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Markdown Inventory",
        f"Generated: {now}",
        f"Root: {csv_root}",
        f"Out: {out_dir}",
        f"Count: {len(entries)}",
        "",
        "## Files",
    ]
    lines.extend(format_entry(csv_root, p) for p in entries)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="List Markdown outputs.")
    parser.add_argument("--year", type=int, required=False)  # accepted but unused
    parser.add_argument("--week", type=int, required=False)  # accepted but unused
    args = parser.parse_args()  # noqa: F841

    script_path = Path(__file__).resolve()
    csv_root = find_csv_root(script_path.parent)
    out_dir = csv_root / "out"
    if not out_dir.exists():
        raise RuntimeError("Missing out/ directory under csv_root")

    all_md = sorted(out_dir.rglob("*.md"))
    core_md: list[Path] = []
    for p in all_md:
        rel_str = p.relative_to(csv_root).as_posix().lower()
        if any(tok in rel_str for tok in EXCLUDE_SUBSTRINGS):
            continue
        core_md.append(p)

    inv_all = out_dir / "md_inventory_all.md"
    inv_core = out_dir / "md_inventory_core.md"
    write_inventory(inv_all, csv_root, out_dir, all_md)
    write_inventory(inv_core, csv_root, out_dir, core_md)

    print("ALL MD:")
    for p in all_md:
        print(p.relative_to(csv_root).as_posix())
    print("")
    print("CORE MD:")
    for p in core_md:
        print(p.relative_to(csv_root).as_posix())
    print(f"Wrote: {inv_all} {inv_core}")


if __name__ == "__main__":
    main()
