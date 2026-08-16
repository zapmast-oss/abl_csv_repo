from __future__ import annotations

import sqlite3
from pathlib import Path

CSV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CSV_ROOT.parent
DEFAULT_DB = REPO_ROOT / "data_work" / "abl.db"
DEFAULT_OUT = CSV_ROOT / "out" / "text_out" / "abl_managers_onepager.txt"

def fetch_data(db_path: Path) -> tuple[int, list[tuple[str, int, int, float, int]]]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        managers = cur.execute("SELECT COUNT(*) FROM dim_manager").fetchone()[0]
        top_rows = cur.execute(
            "SELECT name, career_wins, career_losses, career_win_pct, total_titles "
            "FROM v_manager_career LIMIT 10"
        ).fetchall()
    return managers, top_rows

def render_report(count: int, rows: list[tuple[str, int, int, float, int]]) -> str:
    lines: list[str] = []
    lines.append("ABL Managers Snapshot")
    lines.append("=" * 72)
    lines.append(f"Total managers tracked: {count}")
    lines.append("")
    header = f"{'Rank':<4} {'Name':<28} {'W':>5} {'L':>5} {'Win%':>7} {'Titles':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for idx, (name, wins, losses, pct, titles) in enumerate(rows, 1):
        pct_str = f"{pct:.3f}" if pct is not None else "n/a"
        lines.append(
            f"{idx:<4} {name:<28.28} {wins:>5} {losses:>5} {pct_str:>7} {titles:>7}"
        )
    return "\n".join(lines)

def write_report(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

def main() -> None:
    managers, rows = fetch_data(DEFAULT_DB)
    report = render_report(managers, rows)
    write_report(DEFAULT_OUT, report)
    print(f"Managers one-pager written to {DEFAULT_OUT}")

if __name__ == "__main__":
    main()
