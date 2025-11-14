# show_notes.py (driver script)
import subprocess
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

SCRIPTS = [
    ("Standings & run differential", "abl_standings.py"),
    ("Streaks (winning & losing)", "abl_streaks.py"),
    ("Top players (hitters & pitchers)", "abl_player_leaders.py"),
    ("Fun leaders (speed & strikeout artists)", "abl_fun_leaders.py"),
]


def add_broadcast_line_spacing(text: str) -> str:
    """Insert a blank line after every three broadcast lines for readability."""
    lines = text.splitlines()
    spaced_lines: list[str] = []
    in_broadcast_block = False
    lines_in_block = 0

    for line in lines:
        spaced_lines.append(line)
        indicator = "broadcast lines"

        if indicator in line.lower():
            in_broadcast_block = True
            lines_in_block = 0
            continue

        if in_broadcast_block:
            if line.strip() == "":
                in_broadcast_block = False
                lines_in_block = 0
                continue

            lines_in_block += 1
            if lines_in_block == 3:
                spaced_lines.append("")
                lines_in_block = 0

    return "\n".join(spaced_lines)


def run_script(script_name: str) -> str:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        return f"[Error: script '{script_name}' was not found under {SCRIPT_DIR} .]"
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            input="\n",
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT_DIR,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return f"[Error: unable to execute '{script_path}'.]"
    except subprocess.CalledProcessError as exc:
        return (
            f"[Error while running '{script_name}']\n"
            f"STDOUT:\n{exc.stdout}\n"
            f"STDERR:\n{exc.stderr}"
        )


def main() -> None:
    now = datetime.now()
    header_lines = [
        "=== Action Baseball League - Show Notes ===",
        f"Generated on: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    parts = ["\n".join(header_lines)]

    for title, script in SCRIPTS:
        parts.append(f"=== {title} ===")
        out = run_script(script)
        parts.append(out if out else "[No output]")
        if title.startswith("Standings"):
            parts.append("")
            parts.append("=== Last 10 ===")
            parts.append(run_script("abl_last10_1.py") or "[No output]")
        parts.append("")

    full_text = "\n".join(parts)
    full_text = add_broadcast_line_spacing(full_text)
    out_dir = Path("out") / "text_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ABL_Show_Notes.txt"
    out_path.write_text(full_text, encoding="utf-8")
    print(f"Show notes written to {out_path}")


if __name__ == "__main__":
    main()
