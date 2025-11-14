import subprocess
import datetime


def run_script_capture(script_name: str) -> str:
    """
    Run another Python script in this folder and capture its stdout.
    If it errors, return a nice italicized error message instead of crashing.
    """
    try:
        out = subprocess.check_output(
            ["python", script_name],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        return out.strip()
    except FileNotFoundError:
        return f"[i]({script_name} not found in csv folder.)[/i]"
    except subprocess.CalledProcessError as e:
        return f"[i](Error running {script_name}: {e})[/i]"


def extract_standings_from_show_notes(show_text: str) -> str:
    """
    From the abl_show_notes.py output, pull just the 'Standings & run differential'
    section: the table plus the broadcast lines.

    We look for any line containing 'Standings & run differential' (so it's not
    fragile about ===, spaces, etc.), then capture everything until the next
    major === header, or the end of the text.
    """
    lines = show_text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if "Standings & run differential" in line:
            # We skip the header line itself and start from the next line
            start_idx = i + 1
            break

    if start_idx is None:
        return "[i](Could not find 'Standings & run differential' section in abl_show_notes output.)[/i]"

    section_lines = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        # Stop when we hit the next big section header (=== Something ===)
        if stripped.startswith("===") and "Standings & run differential" not in stripped:
            break
        section_lines.append(line)

    section = "\n".join(section_lines).strip()
    if not section:
        return "[i](Standings section was empty in abl_show_notes output.)[/i]"

    return section


def get_standings_block() -> str:
    """
    Run abl_show_notes.py and pull out the standings section.
    If abl_show_notes.py itself fails, just return its error message.
    """
    raw = run_script_capture("abl_show_notes.py")

    # If we already got an italic error from run_script_capture, just pass it through.
    if raw.startswith("[i]("):
        return raw

    return extract_standings_from_show_notes(raw)


def print_code_block(title: str, content: str) -> None:
    print(f"[b]{title}[/b]")
    print("[code]")
    print(content)
    print("[/code]")
    print()


def main():
    now = datetime.datetime.now()

    # Header
    print("[b][size=150]Around the ABL – Weekly Notebook[/size][/b]")
    print(f"[i]Season 1981 snapshot – standings, streaks, stars and storylines. (Generated {now:%B %d, %Y})[/i]")
    print()

    # 1) Standings & run differential (from abl_show_notes.py)
    standings_text = get_standings_block()
    print_code_block("Standings & run differential", standings_text)

    # 2) Power rankings (from abl_power_rankings_forum.py)
    power_text = run_script_capture("abl_power_rankings_forum.py")
    print_code_block("Power rankings", power_text)

    # 3) Streaks (from abl_streaks.py)
    streaks_text = run_script_capture("abl_streaks.py")
    print_code_block("Heating up & cooling off – streaks", streaks_text)

    # 4) Bats & arms of the week (from abl_player_leaders.py)
    leaders_text = run_script_capture("abl_player_leaders.py")
    print_code_block("Bats & arms of the week", leaders_text)

    # 5) Fun with speed & strikeouts (from abl_fun_leaders.py)
    fun_text = run_script_capture("abl_fun_leaders.py")
    print_code_block("Fun with speed & strikeouts", fun_text)

    # 6) Featured matchup (from abl_featured_matchup.py)
    featured_text = run_script_capture("abl_featured_matchup.py")
    print_code_block("Featured matchup", featured_text)

    # 7) Manager matchup (from abl_manager_matchup.py)
    mgr_text = run_script_capture("abl_manager_matchup.py")
    print_code_block("Manager matchup", mgr_text)

    # 8) News & notes (from abl_news_notes.py)
    news_text = run_script_capture("abl_news_notes.py")
    print_code_block("News & notes", news_text)


if __name__ == "__main__":
    main()
