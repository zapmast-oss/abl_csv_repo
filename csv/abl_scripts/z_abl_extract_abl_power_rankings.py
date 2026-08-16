import argparse
from pathlib import Path
import csv
import sys
import re

from bs4 import BeautifulSoup
from bs4.element import NavigableString


def find_power_rankings_container(soup: BeautifulSoup):
    """
    Find the element that contains the 'Weekly Team Power Rankings' block.

    We look for a <span> whose text includes 'Weekly Team Power Rankings'
    (case-insensitive), then use its parent (the <td> in your snippet)
    as the container. We also try to read the next <span> as the date label.
    """
    heading_span = soup.find(
        "span",
        string=lambda s: isinstance(s, str)
        and "weekly team power rankings" in s.lower(),
    )
    if heading_span is None:
        return None, None

    container = heading_span.parent

    # Try to get the date from the next <span> after the heading_span
    date_span = heading_span.find_next("span")
    date_label = date_span.get_text(strip=True) if date_span else ""

    return container, date_label


def _get_nearby_string(tag, direction: str) -> str:
    """
    Helper: walk siblings in the given direction ('previous' or 'next') until
    we find a non-empty NavigableString. Return its stripped text or ''.
    """
    if direction not in ("previous", "next"):
        raise ValueError("direction must be 'previous' or 'next'")

    attr = f"{direction}_sibling"
    sib = getattr(tag, attr)
    while sib is not None:
        if isinstance(sib, NavigableString):
            text = sib.strip()
            if text:
                return text
        sib = getattr(sib, attr)

    return ""


def parse_power_rankings(container, date_label: str):
    """
    Given the container element (the <td> from your snippet), parse out
    all power ranking entries.

    Expected pattern around each <a>:
      '1) ' [text before link]
      <a href="../teams/team_17.html">Las Vegas Gamblers</a>
      ' (137.0, o)' [text after link]

    We extract:
      - rank (int)
      - team_id (int) from href 'team_17.html'
      - team_name (str)
      - points (str, numeric text)
      - tendency (str, e.g. 'o', '++', '--', '+', '-')
      - date_label (str) as given above
    """
    if container is None:
        raise ValueError("No container provided for power rankings parsing.")

    rows = []

    # Find all team links inside the container
    for a in container.find_all("a", href=True):
        team_name = a.get_text(strip=True)
        href = a["href"]

        # Extract team_id from href like '../teams/team_17.html'
        team_id = None
        m_id = re.search(r"team_(\d+)\.html", href)
        if m_id:
            team_id = int(m_id.group(1))

        # Rank is in the text immediately before the <a>, like '1)'
        rank_text = _get_nearby_string(a, "previous")
        m_rank = re.match(r"(\d+)\)", rank_text)
        if not m_rank:
            print(
                f"WARNING: Could not parse rank from text '{rank_text}' for team '{team_name}'. Skipping.",
                file=sys.stderr,
            )
            continue
        rank = int(m_rank.group(1))

        # Points and tendency are in the text immediately after the <a>,
        # e.g. '(137.0, o)' or '(108.3, ++)'
        after_text = _get_nearby_string(a, "next")
        m_pt = re.search(r"\(([\d\.]+),\s*([+o-]{1,2})\)", after_text)
        if not m_pt:
            print(
                f"WARNING: Could not parse points/tendency from text '{after_text}' for team '{team_name}'. Skipping.",
                file=sys.stderr,
            )
            continue

        points = m_pt.group(1)
        tendency = m_pt.group(2)

        rows.append(
            {
                "rank": rank,
                "team_id": team_id,
                "team_name": team_name,
                "points": points,
                "tendency": tendency,
                "date_label": date_label,
            }
        )

    # Sort rows by rank just to be safe
    rows.sort(key=lambda r: r["rank"])

    if not rows:
        raise ValueError("No power ranking rows were parsed from the container.")

    headers = ["rank", "team_id", "team_name", "points", "tendency", "date_label"]
    return headers, rows


def write_csv(headers, rows, output_path: Path):
    """
    Write a CSV given headers (list of column names) and rows (list of dicts).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow([r[h] for h in headers])


def main():
    parser = argparse.ArgumentParser(
        description="Extract ABL (League 200) Weekly Team Power Rankings from league_200_home.html into CSV."
    )
    parser.add_argument(
        "--input-html",
        required=True,
        help="Path to league_200_home.html (the League 200 home page).",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output CSV file for full power rankings (1–24).",
    )
    args = parser.parse_args()

    input_html = Path(args.input_html).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not input_html.is_file():
        print(f"ERROR: input HTML file does not exist: {input_html}", file=sys.stderr)
        sys.exit(1)

    # Load HTML
    html_text = input_html.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")

    # Find the container holding the Weekly Team Power Rankings block
    container, date_label = find_power_rankings_container(soup)
    if container is None:
        print(
            "ERROR: Could not find a span containing 'Weekly Team Power Rankings' on the page.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        headers, rows = parse_power_rankings(container, date_label)
    except ValueError as e:
        print(f"ERROR while parsing power rankings: {e}", file=sys.stderr)
        sys.exit(1)

    # Sanity check: we expect 24 teams in ABL
    if len(rows) != 24:
        print(
            f"WARNING: Parsed {len(rows)} rows instead of 24. Check the HTML to ensure all teams are present.",
            file=sys.stderr,
        )

    # Write full table (1–24)
    write_csv(headers, rows, output_csv)

    # Derive top 9 by rank for EB's report
    top9 = [r for r in rows if 1 <= r["rank"] <= 9]
    top9_output = output_csv.parent / "abl_power_rankings_top9.csv"
    write_csv(headers, top9, top9_output)

    print("=== Power Rankings Extraction Complete ===")
    print(f"Input HTML:      {input_html}")
    print(f"Output CSV:      {output_csv}")
    print(f"Top 9 Output:    {top9_output}")
    print(f"Columns:         {headers}")
    print(f"Rows parsed:     {len(rows)}")
    print(f"Top 9 rows:      {len(top9)}")


if __name__ == "__main__":
    main()
