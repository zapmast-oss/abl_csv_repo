"""Read the headings, prose and tables in a reviewed Special Coverage manuscript."""

import re


def coverage_block(block):
    if block.startswith("### "):
        return "heading", block[4:]
    if not block.startswith("|"):
        return "paragraph", block
    lines = block.splitlines()
    if len(lines) < 3 or any(not (line.startswith("|") and line.endswith("|")) for line in lines):
        raise ValueError("Invalid Special Coverage table.")
    rows = [[field.strip() for field in line[1:-1].split("|")] for line in lines]
    headings, separator, *data = rows
    if (not all(headings) or any(len(row) != len(headings) for row in rows)
            or any(not re.fullmatch(r"-{3,}", field) for field in separator)):
        raise ValueError("Invalid Special Coverage table columns.")
    return "table", (headings, data)
