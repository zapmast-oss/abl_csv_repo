#!/usr/bin/env python
"""Shared EB markdown/text cleanup helpers."""
from __future__ import annotations

import re

_SEATTLE_RE = re.compile(r"seatlle", re.IGNORECASE)


def normalize_eb_text(text: str) -> str:
    """Normalize common EB text issues (currently Seatlle -> Seattle Comets (SEA))."""
    return _SEATTLE_RE.sub("Seattle Comets (SEA)", text)


def canonicalize_team_city(city: str) -> str:
    """Normalize known team city typos."""
    if city is None:
        return city
    city_clean = str(city).strip()
    if not city_clean:
        return city_clean
    if city_clean.lower() == "seatlle":
        return "Seattle"
    return city_clean


def format_team_label(team_name: str | None, team_abbr: str | None) -> str:
    """
    Return a clean team label, collapsing blanks/nan to 'Free agent'.
    Prefer 'City (ABR)' when possible.
    """

    def _clean(x: str | None) -> str:
        if x is None:
            return ""
        s = str(x).strip()
        if not s:
            return ""
        if s.lower() in {"nan", "none", "null"}:
            return ""
        return s

    name = _clean(team_name)
    abbr = _clean(team_abbr)

    if name and abbr:
        return f"{name} ({abbr})"
    if abbr:
        return abbr
    if name:
        return name
    return "Free agent"
