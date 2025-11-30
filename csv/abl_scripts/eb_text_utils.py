#!/usr/bin/env python
"""Shared EB markdown/text cleanup helpers."""
from __future__ import annotations

import re

_SEATTLE_RE = re.compile(r"seatlle", re.IGNORECASE)


def normalize_eb_text(text: str) -> str:
    """Normalize common EB text issues (currently Seatlle -> Seattle Comets (SEA))."""
    return _SEATTLE_RE.sub("Seattle Comets (SEA)", text)
