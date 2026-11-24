"""Optional Cloudflare Web Analytics integration for EAB HTML."""

import re


BEACON_URL = "https://static.cloudflareinsights.com/beacon.min.js"
TOKEN_FILE = "cloudflare-beacon-token.txt"
TOKEN = re.compile(r"[a-fA-F0-9]{32}\Z")
BEACON_DATA = re.compile(r'\{"token":"[a-f0-9]{32}"\}\Z')
BEACON_TAG = re.compile(
    rf'<script type="module" src="{re.escape(BEACON_URL)}" data-cf-beacon=\'\{{"token":"[a-f0-9]{{32}}"\}}\'></script>'
)


def normalize_token(value):
    token = value.strip()
    if not TOKEN.fullmatch(token):
        raise ValueError("Enter only the 32-character public Cloudflare Web Analytics beacon token.")
    return token.lower()


def load_token(output):
    try:
        return normalize_token((output / TOKEN_FILE).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None


def beacon(token):
    if not token:
        return ""
    return f'''<script type="module" src="{BEACON_URL}" data-cf-beacon='{{"token":"{normalize_token(token)}"}}'></script>'''


def valid_beacon(attributes):
    return (set(attributes) == {"type", "src", "data-cf-beacon"} and attributes["type"] == "module"
            and attributes["src"] == BEACON_URL and bool(BEACON_DATA.fullmatch(attributes["data-cf-beacon"] or "")))
