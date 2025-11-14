# make_standings_snapshot.py
# Rebuilds standings_snapshot.csv from your Team Record file. No other changes.

import pandas as pd
from pathlib import Path
import glob

# 1) Find a source table (Team Record)
SRC = None
for name in ("team_record.csv", "team_records.csv"):
    p = Path(name)
    if p.exists():
        SRC = p
        break
if SRC is None:
    # last resort: any *team*record*.csv
    hits = sorted(glob.glob("*team*record*.csv"))
    if hits:
        SRC = Path(hits[0])

if SRC is None:
    print("[Error] No team record source found (looked for team_record.csv / team_records.csv).")
    raise SystemExit(1)

df = pd.read_csv(SRC)

# 2) Case-insensitive column resolver
lower = {c.lower(): c for c in df.columns}
def pick(*opts):
    for o in opts:
        if o in lower:
            return lower[o]
    return None

team = pick("team_display","team","name","nickname","abbr","city_team")
city = pick("city","team_city")
nick = pick("nickname","team_nickname")

# Build a readable team name if needed
if not team:
    if city and nick:
        df["team_display"] = df[city].astype(str).str.strip() + " " + df[nick].astype(str).str.strip()
    elif city:
        df["team_display"] = df[city].astype(str).str.strip()
    elif pick("abbr"):
        df["team_display"] = df[pick("abbr")].astype(str)
    else:
        df["team_display"] = "Team"
else:
    df["team_display"] = df[team]

w   = pick("w","wins")
l   = pick("l","losses")
pct = pick("pct","winpct","win_pct")
gb  = pick("gb","games_back","gamesbehind")
rs  = pick("rs","runs_scored","r","runs for","runsfor")
ra  = pick("ra","runs_allowed","runs against","runsagainst")

# Compute Pct if missing (and W/L exist)
if pct is None and w and l:
    total = (df[w].astype(float) + df[l].astype(float))
    total = total.where(total != 0, pd.NA)
    df["pct"] = (df[w] / total).round(3)
    pct = "pct"

# Compute RD if possible
if rs and ra:
    df["RD"] = df[rs] - df[ra]

# 3) Write EXACT filename your original script expects
cols = ["team_display"]
if w:   cols.append(w)
if l:   cols.append(l)
if pct: cols.append(pct)
if gb:  cols.append(gb)
if "RD" in df.columns: cols.append("RD")

out = df[cols].rename(columns={
    (w or "w"): "w",
    (l or "l"): "l",
    (pct or "pct"): "pct",
    (gb or "gb"):"gb",
})

out.to_csv("standings_snapshot.csv", index=False)
print(f"[OK] standings_snapshot.csv rebuilt from {SRC.name} with {len(out)} rows.")
