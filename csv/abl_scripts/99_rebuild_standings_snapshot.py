# rebuild_standings_snapshot.py
# Rebuilds standings_snapshot.csv by joining team_record*.csv with teams.csv on team_id.
# No other scripts need to change.

from pathlib import Path
import pandas as pd
import glob

BASE = Path(".")

# --- locate sources ---
rec = None
for name in ("team_record.csv", "team_records.csv"):
    p = BASE / name
    if p.exists():
        rec = p
        break
if rec is None:
    hits = sorted(glob.glob(str(BASE / "*team*record*.csv")))
    if hits:
        rec = Path(hits[0])

teams_path = BASE / "teams.csv"

if rec is None:
    print("[Error] Could not find team_record*.csv")
    raise SystemExit(1)
if not teams_path.exists():
    print("[Error] Could not find teams.csv (export it from OOTP).")
    raise SystemExit(1)

rec_df = pd.read_csv(rec)
tm_df  = pd.read_csv(teams_path)

# --- column pickers (case-insensitive, minimal) ---
rec_low = {c.lower(): c for c in rec_df.columns}
tm_low  = {c.lower(): c for c in tm_df.columns}

def r(*opts):
    for o in opts:
        if o in rec_low: return rec_low[o]
    return None

def t(*opts):
    for o in opts:
        if o in tm_low: return tm_low[o]
    return None

tid_r = r("team_id","tid","teamid")
tid_t = t("team_id","tid","teamid")
if not tid_r or not tid_t:
    print("[Error] team_id column missing in one of the files.")
    raise SystemExit(1)

# Optional: filter to ABL (league_id == 200) if present on team_record
lg_r = r("league_id","leagueid","league","lg_id")
if lg_r is not None:
    try:
        rec_df = rec_df[rec_df[lg_r].astype(int) == 200]
    except Exception:
        rec_df = rec_df[rec_df[lg_r].astype(str) == "200"]

# Join for naming
subset_cols = [c for c in [tid_t, t("abbr","team_abbr"), t("nickname","team_nickname","nick"),
                           t("city","team_city")] if c]
tm_df = tm_df[subset_cols].copy()
merged = rec_df.merge(tm_df, left_on=tid_r, right_on=tid_t, how="left")

# Build team_display: ABBR + Nickname -> City + Nickname -> ABBR -> Nickname -> fallback
abbr = t("abbr","team_abbr")
nick = t("nickname","team_nickname","nick")
city = t("city","team_city")

def build_name(row):
    a = (row.get(abbr) if abbr in merged.columns else None)
    n = (row.get(nick) if nick in merged.columns else None)
    c = (row.get(city) if city in merged.columns else None)
    if isinstance(a, str) and isinstance(n, str): return f"{a.strip()} {n.strip()}"
    if isinstance(c, str) and isinstance(n, str): return f"{c.strip()} {n.strip()}"
    if isinstance(a, str): return a.strip()
    if isinstance(n, str): return n.strip()
    # as a last resort, "Team <id>"
    return f"Team {row.get(tid_r)}"

merged["team_display"] = merged.apply(build_name, axis=1)

# Pull metrics
w   = r("w","wins");  l = r("l","losses")
pct = r("pct","winpct","win_pct")
gb  = r("gb","games_back","gamesbehind")
rs  = r("rs","runs_scored","r","runs for","runsfor")
ra  = r("ra","runs_allowed","runs against","runsagainst")

# Compute pct if missing and W/L exist
if pct is None and w and l:
    total = (merged[w].astype(float) + merged[l].astype(float))
    total = total.where(total != 0, pd.NA)
    merged["pct"] = (merged[w] / total).round(3)
    pct = "pct"

# Compute RD if we have RS/RA
if rs and ra:
    merged["RD"] = merged[rs] - merged[ra]

# Assemble output columns in your original snapshot shape
cols = ["team_display"]
if w:   cols.append(w)
if l:   cols.append(l)
if pct: cols.append(pct)
if gb:  cols.append(gb)
if "RD" in merged.columns: cols.append("RD")

out = merged[cols].rename(columns={
    (w or "w"): "w",
    (l or "l"): "l",
    (pct or "pct"): "pct",
    (gb or "gb"): "gb",
})

out.to_csv(BASE / "standings_snapshot.csv", index=False)
print(f"[OK] Wrote standings_snapshot.csv from {rec.name} + teams.csv ({len(out)} rows)")
