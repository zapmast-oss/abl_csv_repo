import pandas as pd
from pathlib import Path
import re

from abl_config import RAW_CSV_ROOT, TXT_OUT_ROOT

# ===== CONFIG =====
CSV_DIR = RAW_CSV_ROOT
LEAGUE_ID = 200
YEAR = 1981

AWAY_ABBR = "CHI"   # Chicago
HOME_ABBR = "MIA"   # Miami
GAME_LABEL = "Week 6 • Monday, May 11, 1981"
OUT_TXT = TXT_OUT_ROOT / "preview_CHI_at_MIA.txt"
DEBUG_LOG = TXT_OUT_ROOT / "preview_CHI_at_MIA_debug.txt"

def log(lines):
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        for ln in (lines if isinstance(lines, list) else [lines]):
            print(ln)
            f.write(str(ln) + "\n")

def canon(df: pd.DataFrame) -> pd.DataFrame:
    m = {c: re.sub(r'[^0-9a-z]+','_', c.strip().lower()) for c in df.columns}
    df = df.rename(columns=m)
    # common short->canonical
    ren = {
        "teamid":"team_id","leagueid":"league_id","subleagueid":"sub_league_id","divisionid":"division_id",
        "abbr":"abbr","t":"team","w":"w","l":"l","pos":"pos","gb":"gb",
        "r":"r","hr":"hr"
    }
    for k,v in ren.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    return df

def load_csv(name):
    p = CSV_DIR / name
    if not p.exists():
        log(f"Missing CSV: {p}")
        return None
    df = pd.read_csv(p, low_memory=False, encoding_errors="replace")
    return canon(df)

def subleague_name(x):
    try:
        return "NBC" if int(x)==0 else "ABC" if int(x)==1 else str(x)
    except Exception:
        return str(x)

def fmt_gb(val):
    try:
        v = float(val)
        return "—" if abs(v) < 1e-9 else f"{v:.1f}"
    except Exception:
        return str(val)

def pick_team_ids(teams: pd.DataFrame, away_abbr: str, home_abbr: str):
    if teams is None or "abbr" not in teams.columns:
        return None, None, "teams.csv missing or no abbr column"
    teams["abbr_u"] = teams["abbr"].astype(str).str.upper().str.strip()
    log("Known team ABBRs: " + ", ".join(sorted(teams["abbr_u"].unique())))
    away = teams.loc[teams["abbr_u"].eq(away_abbr.upper())]
    home = teams.loc[teams["abbr_u"].eq(home_abbr.upper())]
    if away.empty or home.empty:
        # try by city/nickname contains (fallback)
        msg = []
        if away.empty:
            msg.append(f"Could not find AWAY_ABBR={away_abbr} in teams.csv")
        if home.empty:
            msg.append(f"Could not find HOME_ABBR={home_abbr} in teams.csv")
        return (int(away["team_id"].iloc[0]) if not away.empty else None,
                int(home["team_id"].iloc[0]) if not home.empty else None,
                "; ".join(msg))
    return int(away["team_id"].iloc[0]), int(home["team_id"].iloc[0]), ""

def main():
    # reset debug
    if DEBUG_LOG.exists():
        DEBUG_LOG.unlink()
    log("=== CHI @ MIA PREVIEW (diagnostics) ===")

    teams = load_csv("teams.csv")
    if teams is not None:
        keep_cols = [c for c in ["team_id","abbr","nickname","name","sub_league_id","division_id"] if c in teams.columns]
        log(f"teams.csv rows={len(teams)} cols={keep_cols}")
        teams = teams[keep_cols].drop_duplicates()

    away_id, home_id, abbr_err = pick_team_ids(teams, AWAY_ABBR, HOME_ABBR)
    if abbr_err:
        log(abbr_err)

    # records
    records = load_csv("team_record.csv")
    if records is not None:
        log(f"team_record.csv raw rows={len(records)}")
        # apply filters only if the columns exist
        if "league_id" in records.columns:
            pre=len(records); records = records[records["league_id"]==LEAGUE_ID]; log(f"team_record league_id {pre}->{len(records)}")
        if "year" in records.columns:
            pre=len(records); records = records[records["year"]==YEAR]; log(f"team_record year {pre}->{len(records)}")
        keep = [c for c in ["team_id","w","l","pos","gb","league_id","year","sub_league_id","division_id"] if c in records.columns]
        records = records[keep]
    else:
        records = pd.DataFrame()

    # team batting / pitching (RS/RA)
    tb = load_csv("team_batting_stats.csv")
    if tb is not None:
        pre=len(tb)
        if "split_id" in tb.columns: tb = tb[tb["split_id"].fillna(0).astype(int)==0]
        if "league_id" in tb.columns: tb = tb[tb["league_id"]==LEAGUE_ID]
        if "year" in tb.columns: tb = tb[tb["year"]==YEAR]
        log(f"team_batting_stats filtered {pre}->{len(tb)}")
        # ensure 'r' exists
        if "r" not in tb.columns:
            log("team_batting_stats: missing 'r' (runs scored)")
            tb = None
    tp = load_csv("team_pitching_stats.csv")
    if tp is not None:
        pre=len(tp)
        if "split_id" in tp.columns: tp = tp[tp["split_id"].fillna(0).astype(int)==0]
        if "league_id" in tp.columns: tp = tp[tp["league_id"]==LEAGUE_ID]
        if "year" in tp.columns: tp = tp[tp["year"]==YEAR]
        log(f"team_pitching_stats filtered {pre}->{len(tp)}")
        if "r" not in tp.columns:
            log("team_pitching_stats: missing 'r' (runs allowed)")
            tp = None

    if tb is not None:
        tb = tb.groupby("team_id", as_index=False)[["r"]].sum().rename(columns={"r":"runs_scored"})
    else:
        tb = pd.DataFrame(columns=["team_id","runs_scored"])
    if tp is not None:
        tp = tp.groupby("team_id", as_index=False)[["r"]].sum().rename(columns={"r":"runs_allowed"})
    else:
        tp = pd.DataFrame(columns=["team_id","runs_allowed"])

    # standings merge
    stand = records
    if not stand.empty and teams is not None:
        stand = stand.merge(teams, on="team_id", how="left")
    if not stand.empty:
        stand = stand.merge(tb, on="team_id", how="left").merge(tp, on="team_id", how="left")
        stand["runs_scored"] = pd.to_numeric(stand["runs_scored"], errors="coerce").fillna(0).astype(int)
        stand["runs_allowed"] = pd.to_numeric(stand["runs_allowed"], errors="coerce").fillna(0).astype(int)
        stand["rd"] = stand["runs_scored"] - stand["runs_allowed"]
    log(f"standings rows after merge={len(stand)}")

    # head-to-head (optional)
    games = load_csv("games.csv")
    h2h_line = ""
    if games is not None and away_id and home_id:
        if "year" in games.columns:
            pre=len(games); games = games[games["year"]==YEAR]; log(f"games year {pre}->{len(games)}")
        if "league_id" in games.columns:
            pre=len(games); games = games[games["league_id"]==LEAGUE_ID]; log(f"games league_id {pre}->{len(games)}")
        home_cands = [c for c in ["home_team_id","home_team","home"] if c in games.columns]
        away_cands = [c for c in ["away_team_id","away_team","away"] if c in games.columns]
        if home_cands and away_cands:
            hc, ac = home_cands[0], away_cands[0]
            # map abbr → id when needed
            abbr_map = {}
            if teams is not None and {"abbr","team_id"} <= set(teams.columns):
                abbr_map = {str(a).upper(): int(tid) for a, tid in zip(teams["abbr"], teams["team_id"])}

            def norm_team(v):
                try: return int(v)
                except: return abbr_map.get(str(v).upper().strip(), None)

            games["home_id_n"] = games[hc].apply(norm_team)
            games["away_id_n"] = games[ac].apply(norm_team)
            g2 = games[((games["home_id_n"]==home_id) & (games["away_id_n"]==away_id)) |
                       ((games["home_id_n"]==away_id) & (games["away_id_n"]==home_id))].copy()
            log(f"H2H games found this year: {len(g2)}")
            if not g2.empty:
                # try to compute wins if score columns present
                score_home = [c for c in g2.columns if c.startswith("home_") and c.endswith(("r","runs","score"))]
                score_away = [c for c in g2.columns if c.startswith("away_") and c.endswith(("r","runs","score"))]
                if score_home and score_away:
                    hs, as_ = score_home[0], score_away[0]
                    g2["home_w"] = pd.to_numeric(g2[hs], errors="coerce") > pd.to_numeric(g2[as_], errors="coerce")
                    g2["away_w"] = ~g2["home_w"]
                    aw = int(((g2["away_id_n"]==away_id) & g2["away_w"]).sum())
                    hw = int(((g2["home_id_n"]==home_id) & g2["home_w"]).sum())
                    h2h_line = f"H2H {YEAR}: {AWAY_ABBR} {aw} – {hw} {HOME_ABBR}"
                else:
                    h2h_line = f"H2H {YEAR}: {len(g2)} meetings (scores not available)"
        else:
            log("games.csv missing home/away columns for H2H")

    # Build card (never empty)
    lines = []
    lines.append(f"{AWAY_ABBR} at {HOME_ABBR} — {GAME_LABEL}")
    lines.append("NBC matchup • Action Baseball League")
    lines.append("")

    # Away snapshot
    away_txt = ""
    home_txt = ""
    if away_id and not stand.empty and (stand["team_id"]==away_id).any():
        ar = stand.loc[stand["team_id"]==away_id].iloc[0]
        away_txt = f"{AWAY_ABBR} {int(ar.get('w',0))}-{int(ar.get('l',0))} (Pos {int(ar.get('pos',0))}, GB {fmt_gb(ar.get('gb'))}, RD {int(ar.get('rd',0))})"
    else:
        away_txt = f"{AWAY_ABBR} — standings unavailable (check ABBR or filters)."
    if home_id
