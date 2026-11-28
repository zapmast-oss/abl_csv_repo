"""Independent release checks against the same immutable sources cited by the desk."""
import re
from collections import defaultdict
from pathlib import Path

from abl_newsroom_beats import day, rows


def _ids(pattern, text):
    return [int(n) for m in re.finditer(pattern+r"\s*[=:]?\s*(\d+(?:\s*,\s*\d+)*)",text,re.I)
            for n in re.findall(r"\d+",m[1])]


class ClaimCheck:
    def __init__(self, raw, cutoff, league_date, beats):
        self.raw = Path(raw)
        self.cutoff, self.league_date = day(cutoff), day(league_date)
        self.games = {int(g["game_id"]):g for g in rows(raw,"games") if g["league_id"]=="200"
                      and g["game_type"]=="0" and day(g["date"]).year==self.cutoff.year}
        self.clubs = {int(row["team_id"]):row for row in rows(raw,"teams") if int(row["team_id"]) in beats}
        self.players = {int(row["player_id"]) for row in rows(raw,"players")}
        self.deals = defaultdict(list)
        for deal in rows(raw,"trade_history"):
            if day(deal["date"]) <= self.league_date:
                self.deals[int(deal["message_id"])].append(deal)
        self.records = defaultdict(lambda:[0,0])
        self.scores = defaultdict(list)
        for gid,g in self.games.items():
            if g["played"]!="1" or day(g["date"])>self.cutoff:
                continue
            away,home = int(g["away_team"]),int(g["home_team"])
            a,h=int(g["runs0"]),int(g["runs1"])
            if a!=h:
                winner,loser=(away,home) if a>h else (home,away)
                self.records[winner][0]+=1
                self.records[loser][1]+=1
                self.scores[winner,loser,max(a,h),min(a,h)].append((gid,day(g["date"])))
        self.aliases={spelling:t for t,c in self.clubs.items() for spelling in
                      (c["name"],c["nickname"],c["name"]+" "+c["nickname"])}
        alias="(?:"+"|".join(re.escape(k) for k in sorted(self.aliases,key=len,reverse=True))+")"
        self.result=re.compile(rf"\b({alias}) (beat|defeated|fell to|lost to) (?:the )?({alias}) (\d+)[–-](\d+)(?: on ([A-Z][a-z]+ \d{{1,2}}))?")
        self.standing=re.compile(rf"\b({alias}) (?:is|are|stands at|sits at) (\d+)[–-](\d+)\b(?=\s*(?:[.,;]|overall\b|and (?:lead|trail|share|hold|is |has |sits?|owns?|remain)))")
        for t,b in beats.items():
            if self.records[t] != [b["wins"],b["losses"]]:
                raise ValueError(f"Independent standings contradict club {t}'s notebook")

    def evaluate(self, piece):
        checks={"source_tables":set(),"games":set(),"teams":set(),"players":set(),"transactions":set(),"score_claims":0,"record_claims":0}
        inputs = list(piece["evidence"])
        explicit_games=set()
        for ref in inputs:
            for filename in re.findall(r"[\w-]+\.csv",ref):
                if not (self.raw/filename).is_file():
                    archive_ref=re.search(r"csv/out/story/[\w./-]+\.csv",ref)
                    if not archive_ref or not (Path(__file__).resolve().parents[2]/archive_ref[0]).is_file():
                        raise ValueError(f"Unrecognized evidence source {filename}")
                checks["source_tables"].add(filename)
            for tid in _ids(r"\bteam[ _]ids?",ref):
                if tid not in self.clubs:
                    raise ValueError(f"Unknown ABL franchise {tid}")
                checks["teams"].add(tid)
            for pid in _ids(r"\bplayer[ _]ids?",ref):
                if pid not in self.players:
                    raise ValueError(f"Unknown player {pid}")
                checks["players"].add(pid)
            for mid in _ids(r"\bmessage_id",ref):
                if mid not in self.deals:
                    raise ValueError(f"Trade record {mid} absent at this cutoff")
                checks["transactions"].add(mid)
                date = re.search(r"\bdate=(\d{4}-\d{1,2}-\d{1,2})",ref)
                if date and not any(day(d["date"])==day(date[1]) for d in self.deals[mid]):
                    raise ValueError(f"Trade date contradicts message {mid}")
            gids=_ids(r"\b(?:game[ _]ids?|series IDs|meeting IDs|last seven IDs)",ref)
            schedule=bool(re.search(r"\bscheduled?\b|played=0",ref,re.I))
            for gid in gids:
                g=self.games.get(gid)
                if not g:
                    raise ValueError(f"Game {gid} absent from the ABL season")
                if schedule and g["played"]!="0" or not schedule and (g["played"]!="1" or day(g["date"])>self.cutoff):
                    raise ValueError(f"Game {gid} has the wrong completion status for its citation")
                checks["games"].add(gid)
                if not schedule:
                    explicit_games.add(gid)
                if len(gids)==1:
                    on=re.search(r"\bdate=(\d{4}-\d{1,2}-\d{1,2})",ref)
                    if on and day(on[1])!=day(g["date"]):
                        raise ValueError(f"Wrong date cited for game {gid}")
                    for field, column in [("visitor","away_team"),("home","home_team")]:
                        who=re.search(rf"\b{field}=(\d+)",ref)
                        if who and who[1]!=g[column]:
                            raise ValueError(f"Wrong {field} in game {gid}")
        for text in [piece["title"],*(p for _,ps in piece["sections"] for p in ps)]:
            for city,verb,other,x,y,date in self.result.findall(text):
                a,b=self.aliases[city],self.aliases[other]
                if verb in ("fell to","lost to"):
                    a,b=b,a
                results=self.scores[a,b,int(x),int(y)]
                if date:
                    results=[(i,d) for i,d in results if f"{d:%B} {d.day}"==date]
                related=[gid for gid in explicit_games if
                         {int(self.games[gid]["home_team"]),int(self.games[gid]["away_team"])}=={a,b}]
                if related:
                    results=[(i,d) for i,d in results if i in related]
                elif len(explicit_games)==1:
                    results=[]
                if not results:
                    raise ValueError(f"Score contradicts the verified schedule: {city} {verb} {other} {x}-{y}")
                checks["score_claims"]+=1
            for found in self.standing.finditer(text):
                city,w,l=found.groups()
                lead=re.split(r"[.!?]",text[:found.start()])[-1]
                if re.search(r"\b(?:against|since|during|over (?:the|its)|head-to-head|in (?:their|its|one-run|close))\b",lead,re.I):
                    continue
                if self.records[self.aliases[city]]!=[int(w),int(l)]:
                    raise ValueError(f"Record contradicts the verified standings: {city} {w}-{l}")
                checks["record_claims"]+=1
        return {k:sorted(v) if isinstance(v,set) else v for k,v in checks.items()}
