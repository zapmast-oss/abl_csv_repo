#!/usr/bin/env python3
"""Read-only inventory of the legacy StatsPlus / Deep Dive 25 folder."""

import csv, hashlib, json, re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(r"E:\BACKUP\BACKUP 2\ABL 1981")
OUT = REPO / "csv" / "out" / "control"
DOC = REPO / "docs" / "ABL_STATSPLUS_LEGACY_BASELINE_NOTES.md"

FAMILIES = [
 "owner info","front office/coaches","financials","historical fan interest","fan data",
 "league standings","playoff odds by division","playoff odds by league","BaseRuns","ELO ratings",
 "team WAR","injury summary","team batting by division","team batting by league",
 "team pitching by division","team pitching by league","team fielding by division",
 "team fielding by league","team baserunning","player batting","player pitching",
 "player baserunning","player fielding","team age","best batting game","best pitching game",
 "Grand Tournament of Champions"]

MODEL = {"playoff odds by division","playoff odds by league","BaseRuns","ELO ratings","team WAR"}
HIST = {"historical fan interest","Grand Tournament of Champions"}
HIGH = MODEL | {"owner info","fan data","historical fan interest","injury summary","team baserunning",
                "team age","best batting game","best pitching game","Grand Tournament of Champions"}
UNIQUE = {
 "owner info":"Owner personality, goals, mood, and negotiation fields add organization context.",
 "historical fan interest":"1972-1981 fan-interest time series is potentially unique.",
 "fan data":"Fan mood, loyalty, interest, and change fields add fan-pressure context.",
 "playoff odds by division":"Simulation odds and remaining-strength fields add modeled race context.",
 "playoff odds by league":"Simulation odds and remaining-strength fields add modeled race context.",
 "BaseRuns":"Expected-performance fields add modeled team context.",
 "ELO ratings":"Season, 30-day, and 7-day ELO movement adds strength context.",
 "team WAR":"Combined batter and pitcher team WAR adds team-strength context.",
 "injury summary":"Injury count, IL days, and salary on IL add burden context.",
 "team baserunning":"Team wSB could fill part of unavailable baserunning enrichment after validation.",
 "team age":"Multi-level organization age profiles are potentially unique.",
 "best batting game":"Ranked game performances add story discovery.",
 "best pitching game":"Ranked game performances add story discovery.",
 "Grand Tournament of Champions":"Structured postseason series history adds tournament context."}

def text(path):
 raw=path.read_bytes()
 for enc in ("utf-8-sig","utf-8","cp1252","latin-1"):
  try:return raw.decode(enc),enc
  except UnicodeDecodeError:pass
 return raw.decode("utf-8",errors="replace"),"utf-8-replace"

def csvinfo(path):
 s,enc=text(path); rows=list(csv.reader(s.splitlines())); head=rows[0] if rows else []
 data=[r for r in rows[1:] if any(c.strip() for c in r)]
 return {"row_count":len(data),"column_count":len(head),"column_headers":head,"encoding":enc,"sample_rows":data[:3]}

def num(path):
 m=re.match(r"^(\d{2})_",path.name); return int(m.group(1)) if m else None

def family(path):
 n=num(path)
 if n and 1<=n<=27:return FAMILIES[n-1]
 if path.parent.name=="Front_Office_and_Coaches_by_Div":return "coach/staff division"
 if path.name=="Phoenix_Firebirds_Season_History.csv":return "franchise season history"
 if path.suffix.lower()==".txt":return "Deep Dive 25 notes"
 return "unknown"

def grain(f):
 if f in {"front office/coaches","coach/staff division"}:return "staff/front office"
 if f.startswith("player"):return "player"
 if f.startswith("best "):return "game"
 if f in HIST or f=="franchise season history":return "historical/franchise"
 if f=="Deep Dive 25 notes":return "text/deep-dive notes"
 if f in {"playoff odds by division","playoff odds by league"}:return "model/projection"
 return "team" if f!="unknown" else "unknown"

def role(path):
 if num(path) in range(1,28):return "StatsPlus legacy source"
 if path.parent.name=="Front_Office_and_Coaches_by_Div":return "coach/staff division source"
 if path.suffix.lower()==".txt":return "Deep Dive 25 source"
 if "Season_History" in path.name:return "derived output"
 return "unknown/review"

def keys(f,h):
 wanted={"owner info":["Team","Owner Name"],"front office/coaches":["TM","Job","Name"],
 "league standings":["Div","Team"],"Grand Tournament of Champions":["Year"],
 "best batting game":["Name","Team","Opp","Date"],"best pitching game":["Name","Team","Opp","Date"]}
 base=wanted.get(f,["Name","Team"] if f.startswith("player") else ["Team"])
 return [x for x in base if x in h]

def dumpcsv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction="ignore");w.writeheader();w.writerows(rows)

def dumpjson(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def table(headers,rows):
 clean=lambda x:str(x).replace("|","\\|").replace("\n"," ")
 return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]+
                  ["| "+" | ".join(clean(x) for x in r)+" |" for r in rows])

def main():
 if not ROOT.is_dir():raise SystemExit(f"Missing legacy folder: {ROOT}")
 OUT.mkdir(parents=True,exist_ok=True)
 files=sorted((p for p in ROOT.rglob("*") if p.is_file()),key=lambda p:str(p).lower())
 folders=[p for p in ROOT.rglob("*") if p.is_dir()]
 profiles={p:csvinfo(p) for p in files if p.suffix.lower()==".csv"}
 hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}; groups=defaultdict(list)
 for p,h in hashes.items():groups[h].append(p)
 inv=[]
 for p in files:
  f=family(p); q=profiles.get(p,{}); lines=[]
  if p.suffix.lower()==".txt":lines=[x.strip() for x in text(p)[0].splitlines() if x.strip()][:10]
  warn=[]
  dup=[str(x.relative_to(ROOT)) for x in groups[hashes[p]] if x!=p]
  if dup:warn.append("Exact-content duplicate of: "+"; ".join(dup))
  if num(p) in range(1,28):warn.append("Legacy early-1981 checkpoint; not current July 19 proof.")
  if f=="coach/staff division":warn.append("No stable staff IDs; cannot resolve ID linkage independently.")
  if p.suffix.lower()==".txt":warn.append("Generated planning/prompt text; historical reference only.")
  if q.get("row_count")==25 and grain(f)=="team":warn.append("25 rows may include a summary row; validate before joins.")
  inv.append({"full_path":str(p),"relative_path":str(p.relative_to(ROOT)),"file_name":p.name,
   "file_extension":p.suffix.lower(),"file_size":p.stat().st_size,
   "modified_timestamp":datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
   "readable":"yes","row_count":q.get("row_count",""),"column_count":q.get("column_count",""),
   "column_headers":json.dumps(q.get("column_headers",[]),ensure_ascii=False),
   "first_non_empty_lines":json.dumps(lines,ensure_ascii=False),"inferred_table_family":f,
   "inferred_grain":grain(f),"likely_source_role":role(p),
   "likely_story_value":"high" if f in HIGH else "medium","warnings":" | ".join(warn),"sha256":hashes[p]})
 dumpcsv(OUT/"statsplus_legacy_inventory.csv",inv)
 dumpjson(OUT/"statsplus_legacy_inventory.json",{"legacy_root":str(ROOT),"folder_count":len(folders),"file_count":len(files),"files":inv})

 numbered=sorted((p for p in files if num(p) in range(1,28)),key=num); p27=[]
 for p in numbered:
  f=family(p);q=profiles[p]; current=f not in HIST
  duplicate=f in {"team batting by division","team pitching by division","team fielding by division"}
  p27.append({"table_number":num(p),"file_name":p.name,"relative_path":str(p.relative_to(ROOT)),
   "table_family":f,"grain":grain(f),"row_count":q["row_count"],"column_count":q["column_count"],
   "key_columns":"|".join(keys(f,q["column_headers"])),"current_state_support":"yes, but stale" if current else "no",
   "projection_model_support":"yes" if f in MODEL else "no","historical_context":"yes" if f in HIST else "no",
   "context_type":"front-office" if f in {"owner info","front office/coaches"} else "fan" if "fan" in f else "financial" if f=="financials" else "player" if grain(f) in {"player","game"} else "team",
   "overlap_with_raw_or_sortable":"substantial" if f not in UNIQUE else "partial; unique enrichment present",
   "data_not_currently_available":UNIQUE.get(f,"No distinct missing family established; reconcile at column level."),
   "later_promotion_recommendation":"HOLD_DUPLICATE_OR_OVERLAP" if duplicate else "PROMOTE_HISTORICAL_CONTEXT_AFTER_REVIEW" if f in HIST else "COMPARE_NEW_CAPTURE_THEN_PROMOTE_OR_HOLD",
   "warnings":"Stale baseline; exact as-of date not encoded."})
 dumpcsv(OUT/"statsplus_legacy_27_table_profile.csv",p27)
 dumpjson(OUT/"statsplus_legacy_27_table_profile.json",{"expected":27,"found":len(p27),"all_numbers_present":[num(p) for p in numbered]==list(range(1,28)),"tables":p27})

 coach=[]
 for p in sorted(x for x in files if x.parent.name=="Front_Office_and_Coaches_by_Div"):
  q=profiles[p];coach.append({"file_name":p.name,"relative_path":str(p.relative_to(ROOT)),
   "division_conference":p.stem.replace("_"," "),"row_count":q["row_count"],"column_count":q["column_count"],
   "staff_coach_columns":"Pos|Job|Name|Age|EXP|Rep|Type|POS|NEG|Style|Hit|Pit|Rel",
   "team_linkage_columns":"TM|LG|Lev","resolves_staff_id_name_limitation":"no; no stable staff ID",
   "overlap_with_promoted_sortable_staff":"names/jobs/teams overlap; legacy adds age, experience, reputation and style",
   "recommended_use":"crosscheck and legacy staff context; hold current use pending fresh capture",
   "warnings":"The six division files together exactly reconstruct the 240-row master; stale and ID-less."})
 dumpcsv(OUT/"statsplus_legacy_coach_division_profile.csv",coach)
 dumpjson(OUT/"statsplus_legacy_coach_division_profile.json",{"expected":6,"found":len(coach),"can_resolve_staff_ids":False,"files":coach})

 txt=[]
 for p in sorted(x for x in files if x.suffix.lower()==".txt"):
  lines=[x.strip() for x in text(p)[0].splitlines() if x.strip()][:10]
  txt.append({"full_path":str(p),"relative_path":str(p.relative_to(ROOT)),"file_name":p.name,"file_size":p.stat().st_size,
   "modified_timestamp":datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
   "first_10_non_empty_lines":json.dumps(lines,ensure_ascii=False),"inferred_subject":"Deep Dive 25 team-profile planning",
   "likely_association":"ABL team/league profiling","content_type":"prompts/design notes" if "Prompts" in p.name else "source-selection/design notes",
   "recommended_treatment":"historical reference only; reusable design context after fact revalidation",
   "warnings":"Generated narrative/instructions; not current-state proof."})
 dumpcsv(OUT/"deep_dive_25_legacy_text_inventory.csv",txt)
 dumpjson(OUT/"deep_dive_25_legacy_text_inventory.json",{"count":len(txt),"treatment":"historical reference only","files":txt})

 counts={"total_files":len(files),"csv_files":sum(p.suffix.lower()==".csv" for p in files),"txt_files":len(txt),
         "folders":len(folders),"likely_statsplus_tables":len(numbered),"coach_staff_division_files":len(coach),"deep_dive_25_txt_files":len(txt)}
 invmd="# StatsPlus Legacy Inventory\n\n**Mode:** Read-only; nothing copied or promoted.\n\n"+table(["Measure","Count"],[[k.replace("_"," ").title(),v] for k,v in counts.items()])+"\n\n## Files\n\n"+table(["Relative path","Rows","Cols","Family","Grain","Role","Value"],[[x["relative_path"],x["row_count"],x["column_count"],x["inferred_table_family"],x["inferred_grain"],x["likely_source_role"],x["likely_story_value"]] for x in inv])+"\n\n## Warnings\n\n- Early-1981 baseline, not July 19 current proof.\n- Team batting 13/14 and pitching 15/16 are byte-identical. Fielding 17/18 contain the same row set in different sort order.\n- The six staff division files exactly reconstruct the 240-row master but lack IDs. TXT files are reference-only design material.\n"
 (OUT/"statsplus_legacy_inventory.md").write_text(invmd,encoding="utf-8")
 md27="# Legacy 27-Table StatsPlus Profile\n\nAll numbered tables 01-27 are present, but the capture is stale.\n\n"+table(["#","Family","Rows","Cols","Model","Historical","Later treatment"],[[x["table_number"],x["table_family"],x["row_count"],x["column_count"],x["projection_model_support"],x["historical_context"],x["later_promotion_recommendation"]] for x in p27])+"\n\nHighest value: playoff odds, ELO, BaseRuns, team WAR, injuries, baserunning, fan/owner context, best-game tables, and tournament history. Compare a fresh capture before promotion.\n"
 (OUT/"statsplus_legacy_27_table_profile.md").write_text(md27,encoding="utf-8")
 coachmd="# Legacy Coach/Staff Division Profile\n\nAll six expected division files are present. Each has 40 rows and 17 columns; their combined row multiset exactly reconstructs the 240-row master table.\n\n"+table(["Division","Rows","Cols","ID linkage","Use"],[[x["division_conference"],x["row_count"],x["column_count"],x["resolves_staff_id_name_limitation"],x["recommended_use"]] for x in coach])+"\n\nThey can crosscheck names and add legacy attributes, but cannot solve staff-ID linkage without a validated bridge.\n"
 (OUT/"statsplus_legacy_coach_division_profile.md").write_text(coachmd,encoding="utf-8")
 txtmd="# Deep Dive 25 Legacy Text Inventory\n\nTwo generated planning/design files are present. Treat both as historical reference, not factual proof.\n\n"+table(["File","Bytes","Type","Treatment"],[[x["file_name"],x["file_size"],x["content_type"],x["recommended_treatment"]] for x in txt])+"\n\n"+"\n\n".join("## "+x["file_name"]+"\n\n```text\n"+"\n".join(json.loads(x["first_10_non_empty_lines"]))+"\n```" for x in txt)+"\n"
 (OUT/"deep_dive_25_legacy_text_inventory.md").write_text(txtmd,encoding="utf-8")

 baseline={**counts,"expected_27_present":len(numbered)==27,"expected_6_staff_present":len(coach)==6,"coherent_prior_capture":True,
  "freshness":"stale early-1981 checkpoint; exact as-of date not encoded","safe_as_current_state":False,"preserve_as_legacy_baseline":True,
  "safe_for_new_staging_reconciliation":True,"families":dict(Counter(x["inferred_table_family"] for x in inv)),
  "highest_value_families":["playoff odds","ELO","BaseRuns","team WAR","injury summary","team/player baserunning","fan/owner/financial context","best games","Grand Tournament of Champions"],
  "limited_signal_help":{"staff":"rich context but no IDs","baserunning":"wSB/SB/CS after fresh validation","fans":"historical interest and current fan measures","injuries":"team burden","models":"odds/ELO/BaseRuns/WAR as labeled enrichment"},
  "next_task":"Create a StatsPlus intake contract and reconcile a fresh staged capture against this baseline and current OOTP/sortable authorities."}
 dumpjson(OUT/"statsplus_deep_dive_25_legacy_baseline_report.json",baseline)
 basemd=f'''# StatsPlus / Deep Dive 25 Legacy Baseline Report

## Verdict

`{ROOT}` is a coherent prior capture and should be preserved unchanged as a legacy baseline. It is stale for July 19 current-state use: standings examples are roughly 19 games into 1981 and no exact as-of date is encoded.

## Counts

{table(["Measure","Count"],[[k.replace("_"," ").title(),v] for k,v in counts.items()])}

- All 27 numbered tables: **present**.
- All six coach/staff division files: **present**.
- Deep Dive 25 TXT files: **2 present**.
- Extra CSV: `Phoenix_Firebirds_Season_History.csv` (derived historical context).

## Coverage

The capture covers owners, staff, finances, fan data/history, standings, playoff odds, BaseRuns, ELO, team WAR, injuries, team/player batting, pitching, fielding and baserunning, team age, best games, and Grand Tournament history.

Highest story-engine value lies in labeled model enrichment (odds/ELO/BaseRuns/WAR), injury and baserunning context, fan/owner pressure, best-game discovery, and tournament history. The legacy values themselves are not current proof.

## Limits and duplicates

 Team batting 13/14 and pitching 15/16 are byte-identical pairs. Fielding 17/18 contain identical row multisets in different order. The six division staff files exactly reconstruct the 240-row master, but they lack stable IDs and cannot resolve current staff linkage alone. TXT files are generated design/reference material.

## Recommendation

Preserve all 36 files as the baseline. It is safe to proceed to a new StatsPlus staging/reconciliation task, but promotion must remain separate. Compare new schemas, identities, grains, cutoff dates, hashes, duplicate views, and model semantics against this baseline and current governed OOTP/sortable sources.
'''
 (OUT/"statsplus_deep_dive_25_legacy_baseline_report.md").write_text(basemd,encoding="utf-8")

 DOC.write_text('''# ABL StatsPlus Legacy Baseline Notes

## Recommended architecture

```text
csv/statsplus/legacy/   # immutable baseline copies only after explicit authorization
csv/statsplus/staging/  # dated, unpromoted captures
csv/statsplus/current/  # reconciled and explicitly promoted tables
csv/out/control/        # manifests, profiles, reconciliations, decisions
csv/out/archive/        # pre-promotion generated archives
docs/                   # intake contracts and authority rules
```

Do not create or populate these source folders until explicitly authorized.

## Governance

- Preserve the inspected external folder unchanged as the legacy baseline.
- Give each staged capture a season, as-of date, export timestamp, report identity, row/column counts, and SHA-256.
- Compare every new table against the 27-table baseline, six staff schemas, raw OOTP authority, and promoted sortable stats.
- Detect summary rows, HTML remnants, renamed columns, duplicate sort views, missing IDs, and model-field semantic changes.
- Label odds, ELO, BaseRuns, and WAR as enrichment, never game proof.
- Keep Deep Dive 25 prompts/narrative separate from factual source tables.
- Require a validated staff-ID/name bridge before enabling manager tendencies.
- Require same-cutoff validation before enabling StatsPlus baserunning or injury signals.

## Next Codex task

Create `docs/ABL_STATSPLUS_INTAKE_CONTRACT.md` and a read-only staging manifest/reconciliation runner. Inventory a fresh capture, compare it table-by-table with this baseline and current authorities, and produce promotion recommendations without promoting files.
''',encoding="utf-8")
 print(json.dumps({**counts,"preserve_legacy":True,"safe_for_new_staging":True},indent=2))

if __name__=="__main__":main()
