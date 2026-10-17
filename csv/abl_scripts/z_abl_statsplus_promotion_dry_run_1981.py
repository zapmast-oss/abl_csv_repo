#!/usr/bin/env python3
"""Generate StatsPlus Feed 3 semantic review and dry-run promotion artifacts."""

import csv, json, re
from collections import Counter
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'csv'/'out'/'control'
SLUG='1981_asof_1981-07-19'
LEGACY=Path(r'E:\BACKUP\BACKUP 2\ABL 1981')
STAGING=Path(r'C:\Users\earld\OneDrive\Documents\Out of the Park Developments\OOTP Baseball 26\saved_games\Action Baseball League.lg\import_export\abl_transactions_personnel_statsplus')

def rows(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dumpcsv(path,data):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]),extrasaction='ignore');w.writeheader();w.writerows(data)
def dumpjson(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def clean(v):return str(v).replace('|','\\|').replace('\n',' ')
def table(head,data):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join('---' for _ in head)+'|']+['| '+' | '.join(clean(x) for x in r)+' |' for r in data])
def hdr(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return next(csv.reader(f))
def norm(v):return ' '.join(re.findall(r'[a-z0-9]+',str(v).lower()))

def main():
 profile=rows(OUT/f'statsplus_fresh_capture_profile_{SLUG}.csv')
 manifest=rows(OUT/f'statsplus_fresh_capture_manifest_{SLUG}.csv')
 recon=rows(OUT/f'statsplus_legacy_vs_fresh_reconciliation_{SLUG}.csv')
 pby={int(r['numbered_table']):r for r in profile};mby={int(re.match(r'^(\d{2})_',r['file_name']).group(1)):r for r in manifest}
 rby={int(r['legacy_table_number']):r for r in recon if r['legacy_table_number']}

 # Table 1 semantic reconciliation.
 oldpath=LEGACY/'01_ABL_Owner_Info.csv';newpath=STAGING/'01_abl_transactions_personnel_-_owner.csv'
 old,new=rows(oldpath),rows(newpath)
 teams=rows(REPO/'csv'/'ootp_csv'/'teams.csv')
 abbr={r['abbr']:f"{r['name']} {r['nickname']}" for r in teams if r['league_id']=='200' and r['level']=='1'}
 old_by={r['Team']:r for r in old};new_by={abbr[r['TM']]:r for r in new}
 same_teams=set(old_by)==set(new_by);same_people=sum(old_by[t]['Owner Name']==new_by[t]['Name'] for t in old_by)
 changed=[{'team':t,'legacy_owner':old_by[t]['Owner Name'],'fresh_owner':new_by[t]['Name']} for t in old_by if old_by[t]['Owner Name']!=new_by[t]['Name']]
 mappings=[
  ('Number','','derived/removed','Legacy display ordinal has no source meaning; fresh table is keyed by team/name.'),
  ('Owner Name','Name','retained','Owner name; 23/24 identical, with Chicago showing a current personnel change.'),
  ('Team','TM','retained_normalized','Full team name becomes current team abbreviation; all 24 teams resolve.'),
  ('Conference','','removed_but_derivable','Conference is absent but can be derived from governed current team/division data.'),
  ('Age','Age','retained','Direct field mapping.'),
  ('Years of Experience','EXP','retained_normalized','Numeric years become formatted years.'),
  ('Reputation','Rep','retained_normalized','Same concept; current export uses its present label scale.'),
  ('Personality Type','Type','retained_separated','23/24 exact; changed Chicago owner accounts for the exception.'),
  ('Negotiation Tendencies','POS + NEG','retained_separated','Two grouped values become separate positive/negative relationship columns; 23/24 match.'),
  ('Management Style','Patience + Spending + Involvement','retained_separated','Three grouped values become separate fields; Patience and Involvement match 24/24; Spending matches 16/24, with current/normalized category changes.'),
  ('Financial Goal','Priority','retained_normalized','23/24 exact; changed Chicago owner accounts for the exception.'),
  ('Owner Mood','','removed','No corresponding fresh field.'),
  ('Season Objective','','removed','No corresponding fresh field.'),
  ('Specific Goals','','removed','No corresponding fresh field.'),
  ('','Pos','added_layout','Explicit personnel-position code.'),
  ('','Job','added_layout','Explicit job code.'),
  ('','LG','added_context','Explicit league code; not a replacement for legacy conference.'),
  ('','Lev','added_context','Explicit competition level.'),
  ('','NAT','added_detail','Owner nationality.'),
 ]
 semantic=[]
 for oldel,freshel,status,evidence in mappings:
  semantic.append({'table_number':1,'table_family':'owner info','grain':'team/owner','old_row_count':len(old),'fresh_row_count':len(new),
   'old_key_columns':'Team|Owner Name','fresh_key_columns':'TM|Name','old_data_element':oldel,'fresh_data_element':freshel,
   'mapping_status':status,'evidence':evidence,'table_classification':'PARTIAL_MEANING_LOSS',
   'promotion_recommendation':'PROMOTE_ACCEPT_RESHAPED_SCHEMA','confidence_score':'0.96',
   'risk_note':'Promote retained current owner traits only; Owner Mood, Season Objective, and Specific Goals remain unavailable.'})
 dumpcsv(OUT/f'statsplus_table_1_semantic_reconciliation_{SLUG}.csv',semantic)
 samples=[]
 for team in ('Boston Patriots','Chicago Fire','Las Vegas Gamblers'):
  o,n=old_by[team],new_by[team]
  samples.append({'team':team,'legacy_owner':o['Owner Name'],'fresh_owner':n['Name'],'legacy_personality':o['Personality Type'],'fresh_type':n['Type'],
   'legacy_negotiation':o['Negotiation Tendencies'],'fresh_pos_neg':n['POS']+' / '+n['NEG'],
   'legacy_management':o['Management Style'],'fresh_management':n['Patience']+' / '+n['Spending']+' / '+n['Involvement'],
   'legacy_financial_goal':o['Financial Goal'],'fresh_priority':n['Priority']})
 sem_payload={'season':1981,'as_of_date':'1981-07-19','legacy_file':str(oldpath),'fresh_file':str(newpath),
  'table_family':'owner info','grain':'team/owner','old_row_count':len(old),'fresh_row_count':len(new),
  'same_team_population':same_teams,'same_owner_names':same_people,'changed_entities':changed,
  'old_grouped_columns':['Negotiation Tendencies','Management Style'],
  'fresh_separated_columns':['POS','NEG','Patience','Spending','Involvement'],
  'retained_elements':['owner identity','team linkage','age','experience','reputation','personality type','negotiation traits','management traits','financial priority'],
  'added_elements':['Pos','Job','LG','Lev','NAT'],
  'removed_elements':['Owner Mood','Season Objective','Specific Goals','Conference (derivable elsewhere)','Number (display ordinal)'],
  'same_people_entities_represented':'All 24 teams; 23 owner names match and Chicago has a current owner replacement.',
  'grouped_values_in_separated_columns':True,'actual_information_lost':True,'better_normalized':True,
  'classification':'PARTIAL_MEANING_LOSS','promotion_recommendation':'PROMOTE_ACCEPT_RESHAPED_SCHEMA','confidence_score':0.96,
  'story_engine_impact':'Owner personality/negotiation/management context is better structured; mood/objective/specific-goal signals must remain disabled.',
  'authority_impact':'Feed 3 may govern these current owner-context fields after promotion; it does not authorize motives.',
  'downstream_impact':'Consumers must map legacy grouped traits to separated columns and tolerate removed dynamic-goal fields.',
  'risk_note':'Not a drop-in full semantic replacement, but safe for the retained field contract.',
  'data_element_mappings':semantic,'sample_comparisons':samples}
 dumpjson(OUT/f'statsplus_table_1_semantic_reconciliation_{SLUG}.json',sem_payload)
 sem_md=f'''# StatsPlus Table 1 Semantic Reconciliation

**Classification:** `PARTIAL_MEANING_LOSS`  
**Recommendation:** `PROMOTE_ACCEPT_RESHAPED_SCHEMA`  
**Confidence:** 0.96

## Verdict

Table 1 retains the same owner-context family and all 24 team entities. Twenty-three owner names match; Chicago changes from Art Roy to Willie Roy, consistent with a current personnel replacement rather than a join failure.

The legacy grouped fields are substantially preserved in normalized columns. `Negotiation Tendencies` maps to `POS` and `NEG`; `Management Style` maps to `Patience`, `Spending`, and `Involvement`; `Financial Goal` maps to `Priority`. Personality, patience, and involvement mappings are stable apart from the changed Chicago owner. Several spending labels changed from legacy `Economizer` to current `Normal`, which is a value/category update rather than disappearance of the spending concept.

Actual semantic loss exists: `Owner Mood`, `Season Objective`, and `Specific Goals` have no fresh equivalents. Conference is absent but derivable from governed team data. The fresh table adds explicit position, job, league, level, and nationality fields.

Table 1 is safe to promote for retained current owner traits with a limited field contract. It is not a drop-in replacement for mood/objective/goal signals, which must remain unavailable.

## Data-element map

{table(['Legacy element','Fresh element','Status','Evidence'],[[r['old_data_element'],r['fresh_data_element'],r['mapping_status'],r['evidence']] for r in semantic])}

## Sample comparisons

{table(['Team','Legacy/fresh owner','Legacy negotiation','Fresh POS/NEG','Legacy management','Fresh separated management'],[[r['team'],r['legacy_owner']+' / '+r['fresh_owner'],r['legacy_negotiation'],r['fresh_pos_neg'],r['legacy_management'],r['fresh_management']] for r in samples])}

**Risk:** downstream consumers must not infer missing mood or goals, and must not turn owner traits into motives.
'''
 (OUT/f'statsplus_table_1_semantic_reconciliation_{SLUG}.md').write_text(sem_md,encoding='utf-8')

 # Drift and overlap review.
 drift_specs={
  1:('owner info','semantic reshape with partial loss','ACCEPT_WITH_LIMITED_SIGNALS',0.96,'Promote retained traits; disable mood/objective/specific-goal signals.'),
  4:('historical fan interest','layout-only','ACCEPT_SCHEMA_DRIFT',0.99,'Rank removed; team/year interest values remain.'),
  15:('team pitching by division','additive','ACCEPT_ADDED_DETAIL',0.99,'Adds IP, SV, and BS; overlapping pitching fields remain crosscheck-only.'),
  16:('team pitching by league','additive','ACCEPT_ADDED_DETAIL',0.99,'Adds IP, SV, and BS; overlapping pitching fields remain crosscheck-only.'),
  19:('team baserunning','additive','ACCEPT_ADDED_DETAIL',0.98,'Adds team UBR; use only as a labeled StatsPlus-specific enrichment after promotion.'),
  21:('player pitching','additive','ACCEPT_ADDED_DETAIL',0.98,'Adds rWAR; label as StatsPlus rWAR and do not overwrite WAR or raw proof.'),
  24:('team age','additive','ACCEPT_ADDED_DETAIL',0.99,'Adds short-season and rookie-level age splits.'),
  7:('playoff odds by division','reordered overlap','ACCEPT_SCHEMA_DRIFT',1.0,'Same row multiset as table 8 in a different order; model values agree with raw records.'),
  8:('playoff odds by league','reordered overlap','ACCEPT_SCHEMA_DRIFT',1.0,'Same row multiset as table 7 in a different order; preserve report identity and avoid double counting.'),
 }
 drift=[]
 for n,(fam,change,rec,conf,risk) in drift_specs.items():
  oldfile=next(LEGACY.glob(f'{n:02d}_*.csv'));newfile=Path(mby[n]['source_file']);oh,nh=hdr(oldfile),hdr(newfile)
  os,ns=set(map(norm,oh)),set(map(norm,nh))
  drift.append({'table_number':n,'table_family':fam,'old_column_count':len(oh),'new_column_count':len(nh),
   'old_columns':'|'.join(oh),'new_columns':'|'.join(nh),'retained_columns':'|'.join(sorted(os&ns)),
   'removed_columns':'|'.join(sorted(os-ns)),'added_columns':'|'.join(sorted(ns-os)),'change_type':change,
   'story_engine_impact':risk,'authority_impact':'StatsPlus-specific additions require source labels; overlapping observed fields retain raw/sortable authority.',
   'downstream_impact':'Update field mapping and prevent double counting or silent fallback.',
   'recommendation':rec,'confidence_score':f'{conf:.2f}','risk_note':risk})
 dumpcsv(OUT/f'statsplus_schema_drift_review_{SLUG}.csv',drift)
 drift_payload={'season':1981,'as_of_date':'1981-07-19','tables_reviewed':len(drift),'schema_drift_tables':[1,4,15,16,19,21,24],
  'reordered_overlap_tables':[7,8],'table_19_ubr_usable':True,'table_21_rwar_usable':True,'reviews':drift}
 dumpjson(OUT/f'statsplus_schema_drift_review_{SLUG}.json',drift_payload)
 drift_md='# StatsPlus Schema-Drift and Overlap Review\n\n'+table(['Table','Family','Old/new cols','Change','Recommendation','Confidence'],[[r['table_number'],r['table_family'],str(r['old_column_count'])+'/'+str(r['new_column_count']),r['change_type'],r['recommendation'],r['confidence_score']] for r in drift])+'''\n\n## Specific decisions

- **Table 19 UBR:** usable after promotion as a labeled StatsPlus-specific team baserunning value. It must not silently substitute for a differently defined sortable field.
- **Table 21 rWAR:** usable after promotion as labeled StatsPlus rWAR context. It does not override WAR, raw game proof, or sortable authority.
- **Tables 7 and 8:** same row multiset, different order. Both schemas and values are safe, but their two sort/report identities must not be double counted.
- **Table 1:** retained traits are safe under a reshaped field contract; mood/objective/specific goals remain unavailable.
'''
 (OUT/f'statsplus_schema_drift_review_{SLUG}.md').write_text(drift_md,encoding='utf-8')

 # Dry-run promotion mapping.
 actions={1:'PROMOTE_ACCEPT_RESHAPED_SCHEMA',4:'PROMOTE_ACCEPT_SCHEMA_DRIFT',7:'PROMOTE_REORDERED_OVERLAP',8:'PROMOTE_REORDERED_OVERLAP',
          15:'PROMOTE_ACCEPT_ADDED_DETAIL',16:'PROMOTE_ACCEPT_ADDED_DETAIL',19:'PROMOTE_ACCEPT_ADDED_DETAIL',21:'PROMOTE_ACCEPT_ADDED_DETAIL',24:'PROMOTE_ACCEPT_ADDED_DETAIL'}
 risks={n:drift_specs[n][4] for n in drift_specs}
 dry=[]
 for n in sorted(pby):
  p,m=pby[n],mby[n];action=actions.get(n,'PROMOTE_AS_STATSPLUS_CURRENT')
  dry.append({'table_number':n,'table_family':p['inferred_table_family'],'staging_source_path':m['source_file'],
   'proposed_official_target_path':str(REPO/'csv'/'statsplus'/'current'/m['file_name']),
   'proposed_official_file_name':m['file_name'],'source_type':p['inferred_source_type'],
   'source_authority_classification':p['authority_class'],'story_value':p['story_value'],'current_action':action,
   'row_count':m['row_count'],'column_count':m['column_count'],'sha256':m['sha256'],
   'schema_status':'reshaped_limited' if n==1 else 'reordered_overlap' if n in (7,8) else 'accepted_drift' if n in actions else 'compatible',
   'confidence_score':drift_specs[n][3] if n in drift_specs else p['match_confidence'],
   'risk_note':risks.get(n,'Promote only with field-level authority metadata; no story-engine enablement yet.')})
 for n,reason in ((6,'Raw OOTP governs current standings.'),(27,'GTOC history does not change until postseason.')):
  dry.append({'table_number':n,'table_family':'league standings' if n==6 else 'Grand Tournament of Champions',
   'staging_source_path':'','proposed_official_target_path':'','proposed_official_file_name':'','source_type':'intentionally absent',
   'source_authority_classification':'RAW_OOTP_GOVERNS' if n==6 else 'HISTORICAL_CONTEXT', 'story_value':'hold','current_action':'INTENTIONALLY_EXCLUDED',
   'row_count':'','column_count':'','sha256':'','schema_status':'intentional_exclusion','confidence_score':1.0,'risk_note':reason})
 dry.sort(key=lambda r:r['table_number']);dumpcsv(OUT/f'statsplus_promotion_dry_run_{SLUG}.csv',dry)
 counts=Counter(r['current_action'] for r in dry)
 dry_payload={'season':1981,'as_of_date':'1981-07-19','target_folder':'csv/statsplus/current/','dry_run_only':True,
  'promotion_authorized':False,'counts':dict(counts),'tables':dry}
 dumpjson(OUT/f'statsplus_promotion_dry_run_{SLUG}.json',dry_payload)
 dry_md='# StatsPlus Promotion Dry Run\n\n**No files were copied or promoted.**\n\n'+table(['#','Family','Action','Rows','Cols','Authority','Risk'],[[r['table_number'],r['table_family'],r['current_action'],r['row_count'],r['column_count'],r['source_authority_classification'],r['risk_note']] for r in dry])+f'''\n\n## Counts

{table(['Action','Tables'],sorted(counts.items()))}

Proposed target: `csv/statsplus/current/`. Source filenames are preserved. Promotion requires separate authorization.
'''
 (OUT/f'statsplus_promotion_dry_run_{SLUG}.md').write_text(dry_md,encoding='utf-8')

 # Readiness.
 promoted=[r for r in dry if r['current_action'].startswith('PROMOTE_')];holds=[r for r in dry if r['current_action'].startswith('HOLD_')]
 readiness={'season':1981,'as_of_date':'1981-07-19','fresh_tables_recommended_for_promotion':len(promoted),
  'accepted_schema_drift':counts['PROMOTE_ACCEPT_SCHEMA_DRIFT'],'accepted_reshaped_schema':counts['PROMOTE_ACCEPT_RESHAPED_SCHEMA'],
  'additive_improvements':counts['PROMOTE_ACCEPT_ADDED_DETAIL'],'reordered_overlap_tables':counts['PROMOTE_REORDERED_OVERLAP'],
  'hold_review_tables':len(holds),'intentional_exclusions':[6,27],'blockers':[],
  'feed3_safe_for_separate_authorized_promotion':True,'promotion_authorized':False,
  'conditions':['Retain field-level authority labels.','Disable missing table 1 mood/objective/goal signals.','Do not double count tables 7 and 8.','Do not enable story signals until post-promotion integration.']}
 dumpjson(OUT/f'statsplus_promotion_readiness_{SLUG}.json',readiness)
 readiness_md=f'''# StatsPlus Promotion Readiness

- Fresh tables recommended: **{len(promoted)}**.
- Accepted schema drift: **{counts['PROMOTE_ACCEPT_SCHEMA_DRIFT']}**.
- Accepted reshaped schema: **{counts['PROMOTE_ACCEPT_RESHAPED_SCHEMA']}**.
- Additive improvements: **{counts['PROMOTE_ACCEPT_ADDED_DETAIL']}**.
- Reordered overlap only: **{counts['PROMOTE_REORDERED_OVERLAP']}**.
- Hold/review: **{len(holds)}**.
- Intentional exclusions: **6 and 27**.
- Blockers: **None**, subject to the conditions below.

Feed 3 is safe for a separate authorized promotion task. Promotion is not authorized by this report.

## Conditions

- Preserve authority labels and checksums.
- Disable table 1 mood/objective/specific-goal signals.
- Preserve tables 7 and 8 as separate report identities without double counting.
- Treat UBR and rWAR as labeled StatsPlus-specific enrichment.
- Do not enable story-engine signals until post-promotion integration is separately approved.
'''
 (OUT/f'statsplus_promotion_readiness_{SLUG}.md').write_text(readiness_md,encoding='utf-8')

 # Signal enablement review.
 signals=[
  ('playoff odds pressure','7|8','READY_AFTER_PROMOTION','Model probability; never standings proof.'),
  ('ELO/team strength','10','READY_AFTER_PROMOTION','Labeled StatsPlus model value.'),
  ('BaseRuns over/under-performance','9','READY_AFTER_PROMOTION','Expected-performance model, not observed result.'),
  ('team WAR','11','READY_AFTER_PROMOTION','Labeled StatsPlus team value.'),
  ('injury context','12','READY_AFTER_PROMOTION','Team summary; avoid unsupported player causation.'),
  ('fan interest','4|5','READY_AFTER_PROMOTION','Do not infer emotion beyond measured fields.'),
  ('financial pressure','3','READY_AFTER_PROMOTION','Resource/results context; no blame or motive.'),
  ('owner/front-office/personnel context','1|2','READY_WITH_LIMITED_SIGNALS','Table 1 goals/mood absent; staff IDs remain unavailable.'),
  ('team age/context','24','READY_AFTER_PROMOTION','Adds lower-level age detail.'),
  ('best batting game','25','READY_AFTER_PROMOTION','StatsPlus ranking; raw game proof remains authority.'),
  ('best pitching game','26','READY_AFTER_PROMOTION','StatsPlus ranking; raw game proof remains authority.'),
  ('team/player baserunning','19|22','READY_AFTER_PROMOTION','Label StatsPlus definitions.'),
  ('UBR','19','READY_AFTER_PROMOTION','Usable as StatsPlus-specific team UBR; no silent substitution.'),
  ('rWAR','21','READY_AFTER_PROMOTION','Usable as StatsPlus-specific rWAR; does not override WAR.'),
 ]
 signal_rows=[{'signal':a,'source_tables':b,'review_status':c,'engine_enabled_now':'no','authority_boundary':d} for a,b,c,d in signals]
 sig_payload={'season':1981,'as_of_date':'1981-07-19','engine_changes_authorized':False,'signals':signal_rows}
 dumpjson(OUT/f'statsplus_story_signal_enablement_{SLUG}.json',sig_payload)
 sig_md='# StatsPlus Story-Signal Enablement Review\n\n**No signals are enabled by this review.**\n\n'+table(['Signal','Tables','Review status','Enabled now','Boundary'],[[r['signal'],r['source_tables'],r['review_status'],r['engine_enabled_now'],r['authority_boundary']] for r in signal_rows])+'''\n\nUBR and rWAR are usable after promotion as explicitly labeled StatsPlus values. Owner/personnel context is limited by absent owner goal fields and unresolved staff IDs.\n'''
 (OUT/f'statsplus_story_signal_enablement_{SLUG}.md').write_text(sig_md,encoding='utf-8')

 print(json.dumps({'tables_reviewed':27,'fresh_tables':25,'table_1_classification':'PARTIAL_MEANING_LOSS',
  'promotion_ready_tables':len(promoted),'hold_review_tables':len(holds),'reordered_overlap_tables':[7,8],
  'intentional_exclusions':[6,27],'table_19_ubr_usable':True,'table_21_rwar_usable':True,
  'feed3_ready_for_actual_promotion':True,'promotion_performed':False},indent=2))

if __name__=='__main__':main()
