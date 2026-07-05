#!/usr/bin/env python3
"""Append typed Feed 3 evidence without mutating official story artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[2];SP=REPO/'csv'/'statsplus'/'current';CONTROL=REPO/'csv'/'out'/'control';ENRICH=REPO/'csv'/'out'/'story'/'enrichment'
SLUG='1981_07_20_asof_1981-07-19';CSLUG='1981_asof_1981-07-19'
CAND=REPO/f'csv/out/story/candidates/story_candidates_{SLUG}.csv';EVID=REPO/f'csv/out/story/candidates/story_evidence_{SLUG}.csv'
OVER=ENRICH/f'story_candidates_statsplus_overlay_{SLUG}.csv';DICT=CONTROL/f'statsplus_story_signal_dictionary_{CSLUG}.csv';RECOMMEND=ENRICH/f'statsplus_story_integration_recommendation_{SLUG}.md'
OFFICIAL=[CAND,EVID,REPO/f'csv/out/story/menus/story_menu_{SLUG}.csv',REPO/f'csv/out/story/editorial/observer_editorial_board_{SLUG}.csv',REPO/f'csv/out/story/editorial/observer_story_slate_{SLUG}.md',REPO/f'csv/out/story/production/eb_production_package_{SLUG}.md',REPO/f'csv/out/story/scripts/baseball_observer_segment_{SLUG}.md']

TYPE_CONFIG={
 'statsplus_playoff_odds':('7|8','07_statsplus_Playoff_Odds_Div.csv|08_statsplus_Playoff_Odds_League.csv','race','Div %|PO %|AvgW|rSoS','Modeled race probability and pressure.','Do not replace official standings or treat probability as destiny.','Playoff odds are model context, not destiny.'),
 'statsplus_elo':('10','10_statsplus_Elo.csv','team','Elo|Seas.+/-|30d+/-|7d+/-','Modeled team-strength and movement context.','Do not use ELO as standings proof.','ELO is team-strength context, not standings proof.'),
 'statsplus_baseruns':('9','09_statsplus_Base_Runs.csv','team','xRS|xRA|pWΔ|xWΔ','Underlying expected-performance context.','Do not promise correction or label residuals luck.','BaseRuns describes underlying performance, not guaranteed correction.'),
 'statsplus_team_war':('11','11_statsplus_Team_WAR.csv','team','BatterWAR|PitcherWAR|TotalWAR','Labeled team-value composition.','Do not substitute WAR rank for standings or outcomes.','StatsPlus WAR is enrichment and must remain source-labeled.'),
 'statsplus_injury_context':('12','12_statsplus_injury_sumary.csv','context','Count|DL Days|$ on DL','Aggregate team injury burden.','Do not fabricate player-level cause or explain a result without linkage.','Injury context adds texture but does not establish cause.'),
 'statsplus_fan_interest_context':('4|5','04_statsplus_Historical_Fan_Interest.csv|05_statsplus_Fan_Data.csv','context','Fan Interest|30-dayFI Δ|Avg.Att.|% Full','Measured fan-interest and attendance context.','Do not infer fan emotion or causal effect.','Fan interest is context, not fan emotion.'),
 'statsplus_financial_context':('3','03_statsplus_Finantials.csv','context','Payroll|Budget|TotalRevenue|Cash(for Trades)','Resource and organization-pressure context.','Do not infer blame, intent, or future transactions.','Financial context is pressure/context, not blame.'),
 'statsplus_team_age_context':('24','24_statsplus_Team_Age.csv','team','ML Age|ML P Age|ML B Age|minor-level ages','Organization-age context.','Do not infer development success or decline from age alone.','Team age is descriptive organization context.'),
 'statsplus_team_baserunning':('19','19_statsplus_Team_Baserunning.csv','team','SB|CS|SB %|wSB','Labeled team baserunning context.','Do not silently substitute for other metric definitions or claim causation.','StatsPlus baserunning definitions must remain labeled.'),
 'statsplus_player_baserunning':('22','22_statsplus_Player_Baserunning.csv','player','Name|Team|WSB|SB|CS|SB%','Labeled player baserunning context.','Do not override validated events or other baserunning definitions.','StatsPlus player baserunning is supplemental.'),
 'statsplus_ubr':('19','19_statsplus_Team_Baserunning.csv','team','UBR','Labeled StatsPlus-specific UBR context.','Do not use as an unlabeled replacement metric or causal proof.','UBR is StatsPlus-specific baserunning enrichment.'),
 'statsplus_pitcher_rwar':('21','21_statsplus_Player_Pitching.csv','player','rWAR|WAR|ERA|FIP|xFIP','Alternate labeled pitcher-value lens.','Do not override sortable WAR, raw proof, or decide awards.','rWAR is StatsPlus-specific and must not override existing WAR.'),
}

def parse_args():
 p=argparse.ArgumentParser(description='Build append-only Feed 3 evidence preview.')
 p.add_argument('--disable-feed3-evidence',action='store_true')
 p.add_argument('--enable-feed3-rank-adjustment',action='store_true')
 p.add_argument('--enable-feed3-new-candidates',action='store_true')
 p.add_argument('--enable-owner-front-office-signals',action='store_true')
 p.add_argument('--enable-best-game-discovery-signals',action='store_true')
 p.add_argument('--enable-historical-fan-interpretation',action='store_true')
 return p.parse_args()
def rr(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dumpcsv(p,r,fields=None):
 p.parent.mkdir(parents=True,exist_ok=True);fields=fields or list(r[0])
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(r)
def dumpjson(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def clean(x):return str(x).replace('|','\\|').replace('\n',' ')
def mdtable(h,r):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|']+['| '+' | '.join(clean(v) for v in x)+' |' for x in r])

def evidence_plan(cid):
 if 'national-baseball-conference-central-division' in cid:return ['statsplus_playoff_odds','statsplus_elo','statsplus_team_war','statsplus_injury_context','statsplus_fan_interest_context'],'race','Dallas–Detroit'
 if 'national-baseball-conference-western-division' in cid:return ['statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war'],'race','San Francisco–Phoenix'
 if 'american-baseball-conference-western-division' in cid:return ['statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war'],'race','Las Vegas–Seattle'
 if 'american-baseball-conference-eastern-division' in cid:return ['statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war'],'race','Boston–New York'
 if 'american-baseball-conference-central-division' in cid:return ['statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war'],'race','Houston–Cincinnati'
 if 'scheduled_matchup_stakes' in cid:return ['statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war','statsplus_team_baserunning','statsplus_ubr'],'matchup','Las Vegas–Denver'
 if 'ace_value_leader' in cid:return ['statsplus_pitcher_rwar'],'player','Jose Coronado'
 if 'position_player_value_leader' in cid:return ['statsplus_playoff_odds','statsplus_team_war'],'context','Manny Flores / Charlotte'
 if 'weekly_rise' in cid:return ['statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war','statsplus_injury_context'],'team','Atlanta'
 if 'weekly_fall' in cid:return ['statsplus_injury_context'],'team','Philadelphia'
 if 'pythag_record_gap__3' in cid:return ['statsplus_baseruns','statsplus_elo','statsplus_team_war'],'team','Atlanta'
 if 'pythag_record_gap__24' in cid:return ['statsplus_playoff_odds','statsplus_baseruns','statsplus_elo'],'team','Seattle'
 if 'payroll_record_pressure' in cid:return ['statsplus_financial_context','statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war','statsplus_injury_context','statsplus_fan_interest_context'],'context','Tampa Bay'
 if 'attendance_interest' in cid:return ['statsplus_fan_interest_context'],'context','Detroit'
 if 'prior_qualifier_echo' in cid:return ['statsplus_playoff_odds','statsplus_elo','statsplus_team_war'],'context','Charlotte'
 raise ValueError(f'No Feed 3 plan for {cid}')

def main():
 a=parse_args();flags={'enable_feed3_evidence':not a.disable_feed3_evidence,'enable_feed3_rank_adjustment':a.enable_feed3_rank_adjustment,
  'enable_feed3_new_candidates':a.enable_feed3_new_candidates,'enable_owner_front_office_signals':a.enable_owner_front_office_signals,
  'enable_best_game_discovery_signals':a.enable_best_game_discovery_signals,'enable_historical_fan_interpretation':a.enable_historical_fan_interpretation}
 if not flags['enable_feed3_evidence']:raise SystemExit('Feed 3 evidence is disabled; no output generated.')
 if flags['enable_feed3_rank_adjustment'] or flags['enable_feed3_new_candidates']:raise SystemExit('This append-only adapter does not implement ranking or candidate creation.')
 if any(flags[k] for k in ('enable_owner_front_office_signals','enable_best_game_discovery_signals','enable_historical_fan_interpretation')):raise SystemExit('A reference-only signal flag was enabled; this adapter intentionally holds those signals back.')
 before={str(p):sha(p) for p in OFFICIAL};cands=rr(CAND);official=rr(EVID);overlay=rr(OVER);dictionary=rr(DICT);recommendation=RECOMMEND.read_text(encoding='utf-8')
 if len(list(SP.glob('*.csv')))!=25 or len(cands)!=15 or len(overlay)!=15 or not recommendation:raise SystemExit('Input gate failed.')
 dict_types={r['signal_name'] for r in dictionary};required={'playoff_odds','elo_team_strength','baseruns_context','team_war','injury_context','fan_interest','financial_context','team_age','team_baserunning','player_baserunning','ubr','rwar'}
 if not required<=dict_types:raise SystemExit('Signal dictionary is missing an allowed signal definition.')
 over={r['candidate_id']:r for r in overlay};records=[]
 for c in cands:
  types,entity_type,entity=evidence_plan(c['candidate_id']);ov=over[c['candidate_id']]
  for seq,etype in enumerate(types,1):
   tables,files,default_entity,metric,allowed,disallowed,caution=TYPE_CONFIG[etype]
   eid=f"{c['candidate_id']}__feed3__{etype}__{seq:02d}"
   records.append({'evidence_id':eid,'candidate_id':c['candidate_id'],'candidate_title':c['headline_factual'],'evidence_type':etype,
    'source_feed':'Feed 3 / StatsPlus','source_table':tables,'source_file':'|'.join('csv/statsplus/current/'+x for x in files.split('|')),
    'entity_type':entity_type or default_entity,'entity_name':entity,'metric_name':metric,'metric_value':ov['statsplus_evidence_summary'],
    'interpretation':ov['suggests_new_angle'],'allowed_use':allowed,'disallowed_use':disallowed,
    'authority_caution':'StatsPlus does not prove games, scores, standings, or official current state. '+caution,
    'ranking_effect':'none','confidence':c['confidence'],'notes':f"Feature-flagged append only; priority preview={ov['priority_effect_later']}; official candidate unchanged."})
 astem=ENRICH/f'feed3_evidence_append_{SLUG}';dumpcsv(astem.with_suffix('.csv'),records);dumpjson(astem.with_suffix('.json'),{'feature_flags':flags,'record_count':len(records),'candidate_count':len({r['candidate_id'] for r in records}),'records':records})
 astem.with_suffix('.md').write_text('# Feed 3 Evidence Append\n\n> StatsPlus enhances. It does not replace. StatsPlus annotates before it ranks.\n\n'+mdtable(['Candidate','Type','Entity','Metric','Value','Ranking'],[[r['candidate_title'],r['evidence_type'],r['entity_name'],r['metric_name'],r['metric_value'],r['ranking_effect']] for r in records])+'\n',encoding='utf-8')

 fields=['record_origin','evidence_id','candidate_id','candidate_title','evidence_type','source_feed','source_table','source_file','entity_type','entity_name','metric_name','metric_value','interpretation','allowed_use','disallowed_use','authority_caution','ranking_effect','confidence','notes','original_evidence_json']
 combined=[]
 titles={r['candidate_id']:r['headline_factual'] for r in cands}
 for r in official:
  combined.append({'record_origin':'official','evidence_id':r['evidence_id'],'candidate_id':r['candidate_id'],'candidate_title':titles.get(r['candidate_id'],''),'evidence_type':r['evidence_class'],'source_feed':'Official existing evidence','source_table':'','source_file':r['source_file'],'entity_type':'official','entity_name':'','metric_name':r['metric'],'metric_value':r['value'],'interpretation':r['comparison'],'allowed_use':'Existing governed evidence use.','disallowed_use':'No authority change in this preview.','authority_caution':r['notes'],'ranking_effect':'existing_only','confidence':'','notes':'Official evidence copied into preview; source file untouched.','original_evidence_json':json.dumps(r,ensure_ascii=False)})
 for r in records:combined.append({'record_origin':'feed3',**r,'original_evidence_json':''})
 cstem=ENRICH/f'story_evidence_with_feed3_preview_{SLUG}';dumpcsv(cstem.with_suffix('.csv'),combined,fields);dumpjson(cstem.with_suffix('.json'),{'official_evidence_count':len(official),'feed3_evidence_count':len(records),'combined_count':len(combined),'official_file_replaced':False,'records':combined})
 cstem.with_suffix('.md').write_text(f'''# Story Evidence with Feed 3 — Preview Only

- Official evidence records: **{len(official)}**.
- Appended Feed 3 records: **{len(records)}**.
- Combined preview records: **{len(combined)}**.
- Official evidence file replaced: **No**.
- Ranking effects: **None**.

{mdtable(['Origin','Candidate','Evidence type','Metric','Ranking'],[[r['record_origin'],r['candidate_title'],r['evidence_type'],r['metric_name'],r['ranking_effect']] for r in combined])}
''',encoding='utf-8')
 after={str(p):sha(p) for p in OFFICIAL};untouched=before==after
 used=sorted({r['evidence_type'] for r in records});held=sorted(set(TYPE_CONFIG)-set(used));reference=['owner/front-office traits','best-game discovery tables','historical fan interpretation','dual playoff-odds views as separate ranking signals']
 report={'feature_flags':flags,'candidates_enriched':len({r['candidate_id'] for r in records}),'feed3_evidence_records_created':len(records),'evidence_types_used':used,
  'evidence_types_held_back':held,'reference_only_signals':reference,'ranking_changed':False,'new_candidates_created':False,
  'official_story_files_modified':not untouched,'official_story_artifacts_untouched':untouched,
  'recommended_next_task':'Add this adapter behind an explicit preview flag in the master runner, validate on a later newsroom date, and retain ranking adjustment/new-candidate flags as false.'}
 rstem=ENRICH/f'feed3_evidence_adapter_report_{SLUG}';dumpjson(rstem.with_suffix('.json'),report)
 rstem.with_suffix('.md').write_text('# Feed 3 Evidence Adapter Report\n\n'+mdtable(['Feature flag','Value'],[[k,str(v).lower()] for k,v in flags.items()])+f'''\n\n- Candidates enriched: **{report['candidates_enriched']}**.
- Feed 3 evidence records: **{len(records)}**.
- Evidence types used: {', '.join(used)}.
- Evidence types held back for lack of candidate fit: {', '.join(held) or 'none'}.
- Reference-only signals: {', '.join(reference)}.
- Ranking changed: **No**.
- New candidates created: **No**.
- Official story artifacts untouched: **{'Yes' if untouched else 'No'}**.

Recommended next task: {report['recommended_next_task']}
''',encoding='utf-8')
 if not untouched:raise SystemExit('Official story artifact hash changed.')
 print(json.dumps({'adapter_created':True,'feed3_evidence_records_created':len(records),'candidates_enriched':report['candidates_enriched'],'rankings_changed':False,'new_candidates_created':False,'official_story_artifacts_untouched':untouched,'evidence_types_used':used,'held_types':held},indent=2))

if __name__=='__main__':main()
