#!/usr/bin/env python3
"""Append typed Feed 3 evidence without mutating official story artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[2];CONTROL=REPO/'csv'/'out'/'control'

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
 p.add_argument('--newsroom-date',default='1981-07-20',help='Newsroom date in YYYY-MM-DD form.')
 p.add_argument('--as-of-date',default='1981-07-19',help='Completed-data cutoff in YYYY-MM-DD form.')
 p.add_argument('--feature-flags-path',type=Path)
 p.add_argument('--candidates-path',type=Path)
 p.add_argument('--evidence-path',type=Path)
 p.add_argument('--overlay-path',type=Path)
 p.add_argument('--statsplus-current-folder',type=Path,default=REPO/'csv'/'statsplus'/'current')
 p.add_argument('--output-folder',type=Path,default=REPO/'csv'/'out'/'story'/'enrichment')
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

def evidence_plan(candidate):
 cid=candidate['candidate_id'];signal=candidate.get('signal_type','')
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
 if signal=='division_race':return ['statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war'],'race',candidate.get('subject_name','division race')
 if signal=='scheduled_matchup_stakes':return ['statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war','statsplus_team_baserunning','statsplus_ubr'],'matchup',candidate.get('subject_name','matchup')
 if signal=='ace_value_leader':return ['statsplus_pitcher_rwar'],'player',candidate.get('subject_name','pitcher')
 if signal=='position_player_value_leader':return ['statsplus_playoff_odds','statsplus_team_war'],'context',candidate.get('subject_name','position player')
 if signal=='weekly_rise':return ['statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war','statsplus_injury_context'],'team',candidate.get('subject_name','team')
 if signal=='weekly_fall':return ['statsplus_injury_context'],'team',candidate.get('subject_name','team')
 if signal=='pythag_record_gap':return ['statsplus_playoff_odds','statsplus_baseruns','statsplus_elo'],'team',candidate.get('subject_name','team')
 if signal=='payroll_record_pressure':return ['statsplus_financial_context','statsplus_playoff_odds','statsplus_elo','statsplus_baseruns','statsplus_team_war','statsplus_injury_context','statsplus_fan_interest_context'],'context',candidate.get('subject_name','team')
 if signal=='attendance_interest':return ['statsplus_fan_interest_context'],'context',candidate.get('subject_name','team')
 if signal=='prior_qualifier_echo':return ['statsplus_playoff_odds','statsplus_elo','statsplus_team_war'],'context',candidate.get('subject_name','team')
 raise ValueError(f'No Feed 3 evidence plan for signal={signal!r}, candidate={cid}')

def team_codes(candidate,name_to_abbr,player_view):
 cid=candidate['candidate_id']
 if 'national-baseball-conference-central-division' in cid:return ['DAL','DET']
 if 'national-baseball-conference-western-division' in cid:return ['SF','PHO']
 if 'american-baseball-conference-western-division' in cid:return ['LV','SEA']
 if 'american-baseball-conference-eastern-division' in cid:return ['BOS','NY']
 if 'american-baseball-conference-central-division' in cid:return ['HOU','CIN']
 if 'scheduled_matchup_stakes' in cid:return ['LV','DEN']
 if 'position_player_value_leader' in cid or 'prior_qualifier_echo' in cid:return ['CHA']
 if 'weekly_rise' in cid or 'pythag_record_gap__3' in cid:return ['ATL']
 if 'weekly_fall' in cid:return ['PHI']
 if 'pythag_record_gap__24' in cid:return ['SEA']
 if 'payroll_record_pressure' in cid:return ['TB']
 if 'attendance_interest' in cid:return ['DET']
 text=' | '.join(candidate.get(k,'') for k in ('subject_name','related_subjects','headline_factual'))
 found=[abbr for name,abbr in name_to_abbr.items() if name.lower() in text.lower()]
 if found:return list(dict.fromkeys(found))
 player=candidate.get('subject_name','')
 player_teams=[team for (name,team) in player_view if name.lower()==player.lower()]
 return player_teams[:1]

def exact_metric_value(etype,candidate,team_view,player_view,name_to_abbr):
 codes=team_codes(candidate,name_to_abbr,player_view);rows=[team_view[x] for x in codes if x in team_view]
 fields={
  'statsplus_playoff_odds':[('Div%', 'division_odds_pct'),('PO%', 'playoff_odds_pct'),('AvgW','average_projected_wins'),('rSoS','remaining_sos')],
  'statsplus_elo':[('ELO','elo'),('seasonΔ','elo_season_change'),('30dΔ','elo_30day_change'),('7dΔ','elo_7day_change')],
  'statsplus_baseruns':[('pWΔ','baseruns_pythag_win_delta'),('xWΔ','baseruns_expected_win_delta'),('xW','baseruns_xwins'),('RD','run_differential'),('xRD','expected_run_differential')],
  'statsplus_team_war':[('BatterWAR','batter_war'),('PitcherWAR','pitcher_war'),('TotalWAR','total_war')],
  'statsplus_injury_context':[('injuries','injury_count'),('DL days','dl_days'),('$ on DL','salary_on_dl')],
  'statsplus_fan_interest_context':[('interest','fan_interest'),('30dΔ','fan_interest_30day_change'),('avg attendance','average_attendance'),('% full','capacity_pct_full')],
  'statsplus_financial_context':[('payroll','payroll'),('budget','budget'),('revenue','total_revenue'),('cash','cash_for_trades')],
  'statsplus_team_age_context':[('ML age','ml_age'),('P age','ml_pitcher_age'),('B age','ml_batter_age')],
  'statsplus_team_baserunning':[('SB','team_sb'),('CS','team_cs'),('wSB','team_wsb')],
  'statsplus_ubr':[('UBR','team_ubr')],
 }
 if etype=='statsplus_pitcher_rwar':
  player=candidate.get('subject_name','');matches=[(k,v) for k,v in player_view.items() if k[0].lower()==player.lower()]
  if not matches:raise ValueError(f'No player enrichment row for {player}')
  (name,team),p=matches[0]
  return f"{name}/{team}: rWAR={p['rwar']}, WAR={p['pitching_war']}, ERA={p['era']}, FIP={p['fip']}, xFIP={p['xfip']}"
 if not rows:raise ValueError(f"No team enrichment row for candidate {candidate['candidate_id']} and evidence {etype}")
 selected=fields[etype]
 return '; '.join(f"{r['team_abbr']}: "+', '.join(f"{label}={r[key]}" for label,key in selected) for r in rows)

def main():
 a=parse_args();news_slug=a.newsroom_date.replace('-','_');slug=f'{news_slug}_asof_{a.as_of_date}';season=a.newsroom_date[:4];enrich=a.output_folder.resolve();sp=a.statsplus_current_folder.resolve()
 cand=(a.candidates_path or REPO/f'csv/out/story/candidates/story_candidates_{slug}.csv').resolve();evid=(a.evidence_path or REPO/f'csv/out/story/candidates/story_evidence_{slug}.csv').resolve();over_path=(a.overlay_path or enrich/f'story_candidates_statsplus_overlay_{slug}.csv').resolve()
 dictionary_path=(CONTROL/f'statsplus_story_signal_dictionary_{season}_asof_{a.as_of_date}.csv').resolve();recommend_path=(enrich/f'statsplus_story_integration_recommendation_{slug}.md').resolve()
 team_path=enrich/f'statsplus_team_enrichment_{slug}.csv';player_path=enrich/f'statsplus_player_enrichment_{slug}.csv'
 official=[cand,evid,REPO/f'csv/out/story/menus/story_menu_{slug}.csv',REPO/f'csv/out/story/editorial/observer_editorial_board_{slug}.csv',REPO/f'csv/out/story/editorial/observer_story_slate_{slug}.md',REPO/f'csv/out/story/production/eb_production_package_{slug}.md',REPO/f'csv/out/story/scripts/baseball_observer_segment_{slug}.md']
 required=[cand,evid,over_path,dictionary_path,recommend_path,team_path,player_path]
 missing=[str(p) for p in required if not p.exists()]
 if missing:raise SystemExit('Required Feed 3 preview inputs do not exist for the requested date:\n- '+'\n- '.join(missing))
 missing_official=[str(p) for p in official if not p.exists()]
 if missing_official:raise SystemExit('Required protected story artifacts do not exist for the requested date:\n- '+'\n- '.join(missing_official))
 configured={}
 if a.feature_flags_path:
  if not a.feature_flags_path.exists():raise SystemExit(f'Feature flag file does not exist: {a.feature_flags_path}')
  configured=json.loads(a.feature_flags_path.read_text(encoding='utf-8'))
 flags={'enable_feed3_evidence':configured.get('enable_feed3_evidence_preview',not a.disable_feed3_evidence),'enable_feed3_rank_adjustment':configured.get('enable_feed3_rank_adjustment',a.enable_feed3_rank_adjustment),
  'enable_feed3_new_candidates':a.enable_feed3_new_candidates,'enable_owner_front_office_signals':a.enable_owner_front_office_signals,
  'enable_best_game_discovery_signals':a.enable_best_game_discovery_signals,'enable_historical_fan_interpretation':a.enable_historical_fan_interpretation}
 for key in ('enable_feed3_new_candidates','enable_owner_front_office_signals','enable_best_game_discovery_signals','enable_historical_fan_interpretation'):
  flags[key]=configured.get(key,flags[key])
 if not flags['enable_feed3_evidence']:raise SystemExit('Feed 3 evidence is disabled; no output generated.')
 if flags['enable_feed3_rank_adjustment'] or flags['enable_feed3_new_candidates']:raise SystemExit('This append-only adapter does not implement ranking or candidate creation.')
 if any(flags[k] for k in ('enable_owner_front_office_signals','enable_best_game_discovery_signals','enable_historical_fan_interpretation')):raise SystemExit('A reference-only signal flag was enabled; this adapter intentionally holds those signals back.')
 before={str(p):sha(p) for p in official};cands=rr(cand);official_evidence=rr(evid);overlay=rr(over_path);dictionary=rr(dictionary_path);recommendation=recommend_path.read_text(encoding='utf-8')
 team_view={r['team_abbr']:r for r in rr(team_path)};player_view={(r['player_name'],r['team_abbr']):r for r in rr(player_path)}
 rawteams=rr(REPO/'csv/ootp_csv/teams.csv');name_to_abbr={f"{r['name']} {r['nickname']}":r['abbr'] for r in rawteams if r.get('league_id')=='200' and r.get('level')=='1' and r.get('allstar_team')=='0'}
 if not list(sp.glob('*.csv')) or len(cands)!=len(overlay) or not recommendation:raise SystemExit('Feed 3 input population gate failed.')
 dict_types={r['signal_name'] for r in dictionary};required={'playoff_odds','elo_team_strength','baseruns_context','team_war','injury_context','fan_interest','financial_context','team_age','team_baserunning','player_baserunning','ubr','rwar'}
 if not required<=dict_types:raise SystemExit('Signal dictionary is missing an allowed signal definition.')
 over={r['candidate_id']:r for r in overlay};records=[]
 for c in cands:
  types,entity_type,entity=evidence_plan(c);ov=over[c['candidate_id']]
  for seq,etype in enumerate(types,1):
   tables,files,default_entity,metric,allowed,disallowed,caution=TYPE_CONFIG[etype]
   eid=f"{c['candidate_id']}__feed3__{etype}__{seq:02d}"
   records.append({'evidence_id':eid,'candidate_id':c['candidate_id'],'candidate_title':c['headline_factual'],'evidence_type':etype,
    'source_feed':'Feed 3 / StatsPlus','source_table':tables,'source_file':'|'.join(str((sp/x).relative_to(REPO)).replace('\\','/') if (sp/x).is_relative_to(REPO) else str(sp/x) for x in files.split('|')),
    'entity_type':entity_type or default_entity,'entity_name':entity,'metric_name':metric,'metric_value':exact_metric_value(etype,c,team_view,player_view,name_to_abbr),
    'interpretation':allowed+' Candidate-level angle: '+ov['suggests_new_angle'],'allowed_use':allowed,'disallowed_use':disallowed,
    'authority_caution':'StatsPlus does not prove games, scores, standings, or official current state. '+caution,
    'ranking_effect':'none','confidence':c['confidence'],'notes':f"Feature-flagged append only; priority preview={ov['priority_effect_later']}; official candidate unchanged."})
 astem=enrich/f'feed3_evidence_append_{slug}';dumpcsv(astem.with_suffix('.csv'),records);dumpjson(astem.with_suffix('.json'),{'newsroom_date':a.newsroom_date,'as_of_date':a.as_of_date,'feature_flags':flags,'record_count':len(records),'candidate_count':len({r['candidate_id'] for r in records}),'records':records})
 astem.with_suffix('.md').write_text('# Feed 3 Evidence Append\n\n> StatsPlus enhances. It does not replace. StatsPlus annotates before it ranks.\n\n'+mdtable(['Candidate','Type','Entity','Metric','Value','Ranking'],[[r['candidate_title'],r['evidence_type'],r['entity_name'],r['metric_name'],r['metric_value'],r['ranking_effect']] for r in records])+'\n',encoding='utf-8')

 fields=['record_origin','evidence_id','candidate_id','candidate_title','evidence_type','source_feed','source_table','source_file','entity_type','entity_name','metric_name','metric_value','interpretation','allowed_use','disallowed_use','authority_caution','ranking_effect','confidence','notes','original_evidence_json']
 combined=[]
 titles={r['candidate_id']:r['headline_factual'] for r in cands}
 for r in official_evidence:
  combined.append({'record_origin':'official','evidence_id':r['evidence_id'],'candidate_id':r['candidate_id'],'candidate_title':titles.get(r['candidate_id'],''),'evidence_type':r['evidence_class'],'source_feed':'Official existing evidence','source_table':'','source_file':r['source_file'],'entity_type':'official','entity_name':'','metric_name':r['metric'],'metric_value':r['value'],'interpretation':r['comparison'],'allowed_use':'Existing governed evidence use.','disallowed_use':'No authority change in this preview.','authority_caution':r['notes'],'ranking_effect':'existing_only','confidence':'','notes':'Official evidence copied into preview; source file untouched.','original_evidence_json':json.dumps(r,ensure_ascii=False)})
 for r in records:combined.append({'record_origin':'feed3',**r,'original_evidence_json':''})
 cstem=enrich/f'story_evidence_with_feed3_preview_{slug}';dumpcsv(cstem.with_suffix('.csv'),combined,fields);dumpjson(cstem.with_suffix('.json'),{'newsroom_date':a.newsroom_date,'as_of_date':a.as_of_date,'official_evidence_count':len(official_evidence),'feed3_evidence_count':len(records),'combined_count':len(combined),'official_file_replaced':False,'records':combined})
 cstem.with_suffix('.md').write_text(f'''# Story Evidence with Feed 3 — Preview Only

- Official evidence records: **{len(official_evidence)}**.
- Appended Feed 3 records: **{len(records)}**.
- Combined preview records: **{len(combined)}**.
- Official evidence file replaced: **No**.
- Ranking effects: **None**.

{mdtable(['Origin','Candidate','Evidence type','Metric','Ranking'],[[r['record_origin'],r['candidate_title'],r['evidence_type'],r['metric_name'],r['ranking_effect']] for r in combined])}
''',encoding='utf-8')
 after={str(p):sha(p) for p in official};untouched=before==after
 used=sorted({r['evidence_type'] for r in records});held=sorted(set(TYPE_CONFIG)-set(used));reference=['owner/front-office traits','best-game discovery tables','historical fan interpretation','dual playoff-odds views as separate ranking signals']
 report={'feature_flags':flags,'candidates_enriched':len({r['candidate_id'] for r in records}),'feed3_evidence_records_created':len(records),'evidence_types_used':used,
  'evidence_types_held_back':held,'reference_only_signals':reference,'ranking_changed':False,'new_candidates_created':False,
  'official_story_files_modified':not untouched,'official_story_artifacts_untouched':untouched,
  'recommended_next_task':'Add this adapter behind an explicit preview flag in the master runner, validate on a later newsroom date, and retain ranking adjustment/new-candidate flags as false.'}
 report.update({'newsroom_date':a.newsroom_date,'as_of_date':a.as_of_date,'resolved_paths':{'candidates':str(cand),'evidence':str(evid),'overlay':str(over_path),'statsplus_current':str(sp),'output_folder':str(enrich)}})
 rstem=enrich/f'feed3_evidence_adapter_report_{slug}';dumpjson(rstem.with_suffix('.json'),report)
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
