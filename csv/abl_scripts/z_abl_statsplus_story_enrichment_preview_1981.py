#!/usr/bin/env python3
"""Build a non-destructive StatsPlus enrichment preview for the July 20 board."""

import csv,json,re
from pathlib import Path

REPO=Path(__file__).resolve().parents[2];SP=REPO/'csv'/'statsplus'/'current';CONTROL=REPO/'csv'/'out'/'control'
ENRICH=REPO/'csv'/'out'/'story'/'enrichment';INV=REPO/'csv'/'out'/'story'/'investigations';SLUG='1981_07_20_asof_1981-07-19';CSLUG='1981_asof_1981-07-19'
def rr(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def tab(n):return rr(next(SP.glob(f'{n:02d}_*.csv')))
def dumpcsv(p,r):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def dumpjson(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def clean(x):return str(x).replace('|','\\|').replace('\n',' ')
def mdtable(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|']+['| '+' | '.join(clean(v) for v in r)+' |' for r in rows])
def norm(x):return ' '.join(re.findall(r'[a-z0-9]+',str(x).lower()))
def idx(rows,key='Team'):return {r[key]:r for r in rows if r.get(key)}
def fmt(v,default=''):return v if v not in (None,'') else default

SIGNALS=[
 ('playoff_odds','7|8','07_statsplus_Playoff_Odds_Div.csv|08_statsplus_Playoff_Odds_League.csv','team/model','Team|Div %|PO %|AvgW|rSoS','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Describe modeled race pressure and probability.','Do not replace standings or present probability as destiny.','high','StatsPlus model output; raw OOTP governs records and standings.'),
 ('elo_team_strength','10','10_statsplus_Elo.csv','team/model','Team|Elo|Seas.+/-|30d+/-|7d+/-','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Compare modeled strength and recent movement.','Do not call ELO an official record or guaranteed future result.','high','Use “ELO rates” or “model,” not “proves.”'),
 ('baseruns_context','9','09_statsplus_Base_Runs.csv','team/model','Team|RS|RA|xRS|xRA|pWΔ|xWΔ','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Identify over/under-performance and expected-run context.','Do not label residuals luck or promise regression.','high','Expected-performance model; game results remain raw proof.'),
 ('team_war','11','11_statsplus_Team_WAR.csv','team/model','Team|BatterWAR|PitcherWAR|TotalWAR','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Add team-value composition and strength context.','Do not equate WAR rank with standings or postseason outcome.','high','StatsPlus WAR definition must remain labeled.'),
 ('injury_context','12','12_statsplus_injury_sumary.csv','team','Team|Count|DL Days|$ on DL','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Describe aggregate injury burden.','Do not attribute a result to injury without player/game linkage.','high','Aggregate context only; avoid unsupported causation.'),
 ('fan_interest','4|5','04_statsplus_Historical_Fan_Interest.csv|05_statsplus_Fan_Data.csv','team/historical','Team|Fan Interest|30-dayFI Δ|Avg.Att.|% Full|1972-1981','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Add measured fan-interest, attendance scale, and historical trend.','Do not infer emotion, motives, or clubhouse effects.','high','Measured context is not sentiment testimony.'),
 ('financial_context','3','03_statsplus_Finantials.csv','team','Team|Payroll|Budget|Revenue|Cash(for Trades)','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Compare resources and results.','Do not infer blame, intent, or future transactions.','high','Financial context creates questions, not conclusions.'),
 ('owner_front_office','1|2','01_abl_transactions_personnel_-_owner.csv|02_abl_transactions_personnel_-_all_coaches.csv','staff/front office','TM|Name|Job|Type|POS|NEG|Patience|Spending|Involvement|Priority','LIMITED_STATSPLUS_CONTEXT','Add named personnel and structured current traits.','Do not infer motives/tendencies; owner mood/objectives/goals and staff IDs are unavailable.','medium','Manual reference only until identity/tendency rules expand.'),
 ('team_age','24','24_statsplus_Team_Age.csv','team/organization','Team|ML Age|ML P Age|ML B Age|minor-level ages','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Add organization-age texture.','Do not infer development success or decline from age alone.','medium','Context, not causal proof.'),
 ('best_batting_games','25','25_statsplus_Best_Batting_Game.csv','game/player','Name|Team|Opp|Date|Score|GS','STATSP_RANKING_RAW_CROSSCHECK','Discover ranked standout batting games for investigation.','Do not use as official game proof without raw verification.','medium','StatsPlus ranking; crosscheck raw game data before publication.'),
 ('best_pitching_games','26','26_statsplus_Best_Pitching_Game.csv','game/player','Name|Team|Opp|Date|Score|GS','STATSP_RANKING_RAW_CROSSCHECK','Discover ranked standout pitching games for investigation.','Do not use as official game proof without raw verification.','medium','StatsPlus ranking; crosscheck raw game data before publication.'),
 ('team_baserunning','19','19_statsplus_Team_Baserunning.csv','team','Team|SB|CS|SB %|wSB|UBR','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Add team baserunning-value context.','Do not silently substitute for sortable metrics or explain wins causally.','high','Use StatsPlus definitions and labels.'),
 ('player_baserunning','22','22_statsplus_Player_Baserunning.csv','player','Name|Team|WSB|SB|CS|SB%','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Add individual baserunning-value context.','Do not override validated raw events or other definitions.','medium','Use StatsPlus definition and player identity.'),
 ('ubr','19','19_statsplus_Team_Baserunning.csv','team','Team|UBR','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Use labeled team UBR as enrichment.','Do not treat UBR as official record or unlabeled replacement metric.','high','Newly accepted additive field; model context only.'),
 ('rwar','21','21_statsplus_Player_Pitching.csv','player','Name|Team|rWAR|WAR|ERA|FIP|xFIP','STATSPLUS_SPECIFIC_AFTER_PROMOTION','Use labeled rWAR as an alternate pitcher-value lens.','Do not override sortable WAR or raw proof.','high','Always distinguish rWAR from WAR.'),
]

def main():
 ENRICH.mkdir(parents=True,exist_ok=True);INV.mkdir(parents=True,exist_ok=True)
 sigrows=[{'signal_name':a,'source_table_number':b,'source_file':c,'grain':d,'key_fields':e,'authority_classification':f,'allowed_use':g,'disallowed_use':h,'story_value':i,'caution_language':j} for a,b,c,d,e,f,g,h,i,j in SIGNALS]
 sstem=CONTROL/f'statsplus_story_signal_dictionary_{CSLUG}';dumpcsv(sstem.with_suffix('.csv'),sigrows);dumpjson(sstem.with_suffix('.json'),{'season':1981,'as_of_date':'1981-07-19','doctrine':'StatsPlus enhances. It does not replace.','signals':sigrows})
 sstem.with_suffix('.md').write_text('# StatsPlus Story Signal Dictionary\n\n> **StatsPlus enhances. It does not replace.**\n\n'+mdtable(['Signal','Tables','Grain','Authority','Allowed use','Caution'],[[r['signal_name'],r['source_table_number'],r['grain'],r['authority_classification'],r['allowed_use'],r['caution_language']] for r in sigrows])+'\n',encoding='utf-8')

 teams=rr(REPO/'csv'/'ootp_csv'/'teams.csv'); majors=[t for t in teams if t['league_id']=='200' and t['level']=='1' and t.get('allstar_team')=='0'];names={t['abbr']:f"{t['name']} {t['nickname']}" for t in majors};city={t['name']:t['abbr'] for t in majors}
 odds=idx(tab(7));br=idx(tab(9));elo=idx(tab(10));inj=idx(tab(12));fan=idx(tab(5));hist=idx(tab(4));fin=idx(tab(3));run=idx(tab(19));owner=idx(tab(1),'TM')
 war={next((a for a in names if r['Team'].endswith(a)),r['Team']):r for r in tab(11)}
 age={city.get(r['Team'],r['Team']):r for r in tab(24)}
 bestbat={a:[] for a in names};bestpit={a:[] for a in names}
 for r in tab(25):
  if r['Team'] in bestbat:bestbat[r['Team']].append(f"{r['Name']} vs {r['Opp']} {r['Date']} GS={r['GS']}")
 for r in tab(26):
  if r['Team'] in bestpit:bestpit[r['Team']].append(f"{r['Name']} vs {r['Opp']} {r['Date']} GS={r['GS']}")
 teamrows=[]
 for ab in sorted(names):
  o,b,e,w,i,f,h,fi,ru,ow,ag=[x.get(ab,{}) for x in (odds,br,elo,war,inj,fan,hist,fin,run,owner,age)]
  teamrows.append({'team_abbr':ab,'team_name':names[ab],'feed3_record_crosscheck':f"{o.get('W','')}-{o.get('L','')}",
   'division_odds_pct':o.get('Div %',''),'playoff_odds_pct':o.get('PO %',''),'average_projected_wins':o.get('AvgW',''),'remaining_sos':o.get('rSoS',''),
   'elo':e.get('Elo',''),'elo_season_change':e.get('Seas.+/-',''),'elo_30day_change':e.get('30d+/-',''),'elo_7day_change':e.get('7d+/-',''),
   'baseruns_pythag_win_delta':b.get('pWΔ',''),'baseruns_expected_win_delta':b.get('xWΔ',''),'baseruns_xwins':b.get('xW',''),'run_differential':b.get('RD',''),'expected_run_differential':b.get('xRD',''),
   'batter_war':w.get('BatterWAR',''),'pitcher_war':w.get('PitcherWAR',''),'total_war':w.get('TotalWAR',''),
   'injury_count':i.get('Count',''),'dl_days':i.get('DL Days',''),'salary_on_dl':i.get('$ on DL',''),
   'fan_interest':f.get('Fan Interest',''),'fan_interest_30day_change':f.get('30-dayFI Δ',''),'average_attendance':f.get('Avg.Att.',''),'capacity_pct_full':f.get('% Full',''),'historical_fan_interest_1980':h.get("'80",''),'historical_fan_interest_1981':h.get("'81",''),
   'payroll':fi.get('Payroll',''),'budget':fi.get('Budget',''),'total_revenue':fi.get('TotalRevenue',''),'cash_for_trades':fi.get('Cash(for Trades)',''),
   'owner_name':ow.get('Name',''),'owner_type':ow.get('Type',''),'owner_patience':ow.get('Patience',''),'owner_spending':ow.get('Spending',''),'owner_involvement':ow.get('Involvement',''),'owner_priority':ow.get('Priority',''),
   'ml_age':ag.get('ML Age',''),'ml_pitcher_age':ag.get('ML P Age',''),'ml_batter_age':ag.get('ML B Age',''),
   'team_wsb':ru.get('wSB',''),'team_ubr':ru.get('UBR',''),'team_sb':ru.get('SB',''),'team_cs':ru.get('CS',''),
   'best_batting_game_notes':' | '.join(bestbat[ab]),'best_pitching_game_notes':' | '.join(bestpit[ab]),
   'authority_caution':'Feed 3 model/context only; raw OOTP governs games, records, scores, and standings.'})
 tstem=ENRICH/f'statsplus_team_enrichment_{SLUG}';dumpcsv(tstem.with_suffix('.csv'),teamrows);dumpjson(tstem.with_suffix('.json'),{'newsroom_date':'1981-07-20','as_of_date':'1981-07-19','team_count':len(teamrows),'teams':teamrows})
 tstem.with_suffix('.md').write_text('# StatsPlus Team Enrichment View\n\nCurated overlay only; no source or story artifact is modified.\n\n'+mdtable(['Team','Div odds','PO odds','ELO','xWΔ','Total WAR','Injuries','Fan interest','UBR'],[[r['team_name'],r['division_odds_pct'],r['playoff_odds_pct'],r['elo'],r['baseruns_expected_win_delta'],r['total_war'],r['injury_count'],r['fan_interest'],r['team_ubr']] for r in teamrows])+'\n',encoding='utf-8')
 team={r['team_abbr']:r for r in teamrows}

 bat=tab(20);pit=tab(21);prun=tab(22);pbestb=tab(25);pbestp=tab(26);keys=set()
 for rows_ in (bat,pit,prun,pbestb,pbestp):
  for r in rows_:keys.add((r['Name'],r['Team']))
 bi={(r['Name'],r['Team']):r for r in bat};pi={(r['Name'],r['Team']):r for r in pit};ri={(r['Name'],r['Team']):r for r in prun}
 bb={};bp={}
 for r in pbestb:bb.setdefault((r['Name'],r['Team']),[]).append(f"vs {r['Opp']} {r['Date']} score={r['Score']} GS={r['GS']}")
 for r in pbestp:bp.setdefault((r['Name'],r['Team']),[]).append(f"vs {r['Opp']} {r['Date']} score={r['Score']} GS={r['GS']}")
 prows=[]
 for key in sorted(keys):
  b,p,u=bi.get(key,{}),pi.get(key,{}),ri.get(key,{})
  prows.append({'player_name':key[0],'team_abbr':key[1],'position_or_role':b.get('Pos',p.get('Role',u.get('Pos',''))),
   'batting_war':b.get('WAR',''),'wrc_plus':b.get('WRC+',''),'woba':b.get('WOBA',''),'ops':b.get('OPS   &nbsp;',''),'hr':b.get('HR',''),
   'pitching_war':p.get('WAR',''),'rwar':p.get('rWAR',''),'era':p.get('ERA',''),'fip':p.get('FIP',''),'xfip':p.get('XFIP',''),'innings':p.get('IP   &nbsp;',''),
   'player_wsb':u.get('WSB   &nbsp;',''),'sb':u.get('SB',b.get('SB','')),'cs':u.get('CS',''),'sb_pct':u.get('SB%',''),
   'best_batting_game_notes':' | '.join(bb.get(key,[])),'best_pitching_game_notes':' | '.join(bp.get(key,[])),
   'authority_caution':'StatsPlus-specific enrichment; rWAR does not override WAR and best-game rankings require raw verification.'})
 pstem=ENRICH/f'statsplus_player_enrichment_{SLUG}';dumpcsv(pstem.with_suffix('.csv'),prows);dumpjson(pstem.with_suffix('.json'),{'newsroom_date':'1981-07-20','as_of_date':'1981-07-19','player_rows':len(prows),'players':prows})
 highlights=[r for r in prows if r['player_name'] in ('Jose Coronado','Manny Flores') or r['best_batting_game_notes'] or r['best_pitching_game_notes']]
 pstem.with_suffix('.md').write_text('# StatsPlus Player Enrichment View\n\nCurated Feed 3 view; rankings and modeled value do not replace raw proof.\n\n'+mdtable(['Player','Team','Role','WAR','rWAR','wRC+','wSB','Best-game note'],[[r['player_name'],r['team_abbr'],r['position_or_role'],r['pitching_war'] or r['batting_war'],r['rwar'],r['wrc_plus'],r['player_wsb'],r['best_batting_game_notes'] or r['best_pitching_game_notes']] for r in highlights])+'\n',encoding='utf-8')

 candidates=rr(REPO/f'csv/out/story/candidates/story_candidates_{SLUG}.csv')
 def T(a):return team[a]
 overlay_data={
 'national-baseball-conference-central-division':('DAL/DET playoff odds, ELO, WAR, fan/injury context',f"DAL Div/PO {T('DAL')['division_odds_pct']}/{T('DAL')['playoff_odds_pct']} vs DET {T('DET')['division_odds_pct']}/{T('DET')['playoff_odds_pct']}; ELO {T('DAL')['elo']} vs {T('DET')['elo']}; WAR {T('DAL')['total_war']} vs {T('DET')['total_war']}",'yes','Detroit slightly leads ELO despite trailing in standings.','Detroit fan interest 100 is measured scale, not emotion.','Model-vs-standings tension inside the two-game race.','reinforce_lead','7|10|11|12|4|5','Div %|PO %|Elo|TotalWAR|injuries|Fan Interest'),
 'national-baseball-conference-western-division':('SF/PHO odds, ELO, BaseRuns and WAR',f"SF Div/PO {T('SF')['division_odds_pct']}/{T('SF')['playoff_odds_pct']} vs PHO {T('PHO')['division_odds_pct']}/{T('PHO')['playoff_odds_pct']}; PHO ELO {T('PHO')['elo']} vs SF {T('SF')['elo']}; PHO xWΔ {T('PHO')['baseruns_expected_win_delta']}",'yes','Strongly: the two-game margin masks a wide model gap; Phoenix has higher ELO but much lower odds/WAR.','Do not predict Phoenix collapse or regression.','Investigate why models separate two clubs only two games apart.','consider_raise_secondary','7|9|10|11','Div %|PO %|Elo|xWΔ|TotalWAR'),
 'american-baseball-conference-western-division':('LV/SEA odds, ELO, BaseRuns and WAR',f"LV Div/PO {T('LV')['division_odds_pct']}/{T('LV')['playoff_odds_pct']} vs SEA {T('SEA')['division_odds_pct']}/{T('SEA')['playoff_odds_pct']}; ELO {T('LV')['elo']} vs {T('SEA')['elo']}; both xWΔ +5",'yes','Both clubs are five wins above BaseRuns xW, adding sustainability questions.','No guaranteed regression.','Vegas model advantage is larger than the three-game margin alone.','reinforce_secondary','7|9|10|11','Div %|PO %|Elo|xWΔ|TotalWAR'),
 'american-baseball-conference-eastern-division':('BOS/NY odds, ELO, BaseRuns and WAR',f"BOS Div/PO {T('BOS')['division_odds_pct']}/{T('BOS')['playoff_odds_pct']} vs NY {T('NY')['division_odds_pct']}/{T('NY')['playoff_odds_pct']}; ELO {T('BOS')['elo']} vs {T('NY')['elo']}",'yes','The model treats the four-game race as less balanced than the standings margin.','Odds are not qualification facts.','Boston has a model-backed advantage beyond the current lead.','keep_mention','7|9|10|11','Div %|PO %|Elo|xWΔ|TotalWAR'),
 'american-baseball-conference-central-division':('HOU/CIN odds, ELO, BaseRuns and WAR',f"HOU Div/PO {T('HOU')['division_odds_pct']}/{T('HOU')['playoff_odds_pct']} vs CIN {T('CIN')['division_odds_pct']}/{T('CIN')['playoff_odds_pct']}; TotalWAR HOU {T('HOU')['total_war']} vs CIN {T('CIN')['total_war']}; CIN pWΔ {T('CIN')['baseruns_pythag_win_delta']}",'yes','Cincinnati slightly leads team WAR and is four wins below pW despite Houston’s large odds edge.','Do not convert under-performance into promised rebound.','The challenger may be stronger underneath the five-game gap than odds alone imply.','keep_mention_watch','7|9|10|11','Div %|PO %|Elo|pWΔ|TotalWAR'),
 'scheduled_matchup_stakes':('LV/DEN odds, ELO, BaseRuns, WAR and UBR',f"LV ELO {T('LV')['elo']} vs DEN {T('DEN')['elo']}; PO odds {T('LV')['playoff_odds_pct']} vs {T('DEN')['playoff_odds_pct']}; UBR {T('LV')['team_ubr']} vs {T('DEN')['team_ubr']}",'yes','Both are +5 xWΔ; Denver’s UBR is sharply negative while Vegas is positive.','Pregame context only; no July 20 result and no causal baserunning claim.','Model strength and baserunning contrast sharpen the setup.','reinforce_setup','7|9|10|11|19','PO %|Elo|xWΔ|TotalWAR|UBR'),
 'ace_value_leader':('Coronado rWAR/FIP/xFIP context','rWAR 4.28; WAR 4.34; ERA 2.859; FIP 2.701; xFIP 3.053; K/9 10.10','yes','Adds an alternate value lens that agrees with WAR.','rWAR is labeled StatsPlus value and does not decide awards.','Run prevention and fielding-independent marks strengthen the ace case.','reinforce_secondary','21','rWAR|WAR|ERA|FIP|xFIP|K/9'),
 'position_player_value_leader':('Flores offensive-value context','WAR 4.28; wRC+ 167.97; wOBA .4316; OPS .9546; 18 HR','yes','Adds rate/offensive context behind the value lead.','StatsPlus WAR does not decide awards.','Flores combines current total value with elite rate production.','reinforce_secondary','20','WAR|WRC+|wOBA|OPS|HR'),
 'weekly_rise':('Atlanta ELO, BaseRuns, WAR, odds and injuries',f"ELO 7d {T('ATL')['elo_7day_change']}, 30d {T('ATL')['elo_30day_change']}; xWΔ {T('ATL')['baseruns_expected_win_delta']}; TotalWAR {T('ATL')['total_war']}; PO odds {T('ATL')['playoff_odds_pct']}",'yes','Positive ELO movement and strong WAR coexist with a poor record and only 1.2% playoff odds.','Still not a turnaround; aggregate injuries do not prove cause.','Atlanta underlying-strength/under-performance investigation.','consider_new_investigation','7|9|10|11|12','PO %|Elo changes|xWΔ|TotalWAR|injuries'),
 'weekly_fall':('Philadelphia injury burden','7 injuries; 234 DL days; $168k on DL','limited','Injury burden is relevant context but does not prove the one-week fall.','Do not call collapse or assign injury causation without player linkage.','Investigate roster availability separately.','remain_hold','12','Count|DL Days|$ on DL'),
 'pythag_record_gap__3':('Atlanta BaseRuns/ELO/WAR','xWΔ -6; pWΔ -5; ELO 7d +6.5; TotalWAR 23.06','yes','Underlying strength and recent ELO direction make the gap more interesting.','No “bad luck” claim or promised correction.','Underlying-strength divergence deserves investigation.','consider_raise_watch','9|10|11','xWΔ|pWΔ|7d+/-|TotalWAR'),
 'pythag_record_gap__24':('Seattle BaseRuns/odds/ELO','xWΔ +5; pWΔ +5; Div odds 20.3; PO odds 42.4; ELO 1512.7','yes','Confirms the record/run-balance tension while keeping Seattle live in the race.','No guaranteed regression.','Sustainability question inside a real race.','keep_watch','7|9|10','xWΔ|pWΔ|Div %|PO %|Elo'),
 'payroll_record_pressure':('Tampa Bay finance, odds, BaseRuns, ELO, WAR and fan context',f"Payroll ${int(T('TB')['payroll']):,}; PO odds {T('TB')['playoff_odds_pct']}; ELO {T('TB')['elo']}; xWΔ {T('TB')['baseruns_expected_win_delta']}; Fan interest {T('TB')['fan_interest']}",'yes','BaseRuns sees four more expected wins, complicating a simple spending-failure frame.','No blame, motive, or transaction prediction.','Resources/results/model gap is an organization question.','reinforce_watch','3|5|7|9|10|11','Payroll|PO %|Elo|xWΔ|TotalWAR|Fan Interest'),
 'attendance_interest':('Detroit fan-interest and capacity context',f"Fan Interest {T('DET')['fan_interest']}; Avg attendance {T('DET')['average_attendance']}; {float(T('DET')['capacity_pct_full']):.1f}% full; 1980/81 interest {T('DET')['historical_fan_interest_1980']}/{T('DET')['historical_fan_interest_1981']}",'yes','Adds measured interest and historical continuity.','Does not establish emotion or causal effect on play.','Public scale around the race.','reinforce_watch','4|5','Fan Interest|Avg.Att.|% Full|1980|1981'),
 'prior_qualifier_echo':('Charlotte odds, ELO and WAR',f"Div/PO odds {T('CHA')['division_odds_pct']}/{T('CHA')['playoff_odds_pct']}; ELO {T('CHA')['elo']}; TotalWAR {T('CHA')['total_war']}",'yes','Shows current strength behind the historical echo.','History and models are context, not destiny.','The prior qualifier is also the current model leader.','reinforce_mention','7|10|11','Div %|PO %|Elo|TotalWAR'),
 }
 overlays=[]
 for c in candidates:
  cid=c['candidate_id'];key=next((k for k in overlay_data if k in cid),None);d=overlay_data[key]
  overlays.append({**c,'statsplus_signals':d[0],'statsplus_evidence_summary':d[1],'strengthens_story':d[2],
   'complicates_story':d[3],'creates_caution':d[4],'suggests_new_angle':d[5],'priority_effect_later':d[6],
   'source_tables_used':d[7],'exact_fields_used':d[8],
   'authority_caution':'StatsPlus enhances only; raw OOTP retains standings/game authority. No ranking changed in this preview.'})
 ostem=ENRICH/f'story_candidates_statsplus_overlay_{SLUG}';dumpcsv(ostem.with_suffix('.csv'),overlays);dumpjson(ostem.with_suffix('.json'),{'candidate_count':len(overlays),'original_candidates_modified':False,'ranking_changed':False,'overlays':overlays})
 ostem.with_suffix('.md').write_text('# July 20 Candidate StatsPlus Overlay\n\nThe 15 frozen candidates remain unchanged. This file adds Feed 3 evidence notes only.\n\n'+mdtable(['Candidate','Signals','Feed 3 evidence','Strengthens','Priority later','Caution'],[[r['headline_factual'],r['statsplus_signals'],r['statsplus_evidence_summary'],r['strengthens_story'],r['priority_effect_later'],r['creates_caution']] for r in overlays])+'\n',encoding='utf-8')

 comparison=[
  ('Dallas–Detroit lead','Strengthens',overlay_data['national-baseball-conference-central-division'][1],'Keep lead. Detroit’s slight ELO edge adds competitive tension.'),
  ('Las Vegas–Seattle','Strengthens with caution',overlay_data['american-baseball-conference-western-division'][1],'Keep secondary; both +5 xWΔ discourages certainty.'),
  ('San Francisco–Phoenix','Strongly enriches/complicates',overlay_data['national-baseball-conference-western-division'][1],'Consider raising within secondary board; investigate wide model gap.'),
  ('Las Vegas at Denver','Strengthens setup',overlay_data['scheduled_matchup_stakes'][1],'Keep as unplayed setup.'),
  ('José Coronado','Strengthens',overlay_data['ace_value_leader'][1],'Add labeled rWAR/FIP context; no award claim.'),
  ('Manny Flores','Strengthens',overlay_data['position_player_value_leader'][1],'Add wRC+/wOBA/OPS context; no award claim.'),
  ('Boston–New York','Adds model separation',overlay_data['american-baseball-conference-eastern-division'][1],'Keep mention; model is less balanced than standings.'),
  ('Houston–Cincinnati','Adds useful counterpoint',overlay_data['american-baseball-conference-central-division'][1],'Keep mention/watch; Cincinnati WAR/under-performance complicates odds.'),
  ('Atlanta','Creates investigation angle',overlay_data['weekly_rise'][1],'Underlying strength plus positive ELO movement merits investigation, not promotion yet.'),
  ('Seattle sustainability','Strengthens watch',overlay_data['pythag_record_gap__24'][1],'Keep watch; no regression claim.'),
  ('Detroit attendance','Strengthens measured context',overlay_data['attendance_interest'][1],'Use interest/capacity figures without inferred emotion.'),
  ('Tampa Bay pressure','Strengthens and complicates',overlay_data['payroll_record_pressure'][1],'Keep watch; BaseRuns softens simple failure framing.'),
  ('Chicago–Dallas series','Supports series setup',f"DAL Div/PO 60.0/79.7 vs CHI 15.0/35.9; ELO {T('DAL')['elo']} vs {T('CHI')['elo']}; WAR {T('DAL')['total_war']} vs {T('CHI')['total_war']}",'Use as enhancement to existing raw-proof investigation.'),
 ]
 new_angles=['San Francisco–Phoenix model-gap investigation','Atlanta underlying-strength/under-performance investigation','Philadelphia injury/availability investigation without collapse framing']
 comp={'newsroom_date':'1981-07-20','as_of_date':'1981-07-19','slate_rewrite_recommended':False,'limited_slate_annotation_recommended':True,'new_possible_angles':new_angles,'comparisons':[{'topic':a,'verdict':b,'evidence':c,'recommendation':d} for a,b,c,d in comparison]}
 cstem=ENRICH/f'statsplus_editorial_comparison_{SLUG}';dumpjson(cstem.with_suffix('.json'),comp);cstem.with_suffix('.md').write_text('# StatsPlus Editorial Comparison\n\n> StatsPlus enhances. It does not replace.\n\n'+mdtable(['Topic','Verdict','Feed 3 evidence','Recommendation'],comparison)+'\n\n## New investigation angles\n\n'+'\n'.join('- '+x for x in new_angles)+'\n\nThe existing slate should not be rewritten. Add annotations, consider elevating the San Francisco–Phoenix model-gap discussion, and commission the new investigations separately.\n',encoding='utf-8')

 chi,dal=T('CHI'),T('DAL')
 chinote={'newsroom_date':'1981-07-20','as_of_date':'1981-07-19','classification':'supports_PROMOTE_SERIES_SETUP','raw_ootp_remains_series_authority':True,
  'chicago':chi,'dallas':dal,'comparisons':{
   'playoff_odds':f"Chicago Div/PO {chi['division_odds_pct']}/{chi['playoff_odds_pct']}; Dallas {dal['division_odds_pct']}/{dal['playoff_odds_pct']}",
   'elo':f"Chicago {chi['elo']}; Dallas {dal['elo']}",'baseruns':f"Chicago xWΔ {chi['baseruns_expected_win_delta']}; Dallas {dal['baseruns_expected_win_delta']}",
   'team_war':f"Chicago {chi['total_war']}; Dallas {dal['total_war']}",'injuries':f"Chicago {chi['injury_count']} / {chi['dl_days']} DL days; Dallas {dal['injury_count']} / {dal['dl_days']}",
   'fan_finance':f"Chicago interest {chi['fan_interest']}, {float(chi['capacity_pct_full']):.1f}% full, payroll ${int(chi['payroll']):,}; Dallas interest {dal['fan_interest']}, {float(dal['capacity_pct_full']):.1f}% full, payroll ${int(dal['payroll']):,}",
   'owner_context':f"Chicago owner {chi['owner_name']} ({chi['owner_priority']}, {chi['owner_involvement']}); Dallas owner {dal['owner_name']} ({dal['owner_priority']}, {dal['owner_involvement']})",
   'team_age':f"Chicago ML age {chi['ml_age']}; Dallas {dal['ml_age']}",'best_games':f"Chicago: {chi['best_pitching_game_notes']}; Dallas: {dal['best_pitching_game_notes']}"},
  'verdict':'Feed 3 strengthens the series setup. Dallas has the large odds/WAR advantage, while ELO is close and BaseRuns is nearly neutral. Chicago remains a plausible direct challenger, not a model favorite.',
  'cautions':['Do not prove schedule/results from StatsPlus.','Do not claim owner motives.','Best-game notes require raw verification before publication.','Odds are model output, not destiny.']}
 chistem=INV/f'chicago_dallas_statsplus_enrichment_note_{SLUG}';dumpjson(chistem.with_suffix('.json'),chinote)
 chistem.with_suffix('.md').write_text(f'''# Chicago–Dallas StatsPlus Enrichment Note

**Verdict:** Feed 3 supports the existing `PROMOTE_SERIES_SETUP` classification. Raw OOTP remains the authority for the Detroit split and July 20–23 schedule.

- Playoff odds: {chinote['comparisons']['playoff_odds']}.
- ELO: {chinote['comparisons']['elo']}.
- BaseRuns: {chinote['comparisons']['baseruns']}.
- Team WAR: {chinote['comparisons']['team_war']}.
- Injuries: {chinote['comparisons']['injuries']}.
- Fan/financial: {chinote['comparisons']['fan_finance']}.
- Owner texture: {chinote['comparisons']['owner_context']}.
- Team age: {chinote['comparisons']['team_age']}.
- Ranked best-game notes: {chinote['comparisons']['best_games']}.

Dallas has the clear model-odds and team-WAR edge. ELO is close, and BaseRuns sees both clubs near their expected win levels. That combination strengthens “direct opportunity against the favorite,” not a claim that Chicago is equally likely to win the division.

Do not infer owner motives, use best-game rankings without raw verification, or treat odds as destiny.
''',encoding='utf-8')

 safe=['playoff odds with model label','ELO/team strength with model label','BaseRuns expected-performance context','team WAR with StatsPlus label','aggregate injury burden','fan-interest measures','financial context','team age','team/player baserunning including UBR','pitcher rWAR as a distinct metric']
 manual=['owner/front-office/personnel traits','best batting/pitching game discovery','historical fan-interest interpretation','tables 7/8 dual sort views']
 disabled=['StatsPlus standings authority','manager tendencies or motives','owner mood/season objective/specific goals','causal injury claims','automatic collapse/regression claims','unverified best-game facts']
 rec={'safe_to_enable_later':safe,'manual_editorial_reference_only':manual,'remain_disabled':disabled,'enable_now':False,
  'candidate_ranking_recommendation':'Do not automatically rerank. Add evidence notes; allow editorial review to elevate model-gap investigations.',
  'evidence_note_recommendation':'Feed 3 should initially append typed evidence notes with source/authority labels.',
  'july_20_slate_recommendation':'Do not rewrite the slate. Annotate Dallas–Detroit, LV–Seattle, SF–Phoenix, LV–Denver, Coronado, Flores, Atlanta, Detroit and Tampa Bay; consider SF–Phoenix model gap as a stronger secondary discussion.',
  'chicago_dallas_full_investigation':'Yes. The raw-proof investigation packet already exists; this Feed 3 note should be attached as enrichment.',
  'recommended_next_task':'Design a feature-flagged Feed 3 adapter and evidence schema mapping, then test on a future run without changing ranking by default.'}
 rstem=ENRICH/f'statsplus_story_integration_recommendation_{SLUG}';dumpjson(rstem.with_suffix('.json'),rec)
 rstem.with_suffix('.md').write_text('# StatsPlus Story Integration Recommendation\n\n**Do not enable signals in this task.**\n\n## Safe to enable later\n\n'+'\n'.join('- '+x for x in safe)+'\n\n## Manual/editorial reference only\n\n'+'\n'.join('- '+x for x in manual)+'\n\n## Remain disabled\n\n'+'\n'.join('- '+x for x in disabled)+'''\n\n## Editorial decision

Do not automatically rerank candidates. Feed 3 should first append typed evidence notes with explicit authority labels. The July 20 slate should not be rewritten; it should receive limited annotations, with San Francisco–Phoenix considered for stronger secondary treatment.

Chicago–Dallas deserves a full investigation. Its raw-proof packet already exists, and the Feed 3 enhancement note should be attached to it.
''',encoding='utf-8')
 print(json.dumps({'enrichment_views_created':True,'team_rows':len(teamrows),'player_rows':len(prows),'overlay_created':True,'existing_candidates_enriched':len(overlays),'new_possible_story_angles':len(new_angles),'chicago_dallas_statsplus_support':True,'slate_rewrite_recommended':False},indent=2))

if __name__=='__main__':main()
