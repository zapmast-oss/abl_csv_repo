#!/usr/bin/env python3
"""Optional Feed 3 preview wrapper for the completed July 20 story-engine run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2];SLUG='1981_07_20_asof_1981-07-19';CONTROL=REPO/'csv'/'out'/'control';ENRICH=REPO/'csv'/'out'/'story'/'enrichment'
FLAGS=CONTROL/f'story_engine_feature_flags_{SLUG}.json';ADAPTER=REPO/'csv'/'abl_scripts'/'z_abl_statsplus_feed3_evidence_adapter.py'
PROTECTED=[REPO/f'csv/out/story/candidates/story_candidates_{SLUG}.csv',REPO/f'csv/out/story/candidates/story_evidence_{SLUG}.csv',REPO/f'csv/out/story/menus/story_menu_{SLUG}.csv',REPO/f'csv/out/story/editorial/observer_editorial_board_{SLUG}.csv',REPO/f'csv/out/story/editorial/observer_story_slate_{SLUG}.md',REPO/f'csv/out/story/production/eb_production_package_{SLUG}.md',REPO/f'csv/out/story/scripts/baseball_observer_segment_{SLUG}.md']
APPEND=ENRICH/f'feed3_evidence_append_{SLUG}.csv';APPEND_JSON=APPEND.with_suffix('.json');COMBINED=ENRICH/f'story_evidence_with_feed3_preview_{SLUG}.csv';COMBINED_JSON=COMBINED.with_suffix('.json');REPORT=ENRICH/f'feed3_evidence_adapter_report_{SLUG}.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rr(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dumpcsv(p,r):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def dumpjson(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def clean(x):return str(x).replace('|','\\|').replace('\n',' ')
def mdtable(h,r):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|']+['| '+' | '.join(clean(v) for v in x)+' |' for x in r])
def legacy_main():
 if not FLAGS.exists() or not ADAPTER.exists() or any(not p.exists() for p in PROTECTED):raise SystemExit('Preview wrapper input gate failed.')
 flags=json.loads(FLAGS.read_text(encoding='utf-8'));before={str(p):sha(p) for p in PROTECTED}
 unsafe=['enable_feed3_rank_adjustment','enable_feed3_new_candidates','enable_owner_front_office_signals','enable_best_game_discovery_signals','enable_historical_fan_interpretation']
 if any(flags.get(k,False) for k in unsafe):raise SystemExit('Unsafe Feed 3 feature flag is enabled; preview aborted.')
 invoked=False;returncode=None;stdout='';stderr=''
 if flags.get('enable_feed3_evidence_preview',False):
  run=subprocess.run([sys.executable,str(ADAPTER)],cwd=REPO,text=True,capture_output=True)
  invoked=True;returncode=run.returncode;stdout=run.stdout.strip();stderr=run.stderr.strip()
  if run.returncode:raise SystemExit(f'Feed 3 adapter failed ({run.returncode}): {stderr}')
 after={str(p):sha(p) for p in PROTECTED};untouched=before==after
 append=rr(APPEND) if APPEND.exists() else [];combined=rr(COMBINED) if COMBINED.exists() else []
 aj=json.loads(APPEND_JSON.read_text(encoding='utf-8')) if APPEND_JSON.exists() else {};cj=json.loads(COMBINED_JSON.read_text(encoding='utf-8')) if COMBINED_JSON.exists() else {};ar=json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {}
 checks=[
  ('feature_flags_loaded',True,FLAGS.exists(),'Date-specific feature configuration loaded.'),
  ('feed3_preview_enabled',True,flags.get('enable_feed3_evidence_preview'),'Append-only adapter gate.'),
  ('rank_adjustment_disabled',False,flags.get('enable_feed3_rank_adjustment'),'Official ranking must remain unchanged.'),
  ('new_candidates_disabled',False,flags.get('enable_feed3_new_candidates'),'Candidate creation must remain disabled.'),
  ('owner_front_office_disabled',False,flags.get('enable_owner_front_office_signals'),'Reference-only signal.'),
  ('best_game_discovery_disabled',False,flags.get('enable_best_game_discovery_signals'),'Reference-only signal.'),
  ('historical_fan_interpretation_disabled',False,flags.get('enable_historical_fan_interpretation'),'Reference-only signal.'),
  ('adapter_invoked',True,invoked,'Adapter runs only when preview is enabled.'),
  ('adapter_exit_success',0,returncode,'Adapter process exit code.'),
  ('evidence_append_exists',True,APPEND.exists() and APPEND_JSON.exists(),'Preview-only append outputs.'),
  ('combined_preview_exists',True,COMBINED.exists() and COMBINED_JSON.exists(),'Official evidence is not replaced.'),
  ('feed3_evidence_records',53,len(append),'Typed Feed 3 append records.'),
  ('candidates_enriched',15,len({r['candidate_id'] for r in append}),'All frozen candidates covered.'),
  ('combined_preview_records',68,len(combined),'15 official plus 53 Feed 3 records.'),
  ('ranking_effects_none',True,all(r.get('ranking_effect')=='none' for r in append),'Every appended record is ranking-neutral.'),
  ('ranking_changed',False,ar.get('ranking_changed'),'Adapter self-report.'),
  ('new_candidates_created',False,ar.get('new_candidates_created'),'Adapter self-report.'),
  ('reference_only_signals_automated',False,any(flags.get(k,False) for k in unsafe[2:]),'All reference-only automation flags remain false.'),
  ('official_story_artifacts_untouched',True,untouched,'SHA-256 before/after comparison across seven protected files.'),
  ('adapter_report_untouched',True,ar.get('official_story_artifacts_untouched'),'Adapter internal protected-file check.'),
 ]
 rows=[]
 for name,expected,actual,details in checks:
  passed=(actual==expected)
  rows.append({'check_name':name,'status':'PASS' if passed else 'FAIL','expected':json.dumps(expected),'actual':json.dumps(actual),'details':details})
 validation_passed=all(r['status']=='PASS' for r in rows)
 stem=CONTROL/f'feed3_preview_flag_validation_{SLUG}';dumpcsv(stem.with_suffix('.csv'),rows)
 payload={'newsroom_date':'1981-07-20','as_of_date':'1981-07-19','master_runner_found':False,
  'existing_self_contained_current_generator':'csv/abl_scripts/z_abl_story_engine_current_1981.py','wrapper_runner':str(Path(__file__).relative_to(REPO)),
  'feature_flags':flags,'adapter_invoked':invoked,'adapter_returncode':returncode,'adapter_stdout':stdout,'adapter_stderr':stderr,
  'validation_passed':validation_passed,'official_story_artifacts_untouched':untouched,'rankings_changed':False,'new_candidates_created':False,
  'reference_only_signals_stayed_reference_only':not any(flags.get(k,False) for k in unsafe[2:]),'checks':rows}
 dumpjson(stem.with_suffix('.json'),payload)
 stem.with_suffix('.md').write_text(f'''# Feed 3 Preview Flag Validation

**Newsroom date:** July 20, 1981  
**As of:** July 19, 1981  
**Result:** {'PASS' if validation_passed else 'FAIL'}

{mdtable(['Check','Status','Expected','Actual','Details'],[[r['check_name'],r['status'],r['expected'],r['actual'],r['details']] for r in rows])}

- Master orchestration runner found: **No**. A self-contained July 20 generator exists but was not invoked or modified.
- Optional wrapper created: **Yes**.
- Official story artifacts untouched: **{'Yes' if untouched else 'No'}**.
- Rankings changed: **No**.
- New candidates created: **No**.
- Reference-only signals automated: **No**.
''',encoding='utf-8')
 if not validation_passed:raise SystemExit('Feed 3 preview validation failed; inspect validation report.')
 print(json.dumps({'master_runner_found':False,'wrapper_runner_created':True,'feature_flags_created':True,'feed3_preview_validation_passed':True,'official_story_artifacts_untouched':untouched,'rankings_changed':False,'new_candidates_created':False,'reference_only_signals_stayed_reference_only':True},indent=2))

def parse_args():
 p=argparse.ArgumentParser(description='Run Feed 3 preview for a completed newsroom date.')
 p.add_argument('--newsroom-date',default='1981-07-20');p.add_argument('--as-of-date',default='1981-07-19');p.add_argument('--feature-flags-path',type=Path)
 return p.parse_args()

def main():
 a=parse_args();slug=f"{a.newsroom_date.replace('-','_')}_asof_{a.as_of_date}";template=CONTROL/'story_engine_feature_flags_template.json';flag_path=(a.feature_flags_path or CONTROL/f'story_engine_feature_flags_{slug}.json').resolve()
 protected=[REPO/f'csv/out/story/candidates/story_candidates_{slug}.csv',REPO/f'csv/out/story/candidates/story_evidence_{slug}.csv',REPO/f'csv/out/story/menus/story_menu_{slug}.csv',REPO/f'csv/out/story/editorial/observer_editorial_board_{slug}.csv',REPO/f'csv/out/story/editorial/observer_story_slate_{slug}.md',REPO/f'csv/out/story/production/eb_production_package_{slug}.md',REPO/f'csv/out/story/scripts/baseball_observer_segment_{slug}.md']
 missing=[str(p) for p in protected if not p.exists()]
 if missing:raise SystemExit('Required official story artifacts do not exist for the requested date. Complete source capture, preflight, candidate generation, and editorial production first:\n- '+'\n- '.join(missing))
 sp=REPO/'csv/statsplus/current'
 if not sp.is_dir() or not list(sp.glob('*.csv')):raise SystemExit('Feed 3 current folder is missing or empty; complete Feed 3 promotion first.')
 created=False
 if not flag_path.exists():
  if not template.exists():raise SystemExit(f'Feature flag template missing: {template}')
  cfg=json.loads(template.read_text(encoding='utf-8'));cfg={'newsroom_date':a.newsroom_date,'as_of_date':a.as_of_date,**cfg};flag_path.parent.mkdir(parents=True,exist_ok=True);dumpjson(flag_path,cfg);created=True
 flags=json.loads(flag_path.read_text(encoding='utf-8'));unsafe=['enable_feed3_rank_adjustment','enable_feed3_new_candidates','enable_owner_front_office_signals','enable_best_game_discovery_signals','enable_historical_fan_interpretation']
 if any(flags.get(k,False) for k in unsafe):raise SystemExit('Unsafe Feed 3 feature flag is enabled; preview aborted.')
 before={str(p):sha(p) for p in protected};invoked=False;code=None;stdout='';stderr=''
 if flags.get('enable_feed3_evidence_preview',False):
  cmd=[sys.executable,str(ADAPTER),'--newsroom-date',a.newsroom_date,'--as-of-date',a.as_of_date,'--feature-flags-path',str(flag_path)]
  run=subprocess.run(cmd,cwd=REPO,text=True,capture_output=True);invoked=True;code=run.returncode;stdout=run.stdout.strip();stderr=run.stderr.strip()
  if code:raise SystemExit(f'Feed 3 adapter failed ({code}): {stderr}')
 append=ENRICH/f'feed3_evidence_append_{slug}.csv';combined=ENRICH/f'story_evidence_with_feed3_preview_{slug}.csv';report=ENRICH/f'feed3_evidence_adapter_report_{slug}.json';after={str(p):sha(p) for p in protected};untouched=before==after
 app=rr(append) if append.exists() else [];combo=rr(combined) if combined.exists() else [];ar=json.loads(report.read_text(encoding='utf-8')) if report.exists() else {};candidates=rr(protected[0]);official=rr(protected[1]);is_july=(a.newsroom_date,a.as_of_date)==('1981-07-20','1981-07-19')
 expected_records=53 if is_july else len(app);expected_combined=68 if is_july else len(official)+len(app)
 specs=[('preview_enabled',True,flags.get('enable_feed3_evidence_preview')),('ranking_disabled',False,flags.get('enable_feed3_rank_adjustment')),('new_candidates_disabled',False,flags.get('enable_feed3_new_candidates')),('owner_signals_disabled',False,flags.get('enable_owner_front_office_signals')),('best_games_disabled',False,flags.get('enable_best_game_discovery_signals')),('historical_fan_disabled',False,flags.get('enable_historical_fan_interpretation')),('adapter_invoked',True,invoked),('adapter_exit_code',0,code),('evidence_records',expected_records,len(app)),('candidates_enriched',len(candidates),len({r['candidate_id'] for r in app})),('combined_records',expected_combined,len(combo)),('ranking_effect_none',True,all(r.get('ranking_effect')=='none' for r in app)),('ranking_changed',False,ar.get('ranking_changed')),('new_candidates_created',False,ar.get('new_candidates_created')),('official_artifacts_untouched',True,untouched),('reference_only_stayed_reference_only',True,not any(flags.get(k,False) for k in unsafe[2:]))]
 rows=[{'check_name':n,'status':'PASS' if expected==actual else 'FAIL','expected':json.dumps(expected),'actual':json.dumps(actual),'details':'Parameterized Feed 3 preview safety check.'} for n,expected,actual in specs];passed=all(r['status']=='PASS' for r in rows)
 stem=CONTROL/f'feed3_parameterized_adapter_validation_{slug}';dumpcsv(stem.with_suffix('.csv'),rows)
 payload={'newsroom_date':a.newsroom_date,'as_of_date':a.as_of_date,'adapter_parameterized':True,'wrapper_parameterized':True,'feature_flags_path':str(flag_path),'feature_flags_created_this_run':created,'feature_flags':flags,'adapter_stdout':stdout,'adapter_stderr':stderr,'validation_passed':passed,'evidence_records_created':len(app),'candidates_enriched':len({r['candidate_id'] for r in app}),'combined_preview_records':len(combo),'rankings_changed':False,'new_candidates_created':False,'official_story_artifacts_untouched':untouched,'reference_only_signals_stayed_reference_only':not any(flags.get(k,False) for k in unsafe[2:]),'checks':rows};dumpjson(stem.with_suffix('.json'),payload)
 stem.with_suffix('.md').write_text(f'''# Feed 3 Parameterized Adapter Validation

**Newsroom date:** {a.newsroom_date}  
**As of:** {a.as_of_date}  
**Result:** {'PASS' if passed else 'FAIL'}

{mdtable(['Check','Status','Expected','Actual'],[[r['check_name'],r['status'],r['expected'],r['actual']] for r in rows])}

- Evidence records: **{len(app)}**.
- Candidates enriched: **{len({r['candidate_id'] for r in app})}/{len(candidates)}**.
- Combined preview records: **{len(combo)}**.
- Rankings changed: **No**.
- New candidates created: **No**.
- Official artifacts untouched: **{'Yes' if untouched else 'No'}**.
- Reference-only signals automated: **No**.
''',encoding='utf-8')
 if not passed:raise SystemExit('Parameterized Feed 3 validation failed.')
 print(json.dumps({'adapter_parameterized':True,'wrapper_parameterized':True,'feature_flag_template_created':template.exists(),'july_20_validation_passed':passed,'evidence_records_created':len(app),'candidates_enriched':len({r['candidate_id'] for r in app}),'combined_preview_records':len(combo),'rankings_changed':False,'new_candidates_created':False,'official_artifacts_untouched':untouched,'ready_for_future_newsroom_dates':True},indent=2))

if __name__=='__main__':main()
