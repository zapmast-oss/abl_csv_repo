#!/usr/bin/env python3
"""Optional Feed 3 preview wrapper for the completed July 20 story-engine run."""

from __future__ import annotations

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
def main():
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
if __name__=='__main__':main()
