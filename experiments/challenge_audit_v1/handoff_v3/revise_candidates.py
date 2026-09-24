#!/usr/bin/env python3
"""One-pass, reviewable editorial change set. GT and directed pairs never change.

Usage: python revise_candidates.py --in original_candidates_full.jsonl --out ./output
This only rewrites specified SYNTHETIC development candidates; no LLM calls.
"""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_HASH = '4e2efad0c333f320e9fc69b90b1c87458666d7c5a4c54d0a74a037d8d3a3e4ea'
REWRITES = {
'T049': '''incident=conv_882; streaming agent reports ReadTimeout mainly when long histories are attached.
client ticket: request=chat_882 body_sha256=4f27... estimated_input=near_local_budget; 12:14:35 ReadTimeout waiting for assistant message.
edge intake: chat_882 upload=finished; request_format=accepted; queue_ticket=q_882 issued@12:14:04; queue_lane=weighted.
service scheduler q_882: tenant_slot_group=sg_7 active=8 allowed=8; pending_weight=heavy; processing_started=<none> at 12:14:35.
model registry md_22: max_request_units=32768; admission precheck chat_882 prepared_units=30210 output_reserve=1024.
client ticket chat_883 estimated_input=small body_sha256=6e12... queued@12:14:05 lane=light; model response@12:14:10.
scheduler sg_7: other sessions released slots@12:15:01; ticket q_882 expired at client deadline.
incident follow-up chat_884: body_sha256=4f27... enqueue@12:16:03 sg_7 active=2; processing@12:16:04 model response@12:16:15.
client binary/version/config unchanged; no payload compaction performed between chat_882 and chat_884.''',
'T007': '''job=agent_startup_819
startup: UnsupportedDependencyVersion: model-client release outside configured range
installation record: distribution=model-client/2.4.1+arm64; loaded module=2.4.1; manifest=mc819
runtime capability probe mc819: create_completion=available stream_completion=available
release metadata from admin UI: label=2.4-LTS channel=stable
app gate trace: input_ref=metadata.labels.release; raw_input=2.4-LTS; normalized=2.4; rule_branch=legacy_range; result=blocked
integration smoke artifact mc819: initialized=true streaming_call=finished; package_digest=73c8...
startup replay: manifest=mc819 package_digest=73c8...; app gate config input_ref=runtime.module_release; raw_input=2.4.1; rule_branch=supported; agent_initialized=true''',
'T029': '''task=native_backend_118
host probe: platform=darwin machine=arm64 python=3.12
worker startup: ImportError dlopen failed: mach-o incompatible architecture; requested artifact_ref=bin_07
installer manifest release=r91: bin_07=plugins/native_core_07.so, macho_cpu=X86_64; bin_11=plugins/native_core_11.so, macho_cpu=ARM64; verification=both_present
startup selection trace: os_key=darwin; raw_arch=arm64; arch_normalized=aarch64; routing_table=arch_selector_v4; chosen=bin_07
loader self-test job=native_diag_05: requested=bin_11 loaded=true inference_fixture=passed
startup replay native_backend_119: package_digest unchanged; arch_selector mapping revision=5 chosen=bin_11; backend_initialized=true''',
'T010': '''job=model_adapter_215
package snapshot: model-client=mc18 provider-adapter=pa18 checksum=75ab...
user config: reasoning=medium
startup failure: TypeError: unexpected keyword argument 'reasoning_effort' at ChatAdapter construction
config parser event c_215: canonical.reasoning=medium provenance=user.reasoning
option trace event o_215: input_ref=c_215; transformation_owner=project.provider_wrapper; vendor_alias_export=enabled
constructor capture: reasoning=medium; provider_options.reasoning_effort=medium
isolated integration fixture i_215: mc18+pa18 options.reasoning=medium client_created=true request=complete
workflow replay build=wrapper_216: package checksum=75ab... config_digest unchanged; vendor_alias_export=disabled; client_created=true''',
'T043': '''job=reasoning_agent_92
user configuration: model=agent-model-x reasoning=medium
runtime report: normalizer=n32 mapper=m18 adapter=a18
configuration event c_92: source=user.reasoning; normalized_option=canonical.reasoning; encoding=canonical_v2
mapper event m_92: input_event=c_92; emitted_options=provider_options.reasoning_effort
constructor exception: TypeError ChatAdapter.__init__ unexpected keyword argument 'reasoning_effort'
build manifest n32: output_encoding=canonical_v2; build manifest m18: input_encoding=canonical_legacy; build manifest m32: input_encoding=canonical_v2
fixture reasoning_medium: n18+m18+a18 client_created=true; n32+m32+a18 client_created=true; app configuration digest unchanged'''
}
# Only remove artificial/repetitive naming where the surrounding original facts keep the decisive evidence.
EDITS = {
'T006': [('limit_entry=md_22', 'sequence_descriptor=md_22')],
'T017': [('limit_entry=md_22', 'request_model_ref=md_22'), ('history_summary=sum_28', 'summary_artifact=sum_28')],
'T018': [('limit_entry=md_22', 'reservation_model_ref=md_22')],
'T023': [('limit_entry=md_22', 'input_descriptor=md_22'), ('history_summary=sum_41', 'summary_artifact=sum_41')],
'T025': [('limit_entry=md_22', 'request_model_ref=md_22')],
'T013': [('history_summary=sum_104', 'history_artifact=sum_104')],
'T028': [('history_summary=sum_74', 'compaction_artifact=sum_74')],
'T027': [('component d14 startup:', 'decoder d14 startup:')],
'T048': [('component contract record:', 'tokenizer asset manifest:')],
'T053': [('merge component record:', 'merge build manifest:')],
'T055': [('component contract:', 'schema exchange manifest:')],
'T041': [('ContextLimitError request exceeds remaining session budget', 'SessionBudgetRejected request exceeds remaining session budget')],
'T046': [('state_rev=42 ContextLimitError', 'state_rev=42 ConversationBudgetRejected')],
}
CHANGES = {
'T049': ('mechanism_substitution', 'Replace duplicate downstream stream stall with remote gateway tenant dispatch queue saturation correlated with long history; retains NETWORK->CONTEXT and client timeout; specifically distinguish per-request model window from external API scheduling availability.'),
'T007': ('single_line_shortcut_repair', 'Split application guard input selection from installed package capability and post-change replay; keep app error first.'),
'T029': ('single_line_shortcut_repair', 'Use opaque artifact IDs and separate installer manifest, platform probe, selector mapping, and loader verification.'),
'T010': ('paired_shortcut_repair', 'Distribute source of invalid option into project wrapper trace and isolated library fixture, not a conclusion line.'),
'T043': ('paired_shortcut_repair', 'Describe normalizer/mapper encoding via independent manifests and retain contrasting integration fixtures; remove diagnostic conclusion.'),
}
CLEAR_INTERNAL = {'T007','T010','T029','T043'}
EXTERNAL_PENDING = {'T011','T042','T044'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
 raw=args.input.read_bytes(); sha=hashlib.sha256(raw).hexdigest();assert sha==EXPECTED_HASH, f'Wrong source snapshot {sha}'
 orig=[json.loads(l) for l in raw.decode('utf-8').splitlines() if l.strip()]
 rows=json.loads(json.dumps(orig));changes=[]
 for row in rows:
  k=row['id']; before=row['error_text'];
  if k in REWRITES:
   row['error_text']=REWRITES[k]; typ,note=CHANGES[k];changes.append({'id':k,'type':typ,'description':note})
   if k=='T049':
    row['design_reason']='历史较长与超时相关，但原请求满足模型单次窗口；远端服务的租户调度队列满载，请求排队至客户端超时，负载下降后完全相同 body 成功。'
    row['evidence_pattern']='remote_service_queue_scope_vs_model_per_request_window'
    row['surface_format']='INCIDENT_TIMELINE'
   if k in CLEAR_INTERNAL:
    assert row['review_flags'],k
    changes.append({'id':k,'type':'internal_flag_closed','previous':row['review_flags'],'new_status':'INTERNAL_SHORTCUT_REVIEW_REVISED_NOT_EXTERNALLY_VALIDATED'})
    row['review_flags']=[]
    row['review_status']='INTERNAL_SHORTCUT_REVISION_PENDING_FINAL_BLIND_REVIEW'
  if k in EDITS:
   for old,new in EDITS[k]:
    assert old in row['error_text'], f'{k} did not contain {old}'
    row['error_text']=row['error_text'].replace(old,new)
    changes.append({'id':k,'type':'template_naturalization','old':old,'new':new,'description':'Field/title editing only; numbers, log facts, mechanism and label untouched.'})
  if k in EXTERNAL_PENDING:
   assert row['review_flags']==['challenge_strength_independent_review_pending']
   row['review_status']='REQUIRES_UNINVOLVED_REVIEWERS_FOR_CHALLENGE_STRENGTH'
  assert row['error_text'].strip()
 # Precise GT and identity checks.
 a={v['id']:v for v in orig};b={v['id']:v for v in rows};assert set(a)==set(b)
 for id in a:
  for key in ['id','design_id','label','primary_distractor','boundary_pair','source_type','split']:
   assert a[id][key]==b[id][key],(id,key)
  if a[id]['error_text']!=b[id]['error_text']:
   assert id in REWRITES or id in EDITS,('unplanned change',id)
 assert [r['id'] for r in rows]==[r['id'] for r in orig]
 args.out.mkdir(parents=True,exist_ok=True)
 (args.out/'candidates_full_v3.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
 (args.out/'model_input_only_v3.jsonl').write_text(''.join(json.dumps({'id':x['id'],'error_text':x['error_text']},ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
 (args.out/'change_log.json').write_text(json.dumps({'source_sha256':sha,'changes':changes,'remaining_external_review_ids':sorted(EXTERNAL_PENDING)},ensure_ascii=False,indent=2),encoding='utf-8')
 edited=[r['id'] for r in rows if a[r['id']]['error_text']!=r['error_text']]
 print('Original SHA:',sha)
 print('Rewritten text IDs:',edited)
 print('Text edits:',len(edited),'flags_closed_internal:',sorted(CLEAR_INTERNAL),'external_pending:',sorted(EXTERNAL_PENDING))
 print('All IDs, labels, distractors, 12 boundaries, and row order unchanged.')
 print('New SHA:',hashlib.sha256((args.out/'candidates_full_v3.jsonl').read_bytes()).hexdigest())
if __name__=='__main__': main()
