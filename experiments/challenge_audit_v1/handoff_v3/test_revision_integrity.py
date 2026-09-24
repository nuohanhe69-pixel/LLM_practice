import collections
import hashlib
import json
import re
import unittest
import zipfile
from pathlib import Path

ROOT=Path(__file__).parent
BEFORE=ROOT/'baseline/candidates_full.jsonl'
AFTER=ROOT/'candidates_full_v3.jsonl'
PRED=ROOT/'model_input_only_v3.jsonl'
old=[json.loads(l) for l in BEFORE.read_text(encoding='utf-8').splitlines() if l]
new=[json.loads(l) for l in AFTER.read_text(encoding='utf-8').splitlines() if l]
olds={r['id']:r for r in old};news={r['id']:r for r in new}

class RevisionIntegrity(unittest.TestCase):
 def test_original_snapshot_immutable(self):
  self.assertEqual(hashlib.sha256(BEFORE.read_bytes()).hexdigest(),'4e2efad0c333f320e9fc69b90b1c87458666d7c5a4c54d0a74a037d8d3a3e4ea')
 def test_no_gt_or_pair_changes(self):
  self.assertEqual(len(new),60)
  for a,b in zip(old,new):
   for k in ('id','design_id','label','primary_distractor','boundary_pair','source_type','split'):
    self.assertEqual(a[k],b[k],(a['id'],k))
  self.assertEqual(dict(collections.Counter(r['label'] for r in new)),dict(collections.Counter(r['label'] for r in old)))
  self.assertEqual(dict(collections.Counter((r['label'],r['primary_distractor']) for r in new)),dict(collections.Counter((r['label'],r['primary_distractor']) for r in old)))
 def test_only_declared_edits(self):
  changed={k for k in olds if olds[k]['error_text']!=news[k]['error_text']}
  self.assertEqual(changed,{'T006','T007','T010','T013','T017','T018','T023','T025','T027','T028','T029','T041','T043','T046','T048','T049','T053','T055'})
 def test_remaining_flags_no_false_completion(self):
  flagged={x['id'] for x in new if x.get('review_flags')}
  self.assertEqual(flagged,{'T011','T042','T044'})
  for k in flagged:
   self.assertEqual(news[k]['review_status'],'REQUIRES_UNINVOLVED_REVIEWERS_FOR_CHALLENGE_STRENGTH')
 def test_model_input_strict(self):
  prompts=[json.loads(l) for l in PRED.read_text(encoding='utf-8').splitlines() if l]
  self.assertEqual(len(prompts),60)
  for p,r in zip(prompts,new):
   self.assertEqual(set(p),{'id','error_text'})
   self.assertEqual(p,{'id':r['id'],'error_text':r['error_text']})
 def test_key_mechanism_replacement(self):
  self.assertIn('tenant_slot_group',news['T049']['error_text'])
  self.assertIn('service scheduler',news['T049']['error_text'])
  self.assertNotIn('downstream e1',news['T049']['error_text'])
  self.assertIn('last_forwarded',news['T014']['error_text'])
 def test_blind_packet_contains_no_source_ids_or_labels(self):
  with zipfile.ZipFile(ROOT/'仅发审阅者_匿名容量镜像盲审包.zip') as z:
   names=z.namelist()
   self.assertEqual(len(names),3)
   blob='\n'.join(z.read(n).decode('utf-8-sig') for n in names)
   self.assertNotIn('T011',blob);self.assertNotIn('T042',blob);self.assertNotIn('T044',blob)
   self.assertNotIn('design_id',blob);self.assertNotIn('review_flags',blob)
   self.assertNotIn('GT: CONTEXT_LIMIT',blob)
 def test_only_naturalized_terms_removed(self):
  terms=['limit_entry','component','history_summary']
  for t in terms:
   pattern=re.compile(r'(?i)\b'+re.escape(t)+r'\b')
   self.assertTrue(any(pattern.search(r['error_text']) for r in old))
   self.assertFalse(any(pattern.search(r['error_text']) for r in new))
  self.assertEqual(sum(bool(re.search(r'(?i)\bcapacity\b',r['error_text'])) for r in new),7)

if __name__=='__main__':unittest.main()
