import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from challenge_shortcut_audit import validate, read_jsonl, only_exceptions, only_keys, normalize_text

ROOT=Path(__file__).resolve().parent

class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full=read_jsonl(ROOT/'candidates_full.jsonl')
        cls.prompt=read_jsonl(ROOT/'model_input_only.jsonl')
    def test_positive_structure(self):
        err,_,pairs,counts=validate(self.full,self.prompt)
        self.assertFalse(err)
        self.assertEqual(len(pairs),12)
        self.assertEqual(counts['CONTEXT_LIMIT'],15)
    def test_injected_label_field_fails(self):
        prompts=copy.deepcopy(self.prompt)
        prompts[0]['label']='ENV_DEPENDENCY'
        err,*_=validate(self.full,prompts)
        self.assertTrue(any('prompt keys' in e for e in err))
    def test_text_mutation_fails(self):
        prompts=copy.deepcopy(self.prompt)
        prompts[0]['error_text']+=' fake log'
        err,*_=validate(self.full,prompts)
        self.assertTrue(any('prompt text differs' in e for e in err))
    def test_duplicate_ids_fails(self):
        full=copy.deepcopy(self.full)
        full[1]['id']=full[0]['id']
        err,*_=validate(full,self.prompt)
        self.assertTrue(any('duplicate' in e for e in err))
    def test_surface_extractors(self):
        s='run=xy\nTraceback: ContentDecodingError\ncase_id=221'
        self.assertIn('ContentDecodingError',only_exceptions(s))
        self.assertIn('case_id',only_keys(s))
        self.assertNotIn('221',normalize_text(s))

if __name__=='__main__':unittest.main()
