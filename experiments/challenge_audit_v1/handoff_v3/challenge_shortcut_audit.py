#!/usr/bin/env python3
"""Deterministic developer-only screening. Does not change GT or call any LLM.

Usage:
 python challenge_shortcut_audit.py --dataset full.jsonl --model-input prompt.jsonl --out audit_dir --seeds 10

Outputs: audit.json, audit.md. Metadata is for audit only, never model prompt.
"""
import argparse
import collections
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, pairwise_distances
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ("NETWORK_API", "CONTEXT_LIMIT", "ENV_DEPENDENCY", "CODE_RUNTIME")
EXCEPTION_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*(?:Error|Exception|Timeout|Rejected|NotFound|Exceeded)\b")
KEY_RE = re.compile(r"(?m)\b([a-zA-Z][\w.\-]+)\s*[:=]")
ID_RE = re.compile(r"\b(?:[0-9a-f]{6,}|[A-Za-z]{1,5}[_-]?\d+[A-Za-z0-9_-]*|\d+(?:\.\d+)*)(?:\.\.\.)?\b", re.I)
MASK_TERMS = re.compile(r"\b(?:env_dependency|context_limit|network_api|code_runtime)\b", re.I)

def read_jsonl(path):
    data=[]
    for lineno, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip(): continue
        obj=json.loads(raw)
        if not isinstance(obj,dict): raise ValueError(f'{path}:{lineno} is not an object')
        data.append(obj)
    return data

def hashfile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def normalize_text(s):
    s=s.lower()
    s=ID_RE.sub(' IDVAL ',s)
    return s

def only_exceptions(s):
    return ' '.join(EXCEPTION_RE.findall(s)) or 'no_exception_detected'

def only_keys(s):
    return ' '.join(KEY_RE.findall(s)) or 'no_key_detected'

def shape_row(s):
    ls=s.splitlines()
    return [len(s),len(ls),len(KEY_RE.findall(s)),len(EXCEPTION_RE.findall(s)),
            s.count('='),s.count(':'),sum(line.startswith(' ') for line in ls),
            len(re.findall(r'\b(?:PASS|FAIL|accepted|completed)\b',s,re.I))]

def lexical_cv(xs,ys,seeds):
    results=[]
    for seed in range(seeds):
        cv=StratifiedKFold(5,shuffle=True,random_state=202600+seed)
        pipeline=make_pipeline(TfidfVectorizer(token_pattern=r'(?u)\b[\w][\w.\-]+\b',ngram_range=(1,2),sublinear_tf=True),LogisticRegression(max_iter=2500))
        pr=cross_val_predict(pipeline,xs,ys,cv=cv)
        results.append(float(accuracy_score(ys,pr)))
    return {'n_seeds':seeds,'all_scores':[round(x,4) for x in results],
            'median':round(statistics.median(results),4),'min':round(min(results),4),'max':round(max(results),4)}

def structure_cv(xs,ys,seeds):
    xx=np.array([shape_row(t) for t in xs],dtype=float)
    results=[]
    for seed in range(seeds):
        cv=StratifiedKFold(5,shuffle=True,random_state=202600+seed)
        pr=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2500)),xx,ys,cv=cv)
        results.append(float(accuracy_score(ys,pr)))
    return {'n_seeds':seeds,'all_scores':[round(x,4) for x in results],
            'median':round(statistics.median(results),4),'min':round(min(results),4),'max':round(max(results),4)}

def validate(rows,prompt):
    errors=[]; warnings=[]
    ids=[r.get('id') for r in rows]
    count=collections.Counter(r.get('label') for r in rows)
    if len(rows)!=60: errors.append(f'Expected 60 full rows, got {len(rows)}')
    if len(set(ids))!=len(ids): errors.append('duplicate full IDs')
    if not all(isinstance(i,str) and re.fullmatch(r'T\d{3}',i) for i in ids): errors.append('non-neutral/invalid ID')
    if count!={x:15 for x in LABELS}: errors.append(f'wrong label counts: {dict(count)}')
    pairs=collections.Counter((r.get('label'),r.get('primary_distractor')) for r in rows)
    expected={(a,b):5 for a in LABELS for b in LABELS if a!=b}
    if pairs!=expected: errors.append('not exactly 5 in each of 12 directed boundaries')
    if len(prompt)!=len(rows): errors.append('full/prompt row count mismatch')
    fullmap={r['id']:r for r in rows if 'id' in r}
    pmap={r['id']:r for r in prompt if 'id' in r}
    if set(pmap)!=set(fullmap): errors.append('prompt/full IDs not identical')
    for id,p in pmap.items():
        if set(p)!={'id','error_text'}: errors.append(f'{id}: prompt keys {list(p)} are not id/error_text')
        if id in fullmap and p.get('error_text')!=fullmap[id].get('error_text'): errors.append(f'{id}: prompt text differs from full')
    if any(r.get('source_type')!='SYNTHETIC' for r in rows): errors.append('non-synthetic source entry: verify before claims')
    for r in rows:
        if not isinstance(r.get('error_text'),str) or not r['error_text'].strip(): errors.append(f"{r.get('id')}: empty error_text")
        if MASK_TERMS.search(r['error_text']): warnings.append(f"{r['id']}: raw full label string embedded in error_text")
        if r.get('review_flags'): warnings.append(f"{r['id']}: unresolved flags {r['review_flags']}")
    if sum(bool(r.get('surface_exception')) for r in rows)<len(rows)//2:
        warnings.append('Most surface_exception developer metadata are empty: auto-extract exceptions, do not trust field')
    if sum(bool(r.get('surface_cues')) for r in rows)<len(rows)//2:
        warnings.append('Most surface_cues developer metadata are empty: no claim of completed cue metadata')
    return errors,warnings,pairs,count

def pair_name(r):
    return '<->'.join(sorted([r['label'],r['primary_distractor']]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--model-input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seeds',type=int,default=10)
    args=ap.parse_args()
    if args.seeds<1 or args.seeds>50: raise ValueError('--seeds in [1,50]')
    args.out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(args.dataset);prompt=read_jsonl(args.model_input)
    errors,warnings,pairs,counts=validate(rows,prompt)
    if errors:
        (args.out/'audit.json').write_text(json.dumps({'status':'BLOCKED','errors':errors,'warnings':warnings},indent=2,ensure_ascii=False),encoding='utf-8')
        raise SystemExit('Structural validation FAILED: '+str(errors))
    y=np.array([r['label'] for r in rows]); ids=[r['id'] for r in rows]; texts=[r['error_text'] for r in rows]
    datasets={'full_text_cleaned':[normalize_text(s) for s in texts],
              'field_keys_only':[only_keys(s) for s in texts],
              'exceptions_only':[only_exceptions(s) for s in texts],
              'first_two_lines':[normalize_text(' '.join(s.splitlines()[:2])) for s in texts]}
    cv={name:lexical_cv(xs,y,args.seeds) for name,xs in datasets.items()}
    cv['shape_only']=structure_cv(texts,y,args.seeds)
    # One leave-one-bidirectional-boundary-out check, diagnostic of domain transfer ONLY.
    pairgroups=np.array([pair_name(r) for r in rows]); group_out=[]
    for ixtrain,ixtest in LeaveOneGroupOut().split(texts,y,groups=pairgroups):
        model=make_pipeline(TfidfVectorizer(token_pattern=r'(?u)\b[\w][\w.\-]+\b',ngram_range=(1,2)),LogisticRegression(max_iter=2500))
        xs=datasets['full_text_cleaned'];model.fit([xs[k] for k in ixtrain],y[ixtrain]);yp=model.predict([xs[k] for k in ixtest])
        group_out.append({'boundary':str(pairgroups[ixtest[0]]),'n_test':len(ixtest),
                          'test_label_set':sorted(set(y[ixtest])),
                          'accuracy':round(float(accuracy_score(y[ixtest],yp)),3),
                          'predicted_labels':dict(collections.Counter(yp.tolist()))})
    # Words and bigrams present in >=3 rows, descriptive associations. No p-values with this n.
    v=CountVectorizer(token_pattern=r'(?u)\b[a-z][a-z_]+\b',binary=True,min_df=3,ngram_range=(1,2))
    X=v.fit_transform(datasets['full_text_cleaned']).toarray().astype(bool)
    freq=[]
    for j,token in enumerate(v.get_feature_names_out()):
        if 'idval' in token.split():
            continue  # mask token is not a literal source token; do not report synthetic cleanup artifacts
        by={cl:int(np.sum(X[y==cl,j])) for cl in LABELS}; mx=max(by.values()); total=sum(by.values())
        if mx>=3 and mx-min(by.values())>=3:
            freq.append({'term':token,'n_samples':total,'by_class':by,'dominant_share':round(mx/total,3),'range':mx-min(by.values())})
    freq.sort(key=lambda r:(-r['dominant_share'],-r['n_samples'], -r['range']))
    obvious=[r for r in freq if r['dominant_share']==1 and r['n_samples']>=3]
    # Duplicates and close paraphrases (suggestions, not auto-rejections).
    vect=TfidfVectorizer(token_pattern=r'(?u)\b[\w][\w.\-]+\b',ngram_range=(1,2))
    mat=vect.fit_transform(datasets['full_text_cleaned']); sim=(mat @ mat.T).toarray();
    close=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            if sim[i,j]>=0.35:
                close.append({'id_a':ids[i],'design_a':rows[i].get('design_id'),
                              'label_a':str(y[i]),'id_b':ids[j],'design_b':rows[j].get('design_id'),
                              'label_b':str(y[j]),'cosine':round(float(sim[i,j]),3),
                              'opposite_gt':bool(y[i]!=y[j])})
    close.sort(key=lambda x:-x['cosine'])
    # Label/format counts used for review, not used as model features.
    formats={cl:dict(collections.Counter(r.get('surface_format','') for r in rows if r['label']==cl)) for cl in LABELS}
    lengths={cl:{'min':min(len(t) for t,r in zip(texts,rows) if r['label']==cl),
                 'median':statistics.median(len(t) for t,r in zip(texts,rows) if r['label']==cl),
                 'max':max(len(t) for t,r in zip(texts,rows) if r['label']==cl)} for cl in LABELS}
    exception_counts={e:dict(collections.Counter(r['label'] for r in rows if e in EXCEPTION_RE.findall(r['error_text'])))
                      for e in sorted({v for t in texts for v in EXCEPTION_RE.findall(t)})}
    sample_flags=[]
    for r in rows:
        s=r['error_text'].lower()
        terms=[it['term'] for it in obvious if it['term'] in s and it['by_class'][r['label']]>0]
        sample_flags.append({'id':r['id'],'design_id':r.get('design_id'),'label':r['label'],
                             'existing_review_flags':r.get('review_flags',[]),
                             'class_exclusive_repeated_terms':terms[:12],
                             'exception_names':EXCEPTION_RE.findall(r['error_text']),
                             'n_chars':len(r['error_text']),'n_lines':len(r['error_text'].splitlines())})
    output={'status':'SCREENING_COMPLETED_NOT_ACCEPTANCE','input_sha256':{'dataset':hashfile(args.dataset),'model_input':hashfile(args.model_input)},
            'n':len(rows),'label_counts':dict(counts),'directed_pair_counts':{a+'->'+b:n for (a,b),n in pairs.items()},
            'structural_errors':errors,'warnings':warnings,'surface_baselines_repeated_stratified_5fold':cv,
            'leave_one_bidirectional_boundary_out_domain_shift_only':group_out,
            'exclusive_repeated_terms':obvious[:80],'all_high_association_terms':freq[:120],
            'top_near_neighbors':close[:35], 'format_by_gt':formats, 'length_chars':lengths,
            'exceptions_by_gt':exception_counts, 'samples':sample_flags,
            'cautions':['Synthetic development candidates; no evidence of real-world generalization or difficulty.',
                         'Repeated CV min/max are split sensitivity, not statistical confidence intervals.',
                         'Leave-boundary-out measures domain shift and should NOT be compared directly to simple chance as leakage proof.',
                         'No result can establish unique GT, independent Challenge difficulty, or absence of semantic shortcuts.',
                         'Do not revise GT or repeatedly optimize surface text against these baselines.']}
    (args.out/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# Challenge Set：第三步程序化全局检查','',
        '**性质：合成 Test 候选的开发期筛查；不调用任何正式模型，不替代独立盲审。**','',
        f'- 输入 SHA256：`{output["input_sha256"]["dataset"]}`',
        f'- 结构：{len(rows)} 条；四类各 15；12 个定向边界各 5；中性 ID 和模型输入隔离通过。',
        '- 运行状态：结构检查通过；质量验收未通过（仍需处理下述风险与独立审阅）。','',
        '## 重复分层 5 折浅层筛查','',
        '| 特征 | 中位准确率 | 分割敏感范围 |','|---|---:|---:|']
    for key,c in cv.items():md.append(f'| {key} | {c["median"]:.1%} | {c["min"]:.1%}～{c["max"]:.1%} |')
    md+=['', '参考：四类均衡，均匀随机猜测期望准确率为 25%，但高于 25% 不自动等于不良泄漏。','',
         '## 文本长度','', '| GT | 最短 | 中位数 | 最长 |','|---|---:|---:|---:|']
    for cl,item in lengths.items():md.append(f'| {cl} | {item["min"]} | {item["median"]} | {item["max"]} |')
    md+=['','## 典型跨类词汇关联（按重复专属项优先，不作显著性声明）','',
         '| 词/词组 | 样本数 | 仅出现在哪类 |','|---|---:|---|']
    for item in obvious[:16]:
        cl=next(k for k,v in item['by_class'].items() if v>0)
        md.append(f'| `{item["term"]}` | {item["n_samples"]} | {cl} |')
    md+=['','这些词可能是必要诊断证据，也可能是人工模板；须逐例复核，不得一律删改。',
         '','## leave-one-boundary-out（只作跨主题迁移诊断）','',
         '| 留出边界 | 测试类别 | N | 准确率 |','|---|---|---:|---:|']
    for item in group_out:md.append(f'| {item["boundary"]} | {" / ".join(item["test_label_set"])} | {item["n_test"]} | {item["accuracy"]:.1%} |')
    md+=['','不能将这组成绩直接解释为模型因果推理能力、随机基准比较或无泄漏证据。',
         '','## 推荐优先核对的样本','']
    high_priority=[s for s in sample_flags if s['existing_review_flags'] or len(s['class_exclusive_repeated_terms'])>=3]
    for s in high_priority[:24]:
        if s['existing_review_flags']:reason='existing='+','.join(s['existing_review_flags'])
        else:reason='exclusive terms='+','.join(s['class_exclusive_repeated_terms'][:3])
        md.append(f'- {s["id"]}（{s["design_id"]}）：{reason}')
    md+=['','## 元数据与运行边界','']
    for w in warnings:md.append('- '+w)
    md+=['','## 正确的后续决策','',
         '1. 将上述高关联线索逐条核对成“必要因果证据”或“可消除的人为模板”；一次修订后存新版本，保留原始版本和 SHA。',
         '2. 针对三条容量关键样本及镜像题实施独立 Challenge 盲审；程序化检查不能替代它。',
         '3. 如果修订文本，重新导出仅含 id/error_text 的模型输入，再用同一个固定脚本运行一次，报告变化与局限。',
         '4. 未冻结正式 Test，不运行五个正式实验模型、不修改 GT。']
    (args.out/'audit.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('STRUCTURAL PASS',len(rows),'4x15','12x5','model_input_match')
    print('CV',[(k,v['median'],v['min'],v['max']) for k,v in cv.items()])
    print('EXCLUSIVE_TERMS',[(it['term'],it['n_samples'],it['by_class']) for it in obvious[:12]])
    print('TOP_SIM',close[:3]);print('REVIEW_FLAGS',[(s['id'],s['design_id'],s['existing_review_flags']) for s in sample_flags if s['existing_review_flags']]);
    print('REPORT',args.out/'audit.md')

if __name__=='__main__':main()
