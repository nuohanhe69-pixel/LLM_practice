#!/usr/bin/env python3
"""Build internal review report, anonymized independent-review packet and integrity tests."""
import collections
import csv
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path

ROOT=Path(__file__).parent
SOURCE=ROOT/'baseline'
old=[json.loads(x) for x in (SOURCE/'candidates_full.jsonl').read_text(encoding='utf-8').splitlines() if x]
new=[json.loads(x) for x in (ROOT/'candidates_full_v3.jsonl').read_text(encoding='utf-8').splitlines() if x]
a={x['id']:x for x in old};b={x['id']:x for x in new}
p=json.loads((ROOT/'audit.json').read_text());q=json.loads((SOURCE/'audit.json').read_text())

BULK=['capacity','limit_entry','component','contextlimiterror','history_summary']
def countterm(rows,term):
 pat=re.compile(r'(?i)\b'+re.escape(term)+r'\b')
 return sum(bool(pat.search(x['error_text'])) for x in rows)
def sim(a,b,ids=('T014','T049')):
 from sklearn.feature_extraction.text import TfidfVectorizer
 from challenge_shortcut_audit import normalize_text
 texts=[normalize_text(a[k]['error_text']) for k in ids]
 x=TfidfVectorizer(token_pattern=r'(?u)\b[\w][\w.\-]+\b',ngram_range=(1,2)).fit_transform(texts)
 return float((x@x.T).toarray()[0,1])
# In full corpus similarity, match old script corpus fitting, not just pair fitting.
def fullsim(rows,ids):
 from sklearn.feature_extraction.text import TfidfVectorizer
 from challenge_shortcut_audit import normalize_text
 t=[normalize_text(r['error_text']) for r in rows]
 x=TfidfVectorizer(token_pattern=r'(?u)\b[\w][\w.\-]+\b',ngram_range=(1,2)).fit_transform(t)
 idx={r['id']:i for i,r in enumerate(rows)}
 return float((x@x.T)[idx[ids[0]],idx[ids[1]]])

metrics=['full_text_cleaned','field_keys_only','exceptions_only','first_two_lines','shape_only']
fmt='| 指标 | 修订前 | 修订后 |\n|---|---:|---:|\n'
for m in metrics:
 fmt+=f'| {m} | {q["surface_baselines_repeated_stratified_5fold"][m]["median"]:.1%} | {p["surface_baselines_repeated_stratified_5fold"][m]["median"]:.1%} |\n'
terms='| 词汇 | 原含词样本数 | 现含词样本数 | 决策 |\n|---|---:|---:|---|\n'
notes={
 'capacity':'数值/模型窗口证据与容量故障直接相关：保留，后续审查其日志真实性，不为平衡词频删证据。',
 'limit_entry':'通用人工字段名重复：替换为对应日志内的上下文引用。',
 'component':'用于直指依赖契约的人为总结性标题：移除，保留双方 manifest 事实。',
 'contextlimiterror':'四条 CODE 过度使用相同外层异常：仅对两条应用本地预算异常更换自然包装，另两条作为干扰保留。',
 'history_summary':'与多道容量样本绑定的同质字段名：按工件性质区分。',
}
for term in BULK:
 terms+=f'| `{term}` | {countterm(old,term)} | {countterm(new,term)} | {notes[term]} |\n'
report=f'''# 三个问题专项处理：一次性定向修订 V3

**资料性质：60 条合成开发期候选，非正式 Test，未调用大模型、未动原仓库/Prompt/GT。**

旧文件 SHA256：`{hashlib.sha256((SOURCE/'candidates_full.jsonl').read_bytes()).hexdigest()}`  
新文件 SHA256：`{hashlib.sha256((ROOT/'candidates_full_v3.jsonl').read_bytes()).hexdigest()}`

## 1. 词汇关联：把必要证据与人工写作痕迹分开

{terms}

不以 TF-IDF 降分为修改目标；仅处理人为标签化标题、重复模板和 CODE 过度统一的异常包装。`capacity`、`total_sequence_capacity` 这类字段仍可能与真容量故障关联，因为真实原因需要相关数字证据。它们仍需全局人工判断，**不能宣称不存在任何可利用词汇特征**。

修订前/后使用**同一脚本、10 种相同分层五折划分、同一特征定义**执行一次对照：

{fmt}

受样本数（60）限制，这些分数描述本次有限样本的表面可预测性，不是修订有效性的因果估计，也不设必须降到 25% 的目标。

## 2. T014 / T049：真正消除机制重复，而非改写同义句

- T014 (`N-R-05`) 保留：Provider 已生成、远端传输中途停止，客户端超时后正常取消；真正要排除的是本地 callback/cancellation 代码错误。
- T049 (`N-C-02`) 原机制与 T014 同为 upstream 持续产生 + downstream stall，**撤销原机制并明确替换**为远程网关的**租户加权调度槽满载、长请求排队至客户端超时**。模型的单次序列窗口检查不阻止请求；窗口之后恢复时，未压缩的相同请求可成功。需区分远端服务并发/调度可用性与单请求 context window。
- 两者都属于 NETWORK，但根因层级分别为**响应传输**与**服务排队调度**；属于为已证实重复作出的例外机制变更，已留下差异日志，不偷换 GT/干扰边界。

相同全量语料 TF-IDF 余弦：`{fullsim(old,('T014','T049')):.3f}` → `{fullsim(new,('T014','T049')):.3f}`。词面降低只能作辅助；主要验收依据是不同的故障阶段和判别证据。

## 3. 七个待审标记：四条内部修订，三条外部独立判断仍需要完成

| ID | 类型 | 本轮结论 | 证据 |
|---|---|---|---|
| T007 / R-E-03 | 单行 shortcut | 内部修订完成；待最终盲审 | version guard 使用的元数据、实际模块能力和修复后回放拆为不同记录 |
| T029 / R-E-05 | 单行 shortcut | 内部修订完成；待最终盲审 | 选中匿名 artifact、arch 对应安装清单、同机可加载正确 artifact 需要交叉连接 |
| T010 / R-E-04 | 成对 shortcut | 内部修订完成；待最终盲审 | app wrapper 参数 provenance 对照未改的 library fixture |
| T043 / E-R-02 | 成对 shortcut | 内部修订完成；待最终盲审 | 两组件编码 manifest 与不同配对的调用结果，未直接命名“不兼容” |
| T011 / C-N-03 | Challenge 难度 | **等待两位独立审阅者** | 4 图、多模态联合序列窗口，必须核验是否仍被数值单步计算捷径解决 |
| T042 / C-N-05 | Challenge 难度 | **等待两位独立审阅者** | wire-stream cap 与 decoded-section cap 两项约束，需看是否只需单次数字比较 |
| T044 / C-N-02 | Challenge 难度 | **等待两位独立审阅者** | wire 体积与 decoded-request 作用域，需核验是否满足 Challenge 门槛 |

三条容量镜像 T003/T026/T024 同时进入无 GT 的独立盲审。不存在未完成盲审却视为已通过的自动途径；审阅表已附。

验收规则：两位未参与样本设计的审阅者各自填写 GT、关键证据、竞争根因排除及难度。任何一位认为 GT 不唯一，或双方都认为凭单行/一次直观比较即可分类时，该题不可直接冻结；仅人工修订一次后重新审查。不同意见先讨论题面是否充分，**不许根据模型预测修改 GT**。

## 其他未隐瞒的状态

- 结构检查已通过，GT、directed boundary、ID、排序均未变；源资料仍是 SYNTHETIC。
- 现有 `surface_exception` / `surface_cues` 编辑元数据大面积未核对填写，仍属于正式冻结前的单独 TODO；程序抽取值不能伪称人工审核。
- 不用预留正式模型进行循环试题，不启动正式实验。
'''
(ROOT/'三问题_专项处理与验收报告.md').write_text(report,encoding='utf-8')

# Build genuinely blind packet: never include full rows / mapping / internal report in the reviewer ZIP.
blind=ROOT/'仅发审阅者_匿名容量镜像盲审包.zip'
challenge=['T011','T042','T044'];mirror=['T003','T026','T024'];keys=challenge+mirror
# fixed anonymized IDs assigned in mixed order, not original T IDs nor original design IDs
random.Random(20260923).shuffle(keys)
mapping={k:f'CASE-{i+1:02d}' for i,k in enumerate(keys)}
content=['# 6 条 Agent 故障边界独立盲审（无答案）','',
'这是合成开发调试文本，不代表真实生产数据。请不看作者资料，独立完成后再交回；不要相互讨论或使用原始内部答案文件。',
'判断主要根因类别：NETWORK_API / CONTEXT_LIMIT / ENV_DEPENDENCY / CODE_RUNTIME。按第一因而非最外层异常分类。',
'对每题提供：标签、最强干扰类别、支持 GT 的至少两处原始证据、为什么排除干扰类别、是否有任何单行捷径、难度分级（Easy / Medium / Challenge / Ambiguous）。','']
for k in keys:
 content+=['## '+mapping[k],'','```text',b[k]['error_text'],'```','']
blind_text='\n'.join(content)
form=io.StringIO();w=csv.writer(form)
w.writerow(['case_id','pred_label','primary_competitor','gt_unique(Y/N/U)','decisive_evidence_quotes_at_least_two','why_not_competitor','single_line_shortcut(Y/N/U)','level(Easy/Medium/Challenge/Ambiguous)','notes'])
for k in keys:w.writerow([mapping[k],'','','','','','','',''])
with zipfile.ZipFile(blind,'w',compression=zipfile.ZIP_DEFLATED) as z:
 z.writestr('独立盲审/6条匿名题面.md',blind_text)
 z.writestr('独立盲审/独立填写表.csv',form.getvalue())
 z.writestr('独立盲审/给审阅者的说明.txt','仅可依据匿名题面独立分类与评估。请不要阅读设计者内部包/答案映射。两个审阅者分别提交原样填写表。此包没有 GT 或设计 ID。\n')
private=io.StringIO();w=csv.writer(private);w.writerow(['original_id','blind_id','design_id','gt','distractor','external_review_required'])
for k in keys:w.writerow([k,mapping[k],b[k]['design_id'],b[k]['label'],b[k]['primary_distractor'],k in challenge])
(ROOT/'内部_六条盲审ID映射.csv').write_text(private.getvalue(),encoding='utf-8-sig')
# explicit privacy assertions
with zipfile.ZipFile(blind) as z:
 assert sorted(z.namelist())==sorted(['独立盲审/6条匿名题面.md','独立盲审/独立填写表.csv','独立盲审/给审阅者的说明.txt'])
 assert 'T011' not in blind_text and 'T042' not in blind_text and 'T044' not in blind_text
 assert not any(s in blind_text for s in ['GT:','design_id','primary_distractor','review_flags'])
print('built blinded reviewer packet:',blind.name,'mapping private; 6 anonymous cases.')
print('full-corpus similarity pre and post:',round(fullsim(old,('T014','T049')),3),round(fullsim(new,('T014','T049')),3))
print('term counts pre/post',[(t,countterm(old,t),countterm(new,t)) for t in BULK])
