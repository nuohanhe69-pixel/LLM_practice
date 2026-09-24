# Challenge Set：第三步程序化全局检查

**性质：合成 Test 候选的开发期筛查；不调用任何正式模型，不替代独立盲审。**

- 输入 SHA256：`4e2efad0c333f320e9fc69b90b1c87458666d7c5a4c54d0a74a037d8d3a3e4ea`
- 结构：60 条；四类各 15；12 个定向边界各 5；中性 ID 和模型输入隔离通过。
- 运行状态：结构检查通过；质量验收未通过（仍需处理下述风险与独立审阅）。

## 重复分层 5 折浅层筛查

| 特征 | 中位准确率 | 分割敏感范围 |
|---|---:|---:|
| full_text_cleaned | 59.2% | 56.7%～61.7% |
| field_keys_only | 53.3% | 46.7%～58.3% |
| exceptions_only | 25.0% | 18.3%～28.3% |
| first_two_lines | 33.3% | 28.3%～40.0% |
| shape_only | 38.3% | 23.3%～43.3% |

参考：四类均衡，均匀随机猜测期望准确率为 25%，但高于 25% 不自动等于不良泄漏。

## 文本长度

| GT | 最短 | 中位数 | 最长 |
|---|---:|---:|---:|
| NETWORK_API | 354 | 578 | 716 |
| CONTEXT_LIMIT | 568 | 685 | 1004 |
| ENV_DEPENDENCY | 499 | 658 | 953 |
| CODE_RUNTIME | 487 | 609 | 751 |

## 典型跨类词汇关联（按重复专属项优先，不作显著性声明）

| 词/词组 | 样本数 | 仅出现在哪类 |
|---|---:|---|
| `capacity` | 7 | CONTEXT_LIMIT |
| `component` | 5 | ENV_DEPENDENCY |
| `limit_entry` | 5 | CONTEXT_LIMIT |
| `contextlimiterror` | 4 | CODE_RUNTIME |
| `history_summary` | 4 | CONTEXT_LIMIT |
| `provider request` | 4 | NETWORK_API |
| `total_sequence_capacity` | 4 | CONTEXT_LIMIT |
| `worker none` | 4 | CONTEXT_LIMIT |
| `assembled_units` | 3 | CONTEXT_LIMIT |
| `close_reason` | 3 | NETWORK_API |
| `compact` | 3 | CONTEXT_LIMIT |
| `complete integration` | 3 | ENV_DEPENDENCY |
| `component contract` | 3 | ENV_DEPENDENCY |
| `contextlimiterror request` | 3 | CODE_RUNTIME |
| `dispatch` | 3 | ENV_DEPENDENCY |
| `expects` | 3 | ENV_DEPENDENCY |

这些词可能是必要诊断证据，也可能是人工模板；须逐例复核，不得一律删改。

## leave-one-boundary-out（只作跨主题迁移诊断）

| 留出边界 | 测试类别 | N | 准确率 |
|---|---|---:|---:|
| CODE_RUNTIME<->CONTEXT_LIMIT | CODE_RUNTIME / CONTEXT_LIMIT | 10 | 0.0% |
| CODE_RUNTIME<->ENV_DEPENDENCY | CODE_RUNTIME / ENV_DEPENDENCY | 10 | 0.0% |
| CODE_RUNTIME<->NETWORK_API | CODE_RUNTIME / NETWORK_API | 10 | 0.0% |
| CONTEXT_LIMIT<->ENV_DEPENDENCY | CONTEXT_LIMIT / ENV_DEPENDENCY | 10 | 0.0% |
| CONTEXT_LIMIT<->NETWORK_API | CONTEXT_LIMIT / NETWORK_API | 10 | 10.0% |
| ENV_DEPENDENCY<->NETWORK_API | ENV_DEPENDENCY / NETWORK_API | 10 | 0.0% |

不能将这组成绩直接解释为模型因果推理能力、随机基准比较或无泄漏证据。

## 推荐优先核对的样本

- T004（C-E-05）：exclusive terms=capacity,compact,request_parsed
- T005（R-C-05）：exclusive terms=contextlimiterror,contextlimiterror request,provider_request_id
- T006（C-E-03）：exclusive terms=capacity,limit_entry,assembled_units
- T007（R-E-03）：existing=check_single_line_shortcut
- T010（R-E-04）：existing=check_pair_single_line_shortcut
- T011（C-N-03）：existing=challenge_strength_independent_review_pending
- T015（C-E-02）：exclusive terms=capacity,failed_before_inference,request_parsed
- T017（C-C-02）：exclusive terms=capacity,limit_entry,history_summary
- T018（C-C-03）：exclusive terms=capacity,limit_entry,total_sequence_capacity
- T021（C-E-01）：exclusive terms=capacity,assembled_units,compact
- T022（C-N-04）：exclusive terms=capacity,compact,unrelated
- T023（C-C-01）：exclusive terms=capacity,limit_entry,history_summary
- T025（C-E-04）：exclusive terms=capacity,limit_entry,failed_before_inference
- T028（C-C-04）：exclusive terms=capacity,history_summary,assembled_units
- T029（R-E-05）：existing=check_single_line_shortcut
- T041（R-C-02）：exclusive terms=contextlimiterror,contextlimiterror request,provider_request_id
- T042（C-N-05）：existing=challenge_strength_independent_review_pending
- T043（E-R-02）：existing=check_pair_single_line_shortcut
- T044（C-N-02）：existing=challenge_strength_independent_review_pending
- T048（E-C-02）：exclusive terms=component,component contract,expects
- T051（C-N-01）：exclusive terms=capacity,total_sequence_capacity,compact
- T053（E-C-01）：exclusive terms=component,dispatch,expects

## 元数据与运行边界

- T007: unresolved flags ['check_single_line_shortcut']
- T010: unresolved flags ['check_pair_single_line_shortcut']
- T011: unresolved flags ['challenge_strength_independent_review_pending']
- T029: unresolved flags ['check_single_line_shortcut']
- T042: unresolved flags ['challenge_strength_independent_review_pending']
- T043: unresolved flags ['check_pair_single_line_shortcut']
- T044: unresolved flags ['challenge_strength_independent_review_pending']
- Most surface_exception developer metadata are empty: auto-extract exceptions, do not trust field
- Most surface_cues developer metadata are empty: no claim of completed cue metadata

## 正确的后续决策

1. 将上述高关联线索逐条核对成“必要因果证据”或“可消除的人为模板”；一次修订后存新版本，保留原始版本和 SHA。
2. 针对三条容量关键样本及镜像题实施独立 Challenge 盲审；程序化检查不能替代它。
3. 如果修订文本，重新导出仅含 id/error_text 的模型输入，再用同一个固定脚本运行一次，报告变化与局限。
4. 未冻结正式 Test，不运行五个正式实验模型、不修改 GT。
