# Challenge Set：第三步程序化全局检查

**性质：合成 Test 候选的开发期筛查；不调用任何正式模型，不替代独立盲审。**

- 输入 SHA256：`da66919aa555134ac69e6bce54ddaa9c82b72aeaa297bd0e15bab126060b6544`
- 结构：60 条；四类各 15；12 个定向边界各 5；中性 ID 和模型输入隔离通过。
- 运行状态：结构检查通过；质量验收未通过（仍需处理下述风险与独立审阅）。

## 重复分层 5 折浅层筛查

| 特征 | 中位准确率 | 分割敏感范围 |
|---|---:|---:|
| full_text_cleaned | 58.3% | 50.0%～60.0% |
| field_keys_only | 50.8% | 45.0%～60.0% |
| exceptions_only | 20.0% | 15.0%～23.3% |
| first_two_lines | 34.2% | 30.0%～40.0% |
| shape_only | 30.8% | 18.3%～36.7% |

参考：四类均衡，均匀随机猜测期望准确率为 25%，但高于 25% 不自动等于不良泄漏。

## 文本长度

| GT | 最短 | 中位数 | 最长 |
|---|---:|---:|---:|
| NETWORK_API | 354 | 578 | 1045 |
| CONTEXT_LIMIT | 574 | 685 | 1004 |
| ENV_DEPENDENCY | 499 | 680 | 951 |
| CODE_RUNTIME | 487 | 618 | 751 |

## 典型跨类词汇关联（按重复专属项优先，不作显著性声明）

| 词/词组 | 样本数 | 仅出现在哪类 |
|---|---:|---|
| `capacity` | 7 | CONTEXT_LIMIT |
| `provider request` | 4 | NETWORK_API |
| `total_sequence_capacity` | 4 | CONTEXT_LIMIT |
| `worker none` | 4 | CONTEXT_LIMIT |
| `assembled_units` | 3 | CONTEXT_LIMIT |
| `available` | 3 | CODE_RUNTIME |
| `close_reason` | 3 | NETWORK_API |
| `compact` | 3 | CONTEXT_LIMIT |
| `complete integration` | 3 | ENV_DEPENDENCY |
| `dispatch` | 3 | ENV_DEPENDENCY |
| `failed_before_inference` | 3 | CONTEXT_LIMIT |
| `generation_reservation` | 3 | CONTEXT_LIMIT |
| `none provider` | 3 | CONTEXT_LIMIT |
| `provider diagnostic` | 3 | ENV_DEPENDENCY |
| `provider_request_id` | 3 | CODE_RUNTIME |
| `provider_request_id none` | 3 | CODE_RUNTIME |

这些词可能是必要诊断证据，也可能是人工模板；须逐例复核，不得一律删改。

## leave-one-boundary-out（只作跨主题迁移诊断）

| 留出边界 | 测试类别 | N | 准确率 |
|---|---|---:|---:|
| CODE_RUNTIME<->CONTEXT_LIMIT | CODE_RUNTIME / CONTEXT_LIMIT | 10 | 0.0% |
| CODE_RUNTIME<->ENV_DEPENDENCY | CODE_RUNTIME / ENV_DEPENDENCY | 10 | 0.0% |
| CODE_RUNTIME<->NETWORK_API | CODE_RUNTIME / NETWORK_API | 10 | 0.0% |
| CONTEXT_LIMIT<->ENV_DEPENDENCY | CONTEXT_LIMIT / ENV_DEPENDENCY | 10 | 0.0% |
| CONTEXT_LIMIT<->NETWORK_API | CONTEXT_LIMIT / NETWORK_API | 10 | 10.0% |
| ENV_DEPENDENCY<->NETWORK_API | ENV_DEPENDENCY / NETWORK_API | 10 | 10.0% |

不能将这组成绩直接解释为模型因果推理能力、随机基准比较或无泄漏证据。

## 推荐优先核对的样本

- T004（C-E-05）：exclusive terms=capacity,compact,request_parsed
- T006（C-E-03）：exclusive terms=capacity,assembled_units,failed_before_inference
- T011（C-N-03）：existing=challenge_strength_independent_review_pending
- T015（C-E-02）：exclusive terms=capacity,failed_before_inference,request_parsed
- T018（C-C-03）：exclusive terms=capacity,total_sequence_capacity,generation_reservation
- T021（C-E-01）：exclusive terms=capacity,assembled_units,compact
- T022（C-N-04）：exclusive terms=capacity,compact,unrelated
- T025（C-E-04）：exclusive terms=capacity,failed_before_inference,response_schema
- T028（C-C-04）：exclusive terms=capacity,assembled_units,compact
- T042（C-N-05）：existing=challenge_strength_independent_review_pending
- T044（C-N-02）：existing=challenge_strength_independent_review_pending
- T051（C-N-01）：exclusive terms=capacity,total_sequence_capacity,compact

## 元数据与运行边界

- T011: unresolved flags ['challenge_strength_independent_review_pending']
- T042: unresolved flags ['challenge_strength_independent_review_pending']
- T044: unresolved flags ['challenge_strength_independent_review_pending']
- Most surface_exception developer metadata are empty: auto-extract exceptions, do not trust field
- Most surface_cues developer metadata are empty: no claim of completed cue metadata

## 正确的后续决策

1. 将上述高关联线索逐条核对成“必要因果证据”或“可消除的人为模板”；一次修订后存新版本，保留原始版本和 SHA。
2. 针对三条容量关键样本及镜像题实施独立 Challenge 盲审；程序化检查不能替代它。
3. 如果修订文本，重新导出仅含 id/error_text 的模型输入，再用同一个固定脚本运行一次，报告变化与局限。
4. 未冻结正式 Test，不运行五个正式实验模型、不修改 GT。
