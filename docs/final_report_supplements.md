# 结课报告补充材料：完整 Prompt、错误案例与改进方案

本材料补充[课程研究报告](final_research_report.md)。完整 Prompt 由当前仓库已有渲染函数离线还原，案例预测来自正式 Run1、Run2 的原始 CSV；不发送 API 请求。第 3 节为**待验证方案**，不代表已经修改冻结 Prompt、示例或得到新的实验结果。

## 1. 完整 Prompt 与实际 API 请求结构

### 1.1 模板、渲染正文与请求的区别

仓库 `prompts/` 共四个文件：[标签定义](../prompts/label_definitions.txt)、[Zero-shot 模板](../prompts/zero_shot.txt)、[Few-shot 模板](../prompts/few_shot.txt)、[Demo 模板](../prompts/demo_example.txt)。[render_prompt()](../src/llm_experiment/prompts.py)读取模板并去除模板首尾空白，以数据集中的原始错误文本替换占位符；Few-shot 按数据集顺序插入 D001–D008，每条含错误文本和标签，示例间以一个空行分隔。Demo 的 `reason` 与来源不进入正文；Test 的 `label` 等内部字段不进入正文。

以下两种条件统一使用 **T019** 作为代表性 Test。两个代码块分别是实际实现渲染后的**完整 user 消息正文**，包含所有标签定义，Few-shot 另包含全部 8 条 Demo。代码块中的输入不省略、不润色；样本 ID 用于数据配对，但实现没有把 Test ID 写入 Prompt。

### 1.2 Zero-shot 完整渲染文本（T019）

```text
你是错误类型分类器。请严格依据下面的标签定义进行分类。
按主要根因分类，而不是按报错的表面形式分类。

NETWORK_API：网络连接、远程 API/服务通信、网关/端点可用性、连接中断或超时。
CONTEXT_LIMIT：请求体大小、输入/Token 长度、序列长度、上下文窗口容量或明确的输入/请求大小上限。
ENV_DEPENDENCY：缺少包/模块/系统库、依赖版本不兼容、解释器/运行时版本不匹配或依赖解析失败。
CODE_RUNTIME：在运行环境已可用的前提下，由程序自身运行逻辑、参数或值处理导致的运行时错误。

待分类错误：
req=req_812 timeout_deadline=17:22:14.500
provider final_event@17:22:14.461 completion=finished
edge final_event forwarded@17:22:14.468 close=normal
client transport eof=true@17:22:14.472 response_bytes=9211 sha256=27b4...
app response_future_callback_queued@17:22:14.473; timeout_queued@.500; timeout entered@.501 task_cancelled@.502; response callback@.504 completion_discarded=cancelled
TimeoutError request exceeded deadline
scheduler patch trace: response callback processed before timer, task_state=completed timer_skipped=true

只输出一个标签，不要解释。合法标签：NETWORK_API、CONTEXT_LIMIT、ENV_DEPENDENCY、CODE_RUNTIME。
```

### 1.3 Few-shot 完整渲染文本（T019，固定 8 条 Demo）

```text
你是错误类型分类器。请严格依据下面的标签定义和固定示例进行分类。
按主要根因分类，而不是按报错的表面形式分类。

NETWORK_API：网络连接、远程 API/服务通信、网关/端点可用性、连接中断或超时。
CONTEXT_LIMIT：请求体大小、输入/Token 长度、序列长度、上下文窗口容量或明确的输入/请求大小上限。
ENV_DEPENDENCY：缺少包/模块/系统库、依赖版本不兼容、解释器/运行时版本不匹配或依赖解析失败。
CODE_RUNTIME：在运行环境已可用的前提下，由程序自身运行逻辑、参数或值处理导致的运行时错误。

固定示例：
示例 1
错误：requests.exceptions.ConnectionError: Failed to establish a new connection: [Errno 61] Connection refused
标签：NETWORK_API

示例 2
错误：requests.exceptions.Timeout: HTTPConnectionPool(host='api.example.com', port=80): Read timed out.
标签：NETWORK_API

示例 3
错误：HTTP 413 Payload Too Large: request body exceeds the server's maximum accepted size.
标签：CONTEXT_LIMIT

示例 4
错误：Context length exceeded: this model supports at most 8192 tokens, but the request contains 9347 tokens.
标签：CONTEXT_LIMIT

示例 5
错误：ModuleNotFoundError: No module named 'langchain'
标签：ENV_DEPENDENCY

示例 6
错误：ERROR: Cannot install package_a and package_b because these package versions have conflicting dependencies.
标签：ENV_DEPENDENCY

示例 7
错误：TypeError: expected str, got NoneType
标签：CODE_RUNTIME

示例 8
错误：KeyError: 'messages'
标签：CODE_RUNTIME

待分类错误：
req=req_812 timeout_deadline=17:22:14.500
provider final_event@17:22:14.461 completion=finished
edge final_event forwarded@17:22:14.468 close=normal
client transport eof=true@17:22:14.472 response_bytes=9211 sha256=27b4...
app response_future_callback_queued@17:22:14.473; timeout_queued@.500; timeout entered@.501 task_cancelled@.502; response callback@.504 completion_discarded=cancelled
TimeoutError request exceeded deadline
scheduler patch trace: response callback processed before timer, task_state=completed timer_skipped=true

只输出一个标签，不要解释。合法标签：NETWORK_API、CONTEXT_LIMIT、ENV_DEPENDENCY、CODE_RUNTIME。
```

### 1.4 消息角色与完整调用参数

[API client](../src/llm_experiment/api_client.py)将上述全文作为唯一消息的 `content`，实际结构为 `messages=[{"role": "user", "content": prompt}]`。没有额外的 `system` 消息；Demo 也没有被构造成独立的 `user/assistant` 对话。正文中的“你是错误类型分类器”仍属于 user 内容。以正式 Qwen 配置为例，两种 Prompt 仅替换 `prompt` 字符串，SDK 调用字段如下：

```python
# prompt 为 1.2 或 1.3 的完整正文，不是未替换的模板。
client.chat.completions.create(
    model="qwen3.7-plus-2026-05-26",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    extra_body={"enable_thinking": True},
)
```

这是依据代码和冻结数据还原的 SDK 调用，不是从历史网络流量中抓取的请求报文。`extra_body` 是 SDK 的 provider 扩展参数入口；正式配置不传 `max_tokens`、`max_completion_tokens` 或 `reasoning_effort`。客户端另按[配置](../configs/models.json)设置 `timeout=300.0`、`max_retries=2`、百炼兼容端点及凭据；凭据不属于 Prompt，也不在此展示。Kimi 案例使用 `model="kimi-k3"`，其余上述请求协议一致。分类只使用返回的 `message.content`；完整 `reasoning_content` 未持久化，不能据本材料还原模型内部推理。

### 1.5 ENV_DEPENDENCY 的标注口径与定义局限

冻结 Prompt 对 ENV_DEPENDENCY 的原文为：“缺少包/模块/系统库、依赖版本不兼容、解释器/运行时版本不匹配或依赖解析失败。”[Dataset V2](../dataset_v2.csv)的 `label/reason/source_reference` 与[V4 设计元数据](../experiments/challenge_audit_v1/v4/candidates_full_v4.jsonl)表明，实际标注将**依赖组件之间的兼容性**也纳入该类：组件虽然都存在，但接口、字段协议或 Schema 契约无法互相满足，仍属于依赖组合不兼容，而不只是缺少安装包。例如 T055 的 producer 输出别名映射，旧 validator adapter 只读取实体化字段；其 `reason` 明确记为“跨 schema producer 与 validator adapter 的字段表示契约不一致造成后续 ValidationError。”这是该题归为 ENV_DEPENDENCY 的标注依据。

这一区分取决于题面支持的主要故障位置：若证据指向跨组件契约失配，且匹配的组件组合能够完成处理，则按数据集口径归入 ENV_DEPENDENCY；若环境与依赖契约可用，故障来自应用自身参数、状态或值处理，则归入 CODE_RUNTIME。并非所有接口错误或 Schema 校验错误都归入 ENV_DEPENDENCY，最终异常名称本身不足以确定类别。

**研究局限：** 原始 Prompt 提到了“依赖版本不兼容”，但没有显式说明跨组件接口、字段协议及 Schema 契约失配的适用范围，也没有充分展开其与 CODE_RUNTIME 的边界。因此，冻结标签定义对这类复杂依赖问题可能不够明确，数据集标注口径与模型获得的文字说明之间存在解释空间。这限制了对相关误判的归因：不能把全部差异都解释为模型根因推理能力不足，也不能认定定义不明确已被证实是误判原因。本节仅对既有标注作解释，不追溯修改冻结 Prompt 或 GT。

## 2. 两条完整错误案例

下列“根因”是根据冻结数据标签和合成题面证据作出的解释，不是对生产事故的独立调查结论。预测标签与翻转状态可通过仓库中已提交的[Flip 明细](../experiments/prediction_flip_analysis_v1/flip_details.csv)核查。原始 `raw_output`、`api_status` 和 `finish_reason` 则来自本地未纳入 Git 的 predictions.csv；本次已逐项核对，两条案例的原始可见输出与规范化标签一致。这些本地字段不能仅凭 GitHub 上的 Flip 文件独立验证。

### 2.1 Qwen 3.7 Plus：T019

模型键：`qwen3_7_plus`；API 模型：`qwen3.7-plus-2026-05-26`。Ground Truth 为 `CODE_RUNTIME`，预设干扰类为 `NETWORK_API`。

**完整原始 Input（dataset_v2.csv 的 error_text）：**

```text
req=req_812 timeout_deadline=17:22:14.500
provider final_event@17:22:14.461 completion=finished
edge final_event forwarded@17:22:14.468 close=normal
client transport eof=true@17:22:14.472 response_bytes=9211 sha256=27b4...
app response_future_callback_queued@17:22:14.473; timeout_queued@.500; timeout entered@.501 task_cancelled@.502; response callback@.504 completion_discarded=cancelled
TimeoutError request exceeded deadline
scheduler patch trace: response callback processed before timer, task_state=completed timer_skipped=true
```

| Run | Zero-shot 实际预测 | Few-shot 实际预测 | GT | 状态变化 |
|---|---|---|---|---|
| 1 | CODE_RUNTIME | NETWORK_API | CODE_RUNTIME | 正确→错误 |
| 2 | CODE_RUNTIME | NETWORK_API | CODE_RUNTIME | 正确→错误 |

来源：[Flip 明细](../experiments/prediction_flip_analysis_v1/flip_details.csv)中 `model=qwen3_7_plus`、`sample_id=T019`、`run_id=1/2` 的两行，可核查预测标签、GT、干扰类及翻转状态；错误去向均命中该题的 `primary_distractor`。补充运行字段来自本地 `results/qwen3_7_plus/{zero_shot,few_shot}/run_{1,2}/predictions.csv` 的 T019 行：四次均为 API `SUCCESS`、`finish_reason=stop`。这四份原始 CSV 未纳入 Git，仅本地持有者能核查其原始输出和运行字段，故不提供 GitHub 文件链接。

**证据支持的根因。** provider 已完成生成，edge 正常转发，client 在 deadline 前记录完整 EOF。响应回调已排队，却在 timeout 回调取消任务之后才执行，因此完成结果被本地取消状态丢弃。调整 scheduler 回调顺序后，任务完成且 timer 被跳过。按照题面的事件链和反事实修复，本地回调竞态是根因，最终 `TimeoutError` 是后果；网络交付失败不是这些记录支持的首要故障。

**已证实的现象与待验证解释。** 已证实：相同题面下，两轮 Few-shot 均输出 NETWORK_API，而 Zero-shot 均输出 CODE_RUNTIME。待验证解释：固定 Demo 的连接、通信超时示例可能增强了超时症状与 NETWORK_API 的联系，使模型没有充分利用“已完整到达”和回调竞态的证据。但没有保存模型完整推理，也没有 Demo 替换对照，不能认定模型实际使用了这种启发式，更不能把这一解释写成确定原因。

### 2.2 Kimi K3：T055

模型键：`kimi_k3`；API 模型：`kimi-k3`。Ground Truth 为 `ENV_DEPENDENCY`，预设干扰类为 `CODE_RUNTIME`。

**完整原始 Input（dataset_v2.csv 的 error_text）：**

```text
job=structured_agent_144 ValidationError tool_result.id Field required
tool event evt_83 call_id=call_83 tool=search_repository
producer=sp24 intermediate call_id=call_83 alias_map={call_id:id} content_ref=artifact_83
validator_adapter=va18 materialized_fields=[call_id,content_ref] canonical_id=<absent>
adapter fixture sp24+va24 evt_83 canonical_id=call_83 validation=complete
historical sp18+va18 fixture validation=complete
schema exchange manifest: sp18 outputs materialized canonical_id; sp24 outputs alias_map metadata; va18 only reads materialized canonical_id
```

| Run | Zero-shot 实际预测 | Few-shot 实际预测 | GT | 状态变化 |
|---|---|---|---|---|
| 1 | ENV_DEPENDENCY | CODE_RUNTIME | ENV_DEPENDENCY | 正确→错误 |
| 2 | ENV_DEPENDENCY | CODE_RUNTIME | ENV_DEPENDENCY | 正确→错误 |

来源：[Flip 明细](../experiments/prediction_flip_analysis_v1/flip_details.csv)中 `model=kimi_k3`、`sample_id=T055`、`run_id=1/2` 的两行，可核查预测标签、GT、干扰类及翻转状态；错误去向均命中该题预设干扰类。补充运行字段来自本地 `results/kimi_k3/{zero_shot,few_shot}/run_{1,2}/predictions.csv` 的 T055 行：四次均为 API `SUCCESS`、`finish_reason=stop`。这四份原始 CSV 未纳入 Git，仅本地持有者能核查其原始输出和运行字段，故不提供 GitHub 文件链接。

**证据支持的根因。** producer sp24 提供 `call_id` 和 `alias_map={call_id:id}`，而旧适配器 va18 只读取已经实体化的 canonical_id，不处理别名映射，最终导致 id 校验缺失。sp24+va24 组合可以完成校验，历史 sp18+va18 组合也正常；manifest 明确列出两代 producer 的表示方式差异。题面因此支持跨组件字段表示契约不兼容，对应 ENV_DEPENDENCY；`ValidationError` 是失配之后显现的校验症状，而非单凭异常类型就能认定的应用逻辑根因。

**已证实的现象与待验证解释。** 已证实：两轮 Few-shot 均从正确的 ENV_DEPENDENCY 转为 CODE_RUNTIME。待验证解释：Demo 中依赖类表现为直接缺包或显式依赖冲突，可能未充分展示“组件均存在但表示契约不兼容”；模型可能转而把缺字段与应用值处理联系起来。这只是与数据相容的假设，尚无证据证明模型忽略了哪行日志或如何利用示例。

## 3. 两个具体、可执行的改进方案（均待验证）

[既有 Flip 统计](../experiments/prediction_flip_analysis_v1/flip_summary.csv)显示，600 个模型×Run×题目配对中有 54 次正确→错误、23 次错误→正确；核对[逐题明细](../experiments/prediction_flip_analysis_v1/flip_details.csv)，新增错误中 44/54 命中预设干扰类。以下方案针对这一现象提出，不预设准确率一定提高，也不报告未经运行的提升幅度。

### 3.1 方案 A：用根因与表面症状冲突的对比示例替代简单示例

**解决的问题。** 当前 8 条 Demo 中许多异常名称直接提示类别，不能充分展示 Challenge Test 的多阶段归因。方案检验：更明确的反例能否减少“按表面异常分类”的错误，并抑制部分模型的类别偏移。

**具体修改计划。** 未来另建独立开发示例池，设计 4 对、共 8 条对比 Demo，保留每类 2 条的配额。四对分别覆盖 NETWORK_API/CODE_RUNTIME、CONTEXT_LIMIT/ENV_DEPENDENCY、ENV_DEPENDENCY/CODE_RUNTIME、NETWORK_API/CONTEXT_LIMIT。每对使用相近异常表面，但通过故障阶段、数据是否完整到达、接口契约和单因素修复对照，指向不同 GT。例如，同为超时，一题提供远端连接未建立的证据，另一题提供完整交付后本地状态竞态的证据。只采用新的情境、日志和数值；**不把本报告的 T019、T055 或其他 Test 题改写成 Demo**。沿用当前“错误文本+标签”的 Demo 输出格式，不额外加入答案解析，避免同时改变多个因素。8 条无法覆盖全部 12 个定向边界，因此须预先声明所覆盖范围，不能声称全面覆盖。

**示例选择与防泄漏。** 由开发集独立设计并盲审标签，检查与冻结 Test 的完全重复、近似文本和机制情节复用；不把 Test 的 reason、干扰项字段或预测结果提供给示例生成与筛选过程。按事先登记的类别、边界与证据完整性标准选示例，不按正式 Test 上的准确率筛选。当前方案已受正式 Test 错误分析启发，因此即使不复制题目，也不能把再次使用同一 60 题的结果视为完全独立验证；应将旧 Test 作为探索性回测，并用另行封存的新保留集验证泛化。

**可能有效的理由。** 相近表面、不同根因的成对示例能更直接展示“症状不能单独决定标签”的判定规则；但也可能增加 Prompt 长度或引入新的偏置，效果待检验。

**后续对照。** 固定模型集合、调用协议、标签定义、Test 和两轮运行设置，比较旧 8 Demo 与新 8 Demo 两组 Few-shot，保持类别配额、格式及示例长度尽量相当，登记实际长度差异。事先冻结两个示例集合和顺序，不依据测试结果迭代挑选。比较总体、四类、12 边界准确率及逐题 C→W/W→C；同时保留旧 Zero-shot 为参照。报告探索性旧集回测与独立保留集结果，不混合二者，不把微小差值直接解释为机制证实。

### 3.2 方案 B：在 Prompt 中明确证据定位、故障阶段和干扰排除规则

**解决的问题。** 最后抛出的异常可能来自较晚的 wrapper、validator 或 timeout 回调，模型需要从事件链判断主要故障，而不是把最后一行直接映射为类别。

**具体修改计划。** 未来在原有标签定义之后、示例与待分类输入之前插入统一的判定规则；Zero-shot 和 Few-shot 使用同一段规则，保持四类定义及最终输出协议不变。候选新增文本如下，**这是待验证的拟议规则，不属于第 1 节的实际冻结 Prompt**：

> 判定前请在内部核对：输入与传输是否完整，失败发生在远端通信、容量准入、组件契约还是应用执行阶段；最后出现的异常是否只是更早故障的后果。优先利用时间顺序、作用域和修复对照定位主要根因；当完整交付记录排除通信失败时，不仅凭 TimeoutError 或 BrokenPipe 等名称归为 NETWORK_API；当组件契约不兼容导致后续字段校验失败时，不仅凭 ValidationError 归为 CODE_RUNTIME；当原请求在正确契约下仍因容量准入被拒绝时，再依据容量证据判断。只依题面证据，不补造缺失事实。最终只输出一个合法标签，不输出证据或分析过程。

规则应由独立开发材料审阅后统一冻结，不携带正式 Test ID、逐题 GT 或特定答案。规则关注证据链，并不要求输出完整思维过程，也不改变最终只读取 `message.content` 的分类协议。

**可能有效的理由。** 显式检查交付完整性、故障阶段和组件契约，可能减少将次生症状误作根因的情况。与此同时，新增规则可能诱导模型过度解释证据，或把更多错误归到依赖类；这种反向偏移也需要检查，不能预设只会改善。

**后续对照。** 先固定原有 8 Demo，分别在 Zero-shot 与 Few-shot 下比较“原规则”和“新增规则”，其他配置一致，完成两轮并报告逐题翻转和类别、边界变化。若同时检验方案 A，则预登记“旧/新 Demo × 原/新规则”的 2×2 Few-shot 对照，以区分示例替换、规则添加及共同作用，不能把两者一起修改后的变化全部归因于某一方案。沿用独立保留集和探索性旧集回测的分工；不根据结果修订标签或删除难题。以上均为未来实验计划，本轮未执行。

## 4. 数据与代码来源核查说明

| 材料 | 核查内容与口径 |
|---|---|
| [dataset_v2.csv](../dataset_v2.csv) | 原文件 SHA-256 为 `3471f8572b9953f4de7e8a547fad88893265c3f304d7c2fd5fdf5b38afd4876e`；冻结 canonical fingerprint 为 `7abb151f8a42bca101465011726ffcf6376fb20523e0dd9cdfc559dacf0610f0`。Demo 为 D001–D008；两案例 input 直接提取原始 error_text。 |
| [prompts/](../prompts/) 与 [prompts.py](../src/llm_experiment/prompts.py) | 全部四个模板已核对。第 1 节使用现有 render_prompt() 还原，Few-shot 全量包含 8 Demo；两种条件使用相同 T019 输入。 |
| [api_client.py](../src/llm_experiment/api_client.py) 与 [prediction.py](../src/llm_experiment/prediction.py) | 核对唯一 user 消息、请求参数和只从 content 分类；用本地 Fake SDK 捕获参数进行核查，没有构建真实远端请求。 |
| [V4 元数据](../experiments/challenge_audit_v1/v4/candidates_full_v4.jsonl) | 核对两题 label、primary_distractor、evidence_pattern、design_reason 与 V2 一致的题面及根因解释。 |
| 第 2 节以本地路径注明的 8 份原始 predictions.csv（未纳入 Git） | 本次逐题核对 raw_output、prediction、模型名、Run、api_status 和 finish_reason。原始输出与运行字段仅能由本地文件独立核查；GitHub 上的 Flip 明细仅支持其已保存的预测标签、GT、干扰类、Run 与翻转状态。 |
| [Flip 汇总](../experiments/prediction_flip_analysis_v1/flip_summary.csv)、[明细](../experiments/prediction_flip_analysis_v1/flip_details.csv)及[原分析报告](../experiments/prediction_flip_analysis_v1/report.md) | 两案例四个模型×Run 配对均为 correct_to_wrong；总体 54 次退化、23 次纠正，新增错误命中干扰类 44 次。 |

本次核查所需数据和代码在当前工作区均可访问，但本地原始 predictions.csv 未上传 GitHub；仓库读者可沿 Flip 明细核查预测标签与翻转状态，不能据此核查未保存于该文件的原始输出和运行字段。本材料未新增实验。完整历史 HTTP 请求报文和完整模型 reasoning 文本不在上述预测 artifact 中，因此第 1 节是可复核的代码渲染还原，案例中的内部判断原因及两个方案的效果均尚待验证。所有解释以冻结的合成题面为边界，不宣称已经取得生产环境因果证据。
