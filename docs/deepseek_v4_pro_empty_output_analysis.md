# Difficulty Pilot V1：`deepseek-v4-pro` 24/24 空输出分析

分析日期：2026-09-21
范围：只分析现象、官方参数语义、最小诊断结果与后续建议；未修改实验代码、Prompt、数据集或既有结果 CSV。

## 结论摘要

`deepseek-v4-pro` 的 24 条空 `content` 不是 API 失败，也没有证据表明是模型分类能力失败。根因是当前请求将 `max_tokens` 固定为 16，而百炼官方文档明确规定：对 DeepSeek V4 系列，`max_tokens` 限制的是“思维链 + 最终回答”的总输出 Token 数。

一次受控的 P001 最小诊断完整复现了该机制：

- `finish_reason = "length"`
- `message.content = ""`
- `message.reasoning_content` 非空
- `completion_tokens = 16`
- `completion_tokens_details.reasoning_tokens = 16`

也就是说，16 个生成 Token 全部消耗在 reasoning 阶段，生成在最终标签出现前被截断。项目客户端随后只读取 `message.content`，得到空字符串，再由 normalization 标记为 `INVALID_OUTPUT`。

因此：

1. 当前 24 条 DeepSeek 记录应视为**配置截断导致的无效运行**，不应计入模型准确率或模型分歧分析。
2. 现有报告中的 DeepSeek `0.000000` accuracy 不能解释为能力结果。
3. 方案 B 下应改用 `max_completion_tokens`，并为 reasoning + 最终标签提供足够的总预算。
4. OpenAI Python SDK 3.14.0 已支持 `max_completion_tokens`、`reasoning_effort` 和 `extra_body`；不是 SDK 无法发送参数。
5. SDK 也保留了百炼返回的 `reasoning_content` 扩展字段；信息丢失发生在项目 `api_client.py` 只返回 `message.content` 的封装层。

## 1. 问题复现与现象

### 1.1 既有 Pilot 直接证据

从 `experiments/difficulty_pilot_v1/artifacts/pilot_predictions_v1.csv` 与执行报告可确认：

- `deepseek-v4-pro` 共 24 条记录；
- 24 条 `status` 均为 `INVALID_OUTPUT`；
- 24 条 `raw_output` 均为空字符串；
- 24 条 `normalized_output` 均为 `INVALID_OUTPUT`；
- 24 条 `error_type` 均为空；
- `API_FAILURE = 0`；
- 实际 temperature 均为 `0.0`；
- 单次 latency 为约 1.32–1.88 秒，24 次合计约 36.77 秒。

这些记录直接证明 API 调用正常返回、项目拿到的可见 `content` 为空，但既有 CSV 没有保存 `finish_reason`、`reasoning_content` 或 usage，因此仅靠旧 artifact 不能直接证明每一条为何为空。

### 1.2 当前报告中需要纠正的解释

既有执行报告把 DeepSeek 统计为：

- `SUCCESS = 0`
- `INVALID_OUTPUT = 24`
- `Correct = 0`
- `Accuracy = 0.000000`

这个状态在当前程序的字段定义下是机械正确的，但不适合作为模型能力统计：空输出来自生成预算耗尽，不是模型生成了错误类别，也不是模型拒绝遵循标签集合。

报告中的“24 个模型分歧 item”同样受到该配置问题污染。DeepSeek 并未提供可比较的最终分类，不能把 `INVALID_OUTPUT` 当作第三个有效模型意见。

## 2. 当前调用链

当前调用链如下：

1. `configs/models.json` 为三个模型统一设置：
   - `temperature = 0.0`
   - `max_tokens = 16`
   - `timeout_seconds = 60`
   - `max_retries = 2`
2. `difficulty_pilot.py` 读取只有 `id,error_text` 的 Pilot input，并用冻结的 Zero-shot Prompt 渲染请求。
3. `OpenAICompatibleClient.complete()` 调用：

   ```python
   response = client.chat.completions.create(
       model=config.api_model,
       messages=[{"role": "user", "content": prompt}],
       temperature=config.temperature,
       max_tokens=config.max_tokens,
   )
   ```

4. `api_client.py` 只取：

   ```python
   response.choices[0].message.content
   ```

5. `prediction.normalize_prediction()` 对空字符串执行 trim/newline removal/uppercase，最终得到 `INVALID_OUTPUT`。
6. Pilot 将正常 API 返回但非法标签的记录固化为 `INVALID_OUTPUT`，不会作为 `API_FAILURE` 重试。

### 2.1 Prompt 不是本次空输出根因

Zero-shot Prompt 明确要求只输出四个标签之一。相同 Prompt 下 Qwen 与 GLM 均产生了可见标签；最小诊断也显示 DeepSeek 已开始对同一分类问题进行 reasoning。没有证据表明 Prompt 导致 API 返回结构异常。

### 2.2 `temperature=0` 不是本次空输出根因

百炼文档列出的 `deepseek-v4-pro` 默认 temperature 是 1.0，而当前实验显式覆盖为 0.0。不过诊断请求在 temperature 0.0 下正常返回 HTTP 成功响应、reasoning、usage 和 `finish_reason=length`。因此 temperature 不是空 `content` 的直接原因；是否继续使用 0.0 是实验设计问题，应单独冻结并记录。

## 3. 官方参数语义

### 3.1 OpenAI 标准语义

OpenAI Chat Completions 官方 API 参考将 `max_completion_tokens` 定义为一次 completion 可生成 Token 的上限，包含可见输出 Token 和 reasoning Token。`max_tokens` 是旧参数，OpenAI 已建议迁移到 `max_completion_tokens`。

需要注意：本项目调用的是百炼的 OpenAI-compatible 接口。最终行为要以百炼对具体模型的映射为准，不能只套用 OpenAI 自有模型的语义。

### 3.2 百炼 OpenAI-compatible 语义

百炼官方 OpenAI-compatible Chat 文档给出了模型相关的精确定义：

| 参数 | 百炼语义 | 对 `deepseek-v4-pro` 的影响 |
|---|---|---|
| `max_tokens` | 即将废弃；不同模型语义不同 | DeepSeek V4 系列中，它限制“思维链 + 回答”的总 Token；达到上限时 `finish_reason=length` |
| `max_completion_tokens` | 限制完整输出，即“思维链 + 回答”；达到上限时 `finish_reason=length` | 明确支持 `deepseek-v4-pro`，也是思考模型推荐参数 |

这揭示了当前三模型配置的关键不公平点：虽然 Qwen、GLM、DeepSeek 都配置了 `max_tokens=16`，该值在百炼兼容层上不保证具有相同语义。官方文档对 DeepSeek V4 做了特殊说明：16 是 reasoning 与答案共享的总预算；对未列入特殊规则的模型，`max_tokens` 只限制最终回答。因而“同名、同数值”不等于“同资源约束”。

### 3.3 DeepSeek V4 thinking 行为

百炼官方 DeepSeek 文档确认：

- `deepseek-v4-pro` 是混合思考模型；
- 默认开启 thinking；
- 默认 `reasoning_effort` 为 `high`；
- `deepseek-v4-pro` 支持的主要 reasoning 档位为 `high`、`max`；`low` 只明确列给特定快照型号（如 `deepseek-v4-pro-0813`）；
- thinking 内容返回在 `message.reasoning_content`；
- 最终回答返回在 `message.content`；
- `usage.completion_tokens` 是本次 completion 总输出量；
- reasoning 用量在 `usage.completion_tokens_details.reasoning_tokens`；
- `enable_thinking` 是百炼扩展参数，Python SDK 要通过 `extra_body={"enable_thinking": true}` 传入；
- `reasoning_effort` 是 OpenAI 标准形态的顶层参数；
- 不显式传入 `enable_thinking` 并不会关闭 `deepseek-v4-pro` 的 thinking，因为该模型默认开启；
- 官方模型页给出的最大输出长度为 393,216 Token，但这只是能力上限，不是分类任务的推荐预算。

官方文档没有给出适用于本分类任务的最低 generation budget。因此本文后面的 1,024–2,048 建议属于工程建议，不是官方最低值。

### 3.4 当前 OpenAI SDK 支持情况

仓库锁定 `openai==3.14.0`。本地运行时检查确认：

- `chat.completions.create(...)` 签名原生包含 `max_tokens`；
- 同一签名原生包含 `max_completion_tokens`；
- 同一签名原生包含 `reasoning_effort`；
- 同一签名支持 `extra_body`，可传百炼的 `enable_thinking`；
- `CompletionUsage` 原生包含 `completion_tokens_details`；
- `ChatCompletionMessage` 的静态字段没有声明 `reasoning_content`，但 Pydantic 模型配置为允许额外字段；诊断中 `message.reasoning_content` 与 `message.model_extra["reasoning_content"]` 均可读取。

结论：SDK 3.14.0 不会阻止本项目改用 `max_completion_tokens`，也没有在解析阶段删除 `reasoning_content`。当前缺失是应用封装造成的。

## 4. 最小诊断结果

### 4.1 诊断约束

只执行了一次调用：

- 样本：P001
- 模型：`deepseek-v4-pro`
- Prompt：现有冻结 Zero-shot Prompt
- temperature：0.0
- `max_tokens`：16
- `max_completion_tokens`：未传
- thinking：未显式设置，沿用模型默认开启行为
- SDK：OpenAI Python 3.14.0
- 没有读取 Ground Truth
- 没有覆盖或修改 `pilot_predictions_v1.csv`
- 没有写入临时诊断 artifact
- 没有输出或记录 API Key

### 4.2 返回元数据

```text
model_requested: deepseek-v4-pro
model_returned: deepseek-v4-pro
finish_reason: length
message.content: ""
message.reasoning_content: "我们被要求对错误分类。 先看给定的错误信息。\n\n有两部分"
prompt_tokens: 420
completion_tokens: 16
reasoning_tokens: 16
total_tokens: 436
```

该结果与旧 Pilot 的空 `content` 现象一致，并直接证明：在至少 P001 上，生成停止不是自然结束，而是 16 Token 总预算被 reasoning 完全耗尽。

## 5. 根因分析

### 5.1 直接根因

根因链条为：

```text
deepseek-v4-pro 默认开启 thinking
        ↓
百炼对 DeepSeek V4 将 max_tokens 解释为 reasoning + answer 总预算
        ↓
当前 max_tokens=16
        ↓
16 个 completion token 全部成为 reasoning token
        ↓
finish_reason=length，最终 answer 尚未生成
        ↓
message.content=""
        ↓
项目客户端丢弃 reasoning/finish_reason/usage，只返回空 content
        ↓
normalization 将空 content 标为 INVALID_OUTPUT
```

### 5.2 SDK 是否没有读取 provider 字段

不是。诊断确认 OpenAI SDK 3.14.0 将百炼的 `reasoning_content` 保存在 message 的扩展字段中，并能通过属性读取。问题是 `api_client.py` 在 SDK 解析之后只提取 `message.content`，没有把其它响应元数据传给 prediction 层。

### 5.3 百炼兼容层是否有特殊返回格式

有，但这是官方支持的格式，不是异常：

- reasoning 与最终答案分列在 `reasoning_content` 和 `content`；
- reasoning Token 计入 `completion_tokens`，并在 `completion_tokens_details.reasoning_tokens` 单列；
- DeepSeek V4 的 `max_tokens` 对 reasoning + answer 共同生效。

### 5.4 是否需要显式开启 thinking

当前模型默认开启，因此本次没有显式参数也实际产生了 reasoning。方案 B 为了配置可审计，后续可显式记录 `enable_thinking=true` 与/或 `reasoning_effort=high`，但这不是修复空输出的关键。关键是使用正确的总输出预算。

### 5.5 是否存在其它参数兼容问题

本次没有发现第二个能解释空输出的问题：

- model ID 请求与返回一致；
- temperature 0.0 被接口接受；
- 非流式调用成功；
- SDK 成功解析扩展 reasoning 字段；
- 没有 API error；
- `finish_reason=length` 与 Token 用量完整解释了空 `content`。

## 6. 已确认、推断与未确认

### 6.1 已确认事实

- 历史 Pilot 的 24 个 DeepSeek `content` 均为空，且没有 API failure。
- 24 个请求都使用 `temperature=0.0`、`max_tokens=16`。
- 百炼官方规定 DeepSeek V4 的 `max_tokens` 是 reasoning + answer 的共享总预算。
- `deepseek-v4-pro` 默认开启 thinking，默认 reasoning effort 为 high。
- P001 诊断返回 `finish_reason=length`、16 completion tokens、16 reasoning tokens、空 content。
- SDK 3.14.0 支持 `max_completion_tokens`，并保留 `reasoning_content` 扩展字段。
- 项目客户端没有保存 finish reason、reasoning 或 usage。

### 6.2 官方文档支持的行为

- reasoning 在 `reasoning_content`，最终回复在 `content`。
- `reasoning_tokens` 位于 `completion_tokens_details`。
- `max_completion_tokens` 覆盖 reasoning + visible answer，思考模型推荐使用它。
- 达到生成上限时 `finish_reason=length`。
- DeepSeek V4 的 `max_tokens` 也会限制 reasoning + answer 总量，但该参数已进入废弃路径。

### 6.3 高置信推断

- 其余 23 条历史 DeepSeek 空输出与 P001 原因相同：16 Token 在 reasoning 阶段耗尽。依据是 24 条使用完全相同的配置、全部空 content、全部 API 成功，且 P001 可稳定复现相同外部表现与明确的 `length` 证据。
- 现有 Qwen/GLM 与 DeepSeek 事实上没有受到同语义的输出预算约束，因此原 Pilot 不构成严格公平的三模型比较。

### 6.4 尚未确认

- 旧 24 条中每一条的历史 `finish_reason` 与 reasoning token 数，因为当时没有保存这些字段。
- `max_completion_tokens=512`、1,024 或 2,048 时，各模型在这 24 个样本上的真实 reasoning 分布与截断率。
- 对本任务而言最小且零截断的统一预算值。需要在非 Pilot 校准样本或版本化重跑前预检，不能由单次 P001 诊断严格推出。

## 7. 方案 B 下的参数建议

### 7.1 使用 `max_completion_tokens`

正式实验应从 `max_tokens` 改为 `max_completion_tokens`，理由是：

- 它对三个目标模型都受百炼支持；
- 它明确表示 reasoning + answer 的总预算；
- 它避免 `max_tokens` 的模型特定语义差异；
- 它是百炼对思考模型的推荐参数，也是 OpenAI 侧的新参数方向。

不要同时发送 `max_tokens` 与 `max_completion_tokens`，以免兼容层优先级不清或形成不可审计配置。

### 7.2 预算建议

工程建议如下：

- 预检起点：`max_completion_tokens=1024`
- 正式实验更稳妥的候选：`max_completion_tokens=2048`
- 若预检发现任何 `finish_reason=length`、空 `content` 或 `completion_tokens` 贴近上限，则在正式实验前统一提高预算，并重新冻结配置。

1,024–2,048 不是官方最低值，而是针对“输入约 400+ Token、最终答案只有一个标签、保留正常 reasoning”的保守工程范围。选择 2,048 会增加最坏成本，但更能降低 Hard 样本在 high reasoning 下被截断的风险。

严禁在正式实验中按单条样本动态提高预算后重跑，因为这会为困难样本提供不同资源。任何预算调整都应形成新配置版本，并对同一实验范围统一生效。

### 7.3 Qwen / GLM / DeepSeek 是否统一数值

建议统一 `max_completion_tokens` 数值，而不是统一 `max_tokens` 数值。统一总 completion 上限可以实现“每个模型最多获得相同数量的 reasoning + answer Token”的资源公平。

仍需承认：相同总预算不等于相同推理行为。不同模型的默认 reasoning effort、Token 效率和内部推理长度不同。因此公平性应定义为：

1. 相同 Prompt；
2. 相同总输出 Token 上限；
3. thinking 对所有模型保持开启；
4. reasoning effort 使用预先声明的、各模型支持的等价档位；
5. 不按结果动态调整单个模型或单个样本；
6. 完整记录实际 Token 用量与停止原因。

如果方案 B 的“正常 Thinking”指各模型默认思考行为，可以保留各模型默认 effort，但必须在配置与报告中明确“默认档位可能不同”。如果目标是更强的资源可比性，可对三个模型显式使用共同支持的 `high` 档位；在执行前必须逐模型验证百炼当前文档中的支持范围。

### 7.4 显式 thinking 参数

建议把 thinking 状态显式记录在实验配置中：

- `enable_thinking=true`：清楚表达方案 B；Python SDK 通过 `extra_body` 传递；
- `reasoning_effort="high"`：作为顶层参数，三模型均应先按当前文档验证支持。

这两项的目的主要是可审计性与跨时间复现，不是本次根因修复的必要条件。真正防止空答案的是足够的 `max_completion_tokens`。

## 8. API client 修改建议

### 8.1 返回结构化结果，而不是裸字符串

后续建议让 provider boundary 返回结构化 completion 结果，至少包含：

| 字段 | 用途 | 是否参与最终分类 |
|---|---|---|
| `content` | 模型最终答案；交给现有 normalization | **是，唯一分类输入** |
| `reasoning_content` | 诊断 thinking 是否开启、审计截断 | 否 |
| `finish_reason` | 判断自然停止、长度截断、工具调用等 | 不直接分类，但决定运行是否有效 |
| `completion_tokens` | 总生成 Token | 否，运行元数据 |
| `reasoning_tokens` | reasoning 用量 | 否，运行元数据 |
| `prompt_tokens` / `total_tokens` | 成本与资源审计 | 否 |
| `model_returned` | 确认 provider 实际模型 | 否 |
| 实际 generation 参数 | 复现实验条件 | 否 |

`reasoning_content` 不应回退为分类答案。即使其中出现四个标签之一，也只能使用 `content` 作为模型最终选择，否则会把内部候选分析误当作最终分类。

### 8.2 截断状态应与普通 `INVALID_OUTPUT` 区分

建议未来把以下情况单列为 `GENERATION_TRUNCATED` 或等价状态：

- `finish_reason=length`；
- `content` 为空且 reasoning 非空；
- `completion_tokens` 达到配置上限；
- reasoning tokens 等于 completion tokens，且没有最终答案。

这类结果既不是网络 API 失败，也不是模型输出了非法类别。它是实验配置未给出足够生成预算。正式实验遇到此状态时不应在同一配置下盲目重试；应停止或标记整批配置问题，修订预算后用新版本重跑。

### 8.3 建议新增的测试覆盖

后续实现时应新增测试，但本轮不改代码：

- SDK 返回 `reasoning_content + content + usage + finish_reason` 时完整保留；
- provider 扩展字段通过 `model_extra`/属性可读取；
- `max_completion_tokens` 被正确发送，且不再同时发送 `max_tokens`；
- `finish_reason=length + empty content` 不被计为普通分类错误；
- normalization 仍只作用于 `content`；
- checkpoint 元数据包含实际预算与 thinking 配置；
- Qwen、GLM、DeepSeek 的模型特定参数校验。

## 9. 当前 Pilot 结果处理建议

### 9.1 Qwen / GLM 结果

Qwen 与 GLM 的现有预测可以保留为 V1 历史结果；它们确实返回了最终标签，可用于审查各自样本错误。

但不建议在改变为统一 `max_completion_tokens` 后，直接把旧 Qwen/GLM 与新 DeepSeek 结果拼接成“完全公平”的三模型性能比较。旧请求的 `max_tokens=16` 对不同模型的 reasoning 约束语义不同。

### 9.2 DeepSeek 24 条结果

- 应视为配置截断导致的无效运行；
- 不计入 DeepSeek accuracy 分母；
- 不计入三模型一致/分歧判断；
- 不标记为模型分类错误；
- 保留原始 V1 artifact 作为审计证据。

### 9.3 是否只重跑 DeepSeek 24 条

取决于目标：

- **恢复性诊断/补全**：可以只重跑 DeepSeek 24 条，但必须生成带新版本号的 recovery artifact，并明确其 generation budget 与 Qwen/GLM V1 不同；该结果不应被描述为严格同条件三模型比较。
- **为正式实验验证公平配置**：建议三个模型都在统一的 `max_completion_tokens`、thinking 开启、固定 effort 策略下重跑一个新版本 Pilot。这样才能验证正式配置，而不是只修补 DeepSeek 一列。

### 9.4 Artifact 版本策略

不要覆盖 `pilot_predictions_v1.csv` 或其它 V1 文件。建议：

- 仅 DeepSeek 恢复：生成 `pilot_predictions_v1_1_deepseek_recovery.csv` 等独立文件；
- 三模型统一配置重跑：生成完整的 `difficulty_pilot_v2` 目录与 V2 artifacts；
- 新报告必须引用原 V1、说明根因和配置差异，并保留两版。

## 10. 下一步建议修改项

以下只是 Review 后的建议清单，本轮未实施：

1. 扩展模型配置，区分 `max_completion_tokens`、thinking 开关与 reasoning effort。
2. 三模型改用统一的 `max_completion_tokens` 总预算。
3. 显式记录 `enable_thinking` 与 `reasoning_effort`。
4. 将 API client 返回值改为结构化 completion 结果。
5. 在预测 CSV 或配套 metadata 中保存：
   - `finish_reason`
   - `completion_tokens`
   - `reasoning_tokens`
   - `prompt_tokens`
   - `total_tokens`
   - 实际 model
   - 实际预算参数
   - thinking 配置
6. 将 `GENERATION_TRUNCATED` 与 `INVALID_OUTPUT`、`API_FAILURE` 分离。
7. 在非 Pilot 校准样本上预检 1,024 与 2,048 的截断率，冻结正式预算后再运行。
8. 生成新版本 artifact，绝不覆盖 V1。

## 11. 官方参考资料

- [阿里云百炼：OpenAI 兼容 Chat Completions 参数说明](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)
- [阿里云百炼：DeepSeek V4 / R1 / V3 API](https://help.aliyun.com/zh/model-studio/deepseek-api)
- [阿里云百炼：深度思考模型的用法](https://help.aliyun.com/zh/model-studio/deep-thinking)
- [阿里云百炼：deepseek-v4-pro 模型信息](https://help.aliyun.com/zh/model-studio/deepseek-v4-pro)
- [OpenAI Docs：Chat Completions API Reference](https://developers.openai.com/api/reference/resources/chat)
- [OpenAI Python SDK v3.14.0 release](https://github.com/openai/openai-python/releases/tag/v3.14.0)

## 12. 仓库证据位置

- `configs/models.json:23`：DeepSeek 配置与 `max_tokens=16`
- `src/llm_experiment/api_client.py:31`：请求参数与只返回 `message.content` 的封装
- `src/llm_experiment/prediction.py:66`：normalization 规则
- `src/llm_experiment/difficulty_pilot.py:185`：Pilot Stage A 调用与状态落盘
- `prompts/zero_shot.txt:1`：冻结的 Zero-shot Prompt
- `experiments/difficulty_pilot_v1/artifacts/pilot_predictions_v1.csv`：24 条空输出记录
- `experiments/difficulty_pilot_v1/artifacts/pilot_execution_report_v1.md:17`：现有统计结果
- `tests/test_api_client.py:63`：当前客户端测试只覆盖 `content` 与 `max_tokens`
