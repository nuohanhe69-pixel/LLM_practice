# 实施报告

## 最终项目目录结构

```text
.
├── configs/models.json              # 模型与百炼连接参数
├── docs/implementation_report.md    # 本报告
├── prompts/                         # 标签定义及 zero/few-shot 模板
├── src/llm_experiment/              # 实验流水线包
├── tasks/                            # 规格实施计划与完成清单
├── tests/                            # 单元和本地集成测试
├── .env.example                     # 环境变量示例，不含密钥
├── dataset_v1.csv                   # 已 Review 并冻结的正式数据集
├── pyproject.toml                   # 依赖、测试与 lint 配置
├── run_experiment.py                # 统一运行入口
└── results/                          # 运行时生成，Git 忽略
```

正式输出为 `results/<model>/<prompt_type>/run_<id>/`。带 `--limit N` 的输出位于同一 run 目录下的 `smoke_limit_<N>/`，避免覆盖正式实验。

## 核心文件职责

- `dataset.py`：严格校验冻结数据集；Prediction 只能得到仅含 `id`、`error_text` 的 `TestSample`。
- `config.py`：读取并校验 `configs/models.json`，解析可选端点环境变量。
- `prompts.py`：读取外部模板，生成 zero-shot 或固定 8 条 demo 的 few-shot Prompt。
- `api_client.py`：使用 OpenAI SDK 调用百炼兼容接口，应用配置化 timeout、max retries、temperature 与模型名。
- `prediction.py`：执行预测、有限标准化、区分 API 状态，并按样本原子保存 checkpoint。
- `evaluation.py`：唯一读取 test Ground Truth 的业务模块；计算指标并输出错误案例。
- `pipeline.py`：选择同一批 test 样本，顺序执行 Prediction 和 Evaluation。
- `cli.py` / `run_experiment.py`：解析统一命令并报告产物路径。

## Prediction 数据流

1. `dataset.py` 校验 `dataset_v1.csv`，分别构造 8 个 `DemoSample` 和 60 个安全 `TestSample`。
2. `pipeline.py` 按 `--limit` 选择 test 前缀；不传时选择全部 60 条。
3. `prompts.py` 读取标签定义和对应模板。Few-shot 仅加入 8 个 demo 的 `error_text` 与 `label`。
4. `api_client.py` 从环境变量读取 API Key，并按模型配置调用 Chat Completions。
5. `prediction.py` 保存原始输出；仅去除首尾空白、换行并统一大写。不能精确匹配四个标签时写为 `INVALID_OUTPUT`。
6. 每个样本结束后原子更新 `predictions.csv`；全部目标样本落盘后自动进入 Evaluation。

## Evaluation 数据流

1. `evaluation.py` 重新读取并校验冻结数据集，只取 `split=test` 的 `id`、`error_text`、`label`。
2. 读取 `predictions.csv`，检查每个目标 `sample_id` 恰好出现一次且实验元数据一致。
3. 通过 `sample_id` 连接预测与 Ground Truth。
4. 程序计算总数、API 成功数、无效输出数、API 失败数、正确数、总体 Accuracy，以及各类别正确数和 Accuracy。
5. 自动原子写出 `metrics.json` 与 `error_cases.csv`。

`accuracy = correct_predictions / successful_predictions`。API 失败不作为分类错误进入分母；API 正常返回但不合法的 `INVALID_OUTPUT` 进入分母并计为错误。

## Ground Truth 与 Prediction 隔离

- `TestSample` 类型只有 `id` 和 `error_text`，Prediction 和 Prompt 接口无法访问 test 的 `label`、`reason`、`source_type`、`source_reference`。
- demo 类型只保留 few-shot 必需的 `id`、`error_text`、`label`。
- test `label` 仅在 `evaluation.py` 内重新读取。
- 测试显式检查 test 元数据与 reason/source 标记不会出现在 Prompt 中。

## 断点续跑

- `predictions.csv` 是逐样本 checkpoint，不是模型生成文件。
- 每次启动先校验已有 CSV 的字段、实验元数据、状态和重复 ID。
- CSV 中已有 `SUCCESS` 的样本会跳过；`INVALID_OUTPUT` 属于 API 成功返回，因此同样不会重复调用或覆盖。
- 已有 `API_FAILURE` 的样本会在恢复时重新调用；新结果替换原列表位置，最终 CSV 中该 `sample_id` 仍只有一条记录。
- 每次新建或更新记录都先写入同目录临时文件、刷新并 `fsync`，再通过原子替换提交，降低中断造成半行或损坏文件的风险。

## API 失败处理

- OpenAI SDK 客户端使用模型配置中的 `timeout_seconds` 和有限 `max_retries`；默认 qwen 配置为 60 秒、2 次 SDK 重试。
- SDK 重试耗尽后的异常写为 `api_status=API_FAILURE`，保留异常类型与信息，并保存实际 latency。
- API 正常返回但标签错误属于普通分类错误；API 正常返回但格式不合法写为 `prediction=INVALID_OUTPUT`。
- `API_FAILURE` 与 `INVALID_OUTPUT` 在 `api_status`、`prediction` 和 `error_type` 中保持可区分。

## 增加新模型

在 `configs/models.json` 的 `models` 下新增一个安全名称条目，填写：

- `api_model`
- `base_url` 和可选 `base_url_env`
- `api_key_env`
- `temperature`
- `max_tokens`
- `timeout_seconds`
- `max_retries`

随后把新名称传给 `--model`。无需修改 Prediction Pipeline 核心代码。若模型不支持温度 0，只修改该模型配置为其最低允许值；实际温度会进入产物。

## Smoke Test 执行方式

冻结数据集与 API 环境变量就绪后执行：

```bash
uv run python run_experiment.py \
  --model qwen_plus \
  --prompt-type zero_shot \
  --run-id 1 \
  --limit 10
```

该命令与正式实验使用完全相同的 Pipeline。产物位于 `results/qwen_plus/zero_shot/run_1/smoke_limit_10/`。

## 当前尚未执行的内容

- 未运行任何真实百炼 API 调用。
- 未运行 60 条正式实验、两个 Prompt × 两次重复或多个模型。
- 未比较或人工统计模型结果。
- 未生成课程报告。
- 已原样补入项目方 Review 并冻结的 `dataset_v1.csv`；本次按 CR 要求仍未运行真实 Smoke Test。
