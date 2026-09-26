# LLM Practice Experiment Pipeline

该项目提供一条可复现的四分类实验流水线：默认读取冻结的 `dataset_v2.csv`，调用阿里云百炼 OpenAI-compatible API 生成预测，并自动输出指标与错误案例。Prediction 与 Evaluation 在代码层分离，通过统一 CLI 串联。

## 环境准备

要求 Python 3.10+。推荐使用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --extra dev
```

也可以使用普通虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

冻结的 `dataset_v2.csv` 已原样保存在仓库根目录。程序会严格校验字段、标签、唯一 ID、8 条 demo 和 60 条 test；不会修改该文件。旧版 `dataset_v1.csv` 保留作历史记录。

## API 配置

正式五模型均显式开启 `enable_thinking=true`（通过 `extra_body` 发送），保留各模型默认 reasoning effort。不人为配置或发送 `max_tokens` / `max_completion_tokens`，使用 provider/model 默认最大输出长度。

API Key 只从环境变量读取：

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

如账号使用业务空间专属域名，可覆盖默认端点：

```bash
export DASHSCOPE_BASE_URL="https://<workspace-id>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
```

`.env.example` 仅列出变量名；`.env` 已被 Git 忽略。若使用 `.env` 文件，请先在 shell 中加载它，例如 `set -a; source .env; set +a`。

## 运行实验

Smoke Test 与正式实验使用完全相同的流水线。`--limit` 只限制按数据集顺序选取的 test 数量，并将结果放入独立目录：

```bash
uv run python run_experiment.py \
  --model qwen3_7_plus \
  --prompt-type zero_shot \
  --run-id 1 \
  --limit 10
```

正式实验不传 `--limit`：

```bash
uv run python run_experiment.py \
  --model qwen3_7_plus \
  --prompt-type few_shot \
  --run-id 1
```

同一命令中断后可直接重启。已有 `SUCCESS` 记录（包括 `INVALID_OUTPUT`）会被跳过，不会再次调用或覆盖；`API_FAILURE` 会在恢复时重试，并在原位置更新为最新结果，不产生重复 `sample_id`。每条 API 结果都会原子持久化。

每次完成后自动生成：

- `predictions.csv`
- `metrics.json`
- `error_cases.csv`

正式结果路径为 `results/<model>/<prompt_type>/run_<id>/`；Smoke Test 在其下增加 `smoke_limit_<N>/`，不会覆盖正式结果。

## 指标口径

- `successful_predictions`：API 正常返回的数量，包括 `INVALID_OUTPUT`。
- `api_failures`：SDK 在配置的有限重试后仍失败的数量。
- `accuracy`：`correct_predictions / successful_predictions`。API 失败不进入分母；`INVALID_OUTPUT` 进入分母并作为错误分类。
- 每类 Accuracy 使用同一规则，仅在该 Ground Truth 类别内部计算。

## 增加模型

只需在 `configs/models.json` 的 `models` 下新增条目，配置 API 模型名、端点、API Key 环境变量、实际 temperature、max tokens、timeout 和 max retries。Prediction 核心代码无需修改。

若模型不支持 `temperature = 0`，请在其独立配置中填写允许的最低值；实际值会写入预测文件与指标文件。

## 开发验证

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q src run_experiment.py
```

测试使用生成的本地 fixture 和 fake API client，不访问百炼、不修改冻结数据集，也不启动正式实验。详细设计见 [`docs/implementation_report.md`](docs/implementation_report.md)。
