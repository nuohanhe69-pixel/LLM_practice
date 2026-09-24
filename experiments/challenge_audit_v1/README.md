# Challenge Set V3 开发期审查

`handoff_v3/` 原样保存《三问题_定向修订V3_完整交接包.zip》的全部 22 个文件，包括 V2 基线、60 条 V3 候选、仅含 `id`/`error_text` 的模型输入、修订记录、审查脚本、测试和原交接报告。V3 仍是合成的开发期候选，不是冻结的正式 Test 数据集。内部 GT 文件及盲审 ID 映射不得发送给外部审阅者。

在仓库根目录安装 [uv](https://docs.astral.sh/uv/) 后，运行：

```bash
bash experiments/challenge_audit_v1/run_full_audit.sh
```

脚本使用 Python 3.12，在系统临时目录创建隔离虚拟环境，按 `handoff_v3/audit_requirements.txt` 安装固定版本的 NumPy 和 scikit-learn。它在交接包临时副本中重建 V3、重新执行修订前后相同的 10 seed 审查、重建对比报告及盲审包、运行交接包的 13 项测试，并逐字节核对数据及审查结果。原始交接文件不会被覆盖，输出写入 `artifacts/`。审查流程不调用付费模型 API。

产物：

- `artifacts/baseline/audit.json`、`artifacts/baseline/audit.md`：修订前审查。
- `artifacts/v3/audit.json`、`artifacts/v3/audit.md`：修订后审查，包含重复/相似样本、词汇、异常、字段及结构特征。
- `artifacts/v3/三问题_专项处理与验收报告.md`：修订前后结果与 T014/T049 机制比较。
- `artifacts/integrity_report.json`：完整性、输入字段、T049 机制、待独立验收状态及来源复现核对。
- `artifacts/run_log.txt`、`artifacts/combined_test_results.txt`：本次实际运行输出。

仓库测试另运行 `uv run --extra dev pytest -q`；交接脚本作为原件保存，位于 Ruff 的窄范围排除目录。T011、T042、T044 的 Challenge 难度仍需独立审阅，程序化通过不构成正式验收。
