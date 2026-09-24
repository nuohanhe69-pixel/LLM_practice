# 第三步之后的三个问题：一次性定向修订交接 V3

## 结果范围

- `baseline/candidates_full.jsonl`：不可变 V2 源快照（SHA256 在脚本中锁定）；`baseline/audit.json`：修订前检查结果。
- `candidates_full_v3.jsonl`：仅供开发期内部审查，包含 GT 和设计信息，**禁止用于 Prompt**。
- `model_input_only_v3.jsonl`：仅 `id` 和 `error_text`。仍是候选，不得启动正式五模型实验。
- `change_log.json`：每处文本修订和四条内部快捷审查标记的撤销依据。T049 是明确记账的**机制替换例外**，不是措辞润色。
- `audit.json`、`audit.md`：同脚本、同 seed 的修订后程序化检查；`三问题_专项处理与验收报告.md`：修订前后比较与仍待验收之处。
- `三条容量样本_内部对抗性复核.md`：仅供内部，不能替代盲审。
- `仅发审阅者_匿名容量镜像盲审包.zip`：**唯一可以直接转给外部审阅者的文件**。与本内部交接包分开提供，勿将包含 GT 的完整包发给他们。
- `内部_六条盲审ID映射.csv`：必须内部保密。

## 本地可复现

需要 Python 和 `audit_requirements.txt` 对应的 NumPy/scikit-learn（不要污染正式 API 实验环境）。

```bash
python -m pip install -r audit_requirements.txt
python revise_candidates.py --input baseline/candidates_full.jsonl --out .
python challenge_shortcut_audit.py --dataset candidates_full_v3.jsonl --model-input model_input_only_v3.jsonl --out . --seeds 10
python build_handoff.py
python -m unittest discover -s . -p 'test*py' -v
```

审查脚本 `challenge_shortcut_audit.py` 原样沿用上一轮，不要修改调参或重新选择数据以追求分类器随机准确率。`build_handoff.py` 依赖旧/新审查 JSON，按上述顺序先完成新版本 audit 再执行 build_handoff。

## 建议的 Codex 任务边界

这份包是已完成的开发期定向修订和程序化审查，不要求 Codex 重新生成题目，也不授予调整 GT 的权限。若要集成仓库，先比对当前 `main`，然后只新增 `experiments/challenge_audit_v1/` 中相应数据和脚本，执行只读审查并提供结果/Commit SHA。不要碰 `dataset_v1.csv`、Prompt、正式 pipeline、历史 Pilot 结果、任何 .env/API key。直接在当前分支提交推送之前列出拟操作文件供用户确认。

## 阻塞条件

现有三个待外部盲审样本（T011、T042、T044）**没有宣称通过 Challenge 准入**。此外全套 60 条还未经过独立 GT 审查，编辑元数据的 `surface_exception / surface_cues` 不代表已人工核实，不可冻结 Test。
