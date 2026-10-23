<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/readme/hero-dark.svg">
  <img src="./assets/readme/hero.svg" width="100%" alt="PROSE">
</picture>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.14-1D6B73?style=flat-square&logo=python&logoColor=white&labelColor=16191E" alt="Python 3.14">
  <img src="https://img.shields.io/badge/pytorch-2.11-C2410C?style=flat-square&logo=pytorch&logoColor=white&labelColor=16191E" alt="PyTorch 2.11">
  <img src="https://img.shields.io/badge/env-uv-6A6F77?style=flat-square&logo=uv&logoColor=white&labelColor=16191E" alt="uv">
  <img src="https://img.shields.io/badge/license-MIT-1D6B73?style=flat-square&labelColor=16191E" alt="MIT 许可证">
  <br>
  <a href="./README.md">English</a> · <b>简体中文</b>
</p>

**PROSE**（过程语义历史动力学）是一种即插即用的工业预测历史表示：LLM 在训练前离线描述一次各变量，
运行时这些冻结语义让历史按过程推进而非时间流逝老化。输出与输入同形，任意主干无需改动即可使用。

## 快速开始

```bash
uv sync --extra train
uv run python scripts/train.py --dataset thickener_dewatering --backbone gru --method prose \
  --seed 2024 --set training.max_epochs=2
```

三个数据集的处理后数据（`windows.npz`）与冻结语义已随仓库提供，训练无需 API 密钥；没有 GPU 时加 `--device cpu`。

| 数据集 | 任务 |
| --- | --- |
| `ladle_preheating` | 60 步历史 → 未来 5 步温度（现场数据，已缩放到 [0, 1]） |
| `thickener_dewatering` | 30 分钟历史 → 底流浓度（仿真） |
| `indpensim` | 120 步历史 → 离线青霉素浓度 |

## 用法

- **可选项：** `--backbone` 可选 `gru lstm transformer informer mamba itransformer patchtst moderntcn`；
  `--method` 为 `base` 或 `prose`；任意配置字段可用 `--set section.field=value` 覆盖。
- **批量运行：** `uv run python scripts/run_study.py [configs/studies/smoke.yaml]` 在全部可见 GPU 上运行所列组合
  （默认读取 `configs/run.yaml`），`scripts/collect_results.py` 把指标汇总为 CSV。
- **重建语义：** 把 `configs/env/api.env` 复制为 `api.local.env` 并填写接口，执行 `uv sync --extra llm` 后依次运行
  `scripts/build_semantic_cards.py --dataset all --call-api` 与
  `scripts/build_semantic_artifact.py --dataset all --source llm`。LLM 只读取 `task_spec.json` 与 `PROCESS.md`。

## 许可证

MIT。本项目为研究代码，不适用于生产控制。
