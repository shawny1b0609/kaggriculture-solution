# Kaggriculture 比赛方案开源

[English README](README.md) · [英文完整报告](docs/report/REPORT.md) · [完整架构图](docs/report/figures/detail/architecture_detailed.png)

这是最终提交 **56707930** 的独立开源版本。主线思路是：用可靠的预制经营线路保证生产，再通过市场交易，以及兼容原调度的作物、牲畜替换来参与竞争。最终版本不是 PPO，也不是无限分叉的决策森林。

## 怎么看

- 想了解整体方案：先看英文首页和完整架构图，再看报告。
- 想运行：按下面命令安装、测试、打包。
- 想继续研究：看 [架构与代码对应](docs/ARCHITECTURE.md)、[历史路径](docs/RESEARCH_HISTORY.md)、[验证口径](docs/EVALUATION.md)。
- 想重新训练市场预测器：看 [数据准备](docs/DATA.md) 和 [复现指南](docs/REPRODUCING.md)。

原始研究目录没有被改写；这里不包含登录信息、几十 GB 的回放和临时缓存。

## 安装与运行

```bash
git clone https://github.com/shawny1b0609/kaggriculture-solution.git
cd kaggriculture-solution
conda create -n kaggriculture python=3.11 -y
conda activate kaggriculture
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/evaluate.py --seeds 0
python scripts/package_agent.py
```

已有 conda CV 环境且依赖齐全的话，可以直接使用它，不必再建环境。运行最终模型不需要重新训练，也不要求显卡。

最后生成的 `dist/submission.tar.gz` 是比赛提交包，打包命令不会自动向 Kaggle 提交。

## 成绩怎么看

2026 年 10 月 15 日查询到的线上分数为 **2226.6**；925 场已完成公开对局中 535 胜，胜率 **57.84%**，平均终局资金约 **104,604**，平均资金差 **+890**。这是有日期的历史快照，不是当前排名。

报告也保留本地验证结果。固定产出对手的市场实验与真实线上对局不是同一种样本，分开看即可。验证会检查产出、资金及库存账，不会把“程序跑完”直接当成“经营稳定”。

## 再次修改并更新 GitHub

在此仓库目录修改文件、运行测试后：

```bash
git status
git add README.md
git commit -m "Update documentation"
git push
```

上面只演示提交一个文档；需要更新其他文件时，明确指定文件。不要把 `kaggle.json`、令牌、回放目录或个人浏览器数据加入仓库。

原创代码使用 MIT；第三方引擎及回放衍生内容的来源见 [声明](THIRD_PARTY_NOTICES.md)。
