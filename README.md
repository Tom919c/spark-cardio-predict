# CardioSpark

基于 Spark 的心脑血管风险预测与健康管理平台。阶段一仅包含两项独立风险：心脏事件与脑卒中。

## 功能

- DWD 双标签数据：9 个特征、`label_heart`、`label_stroke`。
- 两个随机森林模型：样本权重与 Isotonic 概率校准。
- B 端：成都市仿真人口风险画像、行政区统计、重点随访名单。
- C 端：个人双风险概率、五级评估与干预建议。

## 运行

Windows 本地演示和测试使用 Conda 环境 `bigdata`。先确认数据已放入 `data/`，再执行：

```powershell
conda run -n bigdata pytest -q
conda run -n bigdata python app.py
```

打开 `http://127.0.0.1:5000/dashboard`，可在顶部切换机构端和个人风险评估。正式训练前先完成测试和接口验收，训练时执行：

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/risk/train?run_label=phase1
```

将 `.env.example` 复制为 `.env` 后按本机环境调整。数据、模型和 `mydocs` 均不进入 Git。没有仿真数据时，使用以下命令生成默认 200 万条记录：

```powershell
conda run -n bigdata python scripts/generate_chengdu_health_data.py
```

## 跨环境

- Windows：Flask、页面调试、单元测试。
- WSL2 Ubuntu 22.04 或 VMware Linux：Hadoop/Spark 伪分布式作业。
- 每个 Linux 环境各自 `git clone` 仓库；不要跨 Windows/WSL/VM 共享同一个工作目录运行。
- Spark 作业默认执行 ETL、特征插补、群体聚合和重点筛查；输入路径可由 `.env` 的 `HDFS_INPUT_PATH` 配置。

详见 `docs/`。

团队协作和智能体开发必须遵守：`docs/团队协作与智能体开发规范.md`。
