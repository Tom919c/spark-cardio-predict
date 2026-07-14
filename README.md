# CardioSpark

基于 Spark 的心脑血管风险预测与健康管理平台。阶段一仅包含两项独立风险：心脏事件与脑卒中。

## 功能

- DWD 双标签数据：9 个特征、`label_heart`、`label_stroke`。
- 两个独立 XGBoost 模型：自然患病率训练、Isotonic 概率校准与疾病专属决策阈值。
- B 端：成都市仿真人口风险画像、行政区统计、重点随访名单。
- C 端：个人双风险概率、五级评估与干预建议。

## 运行
Windows 或 Linux 环境先激活本机用于项目的 Conda 环境，环境名称由各成员自行决定：

```bash
conda activate <your-conda-environment>
```

然后在已激活的环境中执行项目命令。将 `.env.example` 复制为 `.env` 后按本机环境调整。数据、模型和 `mydocs` 均不进入 Git。没有仿真数据时，使用以下命令生成默认 200 万条记录：

```powershell
python scripts/generate_chengdu_health_data.py
```

训练模型前，先基于已清洗的 DWD 数据重建可复现的筛查代理标签：

```powershell
python scripts/refine_training_dataset.py
```

该步骤输出 `CVD_Standard_DWD_refined.csv` 及同名数据质量档案。标签用于课程项目中的风险筛查建模，不等同于真实年度发病率、临床诊断或人群流行病学统计。

## 运行边界

- Flask 和 Spark 在当前成员的开发环境中原生运行；该环境可以是 Windows、WSL2 Ubuntu 或 VMware Ubuntu。
- VMware 或 WSL2 中的 Hadoop/HDFS 负责保存 `raw`、`staging`、`feature` 和 ADS 结果，运行位置不改变接口和结果契约。
- 每个成员在自己的开发环境中单独 `git clone` 仓库，不跨 Windows、WSL2 或 VMware 共享工作目录。
- Spark 作业默认执行 ETL、特征插补、双模型批量评分、群体聚合和重点筛查；输入路径由 `.env` 的 `HDFS_INPUT_PATH` 配置。
- 阶段二评分固定使用 Pandas UDF，依赖见 `requirements-spark.txt`；这些依赖必须安装到 `SPARK_PYTHON` 指向的 Python 环境。缺少依赖时作业直接失败并在任务中记录原因，不自动切换评分实现。
- WebHDFS 如果把请求重定向到当前环境不可达的 DataNode 主机名，在本机 `.env` 中配置 `HDFS_DATANODE_HOST`；该配置不提交到仓库。
- Windows、WSL2 与 VMware 只在运行位置上不同，脚本参数、模型清单格式、HDFS 目录结构和结果契约保持一致。

详见 `docs/`。

团队协作和智能体开发必须遵守：`docs/团队协作与智能体开发规范.md`。
