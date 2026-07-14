# CardioSpark

CardioSpark 是一个面向教学、科研演示和项目答辩的心脑血管风险筛查与健康管理平台。系统围绕“个人筛查 + 机构群体分析 + 数据平台工程”三条主线实现：个人端对心脏事件和脑卒中风险进行双模型评估，机构端对上传的人群数据进行分块分析、区域聚合、重点随访和快照趋势比较，数据平台同时支持本地 SQLite/Pandas 模式和 HDFS/Spark 分布式模式。

> 重要边界：本系统是课程项目中的风险筛查原型。模型输出用于风险排序、健康提示和工程流程演示，不代表临床诊断、处方建议或真实年度发病概率。机构端仿真居民数据中的姓名、电话和编号均为虚拟数据，不能用于真实业务。

## 1. 已实现能力

### 1.1 个人风险筛查

- 使用 9 个统一特征同时评估心脏事件和脑卒中风险：`age`、`gender`、`bmi`、`cholesterol`、`diabetes`、`hypertension`、`smoker`、`alcohol`、`exercise`。
- 两个独立的 XGBoost 二分类模型，不把心脏和卒中标签简单相加。
- 训练阶段使用训练集拟合 `IterativeImputer`，避免测试集信息泄漏。
- 使用 Isotonic Regression 做概率校准，并为两个疾病分别优化决策阈值。
- 返回心脏概率、卒中概率、双模型综合五级风险、关键指标解释、SHAP 风格模型贡献和结构化健康行动建议。
- 支持“情景模拟（what-if）”：修改允许的健康因素，比较基线与情景结果。
- 支持不超过 5,000 行的小批量双模型评估；更大数据应使用机构上传任务。
- 使用 `X-Client-ID` 和 HMAC-SHA256 保存匿名个人评估历史，不保存浏览器原始身份标识。

### 1.2 机构端群体分析

- 支持 CSV 分片上传、任务状态持久化和后台异步分析。
- 本地模式使用分块 Pandas 处理，不要求一次性将完整文件载入前端。
- 计算总人数、心脏高风险率、卒中高风险率、双风险共病率、年龄段统计、风险因素分布和区域统计。
- 生成重点随访名单；姓名和电话在结果中脱敏，社区样本低于隐私阈值时隐藏人数及比例。
- 支持数据集预览、分析结果查询和任务失败状态持久化。
- 任务中的 `storage_engine` 与 `compute_engine` 由服务端真实运行模式生成，浏览器不能伪造 HDFS/Pandas 切换。

### 1.3 快照趋势

- 上传任务要求填写 `data_period`，格式为 `YYYY-MM` 或 `YYYY-Qn`。
- 历史结果只从已登记且分析成功的数据集读取，不伪造不存在的时间点。
- 只有 `data_period`、`coverage_level`、`coverage_name`、`model_version` 兼容的快照才会放入同一条趋势线。
- 趋势接口返回每个时期的心脏风险率、卒中风险率、共病率、样本量和模型版本。

### 1.4 SQLite/MySQL 元数据

任务、上传分片、数据集登记、分析状态和匿名个人评估历史保存在元数据库中。默认使用 SQLite；MySQL DAO 与 SQLite DAO 保持相同业务方法契约，切换数据库只需修改配置和初始化数据库，不需要修改服务层或控制器。

### 1.5 Hadoop/HDFS/Spark 链路

`DATA_MODE=local` 时使用本地文件系统 + Pandas；`DATA_MODE=hdfs` 时使用真实 WebHDFS/HDFS 上传和 `spark-submit` 作业：

```text
CSV 分片
   ├─ local 模式：本地文件系统 → 分块 Pandas → 本地 ADS
   └─ hdfs 模式：WebHDFS raw → Spark ETL/校验/评分 → HDFS feature/ADS
                                      └→ Flask 读取聚合结果
```

Spark 作业负责数据读取、字段校验、缺失值处理、双模型批量评分、区域聚合和 ADS 结果输出。当前阶段二评分引擎固定为 Pandas UDF，依赖 `pyarrow` 的 Spark Python 环境必须与 `spark-submit` 使用的环境一致。

## 2. 技术栈

- Web：Flask 3.0.3、原生 HTML/CSS/JavaScript、ECharts 静态资源。
- 数据处理：Python、Pandas、NumPy、scikit-learn。
- 机器学习：XGBoost、IterativeImputer、Isotonic Regression、joblib。
- 分布式计算：Apache Spark 4.1.2、HDFS/WebHDFS、Pandas UDF、PyArrow。
- 元数据：SQLite（默认）、MySQL/PyMySQL（可选）。
- 测试：pytest。

正式模型清单记录的训练运行时为 Python 3.12.3、pandas 2.2.2、scikit-learn 1.5.1、joblib 1.4.2、xgboost 3.1.2。实际部署环境应优先按仓库中的依赖文件安装。

## 3. 目录说明

```text
app.py                         Flask 启动入口
config/                        配置类和环境变量解析
src/controllers/               页面和 REST API 控制器
src/services/                  训练、预测、上传、分析、趋势和能力检查服务
src/ml/                        特征工程、训练、预测、模型注册、解释
src/dao/                       SQLite、MySQL、HDFS 数据访问层
src/spark_jobs/                Spark 侧可复用逻辑
templates/                     dashboard、个人风险报告、结果报告、解释页面
static/                        CSS、JavaScript、ECharts、地图资源
resources/knowledge/           可审计健康干预知识库
data/raw/                      训练输入数据
data/models/                   正式模型和 active_models.json
data/localstorage/             SQLite、上传文件和本地 ADS 结果
data/demo/                     仿真机构连续快照数据
scripts/                       数据生成、训练准备、HDFS、Spark 和知识库脚本
docs/                          架构、接口、测试、答辩证据和设计说明
tests/                         自动化测试
```

## 4. 环境准备

### 4.1 创建 Python 环境

Windows PowerShell：

```powershell
cd D:\Code\Pycharm\spark-cardio-predict
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux、WSL2 或 VMware Ubuntu：

```bash
cd /path/to/spark-cardio-predict
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果只运行本地 Flask 和个人评估，`requirements.txt` 即可。启用 Spark/Pandas UDF 时，还要在执行 `spark-submit` 的 Python 环境安装：

```bash
python -m pip install -r requirements-spark.txt
```

### 4.2 配置环境变量

复制模板：

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

本地模式最小配置：

```dotenv
APP_ENV=development
SECRET_KEY=change-this-before-deployment
ASSESSMENT_HASH_KEY=change-this-independent-hmac-key
HOST=127.0.0.1
PORT=5000
DEBUG=false
DATA_MODE=local
DATABASE_TYPE=sqlite
DATABASE_PATH=data/localstorage/app.db
DATASET_FILE_PATH=data/raw/CVD_Standard_DWD_refined.csv
MODEL_OUTPUT_DIR=data/models
MODEL_MANIFEST_PATH=data/models/active_models.json
```

程序会自行读取项目根目录 `.env`，不依赖 `python-dotenv`。部署时必须替换默认密钥，并且不要把个人 HDFS、MySQL 密码或本机路径提交到仓库。

## 5. 数据准备与模型训练

### 5.1 生成训练仿真数据

没有训练数据时，生成默认 2,000,000 条记录：

```powershell
python scripts/generate_chengdu_health_data.py
```

常用参数：

```powershell
python scripts/generate_chengdu_health_data.py `
  --rows 2000000 `
  --seed 20260714 `
  --output data/raw/CVD_Standard_DWD.csv `
  --chunk-size 250000
```

### 5.2 重建训练标签和质量档案

训练前建议执行：

```powershell
python scripts/refine_training_dataset.py
```

默认读取 `data/raw/CVD_Standard_DWD.csv`，输出 `data/raw/CVD_Standard_DWD_refined.csv` 及数据质量档案。该步骤构建心脏和卒中筛查代理标签，不等同于真实发病率或临床诊断标签。

### 5.3 启动服务

Windows：

```powershell
python app.py
# 或
run.bat
```

Linux：

```bash
python3 app.py
# 或
./run.sh
```

默认地址：

- 机构端首页：`http://127.0.0.1:5000/dashboard`
- 个人风险评估：`http://127.0.0.1:5000/risk-report`
- 专业模型解释：`http://127.0.0.1:5000/shap-analysis`
- 结果报告：`http://127.0.0.1:5000/result-report`

### 5.4 训练前检查和正式训练

服务启动后检查：

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
Invoke-RestMethod http://127.0.0.1:5000/api/risk/summary
Invoke-RestMethod http://127.0.0.1:5000/api/risk/train-estimate
```

正式训练是 POST 请求，会训练心脏和卒中两个模型，并更新 `active_heart.joblib`、`active_stroke.joblib` 和 `active_models.json`：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:5000/api/risk/train `
  -Method Post `
  -ContentType 'application/json' `
  -Body (@{run_label='phase1'; rounds=1} | ConvertTo-Json)
```

Linux/macOS：

```bash
curl -X POST http://127.0.0.1:5000/api/risk/train \
  -H 'Content-Type: application/json' \
  -d '{"run_label":"phase1","rounds":1}'
```

`rounds` 为多轮训练次数；未传入时服务默认训练 1 轮。训练过程和指标追加写入 `docs/training_results.md`；模型清单同时记录特征契约、SHA256、运行时版本和指标。

## 6. 个人端使用

### 6.1 页面操作

打开 `/risk-report`，填写 9 个必填特征后开始评估。血压、空腹血糖、家族史可作为辅助健康提示字段提交，但不进入当前九特征概率模型。评估结果包含双风险概率、五级风险、关键因素、健康行动、复查建议和急症边界。

### 6.2 单人 API 示例

```powershell
$body = @{
  age=55; gender=1; bmi=24.5; cholesterol=2
  diabetes=0; hypertension=1; smoker=0; alcohol=0; exercise=1
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:5000/api/risk/predict `
  -Method Post -ContentType 'application/json' -Body $body
```

有效范围：年龄 18~95，BMI 10.3~79.8，胆固醇 1~3，吸烟 0~2，其余二元编码为 0/1。可选血压范围为收缩压 60~260、舒张压 30~180；空腹血糖 1~40；家族史为 0/1。

### 6.3 情景模拟和小批量评估

```bash
curl -X POST http://127.0.0.1:5000/api/risk/what-if \
  -H 'Content-Type: application/json' \
  -d '{
    "baseline":{"age":55,"gender":1,"bmi":30,"cholesterol":3,"diabetes":0,"hypertension":1,"smoker":1,"alcohol":1,"exercise":0},
    "scenario":{"age":55,"gender":1,"bmi":27,"cholesterol":2,"diabetes":0,"hypertension":0,"smoker":0,"alcohol":0,"exercise":1}
  }'
```

批量接口要求 `rows` 为非空数组，最多 5,000 行；超过该规模应改用机构上传任务。

## 7. 机构端上传与分析

### 7.1 生成连续月份演示数据

```powershell
python scripts/generate_demo_population_snapshots.py `
  --rows 20000 `
  --months 6 `
  --start-period 2026-01 `
  --output-dir data/demo/organization_snapshots
```

生成的文件不包含疾病标签，包含虚拟居民身份字段和 `data_period`，适合连续上传观察模型高风险率趋势。

### 7.2 分片上传 API

第一步创建任务：

```bash
curl -X POST http://127.0.0.1:5000/api/upload/tasks \
  -H 'Content-Type: application/json' \
  -d '{"filename":"chengdu_demo_population_2026-01.csv","file_size":2824072,"total_chunks":1,"data_period":"2026-01"}'
```

返回 `task_id` 后上传分片：

```bash
curl -X POST http://127.0.0.1:5000/api/upload/tasks/<task_id>/chunks \
  -F 'chunk=@data/demo/organization_snapshots/chengdu_demo_population_2026-01.csv' \
  -F 'chunk_index=0'
```

合并并提交后台分析：

```bash
curl -X POST http://127.0.0.1:5000/api/upload/tasks/<task_id>/finalize
```

轮询任务：

```bash
curl http://127.0.0.1:5000/api/task/status/<task_id>
```

任务进入 `success` 后，根据返回的 `dataset_id` 获取结果：

```bash
curl http://127.0.0.1:5000/api/datasets/<dataset_id>/preview
curl http://127.0.0.1:5000/api/datasets/<dataset_id>/result
```

前端会自动完成分片上传和轮询。生产环境应通过页面或等价的分片客户端上传，不要把超大文件直接放进单次 JSON 请求。

### 7.3 快照趋势

至少成功分析两个时期后：

```bash
curl http://127.0.0.1:5000/api/trends/snapshots
curl -X POST http://127.0.0.1:5000/api/trends/compare \
  -H 'Content-Type: application/json' \
  -d '{"snapshots":[...]}'
```

如果快照模型版本、覆盖层级或覆盖区域不一致，接口会拒绝把它们放进同一趋势线，而不是拼接不具备可比性的结果。

## 8. SQLite 与 MySQL 切换

### SQLite（默认）

无需额外服务，默认数据库为 `data/localstorage/app.db`：

```dotenv
DATABASE_TYPE=sqlite
DATABASE_PATH=data/localstorage/app.db
```

### MySQL

安装 MySQL 并创建数据库后执行：

```bash
mysql -u root -p < scripts/init_mysql.sql
```

配置：

```dotenv
DATABASE_TYPE=mysql
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=cardiospark
MYSQL_PASSWORD=<password>
MYSQL_DATABASE=cardiospark
```

SQLite/MySQL 只保存任务和历史等元数据，不保存完整居民明细。业务层通过数据库工厂和统一 DAO 契约访问数据库，切换时不需要修改 Controller 或 Service 代码。

## 9. HDFS + Spark 运行

### 9.1 配置 HDFS 模式

```dotenv
DATA_MODE=hdfs
HDFS_NAMENODE_URI=hdfs://localhost:9000
HDFS_WEB_URL=http://localhost:9870
HDFS_USER=<linux-user>
HDFS_RAW_PATH=/user/<linux-user>/cardiospark/raw
HDFS_INPUT_PATH=/user/<linux-user>/cardiospark/raw/chengdu_resident_health_simulated.csv
HDFS_STAGING_PATH=/user/<linux-user>/cardiospark/staging
HDFS_FEATURE_PATH=/user/<linux-user>/cardiospark/feature
HDFS_UPLOAD_ROOT=/user/<linux-user>/cardiospark/raw/_uploads
HDFS_RESULT_ROOT=/user/<linux-user>/cardiospark/feature/ads
SPARK_SUBMIT_BIN=spark-submit
SPARK_PYTHON=<spark-python>
SPARK_SCORE_ENGINE=pandas
```

服务端会根据 `DATA_MODE` 强制选择 HDFS 和 Spark；浏览器提交的 `use_hdfs` 字段不会改变服务端模式。

### 9.2 检查 Hadoop、HDFS 和 Spark

```bash
hdfs version
hdfs getconf -confKey fs.defaultFS
hdfs dfs -ls /
hdfs dfsadmin -report
jps
spark-submit --version
curl http://localhost:9870
```

Windows Hadoop 若因 `%HADOOP_HOME%/logs` 无权限导致 `hadoop version` 失败，请先创建 logs 目录并授予当前用户写权限。HDFS 客户端可创建不代表远程集群已经连通，正式运行前应完成一次小文件上传。

### 9.3 上传训练/分析输入到 HDFS

```bash
python scripts/upload_to_hdfs.py data/raw/chengdu_resident_health_simulated.csv \
  --hdfs-dir /user/<linux-user>/cardiospark/raw
```

### 9.4 运行 Spark 阶段二分析

```bash
spark-submit --master local[2] scripts/run_phase2_analysis.py \
  --input hdfs://localhost:9000/user/<linux-user>/cardiospark/raw/chengdu_resident_health_simulated.csv \
  --output hdfs://localhost:9000/user/<linux-user>/cardiospark/feature/ads/demo-task \
  --task-id demo-task \
  --model-manifest data/models/active_models.json \
  --score-engine pandas \
  --min-group-size 5
```

也可以使用：

```bash
./scripts/start_spark_jobs.sh <hdfs-input-file>
```

Spark 作业完成后应检查 staging、feature 和 ADS 目录中的汇总、区域统计和重点筛查结果。缺少 `pyarrow` 或 Spark worker 无法反序列化项目模型类时，作业应修复环境后重试，不会静默切换到另一种评分实现。

### 9.5 Hive 外部表

`scripts/init_hive.hql` 提供 HDFS 数据目录对应的 Hive 外部表初始化脚本，具体表位置按 `.env` 中的 HDFS 路径调整。

## 10. REST API 总览

所有 API 均返回统一结构：

```json
{"success": true, "message": "...", "data": {}}
```

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 服务健康检查 |
| GET | `/api/status` | 系统阶段状态 |
| GET | `/api/capabilities` | 数据库、HDFS、Spark、模型能力快照 |
| GET | `/api/data/profile` | 当前训练数据配置 |
| GET | `/api/data/preview` | 训练数据预览 |
| GET | `/api/data/preprocess` | 数据字段和特征工程校验 |
| GET | `/api/risk/summary` | 双模型状态和特征契约 |
| GET | `/api/risk/train-estimate` | 训练耗时估算，不写模型 |
| POST | `/api/risk/train` | 训练并注册双模型 |
| POST | `/api/risk/predict` | 单人双风险评估 |
| POST | `/api/risk/what-if` | 基线与情景比较 |
| POST | `/api/risk/batch` | 最多 5,000 行小批量评估 |
| GET | `/api/assessments` | 匿名个人评估历史 |
| GET | `/api/assessments/<assessment_id>` | 单条匿名评估历史 |
| GET | `/api/analysis/dashboard` | 机构群体汇总和区域统计 |
| GET | `/api/analysis/follow-ups?limit=100` | 重点随访名单 |
| POST | `/api/upload/tasks` | 创建上传任务 |
| POST | `/api/upload/tasks/<task_id>/chunks` | 上传分片 |
| POST | `/api/upload/tasks/<task_id>/finalize` | 合并并提交分析 |
| GET | `/api/task/status/<task_id>` | 查询任务状态 |
| GET | `/api/datasets/<dataset_id>/preview` | 脱敏数据集预览 |
| GET | `/api/datasets/<dataset_id>/result` | 获取分析结果 |
| GET | `/api/trends/snapshots` | 获取历史快照 |
| POST | `/api/trends/compare` | 比较兼容快照 |

页面入口为 `/`、`/dashboard`、`/risk-report`、`/shap-analysis` 和 `/result-report`。

## 11. 安全、隐私和工程约束

- 默认关闭 Flask Debug 和 reloader；生产部署应使用独立密钥和正式 WSGI/反向代理方案。
- 响应包含 CSP、X-Frame-Options、Referrer-Policy、Permissions-Policy 和 `nosniff` 等安全头。
- API 输出中的动态文本进行 HTML 转义；上传请求有单请求和完整文件大小限制。
- 个人历史使用 HMAC 匿名归属；机构名单对姓名和电话脱敏。
- 区域聚合遵守 `PRIVACY_MIN_GROUP_SIZE`，默认小于 5 人的分组隐藏敏感统计。
- 模型文件只通过 `active_models.json` 定位，清单校验版本、特征契约和文件存在性，防止加载错误模型。
- 训练插补器只在训练子集拟合；机构标签不能覆盖正式双模型概率结果。
- 不要提交 `.env`、个人 HDFS 路径、MySQL 密码、真实居民数据或临时上传目录。

## 12. 测试与验收

运行全部测试：

```bash
python -m pytest -q
```

当前项目的核心回归测试覆盖个人预测、模型清单、健康建议、数据库、上传任务、HDFS/Spark 能力、快照趋势和页面信息架构。前端 JavaScript 可额外检查：

```bash
node --check static/js/phase2.js
```

基础验收顺序：

1. `/health` 返回成功；
2. `/api/capabilities` 显示实际数据库、存储、计算和模型能力；
3. `/api/risk/summary` 显示心脏和卒中模型均就绪；
4. `/api/risk/predict` 返回双风险概率和结构化建议；
5. 机构端上传一份 CSV，任务最终进入 `success`；
6. `/api/datasets/<dataset_id>/result` 返回聚合结果；
7. 连续上传两个以上 `data_period` 后，趋势比较接口返回 `available=true`；
8. HDFS 模式下确认 Spark 作业和 HDFS ADS 目录均有产物。

## 13. 常见问题

### 模型无法加载

检查：

```bash
curl http://127.0.0.1:5000/api/risk/summary
```

确认 `data/models/active_models.json`、`active_heart.joblib`、`active_stroke.joblib` 均存在，且服务已重启。如果特征契约或运行环境确实发生变化，再通过 `POST /api/risk/train` 重新训练。

### 机构任务一直失败

先查询任务：

```bash
curl http://127.0.0.1:5000/api/task/status/<task_id>
```

查看返回的 `error_message`、`storage_engine` 和 `compute_engine`。本地模式检查 CSV 路径和字段；HDFS 模式检查 NameNode、DataNode、WebHDFS、`spark-submit` 和 `pyarrow`。

### 趋势图为空

确认至少有两个分析成功的快照，并且所有快照的 `data_period`、`coverage_level`、`coverage_name`、`model_version` 兼容。模型重新训练后，不同模型版本的历史快照不能直接混成一条趋势线。

### MySQL 或 HDFS 能力显示未就绪

`/api/capabilities` 的就绪状态反映当前实际配置和客户端发现结果。MySQL 需要先初始化数据库并检查账号权限；HDFS 需要检查 Hadoop CLI、WebHDFS 地址和远端集群连通性。

## 14. 关键交付材料

- `docs/architecture.md`：系统架构和模块边界。
- `docs/api_design.md`：接口契约和业务边界。
- `docs/data_platform.md`：SQLite/MySQL、HDFS、Spark 实际链路。
- `docs/model_design.md`：模型设计、校准和解释。
- `docs/training_results.md`：训练记录和评估指标。
- `docs/defense_evidence.md`：用于答辩展示的关键开发过程证据。
- `docs/test_guide.md`：测试和运行验收说明。
- `scripts/init_mysql.sql`、`scripts/init_hive.hql`：数据库和 Hive 初始化脚本。

## 15. 项目定位总结

CardioSpark 不只是一个单模型预测页面，而是一个包含数据准备、标签构建、无泄漏训练、概率校准、可解释性、健康知识库、个人历史、机构异步上传、隐私聚合、快照趋势、SQLite/MySQL 抽象和 Hadoop/Spark 分布式链路的完整课程项目。默认本地模式便于开发和答辩，切换到 HDFS/Spark 后仍沿用同一页面和 API 契约。
