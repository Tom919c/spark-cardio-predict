# 测试说明

在当前成员的开发环境中执行：

```bash
conda activate <your-conda-environment>
python -m pytest -q
python app.py
```

`<your-conda-environment>` 替换为本机实际环境名，不要求所有成员使用同名环境。

基础验收检查：

1. `GET /` 显示机构端首页。
2. `GET /health` 返回成功。
3. `GET /api/data/preprocess` 返回双标签训练表信息。
4. `GET /api/risk/summary` 正确显示两个模型当前是否就绪。
5. 模型未就绪时，`POST /api/risk/predict` 返回可理解的错误消息。
6. 打开 `/dashboard` 与 `/risk-report`，确认 B/C 端切换入口可用。

负责人确认数据、接口和测试通过后，才执行正式训练：

```bash
curl "http://127.0.0.1:5000/api/risk/train?run_label=phase1"
```

训练后再次检查：

1. `GET /api/risk/summary` 显示心脏事件和脑卒中模型均已就绪。
2. `POST /api/risk/predict` 返回两个概率、综合五级风险和干预建议。
3. `docs/training_results.md` 与模型清单指标一致。

在 WSL2 Ubuntu 或 VMware Ubuntu 中执行 Spark 作业前，复制 `.env.example` 为 `.env` 并填写当前环境可访问的 HDFS 参数。默认输入路径为 `HDFS_INPUT_PATH`，也可以显式传入：

```bash
./scripts/start_spark_jobs.sh <hdfs-input-file>
```

作业完成后确认 staging/feature、群体统计和重点筛查输出目录均已生成。

阶段二双模型批量评分使用以下命令模板。`<hdfs-input>`、`<hdfs-output>` 和 `<model-manifest>` 替换为当前成员环境中的路径：

```bash
spark-submit --master local[2] scripts/run_phase2_analysis.py \
  --input <hdfs-input> \
  --output <hdfs-output> \
  --task-id <task-id> \
  --model-manifest <model-manifest> \
  --score-engine pandas
```

模型清单中的模型文件必须位于清单所在目录或使用可访问的相对路径。作业会自动分发项目 `src` 源码包，供 Spark worker 反序列化自定义模型类。

先在当前 Linux 环境中确认 Hadoop、Spark 和 HDFS 服务：

```bash
<hadoop-home>/bin/hdfs version
<hadoop-home>/bin/hdfs getconf -confKey fs.defaultFS
jps
<hadoop-home>/bin/hdfs dfsadmin -report
<spark-home>/bin/spark-submit --version
```

Spark 评分依赖安装在执行 `spark-submit` 的当前环境中：

```bash
conda activate <your-linux-conda-environment>
python -m pip install -r requirements-spark.txt
```

本项目当前配置的 `SPARK_SCORE_ENGINE=pandas` 固定使用向量化 Pandas UDF。缺少 `pyarrow` 或其他评分依赖时，作业应直接失败并先补齐环境，不切换到另一套评分实现。

如果模型使用项目内的自定义类，阶段二作业会自动分发 `src` 源码包，worker 不需要额外手工设置 `PYTHONPATH`。若仍出现 `ModuleNotFoundError: No module named 'src'`，请确认运行的是仓库当前版本的 `scripts/run_phase2_analysis.py`。

如果 WebHDFS 重定向到不可达的 DataNode 主机名，在本机 `.env` 配置 `HDFS_DATANODE_HOST=<当前环境可访问的地址>`，并确保 NameNode、DataNode 和 WebHDFS 端口已开放。
