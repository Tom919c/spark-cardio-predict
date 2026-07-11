# 测试说明

在 Windows 环境执行：

```powershell
conda run -n bigdata pytest -q
conda run -n bigdata python app.py
```

基础验收检查：

1. `GET /` 显示机构端首页。
2. `GET /health` 返回成功。
3. `GET /api/data/preprocess` 返回双标签训练表信息。
4. `GET /api/risk/summary` 正确显示两个模型当前是否就绪。
5. 模型未就绪时，`POST /api/risk/predict` 返回可理解的错误消息。
6. 打开 `/dashboard` 与 `/risk-report`，确认 B/C 端切换入口可用。

负责人确认数据、接口和测试通过后，才执行正式训练：

```powershell
Invoke-RestMethod "http://127.0.0.1:5000/api/risk/train?run_label=phase1"
```

训练后再次检查：

1. `GET /api/risk/summary` 显示心脏事件和脑卒中模型均已就绪。
2. `POST /api/risk/predict` 返回两个概率、综合五级风险和干预建议。
3. `docs/training_results.md` 与模型清单指标一致。

WSL2 或 VMware Linux 的 Spark 作业执行前，复制 `.env.example` 为 `.env` 并填写 HDFS 参数。默认输入路径为 `HDFS_INPUT_PATH`，也可以显式传入：

```bash
./scripts/start_spark_jobs.sh <hdfs-input-file>
```

作业完成后确认 staging/feature、群体统计和重点筛查输出目录均已生成。
