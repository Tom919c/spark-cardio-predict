# 机构端连续月份演示数据

使用 `scripts/generate_demo_population_snapshots.py` 生成不含疾病标签、带姓名和联系电话的机构端仿真数据。数据由同一批居民组成，多个文件分别对应不同月份，适合上传后观察模型高风险率趋势。

```bash
python scripts/generate_demo_population_snapshots.py \
  --rows 20000 \
  --months 6 \
  --start-period 2026-01 \
  --output-dir data/demo/organization_snapshots
```

生成文件示例：

- `chengdu_demo_population_2026-01.csv`
- `chengdu_demo_population_2026-02.csv`
- `chengdu_demo_population_2026-03.csv`

数据不包含 `label_heart`、`label_stroke` 或 `target_disease`。机构端分析会在正式模型可用时使用双模型概率生成高风险率，因此页面应显示“模型高风险率”，而不是“事件标签率”。

姓名、联系电话和居民编号均为 Faker 生成的虚拟数据，只能用于项目演示和测试，不能当作真实居民信息使用。上传页面会优先从文件名中的 `YYYY-MM` 识别数据周期。
