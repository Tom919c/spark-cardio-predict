# CardioSpark 前端简易交付包

纯静态前端交付，零本地依赖，`templates/` 内 HTML 均为自包含单文件（CDN 引入 Tailwind / ECharts / echarts-wordcloud，当前 IS_MOCK=true 自带数据）。

## 目录结构
```
前端简易交付包/
├─ templates/
│  ├─ B端.html      # B端大屏（区域风险决策可视化）
│  └─ C端.html      # C端个人评测页（表单 → 5级风险结果）
├─ static/          # 空目录占位（满足目录规范，当前无本地资源）
├─ 接口对接说明.txt  # 前后端接口契约 + 联调清单（给 D 后端）
└─ README.md        # 本说明
```

## 运行
- 最简单：直接双击打开 `templates/B端.html` / `templates/C端.html`（需联网加载 CDN）。
- 推荐：`python -m http.server 8000` 后访问 `http://127.0.0.1:8000/templates/B端.html`。

## 对接后端
详见 `接口对接说明.txt`：切真实数据只需把两页 `IS_MOCK` 置 false、改 `API_BASE(_URL)`，其余不变。
