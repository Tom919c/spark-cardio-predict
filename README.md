FlaskProject1/
├─ app.py
│  Flask 启动入口。
│  负责启动 Web 服务，当前也包含启动后自动打开页面的逻辑。
│
├─ requirements.txt
│  项目 Python 依赖列表。
│
├─ README.md
│  项目总说明文档占位。
│
├─ run.bat
│  Windows 启动脚本。
│  可用于在本机双击启动项目。
│
├─ run.sh
│  Linux 启动脚本占位。
│
├─ config/
│  项目全局配置层。
│  ├─ __init__.py
│  │  配置包导出入口。
│  ├─ base.py
│  │  核心配置文件。
│  │  管理数据集路径、本地/分布式模式、HDFS 路径、训练参数、模型输出目录等。
│  ├─ dev.py
│  │  开发环境配置占位。
│  └─ prod.py
│     生产环境配置占位。
│
├─ docs/
│  项目文档层。
│  ├─ architecture.md
│  │  系统架构说明占位。
│  ├─ api_design.md
│  │  API 设计文档占位。
│  ├─ model_design.md
│  │  模型设计说明占位。
│  ├─ data_dictionary.md
│  │  数据字段与标签说明文档。
│  ├─ change_notes.md
│  │  项目修改记录文档。
│  ├─ training_results.md
│  │  模型训练结果记录文档。
│  └─ test_guide.md
│     测试说明文档。
│
├─ data/
│  项目数据分层目录。
│  ├─ raw/
│  │  原始数据层。
│  │  └─ datasets/
│  │     本地原始数据集目录。
│  ├─ staging/
│  │  清洗后的中间数据层。
│  │  对应 raw -> staging 过程。
│  ├─ feature/
│  │  特征数据层与模型输出层。
│  │  对应 staging -> feature 过程。
│  │  └─ models/
│  │     训练好的 heart/stroke 模型文件目录。
│  └─ sample/
│     样例数据目录。
│
├─ scripts/
│  运维和数据准备脚本层。
│  ├─ upload_to_hdfs.py
│  │  把本地原始数据上传到 HDFS raw 层。
│  ├─ start_spark_jobs.sh
│  │  串联 Spark ETL 和特征工程作业的启动脚本。
│  ├─ init_mysql.sql
│  │  MySQL 初始化脚本占位。
│  ├─ init_hive.hql
│  │  Hive 初始化脚本占位。
│  └─ redis_cache_demo.py
│     Redis 示例脚本占位。
│
├─ src/
│  项目主源码目录。
│  ├─ __init__.py
│  │  Flask 应用工厂入口。
│  │  负责创建 app、加载配置、注册蓝图、挂载 templates 和 static。
│  │
│  ├─ controllers/
│  │  接口控制层。
│  │  ├─ __init__.py
│  │  │  控制器注册入口。
│  │  ├─ health_controller.py
│  │  │  基础健康检查接口。
│  │  │  用于系统状态检测与服务可用性验证。
│  │  ├─ data_controller.py
│  │  │  数据集配置、预览、预处理接口。
│  │  ├─ risk_controller.py
│  │  │  双模型训练与单人预测接口。
│  │  │  是当前项目最核心的控制器之一。
│  │  ├─ page_controller.py
│  │  │  页面路由控制器。
│  │  │  负责渲染首页和个人风险评估页面。
│  │  ├─ analysis_controller.py
│  │  │  机构分析接口占位。
│  │  │  未来承接群体趋势分析、机构统计接口、高风险筛查接口。
│  │  └─ intervention_controller.py
│  │     干预建议接口占位。
│  │
│  ├─ services/
│  │  业务服务层。
│  │  ├─ __init__.py
│  │  │  服务包入口。
│  │  ├─ data_service.py
│  │  │  统一管理本地与分布式训练数据读取。
│  │  │  负责数据集读取、校验、预处理和训练数据构建入口。
│  │  ├─ risk_service.py
│  │  │  双模型训练、预测、结果整合主服务。
│  │  │  负责 heart/stroke 双模型的训练、预测、联合分类和结果封装。
│  │  ├─ intervention_service.py
│  │  │  个性化干预建议生成服务。
│  │  ├─ feature_service.py
│  │  │  特征服务占位。
│  │  │  当前为空，未来适合承接统一特征流水线服务。
│  │  └─ trend_service.py
│  │     群体趋势分析服务占位。
│  │
│  ├─ ml/
│  │  机器学习层。
│  │  ├─ __init__.py
│  │  │  机器学习包入口。
│  │  ├─ feature_engineering.py
│  │  │  单机清洗、双标签构造和训练表生成。
│  │  │  是当前本地数据清洗与特征工程核心模块。
│  │  ├─ train.py
│  │  │  双随机森林训练、多轮训练和最优模型选择。
│  │  ├─ evaluate.py
│  │  │  双模型二分类评估模块。
│  │  ├─ predict.py
│  │  │  双模型推理模块。
│  │  │  负责加载 heart/stroke 模型并输出预测结果。
│  │  └─ model_registry.py
│  │     模型注册管理占位。
│  │
│  ├─ spark_jobs/
│  │  分布式处理层。
│  │  ├─ __init__.py
│  │  │  Spark 作业包入口。
│  │  ├─ etl_job.py
│  │  │  Spark ETL 清洗作业。
│  │  │  负责 raw -> staging。
│  │  ├─ feature_build_job.py
│  │  │  Spark 特征工程作业。
│  │  │  负责 staging -> feature。
│  │  ├─ population_analysis_job.py
│  │  │  群体分析作业占位。
│  │  └─ high_risk_screening_job.py
│  │     高风险筛查作业占位。
│  │
│  ├─ models/
│  │  结果与领域实体层。
│  │  ├─ __init__.py
│  │  │  实体包入口。
│  │  ├─ risk_result.py
│  │  │  双模型联合后的最终风险结果实体。
│  │  ├─ intervention_plan.py
│  │  │  干预建议结果实体。
│  │  ├─ user_profile.py
│  │  │  用户画像实体占位。
│  │  └─ medical_record.py
│  │     医疗记录实体占位。
│  │
│  ├─ dao/
│  │  数据访问层，占位为主。
│  │  ├─ __init__.py
│  │  │  DAO 包入口。
│  │  ├─ mysql_dao.py
│  │  │  MySQL 访问占位。
│  │  ├─ redis_dao.py
│  │  │  Redis 访问占位。
│  │  ├─ hive_dao.py
│  │  │  Hive 访问占位。
│  │  └─ hbase_dao.py
│  │     HBase 访问占位。
│  │
│  ├─ utils/
│  │  工具层。
│  │  ├─ __init__.py
│  │  │  工具包入口。
│  │  ├─ response.py
│  │  │  统一接口返回格式。
│  │  ├─ validator.py
│  │  │  数据字段和预测参数校验工具。
│  │  ├─ risk_rules.py
│  │  │  中文风险说明与指标解读规则。
│  │  ├─ training_recorder.py
│  │  │  自动记录训练结果到 Markdown。
│  │  └─ logger.py
│  │     日志工具占位。
│  │
│  └─ visualization/
│     可视化服务层，占位为主。
│     ├─ __init__.py
│     │  可视化包入口。
│     ├─ chart_service.py
│     │  图表数据服务占位。
│     └─ dashboard_metrics.py
│        仪表盘指标服务占位。
│
├─ templates/
│  前端模板目录。
│  ├─ dashboard.html
│  │  首页/机构端入口页面。
│  │  当前已调整为偏用户入口展示页。
│  └─ risk_report.html
│     个人风险评估页面。
│     当前已接入双模型预测接口并可展示风险结果。
│
├─ static/
│  前端静态资源目录。
│  ├─ css/
│  │  样式资源目录，占位为主。
│  ├─ js/
│  │  脚本资源目录，占位为主。
│  ├─ img/
│  │  图片资源目录，占位为主。
│  └─ echarts/
│     ECharts 资源目录，占位为主。
│
└─ tests/
   测试目录，目前以占位为主。
   ├─ test_api.py
   │  API 自动化测试占位。
   ├─ test_risk_service.py
   │  风险服务测试占位。
   └─ test_feature_engineering.py
      特征工程测试占位。
