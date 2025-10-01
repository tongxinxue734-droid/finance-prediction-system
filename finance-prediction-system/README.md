# 金融预测系统 (Finance Prediction System)

一个基于微服务架构的金融市场预测系统，能够采集金融数据、处理分析并进行预测，提供Web管理界面和小程序前端。

## 项目结构
finance-prediction-system/
├── README.md               # 项目说明文档
├── docker-compose.yml      # Docker容器编排配置
├── .env.example            # 环境变量示例
├── .gitignore              # Git忽略文件配置
├── backend/                # 后端服务代码
│   ├── api/                # FastAPI接口服务
│   ├── data_collector/     # 数据采集服务
│   ├── data_processor/     # 数据处理服务
│   └── prediction_service/ # 预测模型服务
├── frontend/               # 前端应用代码
│   ├── admin/              # 网页管理后台（React）
│   └── miniprogram/        # 微信小程序
└── init-scripts/           # 数据库初始化脚本
## 功能说明

1. **数据采集服务**：定时从金融API获取市场数据
2. **数据处理服务**：清洗数据、计算技术指标
3. **预测服务**：训练模型并生成预测结果
4. **API服务**：提供统一的接口供前端调用
5. **管理后台**：管理数据、模型和预测结果
6. **微信小程序**：提供移动端访问入口

## 快速开始

1. 复制环境变量示例并修改：
   ```
   cp .env.example .env
   ```

2. 启动服务：
   ```
   docker-compose up -d
   ```

3. 访问管理后台：http://localhost:3000
4. API文档：http://localhost:8000/docs
