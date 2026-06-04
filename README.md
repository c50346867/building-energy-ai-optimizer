<div align="center">
  <img src="docs/banner.svg" width="100%" alt="Building Energy AI Optimizer Banner">
</div>

# 🏢 Building Energy AI Optimizer

> AI 驱动的建筑能耗预测与智能优化工具
> 
> AI-powered building energy prediction and intelligent optimization tool

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/c50346867/building-energy-ai-optimizer/pulls)
[![Stars](https://img.shields.io/github/stars/c50346867/building-energy-ai-optimizer?style=social)](https://github.com/c50346867/building-energy-ai-optimizer)

</div>

## 🌟 项目概览

基于机器学习（LightGBM / XGBoost）和历史建筑运行数据，对建筑能耗进行高精度预测，并提供 HVAC 系统调度优化建议。支持办公、住宅、工业等多种建筑类型。

### 🎯 核心价值

- **🔮 精准预测** — 基于多维特征的 AI 能耗预测模型，R² > 0.92
- **⚡ 智能优化** — HVAC 系统运行策略智能推荐，节能 15-30%
- **💰 降本增效** — 削峰填谷 + 需求响应，降低电费 10-25%
- **🌱 绿色低碳** — 精确计算碳减排量，助力碳中和目标
- **📊 开箱即用** — 内置模拟数据生成器，无需真实数据即可体验

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/c50346867/building-energy-ai-optimizer
cd building-energy-ai-optimizer

# 安装依赖
pip install -r requirements.txt

# 运行模拟演示
python examples/demo.py

# 启动 Web 界面
streamlit run frontend/app.py
```

## 项目结构

```
building-energy-ai-optimizer/
├── backend/
│   ├── api/              # REST API (FastAPI)
│   ├── core/             # 核心算法
│   │   ├── predictor.py   # 能耗预测引擎
│   │   ├── optimizer.py   # 优化引擎
│   │   └── simulator.py   # 数据模拟器
│   └── models/            # 训练好的模型
├── frontend/
│   └── app.py             # Streamlit 演示界面
├── data/                  # 示例数据
├── examples/
│   └── demo.py            # 快速演示脚本
├── docs/                  # 文档
├── tests/                 # 测试
└── README.md              # 本文
```

## 核心功能

### 🔮 能耗预测
- 基于天气、时间、建筑特征等多维特征
- 支持 3 种算法：Prophet, LightGBM, XGBoost
- 小时/日/周多粒度预测

### ⚡ 智能优化
- HVAC 系统启停策略推荐（削峰填谷）
- 室内温度设定点优化
- 需求侧响应潜力评估

### 📊 可视化
- 能耗趋势热力图
- 分项能耗分解图
- 优化前后对比

## 技术栈

- **语言**: Python 3.9+
- **ML 框架**: scikit-learn, LightGBM, XGBoost
- **时序预测**: Prophet, ARIMA
- **Web 框架**: FastAPI (API), Streamlit (Demo)
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, plotly, seaborn

## 数据来源

- 内置模拟数据生成器（用于演示和快速上手）
- 支持接入真实建筑能源管理系统（BEMS）数据
- 兼容 EnergyPlus / OpenStudio 仿真输出格式

## 📄 许可证

MIT License
