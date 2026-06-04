"""
Building Energy AI Optimizer — Streamlit 可视化演示界面

运行: streamlit run frontend/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.simulator import EnergyDataSimulator
from backend.core.predictor import BuildingEnergyPredictor
from backend.core.optimizer import HVACOptimizer, HVACSetpoint

# === 页面配置 ===
st.set_page_config(
    page_title="建筑能耗AI优化器",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === 侧边栏 ===
st.sidebar.image(
    "https://img.icons8.com/fluency/96/building.png",
    width=48,
)
st.sidebar.title("🏢 建筑能耗 AI 优化器")
st.sidebar.markdown("---")

building_type = st.sidebar.selectbox(
    "建筑类型",
    options=[
        "commercial_office",
        "residential",
        "industrial",
        "mall",
        "hospital",
    ],
    format_func=lambda x: {
        "commercial_office": "🏢 商业办公楼",
        "residential": "🏠 住宅",
        "industrial": "🏭 工业厂房",
        "mall": "🛍️ 购物中心",
        "hospital": "🏥 医院",
    }.get(x, x),
)

days = st.sidebar.slider("数据范围（天）", 7, 365, 90)
simulate_btn = st.sidebar.button("🔄 生成并分析", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ 优化参数")
cooling_sp = st.sidebar.slider("制冷设定 (°C)", 22, 30, 26)
heating_sp = st.sidebar.slider("制热设定 (°C)", 16, 22, 20)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 *基于 LightGBM/XGBoost 的建筑能耗预测与优化*"
)

# === 主界面 ===
st.title("🏢 建筑能耗智能预测与优化")
st.markdown(
    "基于机器学习的建筑能耗预测与 HVAC 系统优化调度平台。"
    "支持办公、住宅、工业等多种建筑类型。"
)

# Tab 布局
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 能耗分析",
    "🔮 预测引擎",
    "⚡ 优化建议",
    "📋 数据管理",
])

# 初始化模拟器
simulator = EnergyDataSimulator(building_type=building_type)

with tab1:
    st.header("能耗数据分析")

    if simulate_btn or 'df' not in st.session_state:
        with st.spinner(f"正在生成{building_type}建筑能耗数据..."):
            df = simulator.generate(days=days)
            st.session_state.df = df
            st.session_state.building_type = building_type

    df = st.session_state.get('df')

    if df is not None:
        # 概览指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("平均能耗 (kWh/日)", f"{df['energy_kwh'].mean():.0f}")
        with col2:
            st.metric("峰值能耗 (kW)", f"{df['energy_kwh'].max():.0f}")
        with col3:
            st.metric("最低能耗 (kW)", f"{df['energy_kwh'].min():.0f}")
        with col4:
            days_span = (pd.to_datetime(df['timestamp']).max() - pd.to_datetime(df['timestamp']).min()).days
            st.metric("分析周期", f"{days_span} 天")

        # 能耗时间序列图
        st.subheader("能耗时序图")
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("逐时能耗", "室外温度"),
        )

        fig.add_trace(
            go.Scatter(
                x=df['timestamp'][-336:],  # 最近两周
                y=df['energy_kwh'][-336:],
                mode='lines',
                name='总能耗',
                line=dict(color='#FF6B6B', width=1),
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df['timestamp'][-336:],
                y=df['temperature'][-336:],
                mode='lines',
                name='温度',
                line=dict(color='#4ECDC4', width=1),
            ),
            row=2, col=1,
        )

        fig.update_layout(height=500, hovermode='x unified')
        fig.update_xaxes(title_text="时间", row=2, col=1)
        fig.update_yaxes(title_text="能耗 (kWh)", row=1, col=1)
        fig.update_yaxes(title_text="温度 (°C)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # 分项能耗
        st.subheader("分项能耗构成")
        if all(c in df.columns for c in ['hvac_kwh', 'lighting_kwh', 'equipment_kwh']):
            breakdown = {
                'HVAC 系统': df['hvac_kwh'].sum(),
                '照明系统': df['lighting_kwh'].sum(),
                '设备负荷': df['equipment_kwh'].sum(),
                '其他': df['other_kwh'].sum(),
            }

            col1, col2 = st.columns([1, 1])
            with col1:
                fig_pie = go.Figure(data=[
                    go.Pie(
                        labels=list(breakdown.keys()),
                        values=list(breakdown.values()),
                        hole=0.4,
                        marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                    )
                ])
                fig_pie.update_layout(title="能耗构成占比", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # 典型日负荷曲线
                st.subheader("典型日负荷曲线")
                df_week = df.tail(168)  # 最近一周
                df_week['hour'] = pd.to_datetime(df_week['timestamp']).dt.hour
                hourly_avg = df_week.groupby('hour')['energy_kwh'].agg(['mean', 'std']).reset_index()

                fig_daily = go.Figure()
                fig_daily.add_trace(go.Scatter(
                    x=hourly_avg['hour'],
                    y=hourly_avg['mean'],
                    mode='lines+markers',
                    name='平均负荷',
                    line=dict(color='#FF6B6B', width=2),
                ))
                fig_daily.add_trace(go.Scatter(
                    x=hourly_avg['hour'],
                    y=hourly_avg['mean'] + hourly_avg['std'],
                    mode='lines',
                    name='+1σ',
                    line=dict(color='rgba(255,107,107,0.3)', width=0),
                    showlegend=True,
                ))
                fig_daily.add_trace(go.Scatter(
                    x=hourly_avg['hour'],
                    y=hourly_avg['mean'] - hourly_avg['std'],
                    mode='lines',
                    name='-1σ',
                    line=dict(color='rgba(255,107,107,0.3)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,107,107,0.1)',
                ))
                fig_daily.update_layout(
                    title="逐小时平均负荷 (±1σ)",
                    xaxis_title="小时",
                    yaxis_title="能耗 (kWh)",
                    height=400,
                    hovermode='x unified',
                )
                st.plotly_chart(fig_daily, use_container_width=True)

with tab2:
    st.header("能耗预测引擎")

    if df is not None:
        st.markdown("""
        基于 **LightGBM** 的能耗预测模型。自动提取时间特征、天气特征和滞后特征进行训练。
        """)

        # 训练模型
        with st.spinner("正在训练预测模型..."):
            predictor = BuildingEnergyPredictor(method='lightgbm')
            train_info = predictor.train(df)

        st.success(f"✅ 模型训练完成！特征数: {train_info['n_features']}，样本量: {train_info['n_samples']}")

        # 预测未来
        last_ts = pd.to_datetime(df['timestamp']).max()
        future = pd.date_range(start=last_ts + timedelta(hours=1), periods=72, freq='1h')
        future_df = pd.DataFrame({'timestamp': future})
        future_df['temperature'] = df['temperature'].tail(72).values if len(df) >= 72 else 25
        future_df['humidity'] = df['humidity'].tail(72).values if len(df) >= 72 else 60
        future_df['energy_kwh'] = 0

        predictions = predictor.predict(future_df)

        # 显示预测结果
        st.subheader("未来 72 小时能耗预测")
        fig_pred = go.Figure()

        # 历史数据
        fig_pred.add_trace(go.Scatter(
            x=df['timestamp'].tail(168),
            y=df['energy_kwh'].tail(168),
            mode='lines',
            name='历史能耗',
            line=dict(color='#4ECDC4', width=1.5),
        ))

        # 预测数据
        fig_pred.add_trace(go.Scatter(
            x=future,
            y=predictions,
            mode='lines',
            name='预测值',
            line=dict(color='#FF6B6B', width=2, dash='dash'),
        ))

        fig_pred.add_vline(
            x=last_ts,
            line_dash="dot",
            line_color="gray",
            annotation_text="当前",
        )

        fig_pred.update_layout(
            height=400,
            hovermode='x unified',
            xaxis_title="时间",
            yaxis_title="能耗 (kWh)",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # 特征重要性
        st.subheader("特征重要性分析")
        if hasattr(predictor.model, 'feature_importances_') and predictor.feature_names:
            importance = pd.DataFrame({
                'feature': predictor.feature_names,
                'importance': predictor.model.feature_importances_,
            }).sort_values('importance', ascending=True).tail(15)

            fig_imp = go.Figure(go.Bar(
                x=importance['importance'],
                y=importance['feature'],
                orientation='h',
                marker_color='#45B7D1',
            ))
            fig_imp.update_layout(
                height=500,
                xaxis_title="重要性评分",
                yaxis_title="特征",
                margin=dict(l=5, r=5, t=5, b=5),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            with st.expander("📊 特征说明"):
                st.markdown("""
                | 特征类别 | 说明 |
                |---------|------|
                | hour_sin/cos | 小时周期编码 |
                | dow_sin/cos | 星期周期编码 |
                | temperature | 室外温度 |
                | cooling_degree | 制冷度日数 (>26°C) |
                | heating_degree | 制热度日数 (<18°C) |
                | energy_lag_* | 历史能耗滞后特征 |
                | energy_roll_* | 历史能耗滚动统计 |
                """)

with tab3:
    st.header("⚡ 智能优化建议")

    if df is not None:
        optimizer = HVACOptimizer()
        hvac_params = HVACSetpoint(
            cooling_setpoint=cooling_sp,
            heating_setpoint=heating_sp,
        )

        with st.spinner("正在计算优化方案..."):
            # 准备数据
            df_opt = df.copy()
            df_opt['hour'] = pd.to_datetime(df_opt['timestamp']).dt.hour
            df_opt['is_weekend'] = pd.to_datetime(df_opt['timestamp']).dt.dayofweek >= 5
            df_opt['is_workday'] = (~df_opt['is_weekend']).astype(int)
            result = optimizer.optimize_setpoints(df_opt, hvac_params)

        # 核心指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("节能潜力", f"{result.savings_percent:.1f}%",
                      delta=f"{result.savings_kwh:.0f} kWh")
        with col2:
            st.metric("电费节省", f"¥{result.cost_savings_cny:.0f}")
        with col3:
            st.metric("CO₂ 减排", f"{result.co2_reduction_kg:.1f} kg")
        with col4:
            st.metric("优化后总能耗", f"{result.total_optimized:.0f} kWh")

        # 优化前后对比图
        st.subheader("优化前后能耗对比")
        hourly = result.hourly_schedule

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=hourly.index,
            y=hourly['original_kwh'],
            mode='lines',
            name='优化前',
            line=dict(color='#FF6B6B', width=1),
        ))
        fig_compare.add_trace(go.Scatter(
            x=hourly.index,
            y=hourly['optimized_kwh'],
            mode='lines',
            name='优化后',
            line=dict(color='#4ECDC4', width=1.5),
        ))
        fig_compare.add_trace(go.Scatter(
            x=hourly.index,
            y=hourly['savings_kwh'],
            mode='lines',
            name='节省量',
            line=dict(color='#96CEB4', width=1, dash='dot'),
        ))

        fig_compare.update_layout(
            height=400,
            hovermode='x unified',
            xaxis_title="时间点",
            yaxis_title="能耗 (kWh)",
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # 优化建议列表
        st.subheader("📋 优化建议")
        for i, rec in enumerate(result.recommendations, 1):
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            with st.container():
                st.markdown(f"""
                **{priority_icon.get(rec['priority'], '⚪')} {i}. {rec['title']}**
                > {rec['description']}
                > 
                > **节能潜力**: {rec['savings_potential']}
                """)
                st.markdown("---")

        # 需求响应潜力
        st.subheader("需求响应潜力评估")
        from backend.core.optimizer import calculate_demand_response_potential
        dr_potential = calculate_demand_response_potential(df, {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("15分钟可削减", f"{dr_potential['shed_15min_kw']:.0f} kW")
        with col2:
            st.metric("30分钟可削减", f"{dr_potential['shed_30min_kw']:.0f} kW")
        with col3:
            st.metric("1小时可削减", f"{dr_potential['shed_1hour_kw']:.0f} kW")

        st.markdown(f"**适合的需求响应项目**: {', '.join(dr_potential['suitable_programs'])}")

with tab4:
    st.header("📋 数据管理")

    if df is not None:
        st.subheader("数据预览")
        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=400,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 下载 CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"building_energy_{building_type}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            # 数据统计
            st.info(f"""
            **数据统计**
            - 总记录数: {len(df):,}
            - 日期范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}
            - 平均能耗: {df['energy_kwh'].mean():.1f} kWh
            - 能耗标准差: {df['energy_kwh'].std():.1f} kWh
            - 中位数能耗: {df['energy_kwh'].median():.1f} kWh
            """)

# === 页脚 ===
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🏢 Building Energy AI Optimizer | "
    "基于 LightGBM/XGBoost | "
    "<a href='https://github.com/c50346867/building-energy-ai-optimizer'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
