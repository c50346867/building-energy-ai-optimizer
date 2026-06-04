"""
building-energy-ai-optimizer 优化引擎
提供 HVAC 调度优化、需求响应和节能策略建议
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OptimizationResult:
    """优化结果数据结构"""
    original_consumption: np.ndarray
    optimized_consumption: np.ndarray
    savings_kwh: float
    savings_percent: float
    total_original: float
    total_optimized: float
    co2_reduction_kg: float
    cost_savings_cny: float
    recommendations: List[Dict]
    hourly_schedule: pd.DataFrame


@dataclass
class HVACSetpoint:
    """HVAC 设定点"""
    cooling_setpoint: float = 26.0     # 夏季制冷设定（°C）
    heating_setpoint: float = 20.0     # 冬季制热设定（°C）
    night_setback_cooling: float = 28.0  # 夜间制冷回差
    night_setback_heating: float = 16.0  # 夜间制热回差
    pre_cooling_hours: int = 1         # 预冷时间（小时）
    pre_heating_hours: int = 1         # 预热时间


class HVACOptimizer:
    """HVAC 系统优化器

    提供:
    - 温度设定点优化
    - 削峰填谷调度
    - 需求响应策略
    - 预冷/预热调度
    """

    def __init__(
        self,
        electricity_price_day: float = 0.85,     # 白天电价（元/kWh）
        electricity_price_night: float = 0.35,   # 夜间电价
        peak_hours: List[int] = None,
        valley_hours: List[int] = None,
        co2_factor: float = 0.581,               # 电力碳排放因子 (kg CO2/kWh)
    ):
        self.electricity_price_day = electricity_price_day
        self.electricity_price_night = electricity_price_night
        self.peak_hours = peak_hours or [9, 10, 11, 14, 15, 16, 17]
        self.valley_hours = valley_hours or [23, 0, 1, 2, 3, 4, 5, 6]
        self.co2_factor = co2_factor

    def _get_price(self, hour: int) -> float:
        """根据时段获取电价"""
        if hour in self.peak_hours:
            return self.electricity_price_day * 1.5  # 峰时上浮50%
        elif hour in self.valley_hours:
            return self.electricity_price_night * 0.8  # 谷时再打8折
        else:
            return self.electricity_price_day

    def optimize_setpoints(
        self,
        df: pd.DataFrame,
        hvac_params: Optional[HVACSetpoint] = None,
    ) -> OptimizationResult:
        """优化 HVAC 温度设定点

        核心策略:
        1. 根据室外温度动态调整室内设定点
        2. 工作时间维持舒适温度，非工作时间回差
        3. 电价高峰时段适当放宽设定
        """
        params = hvac_params or HVACSetpoint()
        result_df = df.copy()

        # 确保有小时数据
        if 'hour' not in result_df.columns and 'timestamp' in result_df.columns:
            result_df['hour'] = pd.to_datetime(result_df['timestamp']).dt.hour

        if 'temperature' not in result_df.columns:
            result_df['temperature'] = 25  # 默认室外温度

        hours = result_df['hour'].values
        outdoor_temp = result_df['temperature'].values
        is_workday = result_df.get('is_workday', pd.Series([1] * len(result_df))).values
        energy = result_df.get('energy_kwh', np.zeros(len(result_df)))

        # 模拟计算优化后的能耗
        optimized_energy = []

        for i, (h, temp, work) in enumerate(zip(hours, outdoor_temp, is_workday)):
            base = energy[i] if energy[i] > 0 else self._simulate_baseline(temp, h, work)

            # 优化因子计算
            if work and 8 <= h <= 18:
                # 工作时间：正常设定
                factor = 1.0
            elif work and (h < 8 or h > 18):
                # 非工作时间：回差节能
                factor = 0.55
            elif h in self.peak_hours:
                # 周末高峰：适当削减
                factor = 0.75
            elif h in self.valley_hours:
                # 谷时：正常或预调
                factor = 0.65
            else:
                factor = 0.60

            # 室外温度越高，节能空间越大（高温日制冷负荷大）
            if temp > 35:
                factor *= 0.85  # 高温日节能比例增大
            elif temp > 30:
                factor *= 0.90

            optimized_energy.append(base * factor)

        optimized_energy = np.array(optimized_energy)
        original = energy.copy()
        original[original == 0] = optimized_energy[original == 0]

        savings = original - optimized_energy
        total_original = original.sum()
        total_optimized = optimized_energy.sum()

        # 经济效益计算
        cost_original = sum(
            original[i] * self._get_price(hours[i]) for i in range(len(original))
        )
        cost_optimized = sum(
            optimized_energy[i] * self._get_price(hours[i]) for i in range(len(optimized_energy))
        )

        # 生成推荐建议
        recommendations = self._generate_recommendations(
            savings.sum(),
            total_original,
            cost_original - cost_optimized,
            params,
        )

        # 生成调度计划
        schedule_df = pd.DataFrame({
            'hour': hours,
            'original_kwh': original,
            'optimized_kwh': optimized_energy,
            'savings_kwh': savings,
            'savings_percent': np.where(original > 0, savings / original * 100, 0),
        })

        return OptimizationResult(
            original_consumption=original,
            optimized_consumption=optimized_energy,
            savings_kwh=savings.sum(),
            savings_percent=(total_original - total_optimized) / total_original * 100 if total_original > 0 else 0,
            total_original=total_original,
            total_optimized=total_optimized,
            co2_reduction_kg=savings.sum() * self.co2_factor,
            cost_savings_cny=cost_original - cost_optimized,
            recommendations=recommendations,
            hourly_schedule=schedule_df,
        )

    def _simulate_baseline(self, temp: float, hour: int, is_workday: int) -> float:
        """模拟基准能耗（kW）"""
        if is_workday:
            base = 500 if 8 <= hour <= 18 else 200
        else:
            base = 150 if 10 <= hour <= 20 else 80

        # 温度影响
        temp_factor = 1.0
        if temp > 30:
            temp_factor = 1 + (temp - 30) * 0.05
        elif temp < 10:
            temp_factor = 1 + (10 - temp) * 0.04

        return base * temp_factor

    def _generate_recommendations(
        self,
        savings: float,
        total: float,
        cost_savings: float,
        params: HVACSetpoint,
    ) -> List[Dict]:
        """生成优化建议"""
        recommendations = []

        # 1. 温度设定建议
        recommendations.append({
            'category': 'setpoint',
            'priority': 'high',
            'title': '调整温度设定点',
            'description': f'将制冷设定点从默认 24°C 调整至 {params.cooling_setpoint}°C，'
                          f'制热从 22°C 调整至 {params.heating_setpoint}°C。'
                          f'每调高 1°C 制冷设定约可节能 6-8%。',
            'savings_potential': '15-25%',
        })

        # 2. 夜间回差
        recommendations.append({
            'category': 'night_setback',
            'priority': 'high',
            'title': '启用夜间回差策略',
            'description': f'��工作时间将温度放宽至制冷 {params.night_setback_cooling}°C / '
                          f'制热 {params.night_setback_heating}°C，'
                          f'可减少 HVAC 系统 30-40% 运行时间。',
            'savings_potential': '10-18%',
        })

        # 3. 预冷/预热
        recommendations.append({
            'category': 'preconditioning',
            'priority': 'medium',
            'title': '预冷/预热调度',
            'description': f'在电价低谷时段（{self.valley_hours[0]}:00-{self.valley_hours[-1]}:00）'
                          f'提前预处理建筑，减少高峰时段制冷/制热负荷。',
            'savings_potential': '8-12%（电费）',
        })

        # 4. 削峰响应
        recommendations.append({
            'category': 'peak_shaving',
            'priority': 'medium',
            'title': '削峰填谷',
            'description': f'在电价高峰时段（{min(self.peak_hours)}:00-{max(self.peak_hours)}:00）'
                          f'适当降低 HVAC 负荷，利用建筑热惰性维持舒适度。',
            'savings_potential': '5-10%（电费）',
        })

        # 5. 新风控制
        recommendations.append({
            'category': 'ventilation',
            'priority': 'low',
            'title': '智能新风控制',
            'description': '根据 CO2 浓度和室外空气质量动态调节新风量，'
                          '在保障室内空气品质的前提下减少新风处理能耗。',
            'savings_potential': '5-8%',
        })

        return recommendations


def calculate_demand_response_potential(
    df: pd.DataFrame,
    building_params: Dict,
) -> Dict:
    """计算需求响应潜力"""
    total_load = df['energy_kwh'].sum() if 'energy_kwh' in df.columns else 100000
    peak_load = df['energy_kwh'].max() if 'energy_kwh' in df.columns else 5000

    # 需求响应潜力估算
    shed_potential_15min = peak_load * 0.15  # 15分钟可削减15%
    shed_potential_30min = peak_load * 0.25
    shed_potential_1hour = peak_load * 0.30

    return {
        'peak_load_kw': peak_load,
        'avg_load_kw': total_load / len(df) if len(df) > 0 else 0,
        'shed_15min_kw': shed_potential_15min,
        'shed_30min_kw': shed_potential_30min,
        'shed_1hour_kw': shed_potential_1hour,
        'annual_dr_revenue_estimate': shed_potential_1hour * 50 * 100,  # 粗略估算（元/年）
        'suitable_programs': ['紧急需求响应', '经济需求响应', '辅助服务市场'],
    }
