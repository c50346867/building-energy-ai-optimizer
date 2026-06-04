"""
building-energy-ai-optimizer 数据模拟器
生成示例建筑能耗数据供演示和测试使用
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class EnergyDataSimulator:
    """建筑能耗数据模拟器

    可生成逼真的建筑能耗时序数据，包含：
    - 昼夜/季节周期模式
    - 工作日/周末差异
    - 天气影响（温度、湿度）
    - 随机波动和异常事件
    - 多种建筑类型特征
    """

    # 建筑类型默认参数
    BUILDING_PROFILES = {
        'commercial_office': {
            'name': '商业办公楼',
            'base_load_kw': 150,
            'peak_load_kw': 800,
            'hvac_ratio': 0.45,
            'lighting_ratio': 0.15,
            'equipment_ratio': 0.30,
            'other_ratio': 0.10,
            'work_hours': (8, 19),
            'weekend_factor': 0.35,
            'temp_sensitivity': 0.08,  # 每°C变化带来的负荷变化率
        },
        'residential': {
            'name': '住宅',
            'base_load_kw': 20,
            'peak_load_kw': 80,
            'hvac_ratio': 0.35,
            'lighting_ratio': 0.10,
            'equipment_ratio': 0.40,
            'other_ratio': 0.15,
            'work_hours': (18, 23),
            'weekend_factor': 1.10,
            'temp_sensitivity': 0.06,
        },
        'industrial': {
            'name': '工业厂房',
            'base_load_kw': 400,
            'peak_load_kw': 2000,
            'hvac_ratio': 0.15,
            'lighting_ratio': 0.05,
            'equipment_ratio': 0.70,
            'other_ratio': 0.10,
            'work_hours': (7, 20),
            'weekend_factor': 0.40,
            'temp_sensitivity': 0.04,
        },
        'mall': {
            'name': '购物中心',
            'base_load_kw': 300,
            'peak_load_kw': 1200,
            'hvac_ratio': 0.40,
            'lighting_ratio': 0.20,
            'equipment_ratio': 0.30,
            'other_ratio': 0.10,
            'work_hours': (10, 22),
            'weekend_factor': 1.20,
            'temp_sensitivity': 0.07,
        },
        'hospital': {
            'name': '医院',
            'base_load_kw': 500,
            'peak_load_kw': 1500,
            'hvac_ratio': 0.35,
            'lighting_ratio': 0.15,
            'equipment_ratio': 0.40,
            'other_ratio': 0.10,
            'work_hours': (0, 24),
            'weekend_factor': 0.95,
            'temp_sensitivity': 0.06,
        },
    }

    def __init__(self, building_type: str = 'commercial_office', random_seed: int = 42):
        self.building_type = building_type
        self.profile = self.BUILDING_PROFILES.get(building_type, self.BUILDING_PROFILES['commercial_office'])
        self.rng = np.random.default_rng(random_seed)

    def _generate_temperature(
        self,
        dates: pd.DatetimeIndex,
        location: str = 'Shanghai',
    ) -> np.ndarray:
        """生成模拟温度数据"""
        n = len(dates)
        days = np.arange(n) / 24
        hours = dates.hour

        # 上海典型气温模式（基于2025年气候数据）
        # 年周期（正弦波）
        annual_temp = 18 + 14 * np.sin(2 * np.pi * days / 365 - np.pi / 2)

        # 日周期
        daily_temp = 5 * np.sin(2 * np.pi * (hours - 8) / 24)

        # 天气波动
        weather_noise = self.rng.normal(0, 2, n)
        # 引入一些天气事件（热浪、寒潮）
        heatwave = self.rng.choice([0, 3], n, p=[0.97, 0.03])
        coldwave = self.rng.choice([0, -3], n, p=[0.97, 0.03])

        temperature = annual_temp + daily_temp + weather_noise + heatwave + coldwave
        return np.clip(temperature, -5, 42)

    def _generate_humidity(self, temperature: np.ndarray) -> np.ndarray:
        """生成模拟湿度数据"""
        n = len(temperature)
        base_humidity = 70 - 0.3 * (temperature - 20)  # 温度越高湿度相对降低
        noise = self.rng.normal(0, 8, n)
        humidity = base_humidity + noise
        return np.clip(humidity, 20, 100)

    def generate(
        self,
        days: int = 365,
        start_date: str = '2025-01-01',
        freq: str = '1h',
        add_noise: bool = True,
        add_anomalies: bool = True,
        location: str = 'Shanghai',
    ) -> pd.DataFrame:
        """生成建筑能耗时序数据

        Args:
            days: 天数
            start_date: 起始日期
            freq: 数据频率 ('1h', '30min', '15min')
            add_noise: 是否添加随机噪声
            add_anomalies: 是否添加异常事件

        Returns:
            DataFrame 包含 timestamp, energy_kwh, temperature, humidity 等列
        """
        hours_per_step = {'1h': 1, '30min': 0.5, '15min': 0.25}
        step_hours = hours_per_step.get(freq, 1)
        steps_per_day = int(24 / step_hours)
        n_steps = days * steps_per_day

        # 生成时间轴
        timestamps = pd.date_range(
            start=start_date,
            periods=n_steps,
            freq=freq,
        )

        # 基础特征
        hours = timestamps.hour
        day_of_week = timestamps.dayofweek
        is_weekend = (day_of_week >= 5).astype(int)
        days_elapsed = np.arange(n_steps) / steps_per_day

        # 温度
        temperature = self._generate_temperature(timestamps, location)

        # 湿度
        humidity = self._generate_humidity(temperature)

        # === 生成能耗 ===
        profile = self.profile
        work_start, work_end = profile['work_hours']
        base = profile['base_load_kw']
        peak = profile['peak_load_kw']
        temp_sensitivity = profile['temp_sensitivity']

        # 1. 基础负荷
        energy = np.full(n_steps, base)

        # 2. 工作时间负荷
        if work_start < work_end:
            is_work_time = ((hours >= work_start) & (hours < work_end)).astype(float)
        else:
            # 跨天（如住宅：18-23点）
            is_work_time = ((hours >= work_start) | (hours < work_end)).astype(float)

        # 工作时间负荷：日间渐变模式
        work_load = is_work_time * (
            0.5 + 0.5 * np.sin(np.pi * (hours - work_start) / (work_end - work_start))
        )

        # 3. 周末调节
        weekend_factor = profile['weekend_factor']
        work_load = work_load * (1 + (weekend_factor - 1) * is_weekend)

        # 4. 季节调节：基于温度
        temp_load = temp_sensitivity * np.abs(temperature - 22) * 0.5

        # 5. 组合
        energy += work_load * (peak - base) * 0.5
        energy += temp_load * (peak - base) * 0.3

        # 6. 噪声
        if add_noise:
            noise_level = 0.05 * (peak - base)
            energy += self.rng.normal(0, noise_level, n_steps)

        # 7. 异常事件
        if add_anomalies:
            # 设备故障：负荷突增
            fault_events = self.rng.choice([0, 1], n_steps, p=[0.998, 0.002])
            energy += fault_events * self.rng.uniform(50, 200, n_steps)

        # 确保不出现负值
        energy = np.maximum(energy, base * 0.1)

        # === 分项能耗 ===
        hvac = energy * profile['hvac_ratio']
        lighting = energy * profile['lighting_ratio']
        equipment = energy * profile['equipment_ratio']
        other = energy * profile['other_ratio']

        df = pd.DataFrame({
            'timestamp': timestamps,
            'energy_kwh': energy,
            'temperature': temperature,
            'humidity': humidity,
            'hvac_kwh': hvac,
            'lighting_kwh': lighting,
            'equipment_kwh': equipment,
            'other_kwh': other,
            'is_weekend': is_weekend,
            'is_workday': 1 - is_weekend,
            'building_type': self.building_type,
        })

        return df

    @staticmethod
    def generate_multi_building(
        building_types: List[str],
        days: int = 90,
        start_date: str = '2025-06-01',
    ) -> Dict[str, pd.DataFrame]:
        """生成多个建筑的能耗数据"""
        results = {}
        for bt in building_types:
            sim = EnergyDataSimulator(building_type=bt)
            results[bt] = sim.generate(days=days, start_date=start_date)
        return results


def create_training_dataset(
    building_type: str = 'commercial_office',
    days: int = 365,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """创建训练/测试数据集"""
    sim = EnergyDataSimulator(building_type=building_type)
    df = sim.generate(days=days)

    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)

    print(f"[DataLoader] {sim.profile['name']} 数据: {len(df)} 条")
    print(f"  Train: {len(train)} 条, Test: {len(test)} 条")
    print(f"  平均能耗: {df['energy_kwh'].mean():.1f} kWh, "
          f"峰值: {df['energy_kwh'].max():.1f} kWh")

    return train, test
