"""
building-energy-ai-optimizer 核心预测引擎
基于机器学习的建筑能耗预测模块
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


class BuildingEnergyPredictor:
    """建筑能耗预测器

    支持多种 ML 算法进行能耗预测:
    - LightGBM (默认，速度快精度高)
    - XGBoost
    - Random Forest

    Usage:
        predictor = BuildingEnergyPredictor(method='lightgbm')
        predictor.train(X_train, y_train)
        predictions = predictor.predict(X_test)
    """

    def __init__(
        self,
        method: str = 'lightgbm',
        lookback_hours: int = 168,  # 7天回溯
        forecast_horizon: int = 72,  # 预测未来72小时
        random_state: int = 42,
    ):
        self.method = method
        self.lookback_hours = lookback_hours
        self.forecast_horizon = forecast_horizon
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self._is_trained = False

    def _create_model(self):
        """根据 method 创建模型实例"""
        if self.method == 'lightgbm' and LGBM_AVAILABLE:
            return LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=12,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
        elif self.method == 'xgboost' and XGB_AVAILABLE:
            return XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0,
            )
        else:
            # Fallback: Random Forest
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1,
            )

    @staticmethod
    def extract_time_features(df: pd.DataFrame, datetime_col: str = 'timestamp') -> pd.DataFrame:
        """从时间戳中提取特征"""
        result = df.copy()
        dt = pd.to_datetime(result[datetime_col])

        result['hour'] = dt.dt.hour
        result['day_of_week'] = dt.dt.dayofweek
        result['day_of_year'] = dt.dt.dayofyear
        result['month'] = dt.dt.month
        result['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        result['is_workday'] = result['is_weekend'] * (-1) + 1

        # 周期编码（处理 hour 的周期性）
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

        # 季节标记
        result['season'] = result['month'].apply(
            lambda m: 0 if m in [12, 1, 2] else (1 if m in [3, 4, 5] else (2 if m in [6, 7, 8] else 3))
        )

        return result

    @staticmethod
    def extract_weather_features(
        df: pd.DataFrame,
        temp_col: str = 'temperature',
        humidity_col: str = 'humidity'
    ) -> pd.DataFrame:
        """从天气数据中提取特征"""
        result = df.copy()

        # 温度派生特征
        if temp_col in result.columns:
            result['temp_squared'] = result[temp_col] ** 2  # 非线性关系
            # 度日数（HDD/CDD）
            result['cooling_degree'] = np.maximum(result[temp_col] - 26, 0)  # 制冷度日（>26°C）
            result['heating_degree'] = np.maximum(18 - result[temp_col], 0)  # 制热度日（<18°C）

        # 湿度影响
        if humidity_col in result.columns:
            result['humidity_comfort'] = np.abs(result[humidity_col] - 50)  # 偏离舒适区

        # 温湿指数（THI）
        if temp_col in result.columns and humidity_col in result.columns:
            result['thi'] = (
                result[temp_col]
                - 0.55 * (1 - result[humidity_col] / 100)
                * (result[temp_col] - 14.5)
            )

        return result

    def build_features(
        self,
        df: pd.DataFrame,
        building_params: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """构建完整的特征集"""
        result = df.copy()

        # 1. 时间特征
        result = self.extract_time_features(result)

        # 2. 天气特征
        result = self.extract_weather_features(result)

        # 3. 建筑参数特征
        if building_params:
            result['area_sqm'] = building_params.get('area_sqm', 10000)
            result['floors'] = building_params.get('floors', 10)
            result['window_ratio'] = building_params.get('window_ratio', 0.35)
            result['year_built'] = building_params.get('year_built', 2010)
            result['age'] = 2026 - result['year_built']
            result['baseline_eui'] = building_params.get('baseline_eui_kwh_per_sqm', 100)

            # 窗墙比 × 温度交互
            if 'temperature' in result.columns:
                result['window_temp_interact'] = result['window_ratio'] * result['temperature']

        # 4. 滞后特征（历史能耗模式）
        if 'energy_kwh' in result.columns:
            for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
                result[f'energy_lag_{lag}h'] = result['energy_kwh'].shift(lag)

            # 滚动统计
            for window in [6, 12, 24]:
                result[f'energy_roll_mean_{window}h'] = result['energy_kwh'].rolling(window, min_periods=1).mean()
                result[f'energy_roll_std_{window}h'] = result['energy_kwh'].rolling(window, min_periods=1).std()
                result[f'energy_roll_max_{window}h'] = result['energy_kwh'].rolling(window, min_periods=1).max()
                result[f'energy_roll_min_{window}h'] = result['energy_kwh'].rolling(window, min_periods=1).min()

        return result

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'energy_kwh',
        building_params: Optional[Dict] = None,
    ) -> Dict:
        """训练预测模型"""
        print(f"[Predictor] 训练方法: {self.method}")
        print(f"[Predictor] 数据量: {len(df)} 条")

        # 构建特征
        df_features = self.build_features(df, building_params)

        # 移除含 NaN 的行（滞后特征产生的）
        df_features = df_features.dropna().reset_index(drop=True)

        # 特征列选择
        exclude_cols = ['timestamp', target_col]
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        X = df_features[feature_cols].values
        y = df_features[target_col].values

        self.feature_names = feature_cols

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self._is_trained = True

        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            top_features = sorted(
                zip(feature_cols, importance),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            print(f"[Predictor] 训练完成! 特征数: {len(feature_cols)}")
            print("[Predictor] Top 10 重要特征:")
            for name, score in top_features:
                print(f"  {name}: {score:.4f}")

        return {
            'model_type': self.method,
            'n_samples': len(df_features),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
        }

    def predict(self, df: pd.DataFrame, building_params: Optional[Dict] = None) -> np.ndarray:
        """预测能耗"""
        if not self._is_trained or self.model is None:
            raise ValueError("模型未训练，请先调用 train()")

        df_features = self.build_features(df, building_params)

        if self.feature_names:
            missing = [c for c in self.feature_names if c not in df_features.columns]
            if missing:
                for m in missing:
                    df_features[m] = 0
            feature_cols = [c for c in self.feature_names if c in df_features.columns]
        else:
            exclude_cols = ['timestamp', 'energy_kwh']
            feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        X = df_features[feature_cols].values
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def save(self, path: str):
        """保存模型"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'method': self.method,
        }, path)
        print(f"[Predictor] 模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.method = data.get('method', 'unknown')
        self._is_trained = True
        print(f"[Predictor] 模型已加载: {path}")


class EnsemblePredictor:
    """集成预测器：组合多个模型提升精度"""

    def __init__(self):
        self.predictors = {}
        self.weights = {}

    def add_predictor(
        self,
        name: str,
        predictor: BuildingEnergyPredictor,
        weight: float = 1.0
    ):
        self.predictors[name] = predictor
        self.weights[name] = weight

    def predict(self, df: pd.DataFrame, building_params: Optional[Dict] = None) -> np.ndarray:
        """集成预测（加权平均）"""
        if not self.predictors:
            raise ValueError("没有添加任何预测器")

        predictions = []
        weight_sum = sum(self.weights.values())

        for name, predictor in self.predictors.items():
            pred = predictor.predict(df, building_params)
            weight = self.weights[name] / weight_sum
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)
