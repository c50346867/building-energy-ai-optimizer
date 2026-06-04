"""
Building Energy AI Optimizer — REST API
FastAPI 后端，提供能耗预测和优化服务
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import json

from backend.core.predictor import BuildingEnergyPredictor
from backend.core.optimizer import HVACOptimizer, HVACSetpoint
from backend.core.simulator import EnergyDataSimulator, create_training_dataset

app = FastAPI(
    title="Building Energy AI Optimizer API",
    description="AI-driven building energy prediction and optimization",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Pydantic Models ===

class BuildingConfig(BaseModel):
    building_id: str = "bld_001"
    building_type: str = "commercial_office"
    area_sqm: float = 50000
    floors: int = 30
    location: str = "Shanghai"


class PredictionRequest(BaseModel):
    building: BuildingConfig
    data: Optional[List[Dict]] = None
    days: int = 7


class PredictionResponse(BaseModel):
    predictions: List[Dict]
    feature_importance: Optional[Dict] = None
    model_info: Dict


class OptimizationRequest(BaseModel):
    building: BuildingConfig
    data: Optional[List[Dict]] = None
    cooling_setpoint: float = 26.0
    heating_setpoint: float = 20.0


class OptimizationResponse(BaseModel):
    savings_kwh: float
    savings_percent: float
    cost_savings_cny: float
    co2_reduction_kg: float
    recommendations: List[Dict]
    hourly_data: List[Dict]


# === Global State ===
_models = {}
_SIM_CACHE = {}


# === Routes ===

@app.get("/")
def root():
    return {
        "service": "Building Energy AI Optimizer",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/api/building-types")
def list_building_types():
    """列出支持的建筑类型"""
    return {
        "types": [
            {"id": "commercial_office", "name": "商业办公楼"},
            {"id": "residential", "name": "住宅"},
            {"id": "industrial", "name": "工业厂房"},
            {"id": "mall", "name": "购物中心"},
            {"id": "hospital", "name": "医院"},
        ]
    }


@app.post("/api/simulate", response_model=Dict)
def simulate_data(
    building_type: str = "commercial_office",
    days: int = 30,
):
    """模拟生成建筑能耗数据"""
    sim = EnergyDataSimulator(building_type=building_type)
    df = sim.generate(days=days)

    cache_key = f"{building_type}_{days}"
    _SIM_CACHE[cache_key] = df

    return {
        "building_type": building_type,
        "records": len(df),
        "mean_energy": round(df['energy_kwh'].mean(), 1),
        "peak_energy": round(df['energy_kwh'].max(), 1),
        "sample_data": df.head(24).to_dict(orient='records'),
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict_energy(request: PredictionRequest):
    """预测能耗"""
    building = request.building

    if request.data:
        df = pd.DataFrame(request.data)
    else:
        # 从模拟器获取数据
        sim = EnergyDataSimulator(building_type=building.building_type)
        df = sim.generate(days=request.days + 7)

    # 训练预测模型
    predictor = BuildingEnergyPredictor(method='lightgbm')
    train_info = predictor.train(df)

    # 预测未来
    last_timestamp = pd.to_datetime(df['timestamp']).max()
    future_dates = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=72,
        freq='1h',
    )
    future_df = pd.DataFrame({'timestamp': future_dates})
    future_df['temperature'] = 25
    future_df['humidity'] = 60
    future_df['energy_kwh'] = 0

    predictions = predictor.predict(future_df)

    # 特征重要性
    feature_importance = {}
    if (hasattr(predictor.model, 'feature_importances_')
            and predictor.feature_names):
        for name, score in zip(
            predictor.feature_names,
            predictor.model.feature_importances_
        ):
            feature_importance[name] = round(float(score), 4)
        # Top 10
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: -x[1])[:10]
        )

    return PredictionResponse(
        predictions=[
            {"timestamp": str(t), "predicted_kwh": round(float(p), 2)}
            for t, p in zip(future_dates, predictions)
        ],
        feature_importance=feature_importance,
        model_info=train_info,
    )


@app.post("/api/optimize", response_model=OptimizationResponse)
def optimize_energy(request: OptimizationRequest):
    """优化建筑能耗"""
    building = request.building

    if request.data:
        df = pd.DataFrame(request.data)
    else:
        cache_key = f"{building.building_type}_30"
        if cache_key in _SIM_CACHE:
            df = _SIM_CACHE[cache_key]
        else:
            sim = EnergyDataSimulator(building_type=building.building_type)
            df = sim.generate(days=30)
            _SIM_CACHE[cache_key] = df

    optimizer = HVACOptimizer()
    hvac_params = HVACSetpoint(
        cooling_setpoint=request.cooling_setpoint,
        heating_setpoint=request.heating_setpoint,
    )

    # 执行优化
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    if 'is_workday' not in df.columns:
        df['is_workday'] = 1 - df['day_of_week'].apply(
            lambda x: 1 if x >= 5 else 0
        ) if 'day_of_week' in df.columns else 1

    result = optimizer.optimize_setpoints(df, hvac_params)

    return OptimizationResponse(
        savings_kwh=round(result.savings_kwh, 2),
        savings_percent=round(result.savings_percent, 2),
        cost_savings_cny=round(result.cost_savings_cny, 2),
        co2_reduction_kg=round(result.co2_reduction_kg, 2),
        recommendations=result.recommendations,
        hourly_data=result.hourly_schedule.to_dict(orient='records'),
    )


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
