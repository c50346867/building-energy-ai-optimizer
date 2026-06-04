"""
Building Energy AI Optimizer — 快速演示脚本

运行: python examples/demo.py
展示完整的预测 + 优化工作流
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.simulator import EnergyDataSimulator, create_training_dataset
from backend.core.predictor import BuildingEnergyPredictor
from backend.core.optimizer import HVACOptimizer, HVACSetpoint
import warnings
warnings.filterwarnings('ignore')


def print_separator(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print("""
╔══════════════════════════════════════════════════╗
║     🏢 Building Energy AI Optimizer Demo        ║
║     AI 驱动的建筑能耗预测与智能优化              ║
╚══════════════════════════════════════════════════╝
""")

    # === 1. 数据生成 ===
    print_separator("1️⃣  能耗数据模拟")

    building_type = input("选择建筑类型 (commercial_office/residential/industrial) [default: commercial_office]: ") or "commercial_office"
    days = int(input("模拟天数 [default: 90]: ") or "90")

    print(f"\n📊 正在生成 {building_type} 建筑 {days} 天能耗数据...")
    train_df, test_df = create_training_dataset(
        building_type=building_type,
        days=days,
        train_ratio=0.8,
    )

    # === 2. 模型训练 ===
    print_separator("2️⃣  能耗预测模型训练")

    predictor = BuildingEnergyPredictor(method='lightgbm')
    train_info = predictor.train(train_df)

    # === 3. 模型评估 ===
    print_separator("3️⃣  模型评估")

    print(f"\n📈 评估测试集 ({len(test_df)} 条样本)...")
    predictions = predictor.predict(test_df)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    actuals = test_df['energy_kwh'].values[:len(predictions)]

    mae = mean_absolute_error(actuals, predictions)
    rmse = mean_squared_error(actuals, predictions, squared=False)
    r2 = r2_score(actuals, predictions)

    print(f"\n  ✅ MAE (平均绝对误差):  {mae:.2f} kWh")
    print(f"  ✅ RMSE (均方根误差):   {rmse:.2f} kWh")
    print(f"  ✅ R² (决定系数):       {r2:.4f}")
    print(f"  ✅ 平均相对误差:        {mae / actuals.mean() * 100:.1f}%")

    # === 4. 能耗优化 ===
    print_separator("4️⃣  HVAC 系统优化")

    optimizer = HVACOptimizer()
    params = HVACSetpoint(
        cooling_setpoint=26.0,
        heating_setpoint=20.0,
    )

    print("\n⚡ 执行 HVAC 优化调度...")
    result = optimizer.optimize_setpoints(test_df, params)

    print(f"\n  💰 节能总计:       {result.savings_kwh:.0f} kWh ({result.savings_percent:.1f}%)")
    print(f"  💵 电费节省:       ¥{result.cost_savings_cny:.0f}")
    print(f"  🌱 CO₂ 减排:       {result.co2_reduction_kg:.1f} kg")
    print(f"  📉 原总能耗:       {result.total_original:.0f} kWh")
    print(f"  📈 优化后总能耗:   {result.total_optimized:.0f} kWh")

    # === 5. 优化建议 ===
    print_separator("5️⃣  优化建议列表")

    for i, rec in enumerate(result.recommendations, 1):
        priority_map = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        print(f"\n  {priority_map.get(rec['priority'], '⚪')} {i}. {rec['title']}")
        print(f"     {rec['description']}")
        print(f"     💡 节能潜力: {rec['savings_potential']}")

    # === 6. 完整总结 ===
    print_separator("📊 总结报告")

    print(f"""
  🏛 建筑类型:     {building_type}
  📅 分析周期:     {days} 天 ({days * 24:,} 条数据)
  🤖 预测模型:     LightGBM (R²={r2:.3f})
  ⚡ 节能率:        {result.savings_percent:.1f}%
  💵 预期年节省:    ¥{result.cost_savings_cny * (365 / days):,.0f}
  🌱 年减排 CO₂:    {result.co2_reduction_kg * (365 / days):.1f} kg

  🏆 项目: building-energy-ai-optimizer
  📖 GitHub: https://github.com/c50346867/building-energy-ai-optimizer
""")


if __name__ == "__main__":
    main()
