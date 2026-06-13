import pandas as pd
import numpy as np

# Create a 30-day hourly timeline using lowercase 'h' for compatibility
timeline = pd.date_range(start="2026-06-01", end="2026-06-30 23:00:00", freq="h")
total_hours = len(timeline)
hour_indicators = timeline.hour

# Simulate Weather Supply (Solar curves peaking at 12:00 PM noon)
solar_base = np.where((hour_indicators > 6) & (hour_indicators < 18), np.sin((hour_indicators - 6) / 12 * np.pi) * 850, 0)
solar_telemetry = np.clip(solar_base + np.random.normal(0, 40, total_hours), 0, None)

# Simulate Community Demand (Double peak structure: 9 AM and 8 PM)
demand_base = 60 + 25 * np.sin((hour_indicators - 4) / 24 * 2 * np.pi) + 20 * np.sin((hour_indicators - 16) / 24 * 2 * np.pi)
load_telemetry = np.clip(demand_base + np.random.normal(0, 6, total_hours), 15, None)

grid_dataframe = pd.DataFrame({
    "Timestamp": timeline,
    "Hour": hour_indicators,
    "Solar_Irradiance": solar_telemetry,
    "Community_Demand": load_telemetry
})

grid_dataframe.to_csv("grid_data.csv", index=False)
print("✅ Dataset compiled successfully as 'grid_data.csv'!")