#type:ignore 


import numpy as np
import pandas as pd


def validate_raw_schema(df: pd.DataFrame) -> None:
  """Gelen verinin pipeline standartlarına uygunluğunu denetler."""
  required_cols = {"timestamp", "consumption_mwh"}
  if not required_cols.issubset(df.columns):
    missing = required_cols - set(df.columns)
    raise ValueError(f"Eksik zorunlu kolonlar: {missing}")

  if (df["consumption_mwh"] < 0).any():
    raise ValueError(
        "Negatif tüketim değeri tespit edildi, veri şebeke mantığına aykırı."
    )


def transform_grid_data(df: pd.DataFrame) -> pd.DataFrame:
  """Ham veriyi temizler ve modelleme/analiz özniteliklerini türetir."""
  validate_raw_schema(df)
  data = df.copy().sort_values("timestamp").reset_index(drop=True)

  # Eksik zaman aralıklarını doldur ve enterpole et
  data = data.set_index("timestamp").asfreq("h")
  data["consumption_mwh"] = data["consumption_mwh"].interpolate(
      method="linear"
  )
  data["solar_mwh"] = data.get("solar_mwh", 0.0).fillna(0.0)
  data["wind_mwh"] = data.get("wind_mwh", 0.0).fillna(0.0)
  data = data.reset_index()

  # Zaman ve takvim öznitelikleri
  data["Hour"] = data["timestamp"].dt.hour
  data["DayOfWeek"] = data["timestamp"].dt.dayofweek
  data["Month"] = data["timestamp"].dt.month
  data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

  # Periyodik sinyal dönüşümleri
  data["Hour_Sin"] = np.sin(2 * np.pi * data["Hour"] / 24.0)
  data["Hour_Cos"] = np.cos(2 * np.pi * data["Hour"] / 24.0)

  # Gecikme (Lag) ve hareketli pencere öznitelikleri
  data["Lag_1h"] = data["consumption_mwh"].shift(1)
  data["Lag_24h"] = data["consumption_mwh"].shift(24)
  data["Rolling_Mean_3h"] = data["consumption_mwh"].shift(1).rolling(3).mean()
  data["Volatility_24h"] = data["consumption_mwh"].rolling(24).std().fillna(0)

  # Referans plan simülasyonu
  data["kgup_plan_mwh"] = data["Lag_24h"].fillna(
      data["consumption_mwh"]
  ) * np.random.uniform(0.985, 1.015, len(data))

  return data.dropna().reset_index(drop=True)