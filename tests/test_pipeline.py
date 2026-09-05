#type:ignore 


import numpy as np
import pandas as pd
import pytest
from src.etl.transform import transform_grid_data, validate_raw_schema


@pytest.fixture
def mock_valid_df():
  dates = pd.date_range("2026-01-01", periods=48, freq="h")
  return pd.DataFrame({
      "timestamp": dates,
      "consumption_mwh": np.random.uniform(30000, 40000, size=48),
      "solar_mwh": np.zeros(48),
      "wind_mwh": np.zeros(48),
  })


def test_validate_raw_schema_raises_on_missing_column():
  bad_df = pd.DataFrame({"timestamp": [pd.Timestamp.now()]})
  with pytest.raises(ValueError, match="Eksik zorunlu kolonlar"):
    validate_raw_schema(bad_df)


def test_validate_raw_schema_raises_on_negative_consumption():
  bad_df = pd.DataFrame({
      "timestamp": [pd.Timestamp.now()],
      "consumption_mwh": [-150.0],
  })
  with pytest.raises(ValueError, match="Negatif tüketim"):
    validate_raw_schema(bad_df)


def test_transform_grid_data_feature_generation(mock_valid_df):
  transformed = transform_grid_data(mock_valid_df)

  expected_cols = [
      "Hour",
      "DayOfWeek",
      "Hour_Sin",
      "Hour_Cos",
      "Lag_1h",
      "Lag_24h",
      "Rolling_Mean_3h",
      "Volatility_24h",
  ]
  for col in expected_cols:
    assert col in transformed.columns
  assert not transformed.empty
  assert not transformed["Lag_24h"].isnull().any()