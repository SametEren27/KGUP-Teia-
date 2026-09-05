from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="TEİAŞ Şebeke Tahmin ve Anomali API",
    description="Elektrik şebekesi saatlik yük, net yük ve anomali tespit mikroservisi",
    version="2.0.0"
)

try:
    model = joblib.load("best_model.joblib")
except Exception:
    model = None

class PredictionRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    lag_1h: float = Field(..., gt=0)
    lag_2h: float = Field(..., gt=0)
    lag_24h: float = Field(..., gt=0)
    rolling_mean_3h: float = Field(..., gt=0)
    rolling_mean_6h: float = Field(..., gt=0)
    solar_gen: float = Field(default=0.0, ge=0)
    wind_gen: float = Field(default=0.0, ge=0)
    actual_load: float = Field(default=None, description="Opsiyonel: Anomali kontrolü için gerçekleşen değer")

@app.get("/")
def health():
    return {"status": "online", "model": "LightGBM Production"}

@app.post("/predict")
def predict_and_analyze(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model yüklü değil.")

    hour_sin = np.sin(2 * np.pi * req.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * req.hour / 24.0)
    is_weekend = 1 if req.day_of_week in [5, 6] else 0

    input_df = pd.DataFrame([{
        'Hour': req.hour,
        'DayOfWeek': req.day_of_week,
        'Month': req.month,
        'IsWeekend': is_weekend,
        'Hour_Sin': hour_sin,
        'Hour_Cos': hour_cos,
        'Lag_1h': req.lag_1h,
        'Lag_2h': req.lag_2h,
        'Lag_24h': req.lag_24h,
        'Rolling_Mean_3h': req.rolling_mean_3h,
        'Rolling_Mean_6h': req.rolling_mean_6h,
        'Solar_Gen': req.solar_gen,
        'Wind_Gen': req.wind_gen
    }])

    pred = float(model.predict(input_df)[0])
    renewable_total = req.solar_gen + req.wind_gen
    net_load = pred - renewable_total

    response = {
        "hour": req.hour,
        "predicted_gross_load_mwh": round(pred, 2),
        "predicted_net_load_mwh": round(net_load, 2),
        "is_peak_alert": pred >= 51500,
        "anomaly_status": "NORMAL"
    }

    if req.actual_load is not None:
        error = abs(req.actual_load - pred)
        if error > 1200:  # 1200 MWh sapma kritik eşik
            response["anomaly_status"] = "CRITICAL_ANOMALY_DETECTED"
            response["error_margin_mwh"] = round(error, 2)

    return response