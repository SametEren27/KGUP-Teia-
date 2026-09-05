#type:ignore 


import pandas as pd
from sqlalchemy import Engine, text


def upsert_telemetry_to_sql(df: pd.DataFrame, engine: Engine) -> int:
  """Veriyi ilişkisel veritabanına UPSERT mantığıyla yazar."""
  upsert_stmt = text("""
        INSERT INTO grid_telemetry (timestamp, consumption_mwh, solar_mwh, wind_mwh)
        VALUES (:timestamp, :consumption_mwh, :solar_mwh, :wind_mwh)
        ON CONFLICT (timestamp) 
        DO UPDATE SET 
            consumption_mwh = EXCLUDED.consumption_mwh,
            solar_mwh = EXCLUDED.solar_mwh,
            wind_mwh = EXCLUDED.wind_mwh;
    """)

  records = df[
      ["timestamp", "consumption_mwh", "solar_mwh", "wind_mwh"]
  ].to_dict(orient="records")

  with engine.begin() as conn:
    conn.execute(upsert_stmt, records)

  return len(records)

import os
import pandas as pd
from sqlalchemy import create_engine, text

def get_engine(db_url: str = None):
    url = db_url or os.getenv("DATABASE_URL")
    if not url:
        # Fallback yerel Docker / varsayılan
        url = "postgresql://teias_admin:teias_password_2026@localhost:5432/teias_grid_db"
    return create_engine(url, pool_pre_ping=True)

def fetch_grid_history_from_sql(engine, start_date=None, end_date=None) -> pd.DataFrame:
    """RAM tüketimini önlemek için sorguyu SQL seviyesinde filtreler."""
    query = "SELECT timestamp, consumption_mwh, solar_mwh, wind_mwh FROM grid_telemetry"
    params = {}
    
    if start_date and end_date:
        query += " WHERE timestamp BETWEEN :start_date AND :end_date"
        params = {"start_date": start_date, "end_date": end_date}
        
    query += " ORDER BY timestamp ASC;"
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
        
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df