import sys
import os
sys.path.append(".")

from sqlalchemy import text
from src.etl.extract import extract_from_csv, extract_live_epias
from src.etl.load import get_engine, upsert_telemetry_to_sql

def create_table_if_not_exists(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS grid_telemetry (
        timestamp TIMESTAMP PRIMARY KEY,
        consumption_mwh NUMERIC(10, 2) NOT NULL,
        solar_mwh NUMERIC(10, 2) DEFAULT 0.0,
        wind_mwh NUMERIC(10, 2) DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_grid_telemetry_timestamp 
    ON grid_telemetry (timestamp DESC);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("🛠️ Tablo kontrol edildi / oluşturuldu.")

def run_sync():
    print("⏳ Veritabanı bağlantısı kuruluyor...")
    engine = get_engine()
    
    create_table_if_not_exists(engine)
    
    if os.path.exists("KGUP.csv"):
        print("📁 KGUP.csv veritabanına aktarılıyor...")
        df_csv = extract_from_csv("KGUP.csv")
        count = upsert_telemetry_to_sql(df_csv, engine)
        print(f"✅ {count} adet geçmiş telemetri kaydı SQL'e aktarıldı.")
    else:
        print("📡 Canlı akış veritabanına aktarılıyor...")
        df_live = extract_live_epias(hours_back=168)
        count = upsert_telemetry_to_sql(df_live, engine)
        print(f"✅ {count} adet canlı telemetri kaydı SQL'e aktarıldı.")

if __name__ == "__main__":
    run_sync()