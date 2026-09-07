import os
import sys

sys.path.append(".")

from src.etl.extract import extract_live_epias
from src.etl.load import get_engine, upsert_telemetry_to_sql


def run_daily_ingest():
  db_url = os.getenv("DATABASE_URL")
  if not db_url:
    raise ValueError(
        "❌ DATABASE_URL secret bulunamadı! GitHub Secrets ayarlarını kontrol"
        " et."
    )

  print("📡 EPİAŞ canlı telemetri akışı çekiliyor...")
  api_key = os.getenv("EPIAS_API_KEY")

  df_live = extract_live_epias(api_key=api_key, hours_back=24)

  engine = get_engine(db_url)
  inserted = upsert_telemetry_to_sql(df_live, engine)
  print(f"✅ {inserted} adet yeni canlı kayıt Neon DB'ye eklendi/güncellendi.")


if __name__ == "__main__":
  run_daily_ingest()