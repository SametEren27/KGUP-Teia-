
# TEİAŞ KGÜP & Grid Telemetry ETL Pipeline

End-to-end automated data engineering and machine learning pipeline that monitors, forecasts, and visualizes Turkish electricity transmission grid data (TEİAŞ) and Day-Ahead Market / Balancing Market (EPİAŞ) metrics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sameteren27-kgup-teia--main-app-py.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Database](https://img.shields.io/badge/Database-Neon%20PostgreSQL-green?logo=postgresql)
![Automation](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)
![Framework](https://img.shields.io/badge/ML-PyTorch%20%7C%20Scikit--Learn-orange)

---

## Architecture Overview

```text
       [ EPİAŞ / Grid APIs ]
                 │
                 ▼
    [ GitHub Actions Cron (Daily) ]
     ├── Extract: Daily telemetry
     ├── Transform: Feature engineering & validation
     └── Load: Upsert into Cloud Database
                 │
                 ▼
       [ Neon PostgreSQL ] (Cloud Storage)
                 │
                 ▼
     [ Streamlit Cloud Platform ]
     ├── Hybrid Forecasting (RandomForest / GradientBoosting + LSTM Residuals)
     ├── Grid Imbalance & Market Funnel Analysis
     └── Automated PDF & Analytics Reporting
 Key Features
Automated Live ETL: Scheduled daily cron runs via GitHub Actions (daily_pipeline.yml) to ingest and validate telemetry data with zero manual intervention.

Resilient Fallback: Dynamic generation of realistic seasonal load patterns if upstream EPİAŞ API endpoints face latency or authorization downtime.

Hybrid Machine Learning: Combines tabular regressors with PyTorch LSTM residual learners to minimize forecast error in balancing volume predictions.

Market Funnel & Imbalance Cost Analytics: Real-time calculation of PTF/SMF penalties, tolerance bands, and system settlement discrepancies.

Automated Testing: pytest suite ensuring pipeline integrity, schema validation, and database load guarantees prior to every ingestion.

🛠 Tech Stack
Data Engineering: pandas, sqlalchemy, psycopg2-binary

Database: Serverless Neon PostgreSQL

Orchestration: GitHub Actions (Scheduled Cron)

Machine Learning: torch (LSTM), scikit-learn, joblib

Frontend / Dashboard: Streamlit, plotly

Testing: pytest

 Local Setup
1. Clone & Environment Setup
Bash
git clone [https://github.com/SametEren27/KGUP-Teia-.git](https://github.com/SametEren27/KGUP-Teia-.git)
cd KGUP-Teia-
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
2. Configure Environment Variables
Create a .env or .streamlit/secrets.toml file:

Ini, TOML
DATABASE_URL = "postgresql://user:password@host/neondb?sslmode=require"
EPIAS_API_KEY = "your_optional_api_key"
3. Run Pipeline or Dashboard
Bash
# Run tests
pytest

# Manual Ingestion
python scripts/daily_live_ingest.py

# Launch UI
streamlit run app.py
 Author
Samet Eren

GitHub: @SametEren27