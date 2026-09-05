-- scripts/init_db.sql
CREATE TABLE IF NOT EXISTS grid_telemetry (
    timestamp TIMESTAMP PRIMARY KEY,
    consumption_mwh NUMERIC(10, 2) NOT NULL,
    solar_mwh NUMERIC(10, 2) DEFAULT 0.0,
    wind_mwh NUMERIC(10, 2) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Zaman serisi aralık sorgularını hızlandıran B-Tree indeksi
CREATE INDEX IF NOT EXISTS idx_grid_telemetry_timestamp 
ON grid_telemetry (timestamp DESC);