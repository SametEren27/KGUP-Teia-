import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Modeller
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def prepare_advanced_data(file_path: str):
    print("[*] Veri yükleniyor ve gelişmiş öznitelikler çıkarılıyor...")
    df = pd.read_csv(file_path, sep=';')
    df = df.dropna(how='all', axis=1)
    df.columns = df.columns.str.strip()

    # Tarih-saat formatlama
    df['Tarih_Saat'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'].astype(str) + ':00', format='%d/%m/%Y %H:%M')
    df = df.sort_values('Tarih_Saat').reset_index(drop=True)
    
    # Hedef Değişken (Pozitif Tüketim MWh)
    df['Tuketim_MWh'] = df['Tüketim KGÜP'].abs()
    
    # 1. Takvim ve Döngüsel Öznitelikler
    df['Hour'] = df['Tarih_Saat'].dt.hour
    df['DayOfWeek'] = df['Tarih_Saat'].dt.dayofweek
    df['Month'] = df['Tarih_Saat'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Döngüsel Saat (Trigonometrik Dönüşüm)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)

    # 2. Gecikme (Lag) Öznitelikleri
    df['Lag_1h'] = df['Tuketim_MWh'].shift(1)
    df['Lag_2h'] = df['Tuketim_MWh'].shift(2)
    df['Lag_24h'] = df['Tuketim_MWh'].shift(24)

    # 3. Kayan Ortalamalar (Rolling Statistics)
    df['Rolling_Mean_3h'] = df['Tuketim_MWh'].shift(1).rolling(window=3).mean()
    df['Rolling_Mean_6h'] = df['Tuketim_MWh'].shift(1).rolling(window=6).mean()

    # 4. Yenilenebilir Üretim Etkisi
    if 'gunes' in df.columns and 'Rüzgar' in df.columns:
        df['Solar_Gen'] = df['gunes']
        df['Wind_Gen'] = df['Rüzgar']

    df = df.dropna().reset_index(drop=True)
    return df

def benchmark_models(df: pd.DataFrame):
    feature_cols = [
        'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 
        'Hour_Sin', 'Hour_Cos', 
        'Lag_1h', 'Lag_2h', 'Lag_24h', 
        'Rolling_Mean_3h', 'Rolling_Mean_6h'
    ]
    if 'Solar_Gen' in df.columns:
        feature_cols.extend(['Solar_Gen', 'Wind_Gen'])
        
    target = 'Tuketim_MWh'

    # Son 3 gün (72 saat) Test Seti
    test_size = 72
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:].copy()

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    }

    results = []
    trained_models = {}

    print(f"\n[*] Model Benchmarking Başlatıldı ({len(models)} Algoritma)...")
    print("=" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mape = mean_absolute_percentage_error(y_test, preds) * 100
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results.append({
            "Model": name,
            "MAPE (%)": round(mape, 3),
            "RMSE (MWh)": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(by="MAPE (%)").reset_index(drop=True)
    print(results_df.to_string(index=False))
    print("=" * 65)

    # En iyi modeli seç ve kaydet
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, "best_model.joblib")
    print(f"\n[+] En Başarılı Model: '{best_model_name}' -> 'best_model.joblib' olarak kaydedildi.")

    # Feature Importance Grafiği
    if hasattr(best_model, "feature_importances_"):
        plt.figure(figsize=(9, 5))
        importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
        importances.plot(kind='barh', color='#007acc')
        plt.title(f"Öznitelik Önem Düzeyleri ({best_model_name})")
        plt.xlabel("Önem Skoru")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        print("[+] 'feature_importance.png' grafiği başarıyla kaydedildi.")

if __name__ == "__main__":
    DATA_FILE = "test.csv"
    df = prepare_advanced_data(DATA_FILE)
    benchmark_models(df)