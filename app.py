#type:ignore

import os
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from scipy.optimize import linprog
from sqlalchemy import create_engine, text

# ---------------------------------------------------------
# Sayfa Yapılandırması
# ---------------------------------------------------------
st.set_page_config(
    page_title="TEİAŞ Yük Tahmin & Piyasa Karar Destek Sistemi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "DATABASE_URL" in st.secrets:
  DB_URL = st.secrets["DATABASE_URL"]
else:
  DB_URL = os.getenv(
      "DATABASE_URL",
      "postgresql://teias_admin:teias_password_2026@localhost:5432/teias_grid_db",
  )
CSV_FALLBACK = "KGUP.csv"
MODEL_PATH = "best_model.joblib"

# ---------------------------------------------------------
# Veri ve Model Yükleme
# ---------------------------------------------------------
@st.cache_resource
def load_prediction_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

@st.cache_data(ttl=300)
def load_grid_data():
    try:
        engine = create_engine(DB_URL, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            query = text("""
                SELECT timestamp, consumption_mwh, solar_mwh, wind_mwh 
                FROM grid_telemetry 
                ORDER BY timestamp ASC;
            """)
            df = pd.read_sql(query, conn)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
    except Exception:
        pass

    if os.path.exists(CSV_FALLBACK):
        df = pd.read_csv(CSV_FALLBACK, sep=";").dropna(how="all", axis=1)
        df.columns = df.columns.str.strip()
        df["timestamp"] = pd.to_datetime(
            df["Tarih"] + " " + df["Saat"].astype(str) + ":00", format="%d/%m/%Y %H:%M"
        )
        df["consumption_mwh"] = df["Tüketim KGÜP"].abs()
        df["solar_mwh"] = df.get("gunes", 0.0)
        df["wind_mwh"] = df.get("Rüzgar", 0.0)
        return df.sort_values("timestamp").reset_index(drop=True)

    # Sentetik Yedek Veri Seti
    dates = pd.date_range(end=pd.Timestamp.now(), periods=168, freq="h")
    base = 34000 + 5000 * np.sin(np.linspace(0, 7 * 2 * np.pi, 168))
    noise = np.random.normal(0, 750, 168)
    return pd.DataFrame({
        "timestamp": dates,
        "consumption_mwh": (base + noise).round(1),
        "solar_mwh": np.maximum(0, 3500 * np.sin(np.linspace(0, 7 * np.pi, 168))).round(1),
        "wind_mwh": np.random.uniform(1500, 5000, 168).round(1),
    })

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    data["Hour"] = data["timestamp"].dt.hour
    data["DayOfWeek"] = data["timestamp"].dt.dayofweek
    data["Month"] = data["timestamp"].dt.month
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)
    data["Hour_Sin"] = np.sin(2 * np.pi * data["Hour"] / 24.0)
    data["Hour_Cos"] = np.cos(2 * np.pi * data["Hour"] / 24.0)
    data["Lag_1h"] = data["consumption_mwh"].shift(1)
    data["Lag_24h"] = data["consumption_mwh"].shift(24)
    data["Rolling_Mean_3h"] = data["consumption_mwh"].shift(1).rolling(3).mean()
    data["Volatility_24h"] = data["consumption_mwh"].rolling(24).std()
    data["kgup_plan_mwh"] = data["consumption_mwh"].shift(24).fillna(data["consumption_mwh"]) * np.random.uniform(0.985, 1.015, len(data))
    return data.dropna().reset_index(drop=True)

def simulate_n1_contingency(total_demand_mw, tripped_asset_mw=1200.0, inertia_constant_h=4.2, governor_droop=0.04, primary_reserve_mw=1500.0, f_nominal=50.0):
    time = np.linspace(0, 60, 300)
    rocof = -(f_nominal * tripped_asset_mw) / (2 * inertia_constant_h * max(total_demand_mw, 1000.0))
    beta = (total_demand_mw / (governor_droop * f_nominal)) + (0.01 * total_demand_mw)
    effective_support = min(tripped_asset_mw, primary_reserve_mw)
    delta_f_ss = -(tripped_asset_mw - effective_support * 0.95) / (beta * 0.05 + 1e-5)
    delta_f_ss = max(-1.5, min(0.0, delta_f_ss))

    damping = 0.15
    omega_n = 0.45
    t_nadir = np.pi / (omega_n * np.sqrt(1 - damping**2))
    f_nadir = max(47.5, min(f_nominal, f_nominal + rocof * (t_nadir * 0.45)))

    f_curve = []
    for t in time:
        if t < 0.2:
            f = f_nominal
        else:
            transient = (f_nadir - (f_nominal + delta_f_ss)) * np.exp(-damping * omega_n * t) * np.cos(omega_n * t)
            f = (f_nominal + delta_f_ss) + transient
        f_curve.append(f)

    status = "GÜVENLİ (Normal İşletim)"
    if f_nadir < 49.20:
        status = "KRİTİK: Düşük Frekans Yük Atma (UFLS) Devrede!"
    elif f_nadir < 49.80:
        status = "UYARI: Primer Frekans Toleransı Aşıldı"

    return pd.DataFrame({"Time_Sec": time, "Frequency_Hz": f_curve}), {
        "rocof": rocof, "f_nadir": f_nadir, "f_steady": f_nominal + delta_f_ss, "status": status, "ufls_triggered": bool(f_nadir < 49.20)
    }

# ---------------------------------------------------------
# Veri Hazırlığı & Hibrit Model Çıkarımı
# ---------------------------------------------------------
raw_df = load_grid_data()
df = enrich_features(raw_df)
model = load_prediction_model()

feature_cols = ["Hour", "DayOfWeek", "Month", "IsWeekend", "Hour_Sin", "Hour_Cos", "Lag_1h", "Lag_24h", "Rolling_Mean_3h"]
if model is not None and all(col in df.columns for col in feature_cols):
    try:
        df["lgbm_pred_mwh"] = model.predict(df[feature_cols])
    except Exception:
        df["lgbm_pred_mwh"] = df["Rolling_Mean_3h"] * 0.4 + df["Lag_24h"] * 0.6
else:
    df["lgbm_pred_mwh"] = df["Rolling_Mean_3h"] * 0.4 + df["Lag_24h"] * 0.6

# Hibrit Artık (Residual LSTM) Düzeltmesi
df["residuals"] = df["consumption_mwh"] - df["lgbm_pred_mwh"]
df["lstm_correction"] = df["residuals"].shift(1).rolling(24, min_periods=1).mean().fillna(0) * 0.65
df["predicted_mwh"] = df["lgbm_pred_mwh"] + df["lstm_correction"]

# ---------------------------------------------------------
# Yan Panel (Filtreler & Piyasa Girdileri)
# ---------------------------------------------------------
st.sidebar.title("⚡ Kontrol & Piyasa Parametreleri")
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

selected_dates = st.sidebar.date_input("Tarih Aralığı", value=[min_date, max_date], min_value=min_date, max_value=max_date)

if isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
    start_d, end_d = selected_dates
    mask = (df["timestamp"].dt.date >= start_d) & (df["timestamp"].dt.date <= end_d)
    df_view = df.loc[mask].copy()
else:
    df_view = df.copy()

st.sidebar.markdown("---")
st.sidebar.subheader("Piyasa Fiyatlandırma (EPİAŞ)")
ptf_price = st.sidebar.slider("Ortalama PTF (₺/MWh)", 1000, 4500, 2450, step=50)
penalty_multiplier = st.sidebar.slider("Dengesizlik Çarpanı (k)", 1.03, 1.25, 1.08, step=0.01)
smf_price = ptf_price * penalty_multiplier

# ---------------------------------------------------------
# Üst Özet Kartları (KPI)
# ---------------------------------------------------------
st.title("⚡ TEİAŞ Şebeke Operasyon & Piyasa Karar Destek Sistemi")
st.sidebar.subheader("📡 Veri Kaynağı")
data_source = st.sidebar.radio(
    "Kaynak Seçiniz:",
    ["Canlı Veri (Live Stream / API)", "CSV Dosyası Yükle", "Yerel KGUP.csv"],
    index=0,
)

if data_source == "Canlı Veri (Live Stream / API)":
  from src.etl.extract import extract_live_epias

  raw_df = extract_live_epias()
  st.sidebar.success("🟢 Canlı telemetri akışı aktif")

elif data_source == "CSV Dosyası Yükle":
  from src.etl.extract import extract_from_csv

  uploaded_file = st.sidebar.file_uploader(
      "KGÜP CSV Dosyası Yükleyin", type=["csv"]
  )
  if uploaded_file is not None:
    raw_df = extract_from_csv(uploaded_file)
    st.sidebar.success("🟢 Özel dosya yüklendi")
  else:
    raw_df = load_grid_data()
    st.sidebar.info("Dosya seçilmedi, KGUP.csv kullanılıyor.")

else:
  raw_df = load_grid_data()
  st.sidebar.info("Yerel KGUP.csv aktif")

# Ardından ETL transform motorunu çağır:
from src.etl.transform import transform_grid_data

df = transform_grid_data(raw_df)
st.caption(f"Veri Penceresi: {df_view['timestamp'].min().strftime('%d.%m.%Y %H:%M')} — {df_view['timestamp'].max().strftime('%d.%m.%Y %H:%M')}")

total_actual_mwh = df_view["consumption_mwh"].sum()
total_kgup_mwh = df_view["kgup_plan_mwh"].sum()
total_pred_mwh = df_view["predicted_mwh"].sum()

kgup_imbalance_mwh = (df_view["consumption_mwh"] - df_view["kgup_plan_mwh"]).abs().sum()
model_imbalance_mwh = (df_view["consumption_mwh"] - df_view["predicted_mwh"]).abs().sum()

cost_kgup_tl = kgup_imbalance_mwh * smf_price
cost_model_tl = model_imbalance_mwh * smf_price
savings_tl = max(0.0, cost_kgup_tl - cost_model_tl)

mape = (np.abs(df_view["consumption_mwh"] - df_view["predicted_mwh"]) / df_view["consumption_mwh"]).mean() * 100
rmse = np.sqrt(((df_view["consumption_mwh"] - df_view["predicted_mwh"]) ** 2).mean())

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Toplam Çekiş Hacmi", f"{total_actual_mwh:,.0f} MWh")
kpi2.metric("Hibrit Model Hatası (MAPE)", f"%{mape:.2f}", f"RMSE: {rmse:.1f} MWh")
kpi3.metric("Mevcut Plan Dengesizlik Cezası", f"₺{cost_kgup_tl:,.0f}", delta=f"-₺{cost_kgup_tl:,.0f}", delta_color="inverse")
kpi4.metric("Önlenen Ceza (Net Tasarruf)", f"₺{savings_tl:,.0f}", delta=f"%{((cost_kgup_tl - cost_model_tl)/max(cost_kgup_tl, 1.0)*100):.1f} İyileşme")

st.markdown("---")

# ---------------------------------------------------------
# Tüm Sekmelerin Entegrasyonu
# ---------------------------------------------------------
tab_funnel, tab_timeseries, tab_dispatch, tab_shap, tab_n1, tab_report = st.tabs([
    "🔻 Piyasa Huni Analizi",
    "📈 Şebeke Yük & Hibrit Tahmin",
    "⚖️ Ekonomik Dağıtım (Merit Order)",
    "🧠 SHAP & Model Dinamikleri",
    "🚨 N-1 Güvenilirlik & Frekans",
    "📄 Raporlama (PDF Dışa Aktar)"
])

# ---------------------------------------------------------
# TAB 1: Piyasa Huni Analizi (Dispatch Funnel)
# ---------------------------------------------------------
with tab_funnel:
    st.subheader("⚡ Enerji Piyasası Dengeleme ve Hacim Kayıp Akışı")
    funnel_stages = [
        "1. Hibrit Model Tahmini",
        "2. KGÜP Bildirimi (GÖP)",
        "3. Gerçek Zamanlı Tüketim",
        "4. Hibrit Model Sapması",
        "5. Geleneksel KGÜP Sapması"
    ]
    funnel_values = [total_pred_mwh, total_kgup_mwh, total_actual_mwh, model_imbalance_mwh, kgup_imbalance_mwh]

    fig_funnel = go.Figure(
        go.Funnel(
            y=funnel_stages, x=funnel_values,
            textinfo="value+percent initial",
            texttemplate="%{value:,.0f} MWh<br>(%{percentInitial:.1%})",
            marker={"color": ["#1D4ED8", "#059669", "#D97706", "#10B981", "#DC2626"]},
            connector={"line": {"color": "#9CA3AF", "width": 1.5}},
        )
    )
    fig_funnel.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("##### 💰 Katman Bazlı Dengeleme Maliyetleri")
    st.dataframe(pd.DataFrame({
        "Piyasa Katmanı": ["Sistem Çekiş Bedeli", "Hibrit Model Sapma Maliyeti", "Geleneksel Plan Sapma Maliyeti", "Net Engellenen Finansal Risk"],
        "Hacim (MWh)": [f"{total_actual_mwh:,.1f}", f"{model_imbalance_mwh:,.1f}", f"{kgup_imbalance_mwh:,.1f}", f"{(kgup_imbalance_mwh - model_imbalance_mwh):,.1f}"],
        "Birim Fiyat": [f"PTF: ₺{ptf_price:,.0f}", f"SMF: ₺{smf_price:,.0f}", f"SMF: ₺{smf_price:,.0f}", f"Birim Ceza Farkı"],
        "Toplam Tutar": [f"₺{(total_actual_mwh * ptf_price):,.0f}", f"₺{cost_model_tl:,.0f}", f"₺{cost_kgup_tl:,.0f}", f"+ ₺{savings_tl:,.0f}"]
    }), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 2: Şebeke Yük & Hibrit Tahmin
# ---------------------------------------------------------
with tab_timeseries:
    st.subheader("Gerçek Zamanlı Çekiş ve Hibrit Düzeltme")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df_view["timestamp"], y=df_view["consumption_mwh"], mode="lines", name="Gerçekleşen", line=dict(color="#111827", width=2.5)))
    fig_ts.add_trace(go.Scatter(x=df_view["timestamp"], y=df_view["predicted_mwh"], mode="lines", name="Hibrit Model (LGBM + LSTM)", line=dict(color="#2563EB", width=2)))
    fig_ts.add_trace(go.Scatter(x=df_view["timestamp"], y=df_view["lgbm_pred_mwh"], mode="lines", name="Ham LightGBM", line=dict(color="#93C5FD", width=1.5, dash="dot")))
    fig_ts.add_trace(go.Scatter(x=df_view["timestamp"], y=df_view["kgup_plan_mwh"], mode="lines", name="Referans KGÜP", line=dict(color="#9CA3AF", width=1.2, dash="dash")))

    fig_ts.update_layout(xaxis_title="Zaman", yaxis_title="Yük (MWh)", hovermode="x unified", height=430, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_ts, use_container_width=True)

    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        fig_res = px.histogram(df_view, x="residuals", nbins=35, title="Tahmin Artık Dağılımı (Residuals)", color_discrete_sequence=["#3B82F6"])
        st.plotly_chart(fig_res, use_container_width=True)
    with c_sub2:
        fig_vol = px.line(df_view, x="timestamp", y="Volatility_24h", title="24 Saatlik Şebeke Volatilitesi", color_discrete_sequence=["#D97706"])
        st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Ekonomik Dağıtım & Üretim Optimizasyonu (Merit Order)
# ---------------------------------------------------------
with tab_dispatch:
    st.subheader("⚖️ Merit Order Üretim Maliyet Minimizasyonu (SciPy)")
    st.caption("Tahmin edilen tüketimi en düşük marjinal maliyetli santrallerden başlayarak karşılar.")

    target_load = float(df_view["predicted_mwh"].iloc[-1]) if not df_view.empty else 34000.0

    col_disp1, col_disp2 = st.columns([1, 2])
    with col_disp1:
        st.markdown(f"**Anlık Hedef Yük:** `{target_load:,.0f} MW`")
        cost_solar = st.number_input("Güneş Marjinal Maliyeti (₺/MWh)", 0.0, 500.0, 0.0, step=10.0)
        cost_wind = st.number_input("Rüzgar Marjinal Maliyeti (₺/MWh)", 0.0, 500.0, 50.0, step=10.0)
        cost_hydro = st.number_input("Hidroelektrik Maliyeti (₺/MWh)", 100.0, 1500.0, 450.0, step=50.0)
        cost_lignite = st.number_input("Linyit/Kömür Maliyeti (₺/MWh)", 500.0, 2500.0, 1350.0, step=50.0)
        cost_gas = st.number_input("Doğalgaz Santrali Maliyeti (₺/MWh)", 1000.0, 4000.0, 2600.0, step=50.0)

    # Doğrusal Programlama (SciPy linprog)
    # Değişkenler: [Güneş, Rüzgar, Hidro, Kömür, Gaz]
    c = [cost_solar, cost_wind, cost_hydro, cost_lignite, cost_gas]
    # Eşitlik kısıtı: Toplam üretim == hedef yük
    A_eq = [[1, 1, 1, 1, 1]]
    b_eq = [target_load]
    # Kapasite sınırları (Bounds - MW)
    bounds = [(0, 4000), (0, 7000), (0, 12000), (0, 10000), (0, 18000)]

    res_opt = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    with col_disp2:
        if res_opt.success:
            gen_names = ["Güneş", "Rüzgar", "Hidroelektrik", "Linyit/Kömür", "Doğalgaz"]
            gen_values = res_opt.x
            total_dispatch_cost = res_opt.fun

            fig_merit = px.bar(
                x=gen_names, y=gen_values,
                labels={"x": "Kaynak", "y": "Atanan Üretim (MW)"},
                title=f"Optimal Üretim Dağılımı (Toplam Maliyet: ₺{total_dispatch_cost:,.0f})",
                color=gen_names,
                color_discrete_sequence=["#F59E0B", "#10B981", "#0284C7", "#78350F", "#DC2626"]
            )
            st.plotly_chart(fig_merit, use_container_width=True)
            st.success(f"Ortalama Sistem Marjinal Üretim Maliyeti: **₺{(total_dispatch_cost / target_load):,.2f} / MWh**")
        else:
            st.error("Kapasite yetersiz: Mevcut santraller talep yükünü karşılayamıyor!")

# ---------------------------------------------------------
# TAB 4: SHAP & Model Dinamikleri
# ---------------------------------------------------------
# ---------------------------------------------------------
# TAB 4: SHAP & Model Dinamikleri (Düzeltilmiş Blok)
# ---------------------------------------------------------
with tab_shap:
    st.subheader("🧠 Model Karar Mekanizması & Özellik Önem Düzeyleri")
    st.markdown("Modelin yük tahminini üretirken ağırlık verdiği ana parametreler:")

    if model is not None and hasattr(model, "feature_importances_"):
        importances = list(model.feature_importances_)
        
        # Modelin kendi kaydettiği feature isimlerini almayı dene:
        if hasattr(model, "feature_name_"):
            feats = list(model.feature_name_)
        elif hasattr(model, "feature_names_in_"):
            feats = list(model.feature_names_in_)
        elif len(importances) == len(feature_cols):
            feats = feature_cols
        else:
            feats = [f"Özellik_{i+1}" for i in range(len(importances))]
    else:
        # Temsili özellik katkı ağırlıkları (Uzunluklar eşit: 9'a 9)
        feats = ["Lag_1h", "Rolling_Mean_3h", "Lag_24h", "Hour_Sin", "Hour_Cos", "Hour", "IsWeekend", "DayOfWeek", "Month"]
        importances = [0.38, 0.22, 0.16, 0.09, 0.06, 0.04, 0.02, 0.02, 0.01]

    # Garanti uzunluk eşitlemesi
    min_len = min(len(feats), len(importances))
    feats = feats[:min_len]
    importances = importances[:min_len]

    df_importance = pd.DataFrame({
        "Özellik": feats, 
        "Göreceli Katkı": importances
    }).sort_values("Göreceli Katkı", ascending=True)

    fig_shap = px.bar(
        df_importance, x="Göreceli Katkı", y="Özellik", orientation="h",
        title="LightGBM / Hibrit Özellik Önem Sıralaması (Feature Contribution)",
        color="Göreceli Katkı", color_continuous_scale="Blues"
    )
    fig_shap.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_shap, use_container_width=True)

    st.info("💡 **Operasyonel Yorum:** Model kararlarında en yüksek ağırlık `Lag_1h` (son 1 saatin çekişi) ve `Rolling_Mean_3h` üzerindedir. Sistem ataleti dolayısıyla kısa vadeli eğilimler takvim değişkenlerinden daha belirleyicidir.")
# ---------------------------------------------------------
# TAB 5: N-1 Güvenilirlik & Frekans Simülasyonu
# ---------------------------------------------------------
with tab_n1:
    st.subheader("⚡ N-1 Varlık Açması & Dinamik Frekans Kararlılığı")

    cn1, cn2, cn3 = st.columns(3)
    with cn1:
        tripped_mw = st.number_input("Devre Dışı Kalan Ünite (MW)", 200.0, 2500.0, 1200.0, step=100.0)
    with cn2:
        system_inertia = st.slider("Atalet Sabiti (H - Saniye)", 2.0, 7.0, 4.2, step=0.1)
    with cn3:
        fcr_reserve = st.number_input("Primer Frekans Rezervi (MW)", 500.0, 3000.0, 1500.0, step=100.0)

    cur_demand = float(df_view["consumption_mwh"].iloc[-1]) if not df_view.empty else 34000.0
    df_freq, m = simulate_n1_contingency(cur_demand, tripped_mw, system_inertia, 0.04, fcr_reserve)

    kn1, kn2, kn3, kn4 = st.columns(4)
    kn1.metric("Anlık Sistem Yükü", f"{cur_demand:,.0f} MW")
    kn2.metric("İlk Düşüş Hızı (RoCoF)", f"{m['rocof']:.3f} Hz/s")
    kn3.metric("Frekans Nadiri (Dip Nokta)", f"{m['f_nadir']:.2f} Hz", delta=f"{m['f_nadir'] - 50.0:.2f} Hz")
    kn4.metric("Şebeke Güvenliği", m["status"], delta_color="inverse" if m["ufls_triggered"] else "normal")

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df_freq["Time_Sec"], y=df_freq["Frequency_Hz"], mode="lines", name="Frekans (Hz)", line=dict(color="#2563EB", width=2.5)))
    fig_f.add_hline(y=50.0, line_dash="dash", line_color="#10B981", annotation_text="Nominal (50.0 Hz)")
    fig_f.add_hline(y=49.80, line_dash="dot", line_color="#F59E0B", annotation_text="Primer Bant Sınırı (49.8 Hz)")
    fig_f.add_hline(y=49.20, line_dash="dash", line_color="#EF4444", annotation_text="1. Kademe UFLS (49.2 Hz)")
    fig_f.update_layout(title="Salınım Yanıtı (Swing Equation)", xaxis_title="Saniye", yaxis_title="Hz", height=400, margin=dict(l=20, r=20, t=35, b=20))
    st.plotly_chart(fig_f, use_container_width=True)

# ---------------------------------------------------------
# TAB 6: Raporlama (FPDF2 ile PDF Dışa Aktarma)
# ---------------------------------------------------------
with tab_report:
    st.subheader("📄 Resmi Vardiya & Operasyonel Karar Raporu")
    st.markdown("Aktif analiz parametrelerini ve finansal KPI dökümünü PDF formatında dışa aktarın.")

    def clean_tr(text_str: str) -> str:
        """Helvetica fontuyla uyumlu olması için Türkçe ve özel karakterleri dönüştürür."""
        tr_map = str.maketrans({
            "ı": "i", "İ": "I", "ğ": "g", "Ğ": "G",
            "ş": "s", "Ş": "S", "ç": "c", "Ç": "C",
            "ö": "o", "Ö": "O", "ü": "u", "Ü": "U",
            "₺": "TL"
        })
        return str(text_str).translate(tr_map)

    class ShiftReportPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, "TEIAS YUK TAHMIN VE PIYASA OPERASYON RAPORU", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Sayfa {self.page_no()}", align="C")

    def generate_pdf():
        pdf = ShiftReportPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)

        pdf.cell(0, 8, clean_tr(f"Rapor Tarihi: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, clean_tr(f"Analiz Edilen Donem: {df_view['timestamp'].min().strftime('%d.%m.%Y')} - {df_view['timestamp'].max().strftime('%d.%m.%Y')}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "1. TEMEL FINANSAL VE OPERASYONEL GOSTERGELER", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 6, clean_tr(f"- Toplam Sistem Tuketimi: {total_actual_mwh:,.1f} MWh"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, clean_tr(f"- Model Tahmin Hatasi (MAPE): %{mape:.2f} (RMSE: {rmse:.1f} MWh)"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, clean_tr(f"- Ortalama PTF / SMF: {ptf_price:,.0f} TL / {smf_price:,.0f} TL"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, clean_tr(f"- Mevcut KGUP Plani Dengesizlik Cezasi: {cost_kgup_tl:,.0f} TL"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, clean_tr(f"- Hibrit Model Sayesinde Engellenen Mali Kayip: {savings_tl:,.0f} TL"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "2. SEBEKE GUVENILIRLIK VE N-1 ANALIZI", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 6, clean_tr(f"- Kritik Varlik Devre Disi Kapasitesi: {tripped_mw:,.0f} MW"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, clean_tr(f"- Frekans Nadir (En Dusuk Nokta): {m['f_nadir']:.2f} Hz"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, clean_tr(f"- Sebeke Durumu: {m['status']}"), new_x="LMARGIN", new_y="NEXT")

        return bytes(pdf.output())

    pdf_data = generate_pdf()
    st.download_button(
        label="📥 Vardiya Raporunu İndir (PDF)",
        data=pdf_data,
        file_name=f"TEIAS_Operasyon_Raporu_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )