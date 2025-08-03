# app.py  –  Streamlit interface for forecasting
import streamlit as st
import pickle, pandas as pd
from datetime import date, timedelta
from preprocessing import prepare_data
from forecast_utils import run_forecast

# ---------- 1. Load trained model once (cached) ----------
@st.cache_resource
def load_model():
    with open("deepar_model.pkl", "rb") as f:
        return pickle.load(f)

predictor = load_model()

# ---------- 2. load category-model / mapping ----------
with open("product_category_model.pkl", "rb") as f:
    category_model = pickle.load(f)

with open("index_to_category.pkl", "rb") as f:
    index_to_category = pickle.load(f)

# ---------- 3. App UI ----------
st.set_page_config(page_title="Demand Forecast", layout="centered")
st.title("📈 Demand Forecast • Papa Johns")

file = st.file_uploader("Upload raw sales file (CSV / XLSX)", type=["csv", "xlsx"])

col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Forecast start date",
                          date.today() + timedelta(days=1))
with col2:
    end = st.date_input("Forecast end date",
                        start + timedelta(days=13))

# ---------- 4. Dynamic-feature columns used by the model ----------
DYNAMIC_COLS = [
    "Year", "Month", "Day", "Is_Weekend",
    "is_christian_holiday", "is_jewish_holiday",
    "is_near_jewish_holiday", "is_day_before_new_year",
    "category_encoded", "WeekOfYear", "Season",
    "is_start_of_month", "is_end_of_month",
    "Day_Name_sin", "Day_Name_cos", "Month_sin", "Month_cos",
    "encoded_jewish_holiday", "encoded_christian_holiday"
]

# ---------- 5. Run button ----------
if st.button("🚀 Run forecast") and file:

    # --- range validation ---
    days = (end - start).days + 1
    if start <= date.today():
        st.error("Start date must be tomorrow or later.")
    elif days < 7 or days > 31:
        st.error("Forecast range must be 7-31 days.")
    else:
        # --- read file ---
        if file.name.lower().endswith(".xlsx"):
            df_raw = pd.read_excel(file)
        else:
            df_raw = pd.read_csv(file)

        # --- preprocessing ---
        st.info("⏳ Cleaning & feature-engineering …")
        df_clean = prepare_data(
            df_raw,
            category_model,
            index_to_category=index_to_category
        )

        # --- forecasting ---
        st.info("⏳ Running DeepAR …")
        preds = run_forecast(
            predictor,
            df_clean,
            pd.Timestamp(start),
            pd.Timestamp(end),
            dynamic_features_cols=DYNAMIC_COLS,
        )

        # --- results ---
        st.success("✅ Forecast ready!")
        st.dataframe(preds.head(20))

        csv_bytes = preds.to_csv(index=False, encoding="utf-8-sig").encode()
        st.download_button(
            "⬇️ Download CSV",
            csv_bytes,
            "forecast.csv",
            mime="text/csv",
        )



