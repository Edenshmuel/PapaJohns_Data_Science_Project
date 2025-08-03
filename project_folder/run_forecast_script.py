import pickle
import pandas as pd
from datetime import datetime
from forecast_utils import run_forecast

# === Date input and validation ===
start_input = input("Enter forecast start date (YYYY-MM-DD): ")
end_input   = input("Enter forecast end date   (YYYY-MM-DD): ")

start_date = pd.to_datetime(start_input)
end_date   = pd.to_datetime(end_input)
today = pd.to_datetime(datetime.today().date())

range_days = (end_date - start_date).days + 1

if start_date <= today:
    raise ValueError("⛔ Start date must be tomorrow or later.")
if end_date < start_date:
    raise ValueError("⛔ End date must be after the start date.")
if range_days < 7:
    raise ValueError("⛔ Forecast range must be at least 7 days.")
if range_days > 31:
    raise ValueError("⛔ Forecast range must be up to 31 days only.")

# === Load trained model ===
with open("deepar_model.pkl", "rb") as f:
    predictor = pickle.load(f)

# === Load preprocessed data ===
df = pd.read_csv("cleaned_data.csv")

# === Dynamic features list ===
dynamic_features_cols = [
    'Year', 'Month', 'Day', 'Is_Weekend',
    'is_christian_holiday', 'is_jewish_holiday',
    'is_near_jewish_holiday', 'is_day_before_new_year',
    'category_encoded', 'WeekOfYear', 'Season',
    'is_start_of_month', 'is_end_of_month',
    'Day_Name_sin', 'Day_Name_cos', 'Month_sin', 'Month_cos',
    'encoded_jewish_holiday', 'encoded_christian_holiday'
]

# === Run forecast ===
predictions_df = run_forecast(predictor, df, start_date, end_date, dynamic_features_cols)

# === Save output ===
predictions_df.to_csv("forecast_results.csv", index=False, encoding="utf-8-sig")
print("✅ Forecast completed and saved to forecast_results.csv")


