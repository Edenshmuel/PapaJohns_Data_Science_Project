import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset


def run_forecast(predictor, df, start_date, end_date, dynamic_features_cols,
                 item_col="Clean_Desc_Encoded", target_col="Quantity", freq="D"):
    """
    Run DeepAR forecasts for a given date range for each item.

    Parameters:
    ----------
    predictor : object
        Trained DeepAR model loaded from a .pkl file.
    df : DataFrame
        Preprocessed dataset ready for forecasting (output from prepare_data()).
    start_date : datetime
        First date of the forecast range.
    end_date : datetime
        Last date of the forecast range.
    dynamic_features_cols : list
        List of column names for dynamic real-valued features.
    item_col : str, default="Clean_Desc_Encoded"
        Column that identifies the time series (item).
    target_col : str, default="Quantity"
        Column with the target historical values.
    freq : str, default="D"
        Frequency of the time series ("D" = daily).

    Returns:
    -------
    DataFrame with forecast results for all items.
    """

    # Calculate forecast length
    prediction_length = (end_date - start_date).days + 1
    series_list = []

    # Build time series input for each item
    for item_id in df[item_col].unique():
        df_item = df[df[item_col] == item_id].copy()

        # 1) Historical data only (before the forecast start date)
        df_hist = df_item[df_item["Date"] < start_date].sort_values("Date")
        if df_hist.empty:
            continue

        # 2) Full date range (past + forecast period) to prepare dynamic features
        full_dates = pd.date_range(
            start=df_hist["Date"].iloc[0],
            end=end_date,
            freq=freq
        )
        df_full = pd.DataFrame({"Date": full_dates})
        df_full[item_col] = item_id

        # Merge item-specific features into the full date range
        cols_to_merge = [c for c in df_item.columns if c != target_col]
        df_full = (
            df_full
            .merge(df_item[cols_to_merge], on=["Date", item_col], how="left")
            .sort_values("Date")
        )

        # Extract target history values
        target_hist = df_hist[target_col].astype(float).values

        # Fill missing values for dynamic features
        for col in dynamic_features_cols:
            if col not in df_full.columns:
                df_full[col] = 0
            df_full[col] = (
                df_full[col]
                .fillna(method="ffill")
                .fillna(method="bfill")
                .fillna(0)
            )

        # Transpose features for GluonTS format
        feat_dynamic_real = df_full[dynamic_features_cols].T.values.tolist()

        # Add time series record
        series_list.append(
            {
                "start": df_hist["Date"].iloc[0],
                "target": target_hist,
                "feat_dynamic_real": feat_dynamic_real,
                "item_id": str(item_id),
            }
        )

    # Create GluonTS dataset
    test_ds = ListDataset(series_list, freq=freq)
    forecasts = {}

    # Generate forecasts for each item
    for entry, forecast in zip(test_ds, predictor.predict(test_ds)):
        cat_id = entry["item_id"]
        forecast_index = pd.date_range(start=start_date, periods=prediction_length, freq=freq)

        # Store forecast results
        df_forecast = pd.DataFrame({
            "date": forecast_index,
            "predicted_quantity": forecast.quantile(0.9)  # Use 90th percentile forecast
        })
        df_forecast["predicted_quantity_ceiled"] = np.round(df_forecast["predicted_quantity"])
        forecasts[cat_id] = df_forecast

    # Combine all item forecasts into one DataFrame
    all_forecasts_list = []
    for cat_id, df_cat in forecasts.items():
        df_cat[item_col.lower()] = int(cat_id)
        all_forecasts_list.append(df_cat)

    return pd.concat(all_forecasts_list, ignore_index=True)

