import os
import pandas as pd

def _load_and_prep(filepath, start_date=None, end_date=None):
    extension = os.path.splitext(filepath)[1].lower()
    data = pd.read_csv(filepath) if extension == ".csv" else pd.read_excel(filepath)

    required_columns = {"Date", "Close"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}.")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"]).sort_values("Date").copy()

    if len(data) < 2:
        raise ValueError("At least two valid rows with Date and Close values are required.")

    # Apply date range filtering if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        data = data[data["Date"] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        data = data[data["Date"] <= end_date]

    if len(data) < 2:
        raise ValueError("Not enough data points in the selected date range.")

    data["Days"] = (data["Date"] - data["Date"].min()).dt.days
    
    # Calculate technical indicators
    data["MA_20"] = data["Close"].rolling(window=20).mean()
    data["MA_50"] = data["Close"].rolling(window=50).mean()
    data["MA_200"] = data["Close"].rolling(window=200).mean()
    
    return data

def _calculate_volatility_and_changes(data):
    """Calculate percentage changes."""
    metrics = {}
    
    # Daily percentage changes
    data["Daily_Change_%"] = data["Close"].pct_change() * 100
    metrics["avg_daily_change"] = round(data["Daily_Change_%"].mean(), 4)
    
    # Weekly and monthly percentage changes
    if len(data) >= 7:
        first_price = data["Close"].iloc[0]
        prices_7d_ago = data["Close"].iloc[-7] if len(data) >= 7 else first_price
        metrics["7day_change_%"] = round(((data["Close"].iloc[-1] - prices_7d_ago) / prices_7d_ago) * 100, 4)
    else:
        metrics["7day_change_%"] = 0.0
    
    if len(data) >= 30:
        prices_30d_ago = data["Close"].iloc[-30] if len(data) >= 30 else data["Close"].iloc[0]
        metrics["30day_change_%"] = round(((data["Close"].iloc[-1] - prices_30d_ago) / prices_30d_ago) * 100, 4)
    else:
        metrics["30day_change_%"] = 0.0
    
    overall_change = ((data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0]) * 100
    metrics["overall_change_%"] = round(overall_change, 4)
    
    return metrics
