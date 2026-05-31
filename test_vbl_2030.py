import yfinance as yf
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from predictor import get_stock_data, train_hybrid_model, predict_hybrid_future, get_qualitative_context, get_benchmark_ticker, get_benchmark_forecast

ticker = "VBL.NS"
target_date_str = "2030-06-30"
print(f"Testing {ticker} for {target_date_str} using Growth (Momentum)...")

# 1. Fetch data
df = get_stock_data(ticker, 10)
last_price = df['y'].iloc[-1]
print(f"Current Last Close: ${last_price:.2f}")

# 1.5 Fetch Benchmark
benchmark_ticker = get_benchmark_ticker(ticker)
print(f"Benchmark identified: {benchmark_ticker}")
benchmark_forecast = get_benchmark_forecast(benchmark_ticker, 10, 0.050)

# 2. Train Model (0.500 for Extreme Growth/Momentum)
model_dict = train_hybrid_model(ticker, df, 0.050, 'additive', 0.1, 3, benchmark_forecast)

# 3. Predict
predicted_price, forecast = predict_hybrid_future(model_dict, df, target_date_str, benchmark_forecast)

# 3.5 Gap Smoothing (Replicating UI logic)
first_pred = forecast['hybrid_yhat'].iloc[0]
gap = last_price - first_pred
decay_rate = 0.07 
decay_array = np.exp(-decay_rate * np.arange(len(forecast)))
forecast['hybrid_yhat'] = forecast['hybrid_yhat'] + (gap * decay_array)

target_dt = pd.to_datetime(target_date_str).tz_localize(None).normalize()
pred_row = forecast[forecast['ds'] == target_dt]
if not pred_row.empty:
    predicted_price = pred_row['hybrid_yhat'].values[0]

print(f"Mathematical Forecast for {target_date_str}: ${predicted_price:.2f}")

# 4. Context Adj
qual = get_qualitative_context(ticker)
adj = qual.get('adjustment_factor', 1.0)
adj_predicted = predicted_price * adj

print(f"AI Qualitative Adjusted Forecast: ${adj_predicted:.2f}")
print("Done!")
