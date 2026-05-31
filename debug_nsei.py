import sys
from datetime import datetime, timedelta
from predictor import get_stock_data, train_hybrid_model, predict_hybrid_future

ticker = "^NSEI"
df = get_stock_data(ticker, 10)

print("Data range:", df['ds'].min(), "to", df['ds'].max())
print("Last 5 rows:\n", df.tail())

model_dict = train_hybrid_model(df, 0.05, "multiplicative", 0.05, 3)

target = "2026-09-30"
pred, future = predict_hybrid_future(model_dict, df, target)

print(f"Predicted value for {target}: {pred}")
print("\nFuture predictions:\n", future[['ds', 'yhat', 'hybrid_yhat']].tail(10))

# Print prophet predictions and features
X_future = future[['yhat', 'trend', 'yearly', 'MA20_lag1', 'MA50_lag1', 'RSI_lag1']]
print("Features for last 5 future dates:\n", X_future.tail(5))

