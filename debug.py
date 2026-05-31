import sys
from datetime import datetime, timedelta
from predictor import get_stock_data, train_hybrid_model, predict_hybrid_future

ticker = "RELIANCE.NS"
df = get_stock_data(ticker, 5)

print("Data range:", df['ds'].min(), "to", df['ds'].max())
print("Last 5 rows:\n", df.tail())

model_dict = train_hybrid_model(df, 0.05, "multiplicative", 0.05, 3)

target = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
pred, future = predict_hybrid_future(model_dict, df, target)

print(f"Predicted value for {target}: {pred}")
print("\nFuture predictions:\n", future[['ds', 'yhat', 'hybrid_yhat']].head(10))
