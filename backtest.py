import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import from our new predictor module
from predictor import get_stock_data, train_hybrid_model, predict_hybrid_future, add_technical_features

def backtest_hybrid(ticker, years_back=5, test_days=30):
    print(f"Starting hybrid backtest for {ticker}")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    # Download data
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df = df.reset_index()
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    elif 'index' in df.columns: 
        df = df.rename(columns={'index': 'ds', 'Close': 'y'})
    df = df[['ds', 'y']].dropna()
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None).dt.normalize()
    
    # Split train and test
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    
    print(f"Training on {len(train)} days, testing on {len(test)} days...")
    
    # Train Hybrid model
    model_dict = train_hybrid_model(
        train, 
        changepoint_scale=0.05, 
        seasonality_mode='multiplicative',
        xgb_lr=0.05,
        xgb_depth=3
    )
    
    # Predict (Predicting day by day or just taking the whole block)
    # The hybrid future predict returns a dataframe with all future dates predicted
    _, future_preds = predict_hybrid_future(model_dict, train, str(test['ds'].max().date()))
    
    # Evaluate
    # Merge on date to ensure alignment
    results = test.merge(future_preds[['ds', 'hybrid_yhat']], on='ds', how='left')
    predictions = results['hybrid_yhat'].values
    actuals = results['y'].values
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print("-" * 30)
    print("HYBRID BACKTEST RESULTS:")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("-" * 30)
    
    # Trend direction accuracy
    actual_direction = np.sign(actuals[-1] - actuals[0])
    predicted_direction = np.sign(predictions[-1] - predictions[0])
    print(f"Actual direction (start to end): {'Up' if actual_direction > 0 else 'Down'}")
    print(f"Predicted direction (start to end): {'Up' if predicted_direction > 0 else 'Down'}")
    print(f"Trend Direction Correct: {actual_direction == predicted_direction}")
    
    # Baseline: what if we just guessed the last known price for the whole period?
    baseline_predictions = np.repeat(train['y'].iloc[-1], test_days)
    baseline_mae = np.mean(np.abs(baseline_predictions - actuals))
    baseline_mape = np.mean(np.abs((actuals - baseline_predictions) / actuals)) * 100
    print(f"Baseline (Naive Forecast) MAE: ${baseline_mae:.2f}")
    print(f"Baseline MAPE: {baseline_mape:.2f}%")

if __name__ == "__main__":
    backtest_hybrid("RELIANCE.NS", years_back=5, test_days=30)
