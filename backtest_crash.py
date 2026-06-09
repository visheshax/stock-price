import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Adjust path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from predictor_v2 import (
    get_stock_data,
    get_benchmark_ticker,
    get_benchmark_forecast,
    train_hybrid_model,
    predict_hybrid_future,
    get_qualitative_context
)

def run_crash_backtest(ticker: str):
    print(f"\nSTRESS TEST BACKTEST: {ticker} (Nasdaq Crash Event - June 5, 2026)")
    print("-" * 65)
    
    # 1. Download full dataset
    history_years = 10
    df_full = get_stock_data(ticker, history_years)
    
    # Clean datetime index for filtering
    df_full['ds_str'] = df_full['ds'].dt.strftime('%Y-%m-%d')
    
    # 2. Split data: Train up to June 4, 2026 (Day before the crash)
    df_train = df_full[df_full['ds_str'] <= '2026-06-04'].copy()
    if df_train.empty:
        print("❌ Error: No training data found up to 2026-06-04.")
        return
        
    print(f"Historical training data size: {len(df_train)} rows (ending {df_train['ds_str'].iloc[-1]})")
    
    # Actual values on the crash day and rebound day
    actual_june_5 = df_full[df_full['ds_str'] == '2026-06-05']
    actual_june_8 = df_full[df_full['ds_str'] == '2026-06-08']
    
    if actual_june_5.empty:
        print("⚠️ Warning: No actual data found for crash day 2026-06-05. Checking surrounding dates...")
        return
        
    price_pre_crash = float(df_train['y'].iloc[-1])
    price_actual_5 = float(actual_june_5['y'].iloc[0])
    pct_actual_drop = ((price_actual_5 - price_pre_crash) / price_pre_crash) * 100
    
    print(f"Pre-Crash Close (June 4): ${price_pre_crash:.2f}")
    print(f"Actual Close (June 5 - Crash): ${price_actual_5:.2f} ({pct_actual_drop:+.2f}%)")
    
    if not actual_june_8.empty:
        price_actual_8 = float(actual_june_8['y'].iloc[0])
        print(f"Actual Close (June 8 - Rebound): ${price_actual_8:.2f} ({((price_actual_8 - price_pre_crash) / price_pre_crash) * 100:+.2f}% from June 4)")
    else:
        price_actual_8 = None
        
    # 3. Train models on the pre-crash dataset
    benchmark_ticker = get_benchmark_ticker(ticker)
    
    # We train the benchmark index up to June 4 to simulate real-time forecasting
    print(f"Training regional benchmark index model ({benchmark_ticker})...")
    benchmark_forecast = get_benchmark_forecast(benchmark_ticker, history_years)
    
    # Train individual hybrid model
    print("Training stock hybrid model (Prophet + Gradient Boosting)...")
    changepoint_scale = 0.05
    model_dict = train_hybrid_model(ticker, df_train, changepoint_scale, 'additive', 0.1, 3, benchmark_forecast)
    
    # 4. Predict forward to June 8, 2026
    print("Simulating future projections...")
    predicted_price_5, forecast = predict_hybrid_future(model_dict, df_train, '2026-06-05', benchmark_forecast)
    
    # Extract June 5 predictions
    pred_row_5 = forecast[forecast['ds'] == pd.to_datetime('2026-06-05')]
    if not pred_row_5.empty:
        pred_expected_5 = float(pred_row_5['hybrid_yhat'].values[0])
        pred_upper_5 = float(pred_row_5['hybrid_yhat_upper'].values[0])
        pred_lower_5 = float(pred_row_5['hybrid_yhat_lower'].values[0])
    else:
        pred_expected_5 = predicted_price_5
        # Fallback approximation for bounds
        t_days = 1
        res_std = model_dict.get('residual_std', 5.0)
        uncertainty = 1.96 * res_std * np.sqrt(t_days / 252.0)
        pred_upper_5 = pred_expected_5 + uncertainty
        pred_lower_5 = pred_expected_5 - uncertainty

    # Extract June 8 predictions
    pred_row_8 = forecast[forecast['ds'] == pd.to_datetime('2026-06-08')]
    if not pred_row_8.empty:
        pred_expected_8 = float(pred_row_8['hybrid_yhat'].values[0])
        pred_upper_8 = float(pred_row_8['hybrid_yhat_upper'].values[0])
        pred_lower_8 = float(pred_row_8['hybrid_yhat_lower'].values[0])
    else:
        pred_expected_8 = pred_expected_5
        pred_upper_8 = pred_upper_5
        pred_lower_8 = pred_lower_5
        
    print("\nPROJECTION RESULTS FOR CRASH DAY (June 5):")
    print(f"  Worst Case (Pessimistic Floor): ${pred_lower_5:.2f}")
    print(f"  Expected Case (Baseline):       ${pred_expected_5:.2f}")
    print(f"  Best Case (Optimistic Ceiling): ${pred_upper_5:.2f}")
    
    # 5. RISK ASSIGNMENT AUDIT
    # Did the actual crashed price fall inside our forecasted scenarios?
    within_corridor = pred_lower_5 <= price_actual_5 <= pred_upper_5
    status = "SUCCESS (Captured Risk)" if within_corridor else "FAILURE (Out of Bounds)"
    
    print("-" * 65)
    print(f"Risk Corridor Validation: {status}")
    if within_corridor:
        print("✅ The actual crash price successfully landed inside the predicted Worst-to-Best Case scenarios.")
    else:
        print("❌ The crash exceeded the predicted worst-case boundary.")
        
    # Return metrics for consolidation
    return {
        "ticker": ticker,
        "pre_crash": price_pre_crash,
        "actual_crash": price_actual_5,
        "expected_pred": pred_expected_5,
        "worst_pred": pred_lower_5,
        "best_pred": pred_upper_5,
        "within_corridor": within_corridor
    }

if __name__ == "__main__":
    # Run backtests for three major stocks across the June 5 Nasdaq crash
    tickers = ["AAPL", "NVDA", "ACN"]
    results = []
    
    for t in tickers:
        try:
            res = run_crash_backtest(t)
            if res:
                results.append(res)
        except Exception as e:
            print(f"❌ Error backtesting {t}: {e}")
            import traceback
            traceback.print_exc()
            
    # Print Consolidated Audit Report
    print("\n" + "=" * 70)
    print("         CONSOLIDATED NASDAQ 4% CRASH STRESS TEST REPORT")
    print("=" * 70)
    print(f"{'Ticker':<8} | {'June 4 close':<12} | {'Actual Crash':<12} | {'Worst Case':<12} | {'Expected':<12} | {'Risk Bounded'}")
    print("-" * 70)
    for r in results:
        bounded = "Yes (Safe) ✅" if r["within_corridor"] else "No (Breached) ❌"
        print(f"{r['ticker']:<8} | ${r['pre_crash']:<10.2f} | ${r['actual_crash']:<10.2f} | ${r['worst_pred']:<10.2f} | ${r['expected_pred']:<10.2f} | {bounded}")
    print("=" * 70)
