import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import our quantitative engine
from predictor_v2 import (
    get_stock_data,
    train_hybrid_model,
    predict_hybrid_future,
    search_ticker,
    get_qualitative_context,
    get_benchmark_ticker,
    get_benchmark_forecast
)

# Global in-memory cache to prevent multiple concurrent or sequential benchmark index fits
# that cause Render's 512MB RAM free tier instances to crash with OOM.
BENCHMARK_CACHE = {}

def fetch_cached_benchmark(benchmark_ticker: str, history_years: int) -> pd.DataFrame:
    if benchmark_ticker in BENCHMARK_CACHE:
        print(f"[CACHE] Returning cached benchmark forecast for {benchmark_ticker}")
        return BENCHMARK_CACHE[benchmark_ticker]
    
    print(f"[CACHE] Cache miss for {benchmark_ticker}. Training benchmark forecast...")
    forecast = get_benchmark_forecast(benchmark_ticker, history_years)
    if forecast is not None:
        BENCHMARK_CACHE[benchmark_ticker] = forecast
        print(f"[CACHE] Benchmark forecast for {benchmark_ticker} successfully cached.")
    return forecast

app = FastAPI(
    title="Hybrid Stock Price Predictor API",
    description="Institutional-grade time-series forecasting backend powered by Prophet and Gradient Boosting",
    version="2.0.0"
)

# Configure robust CORS middleware for React frontend (allow localhost and production wildcard/Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to production domain in a real environment
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    ticker: str
    target_date: str

@app.get("/api/search")
async def api_search(q: str):
    """Searches Yahoo Finance for matching stock symbols."""
    if not q:
        return []
    try:
        results = search_ticker(q)
        formatted = [{"label": label, "symbol": symbol} for label, symbol in results.items()]
        return formatted
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/context")
async def api_context(ticker: str):
    """Fetches news sentiment, profit margins, and revenue growth for the stock."""
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker parameter is required.")
    try:
        context = get_qualitative_context(ticker)
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def api_predict(request: PredictRequest):
    """Trains the Prophet + Gradient Boosting model and returns forecast and historical chart arrays."""
    ticker = request.ticker.strip().upper()
    target_date = request.target_date
    
    try:
        # 1. Fetch historical data (10 years)
        history_years = 10
        df = get_stock_data(ticker, history_years)
        
        # 2. Get benchmark mapping
        benchmark_ticker = get_benchmark_ticker(ticker)
        
        # 3. Train benchmark model first (using cached/dynamic scale 0.001)
        benchmark_forecast = fetch_cached_benchmark(benchmark_ticker, history_years)
        
        # 4. Train individual hybrid model (Moderately flexible 0.05 changepoint scale)
        changepoint_scale = 0.05
        model_dict = train_hybrid_model(ticker, df, changepoint_scale, 'additive', 0.1, 3, benchmark_forecast)
        
        # 5. Execute future predictions (applies the dynamic adaptive half-life decay)
        predicted_price, forecast = predict_hybrid_future(model_dict, df, target_date, benchmark_forecast)
        
        # 6. Apply qualitative sentiment adjustment factor smoothly (automatic override)
        qual_context = get_qualitative_context(ticker)
        adj = qual_context.get('adjustment_factor', 1.0)
        steps = len(forecast)
        if steps > 0:
            scaling_array = [1.0 + ((adj - 1.0) * (i / steps)) for i in range(steps)]
            forecast['hybrid_yhat'] = forecast['hybrid_yhat'] * scaling_array
            
        # 7. Apply Autoregressive Anchor Smoothing to prevent Day-1 jumps/cliffs
        last_row = df.iloc[-1]
        last_price = float(last_row['y'])
        last_date = last_row['ds'].strftime('%Y-%m-%d')
        
        first_pred = forecast['hybrid_yhat'].iloc[0]
        gap = last_price - first_pred
        decay_rate = 0.07 
        decay_array = np.exp(-decay_rate * np.arange(len(forecast)))
        forecast['hybrid_yhat'] = forecast['hybrid_yhat'] + (gap * decay_array)
        
        # Recalculate target date price after adjustments
        target_dt = pd.to_datetime(target_date).tz_localize(None).normalize()
        pred_row = forecast[forecast['ds'] == target_dt]
        if not pred_row.empty:
            predicted_price = float(pred_row['hybrid_yhat'].values[0])
        else:
            predicted_price = float(forecast.iloc[-1]['hybrid_yhat'])
            
        # Calculate moves
        delta = predicted_price - last_price
        pct_change = (delta / last_price) * 100
        
        # 8. Compile Chart Arrays (Cap historical data at max 5 years to date to prevent squishing)
        five_years_ago = pd.Timestamp.now().normalize() - pd.DateOffset(years=5)
        df_chart = df[df['ds'] >= five_years_ago].copy()
        
        chart_data = []
        for _, row in df_chart.iterrows():
            chart_data.append({
                "date": row['ds'].strftime('%Y-%m-%d'),
                "price": round(float(row['y']), 2),
                "type": "Historical"
            })
            
        # Filter future forecast only up to target_dt for plotting
        visual_forecast = forecast[forecast['ds'] <= target_dt]
        for _, row in visual_forecast.iterrows():
            if row['ds'] > df['ds'].max():
                chart_data.append({
                    "date": row['ds'].strftime('%Y-%m-%d'),
                    "price": round(float(row['hybrid_yhat']), 2),
                    "type": "Forecast"
                })
        
        # Explicitly run garbage collection to free model-fitting RAM overhead immediately
        import gc
        gc.collect()
        
        return {
            "ticker": ticker,
            "last_price": round(last_price, 2),
            "last_date": last_date,
            "predicted_price": round(predicted_price, 2),
            "predicted_date": target_date,
            "projected_move_val": round(delta, 2),
            "projected_move_pct": round(pct_change, 2),
            "qualitative_context": {
                "sentiment_score": round(qual_context['sentiment_score'], 2),
                "profit_margins": qual_context['profit_margins'],
                "revenue_growth": qual_context['revenue_growth'],
                "news_count": qual_context['news_count']
            },
            "chart_data": chart_data
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
