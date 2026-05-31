import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime, timedelta

def get_stock_data(ticker: str, history_years: int):
    days_back = history_years * 365
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    data = yf.download(ticker, start=start_date, progress=False)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
        
    df = data.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    elif 'index' in df.columns: 
        df = df.rename(columns={'index': 'ds', 'Close': 'y'})
        
    if 'y' not in df.columns:
        possible_cols = [c for c in df.columns if 'Close' in str(c)]
        if possible_cols:
            df = df.rename(columns={possible_cols[0]: 'y'})
        else:
            raise ValueError("Could not determine price column from data.")

    df = df[['ds', 'y']].dropna()
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None).dt.normalize()
    return df

import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def search_ticker(query: str, max_results: int = 5):
    """Searches Yahoo Finance for a company name and returns a dict mapping display names to symbols."""
    try:
        search_results = yf.Search(query, max_results=max_results).quotes
        options = {}
        for q in search_results:
            if 'symbol' in q and 'shortname' in q:
                exch = q.get('exchange', 'Unknown')
                label = f"{q['shortname']} ({q['symbol']}) - {exch}"
                options[label] = q['symbol']
        return options
    except Exception:
        return {}

def get_qualitative_context(ticker: str):
    """Fetches news sentiment and fundamental data to calculate a qualitative adjustment factor."""
    context = {
        'sentiment_score': 0.0,
        'profit_margins': 'N/A',
        'revenue_growth': 'N/A',
        'adjustment_factor': 1.0,
        'news_count': 0
    }
    
    try:
        t = yf.Ticker(ticker)
        
        # 1. Fetch Fundamentals
        info = t.info
        if info:
            if 'profitMargins' in info and info['profitMargins']:
                context['profit_margins'] = f"{info['profitMargins'] * 100:.1f}%"
            if 'revenueGrowth' in info and info['revenueGrowth']:
                context['revenue_growth'] = f"{info['revenueGrowth'] * 100:.1f}%"
                
        # 2. Fetch News and Calculate Sentiment
        news = t.news
        if news:
            sia = SentimentIntensityAnalyzer()
            scores = []
            for article in news:
                title = article.get('title', '')
                if title:
                    scores.append(sia.polarity_scores(title)['compound'])
            
            if scores:
                avg_sentiment = sum(scores) / len(scores)
                context['sentiment_score'] = avg_sentiment
                context['news_count'] = len(scores)
                
                # Calculate automatic adjustment factor (Max +/- 15% based on sentiment)
                # If average sentiment is -1.0, adjustment is 0.85. If +1.0, adjustment is 1.15.
                context['adjustment_factor'] = 1.0 + (avg_sentiment * 0.15)
                
    except Exception as e:
        print(f"Error fetching qualitative data: {e}")
        
    return context

def add_technical_features(df: pd.DataFrame):
    """Calculates technical indicators manually to avoid dependencies."""
    df = df.copy()
    
    # Moving Averages
    df['MA20'] = df['y'].rolling(window=20).mean()
    df['MA50'] = df['y'].rolling(window=50).mean()
    
    # RSI
    delta = df['y'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Lag features
    df['MA20_lag1'] = df['MA20'].shift(1)
    df['MA50_lag1'] = df['MA50'].shift(1)
    df['RSI_lag1'] = df['RSI'].shift(1)
    
    return df

def train_hybrid_model(df: pd.DataFrame, changepoint_scale: float, seasonality_mode: str, xgb_lr: float, xgb_depth: int):
    """Trains a Prophet + Gradient Boosting hybrid model."""
    # 1. Train Prophet
    use_yearly = len(df) >= 365 # Dynamic yearly seasonality
    m_prophet = Prophet(
        growth='linear', 
        daily_seasonality=False, 
        weekly_seasonality=False, # FIX: Stock markets are closed on weekends. Prophet freaks out on weekends if this is True.
        yearly_seasonality=use_yearly,
        changepoint_prior_scale=changepoint_scale,
        seasonality_mode=seasonality_mode
    )
    m_prophet.fit(df[['ds', 'y']])
    
    # 2. Get Prophet predictions for the training set
    forecast = m_prophet.predict(df[['ds']])
    
    # 3. Add Technical Features
    df_tech = add_technical_features(df)
    
    # 4. Merge Prophet output with Technical Features
    hybrid_df = df_tech.merge(forecast[['ds', 'yhat', 'trend', 'yearly'] if use_yearly else ['ds', 'yhat', 'trend']], on='ds')
    hybrid_df = hybrid_df.dropna() # Drop rows where MAs/lags are NaN
    
    if len(hybrid_df) < 10:
        raise ValueError("Not enough historical data to compute technical indicators. Please select a longer history or an older ticker.")
    
    # 5. Train Gradient Boosting on Residuals
    features = ['yhat', 'trend', 'yearly', 'MA20_lag1', 'MA50_lag1', 'RSI_lag1'] if use_yearly else ['yhat', 'trend', 'MA20_lag1', 'MA50_lag1', 'RSI_lag1']
    X = hybrid_df[features]
    y_residual = hybrid_df['y'] - hybrid_df['yhat']
    
    m_xgb = HistGradientBoostingRegressor(
        learning_rate=xgb_lr,
        max_depth=xgb_depth,
        max_iter=100,
        random_state=42
    )
    m_xgb.fit(X, y_residual)
    
    return {
        'prophet': m_prophet,
        'xgb': m_xgb,
        'use_yearly': use_yearly,
        'last_technical_features': {
            'MA20_lag1': df_tech['MA20'].iloc[-1],
            'MA50_lag1': df_tech['MA50'].iloc[-1],
            'RSI_lag1': df_tech['RSI'].iloc[-1]
        }
    }

def predict_hybrid_future(model_dict: dict, df: pd.DataFrame, target_date: str):
    """Predicts future price using the hybrid model."""
    m_prophet = model_dict['prophet']
    m_xgb = model_dict['xgb']
    tech_features = model_dict['last_technical_features']
    
    last_date = df['ds'].max()
    target_dt = pd.to_datetime(target_date)
    
    days_to_predict = (target_dt - last_date).days
    if days_to_predict < 1:
        days_to_predict = 1

    # Prophet forecast
    future = m_prophet.make_future_dataframe(periods=days_to_predict)
    prophet_fcst = m_prophet.predict(future)
    
    future_dates = prophet_fcst[prophet_fcst['ds'] > last_date].copy()
    
    future_dates['MA20_lag1'] = tech_features['MA20_lag1']
    future_dates['MA50_lag1'] = tech_features['MA50_lag1']
    future_dates['RSI_lag1'] = tech_features['RSI_lag1']
    
    use_yearly = model_dict.get('use_yearly', True)
    features_cols = ['yhat', 'trend', 'yearly', 'MA20_lag1', 'MA50_lag1', 'RSI_lag1'] if use_yearly else ['yhat', 'trend', 'MA20_lag1', 'MA50_lag1', 'RSI_lag1']
    X_future = future_dates[features_cols]
    
    residual_preds = m_xgb.predict(X_future)
    future_dates['hybrid_yhat'] = future_dates['yhat'] + residual_preds
    
    # Financial Sanity Check: Stock prices cannot be negative
    future_dates['hybrid_yhat'] = future_dates['hybrid_yhat'].clip(lower=0.01)
    
    prediction_row = future_dates[future_dates['ds'] == target_dt]
    if not prediction_row.empty:
        predicted_mean = prediction_row['hybrid_yhat'].values[0]
    else:
        closest_idx = (future_dates['ds'] - target_dt).abs().idxmin()
        predicted_mean = future_dates.loc[closest_idx, 'hybrid_yhat']
        
    return predicted_mean, future_dates
