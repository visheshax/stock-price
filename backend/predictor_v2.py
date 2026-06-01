import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime, timedelta

# Institutional Constants
BENCHMARK_CHANGEPOINT_SCALE = 0.001  # Macro trajectory scale for benchmark indices

def get_stock_data(ticker: str, history_years: int):
    data = yf.download(ticker, period=f'{history_years}y', auto_adjust=True)
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

def get_benchmark_ticker(ticker: str) -> str:
    """Maps a stock ticker to its regional benchmark index."""
    if '.' in ticker:
        suffix = ticker.split('.')[-1].upper()
        mapping = {
            'NS': '^NSEI', 'BO': '^BSESN',
            'L': '^FTSE', 'IL': '^FTSE',
            'F': '^GDAXI', 'MU': '^GDAXI',
            'TO': '^GSPTSE',
            'AX': '^AXJO',
            'HK': '^HSI',
            'T': '^N225',
            'PA': '^FCHI',
            'MI': 'FTSEMIB.MI'
        }
        return mapping.get(suffix, '^GSPC')
    return '^GSPC'

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

def get_benchmark_forecast(benchmark_ticker: str, history_years: int, changepoint_scale: float = None, days_ahead: int = 7300) -> pd.DataFrame:
    """Trains a Prophet model on the benchmark index and returns its historical+future trend curve."""
    # Always use the institutional constant for benchmark scale, ignoring any passed-in value
    scale = BENCHMARK_CHANGEPOINT_SCALE
    try:
        df = get_stock_data(benchmark_ticker, history_years)
        
        cap_val = df['y'].max() * 5.0
        
        # Dynamic Macro Support Level (same logic as individual stocks)
        if len(df) > 252:
            support_level = df['y'].tail(252).min() * 0.85
        else:
            support_level = df['y'].min() * 0.85
        floor_val = max(support_level, df['y'].min() * 0.5)
        
        df['cap'] = cap_val
        df['floor'] = floor_val
        
        m = Prophet(growth='logistic', changepoint_prior_scale=scale, daily_seasonality=False, weekly_seasonality=False)
        m.fit(df[['ds', 'y', 'cap', 'floor']])
        
        future = m.make_future_dataframe(periods=days_ahead)
        future['cap'] = cap_val
        future['floor'] = floor_val
        forecast = m.predict(future)
        
        return forecast[['ds', 'yhat']].rename(columns={'yhat': 'benchmark_yhat'})
    except Exception:
        return None

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
    ema_down = ema_down.replace(0, 1e-10)  # Guard against division by zero
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Lag features
    df['MA20_lag1'] = df['MA20'].shift(1)
    df['MA50_lag1'] = df['MA50'].shift(1)
    df['RSI_lag1'] = df['RSI'].shift(1)
    
    return df

def train_hybrid_model(ticker: str, df: pd.DataFrame, changepoint_scale: float, seasonality_mode: str, xgb_lr: float, xgb_depth: int, benchmark_forecast: pd.DataFrame = None):
    """Trains a Prophet + Gradient Boosting hybrid model."""
    df = df.copy()
    
    # Fetch Market Cap and normalize to USD to avoid local currency bias
    try:
        info = yf.Ticker(ticker).info
        mc = info.get('marketCap', 0)
        currency = info.get('currency', 'USD')
        
        # Convert non-USD market caps to USD using live Forex rates
        if currency != 'USD' and mc > 0:
            fx_ticker = f"{currency}USD=X"
            fx_history = yf.Ticker(fx_ticker).history(period="1d")
            if not fx_history.empty:
                fx_rate = fx_history['Close'].iloc[-1]
                mc_usd = mc * fx_rate
            else:
                mc_usd = 0
        else:
            mc_usd = mc
    except:
        mc_usd = 0
        
    # Cap is set to 5.0x max to allow aggressive upside compounding without flattening the logistic curve too early
    cap_val = df['y'].max() * 5.0 
    
    # Dynamic Macro Support Level (User requested feature to prevent Black Swan death spirals)
    # Calculate the 1-year (252 trading days) historical low to act as a hard structural support floor
    if len(df) > 252:
        support_level = df['y'].tail(252).min() * 0.85 # Allow a maximum 15% black swan shock below the 1-year low
    else:
        support_level = df['y'].min() * 0.85
        
    # Floor logic: Merge dynamic support with mega-cap absolute floors
    if mc_usd > 2_000_000_000:
        floor_val = max(support_level, df['y'].min() * 0.5)
    else:
        floor_val = max(support_level, 0.01)
        
    df['cap'] = cap_val
    df['floor'] = floor_val

    # 1. Train Prophet with Logistic Growth
    use_yearly = len(df) >= 365 # Dynamic yearly seasonality
    m_prophet = Prophet(
        growth='logistic',
        daily_seasonality=False, 
        weekly_seasonality=False, # FIX: Stock markets are closed on weekends. Prophet freaks out on weekends if this is True.
        yearly_seasonality=use_yearly,
        changepoint_prior_scale=changepoint_scale,
        seasonality_mode=seasonality_mode
    )
    
    # De-Bias Calendar: Inject localized market holidays based on ticker suffix
    if '.' in ticker:
        suffix = ticker.split('.')[-1].upper()
        country_map = {
            'NS': 'IN', 'BO': 'IN', # India
            'L': 'GB', 'IL': 'GB', # UK
            'F': 'DE', 'MU': 'DE', # Germany
            'TO': 'CA', # Canada
            'AX': 'AU', # Australia
            'HK': 'HK', # Hong Kong
            'T': 'JP', # Japan
            'PA': 'FR', # France
            'MI': 'IT' # Italy
        }
        country_code = country_map.get(suffix)
        if country_code:
            m_prophet.add_country_holidays(country_name=country_code)
    else:
        # Standard US tickers have no suffix
        m_prophet.add_country_holidays(country_name='US')
        
    if benchmark_forecast is not None:
        df = pd.merge(df, benchmark_forecast, on='ds', how='left')
        df['benchmark_yhat'] = df['benchmark_yhat'].ffill().bfill()
        m_prophet.add_regressor('benchmark_yhat')
        
    fit_cols = ['ds', 'y', 'cap', 'floor']
    if benchmark_forecast is not None:
        fit_cols.append('benchmark_yhat')
        
    m_prophet.fit(df[fit_cols])
    
    # 2. Get Prophet predictions for the training set
    forecast = m_prophet.predict(df[fit_cols])
    
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
        'cap': cap_val,
        'floor': floor_val,
        'last_technical_features': {
            'MA20_lag1': df_tech['MA20'].iloc[-1],
            'MA50_lag1': df_tech['MA50'].iloc[-1],
            'RSI_lag1': df_tech['RSI'].iloc[-1]
        }
    }

def predict_hybrid_future(model_dict: dict, df: pd.DataFrame, target_date: str, benchmark_forecast: pd.DataFrame = None):
    """Predicts future prices using the trained hybrid model."""
    m_prophet = model_dict['prophet']
    m_xgb = model_dict['xgb']
    cap_val = model_dict.get('cap', 1000000.0)
    floor_val = model_dict.get('floor', 0.01)
    
    # 1. Generate future dates (Always generate at least 20 years for Goal Seek reverse forecasting)
    target_dt = pd.to_datetime(target_date).tz_localize(None).normalize()
    last_dt = df['ds'].max()
    days_to_target = (target_dt - last_dt).days
    
    # Force the engine to forecast 7300 days (20 years) forward regardless of the visual target date
    days_ahead = max(days_to_target, 7300)
    
    if days_ahead <= 0:
        days_ahead = 7300
        
    future_dates = m_prophet.make_future_dataframe(periods=days_ahead)
    future_dates['cap'] = cap_val
    future_dates['floor'] = floor_val
    
    if benchmark_forecast is not None:
        future_dates = pd.merge(future_dates, benchmark_forecast, on='ds', how='left')
        future_dates['benchmark_yhat'] = future_dates['benchmark_yhat'].ffill().bfill()
        
    # 2. Get Prophet future predictions
    forecast = m_prophet.predict(future_dates)
    
    future_dates = forecast[forecast['ds'] > last_dt].copy()
    tech_features = model_dict['last_technical_features']
    
    future_dates['MA20_lag1'] = tech_features['MA20_lag1']
    future_dates['MA50_lag1'] = tech_features['MA50_lag1']
    future_dates['RSI_lag1'] = tech_features['RSI_lag1']
    
    use_yearly = model_dict.get('use_yearly', True)
    features_cols = ['yhat', 'trend', 'yearly', 'MA20_lag1', 'MA50_lag1', 'RSI_lag1'] if use_yearly else ['yhat', 'trend', 'MA20_lag1', 'MA50_lag1', 'RSI_lag1']
    X_future = future_dates[features_cols]
    
    residual_preds = m_xgb.predict(X_future)
    
    # Adaptive Price-to-Trend Divergence Decay:
    # If the stock price has heavily diverged from the macro Prophet trend (e.g. ACN dropping, Broadcom skyrocketing),
    # a rapid 60-day decay causes extreme short-term rebound/crash projections.
    # Symmetrically scale the half-life from 60 days (low divergence) up to 250 trading days (1 year, high divergence).
    last_price = df['y'].iloc[-1]
    first_yhat = future_dates['yhat'].iloc[0]
    divergence = abs(last_price - first_yhat) / first_yhat
    
    if divergence <= 0.10:
        half_life = 60.0
    elif divergence >= 0.20:
        half_life = 250.0
    else:
        # Linear interpolation between 60 and 250 days
        half_life = 60.0 + (250.0 - 60.0) * ((divergence - 0.10) / 0.10)
        
    decay_lambda = np.log(2) / half_life
    decay_weights = np.exp(-decay_lambda * np.arange(len(residual_preds)))
    residual_preds = residual_preds * decay_weights
    
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
