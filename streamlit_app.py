import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

def perform_hybrid_forecast(ticker, target_date, history_years, changepoint_scale, seasonality_mode):
    """
    Fetches data, trains Prophet model, and returns the predicted price 
    for the specific target_date.
    
    Parameters:
    - history_years: Number of years of historical data to fetch.
    - changepoint_scale: Prophet parameter to control trend flexibility.
    - seasonality_mode: 'additive' or 'multiplicative'.
    """
    
    # --- FIX 1: Recency Bias ---
    # Dynamic lookback period instead of hardcoded 2 years.
    # Allows model to see full market cycles (bull and bear).
    days_back = history_years * 365
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    # Download data
    data = yf.download(ticker, start=start_date, progress=False)
    
    if data.empty:
        st.error(f"No data found for ticker {ticker}")
        return None, None, None

    # 2. Prepare data for Prophet
    df = data.reset_index()
    
    # Handle MultiIndex columns (common in new yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # Standardize column names
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    elif 'index' in df.columns: 
        df = df.rename(columns={'index': 'ds', 'Close': 'y'})
        
    # specific check for 'y' column existence
    if 'y' not in df.columns:
        # Try to find a column that looks like 'Close' or 'Adj Close'
        possible_cols = [c for c in df.columns if 'Close' in str(c)]
        if possible_cols:
            df = df.rename(columns={possible_cols[0]: 'y'})
        else:
            st.error("Could not determine price column from data.")
            return None, None, None

    df = df[['ds', 'y']]
    
    # Ensure ds is datetime and normalized to midnight
    # This prevents timestamp mismatches during lookup
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None).dt.normalize()

    # --- FIX 2 & 3: Configuration & Trend Bias ---
    # - daily_seasonality=False: Removes noise from looking for intra-day patterns in daily data.
    # - changepoint_prior_scale: Passed from UI to control overfitting (trend flexibility).
    # - seasonality_mode: Allows for multiplicative seasonality (volatility grows with price).
    m = Prophet(
        growth='linear', 
        daily_seasonality=False,  # Fixes Configuration Bias (fitting to noise)
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=changepoint_scale, # Fixes Trend-Following Bias (overfitting)
        seasonality_mode=seasonality_mode
    )
    m.fit(df)

    # 4. Create Future Dataframe
    last_date = df['ds'].max()
    target_dt = pd.to_datetime(target_date)
    
    days_to_predict = (target_dt - last_date).days
    
    if days_to_predict < 1:
        days_to_predict = 1

    future = m.make_future_dataframe(periods=days_to_predict)

    # 5. Predict
    fc = m.predict(future)

    # 6. Extract Specific Prediction
    prediction_row = fc[fc['ds'] == target_dt]
    
    if not prediction_row.empty:
        predicted_mean = prediction_row['yhat'].values[0]
    else:
        # Fallback to closest date (e.g. if target is a weekend)
        closest_idx = (fc['ds'] - target_dt).abs().idxmin()
        predicted_mean = fc.loc[closest_idx, 'yhat']

    return df, df.iloc[-1], predicted_mean

def main():
    st.title("📈 Robust Stock Price Predictor")
    st.markdown("This version includes fixes for common statistical biases in stock forecasting.")
    
    col_config, col_main = st.columns([1, 3])

    with col_config:
        st.header("Model Settings")
        ticker = st.text_input("Stock Ticker", value="AAPL")
        
        # --- Bias Mitigation Controls ---
        st.subheader("Bias Mitigation")
        
        # 1. Recency Bias Control
        history_years = st.slider(
            "Years of History (Lookback)", 
            min_value=1, 
            max_value=20, 
            value=5,
            help="Using more history (5+ years) helps reduce Recency Bias by including different market cycles."
        )
        
        # 2. Trend Flexibility (Overfitting) Control
        changepoint_scale = st.slider(
            "Trend Flexibility",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.001,
            format="%.3f",
            help="Lower values (e.g., 0.01) make the trend 'stiff' and less reactive to recent hype. Higher values allow the model to fit rapid changes but may overfit."
        )

        # 3. Seasonality Mode
        seasonality_mode = st.selectbox(
            "Seasonality Mode",
            options=["additive", "multiplicative"],
            index=1,
            help="Multiplicative is usually better for stocks: it assumes price swings get larger as the stock price gets higher."
        )
        
        default_date = datetime.now().date() + timedelta(days=1)
        target_date_input = st.date_input("Target Date", value=default_date)

        predict_btn = st.button("Predict Price", type="primary")

    with col_main:
        if predict_btn:
            with st.spinner(f"Training robust model for {ticker} using {history_years} years of data..."):
                full_df, last_row, predicted_price = perform_hybrid_forecast(
                    ticker, 
                    target_date_input, 
                    history_years, 
                    changepoint_scale,
                    seasonality_mode
                )
                
                if predicted_price is not None:
                    st.success("Analysis Complete")
                    
                    # Layout Metrics
                    m1, m2, m3 = st.columns(3)
                    
                    last_price = last_row['y']
                    last_date = last_row['ds'].strftime('%Y-%m-%d')
                    delta = predicted_price - last_price
                    pct_change = (delta / last_price) * 100
                    
                    m1.metric("Last Close", f"${last_price:.2f}", last_date)
                    m2.metric("Prediction", f"${predicted_price:.2f}", f"{target_date_input}")
                    m3.metric("Projected Move", f"{delta:+.2f}", f"{pct_change:+.2f}%")
                    
                    st.divider()
                    st.caption(f"Based on linear growth model trained on data from {full_df['ds'].min().date()} to {full_df['ds'].max().date()}.")
                    
                    # Optional: Show recent data tail to confirm download worked
                    with st.expander("View Recent Data"):
                        st.dataframe(full_df.tail())

if __name__ == "__main__":
    main()
