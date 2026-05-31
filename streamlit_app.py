import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# Import our decoupled hybrid logic
from predictor import get_stock_data, train_hybrid_model, predict_hybrid_future

st.set_page_config(page_title="Hybrid Stock Price Predictor", layout="wide")

# Use Streamlit's cache_data for data downloading
@st.cache_data(show_spinner="Fetching historical data...")
def load_data(ticker, history_years):
    return get_stock_data(ticker, history_years)

# Use cache_resource for the model since it's an object we shouldn't repeatedly train 
# if the underlying data and parameters haven't changed.
@st.cache_resource(show_spinner="Training Hybrid Prophet + XGBoost model...")
def train_model(_df, changepoint_scale, seasonality_mode, xgb_lr, xgb_depth):
    return train_hybrid_model(_df, changepoint_scale, seasonality_mode, xgb_lr, xgb_depth)

def main():
    st.title("📈 Hybrid Stock Price Predictor (Prophet + XGBoost)")
    st.markdown("Forecasting stock prices using Facebook Prophet for macro-trends and XGBoost for micro-volatility based on technical indicators.")
    
    col_config, col_main = st.columns([1, 3])

    with col_config:
        st.header("Model Settings")
        ticker = st.text_input("Stock Ticker", value="AAPL")
        
        st.subheader("General Settings")
        history_years = st.slider(
            "Years of History (Lookback)", min_value=1, max_value=20, value=5,
            help="More history helps Prophet understand long-term cycles."
        )
        
        st.subheader("Prophet Hyperparameters")
        changepoint_scale = st.slider(
            "Trend Flexibility", min_value=0.001, max_value=0.5, value=0.05, step=0.001, format="%.3f"
        )
        seasonality_mode = st.selectbox(
            "Seasonality Mode", options=["additive", "multiplicative"], index=1
        )
        
        st.subheader("XGBoost Hyperparameters")
        xgb_lr = st.slider(
            "Learning Rate", min_value=0.01, max_value=0.5, value=0.05, step=0.01,
            help="Lower values make the model more robust but slower to learn."
        )
        xgb_depth = st.slider(
            "Max Depth", min_value=1, max_value=10, value=3,
            help="Maximum depth of the trees. High values can lead to overfitting."
        )
        
        default_date = datetime.now().date() + timedelta(days=1)
        target_date_input = st.date_input("Target Date", value=default_date)

        predict_btn = st.button("Predict Price", type="primary")

    with col_main:
        if predict_btn:
            try:
                # 1. Fetch data
                df = load_data(ticker, history_years)
                
                # 2. Train hybrid model
                model_dict = train_model(df, changepoint_scale, seasonality_mode, xgb_lr, xgb_depth)
                
                # 3. Predict
                predicted_price, forecast = predict_hybrid_future(model_dict, df, str(target_date_input))
                
                st.success("Analysis Complete")
                
                # Layout Metrics
                last_row = df.iloc[-1]
                last_price = last_row['y']
                last_date = last_row['ds'].strftime('%Y-%m-%d')
                delta = predicted_price - last_price
                pct_change = (delta / last_price) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Last Close", f"${last_price:.2f}", last_date)
                m2.metric("Prediction", f"${predicted_price:.2f}", f"{target_date_input}")
                m3.metric("Projected Move", f"{delta:+.2f}", f"{pct_change:+.2f}%")
                st.divider()

                # Visualizations
                st.subheader("Forecast Overview")
                
                # Prepare data for chart
                hist_chart_data = df.copy()
                hist_chart_data['Type'] = 'Historical'
                
                future_chart_data = forecast[['ds', 'hybrid_yhat']].copy()
                future_chart_data = future_chart_data.rename(columns={'hybrid_yhat': 'y'})
                future_chart_data = future_chart_data[future_chart_data['ds'] > df['ds'].max()]
                future_chart_data['Type'] = 'Forecast'
                
                combined = pd.concat([hist_chart_data, future_chart_data])
                
                # Simple Altair Line Chart
                chart = alt.Chart(combined).mark_line().encode(
                    x=alt.X('ds:T', title='Date'),
                    y=alt.Y('y:Q', title='Price', scale=alt.Scale(zero=False)),
                    color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Forecast'], range=['#1f77b4', '#ff7f0e']))
                ).properties(
                    height=400
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
