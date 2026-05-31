import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import numpy as np

# Import our decoupled hybrid logic
from predictor import get_stock_data, train_hybrid_model, predict_hybrid_future, search_ticker, get_qualitative_context

st.set_page_config(page_title="Hybrid Stock Price Predictor", layout="wide")

# Use Streamlit's cache_data for data downloading
@st.cache_data(show_spinner="Fetching historical data...")
def load_data(ticker, history_years):
    return get_stock_data(ticker, history_years)

@st.cache_data(show_spinner=False)
def get_search_results(query):
    return search_ticker(query)

@st.cache_data(show_spinner=False)
def get_context(ticker):
    return get_qualitative_context(ticker)

# Use cache_resource for the model since it's an object we shouldn't repeatedly train 
# if the underlying data and parameters haven't changed.
@st.cache_resource(show_spinner="Training Hybrid Prophet + Gradient Boosting model...")
def train_model_v8(ticker, _df, changepoint_scale):
    # Hardcode standard sensible defaults for the end-user
    return train_hybrid_model(
        ticker,
        _df, 
        changepoint_scale=changepoint_scale, 
        seasonality_mode="multiplicative", 
        xgb_lr=0.05, 
        xgb_depth=3
    )

def main():
    # Callback to reset prediction state when settings are changed
    def reset_predict_state():
        st.session_state.predict_clicked = False

    st.title("📈 Hybrid Stock Price Predictor")
    st.markdown("Forecasting stock prices using Facebook Prophet for macro-trends and Gradient Boosting for micro-volatility.")
    
    st.header("Comparison Settings")
    
    tickers_to_compare = []
    labels_to_compare = []
    
    # Create 3 horizontal columns for the stock inputs
    input_cols = st.columns(3)
    
    for i in range(1, 4):
        label = f"Stock {i}" if i == 1 else f"Stock {i} (Optional)"
        with input_cols[i-1]:
            with st.expander(label, expanded=True):
                search_query = st.text_input(f"Search Company {i}", key=f"search_{i}", on_change=reset_predict_state)
                if search_query:
                    options = get_search_results(search_query)
                    if not options:
                        options = {search_query: search_query}
                    selected_label = st.selectbox(f"Select {i}", options=list(options.keys()), key=f"select_{i}", on_change=reset_predict_state)
                    tickers_to_compare.append(options[selected_label])
                    labels_to_compare.append(selected_label.split(' (')[0])
    
    # Hardcode history_years to 10 for institutional use cases
    history_years = 10

    # Bottom row of settings
    setting_col1, setting_col2, setting_col3 = st.columns([1.5, 1.5, 1])
    
    with setting_col1:
        strategy = st.selectbox(
            "Investment Strategy",
            ["Value (Mean Reversion)", "Growth (Momentum)"],
            help="Value: Forces the model to stick to long-term 10-year averages. Growth: Allows the model to extrapolate recent compounding momentum."
        )
        # Map strategy to changepoint scale
        changepoint_scale = 0.050 if strategy == "Growth (Momentum)" else 0.010

    with setting_col2:
        default_date = datetime.now().date() + timedelta(days=1)
        max_date = datetime.now().date() + timedelta(days=7300) # 20 years into the future
        target_date_input = st.date_input(
            "Target Date (Forecast Horizon)", 
            value=default_date, 
            min_value=default_date, 
            max_value=max_date,
            help="Forecast horizon up to 20 years for institutional investors.",
            on_change=reset_predict_state
        )

    with setting_col3:
        st.write("") # Spacing to align button with input
        st.write("")
        predict_btn = st.button("Predict Prices", type="primary", use_container_width=True)
        if predict_btn:
            st.session_state.predict_clicked = True

    st.divider()

    if st.session_state.get('predict_clicked', False) and tickers_to_compare:
            # Create dynamic columns
            cols = st.columns(len(tickers_to_compare))
            
            for i, ticker in enumerate(tickers_to_compare):
                with cols[i]:
                    st.subheader(labels_to_compare[i])
                    try:
                        # 1. Fetch data
                        df = load_data(ticker, history_years)
                        
                        # 2. Train hybrid model
                        model_dict = train_model_v8(ticker, df, changepoint_scale)
                        
                        # 3. Predict
                        predicted_price, forecast = predict_hybrid_future(model_dict, df, str(target_date_input))
                        
                        # 4. Qualitative Override (Automatic Blend)
                        qual_context = get_context(ticker)
                        adj = qual_context.get('adjustment_factor', 1.0)
                        
                        # Smoothly apply adjustment
                        steps = len(forecast)
                        if steps > 0:
                            scaling_array = [1.0 + ((adj - 1.0) * (i / steps)) for i in range(steps)]
                            forecast['hybrid_yhat'] = forecast['hybrid_yhat'] * scaling_array
                            
                        # Layout Metrics (Grab last actual price first to calculate gap)
                        last_row = df.iloc[-1]
                        last_price = last_row['y']
                        last_date = last_row['ds'].strftime('%Y-%m-%d')
                        
                        # --- ANCHOR SMOOTHING (Prevent Day-1 Cliffs) ---
                        # Calculate the visual gap between the actual last price and the model's theoretical T+1 price
                        first_pred = forecast['hybrid_yhat'].iloc[0]
                        gap = last_price - first_pred
                        
                        # Exponentially decay this gap over the forecast (half-life of ~10 days) to smoothly merge curves
                        decay_rate = 0.07 
                        decay_array = np.exp(-decay_rate * np.arange(len(forecast)))
                        forecast['hybrid_yhat'] = forecast['hybrid_yhat'] + (gap * decay_array)
                        # -----------------------------------------------

                        # Recalculate predicted_price for the specific target date AFTER scaling and smoothing
                        target_dt = pd.to_datetime(target_date_input).tz_localize(None).normalize()
                        pred_row = forecast[forecast['ds'] == target_dt]
                        if not pred_row.empty:
                            predicted_price = pred_row['hybrid_yhat'].values[0]
                        else:
                            predicted_price = forecast.iloc[-1]['hybrid_yhat']
                            
                        # Slice the visual forecast so the chart only shows up to the target date
                        visual_forecast = forecast[forecast['ds'] <= target_dt]
                        
                        # Qualitative Analyst Block
                        if qual_context.get('news_count', 0) > 0:
                            sentiment_pct = (adj - 1.0) * 100
                            st.info(f"**🤖 AI Adj: {sentiment_pct:+.1f}%**\n"
                                    f"Sentiment: {qual_context['sentiment_score']:+.2f}\n"
                                    f"Margin: {qual_context['profit_margins']} | Rev: {qual_context['revenue_growth']}")
                        
                        delta = predicted_price - last_price
                        pct_change = (delta / last_price) * 100
                        
                        st.metric("Last Close", f"${last_price:.2f}", last_date)
                        st.metric("Prediction", f"${predicted_price:.2f}", f"{target_date_input}")
                        st.metric("Projected Move", f"{delta:+.2f}", f"{pct_change:+.2f}%")
                        
                        st.divider()
                        
                        # Visualizations
                        hist_chart_data = df.copy()
                        hist_chart_data['Type'] = 'Historical'
                        
                        future_chart_data = visual_forecast[['ds', 'hybrid_yhat']].copy()
                        future_chart_data = future_chart_data.rename(columns={'hybrid_yhat': 'y'})
                        future_chart_data = future_chart_data[future_chart_data['ds'] > df['ds'].max()]
                        future_chart_data['Type'] = 'Forecast'
                        
                        combined = pd.concat([hist_chart_data, future_chart_data])
                        
                        # Simple Altair Line Chart
                        chart = alt.Chart(combined).mark_line().encode(
                            x=alt.X('ds:T', title=None, axis=alt.Axis(labels=False)), 
                            y=alt.Y('y:Q', title='Price', scale=alt.Scale(zero=False)),
                            color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Forecast'], range=['#1f77b4', '#ff7f0e']), legend=None)
                        ).properties(
                            height=300
                        ).interactive()
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Goal Seek Logic
                        with st.expander("🎯 Goal Seek (Reverse Forecast)"):
                            st.caption("Calculate the exact date the stock is projected to hit a specific price.")
                            
                            # Constrain width and add explicit button
                            goal_col1, goal_col2 = st.columns([2, 1])
                            with goal_col1:
                                goal_price = st.number_input("Target Price", value=float(last_price * 1.25), key=f"goal_input_{ticker}", step=10.0, label_visibility="collapsed")
                            with goal_col2:
                                calc_btn = st.button("Calculate Date", key=f"goal_btn_{ticker}", use_container_width=True)
                            
                            if calc_btn:
                                if goal_price > last_price:
                                    hit_rows = forecast[forecast['hybrid_yhat'] >= goal_price]
                                else:
                                    hit_rows = forecast[forecast['hybrid_yhat'] <= goal_price]
                                    
                                if not hit_rows.empty:
                                    hit_date = hit_rows['ds'].iloc[0]
                                    days_to_hit = (hit_date - pd.Timestamp.now().normalize()).days
                                    if days_to_hit > 0:
                                        st.success(f"Projected to hit **${goal_price:,.2f}** on **{hit_date.strftime('%b %d, %Y')}** ({days_to_hit/365:.1f} years).")
                                    else:
                                        st.success(f"Already crossed **${goal_price:,.2f}** historically or today.")
                                else:
                                    st.warning("Not projected to hit this price within the 20-year macro forecast horizon.")

                    except Exception as e:
                        st.error(f"Error analyzing {ticker}: {str(e)}")

if __name__ == "__main__":
    main()
