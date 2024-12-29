import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Helper Functions
# -----------------------------

def get_sp500_tickers():
    """Scrape S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to load S&P 500 page")
    
    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    tickers = df['Symbol'].tolist()
    # Handle tickers with dots (e.g., BRK.B) by replacing with hyphens for yfinance
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    """Fetch historical data for a given ticker."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return None

def calculate_atr(df, length):
    """Calculate Average True Range (ATR)."""
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=length, min_periods=1).mean()
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    return df

def calculate_supertrend(df, length, factor):
    """Calculate SuperTrend indicator."""
    atr = df['ATR']
    hl2 = (df['High'] + df['Low']) / 2
    df['Upper'] = hl2 + (atr * factor)
    df['Lower'] = hl2 - (atr * factor)
    df['SuperTrend'] = 0
    df['Trend'] = 0

    for i in range(1, len(df)):
        if df['Close'][i] > df['Upper'][i-1]:
            df.at[i, 'Trend'] = 1
        elif df['Close'][i] < df['Lower'][i-1]:
            df.at[i, 'Trend'] = 0
        else:
            df.at[i, 'Trend'] = df['Trend'][i-1]
            if df['Trend'][i] == 1:
                df.at[i, 'Upper'] = min(df['Upper'][i], df['Upper'][i-1])
            else:
                df.at[i, 'Lower'] = max(df['Lower'][i], df['Lower'][i-1])
        
        if df['Trend'][i] == 1:
            df.at[i, 'SuperTrend'] = df['Lower'][i]
        else:
            df.at[i, 'SuperTrend'] = df['Upper'][i]
    
    return df

def perform_kmeans(data, n_clusters=3, max_iter=1000):
    """Perform K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
    kmeans.fit(data.reshape(-1, 1))
    return kmeans.labels_, kmeans.cluster_centers_

def calculate_bull_market_support_band(df, sma_length=20, ema_length=21):
    """Calculate Bull Market Support Band using weekly SMA and EMA."""
    df['Weekly_SMA'] = df['Close'].rolling(window=sma_length, min_periods=1).mean()
    df['Weekly_EMA'] = df['Close'].ewm(span=ema_length, adjust=False).mean()
    return df

def generate_signals(df, supertrend_col='SuperTrend', trend_col='Trend'):
    """Generate buy and sell signals based on SuperTrend."""
    df['Signal'] = 0
    df['Signal'][1:] = np.where(df[trend_col][1:] > df[trend_col][:-1], 1, 
                                np.where(df[trend_col][1:] < df[trend_col][:-1], -1, 0))
    return df

def apply_trading_strategy(df, atr_length, min_mult, max_mult, step, perf_alpha, from_cluster, max_iter, max_data):
    """Apply the SuperTrend AI (Clustering) and Bull Market Support Band strategy."""
    # Calculate ATR
    df = calculate_atr(df, atr_length)
    
    # Define factors
    factors = np.arange(min_mult, max_mult + step, step)
    factors = np.round(factors, 2)
    
    supertrend_dict = {}
    
    # Calculate SuperTrend for multiple factors
    for factor in factors:
        temp_df = calculate_supertrend(df.copy(), atr_length, factor)
        temp_df['Perf'] = temp_df['Close'].diff().fillna(0) * np.sign(df['Close'].diff().fillna(0))
        perf = temp_df['Perf'].ewm(span=perf_alpha, adjust=False).mean()
        supertrend_dict[factor] = perf
    
    # Prepare data for clustering
    perf_values = []
    factor_values = []
    for factor in factors:
        perf = supertrend_dict[factor].iloc[-1]
        perf_values.append(perf)
        factor_values.append(factor)
    
    perf_array = np.array(perf_values).reshape(-1, 1)
    
    # Perform K-means clustering
    labels, centers = perform_kmeans(perf_array.flatten(), n_clusters=3, max_iter=max_iter)
    
    cluster_info = {}
    for i in range(3):
        cluster_info[i] = {
            'perf': perf_array[labels == i],
            'factor': np.array(factor_values)[labels == i]
        }
    
    # Select target cluster
    if from_cluster == "Best":
        target_cluster = np.argmax(centers)
    elif from_cluster == "Average":
        target_cluster = 1  # Assuming 3 clusters: 0-Worst, 1-Average, 2-Best
    else:
        target_cluster = np.argmin(centers)
    
    target_factors = cluster_info[target_cluster]['factor']
    target_factor = target_factors.mean() if len(target_factors) > 0 else min_mult
    
    # Calculate SuperTrend with target factor
    df = calculate_supertrend(df, atr_length, target_factor)
    
    # Calculate Performance Index (perf_idx)
    den = df['Close'].diff().abs().ewm(span=perf_alpha, adjust=False).mean()
    perf_idx = cluster_info[target_cluster]['perf'].mean() / den.iloc[-1] if den.iloc[-1] != 0 else 0
    
    # Calculate Trailing Stop Adaptive MA
    df['TS'] = df['SuperTrend']
    df['Perf_AMA'] = df['TS'].ewm(alpha=perf_idx, adjust=False).mean()
    
    # Calculate Bull Market Support Band
    df = calculate_bull_market_support_band(df)
    
    # Generate Signals
    df = generate_signals(df)
    
    # Extract the latest signal
    latest_signal = df['Signal'].iloc[-1]
    
    return latest_signal, df

# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.title("S&P 500 Trading Strategy Screener")
    
    st.sidebar.header("Screening Parameters")
    
    # Strategy Settings
    atr_length = st.sidebar.number_input("ATR Length", min_value=1, value=10)
    min_mult = st.sidebar.number_input("Factor Range - Min Mult", min_value=0.0, value=1.0, step=0.1)
    max_mult = st.sidebar.number_input("Factor Range - Max Mult", min_value=0.0, value=5.0, step=0.1)
    step = st.sidebar.number_input("Factor Step", min_value=0.1, value=0.5, step=0.1)
    
    # Validation
    if min_mult > max_mult:
        st.sidebar.error("Minimum factor cannot be greater than maximum factor.")
        st.stop()
    
    perf_alpha = st.sidebar.number_input("Performance Memory (perfAlpha)", min_value=2, value=10, step=1)
    from_cluster = st.sidebar.selectbox("From Cluster", options=["Best", "Average", "Worst"])
    
    # Optimization Settings
    max_iter = st.sidebar.number_input("Maximum Iteration Steps", min_value=1, value=1000, step=100)
    max_data = st.sidebar.number_input("Historical Bars Calculation", min_value=1, value=10000, step=1000)
    
    # Screening Settings
    st.sidebar.subheader("Screening Options")
    signal_filter = st.sidebar.selectbox("Filter by Signal", options=["All", "Buy", "Sell"], index=0)
    
    # Fetch S&P 500 tickers
    st.header("Fetching S&P 500 Tickers...")
    tickers = get_sp500_tickers()
    st.success(f"Retrieved {len(tickers)} tickers.")
    
    # Date Range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    # Progress Indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Results Storage
    results = []
    
    # Function to process each ticker
    def process_ticker(ticker):
        data = fetch_data(ticker, start_date, end_date)
        if data is None or len(data) < atr_length + 1:
            return None
        try:
            signal, processed_df = apply_trading_strategy(
                data,
                atr_length=atr_length,
                min_mult=min_mult,
                max_mult=max_mult,
                step=step,
                perf_alpha=perf_alpha,
                from_cluster=from_cluster,
                max_iter=max_iter,
                max_data=max_data
            )
            return (ticker, signal)
        except Exception as e:
            return None
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
        for idx, future in enumerate(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                pass
            progress_bar.progress((idx + 1) / len(tickers))
            status_text.text(f"Processing {idx + 1} of {len(tickers)} tickers...")
    
    progress_bar.empty()
    status_text.empty()
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results, columns=['Ticker', 'Signal'])
    
    # Apply Signal Filter
    if signal_filter == "Buy":
        screened_df = results_df[results_df['Signal'] == 1]
    elif signal_filter == "Sell":
        screened_df = results_df[results_df['Signal'] == -1]
    else:
        screened_df = results_df.copy()
    
    screened_df = screened_df.sort_values(by='Signal', ascending=False)
    
    st.header("Screening Results")
    st.write(f"Total Stocks Found: {len(screened_df)}")
    st.dataframe(screened_df.reset_index(drop=True))
    
    # Optional: Visualize Selected Stock
    if len(screened_df) > 0:
        selected_ticker = st.selectbox("Select a Ticker to Visualize", options=screened_df['Ticker'].tolist())
        
        if selected_ticker:
            st.subheader(f"Chart for {selected_ticker}")
            data = fetch_data(selected_ticker, start_date, end_date)
            if data is not None and len(data) >= atr_length + 1:
                latest_signal, processed_df = apply_trading_strategy(
                    data,
                    atr_length=atr_length,
                    min_mult=min_mult,
                    max_mult=max_mult,
                    step=step,
                    perf_alpha=perf_alpha,
                    from_cluster=from_cluster,
                    max_iter=max_iter,
                    max_data=max_data
                )
                
                fig = go.Figure()
        
                # Price Candles
                fig.add_trace(go.Candlestick(x=processed_df['Date'],
                                             open=processed_df['Open'],
                                             high=processed_df['High'],
                                             low=processed_df['Low'],
                                             close=processed_df['Close'],
                                             name='Price'))
                
                # SuperTrend
                fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['SuperTrend'], 
                                         line=dict(color='blue', width=1), 
                                         name='SuperTrend'))
                
                # Trailing Stop AMA
                fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['Perf_AMA'], 
                                         line=dict(color='purple', width=1, dash='dash'), 
                                         name='Trailing Stop AMA'))
                
                # Bull Market Support Band
                fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['Weekly_SMA'], 
                                         line=dict(color='red', width=1), 
                                         name='20w SMA'))
                fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['Weekly_EMA'], 
                                         line=dict(color='green', width=1), 
                                         name='21w EMA'))
                fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['Weekly_SMA'],
                                         fill=None, mode='lines', line=dict(color='orange'), showlegend=False))
                fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['Weekly_EMA'],
                                         fill='tonexty', mode='lines', line=dict(color='orange'), 
                                         name='Support Band'))
                
                # Signals
                buy_signals = processed_df[processed_df['Signal'] == 1]
                sell_signals = processed_df[processed_df['Signal'] == -1]
                
                fig.add_trace(go.Scatter(mode='markers',
                                         x=buy_signals['Date'],
                                         y=buy_signals['Low'],
                                         marker=dict(symbol='triangle-up', color='green', size=10),
                                         name='Buy Signal'))
                fig.add_trace(go.Scatter(mode='markers',
                                         x=sell_signals['Date'],
                                         y=sell_signals['High'],
                                         marker=dict(symbol='triangle-down', color='red', size=10),
                                         name='Sell Signal'))
                
                # Update layout
                fig.update_layout(title=f"{selected_ticker} Trading Strategy",
                                  yaxis_title="Price",
                                  xaxis_title="Date",
                                  xaxis_rangeslider_visible=False,
                                  template="seaborn")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Insufficient data to plot the chart.")
    else:
        st.info("No stocks matched the selected criteria.")
    
    st.markdown("""
    ---
    **Disclaimer:** This screener is for educational purposes only and does not constitute financial advice. Trading involves risk.
    """)

if __name__ == "__main__":
    main()
