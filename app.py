import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

# --- 1. Set Page Configuration Before Any Other Streamlit Commands ---
st.set_page_config(page_title="📈 Fibonacci + SuperTrend Screener", layout="wide")

# --- 2. Import Libraries and Define Functions ---

def calculate_supertrend(df, period=10, multiplier=3):
    """
    Calculate SuperTrend indicator for the given DataFrame.

    Parameters:
    - df: Pandas DataFrame with columns ['High', 'Low', 'Close']
    - period: ATR period
    - multiplier: ATR multiplier

    Returns:
    - df: DataFrame with SuperTrend columns added
    """
    df = df.copy()
    # Calculate ATR
    df['ATR'] = df['High'].rolling(window=period).max() - df['Low'].rolling(window=period).min()
    df['ATR'] = df['ATR'].rolling(window=period).mean()

    # Calculate basic upper and lower bands
    hl2 = (df['High'] + df['Low']) / 2
    df['Upper Basic'] = hl2 + (multiplier * df['ATR'])
    df['Lower Basic'] = hl2 - (multiplier * df['ATR'])

    # Initialize Upper Band and Lower Band
    df['Upper Band'] = df[['Upper Basic', 'Close']].apply(
        lambda x: min(x['Upper Basic'], x['Close']) if x['Close'] < x['Upper Basic'] else x['Upper Basic'], axis=1)
    df['Lower Band'] = df[['Lower Basic', 'Close']].apply(
        lambda x: max(x['Lower Basic'], x['Close']) if x['Close'] > x['Lower Basic'] else x['Lower Basic'], axis=1)

    # Initialize SuperTrend
    df['SuperTrend'] = np.nan
    trend = True  # True for uptrend, False for downtrend

    for current in range(1, len(df)):
        previous = current - 1

        if pd.isna(df['SuperTrend'][previous]):
            df.at[current, 'SuperTrend'] = df.at[current, 'Lower Band']
            continue

        if df.at[current, 'Close'] > df.at[previous, 'SuperTrend']:
            trend = True
        elif df.at[current, 'Close'] < df.at[previous, 'SuperTrend']:
            trend = False

        if trend:
            df.at[current, 'SuperTrend'] = df.at[current, 'Lower Band']
        else:
            df.at[current, 'SuperTrend'] = df.at[current, 'Upper Band']

    return df

@st.cache(allow_output_mutation=True)
def get_sp500_tickers():
    """
    Fetch the list of S&P 500 tickers from Wikipedia.

    Returns:
    - List of ticker symbols.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    tickers = df['Symbol'].tolist()
    # Adjust tickers with dots to dashes (e.g., BRK.B to BRK-B)
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

def calculate_fibonacci_levels(df, lookback=100, selected_levels=None):
    """
    Calculate Fibonacci retracement levels.

    Parameters:
    - df: DataFrame with 'High' and 'Low' columns
    - lookback: Number of periods to look back
    - selected_levels: List of Fibonacci levels to calculate

    Returns:
    - Dictionary of Fibonacci levels
    """
    if selected_levels is None:
        selected_levels = ['23.6%', '38.2%', '50%', '61.8%', '78.6%']
    
    recent_data = df.tail(lookback)
    max_price = recent_data['High'].max()
    min_price = recent_data['Low'].min()
    diff = max_price - min_price
    levels = {}
    for level in selected_levels:
        percentage = float(level.strip('%')) / 100
        levels[level] = max_price - percentage * diff
    return levels

def apply_strategy(df, fib_levels, fib_tolerance=0.01):
    """
    Apply Fibonacci + SuperTrend strategy to the DataFrame.

    Parameters:
    - df: DataFrame with necessary columns
    - fib_levels: Dictionary of Fibonacci levels
    - fib_tolerance: Tolerance percentage for proximity

    Returns:
    - signal: 'Buy', 'Sell', or 'Hold'
    """
    # Determine the latest SuperTrend
    supertrend = df['SuperTrend'].iloc[-1]
    close_price = df['Close'].iloc[-1]

    # Determine trend
    if close_price > supertrend:
        trend = 'Uptrend'
    else:
        trend = 'Downtrend'

    # Determine signal based on trend and Fibonacci levels
    signal = 'Hold'
    
    if trend == 'Uptrend':
        target_level = fib_levels.get('61.8%')
        if target_level:
            if abs(close_price - target_level) / target_level <= fib_tolerance:
                signal = 'Buy'
    elif trend == 'Downtrend':
        target_level = fib_levels.get('38.2%')
        if target_level:
            if abs(close_price - target_level) / target_level <= fib_tolerance:
                signal = 'Sell'

    return signal

def process_ticker(ticker, lookback, fib_levels_selected, fib_tolerance, atr_period, multiplier):
    """
    Process a single ticker to determine if it matches the strategy criteria.

    Parameters:
    - ticker: Stock ticker symbol
    - lookback: Fibonacci lookback period
    - fib_levels_selected: List of selected Fibonacci levels
    - fib_tolerance: Tolerance for Fibonacci proximity
    - atr_period: SuperTrend ATR period
    - multiplier: SuperTrend multiplier

    Returns:
    - result: Dictionary with ticker details if criteria met, else None
    """
    try:
        # Fetch historical data (past 1 year)
        df = yf.download(ticker, period='1y', interval='1d', progress=False)
        if df.empty or len(df) < atr_period + 1:
            return None
        # Calculate SuperTrend
        df = calculate_supertrend(df, period=atr_period, multiplier=multiplier)
        # Calculate Fibonacci levels
        fib_levels = calculate_fibonacci_levels(df, lookback=lookback, selected_levels=fib_levels_selected)
        # Apply strategy
        signal = apply_strategy(df, fib_levels, fib_tolerance=fib_tolerance)
        if signal in ['Buy', 'Sell']:
            return {
                'Ticker': ticker,
                'Signal': signal,
                'Close Price': round(df['Close'].iloc[-1], 2),
                'SuperTrend': round(df['SuperTrend'].iloc[-1], 2),
                'Trend': 'Uptrend' if signal == 'Buy' else 'Downtrend'
            }
    except Exception as e:
        # Log the exception if needed
        return None

def main():
    st.title("📈 Fibonacci Golden Level + SuperTrend Strategy Screener for S&P 500")
    st.markdown("""
        This application screens S&P 500 stocks based on the combined **Fibonacci Golden Level** and **SuperTrend** strategy.
        
        **Strategy Overview**:
        - **Uptrend**: Identified by SuperTrend indicator.
        - **Buy Signal**: Price is near the selected Fibonacci retracement level during an uptrend.
        - **Downtrend**: Identified by SuperTrend indicator.
        - **Sell Signal**: Price is near the selected Fibonacci retracement level during a downtrend.
    """)

    # --- Sidebar for Configuration ---
    st.sidebar.header("🛠️ Configuration Settings")

    # Fibonacci Settings
    st.sidebar.subheader("🔢 Fibonacci Settings")
    fib_levels_available = ['23.6%', '38.2%', '50%', '61.8%', '78.6%']
    fib_levels_selected = st.sidebar.multiselect(
        "Select Fibonacci Levels to Use",
        options=fib_levels_available,
        default=['61.8%', '38.2%']
    )
    fib_tolerance = st.sidebar.slider(
        "Fibonacci Proximity Tolerance (%)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1
    ) / 100  # Convert to decimal
    fib_lookback = st.sidebar.slider(
        "Fibonacci Lookback Period (days)",
        min_value=50,
        max_value=200,
        value=100,
        step=10
    )

    # SuperTrend Settings
    st.sidebar.subheader("📈 SuperTrend Settings")
    atr_period = st.sidebar.slider(
        "SuperTrend ATR Period",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    multiplier = st.sidebar.slider(
        "SuperTrend Multiplier",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5
    )

    # Option to select whether to include Buy, Sell signals
    st.sidebar.subheader("📊 Signal Selection")
    include_buy = st.sidebar.checkbox("Include Buy Signals", value=True)
    include_sell = st.sidebar.checkbox("Include Sell Signals", value=True)

    # Button to run screener
    if st.button("🔍 Run Screener"):
        with st.spinner("Fetching and processing data..."):
            tickers = get_sp500_tickers()
            results = []

            # Initialize ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=20) as executor:
                # Prepare futures
                futures = {
                    executor.submit(
                        process_ticker, ticker, fib_lookback, fib_levels_selected, fib_tolerance, atr_period, multiplier
                    ): ticker for ticker in tickers
                }
                # Progress bar
                progress_bar = st.progress(0)
                total = len(futures)
                processed = 0

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        # Filter based on user selection
                        if (result['Signal'] == 'Buy' and include_buy) or (result['Signal'] == 'Sell' and include_sell):
                            results.append(result)
                    processed += 1
                    progress_bar.progress(processed / total)

            progress_bar.empty()

            if results:
                result_df = pd.DataFrame(results)
                # Sort by Signal for better visualization
                result_df = result_df.sort_values(by='Signal', ascending=False)
                st.success("📊 Screening Complete!")
                st.write(f"**Found {len(result_df)} stocks matching the criteria:**")

                # Split the results into Buy and Sell for better visualization
                buy_df = result_df[result_df['Signal'] == 'Buy']
                sell_df = result_df[result_df['Signal'] == 'Sell']

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🔼 Buy Signals")
                    st.dataframe(buy_df.reset_index(drop=True))

                with col2:
                    st.subheader("🔻 Sell Signals")
                    st.dataframe(sell_df.reset_index(drop=True))

                # Option to download results
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Results as CSV",
                    data=csv,
                    file_name='fibo_supertrend_signals.csv',
                    mime='text/csv',
                )
            else:
                st.warning("⚠️ No stocks found matching the criteria.")

    st.markdown("---")
    st.markdown("**Disclaimer**: This tool is for educational purposes only and does not constitute financial advice. Always do your own research before making any investment decisions.")

if __name__ == "__main__":
    main()
