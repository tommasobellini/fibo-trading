import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

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
    df['ATR'] = df['High'].rolling(window=period).max() - df['Low'].rolling(window=period).min()
    df['ATR'] = df['ATR'].rolling(window=period).mean()

    df['Upper Basic'] = (df['High'] + df['Low']) / 2 + multiplier * df['ATR']
    df['Lower Basic'] = (df['High'] + df['Low']) / 2 - multiplier * df['ATR']

    df['Upper Band'] = df[['Upper Basic', 'Close']].apply(
        lambda x: min(x['Upper Basic'], x['Close']) if x['Close'] < x['Upper Basic'] else x['Upper Basic'], axis=1)
    df['Lower Band'] = df[['Lower Basic', 'Close']].apply(
        lambda x: max(x['Lower Basic'], x['Close']) if x['Close'] > x['Lower Basic'] else x['Lower Basic'], axis=1)

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

def calculate_fibonacci_levels(df, lookback=100):
    """
    Calculate Fibonacci retracement levels.
    
    Parameters:
    - df: DataFrame with 'High' and 'Low' columns
    - lookback: Number of periods to look back
    
    Returns:
    - Dictionary of Fibonacci levels
    """
    recent_data = df.tail(lookback)
    max_price = recent_data['High'].max()
    min_price = recent_data['Low'].min()
    diff = max_price - min_price
    levels = {
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '78.6%': max_price - 0.786 * diff
    }
    return levels

def apply_strategy(df):
    """
    Apply Fibonacci + SuperTrend strategy to the DataFrame.
    
    Parameters:
    - df: DataFrame with necessary columns
    
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
    
    # Get Fibonacci levels
    levels = calculate_fibonacci_levels(df)
    
    # Determine proximity to Fibonacci levels (±1%)
    tolerance = 0.01
    signal = 'Hold'
    
    if trend == 'Uptrend':
        target_level = levels['61.8%']
        if abs(close_price - target_level) / target_level <= tolerance:
            signal = 'Buy'
    elif trend == 'Downtrend':
        target_level = levels['38.2%']
        if abs(close_price - target_level) / target_level <= tolerance:
            signal = 'Sell'
    
    return signal

def main():
    st.set_page_config(page_title="Fibonacci + SuperTrend Screener", layout="wide")
    st.title("📈 Fibonacci Golden Level + SuperTrend Strategy Screener for S&P 500")
    st.markdown("""
        This application screens S&P 500 stocks based on the combined **Fibonacci Golden Level** and **SuperTrend** strategy.
        
        **Strategy Overview**:
        - **Uptrend**: Identified by SuperTrend indicator.
        - **Buy Signal**: Price is near the 61.8% Fibonacci retracement level during an uptrend.
        - **Downtrend**: Identified by SuperTrend indicator.
        - **Sell Signal**: Price is near the 38.2% Fibonacci retracement level during a downtrend.
    """)
    
    if st.button("Run Screener"):
        with st.spinner("Fetching and processing data..."):
            tickers = get_sp500_tickers()
            results = []
            
            for ticker in tickers:
                try:
                    # Fetch historical data (past 1 year)
                    df = yf.download(ticker, period='1y', interval='1d', progress=False)
                    if df.empty:
                        continue
                    # Calculate SuperTrend
                    df = calculate_supertrend(df)
                    # Apply strategy
                    signal = apply_strategy(df)
                    if signal in ['Buy', 'Sell']:
                        results.append({
                            'Ticker': ticker,
                            'Signal': signal,
                            'Close Price': round(df['Close'].iloc[-1], 2),
                            'SuperTrend': round(df['SuperTrend'].iloc[-1], 2),
                            'Trend': 'Uptrend' if signal == 'Buy' else 'Downtrend'
                        })
                except Exception as e:
                    # Handle exceptions (e.g., data fetching issues)
                    continue
            
            if results:
                result_df = pd.DataFrame(results)
                result_df = result_df.sort_values(by='Signal', ascending=False)
                st.success("Screening Complete!")
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
            else:
                st.warning("No stocks found matching the criteria.")
    
    st.markdown("---")
    st.markdown("**Disclaimer**: This tool is for educational purposes only and does not constitute financial advice. Always do your own research before making any investment decisions.")

if __name__ == "__main__":
    main()
