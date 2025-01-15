import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# If TA-Lib is not available, you can implement an RSI function manually.
# Example RSI implementation:
def compute_rsi(prices, period=14):
    """
    Compute RSI for a pandas Series of prices.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_bullish_rsi_divergence(df, rsi_column='RSI', lookback=5):
    """
    Simple check for bullish RSI divergence in the most recent segment.
    - We check if price made a lower low but RSI made a higher low
      over a certain lookback window.
    - This is a simplified illustration and may not be robust for all cases.
    """
    # Ensure we have enough rows
    if len(df) < lookback * 2:
        return False
    
    # Consider the last 'lookback' bars for analysis
    recent_df = df.iloc[-lookback:]
    
    # Price lows
    min_price_1 = df['Close'].iloc[-(lookback * 2):-lookback].min()
    min_price_2 = recent_df['Close'].min()
    
    # RSI lows
    min_rsi_1 = df[rsi_column].iloc[-(lookback * 2):-lookback].min()
    min_rsi_2 = recent_df[rsi_column].min()

    # Condition for bullish divergence:
    # Price forms a lower low, but RSI forms a higher low.
    price_divergence = (min_price_2 < min_price_1)
    rsi_divergence = (min_rsi_2 > min_rsi_1)
    
    return price_divergence and rsi_divergence


def main():
    st.title("Small-Cap Stocks Screener: Divergence (RSI) Example")
    
    # 1. Input parameters / user selections
    st.sidebar.header("Screener Settings")
    lookback_days = st.sidebar.number_input("Lookback for Divergence Detection", 
                                            min_value=2, max_value=30, value=5)
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

    # 2. Placeholder list of small-cap US stocks:
    #    In reality, you would fetch from a real source (e.g. an API or CSV).
    small_cap_tickers = [
        "PLUG",  # Just examples
        "FCEL",
        "WKHS",
        "GNUS",
        "BBIG",
        "SNDL"
    ]
    
    st.write("Analyzing the following small-cap tickers:", small_cap_tickers)
    
    # 3. Fetch data, compute RSI, and detect divergence
    results = []
    
    for ticker in small_cap_tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                continue
            
            # Compute RSI
            df['RSI'] = compute_rsi(df['Close'])
            
            # Drop initial NaN rows from RSI calculation
            df = df.dropna()
            if df.empty:
                continue
            
            # Detect bullish divergence
            has_divergence = detect_bullish_rsi_divergence(df, 'RSI', lookback_days)
            
            if has_divergence:
                # For demonstration, let’s store the last close, RSI, etc.
                last_close = df['Close'].iloc[-1]
                last_rsi = df['RSI'].iloc[-1]
                
                results.append({
                    'Ticker': ticker,
                    'Last Close': last_close,
                    'Last RSI': round(last_rsi, 2),
                    'Divergence': "Bullish RSI Divergence"
                })
        
        except Exception as e:
            st.write(f"Error fetching data for {ticker}: {e}")
    
    # 4. Display results in a DataFrame
    if results:
        results_df = pd.DataFrame(results)
        st.subheader("Stocks with Detected Divergence")
        st.dataframe(results_df)
    else:
        st.subheader("No divergences detected with current settings.")

if __name__ == "__main__":
    main()
