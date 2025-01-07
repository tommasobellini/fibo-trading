import streamlit as st
import requests
import pandas as pd
import yfinance as yf

def get_sp500_tickers():
    """
    Scrapes Wikipedia to retrieve the current list of S&P 500 companies and returns the tickers.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    sp500_table = pd.read_html(html, header=0)[0]
    tickers = sp500_table['Symbol'].unique().tolist()
    # Some tickers contain '.' such as 'BRK.B' on Wikipedia, 
    # but yfinance expects '-' for those. Replace as needed:
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers

def is_white_marubozu_yesterday(df, threshold=0.01):
    """
    Determines if yesterday's candle (the second-to-last row) in a DataFrame has a white marubozu.
    
    threshold: Float between 0 and 1 indicating the fraction of the day's range 
               that upper/lower wick can occupy.
    """
    # Need at least 2 rows to check "yesterday"
    if len(df) < 2:
        return False
    
    candle = df.iloc[-2]  # second-to-last row
    open_price = candle['Open']
    high_price = candle['High']
    low_price = candle['Low']
    close_price = candle['Close']
    
    # Must be a bullish candle
    if close_price <= open_price:
        return False
    
    candle_range = high_price - low_price
    if candle_range == 0:
        return False
    
    # Check if open is near the low
    if (open_price - low_price) > threshold * candle_range:
        return False
    
    # Check if close is near the high
    if (high_price - close_price) > threshold * candle_range:
        return False
    
    return True

def screen_sp500_marubozu_yesterday(threshold=0.01):
    """
    Screens through the S&P 500 for a white Marubozu candlestick on yesterday's candle.
    Logs progress in the Streamlit app.
    """
    st.write("Fetching S&P 500 ticker list...")
    tickers = get_sp500_tickers()
    st.write(f"Found {len(tickers)} tickers. Starting the screening...")

    marubozu_tickers = []
    
    # Download all tickers in a single call
    st.write("Downloading data for all tickers (3mo daily bars)...")
    df = yf.download(
        tickers, 
        period="3mo", 
        interval="1d", 
        group_by='ticker', 
        progress=False
    )

    for ticker in tickers:
        try:
            st.write(f"Processing {ticker} ...")
            ticker_df = df[ticker].dropna()
            
            if is_white_marubozu_yesterday(ticker_df, threshold=threshold):
                st.write(f">>> Found White Marubozu (yesterday): {ticker}")
                marubozu_tickers.append(ticker)
        except Exception as e:
            st.write(f"Error retrieving data for {ticker}: {e}")
    
    return marubozu_tickers

# -------------------------
#  STREAMLIT APP LAYOUT
# -------------------------
st.title("S&P 500 White Marubozu Screener (Yesterday's Candle)")

st.write("Click the **Start Screening** button below to fetch S&P 500 stocks and identify any yesterday White Marubozu candles.")

if st.button("Start Screening"):
    marubozu_results = screen_sp500_marubozu_yesterday(threshold=0.01)
    st.write("#### Screening Complete!")
    
    if marubozu_results:
        st.write("**S&P 500 stocks with yesterday's White Marubozu candles:**")
        for stock in marubozu_results:
            st.write(f"- {stock}")
    else:
        st.write("No White Marubozu candles found for yesterday.")
