import streamlit as st
import requests
import pandas as pd
import yfinance as yf

# -------------------------
# 1) HELPER FUNCTIONS
# -------------------------
def get_sp500_tickers():
    """
    Scrapes Wikipedia to retrieve the current list of S&P 500 companies and returns the tickers.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    # Parse the HTML using pandas
    sp500_table = pd.read_html(html, header=0)[0]
    # The tickers are typically in the first column under 'Symbol'
    tickers = sp500_table['Symbol'].unique().tolist()
    # Some tickers contain '.' such as 'BRK.B' on Wikipedia, 
    # but yfinance expects '-' for those. Replace as needed:
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers

def is_white_marubozu(df, threshold=0.01):
    """
    Determines if the last row in a DataFrame has a white marubozu candle.

    threshold: Float between 0 and 1 indicating the fraction of the day's range 
               that upper/lower wick can occupy. A smaller value => stricter marubozu.
    
    A 'white marubozu' is assumed if:
        - Close > Open (bullish candle)
        - The difference between Open and Low is < threshold * (High - Low)
        - The difference between High and Close is < threshold * (High - Low)
    """
    if df.empty:
        return False
    
    last_candle = df.iloc[-1]
    open_price = last_candle['Open']
    high_price = last_candle['High']
    low_price = last_candle['Low']
    close_price = last_candle['Close']
    
    # If not a bullish candle, skip
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

def screen_sp500_marubozu(threshold=0.01):
    """
    Screens through the S&P 500 for a white Marubozu candlestick on the latest day.
    Logs progress in the Streamlit app.
    """
    st.write("Fetching S&P 500 ticker list...")
    tickers = get_sp500_tickers()
    st.write(f"Found {len(tickers)} tickers. Starting the screening...")

    marubozu_tickers = []
    
    for ticker in tickers:
        st.write(f"Processing {ticker} ...")
        try:
            # Use period="3mo" to avoid yfinance period errors
            df = yf.download(ticker, period="3mo", interval="1d", progress=False)
            
            if df.shape[0] == 0:
                st.write(f"No data for {ticker}, skipping.")
                continue
            
            if is_white_marubozu(df, threshold=threshold):
                st.write(f">>> Found White Marubozu: {ticker}")
                marubozu_tickers.append(ticker)
        except Exception as e:
            st.write(f"Error retrieving data for {ticker}: {e}")
    
    return marubozu_tickers

# -------------------------
# 2) STREAMLIT APP LAYOUT
# -------------------------
st.title("S&P 500 White Marubozu Screener")

st.write("Click the **Start Screening** button below to fetch S&P 500 stocks and identify any last-day White Marubozu candles.")

if st.button("Start Screening"):
    marubozu_results = screen_sp500_marubozu(threshold=0.01)
    st.write("#### Screening Complete!")
    
    if marubozu_results:
        st.write("**S&P 500 stocks with last-day White Marubozu candles:**")
        for stock in marubozu_results:
            st.write(f"- {stock}")
    else:
        st.write("No White Marubozu candles found today.")
