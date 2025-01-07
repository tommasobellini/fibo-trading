import streamlit as st
import pandas as pd
import requests
import yfinance as yf

# -------------------------
# 1) HELPER FUNCTIONS
# -------------------------
def get_sp500_tickers():
    """
    Scrapes Wikipedia for the current list of S&P 500 companies and returns the tickers.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    
    sp500_table = pd.read_html(html, header=0)[0]
    tickers = sp500_table['Symbol'].unique().tolist()
    # Replace '.' with '-' for yfinance (e.g. 'BRK.B' -> 'BRK-B')
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

def is_white_marubozu(row, threshold=0.01):
    """
    Checks if a single candle (row) is a White Marubozu:
      - Close > Open (bullish)
      - (Open - Low)  < threshold * (High - Low)
      - (High - Close) < threshold * (High - Low)
    Returns True/False.
    """
    open_price  = row['Open']
    close_price = row['Close']
    high_price  = row['High']
    low_price   = row['Low']
    
    # Must be bullish
    if close_price <= open_price:
        return False
    
    candle_range = high_price - low_price
    if candle_range == 0:
        return False
    
    # Check lower wick
    if (open_price - low_price) > threshold * candle_range:
        return False
    
    # Check upper wick
    if (high_price - close_price) > threshold * candle_range:
        return False
    
    return True

def find_marubozu_in_lookback(df, lookback=1, threshold=0.01):
    """
    Checks if there is ANY White Marubozu candle among the last `lookback` fully-closed candles.

    By default, `lookback=1` => checks ONLY 'yesterday' (the second-to-last row, i.e. df.iloc[-2]).
    If `lookback=10`, it checks df.iloc[-2] through df.iloc[-(2+lookback-1)], i.e. the 10 most recent
    fully-closed daily candles.

    Returns True if at least one of those candles is White Marubozu, else False.
    """
    # We need at least (lookback + 1) rows because we skip the last row (-1) 
    # which may be incomplete or "today."
    if len(df) < (lookback + 1):
        return False
    
    # Check from -2 (yesterday) back to -2 - (lookback-1).
    start_index = -2
    end_index = -(2 + lookback - 1)  # inclusive

    for i in range(start_index, end_index - 1, -1):
        candle = df.iloc[i]
        if is_white_marubozu(candle, threshold=threshold):
            return True
    
    return False

# -------------------------
#  2) FIXED SCREEN FUNCTION
# -------------------------
def screen_sp500_marubozu_yf(lookback=1, threshold=0.01):
    """
    Screens through the S&P 500 for any White Marubozu candles within the specified lookback
    (defaults to checking just 'yesterday'), but downloads in chunks to avoid the
    'Cannot join tz-naive with tz-aware DatetimeIndex' error.

    Logs progress in the Streamlit app.
    """
    st.write("Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    st.write(f"Found {len(tickers)} tickers. Downloading data in batches...")

    # Helper to create chunks/batches of a list
    def chunker(seq, size):
        for pos in range(0, len(seq), size):
            yield seq[pos : pos + size]

    batch_size = 50  # 50 tickers per batch; adjust as you see fit
    all_batches = []

    # 1) Download in chunks
    for chunk in chunker(tickers, batch_size):
        st.write(f"Downloading batch of size {len(chunk)}: {chunk[:3]}... + others")
        # Download for this chunk
        df_chunk = yf.download(
            chunk,
            period="3mo",
            interval="1d",
            group_by="ticker",
            progress=False
        )
        # 2) Remove timezone from each ticker's index (tz_localize(None)) to unify
        for t in chunk:
            try:
                if not df_chunk[t].empty:
                    df_chunk[t].index = df_chunk[t].index.tz_localize(None)
            except Exception as e:
                st.write(f"Timezone fix error for {t}: {e}")
        # Store the chunk
        all_batches.append(df_chunk)
    
    # 3) Concatenate all chunked data horizontally
    try:
        df_all = pd.concat(all_batches, axis=1)
    except Exception as e:
        st.write("Error concatenating batch data:", e)
        return []

    # 4) Now perform the marubozu screening
    marubozu_tickers = []
    st.write(f"Checking the last {lookback} fully-closed candles for each ticker...")

    for ticker in tickers:
        st.write(f"Processing {ticker}...")
        try:
            df_ticker = df_all[ticker].dropna()
            if find_marubozu_in_lookback(df_ticker, lookback=lookback, threshold=threshold):
                st.write(f"**>>> Found White Marubozu in last {lookback} candles: {ticker}**")
                marubozu_tickers.append(ticker)
        except Exception as e:
            st.write(f"Error with {ticker}: {e}")

    return marubozu_tickers

# -------------------------
# 3) STREAMLIT APP
# -------------------------
def main():
    st.title("S&P 500 White Marubozu Screener")
    st.write(
        """
        This tool checks for a White Marubozu candle in the **last N** fully-closed daily bars.  
        By default, N=1 (i.e., 'yesterday'). Increase N to check more recent days (e.g., last 10).
        """
    )

    lookback = st.number_input(
        "How many past candles do you want to check?",
        min_value=1,
        max_value=30,
        value=1,   # Default is 1 => 'yesterday'
        step=1
    )

    threshold = st.slider(
        "Marubozu threshold (fraction of candle range allowed for wicks)",
        min_value=0.0001,
        max_value=0.05,
        value=0.01,
        step=0.001
    )

    if st.button("Start Screening"):
        st.write(f"Scanning for White Marubozu in last {lookback} fully-closed candles...")
        marubozu_results = screen_sp500_marubozu_yf(lookback=lookback, threshold=threshold)
        st.write("#### Screening Complete!")

        if marubozu_results:
            st.write(
                f"**Found {len(marubozu_results)} tickers with at least one White Marubozu** "
                f"among the last {lookback} bars:"
            )
            for stock in marubozu_results:
                st.write(f"- {stock}")
        else:
            st.write("No White Marubozu candles found in that window.")

if __name__ == "__main__":
    main()
