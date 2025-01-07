import yfinance as yf
import pandas as pd
import requests

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    sp500_table = pd.read_html(html, header=0)[0]
    tickers = sp500_table['Symbol'].unique().tolist()
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

def is_white_marubozu(df, threshold=0.01):
    """
    Determines if the second-to-last row in a DataFrame has a white marubozu candle
    (i.e., 'yesterday's' candle).
    """
    # Need at least 2 rows
    if len(df) < 2:
        return False
    
    # Instead of the last row (-1), use the second to last row (-2)
    last_candle = df.iloc[-2]
    
    open_price = last_candle['Open']
    high_price = last_candle['High']
    low_price  = last_candle['Low']
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

def screen_sp500_marubozu():
    tickers = get_sp500_tickers()
    print(f"Downloading data for {len(tickers)} tickers...")

    # STEP 1: Download data for all S&P 500 tickers in 1–5 batches
    df = yf.download(tickers, period="3mo", interval="1d", group_by='ticker')

    # STEP 2: Analyze
    marubozu_list = []
    for ticker in tickers:
        try:
            # Each ticker's data is in df[ticker]
            ticker_df = df[ticker].dropna()
            if is_white_marubozu(ticker_df):
                marubozu_list.append(ticker)
        except:
            pass

    return marubozu_list

if __name__ == "__main__":
    results = screen_sp500_marubozu()
    print("Marubozu stocks:")
    for r in results:
        print(r)
