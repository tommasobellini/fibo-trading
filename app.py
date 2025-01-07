import requests
import pandas as pd
import yfinance as yf

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
        - The difference between Open and Low is less than threshold * (High - Low)
        - The difference between High and Close is less than threshold * (High - Low)
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

def screen_sp500_marubozu():
    """
    Screens through the S&P 500 for a white Marubozu candlestick on the latest day.
    """
    tickers = get_sp500_tickers()
    marubozu_tickers = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False)
            
            # If there's not enough data, skip
            if df.shape[0] == 0:
                continue
            
            if is_white_marubozu(df):
                marubozu_tickers.append(ticker)
        except Exception as e:
            # In case of any download error or data parsing error, just skip
            print(f"Error retrieving data for {ticker}: {e}")
    
    return marubozu_tickers

if __name__ == "__main__":
    # Run the screening
    white_marubozu_stocks = screen_sp500_marubozu()
    
    print("S&P 500 stocks with last-day White Marubozu candles:")
    for stock in white_marubozu_stocks:
        print(stock)
