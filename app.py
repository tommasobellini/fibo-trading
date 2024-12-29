import yfinance as yf
import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
import streamlit as st

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def screen_stocks(stock_symbols):
    results = []

    for symbol in stock_symbols:
        try:
            # Fetch 2 years of weekly data
            data = yf.download(symbol, period='2y', interval='1wk')

            if data.empty:
                print(f"No data for {symbol}")
                continue

            data['SMA_20'] = calculate_sma(data['Close'], 20)
            data['EMA_21'] = calculate_ema(data['Close'], 21)

            # Check the latest values for SMA and EMA
            latest_sma = data['SMA_20'].iloc[-1]
            latest_ema = data['EMA_21'].iloc[-1]
            latest_close = data['Close'].iloc[-1]

            # Conditions for the Bull Market Support Band
            if latest_close > latest_ema > latest_sma:
                results.append({'Symbol': symbol, 'Close': latest_close, 'EMA_21': latest_ema, 'SMA_20': latest_sma})

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    return pd.DataFrame(results)

# Streamlit Interface
st.title("S&P 500 Bull Market Support Band Screener")

# Fetch the list of S&P 500 tickers dynamically
sp500_symbols = si.tickers_sp500()

# Streamlit select box to choose tickers
selected_symbols = st.multiselect("Select S&P 500 Tickers to Screen:", sp500_symbols, default=sp500_symbols[:5])

if st.button("Run Screener"):
    screened_stocks = screen_stocks(selected_symbols)

    if not screened_stocks.empty:
        st.write("Bull Market Support Band Stocks:")
        st.dataframe(screened_stocks)
    else:
        st.write("No stocks meet the criteria.")
