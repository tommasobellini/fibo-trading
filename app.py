import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import List
import time

# Set Streamlit page configuration
st.set_page_config(
    page_title="S&P 500 Stock Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("📈 S&P 500 Stock Screener: Undervalued & Oversold Stocks")

# Sidebar for user inputs
st.sidebar.header("Screening Parameters")

# User inputs
rsi_threshold = st.sidebar.number_input(
    "RSI Oversold Threshold (Default: 30)",
    min_value=10,
    max_value=50,
    value=30,
    step=1,
)

pe_threshold = st.sidebar.number_input(
    "P/E Ratio Maximum (Default: 15)",
    min_value=5.0,
    max_value=50.0,
    value=15.0,
    step=0.5,
)

lookback_days = st.sidebar.number_input(
    "Lookback Period (Days, Default: 90)",
    min_value=30,
    max_value=365,
    value=90,
    step=30,
)

# Button to start screening
start_screening = st.sidebar.button("Start Screening")

# Function to fetch S&P 500 tickers
@st.cache_data(show_spinner=False)
def fetch_sp500_tickers() -> List[str]:
    """
    Fetch S&P 500 tickers from Wikipedia.
    Returns a list of ticker symbols.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    response.raise_for_status()
    html = response.text

    # Parse the HTML table using pandas
    sp500_table = pd.read_html(html, attrs={"id": "constituents"})[0]
    tickers = sp500_table["Symbol"].tolist()
    # Convert tickers to Yahoo Finance format
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers

# Function to calculate RSI
def calculate_RSI(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
    """
    delta = series.diff()
    gains = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
    losses = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
    # Prevent divide-by-zero
    losses = losses.replace(0, 1e-10)
    rs = gains / losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to check oversold condition
def is_oversold(rsi_value: float, threshold: float) -> bool:
    return rsi_value < threshold

# Function to check undervalued condition
def is_undervalued(pe_ratio: float, max_pe: float) -> bool:
    if pd.isna(pe_ratio) or pe_ratio <= 0:
        return False
    return pe_ratio < max_pe

# Main screening function
def screen_stocks(
    tickers: List[str],
    rsi_threshold: float,
    pe_threshold: float,
    lookback_days: int,
) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)

    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(tickers)

    for idx, ticker in enumerate(tickers):
        try:
            # Fetch historical price data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                continue

            # Calculate RSI
            price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            df["RSI"] = calculate_RSI(df[price_col])

            # Get the latest RSI
            latest_rsi = df["RSI"].iloc[-1] if not df["RSI"].isna().all() else np.nan

            # Fetch fundamental data
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info

            # Get trailing P/E ratio
            pe_ratio = info.get("trailingPE", np.nan)

            # Check criteria
            if is_oversold(latest_rsi, rsi_threshold) and is_undervalued(pe_ratio, pe_threshold):
                results.append({
                    "Ticker": ticker,
                    "RSI": round(latest_rsi, 2) if pd.notna(latest_rsi) else "N/A",
                    "Trailing P/E": round(pe_ratio, 2) if pd.notna(pe_ratio) else "N/A",
                    "Company Name": info.get("shortName", "N/A"),
                    "Sector": info.get("sector", "N/A"),
                })

        except Exception as e:
            # Optionally, log the error or pass
            pass

        # Update progress
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Processing {idx + 1} of {total} tickers...")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by RSI ascending
    results_df.sort_values(by="RSI", inplace=True)

    return results_df

# Display the screening results
if start_screening:
    with st.spinner("Fetching S&P 500 tickers..."):
        try:
            sp500_tickers = fetch_sp500_tickers()
        except Exception as e:
            st.error(f"Error fetching S&P 500 tickers: {e}")
            sp500_tickers = []

    if sp500_tickers:
        with st.spinner("Screening stocks based on your criteria..."):
            screened_df = screen_stocks(
                tickers=sp500_tickers,
                rsi_threshold=rsi_threshold,
                pe_threshold=pe_threshold,
                lookback_days=lookback_days,
            )

        if not screened_df.empty:
            st.success("Screening complete! Found the following stocks:")
            # Display the DataFrame
            st.dataframe(
                screened_df.style.format({
                    "RSI": "{:.2f}",
                    "Trailing P/E": "{:.2f}",
                }),
                height=600,
            )
            # Optionally, provide download option
            csv = screened_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='sp500_undervalued_oversold.csv',
                mime='text/csv',
            )
        else:
            st.info("No stocks found matching the criteria.")
    else:
        st.error("Failed to retrieve S&P 500 tickers.")

# Optional: Display app information or disclaimers
st.markdown("---")
st.markdown("""
**Disclaimer**: This app is for educational purposes only and does not constitute financial advice. Always conduct your own research or consult a financial professional before making investment decisions.
""")
