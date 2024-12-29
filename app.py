import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import List
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="S&P 500 Stock Screener with Dynamic Price Oscillator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("📈 S&P 500 Stock Screener: Undervalued & Oversold Stocks with DPO")

# Sidebar for user inputs
st.sidebar.header("Screening Parameters")

# User inputs for RSI, P/E, Lookback
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

# User inputs for DPO parameters
dpo_length = st.sidebar.number_input(
    "DPO Lookback Length (Default: 33)",
    min_value=1,
    max_value=100,
    value=33,
    step=1,
)

dpo_smooth_factor = st.sidebar.number_input(
    "DPO Smoothing Factor (Default: 5)",
    min_value=1,
    max_value=50,
    value=5,
    step=1,
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

# Function to calculate True Range
def calculate_true_range(high, low, close):
    """
    Calculate True Range (TR).
    TR = max(high - low, abs(high - previous close), abs(low - previous close))
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(src, length, mult):
    """
    Calculate Bollinger Bands.
    """
    basis = src.rolling(window=length, min_periods=1).mean()
    dev = mult * src.rolling(window=length, min_periods=1).std()
    upper = basis + dev
    lower = basis - dev
    return upper, lower

# Function to calculate Dynamic Price Oscillator
def calculate_dynamic_price_oscillator(df, length=33, smooth_factor=5):
    """
    Calculate the Dynamic Price Oscillator and Bollinger Bands.
    Adds oscillator and Bollinger Bands columns to the DataFrame.
    
    Parameters:
    - df: DataFrame with 'High', 'Low', 'Close' columns.
    - length: Lookback period for calculations.
    - smooth_factor: Smoothing factor for EMA.
    
    Returns:
    - df: DataFrame with added oscillator and Bollinger Bands columns.
    """
    # Calculate True Range
    df['True_Range'] = calculate_true_range(df['High'], df['Low'], df['Close'])
    
    # Volume-Adjusted Price (EMA of True Range)
    df['VolAdjPrice'] = df['True_Range'].ewm(span=length, adjust=False).mean()
    
    # Price Change and Price Delta
    df['PriceChange'] = df['Close'] - df['Close'].shift(length)
    df['PriceDelta'] = df['Close'] - df['VolAdjPrice']
    
    # Oscillator Calculation
    df['Oscillator'] = (df['PriceDelta'].combine(df['PriceChange'], func=lambda x, y: np.nan if np.isnan(x) or np.isnan(y) else (x + y) / 2)).ewm(span=smooth_factor, adjust=False).mean()
    
    # Bollinger Bands on Oscillator
    boll_length = length * 5
    df['BB_High'], df['BB_Low'] = calculate_bollinger_bands(df['Oscillator'], boll_length, 1)
    df['BB_HighExp'], df['BB_LowExp'] = calculate_bollinger_bands(df['Oscillator'], boll_length, 2)
    df['Mean'] = (df['BB_HighExp'] + df['BB_LowExp']) / 2
    
    return df

# Function to calculate RSI
def calculate_RSI(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
    # Prevent division by zero
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to check if oscillator is below Bollinger Bands
def is_oscillator_below_bbands(df):
    """
    Check if oscillator is below the lower Bollinger Bands.
    Returns True if oscillator is below BB_Low or BB_LowExp.
    """
    return (df['Oscillator'] < df['BB_Low']) | (df['Oscillator'] < df['BB_LowExp'])

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
    dpo_length: int,
    dpo_smooth_factor: int
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
            
            # Ensure necessary columns are present
            if not {'High', 'Low', 'Close'}.issubset(df.columns):
                continue
            
            # Calculate RSI
            df['RSI'] = calculate_RSI(df['Close'])
            
            # Calculate Dynamic Price Oscillator
            df = calculate_dynamic_price_oscillator(df, length=dpo_length, smooth_factor=dpo_smooth_factor)
            
            # Get the latest RSI and Oscillator values
            latest_rsi = df['RSI'].iloc[-1] if not df['RSI'].isna().all() else np.nan
            latest_oscillator = df['Oscillator'].iloc[-1] if not df['Oscillator'].isna().all() else np.nan
            
            # Check oscillator condition
            oscillator_condition = False
            if not df[['Oscillator', 'BB_Low', 'BB_LowExp']].isna().all().all():
                oscillator_condition = is_oscillator_below_bbands(df).iloc[-1]
            
            # Fetch fundamental data
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # Get trailing P/E ratio
            pe_ratio = info.get("trailingPE", np.nan)
            
            # Get Fair Value Price (targetMeanPrice)
            fair_value_price = info.get("targetMeanPrice", np.nan)
            
            # Get Current Price with fallback
            current_price = info.get("regularMarketPrice", np.nan)
            if pd.isna(current_price):
                # Fallback to the latest Close price from historical data
                if 'Close' in df.columns and not df['Close'].isna().all():
                    current_price = df['Close'].iloc[-1]
                else:
                    current_price = np.nan
            
            # Check combined criteria
            if (is_oversold(latest_rsi, rsi_threshold) and
                is_undervalued(pe_ratio, pe_threshold) and
                oscillator_condition):
                
                results.append({
                    "Ticker": ticker,
                    "Current Price": round(current_price, 2) if pd.notna(current_price) else np.nan,
                    "RSI": round(latest_rsi, 2) if pd.notna(latest_rsi) else np.nan,
                    "Trailing P/E": round(pe_ratio, 2) if pd.notna(pe_ratio) else np.nan,
                    "Fair Value Price": round(fair_value_price, 2) if pd.notna(fair_value_price) else np.nan,
                    "Oscillator": round(latest_oscillator, 2) if pd.notna(latest_oscillator) else np.nan,
                    "Company Name": info.get("shortName", "N/A"),
                    "Sector": info.get("sector", "N/A"),
                })
        
        except Exception as e:
            # Log the error with ticker information
            logging.error(f"Error processing {ticker}: {e}")
            pass
        
        # Update progress
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Processing {idx + 1} of {total} tickers...")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Conditional Sorting
    if not results_df.empty and "RSI" in results_df.columns:
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
                dpo_length=dpo_length,
                dpo_smooth_factor=dpo_smooth_factor
            )
    
        if not screened_df.empty:
            st.success("Screening complete! Found the following stocks:")
            # Display the DataFrame
            st.dataframe(
                screened_df.style.format({
                    "Current Price": "${:.2f}",
                    "RSI": "{:.2f}",
                    "Trailing P/E": "{:.2f}",
                    "Fair Value Price": "${:.2f}",
                    "Oscillator": "{:.2f}",
                }),
                height=600,
            )
            # Optionally, provide download option
            csv = screened_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='sp500_undervalued_oversold_dpo.csv',
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
