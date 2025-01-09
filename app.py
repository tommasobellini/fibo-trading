import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import ta

# -----------------------------
# 1) Get S&P 500 Tickers (Wikipedia scraping)
# -----------------------------
@st.cache_data
def get_sp500_tickers():
    """
    Fetch the list of S&P 500 companies from Wikipedia and return as a list.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    tables = pd.read_html(html)
    # The first table on the page typically has the current S&P 500 constituents
    df_sp500 = tables[0]
    # Some symbols contain "." instead of "-", etc. Let's do minimal cleaning.
    tickers = df_sp500['Symbol'].unique().tolist()
    # Remove any weird tickers like BF.B or BRK.B if you want (optional).
    # For simplicity, let's keep them but note that Yahoo's API may expect BRK-B as BRK-B, etc.
    # We'll do a small replace for any that use periods:
    cleaned_tickers = []
    for t in tickers:
        # E.g. 'BRK.B' -> 'BRK-B'
        if "." in t:
            t = t.replace(".", "-")
        cleaned_tickers.append(t)
    return cleaned_tickers


# -----------------------------
# 2) Pivot Calculation
# -----------------------------
def pivot_high(series, left_bars, right_bars):
    """
    Identify pivot highs where 'series' is greater than the left_bars bars to the left
    and greater than the right_bars bars to the right.
    Return a boolean Series of the same length as 'series'.
    """
    cond = pd.Series([True] * len(series), index=series.index)

    # Compare to the left
    for i in range(1, left_bars + 1):
        cond_left = series > series.shift(i)
        cond = cond & cond_left.fillna(False)

    # Compare to the right
    for j in range(1, right_bars + 1):
        cond_right = series > series.shift(-j)
        cond = cond & cond_right.fillna(False)

    return cond.fillna(False)


def pivot_low(series, left_bars, right_bars):
    """
    Identify pivot lows where 'series' is less than the left_bars bars to the left
    and less than the right_bars bars to the right.
    Return a boolean Series of the same length as 'series'.
    """
    cond = pd.Series([True] * len(series), index=series.index)

    # Compare to the left
    for i in range(1, left_bars + 1):
        cond_left = series < series.shift(i)
        cond = cond & cond_left.fillna(False)

    # Compare to the right
    for j in range(1, right_bars + 1):
        cond_right = series < series.shift(-j)
        cond = cond & cond_right.fillna(False)

    return cond.fillna(False)


# -----------------------------
# 3) Indicator Calculations
# -----------------------------
def compute_indicators(df):
    """
    Given a DataFrame with columns: ['Open','High','Low','Close','Volume']
    Return a dict of Series for various indicators.
    """

    # RSI (14)
    rsi_14 = ta.momentum.rsi(df['Close'], window=14)

    # MACD
    macd_inst = ta.trend.MACD(
        df['Close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    macd_line = macd_inst.macd()          # MACD line
    macd_hist = macd_inst.macd_diff()     # MACD histogram

    # Stochastic (14,3)
    stoch_k = ta.momentum.stoch(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14,
        smooth_window=3
    )

    # CCI (20)
    cci_20 = ta.trend.cci(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=20
    )

    # Momentum (10) - replicate a simple approach: (Close - Close.shift(10))
    momentum_10 = df['Close'] - df['Close'].shift(10)

    # OBV
    obv_series = ta.volume.on_balance_volume(
        df['Close'],
        df['Volume']
    )

    # CMF (21)
    cmf_21 = ta.volume.chaikin_money_flow(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume'],
        window=21
    )

    # MFI (14)
    mfi_14 = ta.volume.money_flow_index(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume'],
        window=14
    )

    return {
        "rsi": rsi_14,
        "macd_line": macd_line,
        "macd_hist": macd_hist,
        "stoch": stoch_k,
        "cci": cci_20,
        "momentum": momentum_10,
        "obv": obv_series,
        "cmf": cmf_21,
        "mfi": mfi_14,
    }


# -----------------------------
# 4) Divergence Detection
# -----------------------------
def detect_divergences(df, indicator, pivot_high_col, pivot_low_col, lookback_bars=100):
    """
    Checks for:
      - Positive Regular  (price forms higher low, indicator forms lower low)
      - Negative Regular  (price forms lower high, indicator forms higher high)
      - Positive Hidden   (price forms lower low,  indicator forms higher low)
      - Negative Hidden   (price forms higher high, indicator forms lower high)

    Returns a dict of boolean arrays for each type, e.g. signals['positive_regular']
    is a boolean array indicating where a positive regular divergence was found.
    """
    signals = {
        'positive_regular': np.zeros(len(df), dtype=bool),
        'negative_regular': np.zeros(len(df), dtype=bool),
        'positive_hidden':  np.zeros(len(df), dtype=bool),
        'negative_hidden':  np.zeros(len(df), dtype=bool),
    }

    pivot_low_idx = df.index[df[pivot_low_col]]
    pivot_high_idx = df.index[df[pivot_high_col]]

    # Positive Regular Divergence
    for i in range(1, len(pivot_low_idx)):
        curr = pivot_low_idx[i]
        prev = pivot_low_idx[i-1]
        # Skip if too far apart
        if (curr - prev).days > lookback_bars:
            continue

        # Price
        price_prev = df.loc[prev, 'Close']
        price_curr = df.loc[curr, 'Close']
        # Indicator
        indi_prev = indicator.loc[prev]
        indi_curr = indicator.loc[curr]

        # Price forms higher low, indicator forms lower low
        if price_curr > price_prev and indi_curr < indi_prev:
            signals['positive_regular'][df.index.get_loc(curr)] = True

    # Negative Regular Divergence
    for i in range(1, len(pivot_high_idx)):
        curr = pivot_high_idx[i]
        prev = pivot_high_idx[i-1]
        if (curr - prev).days > lookback_bars:
            continue

        price_prev = df.loc[prev, 'Close']
        price_curr = df.loc[curr, 'Close']
        indi_prev = indicator.loc[prev]
        indi_curr = indicator.loc[curr]

        # Price forms lower high, indicator forms higher high
        if price_curr < price_prev and indi_curr > indi_prev:
            signals['negative_regular'][df.index.get_loc(curr)] = True

    # Positive Hidden Divergence
    for i in range(1, len(pivot_low_idx)):
        curr = pivot_low_idx[i]
        prev = pivot_low_idx[i-1]
        if (curr - prev).days > lookback_bars:
            continue

        price_prev = df.loc[prev, 'Close']
        price_curr = df.loc[curr, 'Close']
        indi_prev = indicator.loc[prev]
        indi_curr = indicator.loc[curr]

        # Price forms lower low, indicator forms higher low
        if price_curr < price_prev and indi_curr > indi_prev:
            signals['positive_hidden'][df.index.get_loc(curr)] = True

    # Negative Hidden Divergence
    for i in range(1, len(pivot_high_idx)):
        curr = pivot_high_idx[i]
        prev = pivot_high_idx[i-1]
        if (curr - prev).days > lookback_bars:
            continue

        price_prev = df.loc[prev, 'Close']
        price_curr = df.loc[curr, 'Close']
        indi_prev = indicator.loc[prev]
        indi_curr = indicator.loc[curr]

        # Price forms higher high, indicator forms lower high
        if price_curr > price_prev and indi_curr < indi_prev:
            signals['negative_hidden'][df.index.get_loc(curr)] = True

    return signals


def screen_for_divergences(ticker, period="6mo", interval="1d", prd=5, showlimit=1):
    """
    1. Download data from yfinance
    2. Compute pivot highs & lows
    3. Compute indicators
    4. Detect divergences
    5. Check total divergences on the last bar
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or len(df) == 0:
        return None

    df.dropna(inplace=True)
    if any(col not in df.columns for col in ["Open","High","Low","Close","Volume"]):
        return None

    # Pivot columns
    df["pivot_high"] = pivot_high(df["High"], prd, prd)
    df["pivot_low"]  = pivot_low(df["Low"],  prd, prd)

    # Indicators
    inds = compute_indicators(df)

    # Gather signals into a DataFrame
    all_signals_df = pd.DataFrame(index=df.index)
    for name, indi_series in inds.items():
        if indi_series.isnull().all():
            continue
        signals = detect_divergences(
            df=df,
            indicator=indi_series,
            pivot_high_col="pivot_high",
            pivot_low_col="pivot_low",
            lookback_bars=100
        )
        for sig_name, arr in signals.items():
            col = f"{name}_{sig_name}"
            all_signals_df[col] = arr

    if len(all_signals_df) == 0:
        return None

    # Count divergences in the last bar
    last_bar_signals = all_signals_df.iloc[-1]
    total_divergences = last_bar_signals.sum()
    meets_threshold = (total_divergences >= showlimit)

    result = {
        "ticker": ticker,
        "data_points": len(df),
        "last_bar_date": df.index[-1],
        "divergences_last_bar": int(total_divergences),
        "signal": meets_threshold,
    }
    return result


# -----------------------------
# 5) Streamlit App
# -----------------------------
def main():
    st.title("Full S&P 500 Divergence Screener")
    st.write("Scrapes current S&P 500 constituents from Wikipedia, then analyzes all for divergences.")

    period = st.sidebar.selectbox("Yahoo Finance Period:", ["3mo","6mo","1y","2y","5y"], index=1)
    interval = st.sidebar.selectbox("Data Interval:", ["1d","1h","15m"], index=0)
    prd = st.sidebar.slider("Pivot Period (prd)", min_value=2, max_value=10, value=5, step=1)
    showlimit = st.sidebar.slider("Minimum divergences on last bar to flag signal", min_value=1, max_value=5, value=1)

    if st.button("Run Screener"):
        tickers = get_sp500_tickers()
        results = []

        for ticker in tickers:
            try:
                out = screen_for_divergences(
                    ticker=ticker,
                    period=period,
                    interval=interval,
                    prd=prd,
                    showlimit=showlimit
                )
                if out is not None:
                    results.append(out)
                else:
                    # If no data, or error in retrieval
                    results.append({
                        "ticker": ticker,
                        "data_points": 0,
                        "last_bar_date": None,
                        "divergences_last_bar": 0,
                        "signal": False
                    })
            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "data_points": 0,
                    "last_bar_date": None,
                    "divergences_last_bar": 0,
                    "signal": False,
                    "error": str(e)
                })

        if results:
            df_res = pd.DataFrame(results)
            st.write(f"**Analyzed {len(tickers)} tickers.**")
            st.dataframe(df_res)
        else:
            st.write("No results to display.")


if __name__ == "__main__":
    main()
