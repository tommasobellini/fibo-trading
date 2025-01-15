import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import requests

# -----------------------------------
# 1) Get a universe of US stocks
#    For demonstration, let's use S&P 500 from Wikipedia
# -----------------------------------
@st.cache_data
def get_sp500_tickers():
    """Scrapes S&P 500 from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    tables = pd.read_html(html)
    df_sp500 = tables[0]
    tickers = df_sp500['Symbol'].unique().tolist()
    # Clean up . to - for Yahoo (e.g. BRK.B -> BRK-B)
    cleaned = []
    for t in tickers:
        if "." in t:
            t = t.replace(".", "-")
        cleaned.append(t)
    return cleaned

# -----------------------------------
# 2) Pivot High / Pivot Low detection
#    This mimics the "prd" pivot logic from PineScript.
# -----------------------------------
def pivot_high(series, left_bars, right_bars):
    """
    Return True at index i if series[i] is greater than
    series[i +/- k] for k in [1..left_bars] and [1..right_bars].
    """
    cond = pd.Series([True]*len(series), index=series.index)
    for i in range(1, left_bars+1):
        cond = cond & (series > series.shift(i)).fillna(False)
    for j in range(1, right_bars+1):
        cond = cond & (series > series.shift(-j)).fillna(False)
    return cond.fillna(False)

def pivot_low(series, left_bars, right_bars):
    """
    Return True at index i if series[i] is less than
    series[i +/- k] for k in [1..left_bars] and [1..right_bars].
    """
    cond = pd.Series([True]*len(series), index=series.index)
    for i in range(1, left_bars+1):
        cond = cond & (series < series.shift(i)).fillna(False)
    for j in range(1, right_bars+1):
        cond = cond & (series < series.shift(-j)).fillna(False)
    return cond.fillna(False)

# -----------------------------------
# 3) Indicators (matching PineScript's “many indicators”)
# -----------------------------------
def compute_indicators(df):
    """
    We replicate some of the built-in indicators from the Pine script:
      - MACD line, MACD histogram
      - RSI
      - Stoch
      - CCI
      - Momentum (10)
      - OBV
      - CMF
      - MFI
    For "VWmacd" you can define it similarly or skip.
    """
    # Ensure no missing columns
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    # MACD
    macd_obj = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_obj.macd()
    macd_hist = macd_obj.macd_diff()

    # RSI
    rsi_14 = ta.momentum.rsi(df["Close"], window=14)

    # Stochastic
    stoch_k = ta.momentum.stoch(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14,
        smooth_window=3
    )

    # CCI
    cci_20 = ta.trend.cci(
        high=df["High"], 
        low=df["Low"], 
        close=df["Close"], 
        window=20
    )

    # Momentum(10) = close - close.shift(10)
    momentum_10 = df["Close"] - df["Close"].shift(10)

    # OBV
    obv_series = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # CMF (21)
    cmf_21 = ta.volume.chaikin_money_flow(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"],
        window=21
    )

    # MFI (14)
    mfi_14 = ta.volume.money_flow_index(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"],
        window=14
    )

    return {
        "macd_line": macd_line,
        "macd_hist": macd_hist,
        "rsi": rsi_14,
        "stoch": stoch_k,
        "cci": cci_20,
        "momentum": momentum_10,
        "obv": obv_series,
        "cmf": cmf_21,
        "mfi": mfi_14,
    }

# -----------------------------------
# 4) Divergence logic
#    We check the 4 typical divergences (regular & hidden, positive & negative).
# -----------------------------------
def detect_divergences(df, indicator, pivot_high_col, pivot_low_col, lookback_bars=100):
    """
    Return dict of 4 arrays:
      - positive_regular
      - negative_regular
      - positive_hidden
      - negative_hidden

    "positive_regular": Price forms higher low, indicator forms lower low.
    "negative_regular": Price forms lower high, indicator forms higher high.
    "positive_hidden":  Price forms lower low, indicator forms higher low.
    "negative_hidden":  Price forms higher high, indicator forms lower high.

    We skip pivot pairs if they're too far apart in days ((curr - prev).days > lookback_bars).
    """
    signals = {
        'positive_regular': np.zeros(len(df), dtype=bool),
        'negative_regular': np.zeros(len(df), dtype=bool),
        'positive_hidden':  np.zeros(len(df), dtype=bool),
        'negative_hidden':  np.zeros(len(df), dtype=bool),
    }

    pivot_low_idx  = df.index[df[pivot_low_col]]
    pivot_high_idx = df.index[df[pivot_high_col]]

    # 1) Positive Regular (Price HL, Indicator LL)
    for i in range(1, len(pivot_low_idx)):
        curr = pivot_low_idx[i]
        prev = pivot_low_idx[i-1]
        # skip if date difference is too big
        if (curr - prev).days > lookback_bars:
            continue
        price_prev, price_curr = df.loc[prev, "Close"], df.loc[curr, "Close"]
        indi_prev, indi_curr = indicator.loc[prev], indicator.loc[curr]
        # Price forms higher low, indicator forms lower low
        if price_curr > price_prev and indi_curr < indi_prev:
            signals["positive_regular"][df.index.get_loc(curr)] = True

    # 2) Negative Regular (Price LH, Indicator HH)
    for i in range(1, len(pivot_high_idx)):
        curr = pivot_high_idx[i]
        prev = pivot_high_idx[i-1]
        if (curr - prev).days > lookback_bars:
            continue
        price_prev, price_curr = df.loc[prev, "Close"], df.loc[curr, "Close"]
        indi_prev, indi_curr = indicator.loc[prev], indicator.loc[curr]
        # Price forms lower high, indicator forms higher high
        if price_curr < price_prev and indi_curr > indi_prev:
            signals["negative_regular"][df.index.get_loc(curr)] = True

    # 3) Positive Hidden (Price LL, Indicator HL)
    for i in range(1, len(pivot_low_idx)):
        curr = pivot_low_idx[i]
        prev = pivot_low_idx[i-1]
        if (curr - prev).days > lookback_bars:
            continue
        price_prev, price_curr = df.loc[prev, "Close"], df.loc[curr, "Close"]
        indi_prev, indi_curr = indicator.loc[prev], indicator.loc[curr]
        # Price forms lower low, indicator forms higher low
        if price_curr < price_prev and indi_curr > indi_prev:
            signals["positive_hidden"][df.index.get_loc(curr)] = True

    # 4) Negative Hidden (Price HH, Indicator LH)
    for i in range(1, len(pivot_high_idx)):
        curr = pivot_high_idx[i]
        prev = pivot_high_idx[i-1]
        if (curr - prev).days > lookback_bars:
            continue
        price_prev, price_curr = df.loc[prev, "Close"], df.loc[curr, "Close"]
        indi_prev, indi_curr = indicator.loc[prev], indicator.loc[curr]
        # Price forms higher high, indicator forms lower high
        if price_curr > price_prev and indi_curr < indi_prev:
            signals["negative_hidden"][df.index.get_loc(curr)] = True

    return signals

# -----------------------------------
# 5) Stock scanning function
# -----------------------------------
def screen_for_divergences(ticker, period, interval, prd, showlimit, lookback_bars, recent_days):
    """
    1. Download data
    2. Compute pivot highs/lows
    3. Compute the "many" indicators
    4. Detect divergences for each indicator
    5. Summarize how many divergences occurred in the last `recent_days` bars
    6. Return a dict of results or None if no data
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or len(df) == 0:
        return None
    df.dropna(inplace=True)

    # Add pivot columns
    df["pivot_high"] = pivot_high(df["High"], prd, prd)
    df["pivot_low"]  = pivot_low(df["Low"],  prd, prd)

    # Compute all indicators
    inds = compute_indicators(df)

    # We'll store each (indicator -> signals) in a big DataFrame
    all_signals_df = pd.DataFrame(index=df.index)
    for name, indi_series in inds.items():
        if indi_series.isnull().all():
            # skip if indicator is all NaN
            continue

        div_signals = detect_divergences(
            df=df,
            indicator=indi_series,
            pivot_high_col="pivot_high",
            pivot_low_col="pivot_low",
            lookback_bars=lookback_bars
        )
        # Each of the 4 signal types => columns
        for sig_name, arr in div_signals.items():
            col = f"{name}_{sig_name}"
            all_signals_df[col] = arr

    if len(all_signals_df) == 0:
        return None

    # For each bar, count how many divergences (True) across all columns
    total_div_each_bar = all_signals_df.sum(axis=1)

    # Focus on the last N bars
    last_n = total_div_each_bar.tail(recent_days)
    total_in_last_n_days = last_n.sum()

    # If total >= showlimit, we say "recent_divergence = True"
    recent_divergence = (total_in_last_n_days >= showlimit)

    return {
        "ticker": ticker,
        "data_points": len(df),
        "last_bar_date": df.index[-1],
        "divergences_in_last_n_days": int(total_in_last_n_days),
        "recent_divergence": recent_divergence,
    }

# -----------------------------------
# 6) Streamlit App
# -----------------------------------
def main():
    st.title("Divergence Screener (Adapted from PineScript ‘Divergence for Many Indicators v4’)")
    st.write("""
    This app scans a list of US stocks (S&P 500 for demo) to find 
    stocks with divergences (regular/hidden, positive/negative) 
    on multiple indicators in the last N bars.
    """)

    # Sidebar inputs
    period = st.sidebar.selectbox("Yahoo Finance Period", ["3mo","6mo","1y","2y","5y"], index=1)
    interval = st.sidebar.selectbox("Data Interval", ["1d","1h"], index=0)
    prd = st.sidebar.slider("Pivot Period (prd)", 1, 10, 5)
    lookback_bars = st.sidebar.slider("Max pivot spacing (days)", 10, 365, 100, step=5)
    recent_days = st.sidebar.slider("Look for divergences in last N bars", 1, 20, 5)
    showlimit = st.sidebar.slider("Min # divergences to flag signal", 1, 5, 1)

    if st.button("Run Screener"):
        with st.spinner("Fetching ticker list..."):
            tickers = get_sp500_tickers()  # or your own list of US stocks

        results = []
        with st.spinner("Scanning tickers... please wait."):
            for i, ticker in enumerate(tickers):
                try:
                    out = screen_for_divergences(
                        ticker=ticker,
                        period=period,
                        interval=interval,
                        prd=prd,
                        showlimit=showlimit,
                        lookback_bars=lookback_bars,
                        recent_days=recent_days
                    )
                    if out is not None:
                        # only keep if there's a "recent_divergence"
                        if out["recent_divergence"]:
                            results.append(out)
                except Exception as e:
                    # optionally log or print e
                    pass

        if results:
            df_res = pd.DataFrame(results)
            # sort by # of divergences
            df_res.sort_values("divergences_in_last_n_days", ascending=False, inplace=True)
            st.write(f"Found {len(df_res)} stocks with >={showlimit} divergences in last {recent_days} bars.")
            st.dataframe(df_res)
        else:
            st.write(f"No stocks found with divergences in the last {recent_days} bars.")

if __name__ == "__main__":
    main()
