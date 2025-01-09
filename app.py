import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import ta

# --------------------------------
# 1) Fetch S&P 500 Tickers from Wikipedia
# --------------------------------
@st.cache_data
def get_sp500_tickers():
    """
    Fetch the list of S&P 500 companies from Wikipedia and return a list of tickers.
    Replaces '.' with '-' (e.g. BRK.B -> BRK-B) for Yahoo Finance.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    tables = pd.read_html(html)
    df_sp500 = tables[0]
    tickers = df_sp500['Symbol'].unique().tolist()
    cleaned_tickers = []
    for t in tickers:
        if "." in t:
            t = t.replace(".", "-")
        cleaned_tickers.append(t)
    return cleaned_tickers

# --------------------------------
# 2) Pivot High / Low
# --------------------------------
def pivot_high(series, left_bars, right_bars):
    """
    True at index i if 'series[i]' is greater than 'series[i +/- k]' for k in [1..left_bars]
    and [1..right_bars].
    """
    cond = pd.Series([True] * len(series), index=series.index)
    for i in range(1, left_bars + 1):
        cond = cond & (series > series.shift(i)).fillna(False)
    for j in range(1, right_bars + 1):
        cond = cond & (series > series.shift(-j)).fillna(False)
    return cond.fillna(False)

def pivot_low(series, left_bars, right_bars):
    """
    True at index i if 'series[i]' is lower than 'series[i +/- k]' for k in [1..left_bars]
    and [1..right_bars].
    """
    cond = pd.Series([True] * len(series), index=series.index)
    for i in range(1, left_bars + 1):
        cond = cond & (series < series.shift(i)).fillna(False)
    for j in range(1, right_bars + 1):
        cond = cond & (series < series.shift(-j)).fillna(False)
    return cond.fillna(False)

# --------------------------------
# 3) Compute Indicators
# --------------------------------
def compute_indicators(df):
    """
    Return a dict of Series: RSI, MACD line, MACD histogram, Stoch, CCI, Momentum, OBV, CMF, MFI.
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
    cci_20 = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)

    # Momentum(10) = Close - Close.shift(10)
    momentum_10 = df['Close'] - df['Close'].shift(10)

    # OBV
    obv_series = ta.volume.on_balance_volume(df['Close'], df['Volume'])

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

# --------------------------------
# 4) Detect Divergences
# --------------------------------
def detect_divergences(df, indicator, pivot_high_col, pivot_low_col, lookback_bars=100):
    """
    Checks for:
      - Positive Regular   (price HL, indicator LL)
      - Negative Regular   (price LH, indicator HH)
      - Positive Hidden    (price LL, indicator HL)
      - Negative Hidden    (price HH, indicator LH)
    Returns dict of arrays, e.g. signals['positive_regular'] = [bool,...].
    """
    signals = {
        'positive_regular': np.zeros(len(df), dtype=bool),
        'negative_regular': np.zeros(len(df), dtype=bool),
        'positive_hidden':  np.zeros(len(df), dtype=bool),
        'negative_hidden':  np.zeros(len(df), dtype=bool),
    }

    pivot_low_idx  = df.index[df[pivot_low_col]]
    pivot_high_idx = df.index[df[pivot_high_col]]

    # 1) Positive Regular Divergence
    #    Price: higher low, Indicator: lower low
    for i in range(1, len(pivot_low_idx)):
        curr = pivot_low_idx[i]
        prev = pivot_low_idx[i - 1]
        # If they're too far apart in time, skip
        if (curr - prev).days > lookback_bars:
            continue
        price_prev, price_curr = df.loc[prev, 'Close'], df.loc[curr, 'Close']
        indi_prev, indi_curr = indicator.loc[prev], indicator.loc[curr]
        if price_curr > price_prev and indi_curr < indi_prev:
            signals['positive_regular'][df.index.get_loc(curr)] = True

    # 2) Negative Regular Divergence
    #    Price: lower high, Indicator: higher high
    for i in range(1, len(pivot_high_idx)):
        curr = pivot_high_idx[i]
        prev = pivot_high_idx[i - 1]
        if (curr - prev).days > lookback_bars:
            continue
        price_prev, price_curr = df.loc[prev, 'Close'], df.loc[curr, 'Close']
        indi_prev, indi_curr = indicator.loc[prev], indicator.loc[curr]
        if price_curr < price_prev and indi_curr > indi_prev:
            signals['negative_regular'][df.index.get_loc(curr)] = True

    # 3) Positive Hidden Divergence
    #    Price: lower low, Indicator: higher low
    for i in range(1, len(pivot_low_idx)):
        curr = pivot_low_idx[i]
        prev = pivot_low_idx[i - 1]
        if (curr - prev).days > lookback_bars:
            continue
        price_prev, price_curr = df.loc[prev, 'Close'], df.loc[curr, 'Close']
        indi_prev, indi_curr = indicator.loc[prev], indicator.loc[curr]
        if price_curr < price_prev and indi_curr > indi_prev:
            signals['positive_hidden'][df.index.get_loc(curr)] = True

    # 4) Negative Hidden Divergence
    #    Price: higher high, Indicator: lower high
    for i in range(1, len(pivot_high_idx)):
        curr = pivot_high_idx[i]
        prev = pivot_high_idx[i - 1]
        if (curr - prev).days > lookback_bars:
            continue
        price_prev, price_curr = df.loc[prev, 'Close'], df.loc[curr, 'Close']
        indi_prev, indi_curr = indicator.loc[prev], indicator.loc[curr]
        if price_curr > price_prev and indi_curr < indi_prev:
            signals['negative_hidden'][df.index.get_loc(curr)] = True

    return signals

# --------------------------------
# 5) Screening Function
# --------------------------------
def screen_for_divergences(
    ticker, period="6mo", interval="1d", 
    prd=5, showlimit=1, lookback_bars=100, 
    recent_days=5
):
    """
    1. Download data (last X months, etc.)
    2. Compute pivot highs & lows
    3. Compute indicators
    4. Detect divergences across entire data
    5. We only care if there's at least ONE divergence in the last `recent_days` bars.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or len(df) == 0:
        return None
    df.dropna(inplace=True)

    # Ensure needed columns exist
    if any(col not in df.columns for col in ["Open","High","Low","Close","Volume"]):
        return None

    # Calculate pivot columns
    df["pivot_high"] = pivot_high(df["High"], prd, prd)
    df["pivot_low"]  = pivot_low(df["Low"],  prd, prd)

    # Calculate indicators
    inds = compute_indicators(df)

    # Gather all signals
    all_signals_df = pd.DataFrame(index=df.index)
    for name, indi_series in inds.items():
        # skip if the entire indicator is NaN
        if indi_series.isnull().all():
            continue

        signals = detect_divergences(
            df=df,
            indicator=indi_series,
            pivot_high_col="pivot_high",
            pivot_low_col="pivot_low",
            lookback_bars=lookback_bars
        )
        # Add them as columns
        for sig_name, arr in signals.items():
            col = f"{name}_{sig_name}"
            all_signals_df[col] = arr

    if len(all_signals_df) == 0:
        return None

    # Count total divergences for each bar
    #   (Summation across all columns in all_signals_df.)
    total_div_each_bar = all_signals_df.sum(axis=1)

    # Focus only on the last `recent_days` bars
    last_n = total_div_each_bar.tail(recent_days)
    # If ANY divergence in that window, we consider it "recent_divergence"
    total_in_last_n_days = last_n.sum()
    recent_divergence = (total_in_last_n_days >= showlimit)

    # Return summary
    return {
        "ticker": ticker,
        "data_points": len(df),
        "last_bar_date": df.index[-1],
        "divergences_in_last_n_days": int(total_in_last_n_days),
        "recent_divergence": recent_divergence,
    }

# --------------------------------
# 6) Streamlit App
# --------------------------------
def main():
    st.title("S&P 500 Screener for **Recent** Divergences")
    st.write("""
    This app fetches the S&P 500 constituents from Wikipedia, downloads historical 
    data from Yahoo Finance, and **only shows** those symbols where a divergence 
    (regular or hidden) occurred in the **last N days**.
    """)

    # Sidebar controls
    period = st.sidebar.selectbox(
        "Yahoo Finance Period", 
        ["3mo","6mo","1y","2y","5y"], 
        index=1
    )
    interval = st.sidebar.selectbox(
        "Data Interval", 
        ["1d","1h","15m"], 
        index=0
    )
    prd = st.sidebar.slider(
        "Pivot Period (prd)", 
        min_value=2, 
        max_value=10, 
        value=5, 
        step=1
    )
    showlimit = st.sidebar.slider(
        "Minimum number of divergences to flag signal in that timeframe", 
        min_value=1, 
        max_value=5, 
        value=1
    )
    lookback_bars = st.sidebar.slider(
        "Max pivot spacing in days (lookback_bars)", 
        min_value=30, 
        max_value=200, 
        value=100, 
        step=10
    )
    recent_days = st.sidebar.slider(
        "Look for divergences in the last N bars", 
        min_value=1, 
        max_value=30, 
        value=5
    )

    if st.button("Run Screener"):
        with st.spinner("Scraping S&P 500..."):
            tickers = get_sp500_tickers()

        results = []
        with st.spinner("Analyzing tickers..."):
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
                        # Only add to results if there's a "recent_divergence"
                        if out["recent_divergence"]:
                            results.append(out)
                except Exception as e:
                    pass  # or log the error if you want

        if results:
            df_res = pd.DataFrame(results)
            # Sort by how many divergences in the last N days
            df_res.sort_values("divergences_in_last_n_days", ascending=False, inplace=True)
            st.write(f"Found {len(results)} stocks with divergences in the last {recent_days} bars.")
            st.dataframe(df_res)
        else:
            st.write(f"No stocks found with divergences in the last {recent_days} bars.")


if __name__ == "__main__":
    main()
