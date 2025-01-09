import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import ta

# -----------------------------
# 1) Ticker Universe
# -----------------------------
# Sample subset of S&P 500 (you can replace with the full list)
SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "JNJ", "XOM", "V", "NVDA", "JPM",  # ... etc.
]


# -----------------------------
# 2) Pivot Calculation
# -----------------------------
def pivot_high(series, left_bars, right_bars):
    """
    Identify pivot highs where 'series' is greater than the left_bars bars to the left
    and greater than the right_bars bars to the right.
    Return a boolean Series of the same length as 'series'.
    """
    cond = pd.Series([True] * len(series), index=series.index)  # Start as all True, refine below

    # Compare to left side
    for i in range(1, left_bars + 1):
        cond_left = series > series.shift(i)
        cond = cond & cond_left.fillna(False)

    # Compare to right side
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

    # Compare to left side
    for i in range(1, left_bars + 1):
        cond_left = series < series.shift(i)
        cond = cond & cond_left.fillna(False)

    # Compare to right side
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
    macd_signal = macd_inst.macd_signal() # signal line
    macd_hist = macd_inst.macd_diff()     # histogram

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

    # Momentum (10) - simple approach is close.diff(10)
    # or we can replicate the "mom" logic from Pine (Close - Close[n])
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

    # For demonstration, we'll skip VWmacd replication. You can add it if needed.
    # Return a dictionary of indicator series
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
    Attempt to replicate the logic for:
      - Positive Regular Divergence (price forms higher low, indicator forms lower low)
      - Negative Regular Divergence (price forms lower high, indicator forms higher high)
      - Positive Hidden Divergence (price forms lower low, indicator forms higher low)
      - Negative Hidden Divergence (price forms higher high, indicator forms lower high)

    Return a dict of boolean arrays for each type:
      {
        'positive_regular': [bool, bool, ...],
        'negative_regular': [...],
        'positive_hidden':  [...],
        'negative_hidden':  [...],
      }

    Simplified approach: we only compare consecutive pivots. 
    For exact replication of Pine's slope checks & multi-pivot checks, 
    you must add more advanced logic. 
    """
    signals = {
        'positive_regular': np.zeros(len(df), dtype=bool),
        'negative_regular': np.zeros(len(df), dtype=bool),
        'positive_hidden':  np.zeros(len(df), dtype=bool),
        'negative_hidden':  np.zeros(len(df), dtype=bool),
    }

    # Identify pivot locations
    pivot_low_idx = df.index[df[pivot_low_col]]
    pivot_high_idx = df.index[df[pivot_high_col]]

    # 1) Positive Regular Divergence = 
    #    Price: Higher Low, Indicator: Lower Low
    for i in range(1, len(pivot_low_idx)):
        curr = pivot_low_idx[i]
        prev = pivot_low_idx[i-1]
        if (curr - prev).days > lookback_bars:
            continue

        # Price
        price_prev = df.loc[prev, 'Close']
        price_curr = df.loc[curr, 'Close']
        # Indicator
        indi_prev = indicator.loc[prev]
        indi_curr = indicator.loc[curr]

        # Price forms a higher low
        # Indicator forms a lower low
        if price_curr > price_prev and indi_curr < indi_prev:
            signals['positive_regular'][df.index.get_loc(curr)] = True

    # 2) Negative Regular Divergence = 
    #    Price: Lower High, Indicator: Higher High
    for i in range(1, len(pivot_high_idx)):
        curr = pivot_high_idx[i]
        prev = pivot_high_idx[i-1]
        if (curr - prev).days > lookback_bars:
            continue

        # Price
        price_prev = df.loc[prev, 'Close']
        price_curr = df.loc[curr, 'Close']
        # Indicator
        indi_prev = indicator.loc[prev]
        indi_curr = indicator.loc[curr]

        # Price forms a lower high
        # Indicator forms a higher high
        if price_curr < price_prev and indi_curr > indi_prev:
            signals['negative_regular'][df.index.get_loc(curr)] = True

    # 3) Positive Hidden Divergence = 
    #    Price: Lower Low, Indicator: Higher Low
    #    Typically we use pivot lows for that
    for i in range(1, len(pivot_low_idx)):
        curr = pivot_low_idx[i]
        prev = pivot_low_idx[i-1]
        if (curr - prev).days > lookback_bars:
            continue

        price_prev = df.loc[prev, 'Close']
        price_curr = df.loc[curr, 'Close']
        indi_prev = indicator.loc[prev]
        indi_curr = indicator.loc[curr]

        # Price forms a lower low
        # Indicator forms a higher low
        if price_curr < price_prev and indi_curr > indi_prev:
            signals['positive_hidden'][df.index.get_loc(curr)] = True

    # 4) Negative Hidden Divergence = 
    #    Price: Higher High, Indicator: Lower High
    #    Typically we use pivot highs for that
    for i in range(1, len(pivot_high_idx)):
        curr = pivot_high_idx[i]
        prev = pivot_high_idx[i-1]
        if (curr - prev).days > lookback_bars:
            continue

        price_prev = df.loc[prev, 'Close']
        price_curr = df.loc[curr, 'Close']
        indi_prev = indicator.loc[prev]
        indi_curr = indicator.loc[curr]

        # Price forms a higher high
        # Indicator forms a lower high
        if price_curr > price_prev and indi_curr < indi_prev:
            signals['negative_hidden'][df.index.get_loc(curr)] = True

    return signals


def screen_for_divergences(ticker, period="6mo", interval="1d", prd=5, showlimit=1):
    """
    1. Download data
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

    # Calculate pivot columns
    df["pivot_high"] = pivot_high(df["High"], prd, prd)
    df["pivot_low"] = pivot_low(df["Low"], prd, prd)

    # Compute indicators
    inds = compute_indicators(df)
    all_signals_df = pd.DataFrame(index=df.index)

    # For each indicator, detect divergences
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
        # Convert dict of signals to columns
        for sig_name, arr in signals.items():
            col = f"{name}_{sig_name}"
            all_signals_df[col] = arr

    # Now check how many divergences are present in the most recent bar
    if len(all_signals_df) == 0:
        return None

    last_bar_signals = all_signals_df.iloc[-1]
    total_divergences = last_bar_signals.sum()

    # If total_divergences < showlimit => "No signal" in PineScript style
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
# 5) Streamlit Front-End
# -----------------------------
def main():
    st.title("S&P 500 Divergence Screener (Regular + Hidden)")

    # Sidebar inputs
    period = st.sidebar.selectbox("Yahoo Finance Period:", ["3mo","6mo","1y","2y","5y"], index=1)
    interval = st.sidebar.selectbox("Data Interval:", ["1d","1h","15m"], index=0)
    prd = st.sidebar.slider("Pivot Period", min_value=2, max_value=10, value=5, step=1)
    showlimit = st.sidebar.slider("Minimum divergences to flag signal", min_value=1, max_value=5, value=1)

    if st.button("Run Screener"):
        results = []
        for ticker in SP500_TICKERS:
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
            st.dataframe(df_res)
        else:
            st.write("No results to display.")


if __name__ == "__main__":
    main()
