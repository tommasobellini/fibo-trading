import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import io

############################
# Funzioni di supporto
############################

@st.cache_data
def get_sp500_companies():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    return df

@st.cache_data
def get_market_caps_bulk(tickers):
    joined_tickers = " ".join(tickers)
    bulk_obj = yf.Tickers(joined_tickers)
    caps = {}
    for t in tickers:
        try:
            info = bulk_obj.tickers[t].info
            caps[t] = info.get('marketCap', 0)
        except Exception:
            caps[t] = 0
    return caps

def filter_stocks_by_market_cap(tickers, min_market_cap=10_000_000_000):
    caps_dict = get_market_caps_bulk(tickers)
    filtered = [t for t in tickers if caps_dict.get(t, 0) >= min_market_cap]
    return filtered

def find_previous_swing_points(df, exclude_days=5):
    if len(df) < exclude_days + 2:
        return None, None
    df_past = df.iloc[:-exclude_days]
    if df_past.empty:
        return None, None
    max_price = df_past['High'].max()
    min_price = df_past['Low'].min()
    return max_price, min_price

def compute_fibonacci_levels(max_price, min_price):
    if max_price is None or min_price is None:
        return {}
    diff = max_price - min_price
    levels = {
        '0%': max_price,
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50%': max_price - 0.50  * diff,
        '61.8%': max_price - 0.618 * diff,
        '78.6%': max_price - 0.786 * diff,
        '100%': min_price
    }
    return levels

def is_near_level(current_price, target_price, tolerance=0.01):
    # Se target_price fosse 0, evitiamo divisione per zero
    if target_price == 0:
        return False
    return abs(current_price - target_price) / abs(target_price) <= tolerance

############################
# Indicatori classici
############################
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1+rs))
    return rsi

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d

def compute_macd(close, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = close.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = close.ewm(span=slowperiod, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

############################
# Dynamic Price Oscillator (Zeiierman)
############################
def compute_dynamic_price_oscillator(df, length=33, smooth_factor=5):
    """
    Calcola il Dynamic Price Oscillator (Zeiierman) + Bollinger Bands:
      - bbHigh, bbLow (dev=1)
      - bbHighExp, bbLowExp (dev=2)
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    vol_adj_price = true_range.ewm(span=length, adjust=False).mean()

    # priceChange = close - close.shift(length)
    price_change = close - close.shift(length)
    # priceDelta = close - vol_adj_price
    price_delta = close - vol_adj_price

    # oscillator = EMA( avg(priceDelta, priceChange), smooth_factor )
    avg_series = (price_delta + price_change) / 2
    oscillator = avg_series.ewm(span=smooth_factor, adjust=False).mean()

    # Bollinger su oscillator
    bb_length = length * 5
    basis_1dev = oscillator.rolling(bb_length).mean()
    std_1dev   = oscillator.rolling(bb_length).std()

    bbHigh = basis_1dev + 1 * std_1dev
    bbLow  = basis_1dev - 1 * std_1dev

    bbHighExp = basis_1dev + 2 * std_1dev
    bbLowExp  = basis_1dev - 2 * std_1dev

    return oscillator, bbHigh, bbLow, bbHighExp, bbLowExp

############################
# Logica di oversold/overbought
############################
def passes_osc_conditions(
    rsi_val, stoch_k_val, macd_val, signal_val,
    check_rsi=True, check_stoch=True, check_macd=True
):
    """
    Restituisce True se il titolo passa TUTTI i filtri RSI, Stoch, MACD attivati.
    """
    # 1) RSI oversold: rsi < 30
    if check_rsi and rsi_val >= 30:
        return False

    # 2) Stoch oversold: stoch_k < 20
    if check_stoch and stoch_k_val >= 20:
        return False

    # 3) MACD oversold: macd < 0 e macd < signal
    if check_macd and (macd_val >= 0 or macd_val >= signal_val):
        return False

    return True

def passes_dpo_condition(
    dpo_val, bb_low, bb_high,
    dpo_mode="Nessuna"
):
    """
    dpo_mode: "Nessuna", "Oversold (DPO < BB Low)", "Overbought (DPO > BB High)"
    """
    # Se l'utente non vuole filtrare per DPO, passiamo tutto
    if dpo_mode == "Nessuna":
        return True
    # Evitiamo errori
    if dpo_val is None or pd.isna(dpo_val) or bb_low is None or pd.isna(bb_low) or bb_high is None or pd.isna(bb_high):
        return False

    if dpo_mode.startswith("Oversold") and dpo_val >= bb_low:
        return False
    if dpo_mode.startswith("Overbought") and dpo_val <= bb_high:
        return False
    return True

############################
# Funzione di entry & stop
############################
def define_trade_levels(current_price, min_price):
    """
    Entry Price = current_price
    Stop Price = 1% sotto min_price
    """
    entry_price = current_price
    stop_price = min_price * 0.99
    return round(entry_price, 2), round(stop_price, 2)

############################
# Screener principale
############################
def fibonacci_screener_entry_stop(
    tickers, 
    exclude_days=5, 
    check_rsi=True, 
    check_stoch=True, 
    check_macd=True,
    dpo_mode="Nessuna",
    dpo_length=33,
    dpo_smooth=5
):
    """
    1) Scarica ~3 mesi di dati orari
    2) Trova swing max/min precedenti (escludendo ultimi exclude_days)
    3) Se prezzo attuale è vicino al Fib 61,8% e oversold/overbought => definisce entry e stop
    """
    end = datetime.now()
    start = end - timedelta(days=90)

    data = yf.download(
        tickers, 
        start=start, 
        end=end, 
        interval='1h',
        group_by="ticker"
    )

    single_ticker = (len(tickers) == 1)
    results = []

    for ticker in tickers:
        try:
            if single_ticker:
                df_ticker = data
            else:
                if ticker not in data.columns.levels[0]:
                    continue
                df_ticker = data[ticker]

            if df_ticker.empty:
                continue
            
            max_price, min_price = find_previous_swing_points(df_ticker, exclude_days=exclude_days)
            if max_price is None or min_price is None:
                continue
            fib_levels = compute_fibonacci_levels(max_price, min_price)
            if not fib_levels:
                continue

            current_price = df_ticker['Close'].iloc[-1]

            # RSI, Stoch, MACD
            close_series = df_ticker['Close']
            high_series  = df_ticker['High']
            low_series   = df_ticker['Low']

            rsi_series = compute_rsi(close_series, period=14)
            if rsi_series.dropna().empty:
                continue
            rsi_val = rsi_series.iloc[-1]

            stoch_k, stoch_d = compute_stochastic(high_series, low_series, close_series)
            if stoch_k.dropna().empty:
                continue
            stoch_k_val = stoch_k.iloc[-1]

            macd_line, signal_line, _ = compute_macd(close_series)
            if macd_line.dropna().empty:
                continue
            macd_val   = macd_line.iloc[-1]
            signal_val = signal_line.iloc[-1]

            # DPO
            dpo_val = None
            dpo_bb_low = None
            dpo_bb_high = None

            if dpo_mode != "Nessuna":
                oscillator, bbHigh, bbLow, bbHighExp, bbLowExp = compute_dynamic_price_oscillator(
                    df_ticker, length=dpo_length, smooth_factor=dpo_smooth
                )
                # Evitiamo situazioni in cui la rolling sia troppo corta
                if oscillator.dropna().empty or bbLow.dropna().empty or bbHigh.dropna().empty:
                    continue
                dpo_val      = oscillator.iloc[-1]
                dpo_bb_low   = bbLow.iloc[-1]
                dpo_bb_high  = bbHigh.iloc[-1]

            # Controllo Fib 61.8%
            fib_618 = fib_levels['61.8%']
            # Vicinanza ±1%
            if is_near_level(current_price, fib_618, tolerance=0.01):

                # Calcoliamo se passa i filtri RSI, Stoch, MACD
                pass_osc = passes_osc_conditions(
                    rsi_val, stoch_k_val, macd_val, signal_val,
                    check_rsi, check_stoch, check_macd
                )
                if not pass_osc:
                    continue

                # Calcoliamo se passa il filtro DPO (Oversold / Overbought)
                pass_dpo = passes_dpo_condition(
                    dpo_val, dpo_bb_low, dpo_bb_high,
                    dpo_mode
                )
                if not pass_dpo:
                    continue

                # Se tutti i filtri passano => definisci entry e stop
                if current_price >= fib_618:
                    fib_trend = "Bullish"
                else:
                    fib_trend = "Bearish"

                entry_price, stop_price = define_trade_levels(current_price, min_price)

                results.append({
                    'Ticker': ticker,
                    'Max Price (prev.)': round(max_price, 2),
                    'Min Price (prev.)': round(min_price, 2),
                    'Fib 61.8%': round(fib_618, 2),
                    'Current Price': round(current_price, 2),
                    'Fib Trend': fib_trend,
                    'RSI': round(rsi_val, 2),
                    'StochK': round(stoch_k_val, 2),
                    'MACD': round(macd_val, 2),
                    'Signal': round(signal_val, 2),
                    'DPO': round(dpo_val, 2) if dpo_val is not None else None,
                    'DPO BB Low':  round(dpo_bb_low, 2) if dpo_bb_low is not None else None,
                    'DPO BB High': round(dpo_bb_high, 2) if dpo_bb_high is not None else None,
                    'Entry Price': entry_price,
                    'Stop Price': stop_price
                })

        except Exception as e:
            st.warning(f"Errore su {ticker}: {e}")
            continue

    return pd.DataFrame(results)

############################
# App Streamlit
############################
def main():
    st.title("Fibonacci Screener + RSI/Stoch/MACD + DPO (oversold/overbought)")

    # Parametri capitalizzazione e giorni di swing
    min_market_cap = st.number_input("Min Market Cap", value=10_000_000_000, step=1_000_000_000)
    exclude_days = st.number_input("Escludi ultimi N giorni (swing precedenti)", value=5, step=1)
    
    # Flag oversold classici
    rsi_flag = st.checkbox("RSI < 30 (Oversold)", value=True)
    stoch_flag = st.checkbox("Stocastico < 20 (Oversold)", value=True)
    macd_flag = st.checkbox("MACD < 0 e MACD < Signal (Oversold)", value=True)

    # Filtro DPO: Nessuno, Oversold, Overbought
    dpo_mode = st.selectbox(
        "DPO Condition",
        ["Nessuna", "Oversold (DPO < BB Low)", "Overbought (DPO > BB High)"],
        index=0
    )
    dpo_length_val = st.number_input("DPO Length (default=33)", value=33, step=1)
    dpo_smooth_val = st.number_input("DPO Smoothing Factor (default=5)", value=5, step=1)

    if st.button("Esegui Screener"):
        sp500_df = get_sp500_companies()
        if sp500_df.empty:
            st.error("Impossibile scaricare la lista S&P 500.")
            return
        
        tickers = sp500_df['Symbol'].tolist()
        # In Yahoo Finance, i ticker con punto vanno sostituiti col trattino
        tickers = [t.replace('.', '-') for t in tickers]

        filtered = filter_stocks_by_market_cap(tickers, min_market_cap)
        st.write("Numero di ticker dopo il filtro Market Cap:", len(filtered))
        if not filtered:
            st.warning("Nessun ticker supera la capitalizzazione richiesta.")
            return

        df_results = fibonacci_screener_entry_stop(
            tickers=filtered, 
            exclude_days=exclude_days, 
            check_rsi=rsi_flag, 
            check_stoch=stoch_flag, 
            check_macd=macd_flag,
            dpo_mode=dpo_mode,
            dpo_length=dpo_length_val,
            dpo_smooth=dpo_smooth_val
        )

        if df_results.empty:
            st.info("Nessun titolo rispetta i criteri selezionati.")
        else:
            st.success(f"Trovati {len(df_results)} titoli. Mostriamo entry e stop proposti:")
            st.dataframe(df_results)

            # Export
            watchlist = df_results['Ticker'].tolist()
            csv_buffer = io.StringIO()
            for t in watchlist:
                csv_buffer.write(t + "\n")
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="Scarica Tickers",
                data=csv_data,
                file_name="my_watchlist_fib.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
