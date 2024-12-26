import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import io

##########################
# Funzioni di supporto
##########################

@st.cache_data
def get_sp500_companies(log_messages):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        log_messages.append("S&P 500 list fetched successfully from Wikipedia.")
        return df
    except Exception as e:
        msg = f"Error fetching S&P 500 companies: {e}"
        log_messages.append(msg)
        return pd.DataFrame()

@st.cache_data
def get_market_caps_bulk(tickers, log_messages):
    joined_tickers = " ".join(tickers)
    bulk_obj = yf.Tickers(joined_tickers)
    caps = {}
    for t in tickers:
        try:
            info = bulk_obj.tickers[t].info
            mcap = info.get('marketCap', 0)
            caps[t] = mcap
            log_messages.append(f"Ticker {t}, MarketCap={mcap}")
        except Exception as e:
            caps[t] = 0
            log_messages.append(f"Error fetching market cap for {t}: {e}")
    return caps

def filter_stocks_by_market_cap(tickers, min_market_cap, log_messages):
    caps_dict = get_market_caps_bulk(tickers, log_messages)
    filtered = []
    for t in tickers:
        if caps_dict.get(t, 0) >= min_market_cap:
            filtered.append(t)
            log_messages.append(f"Ticker {t} passed the MCAP filter.")
        else:
            log_messages.append(f"Ticker {t} did NOT pass the MCAP filter.")
    return filtered

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta>0, 0.0)
    loss = -delta.where(delta<0, 0.0)
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

def is_oversold(rsi_val, stoch_k_val, macd_val, signal_val,
                check_rsi=True, check_stoch=True, check_macd=True):
    if check_rsi and rsi_val >= 30:
        return False
    if check_stoch and stoch_k_val >= 20:
        return False
    if check_macd and (macd_val >= 0 or macd_val >= signal_val):
        return False
    return True

def define_trade_levels(current_price, swing_low, log_messages):
    """
    Semplice esempio:
    - Entry = current_price
    - Stop  = swing_low * 0.99
    """
    entry_price = current_price
    stop_price = swing_low * 0.99
    log_messages.append(
        f"Define trade levels -> Entry={round(entry_price,2)}, Stop={round(stop_price,2)}"
    )
    return round(entry_price, 2), round(stop_price, 2)

##########################
# Aggiungiamo una check_volume semplice
##########################
def check_volume_confirmation(df, ratio=1.2, lookback=20):
    if len(df) < lookback:
        return False
    avg_vol = df['Volume'].tail(lookback).mean()
    last_vol = df['Volume'].iloc[-1]
    return last_vol >= ratio * avg_vol

##########################
# Screener principale
##########################
def fibonacci_screener(
    tickers,
    min_market_cap,
    check_rsi,
    check_stoch,
    check_macd,
    check_volume,
    volume_ratio,
    volume_lookback,
    log_messages
):
    """
    Esempio semplificato di screener:
    - Scarica 3 mesi di dati
    - Prende max e min (ultimo mese)
    - Verifica se prezzo vicino a fib 61,8% e oversold
    - Definisce entry e stop
    - Aggiunge log
    """
    end = datetime.now()
    start = end - timedelta(days=90)  # 3 mesi
    results = []

    log_messages.append("Downloading historical data in bulk...")
    try:
        data = yf.download(tickers, start=start, end=end, group_by="ticker")
        log_messages.append("Data downloaded successfully.")
    except Exception as e:
        log_messages.append(f"Error downloading data: {e}")
        return pd.DataFrame()

    single_ticker = (len(tickers) == 1)

    for ticker in tickers:
        try:
            if single_ticker:
                df_ticker = data
            else:
                if ticker not in data.columns.levels[0]:
                    log_messages.append(f"{ticker} not in downloaded columns.")
                    continue
                df_ticker = data[ticker]

            if df_ticker.empty:
                log_messages.append(f"No data for {ticker}, skipping.")
                continue

            # Calcoliamo max e min (ultimo mese)
            df_last_month = df_ticker.iloc[-22:]  # ~22 trading days = ~1 month
            max_price = df_last_month['High'].max()
            min_price = df_last_month['Low'].min()
            diff = max_price - min_price
            fib_618 = max_price - 0.618 * diff

            current_price = df_ticker['Close'].iloc[-1]
            log_messages.append(f"{ticker} -> Max={round(max_price,2)}, Min={round(min_price,2)}, Fib618={round(fib_618,2)}, Current={round(current_price,2)}")

            # Check se vicino fib 61,8%
            tolerance = 0.01
            fib_diff = abs(current_price - fib_618)/abs(fib_618)*100  # in %
            log_messages.append(f"{ticker} -> Fib difference={round(fib_diff,2)}%")
            if fib_diff <= 1.0:  # entro 1%
                # Oversold?
                close_series = df_ticker['Close']
                high_series = df_ticker['High']
                low_series = df_ticker['Low']
                
                # RSI
                rsi_series = compute_rsi(close_series)
                if rsi_series.dropna().empty:
                    log_messages.append(f"{ticker} -> Not enough data for RSI.")
                    continue
                rsi_val = rsi_series.iloc[-1]
                
                # Stoc
                stoch_k, stoch_d = compute_stochastic(high_series, low_series, close_series)
                if stoch_k.dropna().empty:
                    log_messages.append(f"{ticker} -> Not enough data for Stoch.")
                    continue
                stoch_k_val = stoch_k.iloc[-1]
                
                # MACD
                macd_line, signal_line, _ = compute_macd(close_series)
                if macd_line.dropna().empty:
                    log_messages.append(f"{ticker} -> Not enough data for MACD.")
                    continue
                macd_val = macd_line.iloc[-1]
                signal_val = signal_line.iloc[-1]

                oversold_flag = is_oversold(rsi_val, stoch_k_val, macd_val, signal_val,
                                            check_rsi, check_stoch, check_macd)
                log_messages.append(
                    f"{ticker} -> RSI={round(rsi_val,2)}, StochK={round(stoch_k_val,2)}, MACD={round(macd_val,2)}, Signal={round(signal_val,2)}"
                )
                
                if oversold_flag:
                    log_messages.append(f"{ticker} -> Oversold confirmed.")
                    # Check volume se richiesto
                    volume_ok = True
                    if check_volume:
                        volume_ok = check_volume_confirmation(df_ticker, ratio=volume_ratio, lookback=volume_lookback)
                        log_messages.append(f"{ticker} -> Volume check = {volume_ok}")

                    if volume_ok:
                        # Definisci entry & stop
                        entry_price, stop_price = define_trade_levels(current_price, min_price, log_messages)
                        results.append({
                            'Ticker': ticker,
                            'Max Price': round(max_price, 2),
                            'Min Price': round(min_price, 2),
                            'Fib 61.8%': round(fib_618, 2),
                            'Current Price': round(current_price, 2),
                            'RSI': round(rsi_val,2),
                            'StochK': round(stoch_k_val,2),
                            'MACD': round(macd_val,2),
                            'Signal': round(signal_val,2),
                            'Entry': entry_price,
                            'Stop': stop_price
                        })
                else:
                    log_messages.append(f"{ticker} -> Not oversold, skip.")
            else:
                log_messages.append(f"{ticker} -> Not near Fib 61.8%. Skip.")

        except Exception as e:
            log_messages.append(f"Error processing {ticker}: {e}")
            continue

    return pd.DataFrame(results)

##########################
# WebApp main
##########################
def main():
    st.title("Fibonacci Screener + LOG")

    # Variabile per accumulare i log
    log_messages = []

    # Parametri
    min_mcap = st.number_input("Min Market Cap", value=10_000_000_000, step=1_000_000_000)
    check_rsi = st.checkbox("Check RSI < 30", value=True)
    check_stoch = st.checkbox("Check StochK < 20", value=True)
    check_macd = st.checkbox("Check MACD < 0 & < Signal", value=True)
    check_volume = st.checkbox("Volume Confirmation?", value=False)
    volume_ratio = st.number_input("Volume Ratio", value=1.2, step=0.1)
    volume_lookback = st.number_input("Volume Lookback Days", value=20, step=1)

    if st.button("Esegui Screener"):
        # 1) Recupera S&P 500
        df_sp500 = get_sp500_companies(log_messages)
        if df_sp500.empty:
            st.error("Impossibile scaricare la lista S&P 500.")
            st.write("### Log:")
            for msg in log_messages:
                st.write("-", msg)
            return
        
        tickers = df_sp500['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]

        # 2) Filtro su MCAP
        filtered = filter_stocks_by_market_cap(tickers, min_mcap, log_messages)
        log_messages.append(f"Filtered Tickers = {len(filtered)}")
        if not filtered:
            st.warning("Nessun ticker oltre il min market cap.")
            st.write("### Log:")
            for msg in log_messages:
                st.write("-", msg)
            return

        # 3) Screener fib
        df_results = fibonacci_screener(
            filtered, 
            min_mcap,
            check_rsi,
            check_stoch,
            check_macd,
            check_volume,
            volume_ratio,
            volume_lookback,
            log_messages
        )

        if df_results.empty:
            st.warning("Nessun titolo rispetta i criteri.")
        else:
            st.success(f"Trovati {len(df_results)} titoli:")
            st.dataframe(df_results)

        # Mostriamo i log in un riquadro
        st.write("### Log:")
        for msg in log_messages:
            st.write("-", msg)


if __name__ == "__main__":
    main()
