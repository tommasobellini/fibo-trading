import streamlit as st
import pandas as pd
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
    if target_price == 0:
        return False
    return abs(current_price - target_price) / abs(target_price) <= tolerance

############################
# Indicatori "oversold"
############################
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
    # RSI oversold: rsi < 30
    # Stoch oversold: stoch_k < 20
    # MACD oversold: macd < 0 e macd < signal
    if check_rsi and rsi_val >= 30:
        return False
    if check_stoch and stoch_k_val >= 20:
        return False
    if check_macd and (macd_val >= 0 or macd_val >= signal_val):
        return False
    return True

############################
# Funzione di "entry & stop"
############################
def define_trade_levels(current_price, min_price):
    """
    Esempio: 
      - Entry Price = current_price (ipotizziamo ingresso immediato)
      - Stop Price = 1% sotto il min_price (lo swing low precedente),
        così se il mercato rompe quel minimo, esce in stop.
    """
    entry_price = current_price
    stop_price = min_price * 0.99  # 1% sotto
    return round(entry_price, 2), round(stop_price, 2)

############################
# Screener principale
############################
def fibonacci_screener_entry_stop(
    tickers, exclude_days=5, 
    check_rsi=True, check_stoch=True, check_macd=True
):
    """
    1) Scarica 3 mesi di dati a 1h
    2) Trova swing max/min precedenti (escludendo ultimi exclude_days)
    3) Se prezzo attuale è vicino al fib 61,8% e oversold:
       - Definisce entry e stop
    """
    end = datetime.now()
    # Circa 90 giorni di storico orario (circa 3 mesi)
    start = end - timedelta(days=90)

    # <-- Aggiunto "interval='1h'" per avere dati orari
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
                # Verifica che il ticker sia effettivamente presente nei dati scaricati
                if ticker not in data.columns.levels[0]:
                    continue
                df_ticker = data[ticker]

            if df_ticker.empty:
                continue
            
            # Trova swing
            max_price, min_price = find_previous_swing_points(df_ticker, exclude_days=exclude_days)
            if max_price is None or min_price is None:
                continue
            fib_levels = compute_fibonacci_levels(max_price, min_price)
            if not fib_levels:
                continue

            current_price = df_ticker['Close'].iloc[-1]

            # Calcolo indicatori
            close_series = df_ticker['Close']
            high_series = df_ticker['High']
            low_series = df_ticker['Low']

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
            macd_val = macd_line.iloc[-1]
            signal_val = signal_line.iloc[-1]

            # Scegliamo di controllare solo 61.8% per esempio
            fib_618 = fib_levels['61.8%']
            # Se prezzo attuale è vicino al 61,8% e "oversold"
            if is_near_level(current_price, fib_618, tolerance=0.01):
                # Bullish/Bearish
                if current_price >= fib_618:
                    fib_trend = "Bullish"
                else:
                    fib_trend = "Bearish"

                # Check oversold
                if is_oversold(rsi_val, stoch_k_val, macd_val, signal_val,
                               check_rsi, check_stoch, check_macd):
                    # Definiamo Entry e Stop
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
                        # Nuove colonne
                        'Entry Price': entry_price,
                        'Stop Price': stop_price
                    })

        except Exception as e:
            st.warning(f"Errore su {ticker}: {e}")
            continue

    return pd.DataFrame(results)

############################
# App Streamlit di esempio
############################
def main():
    st.title("Fibonacci Screener + RSI/Stoch/MACD + Entry & Stop (1H Timeframe)")

    min_market_cap = st.number_input("Min Market Cap", value=10_000_000_000, step=1_000_000_000)
    exclude_days = st.number_input("Escludi ultimi N giorni (swing precedenti)", value=5, step=1)
    
    # Flag oversold
    rsi_flag = st.checkbox("RSI < 30", value=True)
    stoch_flag = st.checkbox("Stocastico < 20", value=True)
    macd_flag = st.checkbox("MACD < 0 e MACD < Signal", value=True)

    if st.button("Esegui Screener"):
        sp500_df = get_sp500_companies()
        if sp500_df.empty:
            st.error("Impossibile scaricare la lista S&P 500.")
            return
        
        tickers = sp500_df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]

        filtered = filter_stocks_by_market_cap(tickers, min_market_cap)
        st.write("Ticker dopo filtro:", len(filtered))
        if not filtered:
            st.warning("Nessun ticker supera la capitalizzazione richiesta.")
            return

        df_results = fibonacci_screener_entry_stop(
            filtered, 
            exclude_days=exclude_days, 
            check_rsi=rsi_flag, 
            check_stoch=stoch_flag, 
            check_macd=macd_flag
        )

        if df_results.empty:
            st.info("Nessun titolo rispetta i criteri (vicino a Fib 61.8%, oversold).")
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
