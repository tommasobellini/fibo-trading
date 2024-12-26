import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import io

# =====================================
#            FUNZIONI DI BASE
# =====================================
@st.cache_data
def get_sp500_companies():
    """
    Recupera la lista delle aziende S&P 500 da Wikipedia.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    return df

@st.cache_data
def get_market_caps_bulk(tickers):
    """
    Recupera in blocco i Market Cap dei ticker, usando yf.Tickers(...) invece
    di invocare singolarmente Ticker(t).info per ognuno.
    Restituisce un dizionario {ticker: market_cap}.
    """
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
    """
    Filtra i ticker in base alla capitalizzazione di mercato.
    """
    caps_dict = get_market_caps_bulk(tickers)
    filtered = [t for t in tickers if caps_dict.get(t, 0) >= min_market_cap]
    return filtered

# ======================
#   INDICATORI TECNICI
# ======================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
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

# ======================
#  FIBONACCI & OVERSOLD
# ======================
def fibonacci_level(df, retracement=0.618):
    """
    Calcola il livello di Fibonacci personalizzato
    (es. retracement=0.618 => 61,8%).
    """
    if df.empty:
        return None
    max_price = df['High'].max()
    min_price = df['Low'].min()
    diff = max_price - min_price
    level = max_price - retracement * diff
    return level

def is_near_price(current_price, target_price, tolerance=0.01):
    """
    Verifica se il prezzo corrente è entro la tolleranza di (1% default)
    rispetto al livello target.
    """
    if target_price == 0:
        return False
    return abs(current_price - target_price) / abs(target_price) <= tolerance

def meets_oversold_condition(
    rsi_latest, stoch_k_latest, macd_latest, signal_latest,
    check_rsi, check_stoch, check_macd
):
    """
    Applica i filtri di oversold solo se i flag (check_rsi, check_stoch, check_macd)
    sono True.
    - RSI oversold: rsi < 30
    - Stocastico oversold: stoch_k < 20
    - MACD oversold: macd < 0 e macd < signal
    Restituisce True se TUTTE le condizioni scelte sono soddisfatte.
    """
    if check_rsi and (rsi_latest >= 30):
        return False
    if check_stoch and (stoch_k_latest >= 20):
        return False
    if check_macd and (macd_latest >= 0 or macd_latest >= signal_latest):
        return False
    
    return True

# ======================
#     SCREENER IN BULK
# ======================
@st.cache_data
def download_data_in_bulk(tickers, start, end):
    """
    Scarica i dati storici in blocco. Se i ticker sono >200, li dividiamo in chunk.
    """
    CHUNK_SIZE = 100
    data_list = []
    
    for i in range(0, len(tickers), CHUNK_SIZE):
        subset = tickers[i:i+CHUNK_SIZE]
        try:
            temp = yf.download(subset, start=start, end=end, group_by="ticker")
            data_list.append(temp)
        except Exception as e:
            st.warning(f"Errore nello scaricare chunk: {e}")
    
    if not data_list:
        return pd.DataFrame()
    
    data = data_list[0]
    for extra_data in data_list[1:]:
        data = pd.concat([data, extra_data], axis=1)
    
    return data

def fibonacci_retracement_screener(
    tickers, retracement=0.618,
    check_rsi=True, check_stoch=True, check_macd=True
):
    """
    Screener:
    - Scarica i dati dell'ultimo mese
    - Calcola il livello di Fibonacci personalizzato
    - Verifica se current_price è vicino a tale livello
    - Calcola RSI, Stocastico, MACD
    - Verifica oversold su RSI, Stoch, MACD SOLO se selezionati
    """
    end = datetime.now()
    start = end - timedelta(days=30)

    data = download_data_in_bulk(tickers, start, end)
    if data.empty:
        return pd.DataFrame()

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

            current_price = df_ticker['Close'].iloc[-1]
            fib_lv = fibonacci_level(df_ticker, retracement)
            if fib_lv is None:
                continue

            near_fib = is_near_price(current_price, fib_lv, tolerance=0.01)

            close_series = df_ticker['Close']
            high_series  = df_ticker['High']
            low_series   = df_ticker['Low']

            # RSI
            rsi_series = compute_rsi(close_series, period=14)
            if rsi_series.dropna().empty:
                continue
            rsi_latest = rsi_series.iloc[-1]

            # Stocastico
            stoch_k, stoch_d = compute_stochastic(high_series, low_series, close_series)
            if stoch_k.dropna().empty:
                continue
            stoch_k_latest = stoch_k.iloc[-1]

            # MACD
            macd_line, signal_line, _ = compute_macd(close_series)
            if macd_line.dropna().empty:
                continue
            macd_latest = macd_line.iloc[-1]
            signal_latest = signal_line.iloc[-1]

            # Oversold condition
            oversold = meets_oversold_condition(
                rsi_latest, stoch_k_latest, macd_latest, signal_latest,
                check_rsi, check_stoch, check_macd
            )

            if near_fib and oversold:
                results.append({
                    'Ticker': ticker,
                    'Current Price': round(current_price, 2),
                    f'Fib {int(retracement*100)}%': round(fib_lv, 2),
                    'RSI': round(rsi_latest, 2),
                    'StochK': round(stoch_k_latest, 2),
                    'MACD': round(macd_latest, 2),
                    'Signal': round(signal_latest, 2),
                })

        except Exception as e:
            st.warning(f"Errore su {ticker}: {e}")
    
    return pd.DataFrame(results)

# ======================
#         STREAMLIT
# ======================
def main():
    st.title("Fibonacci Screener + Oversold Filters + Export to TradingView Watchlist")
    st.write("**Ottimizzato per ridurre le richieste a yfinance**")

    # Parametri input
    min_market_cap = st.number_input("Minimum Market Cap (USD)", value=10_000_000_000, step=1_000_000_000)
    
    fib_input = st.number_input(
        "Fibonacci Retracement (0.0 - 1.0)", 
        value=0.618, 
        min_value=0.0, 
        max_value=1.0, 
        step=0.1
    )

    # Flag per oversold
    rsi_flag = st.checkbox("RSI < 30", value=True)
    stoch_flag = st.checkbox("Stocastico K < 20", value=True)
    macd_flag = st.checkbox("MACD < 0 e MACD < Signal", value=True)

    if st.button("Esegui Screener"):
        st.write("1) Recupero lista S&P 500...")
        sp500_df = get_sp500_companies()
        if sp500_df.empty:
            st.error("Impossibile scaricare la lista S&P 500.")
            return
        
        # Pulizia ticker
        tickers = sp500_df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]  # yfinance usa i trattini

        st.write(f"2) Filtro per Market Cap >= {min_market_cap} ...")
        filtered_tickers = filter_stocks_by_market_cap(tickers, min_market_cap)
        st.write("Ticker trovati dopo il filtro:", len(filtered_tickers))
        if not filtered_tickers:
            st.warning("Nessun ticker supera la capitalizzazione richiesta.")
            return

        st.write("3) Calcolo Fibonacci + Indicatori (RSI, Stoch, MACD)...")
        results_df = fibonacci_retracement_screener(
            filtered_tickers,
            retracement=fib_input,
            check_rsi=rsi_flag,
            check_stoch=stoch_flag,
            check_macd=macd_flag
        )

        if results_df.empty:
            st.info("Nessun titolo soddisfa i criteri (Fib vicino + oversold richiesti).")
        else:
            st.success(f"Trovati {len(results_df)} titoli:")
            st.dataframe(results_df)

            # ==========================
            #    EXPORT TICKERS
            # ==========================
            st.write("### Esporta Tickers per TradingView")

            # 1) Creiamo la lista di ticker
            watchlist_tickers = results_df['Ticker'].tolist()

            # 2) Generiamo un CSV/txt in memoria
            csv_buffer = io.StringIO()
            # TradingView in genere riconosce un ticker per riga, ad es. "AAPL"
            for t in watchlist_tickers:
                csv_buffer.write(t + "\n")

            csv_data = csv_buffer.getvalue()

            # 3) Permettiamo di scaricarlo
            st.download_button(
                label="Scarica la Watchlist (txt)",
                data=csv_data,
                file_name="my_tradingview_watchlist.txt",
                mime="text/plain"
            )

            st.info("Dopo il download, su TradingView > Watchlist > Import, seleziona il file .txt con i ticker.")

if __name__ == "__main__":
    main()
