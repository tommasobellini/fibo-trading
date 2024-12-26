import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ======================
#  FUNZIONI DI SUPPORTO
# ======================

def get_sp500_companies():
    """
    Fetch the list of S&P 500 companies from Wikipedia.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]  # La prima tabella contiene la lista delle aziende S&P 500
        return df
    except Exception as e:
        print(f"Error fetching S&P 500 companies: {e}")
        return pd.DataFrame()

def get_market_caps(tickers):
    """
    Restituisce un dizionario {ticker: market_cap} per i ticker specificati,
    recuperando l'informazione da yfinance.
    """
    caps = {}
    for ticker in tickers:
        try:
            ticker_data = yf.Ticker(ticker)
            market_cap = ticker_data.info.get('marketCap', 0)
            caps[ticker] = market_cap
        except Exception as e:
            print(f"Error fetching market cap for {ticker}: {e}")
            caps[ticker] = 0
    return caps

def filter_stocks_by_market_cap(tickers, min_market_cap=10_000_000_000):
    """
    Filtra i ticker in base alla capitalizzazione di mercato minima.
    """
    # Recupera i market cap in batch
    caps_dict = get_market_caps(tickers)

    filtered_tickers = []
    for ticker in tickers:
        if caps_dict.get(ticker, 0) >= min_market_cap:
            filtered_tickers.append(ticker)
    return filtered_tickers

def fibonacci_618_level(df):
    """
    Calcola il livello di Fibonacci 61,8% tra il massimo e il minimo dell'ultimo mese.
    Restituisce il valore corrispondente.
    """
    if df.empty:
        return None
    max_price = df['High'].max()
    min_price = df['Low'].min()
    diff = max_price - min_price

    # 61,8% level (ritracciamento)
    level_618 = max_price - 0.618 * diff
    return level_618

def is_near_price(current_price, target_price, tolerance=0.01):
    """
    Verifica se current_price è entro la tolleranza (in percentuale) rispetto al valore target_price.
    Esempio di default: 1% di tolleranza.
    """
    if target_price == 0:
        return False
    return abs(current_price - target_price) / abs(target_price) <= tolerance

# ======================
#    INDICATORI TECNICI
# ======================

def compute_rsi(series, period=14):
    """
    Calcolo RSI (Relative Strength Index) manuale,
    per scopi dimostrativi. Restituisce una Serie contenente i valori di RSI.
    
    Formula semplificata:
    1) Calcoliamo la differenza tra i prezzi consecutivi
    2) Ripartiamo le differenze in 'guadagni' (positive) e 'perdite' (negative) 
    3) Calcoliamo la media esponenziale dei guadagni (avg_gain) e delle perdite (avg_loss)
    4) RSI = 100 - (100 / (1 + (avg_gain/avg_loss)))
    """
    delta = series.diff()
    # Guadagni (up) e perdite (down)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Usiamo EMA per calcolare le medie esponenziali
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    """
    Calcolo manuale dello Stocastico (valore %K e %D).
    %K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)
    %D = media mobile semplice di K su d_period
    
    Ritorna due Serie: stochK, stochD
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    stochK = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stochD = stochK.rolling(d_period).mean()
    return stochK, stochD

def compute_macd(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Calcolo MACD base:
    - MACD = EMA(fastperiod) - EMA(slowperiod)
    - Signal = EMA(MACD, signalperiod)
    - Histogram = MACD - Signal
    """
    # EMA veloci/lente
    ema_fast = close.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = close.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# ======================
#       SCREENER
# ======================

def fibonacci_retracement_screener(tickers):
    """
    Scarica i dati per l'ultimo mese di tutti i ticker,
    calcola il livello di Fibonacci 61,8% e verifica se il prezzo
    corrente è vicino a tale livello. Inoltre, calcola RSI, Stocastico e MACD
    e verifica se indicano condizioni di oversold (in modo basilare).
    
    Restituisce un DataFrame con i risultati.
    """
    results = []

    # Definiamo la finestra temporale per l'ultimo mese
    end = datetime.now()
    start = end - timedelta(days=30)

    # Scarichiamo in batch i dati storici di tutti i ticker filtrati
    try:
        data = yf.download(tickers, start=start, end=end, group_by="ticker")
    except Exception as e:
        print(f"Error downloading data for multiple tickers: {e}")
        return pd.DataFrame()

    # Se abbiamo un singolo ticker, yfinance non crea un MultiIndex
    single_ticker_mode = (len(tickers) == 1)

    for ticker in tickers:
        try:
            # Estraiamo i dati del singolo ticker
            if single_ticker_mode:
                df_ticker = data
            else:
                # Quando i dati sono scaricati per più ticker,
                # 'df_ticker' si trova in data[ticker].
                if ticker not in data.columns.levels[0]:
                    # Se il ticker non è presente, saltiamo
                    print(f"No data found for {ticker}. Skipping.")
                    continue
                df_ticker = data[ticker]

            if df_ticker.empty:
                print(f"Data for {ticker} is empty. Skipping.")
                continue

            # ======================
            #  FIBONACCI 61.8% CHECK
            # ======================
            current_price = df_ticker['Close'][-1]
            level_618 = fibonacci_618_level(df_ticker)

            if level_618 is None:
                continue

            near_fib_618 = False
            if is_near_price(current_price, level_618, tolerance=0.01):
                near_fib_618 = True

            # ======================
            #   INDICATORI TECNICI
            # ======================
            close_series = df_ticker['Close']
            high_series = df_ticker['High']
            low_series  = df_ticker['Low']

            # Calcolo RSI
            rsi_series = compute_rsi(close_series, period=14)
            rsi_latest = rsi_series.iloc[-1]

            # Calcolo Stocastico
            stochK, stochD = compute_stochastic(
                high_series, low_series, close_series, k_period=14, d_period=3
            )
            stochK_latest = stochK.iloc[-1]
            stochD_latest = stochD.iloc[-1]

            # Calcolo MACD
            macd_line, signal_line, hist_line = compute_macd(close_series)
            macd_latest = macd_line.iloc[-1]
            signal_latest = signal_line.iloc[-1]

            # ======================
            #   CONDIZIONI OVERSOLD
            # ======================
            # Esempio di condizioni "basiche" di oversold:
            # - RSI < 30
            # - Stocastico < 20
            # - MACD al di sotto della signal line e (macd_line < 0) => momentum negativo
            rsi_oversold = (rsi_latest < 30)
            stoch_oversold = (stochK_latest < 20)
            macd_oversold = (macd_latest < signal_latest and macd_latest < 0)

            # Se siamo vicini al fib 61.8% e tutti e tre gli indicatori sono in oversold
            # (puoi personalizzare la condizione in base alle tue esigenze)
            if near_fib_618 and rsi_oversold and stoch_oversold and macd_oversold:
                results.append({
                    'Ticker': ticker,
                    'Current Price': round(current_price, 2),
                    'Fibonacci 61,8%': round(level_618, 2),
                    'RSI': round(rsi_latest, 2),
                    'Stoch K': round(stochK_latest, 2),
                    'Stoch D': round(stochD_latest, 2),
                    'MACD': round(macd_latest, 3),
                    'Signal': round(signal_latest, 3),
                })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    return pd.DataFrame(results)

# main semplificato
def run_screener():
    st.title("Fibonacci 61.8% Screener + RSI, Stoch, MACD (Oversold)")

    # input eventuali (puoi aggiungere parametri personalizzabili)
    min_market_cap = st.number_input("Minimum Market Cap (USD)", value=10000000000, step=1000000000)
    
    if st.button("Run Screener"):
        st.write("Recupero la lista S&P 500 e calcolo i dati...")
        sp500_df = get_sp500_companies()
        if sp500_df.empty:
            st.warning("Failed to retrieve the list of S&P 500 companies.")
            return
        
        tickers = sp500_df['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        st.write("Filtraggio per market cap...")
        filtered_tickers = filter_stocks_by_market_cap(tickers, min_market_cap)
        
        st.write("Download dati e calcolo Fibonacci, RSI, Stocastico, MACD...")
        results_df = fibonacci_retracement_screener(filtered_tickers)
        
        if results_df.empty:
            st.info("Nessun titolo soddisfa i criteri (Fib 61.8% + oversold).")
        else:
            st.success("Titoli trovati:")
            st.dataframe(results_df)

if __name__ == "__main__":
    run_screener()
