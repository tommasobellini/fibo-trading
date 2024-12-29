#####################################
# script_streamlit_screener.py
#####################################

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import io

############################
# FUNZIONI DI SUPPORTO
############################

@st.cache_data
def get_sp500_companies():
    """
    Restituisce la lista dei componenti dell'S&P 500 leggendo la tabella da Wikipedia.
    In caso di errore o mancanza di connessione, restituisce un DataFrame vuoto.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        return df
    except Exception as e:
        st.warning(f"Impossibile scaricare la lista S&P 500: {e}")
        return pd.DataFrame()

def chunk_list(lst, chunk_size=50):
    """
    Suddivide una lista in chunk di dimensione fissa, 
    per evitare chiamate troppo grandi alle API di Yahoo.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

@st.cache_data
def get_market_caps_bulk(tickers):
    """
    Scarica la capitalizzazione di mercato per ogni ticker in 'tickers'.
    Utilizza un meccanismo a chunk per evitare chiamate troppo grandi a Yahoo Finance.
    """
    caps = {}
    for tk_chunk in chunk_list(tickers, chunk_size=50):
        joined_tickers = " ".join(tk_chunk)
        bulk_obj = yf.Tickers(joined_tickers)
        for t in tk_chunk:
            try:
                info = bulk_obj.tickers[t].info
                caps[t] = info.get('marketCap', 0)
            except Exception:
                caps[t] = 0
    return caps

def filter_stocks_by_market_cap(tickers, min_market_cap=10_000_000_000):
    """
    Filtra i ticker in base alla capitalizzazione minima di mercato (di default 10 mld).
    """
    caps_dict = get_market_caps_bulk(tickers)
    filtered = [t for t in tickers if caps_dict.get(t, 0) >= min_market_cap]
    return filtered

############################
# FUNZIONI PER L'ANALISI TECNICA (Fibonacci Screener)
############################

def find_previous_swing_points(df, exclude_days=5):
    """
    Trova i massimi e minimi precedenti, escludendo gli ultimi 'exclude_days' dati.
    """
    if len(df) < exclude_days + 2:
        return None, None
    df_past = df.iloc[:-exclude_days]
    if df_past.empty:
        return None, None
    max_price = df_past['High'].max()
    min_price = df_past['Low'].min()
    return max_price, min_price

def compute_fibonacci_levels(max_price, min_price):
    """
    Calcola i livelli di Fibonacci principali: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%.
    """
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
    """
    Verifica se 'current_price' è entro +/- (tolerance * 100)% rispetto a 'target_price'.
    Evita la divisione per zero se 'target_price' è zero.
    """
    if target_price == 0:
        return False
    return abs(current_price - target_price) / abs(target_price) <= tolerance

############################
# INDICATORI CLASSICI
############################

def compute_rsi(series, period=14):
    """
    Calcola l'RSI (Relative Strength Index) a 14 periodi di default.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1+rs))
    return rsi

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    """
    Calcola lo Stocastico (K e D) con periodi di default 14 (K) e 3 (D).
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d

def compute_macd(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Calcola la MACD standard (EMA12 - EMA26) e linea di segnale (EMA9).
    Restituisce la linea MACD, la linea segnale e l'istogramma.
    """
    ema_fast = close.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = close.ewm(span=slowperiod, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

############################
# DYNAMIC PRICE OSCILLATOR (Zeiierman)
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

    price_change = close - close.shift(length)
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
# LOGICA OVERBOUGHT/OVERSOLD
############################

def passes_osc_conditions(
    rsi_val, stoch_k_val, macd_val, signal_val,
    check_rsi=True, check_stoch=True, check_macd=True,
    rsi_threshold=30, stoch_threshold=20
):
    """
    Restituisce True se il titolo passa TUTTI i filtri RSI, Stoch, MACD attivati.
    Esempio oversold: RSI < 30, Stoch K < 20, MACD < 0 e < Signal.
    """
    # RSI
    if check_rsi and rsi_val >= rsi_threshold:
        return False
    # Stocastico
    if check_stoch and stoch_k_val >= stoch_threshold:
        return False
    # MACD
    if check_macd and (macd_val >= 0 or macd_val >= signal_val):
        return False
    return True

def passes_dpo_condition(dpo_val, dpo_bb_low, dpo_bb_high, dpo_mode="Nessuna"):
    """
    dpo_mode: "Nessuna", "Oversold (DPO < BB Low)", "Overbought (DPO > BB High)"
    """
    if dpo_mode == "Nessuna":
        return True
    if (
        dpo_val is None or pd.isna(dpo_val) or
        dpo_bb_low is None or pd.isna(dpo_bb_low) or
        dpo_bb_high is None or pd.isna(dpo_bb_high)
    ):
        return False

    # Oversold
    if dpo_mode.startswith("Oversold") and dpo_val >= dpo_bb_low:
        return False
    # Overbought
    if dpo_mode.startswith("Overbought") and dpo_val <= dpo_bb_high:
        return False

    return True

############################
# FUNZIONE DI ENTRY & STOP (FIBONACCI)
############################

def define_trade_levels(current_price, min_price, stop_pct=0.01):
    """
    Definisce entry e stop loss:
    - Entry Price = current_price
    - Stop Price = (1 - stop_pct) * min_price
    """
    entry_price = current_price
    stop_price = min_price * (1 - stop_pct)
    return round(entry_price, 2), round(stop_price, 2)

############################
# FIBONACCI SCREENER
############################

def fibonacci_screener_entry_stop(
    tickers, 
    exclude_days=5, 
    check_rsi=True, 
    check_stoch=True, 
    check_macd=True,
    dpo_mode="Nessuna",
    dpo_length=33,
    dpo_smooth=5,
    fib_tolerance=0.01,
    rsi_threshold=30,
    stoch_threshold=20,
    stop_pct=0.01
):
    """
    Screener basato su rimbalzo al 61.8% di Fibonacci + indicatori oversold + DPO.
    """
    end = datetime.now()
    start = end - timedelta(days=90)

    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval='1h',
        group_by="ticker",
        progress=False
    )

    single_ticker = (len(tickers) == 1)
    results = []

    my_bar = st.progress(0)
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        my_bar.progress((i + 1) / total)

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

            # Indicatori
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
            dpo_val      = None
            dpo_bb_low   = None
            dpo_bb_high  = None
            if dpo_mode != "Nessuna":
                oscillator, bbHigh, bbLow, bbHighExp, bbLowExp = compute_dynamic_price_oscillator(
                    df_ticker, length=dpo_length, smooth_factor=dpo_smooth
                )
                if oscillator.dropna().empty or bbLow.dropna().empty or bbHigh.dropna().empty:
                    continue
                dpo_val     = oscillator.iloc[-1]
                dpo_bb_low  = bbLow.iloc[-1]
                dpo_bb_high = bbHigh.iloc[-1]

            # Fib 61.8%
            fib_618 = fib_levels['61.8%']

            # Verifichiamo se il current_price è vicino al 61.8% di Fibonacci
            if is_near_level(current_price, fib_618, tolerance=fib_tolerance):
                # Filtri RSI, Stoch, MACD
                pass_osc = passes_osc_conditions(
                    rsi_val, stoch_k_val, macd_val, signal_val,
                    check_rsi, check_stoch, check_macd,
                    rsi_threshold=rsi_threshold,
                    stoch_threshold=stoch_threshold
                )
                if not pass_osc:
                    continue

                # Filtro DPO
                pass_dpo = passes_dpo_condition(
                    dpo_val, dpo_bb_low, dpo_bb_high,
                    dpo_mode
                )
                if not pass_dpo:
                    continue

                # Se tutti i filtri passano => definiamo entry e stop
                fib_trend = "Bullish" if current_price >= fib_618 else "Bearish"
                entry_price, stop_price = define_trade_levels(current_price, min_price, stop_pct=stop_pct)

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
                    'DPO BB Low': round(dpo_bb_low, 2) if dpo_bb_low is not None else None,
                    'DPO BB High': round(dpo_bb_high, 2) if dpo_bb_high is not None else None,
                    'Entry Price': entry_price,
                    'Stop Price': stop_price
                })

        except Exception as e:
            st.warning(f"Errore su {ticker}: {e}")
            continue

    my_bar.empty()
    return pd.DataFrame(results)

############################
# BREAKOUT MENSILE (CON CHECK NEGLI ULTIMI N GIORNI)
############################

def monthly_breakout_screener_recent(
    tickers,
    n_months=3,         # Calcolo breakout sugli ultimi N mesi
    last_n_days=3,      # Controlliamo se il breakout è avvenuto negli ultimi N giorni
    start_date="2020-01-01",
    end_date=None
):
    """
    Scarica i dati giornalieri per ogni ticker, ricampiona mensilmente,
    calcola rolling max/min sugli ultimi n_months (shift(1)).
    Poi controlla negli ultimi 'last_n_days' se c'è stata una chiusura > rolling_max (breakout long)
    o < rolling_min (breakout short).
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    results = []
    my_bar = st.progress(0)
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        my_bar.progress((i + 1) / total)
        try:
            df_daily = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df_daily.empty:
                continue

            # Ricampioniamo su base mensile
            df_monthly = df_daily.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()

            # rolling_max e rolling_min
            df_monthly['rolling_max'] = df_monthly['High'].rolling(n_months).max().shift(1)
            df_monthly['rolling_min'] = df_monthly['Low'].rolling(n_months).min().shift(1)

            if len(df_monthly) < 1:
                continue

            latest_month = df_monthly.iloc[-1]
            rmax = latest_month['rolling_max']
            rmin = latest_month['rolling_min']
            if pd.isna(rmax) or pd.isna(rmin):
                continue

            # Controlliamo negli ultimi 'last_n_days'
            if len(df_daily) < last_n_days:
                continue

            df_recent = df_daily.tail(last_n_days)
            max_recent_close = df_recent['Close'].max()
            min_recent_close = df_recent['Close'].min()

            signal = "No breakout"
            if max_recent_close > rmax:
                signal = "Breakout Long"
            elif min_recent_close < rmin:
                signal = "Breakout Short"

            results.append({
                "Ticker": ticker,
                "Ultima Chiusura": round(df_daily['Close'].iloc[-1], 2),
                f"rolling_max_{n_months}m": round(rmax, 2),
                f"rolling_min_{n_months}m": round(rmin, 2),
                "Max Close (ultimi N gg)": round(max_recent_close, 2),
                "Min Close (ultimi N gg)": round(min_recent_close, 2),
                "Breakout in ultimi N giorni": signal
            })

        except Exception as e:
            st.warning(f"Errore su {ticker}: {e}")
            continue

    my_bar.empty()
    return pd.DataFrame(results)

############################
# STOCK PICKING (FILTRI FONDAMENTALI)
############################

@st.cache_data
def get_fundamentals_bulk(tickers):
    """
    Scarica alcuni dati fondamentali per ogni ticker (es. trailingPE, forwardPE, priceToBook, pegRatio, dividendYield).
    Ritorna un DataFrame con colonne: [symbol, trailingPE, forwardPE, priceToBook, pegRatio, dividendYield].
    """
    fundamentals_list = []
    for tk_chunk in chunk_list(tickers, chunk_size=50):
        joined_tickers = " ".join(tk_chunk)
        bulk_obj = yf.Tickers(joined_tickers)
        for t in tk_chunk:
            try:
                info = bulk_obj.tickers[t].info
                trailing_pe = info.get("trailingPE", None)
                forward_pe = info.get("forwardPE", None)
                pb = info.get("priceToBook", None)
                peg = info.get("pegRatio", None)
                dividend_yield = info.get("dividendYield", None)

                fundamentals_list.append({
                    "symbol": t,
                    "trailingPE": trailing_pe,
                    "forwardPE": forward_pe,
                    "priceToBook": pb,
                    "pegRatio": peg,
                    "dividendYield": dividend_yield
                })
            except Exception:
                fundamentals_list.append({
                    "symbol": t,
                    "trailingPE": None,
                    "forwardPE": None,
                    "priceToBook": None,
                    "pegRatio": None,
                    "dividendYield": None
                })
    df_fund = pd.DataFrame(fundamentals_list)
    return df_fund

def stock_picking_screener(
    tickers, 
    max_pe=20, 
    max_pb=3, 
    min_dividend=0.02
):
    """
    Esempio di stock picking screener:
      - trailingPE < max_pe
      - priceToBook < max_pb
      - dividendYield >= min_dividend
    """
    df_fund = get_fundamentals_bulk(tickers)

    df_fund['trailingPE'] = df_fund['trailingPE'].fillna(999999)
    df_fund['priceToBook'] = df_fund['priceToBook'].fillna(999999)
    df_fund['dividendYield'] = df_fund['dividendYield'].fillna(0)

    filtered_df = df_fund[
        (df_fund['trailingPE'] > 0) & (df_fund['trailingPE'] < max_pe) &
        (df_fund['priceToBook'] > 0) & (df_fund['priceToBook'] < max_pb) &
        (df_fund['dividendYield'] >= min_dividend)
    ].copy()

    # Ordina per trailingPE (o come preferisci)
    filtered_df.sort_values(by=['trailingPE'], inplace=True)
    return filtered_df

############################
# APP STREAMLIT
############################

def main():
    st.title("Screener Multiplo: Fibonacci / Breakout Mensile / Stock Picking")
    
    # Scelta strategia
    st.subheader("Seleziona la strategia da eseguire:")
    strategy_choice = st.selectbox(
        "Strategia",
        [
            "Fibonacci RSI/Stoch/MACD + DPO",
            "Breakout Mensile (ultimi N giorni)",
            "Stock Picking (fondamentale)"
        ]
    )

    st.subheader("1) Parametri di filtraggio (Market Cap)")
    min_market_cap = st.number_input("Min Market Cap (in $)", value=10_000_000_000, step=1_000_000_000)

    # Parametri personalizzati in base alla strategia scelta
    if strategy_choice == "Fibonacci RSI/Stoch/MACD + DPO":
        st.subheader("2) Parametri Fibonacci")
        exclude_days = st.number_input("Escludi ultimi N giorni (swing precedenti)", value=5, step=1)
        
        st.subheader("Indicatori di 'Oversold'")
        rsi_flag = st.checkbox("RSI < soglia (default 30)?", value=True)
        rsi_thr = st.number_input("Soglia RSI", value=30, step=1)

        stoch_flag = st.checkbox("Stocastico < soglia (default 20)?", value=True)
        stoch_thr = st.number_input("Soglia Stocastico", value=20, step=1)

        macd_flag = st.checkbox("MACD < 0 e MACD < Signal?", value=True)

        st.subheader("3) Parametri DPO")
        dpo_mode = st.selectbox(
            "DPO Condition",
            ["Nessuna", "Oversold (DPO < BB Low)", "Overbought (DPO > BB High)"],
            index=0
        )
        dpo_length_val = st.number_input("DPO Length (default=33)", value=33, step=1)
        dpo_smooth_val = st.number_input("DPO Smoothing Factor (default=5)", value=5, step=1)

        st.subheader("4) Parametri Fibonacci e Stop")
        fib_tolerance = st.number_input("Tolleranza Fib 61.8% (es: 0.01 = 1%)", value=0.01, step=0.001)
        stop_pct_val = st.number_input("Stop Loss % sotto il minimo swing", value=0.01, step=0.001)

    elif strategy_choice == "Breakout Mensile (ultimi N giorni)":
        st.subheader("2) Parametri Breakout Mensile")
        n_months = st.number_input("N mesi di lookback (es: 3)", value=3, step=1)
        last_n_days = st.number_input("Controlla breakout negli ultimi N giorni", value=3, step=1)
        start_str = st.text_input("Data inizio (YYYY-MM-DD)", "2020-01-01")

    else:  # Stock Picking
        st.subheader("2) Parametri Stock Picking")
        max_pe_val = st.number_input("Max Trailing P/E", value=20, step=1)
        max_pb_val = st.number_input("Max Price/Book", value=3, step=1)
        min_div_val = st.number_input("Min Dividend Yield (es. 0.02 = 2%)", value=0.02, step=0.01)

    # Avvio Screener
    if st.button("Esegui Screener"):
        sp500_df = get_sp500_companies()
        if sp500_df.empty:
            st.error("Impossibile scaricare la lista S&P 500 o tabella vuota.")
            return
        
        tickers = sp500_df['Symbol'].tolist()
        # In Yahoo Finance, i ticker con punto vanno sostituiti col trattino
        tickers = [t.replace('.', '-') for t in tickers]

        # Filtro per capitalizzazione
        filtered = filter_stocks_by_market_cap(tickers, min_market_cap)
        st.write("Numero di ticker dopo il filtro Market Cap:", len(filtered))
        if not filtered:
            st.warning("Nessun ticker supera la capitalizzazione richiesta.")
            return

        # Selezioniamo la logica in base alla strategia
        if strategy_choice == "Fibonacci RSI/Stoch/MACD + DPO":
            df_results = fibonacci_screener_entry_stop(
                tickers=filtered,
                exclude_days=exclude_days,
                check_rsi=rsi_flag,
                check_stoch=stoch_flag,
                check_macd=macd_flag,
                dpo_mode=dpo_mode,
                dpo_length=dpo_length_val,
                dpo_smooth=dpo_smooth_val,
                fib_tolerance=fib_tolerance,
                rsi_threshold=rsi_thr,
                stoch_threshold=stoch_thr,
                stop_pct=stop_pct_val
            )
            if df_results.empty:
                st.info("Nessun titolo rispetta i criteri selezionati.")
            else:
                st.success(f"Trovati {len(df_results)} titoli (strategia Fibonacci):")
                st.dataframe(df_results)

                # Download
                watchlist = df_results['Ticker'].tolist()
                csv_buffer = io.StringIO()
                for t in watchlist:
                    csv_buffer.write(t + "\n")
                csv_data = csv_buffer.getvalue()
                st.download_button(
                    label="Scarica Tickers",
                    data=csv_data,
                    file_name="fibonacci_watchlist.txt",
                    mime="text/plain"
                )

        elif strategy_choice == "Breakout Mensile (ultimi N giorni)":
            df_breakout = monthly_breakout_screener_recent(
                tickers=filtered,
                n_months=n_months,
                last_n_days=last_n_days,
                start_date=start_str
            )
            if df_breakout.empty:
                st.info("Nessun segnale di breakout negli ultimi giorni.")
            else:
                st.success(f"Trovati {len(df_breakout)} tickers con breakout (ultimi {last_n_days} giorni):")
                st.dataframe(df_breakout)

                # Download
                watchlist = df_breakout['Ticker'].tolist()
                csv_buffer = io.StringIO()
                for t in watchlist:
                    csv_buffer.write(t + "\n")
                csv_data = csv_buffer.getvalue()
                st.download_button(
                    label="Scarica Tickers",
                    data=csv_data,
                    file_name="breakout_watchlist.txt",
                    mime="text/plain"
                )

        else:  # Stock Picking (fondamentale)
            df_picked = stock_picking_screener(
                filtered, 
                max_pe=max_pe_val,
                max_pb=max_pb_val,
                min_dividend=min_div_val
            )
            if df_picked.empty:
                st.info("Nessun titolo rispetta i criteri fondamentali.")
            else:
                st.success(f"Trovati {len(df_picked)} titoli con parametri fondamentali richiesti:")
                st.dataframe(df_picked)

                # Download
                watchlist = df_picked['symbol'].tolist()
                csv_buffer = io.StringIO()
                for t in watchlist:
                    csv_buffer.write(t + "\n")
                csv_data = csv_buffer.getvalue()
                st.download_button(
                    label="Scarica Tickers",
                    data=csv_data,
                    file_name="fundamentals_watchlist.txt",
                    mime="text/plain"
                )

    # Disclaimer
    st.write("---")
    st.markdown("""
    **Disclaimer**: Questo screener è a puro scopo didattico. 
    Non costituisce consiglio finanziario. Si raccomanda di fare le proprie analisi 
    prima di effettuare operazioni di trading.
    """)

if __name__ == "__main__":
    main()
