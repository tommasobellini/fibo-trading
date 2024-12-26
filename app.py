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
    Recupera in blocco i Market Cap dei ticker, usando yf.Tickers(...)
    invece di invocare Ticker(t).info per ognuno.
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

############################
# Trova max/min "precedenti"
############################
def find_previous_swing_points(df, exclude_days=5):
    """
    Cerca il "massimo precedente" e il "minimo precedente" in df,
    ignorando (escludendo) gli ultimi `exclude_days` giorni.
    Restituisce (max_price, min_price).
    """
    if len(df) < exclude_days + 2:
        return None, None
    
    # Escludiamo gli ultimi N giorni
    df_past = df.iloc[:-exclude_days]
    if df_past.empty:
        return None, None
    
    max_price = df_past['High'].max()
    min_price = df_past['Low'].min()
    return max_price, min_price

############################
# Livelli di Fibonacci
############################
def compute_fibonacci_levels(max_price, min_price):
    """
    Calcolo classico del retracement:
    0% = max_price, 100% = min_price (dall'alto al basso).
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
    Verifica se current_price è vicino a target_price
    con una tolleranza percentuale (default 1%).
    """
    if target_price == 0:
        return False
    return abs(current_price - target_price) / abs(target_price) <= tolerance


############################
# Stagionalità su 10 anni
############################
@st.cache_data
def compute_seasonality_10y(ticker):
    """
    Calcola la stagionalità rispetto allo stesso mese
    negli ultimi 10 anni.
    
    - Scarica 10 anni di dati daily.
    - Per ogni mese (1..12), calcola la performance media storica.
    - Calcola la performance (somma daily_return) del mese corrente.
    - Restituisce: differenza (mese_corrente - media_storica_mese) * 100
      in punti percentuali.
    
    Se mancano dati o non possiamo calcolare, restituisce None.
    """
    end = datetime.now()
    start = end - timedelta(days=365 * 10)  # 10 anni

    try:
        df = yf.download(ticker, start=start, end=end, interval="1d")
    except Exception as e:
        st.warning(f"Error downloading data for {ticker}: {e}")
        return None
    
    if df.empty:
        return None
    
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    # Rendimento giornaliero
    df['Daily_Return'] = df['Close'].pct_change()

    # Raggruppiamo Year, Month → somma daily_return
    monthly = df.groupby(['Year','Month'])['Daily_Return'].sum().reset_index()
    
    # Media storica per ogni mese (1..12)
    monthly_avg = monthly.groupby('Month')['Daily_Return'].mean()

    current_year = end.year
    current_month = end.month
    
    # Trova la performance del mese attuale
    row_current = monthly[(monthly['Year'] == current_year) & (monthly['Month'] == current_month)]
    if row_current.empty:
        # Se il mese non è finito, facciamo partial:
        df_this_month = df[(df.index.year == current_year) & (df.index.month == current_month)]
        if df_this_month.empty:
            return None
        actual_return = df_this_month['Daily_Return'].sum()
    else:
        actual_return = row_current['Daily_Return'].iloc[0]

    if current_month not in monthly_avg.index:
        return None
    
    avg_return = monthly_avg.loc[current_month]

    diff = (actual_return - avg_return) * 100.0
    return diff  # ex: +2.5 => 2,5% sopra la media storica del mese


############################
# Screener principale
############################
def fibonacci_screener_previous_swings(tickers, exclude_days=5):
    """
    1) Scarica 6 mesi di dati.
    2) Trova max e min "precedenti" escludendo ultimi exclude_days giorni.
    3) Calcola i livelli Fib, controlla se l'ultimo prezzo è vicino a 61,8%.
    4) Determina bull/bear.
    Restituisce un DataFrame con i titoli che rispettano la condizione.
    """
    end = datetime.now()
    start = end - timedelta(days=6*30)  # ~6 mesi
    
    try:
        data = yf.download(tickers, start=start, end=end, group_by="ticker")
    except Exception as e:
        st.warning(f"Error downloading data in bulk: {e}")
        return pd.DataFrame()

    single_ticker = (len(tickers) == 1)
    results = []

    # Interessati di default al 61,8% (puoi aggiungere altri se vuoi)
    selected_levels = ['61.8%']
    tolerance = 0.01  # 1%

    for ticker in tickers:
        try:
            if single_ticker:
                df_ticker = data
            else:
                # MultiIndex
                if ticker not in data.columns.levels[0]:
                    continue
                df_ticker = data[ticker]
            
            if df_ticker.empty:
                continue

            # Trova swing prev
            max_price, min_price = find_previous_swing_points(df_ticker, exclude_days=exclude_days)
            if max_price is None or min_price is None:
                continue
            
            fib_levels = compute_fibonacci_levels(max_price, min_price)
            if not fib_levels:
                continue

            current_price = df_ticker['Close'].iloc[-1]

            for lvl_name in selected_levels:
                lvl_price = fib_levels[lvl_name]
                if is_near_level(current_price, lvl_price, tolerance=tolerance):
                    # Bullish o Bearish?
                    fib_trend = "Bullish" if (current_price >= lvl_price) else "Bearish"
                    results.append({
                        'Ticker': ticker,
                        'Max Price (prev.)': round(max_price, 2),
                        'Min Price (prev.)': round(min_price, 2),
                        'Fib Level': lvl_name,
                        'Level Price': round(lvl_price, 2),
                        'Current Price': round(current_price, 2),
                        'Fib Trend': fib_trend,
                    })
        except Exception as e:
            st.warning(f"Errore su {ticker}: {e}")
            continue

    return pd.DataFrame(results)


############################
# App Streamlit
############################
def main():
    st.title("Fibonacci Screener (massimi/minimi precedenti) + Stagionalità 10y")

    min_mcap = st.number_input("Min Market Cap (USD)", value=10_000_000_000, step=1_000_000_000)
    exclude_days = st.number_input("Escludi ultimi N giorni per trovare max/min precedente", value=5, step=1)

    if st.button("Esegui Screener"):
        # 1) Lista S&P500
        sp500_df = get_sp500_companies()
        if sp500_df.empty:
            st.error("Impossibile scaricare la lista S&P 500.")
            return
        
        # Pulizia ticker
        tickers = sp500_df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]

        # 2) Filtro
        filtered_tickers = filter_stocks_by_market_cap(tickers, min_mcap)
        st.write("Ticker dopo filtro:", len(filtered_tickers))
        if not filtered_tickers:
            st.warning("Nessun ticker supera la capitalizzazione richiesta.")
            return

        # 3) Screener fib
        df_results = fibonacci_screener_previous_swings(filtered_tickers, exclude_days=exclude_days)
        if df_results.empty:
            st.info("Nessun titolo si trova vicino al livello Fib 61.8% (precedente swing).")
            return

        st.success(f"Troviamo {len(df_results)} titoli. Calcoliamo la Stagionalità (10 anni)...")

        # 4) Calcolo stagionalità 10y
        seasonality_scores = []
        for t in df_results['Ticker']:
            score = compute_seasonality_10y(t)
            if score is None:
                seasonality_scores.append(None)
            else:
                # round a due decimali
                seasonality_scores.append(round(score, 2))

        df_results['Seasonality (10y)'] = seasonality_scores

        st.write("""
        - **Fib Trend**: 'Bullish' se prezzo >= livello 61,8%, 'Bearish' se < livello.
        - **Seasonality (10y)**: differenza (in %) tra la performance di questo mese e la media storica
          dello stesso mese, calcolata sugli ultimi 10 anni.
          Esempio: +2.5 => il titolo sta rendendo 2.5 punti percentuali in più rispetto
          alla sua media storica del mese.
        """)

        st.dataframe(df_results)

        # Opzionale: esporta
        watchlist_tickers = df_results['Ticker'].tolist()
        csv_buffer = io.StringIO()
        for t in watchlist_tickers:
            csv_buffer.write(t + "\n")
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Scarica Tickers",
            data=csv_data,
            file_name="my_fibonacci_watchlist.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
