import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import io

############################
# Funzioni di supporto
############################

def get_sp500_companies():
    """
    Recupera la lista delle aziende S&P 500 da Wikipedia.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    return df

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
# Nuova logica:
# - Scarichiamo 6 mesi
# - Escludiamo ultimi 5 giorni
# - Troviamo max/min in quel periodo
############################
def find_previous_swing_points(df, exclude_days=5):
    """
    Cerca il "massimo precedente" e il "minimo precedente" in df,
    ignorando (escludendo) gli ultimi `exclude_days` giorni,
    così da ottenere un movimento compiuto in passato.
    Restituisce (max_price, min_price).
    Se df è troppo corto, restituisce (None, None).
    """
    if len(df) < exclude_days + 2:
        return None, None
    
    # Escludiamo gli ultimi `exclude_days` giorni
    df_past = df.iloc[:-exclude_days]  # tutti tranne gli ultimi n
    if df_past.empty:
        return None, None
    
    max_price = df_past['High'].max()
    min_price = df_past['Low'].min()
    return max_price, min_price

def compute_fibonacci_levels(max_price, min_price):
    """
    Ricalcola i livelli di Fibonacci 0%, 23,6%, 38,2%, 50%, 61,8%, 78,6%, 100%
    basandosi su un max_price "precedente" e un min_price "precedente".
    Di default, 0% corrisponde al max_price, 100% corrisponde al min_price
    (ritracciamento dall'alto verso il basso).
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
# Scarico dati e calcolo
############################
def fibonacci_screener_previous_swings(tickers, exclude_days=5):
    """
    1) Scarica i dati degli ultimi 6 mesi.
    2) Trova max e min "precedenti" (escludendo ultimi X giorni).
    3) Calcola i livelli di Fibonacci da quei due punti.
    4) Verifica se il prezzo attuale è vicino a uno di quei livelli.
       (per esempio, di default cerchiamo il 61,8%, ma puoi modificare).
    """
    end = datetime.now()
    start = end - timedelta(days=6*30)  # 6 mesi circa
    
    # Scarichiamo i dati (in blocco o a chunk se vuoi)
    data = yf.download(tickers, start=start, end=end, group_by="ticker")
    # Se abbiamo un solo ticker, no MultiIndex
    single_ticker = (len(tickers) == 1)

    result = []
    # Scegliamo a quali livelli prestare attenzione
    selected_levels = ['61.8%']  # ad esempio solo 61,8%

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
            
            # Troviamo i "precedenti" max e min
            max_price, min_price = find_previous_swing_points(df_ticker, exclude_days=exclude_days)
            if max_price is None or min_price is None:
                continue
            
            # Calcoliamo i livelli
            fib_levels = compute_fibonacci_levels(max_price, min_price)
            if not fib_levels:
                continue
            
            # Prezzo attuale (ultimo close)
            current_price = df_ticker['Close'].iloc[-1]
            
            # Verifichiamo se siamo vicini a uno dei livelli selezionati
            for lvl_name in selected_levels:
                lvl_price = fib_levels[lvl_name]
                if is_near_level(current_price, lvl_price, tolerance=0.01):
                    # Determiniamo se siamo bullish o bearish rispetto a quel livello
                    if current_price >= lvl_price:
                        fib_trend = "Bullish"
                    else:
                        fib_trend = "Bearish"

                    result.append({
                        'Ticker': ticker,
                        'Max Price (prev.)': round(max_price, 2),
                        'Min Price (prev.)': round(min_price, 2),
                        'Fibonacci Level': lvl_name,
                        'Level Price': round(lvl_price, 2),
                        'Current Price': round(current_price, 2),
                        'Fib Trend': fib_trend,
                    })
                    
        except Exception as e:
            print(f"Errore su {ticker}: {e}")
            continue

    return pd.DataFrame(result)

############################
# Esempio Streamlit
############################
def main():
    st.title("Fibonacci Screener con massimi/minimi precedenti")
    st.write("""
    Invece di prendere l'ultimo mese, cerchiamo un "massimo precedente" e "minimo precedente"
    (escludendo gli ultimi N giorni) e tracciamo Fibonacci su quei punti.
    """)

    # Parametro per la capitalizzazione
    min_market_cap = st.number_input("Min Market Cap (USD)", value=10_000_000_000, step=1_000_000_000)
    # Parametro exclude_days
    exclude_days = st.number_input("Escludi ultimi N giorni per trovare max/min precedente", value=5, step=1)

    if st.button("Esegui Screener"):
        # 1) Lista S&P 500
        sp500_df = get_sp500_companies()
        if sp500_df.empty:
            st.error("Impossibile scaricare la lista S&P 500.")
            return
        
        # Pulizia ticker
        tickers = sp500_df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        
        # 2) Filtro market cap
        filtered_tickers = filter_stocks_by_market_cap(tickers, min_market_cap)
        st.write("Ticker dopo filtro:", len(filtered_tickers))
        if not filtered_tickers:
            st.warning("Nessun ticker supera la capitalizzazione richiesta.")
            return

        # 3) Calcolo fibonacci_screener_previous_swings
        df_results = fibonacci_screener_previous_swings(filtered_tickers, exclude_days=exclude_days)
        if df_results.empty:
            st.info("Nessun titolo è vicino al livello Fibonacci selezionato (ex: 61,8%).")
        else:
            st.success(f"Risultati trovati: {len(df_results)}")
            st.dataframe(df_results)

if __name__ == "__main__":
    main()
