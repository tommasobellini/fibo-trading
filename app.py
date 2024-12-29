import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Funzione per ottenere i ticker dell'S&P 500
@st.cache_data
def get_sp500_tickers():
    # Fonte alternativa per ottenere i ticker, ad esempio Wikipedia
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    return df['Symbol'].tolist()

# Funzione per calcolare il DPO
def calculate_dpo(df, length=33, smooth_factor=5):
    if len(df) < length + 1:
        return None
    
    # Calcolo True Range
    df['prev_close'] = df['Close'].shift(1)
    df['TR'] = df.apply(lambda row: max(row['High'] - row['Low'], 
                                       abs(row['High'] - row['prev_close']),
                                       abs(row['Low'] - row['prev_close'])), axis=1)
    
    # Calcolo volAdjPrice
    df['volAdjPrice'] = df['TR'].ewm(span=length, adjust=False).mean()
    
    # Calcolo priceChange e priceDelta
    df['priceChange'] = df['Close'] - df['Close'].shift(length)
    df['priceDelta'] = df['Close'] - df['volAdjPrice']
    
    # Calcolo oscillator
    df['oscillator'] = (df['priceDelta'] + df['priceChange']) / 2
    df['oscillator'] = df['oscillator'].ewm(span=smooth_factor, adjust=False).mean()
    
    # Bollinger Bands sull'oscillator
    bollinger = BollingerBands(close=df['oscillator'], window=length*5, window_dev=1)
    df['bbHigh'] = bollinger.bollinger_hband()
    df['bbLow'] = bollinger.bollinger_lband()
    
    bollinger_exp = BollingerBands(close=df['oscillator'], window=length*5, window_dev=2)
    df['bbHighExp'] = bollinger_exp.bollinger_hband()
    df['bbLowExp'] = bollinger_exp.bollinger_lband()
    
    # Media delle Bollinger Bands espanse
    df['mean'] = (df['bbHighExp'] + df['bbLowExp']) / 2
    
    return df

# Funzione per screening basato sul DPO
def screen_stock(ticker, start_date, end_date, length=33, smooth_factor=5):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return None
        df = calculate_dpo(df, length, smooth_factor)
        if df is None:
            return None
        latest = df.iloc[-1]
        
        # Criteri di screening:
        # Esempio: Oscillator sopra Bollinger High e sopra Bollinger High Exp
        # oppure Oscillator sotto Bollinger Low e sotto Bollinger Low Exp
        if (latest['oscillator'] > latest['bbHigh']) or (latest['oscillator'] < latest['bbLow']):
            return {
                'Ticker': ticker,
                'Oscillator': latest['oscillator'],
                'BB High': latest['bbHigh'],
                'BB Low': latest['bbLow'],
                'BB High Exp': latest['bbHighExp'],
                'BB Low Exp': latest['bbLowExp'],
                'Mean': latest['mean']
            }
        else:
            return None
    except Exception as e:
        # Gestione degli errori, ad esempio ticker non valido
        return None

def main():
    st.title("Screener SP500 basato sul Dynamic Price Oscillator (DPO)")
    st.write("""
        Questo applicativo utilizza l'indicatore Dynamic Price Oscillator (DPO) per eseguire uno screening delle azioni dell'S&P 500.
        Le azioni vengono selezionate in base a criteri specifici legati all'oscillatore e alle Bollinger Bands.
    """)

    # Parametri
    length = st.sidebar.slider("Length", min_value=10, max_value=100, value=33)
    smooth_factor = st.sidebar.slider("Smoothing Factor", min_value=1, max_value=20, value=5)
    start_date = st.sidebar.date_input("Data Inizio", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("Data Fine", datetime.now())

    if start_date >= end_date:
        st.sidebar.error("La data di inizio deve essere precedente alla data di fine.")
        return

    # Ottieni i ticker
    tickers = get_sp500_tickers()
    st.write(f"**Totale azioni nell'S&P 500:** {len(tickers)}")

    # Avvia il processo di screening
    with st.spinner("Eseguendo lo screening delle azioni..."):
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(screen_stock, ticker, start_date, end_date, length, smooth_factor) for ticker in tickers]
            for future in futures:
                res = future.result()
                if res:
                    results.append(res)

    if results:
        df_results = pd.DataFrame(results)
        st.success(f"Trovate {len(df_results)} azioni che soddisfano i criteri di screening.")
        st.dataframe(df_results)
        # Opzione per scaricare i risultati
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Scarica risultati in CSV",
            data=csv,
            file_name='sp500_dpo_screening.csv',
            mime='text/csv',
        )
    else:
        st.warning("Nessuna azione ha soddisfatto i criteri di screening.")

if __name__ == "__main__":
    main()
