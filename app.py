import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# Funzione per calcolare il CVD
def calculate_cvd(data):
    data['Price Change'] = data['Close'].diff().fillna(0)
    data['Volume Flow'] = np.where(data['Price Change'] > 0, data['Volume'],
                                   np.where(data['Price Change'] < 0, -data['Volume'], 0))
    data['CVD'] = data['Volume Flow'].cumsum()
    return data

# Funzione per trovare divergenze e dare indicazioni di trading
def find_divergences(data):
    data = calculate_cvd(data)
    data['Price Change'] = data['Close'].pct_change().fillna(0)
    data['CVD Change'] = data['CVD'].pct_change().fillna(0)
    data['Divergence'] = np.where(
        (data['Price Change'] > 0) & (data['CVD Change'] < 0), 1,
        np.where((data['Price Change'] < 0) & (data['CVD Change'] > 0), -1, 0)
    )
    data['Signal'] = np.where(data['Divergence'] == 1, 'Long',
                              np.where(data['Divergence'] == -1, 'Short', 'Hold'))
    return data

# Funzione per scaricare i dati dei titoli
def fetch_ticker_data(ticker, period="1mo", interval="1d"):
    try:
        stock_data = yf.download(ticker, period=period, interval=interval)
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        st.error(f"Errore nel recupero dei dati per {ticker}: {e}")
        return None

# Streamlit app
st.title("Analisi Divergenze CVD con Filtri Long/Short e Range di Prezzo")
st.write("Carica un file CSV contenente i ticker da analizzare. Il file deve avere una colonna 'Ticker'.")

uploaded_file = st.file_uploader("Carica il file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Anteprima dei dati caricati:")
    st.dataframe(df.head())

    if 'Ticker' not in df.columns:
        st.error("Il file CSV deve contenere una colonna denominata 'Ticker'.")
    else:
        tickers = df['Ticker'].unique()
        period = st.selectbox("Periodo", options=["1mo", "3mo", "6mo", "1y"], index=0)
        interval = st.selectbox("Intervallo", options=["1d", "1h", "5m"], index=0)
        filter_signal = st.selectbox("Filtra i risultati", options=["Tutti", "Long", "Short"], index=0)
        
        # Input per range di prezzo
        st.write("Definisci un range di prezzo:")
        min_price = st.number_input("Prezzo minimo", value=0.0, step=1.0)
        max_price = st.number_input("Prezzo massimo", value=1000.0, step=1.0)

        if st.button("Analizza"):
            results = []

            for ticker in tickers:
                try:
                    st.write(f"**Analizzando {ticker}...**")
                    data = fetch_ticker_data(ticker, period=period, interval=interval)

                    if data is not None and not data.empty:
                        data = find_divergences(data)
                        latest_price = data.iloc[-1]['Close']
                        latest_signal = data.iloc[-1]['Signal']

                        # Applica i filtri per segnale e prezzo
                        if (filter_signal == "Tutti" or latest_signal == filter_signal) and (min_price <= latest_price <= max_price):
                            results.append((ticker, latest_signal, latest_price))
                            
                            st.write(f"**Segnale per {ticker}: {latest_signal} (Prezzo: {latest_price:.2f})**")
                            
                            # Plot grafico con divergenza
                            fig, ax1 = plt.subplots()
                            ax1.set_xlabel('Data')
                            ax1.set_ylabel('Prezzo', color='tab:blue')
                            ax1.plot(data['Date'], data['Close'], label='Prezzo', color='tab:blue')
                            ax1.tick_params(axis='y', labelcolor='tab:blue')

                            ax2 = ax1.twinx()
                            ax2.set_ylabel('CVD', color='tab:red')
                            ax2.plot(data['Date'], data['CVD'], label='CVD', color='tab:red')
                            ax2.tick_params(axis='y', labelcolor='tab:red')

                            # Aggiungi linea delle divergenze
                            ax1.plot(data['Date'], data['Divergence'] * data['Close'].max() * 0.1, 
                                    label='Divergenza', linestyle='--', color='green')

                            fig.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.warning(f"Dati non disponibili per {ticker}.")
                except Exception as e:
                    st.error(f"Errore durante l'analisi di {ticker}: {e}")
            if results:
                st.write("**Riepilogo Segnali:**")
                summary_df = pd.DataFrame(results, columns=['Ticker', 'Segnale', 'Prezzo'])
                st.dataframe(summary_df)
