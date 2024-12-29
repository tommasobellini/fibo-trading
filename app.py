import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Set Streamlit page configuration
st.set_page_config(
    page_title="S&P 500 Dynamic Price Oscillator (DPO) Strategy",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("📈 S&P 500 Dynamic Price Oscillator (DPO) Strategy")

# Sidebar for user inputs
st.sidebar.header("🔧 Parameters")

def user_input_features():
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020,1,1))
    end_date = st.sidebar.date_input("End Date", value=datetime.today())
    dpo_period = st.sidebar.number_input("DPO Period", min_value=1, max_value=100, value=20, step=1)
    initial_cash = st.sidebar.number_input("Initial Cash ($)", min_value=1000, value=10000, step=100)
    analysis_type = st.sidebar.selectbox("Analysis Type", options=["Aggregate Performance", "Individual Stock Analysis"])
    return start_date, end_date, dpo_period, initial_cash, analysis_type

start_date, end_date, dpo_period, initial_cash, analysis_type = user_input_features()

# Function to fetch S&P 500 tickers
@st.cache_data
def get_sp500_tickers():
    # Fetch the list from Wikipedia
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()  # Adjust tickers for yfinance
        return tickers
    except Exception as e:
        st.error("Error fetching S&P 500 tickers.")
        return []

sp500_tickers = get_sp500_tickers()
st.sidebar.markdown(f"**Total S&P 500 Stocks:** {len(sp500_tickers)}")

# Option to select specific stocks or analyze all
if analysis_type == "Individual Stock Analysis":
    selected_tickers = st.sidebar.multiselect("Select Stock(s) for Analysis", options=sp500_tickers, default=["AAPL", "MSFT", "GOOGL"])
else:
    selected_tickers = sp500_tickers  # All tickers

st.subheader(f"📊 Selected Stocks for DPO Strategy: {len(selected_tickers)}")

# Function to load data for a single ticker
@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            return None
        data.dropna(inplace=True)
        data['Ticker'] = ticker
        return data
    except Exception as e:
        return None

# Function to calculate DPO
def calculate_dpo(df, period):
    shift = int(period / 2 + 1)
    sma = df['Close'].rolling(window=period).mean()
    dpo = df['Close'] - sma.shift(shift)
    df['DPO'] = dpo
    return df

# Function to generate signals
def generate_signals(df):
    df['Signal'] = 0
    df['Signal'] = np.where((df['DPO'] > 0) & (df['DPO'].shift(1) <= 0), 1, df['Signal'])
    df['Signal'] = np.where((df['DPO'] < 0) & (df['DPO'].shift(1) >= 0), -1, df['Signal'])
    return df

# Function to backtest strategy
def backtest_strategy(df, initial_cash):
    cash = initial_cash
    position = 0  # Number of shares
    portfolio = []  # Portfolio value over time

    for index, row in df.iterrows():
        # Buy signal
        if row['Signal'] == 1 and cash > 0:
            position = cash / row['Close']
            cash = 0
        # Sell signal
        elif row['Signal'] == -1 and position > 0:
            cash = position * row['Close']
            position = 0
        # Calculate portfolio value
        portfolio_value = cash + position * row['Close']
        portfolio.append(portfolio_value)

    df['Portfolio Value'] = portfolio
    return df

# Function to process a single ticker
def process_ticker(ticker, start, end, period, initial_cash):
    data = load_data(ticker, start, end)
    if data is None:
        return None
    data = calculate_dpo(data, period)
    data = generate_signals(data)
    data = backtest_strategy(data, initial_cash)
    final_portfolio = data['Portfolio Value'].iloc[-1]
    profit = final_portfolio - initial_cash
    roi = (profit / initial_cash) * 100
    return {
        'Ticker': ticker,
        'Initial Cash': initial_cash,
        'Final Portfolio Value': final_portfolio,
        'Profit': profit,
        'ROI (%)': roi
    }

# Display progress
progress_bar = st.progress(0)
status_text = st.empty()

# Function to update progress
def update_progress(count, total):
    progress = count / total
    progress_bar.progress(progress)
    status_text.text(f"Processing {count} of {total} stocks...")

# Use multiprocessing for faster processing
def run_backtest(tickers, start, end, period, initial_cash):
    results = []
    total = len(tickers)
    with Pool(processes=cpu_count()) as pool:
        func = partial(process_ticker, start=start, end=end, period=period, initial_cash=initial_cash)
        for i, result in enumerate(pool.imap(func, tickers), 1):
            if result is not None:
                results.append(result)
            update_progress(i, total)
    return results

# Run backtest
with st.spinner("Running DPO strategy on selected stocks..."):
    results = run_backtest(selected_tickers, start_date, end_date, dpo_period, initial_cash)

progress_bar.empty()
status_text.empty()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

if results_df.empty:
    st.warning("No data available for the selected stocks and date range.")
else:
    # Display aggregate performance
    if analysis_type == "Aggregate Performance":
        st.subheader("💼 Aggregate Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Stocks Analyzed", len(results_df))
        total_profit = results_df['Profit'].sum()
        total_initial = results_df['Initial Cash'].sum()
        col2.metric("Total Profit", f"${total_profit:,.2f}")
        col3.metric("Average ROI", f"{results_df['ROI (%)'].mean():.2f}%")
        col4.metric("Best Performer", results_df.loc[results_df['Profit'].idxmax()]['Ticker'])

        # Top 10 Performers
        st.subheader("🏆 Top 10 Stocks by ROI")
        top10 = results_df.sort_values(by='ROI (%)', ascending=False).head(10)
        st.dataframe(top10[['Ticker', 'Final Portfolio Value', 'Profit', 'ROI (%)']])

        # Distribution of ROI
        st.subheader("📈 ROI Distribution")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(results_df['ROI (%)'], bins=30, kde=True, ax=ax, color='skyblue')
        ax.set_title("Distribution of ROI (%) Across S&P 500 Stocks")
        ax.set_xlabel("ROI (%)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Scatter Plot: Initial Cash vs Final Portfolio Value
        st.subheader("📊 Initial Cash vs Final Portfolio Value")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.scatterplot(data=results_df, x='Initial Cash', y='Final Portfolio Value', hue='ROI (%)', palette='viridis', ax=ax2)
        ax2.set_title("Initial Cash vs Final Portfolio Value")
        ax2.set_xlabel("Initial Cash ($)")
        ax2.set_ylabel("Final Portfolio Value ($)")
        st.pyplot(fig2)

    elif analysis_type == "Individual Stock Analysis":
        st.subheader("📄 Individual Stock Performance")
        st.dataframe(results_df[['Ticker', 'Final Portfolio Value', 'Profit', 'ROI (%)']].sort_values(by='ROI (%)', ascending=False))

        # Allow user to select a stock to visualize
        selected_stock = st.selectbox("Select a Stock to View Detailed Analysis", options=results_df['Ticker'].tolist())
        if selected_stock:
            # Load data again for the selected stock
            stock_data = load_data(selected_stock, start_date, end_date)
            if stock_data is not None:
                stock_data = calculate_dpo(stock_data, dpo_period)
                stock_data = generate_signals(stock_data)
                stock_data = backtest_strategy(stock_data, initial_cash)

                # Plot Close Price with Buy/Sell Signals
                fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,10), sharex=True)

                ax1.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
                ax1.set_title(f"{selected_stock} Close Price with Buy/Sell Signals")
                ax1.set_ylabel("Price ($)")
                ax1.legend()

                # Plot Buy Signals
                buy_signals = stock_data[stock_data['Signal'] == 1]
                ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=100)

                # Plot Sell Signals
                sell_signals = stock_data[stock_data['Signal'] == -1]
                ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=100)

                ax1.legend()

                # Plot DPO
                ax2.plot(stock_data.index, stock_data['DPO'], label='DPO', color='purple')
                ax2.axhline(0, color='black', linestyle='--', linewidth=1)
                ax2.set_title("Dynamic Price Oscillator (DPO)")
                ax2.set_ylabel("DPO Value")
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig3)

                # Plot Portfolio Value
                st.subheader("💼 Portfolio Value Over Time")
                fig4, ax3 = plt.subplots(figsize=(14,6))
                ax3.plot(stock_data.index, stock_data['Portfolio Value'], label='Portfolio Value', color='orange')
                ax3.set_title("Portfolio Value Over Time")
                ax3.set_xlabel("Date")
                ax3.set_ylabel("Value ($)")
                ax3.legend()

                plt.tight_layout()
                st.pyplot(fig4)

                # Performance Metrics
                final_portfolio = stock_data['Portfolio Value'].iloc[-1]
                profit = final_portfolio - initial_cash
                roi = (profit / initial_cash) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Initial Cash", f"${initial_cash:,.2f}")
                col2.metric("Final Portfolio Value", f"${final_portfolio:,.2f}")
                col3.metric("Profit", f"${profit:,.2f}", f"{roi:.2f}% ROI")

                # Show recent signals and portfolio values
                st.subheader("🔍 Recent Trades and Portfolio Values")
                st.write(stock_data[['Close', 'DPO', 'Signal', 'Portfolio Value']].tail(100))
            else:
                st.warning("No data available for the selected stock.")

    # Option to download results
    st.subheader("💾 Download Results")
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='dpo_sp500_strategy_results.csv',
        mime='text/csv',
    )

    # Footer
    st.markdown("""
    ---
    **Note:** This application performs a simplified backtest of the DPO strategy across S&P 500 stocks. It does not account for transaction costs, slippage, or other real-world trading factors. Always perform thorough testing and consider additional factors before deploying any trading strategy.
    """)
