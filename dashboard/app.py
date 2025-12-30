"""
QS Hedge Fund Dashboard

Main Streamlit application for monitoring the quant hedge fund.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# Page configuration
st.set_page_config(
 page_title="QS Hedge Fund Dashboard",
 page_icon="📈",
 layout="wide",
 initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
 .main-header {
 font-size: 2.5rem;
 font-weight: bold;
 color: #1f77b4;
 margin-bottom: 1rem;
 }
 .metric-card {
 background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
 border-radius: 10px;
 padding: 1rem;
 box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
 }
 .positive {
 color: #00ff88;
 }
 .negative {
 color: #ff4444;
 }
</style>
""", unsafe_allow_html=True)


def main():
 """Main dashboard application."""
 
 # Sidebar
 with st.sidebar:
 st.title("⚙️ Settings")
 
 st.subheader("Data Settings")
 data_dir = st.text_input(
 "Dataset directory",
 value="data/outputs",
 )
 
 deployment_state = st.text_input(
 "Deployment state file",
 value="model_serving/deployment_state.json",
 )
 
 if st.button("🔄 Reload dataset from disk"):
 st.cache_data.clear()
 st.success("Cache cleared!")
 
 st.divider()
 
 st.subheader("AI Settings")
 openai_model = st.selectbox(
 "OpenAI model",
 ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
 )
 
 st.divider()
 
 st.subheader("MLflow Settings")
 mlflow_uri = st.text_input(
 "MLflow tracking URI",
 value="http://127.0.0.1:5050",
 )
 
 # Main content
 st.markdown('<div class="main-header">📊 QS Hedge Fund Dashboard</div>', unsafe_allow_html=True)
 
 # Tabs
 tab1, tab2, tab3, tab4 = st.tabs([
 "📈 Live Ops",
 "🔬 Research Lab",
 "📊 Signal Charts",
 "🤖 AI Quant Team",
 ])
 
 with tab1:
 render_live_ops()
 
 with tab2:
 render_research_lab()
 
 with tab3:
 render_signal_charts()
 
 with tab4:
 render_ai_quant()


def render_live_ops():
 """Render the Live Operations tab."""
 
 # Key metrics row
 col1, col2, col3, col4 = st.columns(4)
 
 with col1:
 st.metric(
 label="Portfolio Value",
 value="$5.2M",
 delta="+12.3%",
 )
 
 with col2:
 st.metric(
 label="Positions",
 value="19",
 delta="+2",
 )
 
 with col3:
 st.metric(
 label="Daily P&L",
 value="+$24,500",
 delta="+0.47%",
 )
 
 with col4:
 st.metric(
 label="Sharpe Ratio",
 value="1.85",
 delta="+0.12",
 )
 
 st.divider()
 
 # Performance chart
 st.subheader("📈 Performance")
 
 # Generate sample data
 dates = pd.date_range(start="2024-01-01", end=datetime.now(), freq="D")
 portfolio_values = [1_000_000]
 benchmark_values = [1_000_000]
 
 for i in range(1, len(dates)):
 portfolio_values.append(portfolio_values[-1] * (1 + 0.0008 + 0.003 * (i % 10 - 5) / 100))
 benchmark_values.append(benchmark_values[-1] * (1 + 0.0005 + 0.002 * (i % 10 - 5) / 100))
 
 fig = go.Figure()
 
 fig.add_trace(go.Scatter(
 x=dates,
 y=portfolio_values,
 name="Portfolio",
 line=dict(color="#00ff88", width=2),
 fill="tozeroy",
 fillcolor="rgba(0, 255, 136, 0.1)",
 ))
 
 fig.add_trace(go.Scatter(
 x=dates,
 y=benchmark_values,
 name="Benchmark (SPY)",
 line=dict(color="#666666", width=2, dash="dash"),
 ))
 
 fig.update_layout(
 template="plotly_dark",
 height=400,
 margin=dict(l=50, r=50, t=30, b=50),
 xaxis_title="Date",
 yaxis_title="Portfolio Value",
 legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
 hovermode="x unified",
 )
 
 st.plotly_chart(fig, use_container_width=True)
 
 # Rolling returns table
 st.subheader("📊 Rolling Returns")
 
 rolling_data = pd.DataFrame({
 "Horizon": ["MTD", "3 Month", "6 Month", "YTD", "1 Year"],
 "Portfolio": ["1.4%", "20.6%", "49.8%", "66.9%", "50.9%"],
 "Benchmark": ["0.6%", "5.4%", "14.0%", "17.3%", "14.0%"],
 })
 
 st.dataframe(rolling_data, use_container_width=True, hide_index=True)
 
 # Active weights table
 st.subheader("⚖️ Active Weights")
 
 weights_data = pd.DataFrame({
 "Rank": [1, 2, 3, 4, 5, 6, 7, 8],
 "Symbol": ["RGTI", "QBTS", "EOSE", "OKLO", "WDC", "GE", "BE", "APH"],
 "Factor Signal": [85.66, 80.99, 73.57, 63.98, 63.83, 62.48, 59.28, 56.76],
 "Weight %": ["10%"] * 8,
 "12M Return": ["484.6%", "476.3%", "116.5%", "400.7%", "239.1%", "65.5%", "281.0%", "88.2%"],
 })
 
 st.dataframe(weights_data, use_container_width=True, hide_index=True)


def render_research_lab():
 """Render the Research Lab tab."""
 
 st.subheader("🔬 Research Lab")
 
 # Symbol selector
 col1, col2 = st.columns([1, 3])
 
 with col1:
 selected_symbol = st.selectbox(
 "Select Symbol",
 ["RGTI", "QBTS", "EOSE", "OKLO", "WDC", "GE", "BE", "APH"],
 )
 
 lookback = st.slider("Lookback (days)", 30, 504, 252)
 
 with col2:
 # Company info
 st.info(f"**{selected_symbol}** - Viewing {lookback} trading days")
 
 # Factor history chart
 fig = go.Figure()
 
 dates = pd.date_range(end=datetime.now(), periods=lookback, freq="D")
 factor_values = [50 + i * 0.15 + (i % 20 - 10) for i in range(lookback)]
 
 fig.add_trace(go.Scatter(
 x=dates,
 y=factor_values,
 name="Factor Signal",
 line=dict(color="#1f77b4", width=2),
 ))
 
 fig.update_layout(
 template="plotly_dark",
 height=300,
 title=f"{selected_symbol} Factor Signal History",
 )
 
 st.plotly_chart(fig, use_container_width=True)
 
 # Rankings table
 st.subheader("📊 Current Rankings")
 
 rankings = pd.DataFrame({
 "Rank": range(1, 21),
 "Symbol": ["RGTI", "QBTS", "EOSE", "OKLO", "WDC", "GE", "BE", "APH", "PLTR", "HOOD",
 "MU", "GLW", "IREN", "SBSW", "NEM", "IAG", "BCS", "KGC", "GFI", "TSLA"],
 "Factor Signal": [85.66, 80.99, 73.57, 63.98, 63.83, 62.48, 59.28, 56.76, 55.12, 54.89,
 52.34, 51.22, 50.11, 49.88, 48.76, 47.65, 46.54, 45.43, 44.32, 43.21],
 "Sector": ["Tech", "Tech", "Energy", "Energy", "Tech", "Industrial", "Energy", "Tech",
 "Tech", "Finance", "Tech", "Tech", "Tech", "Mining", "Mining", "Mining",
 "Finance", "Mining", "Mining", "Automotive"],
 })
 
 st.dataframe(rankings, use_container_width=True, hide_index=True)


def render_signal_charts():
 """Render the Signal Charts tab."""
 
 st.subheader("📈 Technical Analysis")
 
 # Symbol selector
 col1, col2 = st.columns([1, 4])
 
 with col1:
 symbol = st.selectbox(
 "Symbol",
 ["RGTI", "AAPL", "MSFT", "GOOGL", "AMZN"],
 key="signal_symbol",
 )
 
 timeframe = st.selectbox(
 "Timeframe",
 ["1M", "3M", "6M", "YTD", "1Y", "2Y"],
 )
 
 st.write("**Indicators**")
 show_bb = st.checkbox("Bollinger Bands", value=True)
 show_ema = st.checkbox("EMA 21", value=True)
 show_sma = st.checkbox("SMA 50/200", value=True)
 
 with col2:
 # Candlestick chart
 dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
 
 # Generate OHLC data
 import numpy as np
 np.random.seed(42)
 
 base_price = 50
 prices = []
 for i in range(100):
 open_price = base_price + np.random.randn() * 2
 close_price = open_price + np.random.randn() * 3
 high_price = max(open_price, close_price) + abs(np.random.randn()) * 1.5
 low_price = min(open_price, close_price) - abs(np.random.randn()) * 1.5
 prices.append([open_price, high_price, low_price, close_price])
 base_price = close_price
 
 df = pd.DataFrame(prices, columns=["Open", "High", "Low", "Close"], index=dates)
 
 fig = go.Figure()
 
 # Candlestick
 fig.add_trace(go.Candlestick(
 x=df.index,
 open=df["Open"],
 high=df["High"],
 low=df["Low"],
 close=df["Close"],
 name=symbol,
 ))
 
 # Bollinger Bands
 if show_bb:
 sma = df["Close"].rolling(20).mean()
 std = df["Close"].rolling(20).std()
 
 fig.add_trace(go.Scatter(
 x=df.index, y=sma + 2*std,
 name="BB Upper", line=dict(color="rgba(100,100,100,0.5)"),
 ))
 fig.add_trace(go.Scatter(
 x=df.index, y=sma - 2*std,
 name="BB Lower", line=dict(color="rgba(100,100,100,0.5)"),
 fill="tonexty", fillcolor="rgba(100,100,100,0.1)",
 ))
 
 # EMA 21
 if show_ema:
 ema = df["Close"].ewm(span=21).mean()
 fig.add_trace(go.Scatter(
 x=df.index, y=ema,
 name="EMA 21", line=dict(color="#ff7f0e", width=1),
 ))
 
 # SMA 50/200
 if show_sma:
 sma50 = df["Close"].rolling(50).mean()
 fig.add_trace(go.Scatter(
 x=df.index, y=sma50,
 name="SMA 50", line=dict(color="#2ca02c", width=1),
 ))
 
 fig.update_layout(
 template="plotly_dark",
 height=500,
 xaxis_rangeslider_visible=False,
 title=f"{symbol} Price Chart",
 )
 
 st.plotly_chart(fig, use_container_width=True)


def render_ai_quant():
 """Render the AI Quant Team tab."""
 
 st.subheader("🤖 AI Quant Team")
 
 st.info("Ask questions about the portfolio, strategy performance, or market research.")
 
 # Chat interface
 if "messages" not in st.session_state:
 st.session_state.messages = []
 
 for message in st.session_state.messages:
 with st.chat_message(message["role"]):
 st.markdown(message["content"])
 
 if prompt := st.chat_input("Ask the AI Quant Team..."):
 st.session_state.messages.append({"role": "user", "content": prompt})
 
 with st.chat_message("user"):
 st.markdown(prompt)
 
 # Simulated AI response
 response = f"I analyzed your question about **{prompt[:50]}**... This is a placeholder response. Connect your OpenAI API key to enable AI-powered research assistance."
 
 with st.chat_message("assistant"):
 st.markdown(response)
 
 st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
 try:
 main()
 except Exception as e:
 st.error(f"Dashboard error: {e}")
 st.info("Please check that the database is initialized. Run: python scripts/setup_database.py")
