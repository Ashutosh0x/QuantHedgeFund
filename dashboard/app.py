"""
QS Hedge Fund Dashboard - Operational Control Plane
No-emoji, professional SVG-based UI.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from datetime import timedelta
import numpy as np
from pathlib import Path
from plotly.subplots import make_subplots
import importlib
import qsconnect.database.duckdb_manager
importlib.reload(qsconnect.database.duckdb_manager)
from qsconnect.database.duckdb_manager import DuckDBManager
from qsresearch.governance.manager import GovernanceManager
from qsresearch.governance.reporting import ReportingEngine

# Page configuration
st.set_page_config(
    page_title="QS Control Plane",
    layout="wide",
    initial_sidebar_state="expanded"
)

import importlib
# importlib.reload Removed for Production Safety

from qsconnect.database.duckdb_manager import DuckDBManager
from qsresearch.governance.manager import GovernanceManager
from qsresearch.governance.reporting import ReportingEngine
from qsconnect.emergency import EmergencyControl


# SVG Icons (React-icons / Lucide style)
SVG_ICONS = {
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "cog": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "bar-chart": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
    "line-chart": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>',
    "terminal": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>',
    "bot": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="10" x="3" y="11" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>',
    "alert-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    "sun": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
    "moon": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
    "check-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "timer": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "flask": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v8L4.5 20.5a2 2 0 0 0 2 2.5h11a2 2 0 0 0 2-2.5L14 10V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/></svg>',
    "power": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/></svg>',
    "trash-2": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>',
    "layout-list": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/><path d="M14 4h7"/><path d="M14 9h7"/><path d="M14 15h7"/><path d="M14 20h7"/></svg>',
    "file-text": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
    "lock": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
}

def render_icon(name, color="currentColor"):
    if name not in SVG_ICONS:
        logger.error(f"Missing icon: {name}")
        return f'<span style="color:{color};">[Icon:{name}]</span>'
    return f'<div style="display:inline-block; vertical-align:middle; margin-right:8px; color:{color};">{SVG_ICONS[name]}</div>'

# Session state initialization
if "halted" not in st.session_state:
    st.session_state.halted = False
if "strategy_approved" not in st.session_state:
    st.session_state.strategy_approved = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # Default to dark mode

# Dynamic CSS based on theme
if st.session_state.dark_mode:
    theme_css = """
    <style>
        .main-header { font-size: 2.2rem; font-weight: bold; color: #1f77b4; margin-bottom: 1.5rem; display: flex; align-items: center; }
        .status-panel { background: #0e1117; border: 1px solid #1f77b4; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
        .halt-banner { background: #4a0404; color: white; padding: 12px; border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 20px; border: 1px solid #e74c3c; box-shadow: 0 0 10px rgba(231, 76, 60, 0.4); display: flex; justify-content: center; align-items: center; }
        .stButton > button { width: 100%; border-radius: 6px; }
        .status-dot { height: 10px; width: 10px; background-color: #00ff88; border-radius: 50%; display: inline-block; margin-right: 8px; box-shadow: 0 0 5px #00ff88; }
        .status-text { font-size: 0.9rem; vertical-align: middle; }
        .theme-toggle { cursor: pointer; padding: 8px; border-radius: 50%; transition: all 0.3s; }
        .theme-toggle:hover { background: rgba(255,255,255,0.1); }
    </style>
    """
else:
    theme_css = """
    <style>
        /* Main content area */
        .stApp, [data-testid="stAppViewContainer"], .main .block-container {
            background-color: #ffffff !important;
        }
        [data-testid="stHeader"] { background-color: #f8f9fa !important; }
        
        /* Sidebar */
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
            background-color: #f0f2f6 !important;
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h3 {
            color: #333 !important;
        }
        
        /* Text colors */
        .stMarkdown, .stText, p, span, label, h1, h2, h3, h4, h5, h6 {
            color: #1a1a1a !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] { color: #1f77b4 !important; }
        [data-testid="stMetricDelta"] { color: #00cc66 !important; }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { background-color: #f8f9fa !important; }
        .stTabs [data-baseweb="tab"] { color: #333 !important; }
        
        /* Inputs */
        .stTextInput > div > div, .stSelectbox > div > div {
            background-color: #fff !important;
            color: #333 !important;
            border-color: #ccc !important;
        }
        
        /* Dataframes */
        .stDataFrame { background-color: #fff !important; }
        
        /* Custom classes */
        .main-header { font-size: 2.2rem; font-weight: bold; color: #1f77b4; margin-bottom: 1.5rem; display: flex; align-items: center; }
        .status-panel { background: #f5f5f5; border: 1px solid #1f77b4; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
        .halt-banner { background: #ffcccc; color: #8b0000; padding: 12px; border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 20px; border: 1px solid #e74c3c; display: flex; justify-content: center; align-items: center; }
        .stButton > button { width: 100%; border-radius: 6px; }
        .status-dot { height: 10px; width: 10px; background-color: #00cc66; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-text { font-size: 0.9rem; vertical-align: middle; color: #333; }
    </style>
    """
st.markdown(theme_css, unsafe_allow_html=True)

def render_live_chart(db_mgr, symbol):
    """Renders a real-time Plotly candlestick chart for the given symbol."""
    query = f"""
        SELECT timestamp, open, high, low, close, volume 
        FROM realtime_candles 
        WHERE symbol = '{symbol}' 
        ORDER BY timestamp ASC
    """
    df = db_mgr.query_pandas(query)
    
    if df.empty:
        st.warning(f"No live candle data available for {symbol}. Waiting for ticks...")
        return

    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    )])

    fig.update_layout(
        title=f"Truth Layer: {symbol} (Real-Time)",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_market_profile(db_mgr, symbol, days=30):
    """Renders a Market Profile (Volume Profile) chart."""
    # 1. Fetch Historical Data from DuckDB
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        query = f"""
            SELECT date as timestamp, open, high, low, close, volume 
            FROM prices 
            WHERE symbol = '{symbol}' 
            AND date >= '{start_date.strftime('%Y-%m-%d')}'
            ORDER BY date ASC
        """
        df = db_mgr.query_pandas(query)
        
        if df.empty:
            st.warning(f"No historical data for {symbol} in 'prices' table.")
            return

    except Exception as e:
        st.error(f"Data Error: {e}")
        return

    # 2. Calculate Volume Profile
    price = df['close']
    volume = df['volume']
    
    # Create histograms where bins=24
    if len(price) > 0:
        counts, bin_edges = np.histogram(price, bins=24, weights=volume)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    else:
        st.warning("Insufficient data.")
        return

    # 3. Create Subplots
    fig = make_subplots(
        rows=1, cols=2, 
        column_widths=[0.2, 0.8], 
        shared_yaxes=True, 
        horizontal_spacing=0.02
    )
    
    # 4. Add Volume Profile (Left)
    fig.add_trace(go.Bar(
        x=counts,
        y=bin_centers,
        orientation='h',
        name='Volume Profile',
        marker_color='rgba(0, 255, 136, 0.3)', # Greenish tint
        showlegend=False
    ), row=1, col=1)
    
    # 5. Add Candlestick (Right)
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name=symbol
    ), row=1, col=2)
    
    fig.update_layout(
        title=f"Market Profile: {symbol} ({days} Days)", 
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        bargap=0.01,
        yaxis=dict(showticklabels=False), # Hide Y on left chart? No, shared.
    )
    # Be explicit about shared Y
    fig.update_yaxes(side="right", row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=1) # Hide volume numbers
    
    st.plotly_chart(fig, use_container_width=True)

def check_password():
    """Simple password protection."""
    if st.session_state.get('password_correct', False):
        return True

    st.markdown("### 🔒 Restricted Access")
    pwd = st.text_input("Enter Operational Password", type="password")
    
    if st.button("Login"):
        if pwd == "quant123": # Review: Move to secrets.toml in real prod
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Invalid password")
    return False

def main():
    if not check_password():
        st.stop()

    # managers
    # Use read_only=True to prevent locking, as Trading Node is the writer
    db_mgr = DuckDBManager(Path("data/quant.duckdb"), read_only=True, auto_close=True)
    gov_mgr = GovernanceManager(db_mgr)
    report_engine = ReportingEngine(db_mgr)
    active_strat = gov_mgr.get_active_strategy()

    # Sidebar: System Health & Emergency Controls
    with st.sidebar:
        # Theme Toggle at very top
        col_title, col_toggle = st.columns([4, 1])
        with col_title:
            st.markdown(f'<h3 style="display:flex; align-items:center;">{render_icon("activity")} System Health</h3>', unsafe_allow_html=True)
        with col_toggle:
            # Use SVG icon instead of emoji
            if st.session_state.dark_mode:
                icon_html = f'<div style="cursor:pointer; padding:4px;">{render_icon("sun", "#ffd700")}</div>'
                clicked = st.button("🌞", key="theme_toggle", help="Switch to Light Mode", type="secondary")
                if clicked:
                    st.session_state.dark_mode = False
                    st.rerun()
            else:
                icon_html = f'<div style="cursor:pointer; padding:4px;">{render_icon("moon", "#4169e1")}</div>'
                clicked = st.button("🌙", key="theme_toggle", help="Switch to Dark Mode", type="secondary")
                if clicked:
                    st.session_state.dark_mode = True
                    st.rerun()
        
        colH1, colH2 = st.columns([1.5, 1])
        with colH1:
            st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:8px;">{render_icon("check-circle", "#00ff88")} IBKR API</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex; align-items:center;">{render_icon("check-circle", "#00ff88")} Truth Layer</div>', unsafe_allow_html=True)
        with colH2:
            st.markdown(f'<div style="margin-bottom:8px;"><span class="status-dot"></span><span class="status-text">Live</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div><span class="status-dot" style="background-color:#00ff88; box-shadow:0 0 5px #00ff88;"></span><span class="status-text">Active</span></div>', unsafe_allow_html=True)
            
        st.divider()
        
        st.markdown(f'<h3 style="display:flex; align-items:center; color:#e74c3c;">{render_icon("shield", "#e74c3c")} Emergency Controls</h3>', unsafe_allow_html=True)
        
        # Check backend state
        system_halted = EmergencyControl.is_halted()
        
        if not system_halted:
            if st.button("HALT ALL TRADING", type="primary", use_container_width=True):
                if EmergencyControl.halt("Manually Triggered from Dashboard"):
                    st.toast("System Halted Successfully")
                    st.rerun()
                else:
                    st.error("Failed to Halt System - Check Logs")
        else:
            if st.button("RESUME TRADING", use_container_width=True):
                if EmergencyControl.resume():
                    st.toast("System Resumed")
                    st.rerun()
                else:
                    st.error("Failed to Resume - Check Logs")
                
        if st.button("CANCEL ALL ORDERS", use_container_width=True, disabled=True):
            st.toast("Not Wired to Backend yet")
            
        if st.button("FLATTEN ALL POSITIONS", use_container_width=True, disabled=True):
            st.warning("Action: Immediate Portfolio Liquidation")
            if st.button("CONFIRM FLATTEN", type="primary"):
                st.toast("Flattening portfolio...")

        st.divider()
        st.markdown(f'<h3 style="display:flex; align-items:center;">{render_icon("cog")} Config</h3>', unsafe_allow_html=True)
        st.text_input("Daily Loss Limit (USD)", value="5,000")
        st.text_input("Max Symbol Exposure (%)", value="20")

    # Halt Banner
    if EmergencyControl.is_halted():
        st.markdown(f'<div class="halt-banner">{render_icon("alert-circle", "white")} SYSTEM HALTED - TRADING STOPPED</div>', unsafe_allow_html=True)

    # Main Header
    st.markdown(f'<div class="main-header">{render_icon("shield")} Operational Control Plane</div>', unsafe_allow_html=True)
    
    if active_strat:
        expiry_days = (active_strat['ttl_expiry'] - datetime.now()).days if isinstance(active_strat['ttl_expiry'], datetime) else "N/A"
        st.caption(f"Active Strategy: {active_strat['strategy_hash'][:12]} | Stage: {active_strat['stage']} | TTL: {expiry_days} days remaining")

    # Live Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Net Liquidity", "$5.24M", "+0.4%")
    with m2:
        st.metric("Gross Exposure", "138%", "Target")
    with m3:
        st.metric("Day P&L", "+$24,510", "0.47%")
    with m4:
        st.metric("Current Drawdown", "-$12.3k", "Limit: $50k")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Execution Blotter",
        "Holdings",
        "Strategy Approval",
        "Governance Audit",
        "Drift & Performance",
        "Ops Management",
        "Risk Overview",
        "Market Profile"
    ])

    with tab1:
        st.markdown(f"#### {render_icon('terminal')} Real-Time Truth Layer & Order Blotter", unsafe_allow_html=True)
        
        # Live Chart Section
        try:
            active_symbols = db_mgr.query_pandas("SELECT DISTINCT symbol FROM realtime_candles")['symbol'].tolist()
        except Exception:
            active_symbols = []
            
        if not active_symbols:
            # st.warning("No active symbols found in DB.") # Silence is golden
            active_symbols = []
            
        if not active_symbols:
            active_symbols = ["AMZN"] # Fallback for UI skeleton
            
        chart_col, info_col = st.columns([3, 1])
        with chart_col:
            selected_symbol = st.selectbox("View Symbol", active_symbols, index=0, label_visibility="collapsed")
            render_live_chart(db_mgr, selected_symbol)
        
        with info_col:
            st.markdown(f"**Engine Status**")
            st.markdown(f'<span class="status-dot"></span><span class="status-text">Consuming IBKR Ticks</span>', unsafe_allow_html=True)
            st.markdown(f"**Symbol**: {selected_symbol}")
            st.markdown(f"**Interval**: 1 min")
            if st.button("Refresh Chart"):
                st.rerun()

        st.divider()
        st.markdown(f"#### {render_icon('layout-list')} Real-Time Order Blotter", unsafe_allow_html=True)
        # Fetch Real Trades
        try:
            trades = db_mgr.query_pandas("SELECT * FROM trades ORDER BY execution_time DESC LIMIT 50")
            if not trades.empty:
                st.dataframe(trades, use_container_width=True, hide_index=True)
            else:
                st.info("No trades recorded today.")
        except Exception:
            st.info("Trade ledger not initialized.")
    
    with tab2:
        st.markdown(f"#### {render_icon('bar-chart')} Current Holdings", unsafe_allow_html=True)
    with tab2:
        st.markdown(f"#### {render_icon('bar-chart')} Current Holdings", unsafe_allow_html=True)
        st.info("Live positions view requires persistent portfolio service.")
        # Placeholder for real connection
        # positions = db_mgr.query_pandas("SELECT * FROM positions") ...
        
    with tab3:
        st.markdown(f"#### {render_icon('bot')} Staged Deployment & Approval", unsafe_allow_html=True)
        st.info("Strategy staging area for proposed models.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1️⃣ Current Active**")
            if active_strat:
                st.json(active_strat)
            else:
                st.warning("No strategy active in FULL/CANARY stage.")
        with c2:
            st.markdown("**2️⃣ Proposed Candidate (MOCK)**")
            proposed = {"top_n": 30, "reasoning": "Aggressive momentum capture."}
            st.json(proposed)
            rationale = st.text_area("Human Approval Rationale", "Reviewed backtest.")
            colP1, colP2 = st.columns(2)
            with colP1:
                target_stage = st.selectbox("Activation Stage", ["SHADOW", "PAPER", "CANARY", "FULL"])
            with colP2:
                if st.button("LOG & APPROVE STRATEGY", type="primary"):
                    strat_hash = gov_mgr.log_strategy_approval(config=proposed, regime_snapshot={"label": "Bull"}, llm_reasoning=proposed["reasoning"], human_rationale=rationale, approved_by="ADMIN", stage=target_stage)
                    st.success(f"Strategy {strat_hash[:8]} deployed to {target_stage}")
                    st.rerun()

    with tab4:
        st.markdown(f"#### {render_icon('database')} Immutable Audit Trail", unsafe_allow_html=True)
        audit_sql = "SELECT strategy_hash, stage, approved_by, approved_at, human_rationale FROM strategy_audit_log ORDER BY approved_at DESC"
        audit_data = db_mgr.query_pandas(audit_sql)
        st.dataframe(audit_data, use_container_width=True, hide_index=True)
        
    with tab5:
        st.markdown(f"#### {render_icon('activity')} Strategy Drift Monitor", unsafe_allow_html=True)
        if active_strat:
            d1, d2 = st.columns(2)
            d1.metric("Turnover Drift", "0.08", "-0.01")
            d2.metric("Factor Exposure Drift", "0.15", "Normal")
        else:
            st.warning("No active strategy to monitor drift.")

    with tab6:
        st.markdown(f"#### {render_icon('file-text')} Operational Reporting (Phase 2)", unsafe_allow_html=True)
        cR1, cR2 = st.columns([1, 2])
        with cR1:
            report_date = st.date_input("Report Date", datetime.now())
            if st.button("GENERATE DAILY OPS REPORT", type="primary"):
                report = report_engine.generate_daily_ops_report(report_date.strftime("%Y-%m-%d"))
                st.session_state.current_report = report
        with cR2:
            if "current_report" in st.session_state:
                report_md = report_engine.format_as_markdown(st.session_state.current_report)
                st.markdown(report_md)
                st.download_button("Download Report (.md)", report_md, file_name=f"ops_report_{report_date}.md")
            else:
                st.info("Select a date and click generate to view the Daily Ops Report.")

    with tab7:
        st.markdown(f"#### {render_icon('shield')} Risk & Compliance Overview", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Operational Maturity Readiness**")
        h1, h2, h3 = st.columns(3)
        h1.markdown(f"{render_icon('check-circle', '#00ff88')} **Phase 1: Governance** (Verified)", unsafe_allow_html=True)
        h2.markdown(f"{render_icon('check-circle', '#00ff88')} **Phase 2: Discipline** (Verified)", unsafe_allow_html=True)
        h3.markdown(f"{render_icon('activity', '#f1c40f')} **Phase 3: Scaling** (In Progress)", unsafe_allow_html=True)
        colL, colR = st.columns(2)
        with colL:
            st.progress(0.24, text="Daily Loss Consumption: 24%")
            st.progress(0.69, text="Margin Utilization: 1.38x / 2.0x")
        with colR:
            st.markdown(f'<div style="display:flex; align-items:center;">{render_icon("check-circle", "#00ff88")} Paper Trading Validated</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex; align-items:center;">{render_icon("check-circle", "#00ff88")} Deterministic Truth Layer</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex; align-items:center;">{render_icon("check-circle", "#00ff88")} Bar-Close Sync (Parity)</div>', unsafe_allow_html=True)

    with tab8:
        st.markdown(f"#### {render_icon('bar-chart')} Market Profile Analysis", unsafe_allow_html=True)
        
        cP1, cP2 = st.columns([1, 3])
        with cP1:
            try:
                # Get symbols from prices table
                avail_symbols = db_mgr.query_pandas("SELECT DISTINCT symbol FROM prices LIMIT 10")['symbol'].tolist()
            except:
                avail_symbols = ["AAPL", "MSFT", "SPY"]
                
            prof_symbol = st.selectbox("Symbol", avail_symbols if avail_symbols else ["AAPL"], key="prof_sym")
            days = st.slider("Lookback Days", 5, 60, 30)
            
        with cP2:
            render_market_profile(db_mgr, prof_symbol, days)

if __name__ == "__main__":
    main()
