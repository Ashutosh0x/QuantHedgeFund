"""
QS Hedge Fund Dashboard - Operational Control Plane
No-emoji, professional SVG-based UI.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="QS Control Plane",
    page_icon="https://quantscience.io/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    "check-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "timer": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "flask": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v8L4.5 20.5a2 2 0 0 0 2 2.5h11a2 2 0 0 0 2-2.5L14 10V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/></svg>',
    "power": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/></svg>',
    "trash-2": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>',
    "layout-list": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/><path d="M14 4h7"/><path d="M14 9h7"/><path d="M14 15h7"/><path d="M14 20h7"/></svg>',
}

def render_icon(name, color="currentColor"):
    return f'<div style="display:inline-block; vertical-align:middle; margin-right:8px; color:{color};">{SVG_ICONS[name]}</div>'

# Session state initialization
if "halted" not in st.session_state:
    st.session_state.halted = False
if "strategy_approved" not in st.session_state:
    st.session_state.strategy_approved = False

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: bold; color: #1f77b4; margin-bottom: 1.5rem; display: flex; align-items: center; }
    .status-panel { background: #0e1117; border: 1px solid #1f77b4; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
    .halt-banner { background: #4a0404; color: white; padding: 12px; border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 20px; border: 1px solid #e74c3c; box-shadow: 0 0 10px rgba(231, 76, 60, 0.4); display: flex; justify-content: center; align-items: center; }
    .stButton > button { width: 100%; border-radius: 6px; }
    .status-dot { height: 10px; width: 10px; background-color: #00ff88; border-radius: 50%; display: inline-block; margin-right: 8px; box-shadow: 0 0 5px #00ff88; }
    .status-text { font-size: 0.9rem; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar: System Health & Emergency Controls
    with st.sidebar:
        st.markdown(f'<h3 style="display:flex; align-items:center;">{render_icon("activity")} System Health</h3>', unsafe_allow_html=True)
        
        colH1, colH2 = st.columns([1.5, 1])
        with colH1:
            st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:8px;">{render_icon("check-circle", "#00ff88")} IBKR API</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex; align-items:center;">{render_icon("check-circle", "#00ff88")} Data Feed</div>', unsafe_allow_html=True)
        with colH2:
            st.markdown(f'<div style="margin-bottom:8px;"><span class="status-dot"></span><span class="status-text">Live</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div><span class="status-dot"></span><span class="status-text">Online</span></div>', unsafe_allow_html=True)
            
        st.divider()
        
        st.markdown(f'<h3 style="display:flex; align-items:center; color:#e74c3c;">{render_icon("shield", "#e74c3c")} Emergency Controls</h3>', unsafe_allow_html=True)
        
        if not st.session_state.halted:
            if st.button("HALT ALL TRADING", type="primary", use_container_width=True):
                st.session_state.halted = True
                st.rerun()
        else:
            if st.button("RESUME TRADING", use_container_width=True):
                st.session_state.halted = False
                st.rerun()
                
        if st.button("CANCEL ALL ORDERS", use_container_width=True):
            st.toast("Cancelling all open orders...")
            
        if st.button("FLATTEN ALL POSITIONS", use_container_width=True):
            st.warning("Action: Immediate Portfolio Liquidation")
            if st.button("CONFIRM FLATTEN", type="primary"):
                st.toast("Flattening portfolio...")

        st.divider()
        st.markdown(f'<h3 style="display:flex; align-items:center;">{render_icon("cog")} Config</h3>', unsafe_allow_html=True)
        st.text_input("Daily Loss Limit (USD)", value="5,000")
        st.text_input("Max Symbol Exposure (%)", value="20")



    # Halt Banner
    if st.session_state.halted:
        st.markdown(f'<div class="halt-banner">{render_icon("alert-circle", "white")} SYSTEM HALTED - NO TRADES ALLOWED</div>', unsafe_allow_html=True)

    # Main Header
    st.markdown(f'<div class="main-header">{render_icon("shield")} Operational Control Plane</div>', unsafe_allow_html=True)

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Execution Blotter",
        "Holdings",
        "Strategy Controls",
        "Latency Monitor",
        "Risk Overview"
    ])


    with tab1:
        st.markdown(f"#### {render_icon('terminal')} Real-Time Order Blotter", unsafe_allow_html=True)
        orders = pd.DataFrame([
            {"ID": "ORD_101", "Time": "17:28:01", "Symbol": "AAPL", "Side": "BUY", "Qty": 100, "Fill": 100, "Type": "ADAPTIVE", "Status": "FILLED", "Slippage": "1.2 bps"},
            {"ID": "ORD_102", "Time": "17:29:10", "Symbol": "MSFT", "Side": "SELL", "Qty": 50, "Fill": 0, "Type": "LIMIT", "Status": "SUBMITTED", "Slippage": "0.0 bps"},
            {"ID": "ORD_103", "Time": "17:29:45", "Symbol": "TSLA", "Side": "BUY", "Qty": 200, "Fill": 45, "Type": "ADAPTIVE", "Status": "PARTIAL", "Slippage": "0.8 bps"},
        ])
        st.dataframe(orders, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown(f"#### {render_icon('bar-chart')} Current Holdings", unsafe_allow_html=True)
        positions = pd.DataFrame({
            "Symbol": ["RGTI", "QBTS", "EOSE", "OKLO", "WDC"],
            "Qty": [5000, 2000, 1000, 500, 300],
            "Market Value": ["$520,000", "$210,000", "$98,000", "$45,000", "$32,000"],
            "Unrealized P&L": ["+$42k", "-$1.2k", "+$5.5k", "+$8k", "-$500"],
            "Weight %": ["9.9%", "4.0%", "1.9%", "0.9%", "0.6%"]
        })
        st.dataframe(positions, use_container_width=True, hide_index=True)
        
    with tab3:
        st.markdown(f"#### {render_icon('bot')} Human-in-the-Loop Strategy Approval", unsafe_allow_html=True)
        st.info("Strategy staging area for 'openai/gpt-oss-120b' proposals.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Active Strategy: v2**")
            st.json({"momentum": 0.4, "value": 0.3, "quality": 0.3})
        with c2:
            st.markdown("**Proposed Strategy: v3**")
            st.json({
                "momentum": 0.6,
                "value": 0.2,
                "quality": 0.2,
                "reasoning": "Increased momentum allocation for high-volatility bull regime."
            })
        
        if not st.session_state.strategy_approved:
            if st.button("APPROVE AND DEPLOY v3", type="primary"):
                st.session_state.strategy_approved = True
                st.success("Configuration v3 pushed to Live Engine.")
                st.rerun()
        else:
            st.success("Active Engine State: v3 (Approved)")
            if st.button("REVERT TO PREVIOUS"):
                st.session_state.strategy_approved = False
                st.rerun()
                
    with tab4:
        st.markdown(f"#### {render_icon('timer')} Network & Inference Latency", unsafe_allow_html=True)
        l1, l2, l3 = st.columns(3)
        l1.metric("Signal Latency (P50)", "12ms")
        l2.metric("Execution Latency (P95)", "45ms")
        l3.metric("IBKR Ack Latency (P99)", "142ms")
        chart_data = pd.DataFrame(np.random.randn(20, 3) + 50, columns=['P50', 'P95', 'P99'])
        st.line_chart(chart_data)
        
    with tab5:
        st.markdown(f"#### {render_icon('shield')} Risk & Compliance Overview", unsafe_allow_html=True)
        colL, colR = st.columns(2)
        with colL:
            st.markdown(f'<div style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
            st.progress(0.24, text="Daily Loss Consumption: 24%")
            st.markdown(f'</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
            st.progress(0.69, text="Margin Utilization: 1.38x / 2.0x")
            st.markdown(f'</div>', unsafe_allow_html=True)
        
        with colR:
            st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:12px;">{render_icon("check-circle", "#00ff88")} Paper Trading Validated</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:12px;">{render_icon("check-circle", "#00ff88")} Latency Limits Checked</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex; align-items:center; color:#f39c12;">{render_icon("terminal", "#f39c12")} Pre-Trade Gate Enabled</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
