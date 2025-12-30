"""
Twelve Data Real-Time Feed
Uses WebSocket to stream real-time price quotes and feed them to the dashboard.
API Key provided by user.
"""
import websocket
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qsconnect.database.duckdb_manager import DuckDBManager
from omega.data.candle_engine import CandleAggregator, Tick, SessionManager

# API Configuration
API_KEY = "2b20a28a0c5040f082c35cb6f95a75c2"
SYMBOLS = ["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "EUR/USD", "BTC/USD"]

# Force override session check to allow 24/7 processing (Crypto/Forex/Testing)
class PermissiveSession(SessionManager):
    def is_in_session(self, ts, symbol): return True

# Globals
db_path = Path(__file__).parent.parent / "data" / "quant.duckdb"
DB_MGR = DuckDBManager(db_path, read_only=False)
AGGREGATORS = {}

def on_message(ws, message):
    try:
        data = json.loads(message)
        event = data.get("event")
        
        # Twelve Data 'price' event (Quote)
        if event == "price":
            # Sample: {"event":"price","symbol":"AAPL","currency":"USD","exchange":"NASDAQ",
            #          "type":"Common Stock","timestamp":1616...,"price":121.21,"day_volume":0}
            symbol = data.get("symbol")
            price = float(data.get("price", 0))
            ts = data.get("timestamp", time.time())
            
            # Map common variations
            if symbol == "EUR/USD": symbol = "EURUSD" # Adjust if needed by DB
            
            if symbol in AGGREGATORS:
                tick = Tick(
                    symbol=symbol,
                    price=price,
                    size=100, # Fake size if missing
                    exchange_ts=ts,
                    recv_ts=time.time()
                )
                
                # Process Tick
                AGGREGATORS[symbol].process_tick(tick)
                print(f"Update {symbol}: {price:.2f}")
                
                # Release Lock immediately
                DB_MGR.close()
                
        elif event == "heartbeat":
            print(".", end="", flush=True)
            
        elif data.get("status") == "error":
            print(f"\n❌ API Error: {data.get('message')}")
            
    except Exception as e:
        print(f"\nError processing message: {e}")

def on_error(ws, error):
    print(f"\nWS Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("\nWS Connection Closed")

def on_open(ws):
    print("✅ Connected to Twelve Data. Subscribing...")
    
    # Subscribe to price quotes (no interval = ticks/quotes)
    payload = {
        "action": "subscribe",
        "params": {
            "symbols": ",".join(SYMBOLS)
        }
    }
    ws.send(json.dumps(payload))

def main():
    print("🚀 Starting Twelve Data Real-Time Feed...")
    print(f"   Symbols: {SYMBOLS}")
    
    # Initialize Aggregators
    for sym in SYMBOLS:
        AGGREGATORS[sym] = CandleAggregator(
            symbol=sym, 
            session_mgr=PermissiveSession(),
            db_mgr=DB_MGR
        )
    
    # Run WebSocket
    # URL: The main endpoint for streaming
    ws_url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={API_KEY}"
    
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")

if __name__ == "__main__":
    main()
