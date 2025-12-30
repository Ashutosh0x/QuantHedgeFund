"""
Start Mock Live Feed
Feeds simulated ticks to the real-time database for testing the dashboard.
Use this if you don't have active Interactive Brokers market data subscriptions.
"""
import sys
import os
import time
import random
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qsconnect.database.duckdb_manager import DuckDBManager
from omega.data.candle_engine import CandleAggregator, Tick, SessionManager

# Force override session check to allow 24/7 testing
class MockSession(SessionManager):
    def is_in_session(self, ts, symbol): return True

def main():
    print("🚀 Starting MOCK Data Feed (Simulated Exchange)...")
    print("   Target DB: data/quant.duckdb")
    
    # Connect to PROD database (same as Dashboard)
    db_mgr = DuckDBManager(db_path="data/quant.duckdb")
    
    symbols = ["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA"]
    aggs = {}
    
    # Initialize aggregators
    for sym in symbols:
        # Create table if not exists (handled by candle engine logic or first insert)
        aggs[sym] = CandleAggregator(
            symbol=sym, 
            session_mgr=MockSession(),
            db_mgr=db_mgr
        )
        print(f"   Initialized {sym} aggregator")
        
    # Initial prices
    prices = {
        "AMZN": 180.0,
        "AAPL": 170.0,
        "MSFT": 420.0,
        "GOOGL": 140.0,
        "TSLA": 200.0
    }
    
    print("\n✅ Simulation Running. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            for sym in symbols:
                # Random walk simulation
                prices[sym] += (random.random() - 0.5) * 0.5 # +/- $0.25 move
                
                now = time.time()
                tick = Tick(
                    symbol=sym,
                    price=round(prices[sym], 2),
                    size=random.randint(10, 500),
                    exchange_ts=now,
                    recv_ts=now
                )
                
                # print(f"Update {sym}: {tick.price:.2f}") # Too noisy
                aggs[sym].process_tick(tick)
            
            db_mgr.close()
            
            # Show heartbeat
            print(".", end="", flush=True)
            time.sleep(0.5) # 2 updates per second
            
    except KeyboardInterrupt:
        print("\n🛑 Stopped Mock Feed.")

if __name__ == "__main__":
    main()
