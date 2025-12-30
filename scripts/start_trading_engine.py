"""
Start Trading Engine Script

Launches the Omega Trading App (Backend) to:
1. Connect to Interactive Brokers
2. Subscribe to Market Data (Truth Layer)
3. Execute Algo Strategies
"""
import sys
import os
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omega.trading_app import TradingApp
from loguru import logger

def main():
    logger.info("🚀 Starting Omega Trading Engine...")
    
    # 1. Initialize
    app = TradingApp(paper_trading=True)
    
    # 2. Connect
    if not app.connect():
        logger.error("❌ Failed to connect to IB Gateway. Exiting.")
        sys.exit(1)
        
    # 3. Subscribe to Data (The Fix for Dashboard)
    symbols = ["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA"]
    logger.info(f"Subscribing to Truth Layer for: {symbols}")
    
    for symbol in symbols:
        app.subscribe_truth_layer(symbol)
        
    logger.success("✅ Trading Engine Running. Press Ctrl+C to stop.")
    
    # 4. Event Loop
    try:
        app.run_blocking()
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping Trading Engine...")
        app.disconnect()
        sys.exit(0)

if __name__ == "__main__":
    main()
