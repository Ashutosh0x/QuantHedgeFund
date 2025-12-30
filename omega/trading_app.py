"""
Omega - Trading Application

Main trading application for executing trades via Interactive Brokers.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from loguru import logger

from config.settings import get_settings
from qsresearch.governance.manager import GovernanceManager
from omega.data.candle_engine import CandleAggregator, BarCloseEventBus, Tick



class TradingApp:
    """
    Main trading application for Omega.
    
    Provides interface to Interactive Brokers for:
    - Account management
    - Position tracking
    - Order execution
    - Market data retrieval
    
    Example:
        >>> from omega import TradingApp
        >>> app = TradingApp()
        >>> positions = app.get_positions()
        >>> app.order_target_percent("AAPL", 0.05)
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        paper_trading: bool = True,
    ):
        """
        Initialize the trading application.
        
        Args:
            host: IB Gateway host (default from settings)
            port: IB Gateway port (default from settings)
            client_id: Client ID for connection
            paper_trading: Whether to use paper trading mode
        """
        settings = get_settings()
        
        self.host = host or settings.ib_host
        self.port = port or settings.ib_port
        self.client_id = client_id or settings.ib_client_id
        self.paper_trading = paper_trading
        
        self._connected = False
        self._ib = None
        self._halted = False  # Critical safety flag
        
        # Governance & Strategy Layer
        if not hasattr(self, '_db_manager'):
            from qsconnect.database.duckdb_manager import DuckDBManager
            from pathlib import Path
            self._db_manager = DuckDBManager(Path("data/quant.duckdb"))
            
        self.gov = GovernanceManager(self._db_manager)
        self.active_strategy: Optional[Dict[str, Any]] = None

        # --- Phase 3: Candle Truth Layer ---
        self.event_bus = BarCloseEventBus()
        self.aggregators: Dict[str, CandleAggregator] = {}
        
        # Telemetry & Metrics
        self.metrics = {
            "last_tick_time": None,
            "last_order_time": None,
            "order_latencies": [],
            "daily_pnl_initial_value": None
        }
        
        logger.info(
            f"TradingApp initialized: {self.host}:{self.port} "
            f"(paper={self.paper_trading})"
        )
    
    # =====================
    # Connection Management
    # =====================
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers.
        
        Returns:
            True if connection successful
        """
        try:
            from ib_insync import IB
            
            self._ib = IB()
            self._ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
            )
            
            # Register Global Tick Event
            self._ib.tickByTickEvent += self._on_tick_by_tick_all
            
            self._connected = True
            logger.info("Connected to Interactive Brokers. Tick engine initialized.")
            return True
            
        except ImportError:
            logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from Interactive Brokers")
    
    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self._connected and self._ib and self._ib.isConnected()

    def halt(self) -> None:
        """Emergency halt all trading activity."""
        self._halted = True
        logger.warning("TRADING HALTED: Manual emergency halt triggered.")

    def resume(self) -> None:
        """Resume trading activity after halt."""
        self._halted = False
        logger.info("TRADING RESUMED: Manual resume triggered.")

    def is_halted(self) -> bool:
        """Check if system is currently halted."""
        return self._halted

    def _on_tick_by_tick_all(self, ticker: Any, tick: Any):
        """Global handler for raw trade ticks from IBKR."""
        try:
            # We ONLY care about TRADE ticks (Last)
            if not hasattr(tick, 'price') or not hasattr(tick, 'size'):
                return
                
            symbol = ticker.contract.symbol
            if symbol not in self.aggregators:
                return
                
            # --- Safety: Clock-Skew Detection ---
            exchange_ts = tick.time.timestamp()
            recv_ts = datetime.now().timestamp()
            skew = abs(recv_ts - exchange_ts)
            
            from omega.data.candle_engine import MAX_CLOCK_SKEW_SEC
            if skew > MAX_CLOCK_SKEW_SEC:
                logger.error(f"CRITICAL CLOCK SKEW for {symbol}: {skew:.2f}s. Halting Truth Layer for safety.")
                self._halted = True
                return

            # Convert to our internal Truth Layer format
            truth_tick = Tick(
                symbol=symbol,
                price=float(tick.price),
                size=float(tick.size),
                exchange_ts=exchange_ts,
                recv_ts=recv_ts
            )
            
            self.aggregators[symbol].process_tick(truth_tick)
            self.metrics["last_tick_time"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Tick Processing Error for {ticker.contract.symbol}: {e}")

    def subscribe_truth_layer(self, symbol: str, interval_sec: int = 60):
        """Subscribes to raw trade ticks and attaches a deterministic aggregator."""
        if not self.is_connected():
            return
            
        if symbol in self.aggregators:
            return
            
        contract = self.create_contract(symbol)
        
        # Initialize Aggregator with DB support
        agg = CandleAggregator(
            symbol=symbol,
            interval_sec=interval_sec,
            event_bus=self.event_bus,
            db_mgr=self._db_manager
        )
        self.aggregators[symbol] = agg
        
        # Request stream (Last trade only - exact IBKR candle source)
        self._ib.reqTickByTickData(contract, "Last")
        logger.info(f"TRUTH LAYER ACTIVE for {symbol} ({interval_sec}s bars)")

    def _log_trade(self, trade: Any):
        """Log a filled trade to DuckDB for audit."""
        try:
            # Extract fill info
            fill = trade.fills[-1] if trade.fills else None
            if not fill:
                return

            strat_hash = self.active_strategy.get("strategy_hash", "UNKNOWN") if self.active_strategy else "MANUAL"
            
            conn = self._db_manager.connect()
            conn.execute("""
                INSERT INTO trades (
                    trade_id, strategy_hash, symbol, side, quantity, 
                    fill_price, execution_time, commission, slippage_bps, 
                    order_type, account_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{trade.order.orderId}_{int(fill.time.timestamp())}",
                strat_hash,
                trade.contract.symbol,
                trade.order.action,
                float(fill.execution.shares),
                float(fill.execution.price),
                fill.time,
                float(fill.commissionReport.commission if fill.commissionReport else 0.0),
                0.0, # Slippage calc placeholder
                trade.order.orderType,
                self.client_id
            ))
            logger.info(f"Trade LOGGED: {trade.contract.symbol} {trade.order.action} {fill.execution.shares} @ {fill.execution.price}")
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    
    # =====================
    # Account Information
    # =====================
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account values
        """
        if not self.is_connected():
            self.connect()
        
        account_values = self._ib.accountValues()
        
        info = {}
        for av in account_values:
            if av.tag in ["NetLiquidation", "TotalCashValue", "BuyingPower", "GrossPositionValue"]:
                info[av.tag] = float(av.value)
        
        return info
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        info = self.get_account_info()
        return info.get("NetLiquidation", 0.0)
    
    # =====================
    # Position Management
    # =====================
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of position dictionaries
        """
        if not self.is_connected():
            self.connect()
        
        positions = []
        
        for pos in self._ib.positions():
            avg_cost = pos.avgCost if pos.avgCost else 0.0
            quantity = pos.position
            
            # Get current market price for accurate valuation
            try:
                contract = pos.contract
                ticker = self._ib.reqMktData(contract, snapshot=True)
                self._ib.sleep(0.5)  # Brief wait for snapshot
                current_price = ticker.marketPrice() if ticker.marketPrice() else avg_cost
                self._ib.cancelMktData(contract)
            except Exception:
                current_price = avg_cost  # Fallback to avg_cost if price unavailable
                logger.warning(f"Could not get market price for {pos.contract.symbol}, using avg_cost")
            
            positions.append({
                "symbol": pos.contract.symbol,
                "quantity": quantity,
                "avg_cost": avg_cost,
                "current_price": current_price,
                "cost_basis": quantity * avg_cost,
                "market_value": quantity * current_price,
                "unrealized_pnl": quantity * (current_price - avg_cost),
                "contract": pos.contract,
            })
        
        return positions
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        positions = self.get_positions()
        for pos in positions:
            if pos["symbol"] == symbol:
                return pos
        return None
    
    # =====================
    # Order Execution
    # =====================
    
    def create_contract(self, symbol: str, sec_type: str = "STK", exchange: str = "SMART") -> Any:
        """
        Create an IB contract object.
        
        Args:
            symbol: Stock symbol
            sec_type: Security type (STK, OPT, FUT, etc.)
            exchange: Exchange name
            
        Returns:
            IB Contract object
        """
        from ib_insync import Stock
        
        return Stock(symbol, exchange, "USD")

    def _validate_risk(self, symbol: str, shares: int, current_price: float, side: str) -> bool:
        """
        Internal pre-trade risk validation.
        
        Checks:
        1. System halt state
        2. Max symbol exposure
        3. Portfolio leverage
        4. (Placeholder) Daily loss limit
        
        Returns:
            True if trade is safe to proceed
        """
        if self._halted:
            logger.error(f"RISK REJECTED: System is HALTED. Cannot {side} {symbol}.")
            return False

        # --- Governance Check: Staged Deployment & Expiry ---
        if not self.active_strategy:
            self.active_strategy = self.gov.get_active_strategy()
            
        if not self.active_strategy:
            logger.warning(f"RISK REJECTED: No active approved strategy found for {symbol}")
            return False
            
        # Check Expiry
        if datetime.now() > self.active_strategy.get("ttl_expiry", datetime.max):
            logger.error(f"RISK REJECTED: Strategy {self.active_strategy['strategy_hash'][:8]} has EXPIRED")
            self._halted = True # Fail safe
            return False
            
        # Deployment Stage Enforcement
        stage = self.active_strategy.get("stage", "SHADOW")
        if stage == "SHADOW":
            logger.info(f"SHADOW MODE: Signal for {side} {shares} {symbol} recorded. Execution suppressed.")
            return False
        
        if stage == "PAPER" and not self.paper_trading:
            logger.error(f"RISK REJECTED: PAPER strategy cannot run on LIVE account.")
            return False


        settings = get_settings()
        portfolio_value = self.get_portfolio_value()
        order_value = abs(shares * current_price)
        
        # 1. Individual Symbol Exposure
        current_pos = self.get_position(symbol)
        current_pos_value = current_pos["market_value"] if current_pos else 0.0
        new_pos_value = current_pos_value + (order_value if side == "BUY" else -order_value)
        
        exposure_pct = abs(new_pos_value) / portfolio_value
        if exposure_pct > settings.max_symbol_exposure_pct:
            logger.error(f"RISK REJECTED: {symbol} exposure would be {exposure_pct:.1%}, exceeds limit of {settings.max_symbol_exposure_pct:.1%}")
            return False

        # 2. Leverage Check
        # Gross position value = total value of all positions
        info = self.get_account_info()
        gross_value = info.get("GrossPositionValue", 0.0)
        new_gross_value = gross_value + order_value
        leverage = new_gross_value / portfolio_value
        
        if leverage > settings.max_leverage:
            logger.error(f"RISK REJECTED: Portfolio leverage would be {leverage:.2f}, exceeds limit of {settings.max_leverage}")
            return False

        return True
    
    def order_target_percent(
        self,
        symbol: str,
        target_percent: float,
        order_type: str = "ADAPTIVE",  # Changed from MKT for safety
    ) -> Optional[Any]:
        """
        Place order to reach target portfolio percentage.
        
        This is the primary method for portfolio rebalancing.
        
        Args:
            symbol: Stock symbol
            target_percent: Target allocation (0.0 to 1.0)
            order_type: Order type (ADAPTIVE, LMT, MKT - use MKT with caution)
            
        Returns:
            Order trade object
        """
        if not self.is_connected():
            self.connect()
        
        from ib_insync import MarketOrder, LimitOrder
        
        # Get portfolio value
        portfolio_value = self.get_portfolio_value()
        target_value = portfolio_value * target_percent
        
        # Get current position
        current_pos = self.get_position(symbol)
        current_value = current_pos["market_value"] if current_pos else 0.0
        
        # Calculate difference
        diff_value = target_value - current_value
        
        # Configurable minimum order threshold (default $100)
        min_order_threshold = getattr(get_settings(), 'min_order_threshold', 100)
        if abs(diff_value) < min_order_threshold:
            logger.info(f"Skipping {symbol}: difference too small (${diff_value:.2f}, threshold: ${min_order_threshold})")
            return None
        
        # Get current price for share calculation
        contract = self.create_contract(symbol)
        ticker = self._ib.reqMktData(contract, snapshot=True)
        
        # Event-driven wait with timeout (non-blocking pattern)
        max_wait = 2.0  # seconds
        waited = 0.0
        while ticker.marketPrice() is None and waited < max_wait:
            self._ib.sleep(0.1)
            waited += 0.1
        
        current_price = ticker.marketPrice()
        self._ib.cancelMktData(contract)  # Clean up subscription
        
        if not current_price or current_price <= 0:
            logger.warning(f"Could not get price for {symbol}")
            return None
        
        # Calculate shares
        shares = int(diff_value / current_price)
        
        if shares == 0:
            logger.info(f"Skipping {symbol}: calculated 0 shares")
            return None
        
        # Determine order action
        action = "BUY" if shares > 0 else "SELL"
        shares = abs(shares)
        
        # --- Pre-Trade Risk Gate ---
        import time
        start_time = time.time()
        
        if not self._validate_risk(symbol, shares, current_price, action):
            return None
            
        # Create order (Adaptive LMT preferred for production)
        if order_type.upper() == "MKT":
            logger.warning(f"Using MKT order for {symbol} - consider ADAPTIVE for production")
            order = MarketOrder(action, shares)
        elif order_type.upper() == "ADAPTIVE":
            # Adaptive algo order - best for production
            order = LimitOrder(action, shares, current_price)
            order.algoStrategy = "Adaptive"
            order.algoParams = [("adaptivePriority", "Normal")]
        else:
            order = LimitOrder(action, shares, current_price)
        
        # Submit order
        trade = self._ib.placeOrder(contract, order)
        
        # Hook fill event for logging
        def onFill(trade, fill):
            self._log_trade(trade)
            
        trade.fillEvent += onFill
        
        # Telemetry
        latency = (time.time() - start_time) * 1000

        self.metrics["order_latencies"].append(latency)
        self.metrics["last_order_time"] = datetime.now()
        
        logger.info(f"Placed {order_type} {action} order for {shares} shares of {symbol} (Latency: {latency:.2f}ms)")
        
        return trade
    
    def liquidate_position(self, symbol: str) -> Optional[Any]:
        """
        Liquidate entire position in a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Order trade object
        """
        return self.order_target_percent(symbol, 0.0)
    
    def submit_order(self, order: Dict[str, Any]) -> Optional[Any]:
        """
        Submit a pre-built order dictionary.
        
        Args:
            order: Order specification dict
            
        Returns:
            Order trade object
        """
        symbol = order.get("symbol")
        quantity = order.get("quantity", 0)
        side = order.get("side", "BUY")
        order_type = order.get("order_type", "MKT")
        
        if not symbol or quantity == 0:
            return None
        
        if not self.is_connected():
            self.connect()
        
        from ib_insync import MarketOrder, LimitOrder
        
        contract = self.create_contract(symbol)
        
        if order_type.upper() == "MKT":
            ib_order = MarketOrder(side, abs(quantity))
        else:
            price = order.get("limit_price", 0)
            ib_order = LimitOrder(side, abs(quantity), price)
        
        trade = self._ib.placeOrder(contract, ib_order)
        
        logger.info(f"Submitted order: {side} {quantity} {symbol}")
        
        return trade
    
    # =====================
    # Order Management
    # =====================
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get list of open orders."""
        if not self.is_connected():
            self.connect()
        
        orders = []
        for trade in self._ib.openTrades():
            orders.append({
                "symbol": trade.contract.symbol,
                "action": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "order_type": trade.order.orderType,
                "status": trade.orderStatus.status,
            })
        
        return orders
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        if not self.is_connected():
            self.connect()
        
        open_trades = self._ib.openTrades()
        
        for trade in open_trades:
            self._ib.cancelOrder(trade.order)
        
        logger.info(f"Cancelled {len(open_trades)} orders")
        return len(open_trades)
    
    # =====================
    # Market Data
    # =====================
    
    def get_quote(self, symbol: str) -> Dict[str, float]:
        """
        Get current quote for a symbol.
        
        Returns:
            Dictionary with bid, ask, last, volume
        """
        if not self.is_connected():
            self.connect()
        
        contract = self.create_contract(symbol)
        ticker = self._ib.reqMktData(contract)
        self._ib.sleep(1)
        
        return {
            "bid": ticker.bid or 0.0,
            "ask": ticker.ask or 0.0,
            "last": ticker.last or 0.0,
            "volume": ticker.volume or 0,
        }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status for the control plane.
        """
        status = {
            "ib_connected": self.is_connected(),
            "engine_halted": self._halted,
            "last_heartbeat": datetime.now().isoformat(),
            "latency_p50_ms": 0.0,
            "latency_p99_ms": 0.0,
            "truth_layer_active": len(self.aggregators) > 0,
            "active_symbols": list(self.aggregators.keys())
        }
        
        if self.metrics["order_latencies"]:
            import numpy as np
            status["latency_p50_ms"] = float(np.percentile(self.metrics["order_latencies"], 50))
            status["latency_p99_ms"] = float(np.percentile(self.metrics["order_latencies"], 99))
            
        return status

    def get_live_candles(self) -> Dict[str, Dict[str, Any]]:
        """Returns the current forming candle state for all active symbols."""
        results = {}
        for symbol, agg in self.aggregators.items():
            if agg.current_candle:
                c = agg.current_candle
                results[symbol] = {
                    "start_ts": c.start_ts,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume
                }
        return results

    def flatten_all_positions(self) -> int:
        """
        Emergency: Liquidate all positions immediately.
        """
        logger.warning("EMERGENCY: Flattening all positions!")
        positions = self.get_positions()
        count = 0
        for pos in positions:
            self.liquidate_position(pos["symbol"])
            count += 1
        return count
    def run_blocking(self) -> None:
        """
        Run the IB event loop blocking execution.
        Use this for the main entry point script.
        """
        if self.is_connected():
            logger.info("Starting IB Event Loop...")
            self._ib.run()
