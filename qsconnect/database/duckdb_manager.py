"""
QS Connect - DuckDB Database Manager

High-performance columnar database management using DuckDB.
Handles all data storage, querying, and schema management.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import duckdb
import polars as pl
import pandas as pd
from loguru import logger


class DuckDBManager:
    """
    Manager for DuckDB database operations.
    
    DuckDB is an embedded analytical database that provides:
    - Columnar storage for fast analytics
    - SQL interface
    - Direct Polars/Pandas integration
    - Zero-copy data sharing
    """
    
    def __init__(self, db_path: Path, read_only: bool = False, auto_close: bool = False):
        """
        Initialize DuckDB manager.
        
        Args:
            db_path: Path to the DuckDB database file
            read_only: If True, open connection in read-only mode
            auto_close: If True, close connection after each query (helper for file locking)
        """
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.auto_close = auto_close
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        
        # Initialize schema on first connect (ONLY if writable)
        if not self.read_only:
            self.connect()
            self._init_schema()
        
        logger.info(f"DuckDB manager initialized: {self.db_path} (read_only={self.read_only})")
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Establish database connection."""
        if self._conn is None:
            config = {}
            if self.read_only:
                config['access_mode'] = 'READ_ONLY'
                
            self._conn = duckdb.connect(str(self.db_path), read_only=self.read_only, config=config)
        return self._conn
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("DuckDB connection closed")
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self.connect()
        
        # Prices table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                change DOUBLE,
                change_percent DOUBLE,
                vwap DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        # Income statements
        conn.execute("""
            CREATE TABLE IF NOT EXISTS income_statements (
                symbol VARCHAR,
                date DATE,
                period VARCHAR,
                revenue DOUBLE,
                cost_of_revenue DOUBLE,
                gross_profit DOUBLE,
                operating_expenses DOUBLE,
                operating_income DOUBLE,
                net_income DOUBLE,
                eps DOUBLE,
                eps_diluted DOUBLE,
                shares_outstanding BIGINT,
                ebitda DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, period)
            )
        """)
        
        # Balance sheets
        conn.execute("""
            CREATE TABLE IF NOT EXISTS balance_sheets (
                symbol VARCHAR,
                date DATE,
                period VARCHAR,
                total_assets DOUBLE,
                total_liabilities DOUBLE,
                total_equity DOUBLE,
                cash_and_equivalents DOUBLE,
                short_term_debt DOUBLE,
                long_term_debt DOUBLE,
                total_debt DOUBLE,
                retained_earnings DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, period)
            )
        """)
        
        # Cash flow statements
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cash_flows (
                symbol VARCHAR,
                date DATE,
                period VARCHAR,
                operating_cash_flow DOUBLE,
                investing_cash_flow DOUBLE,
                financing_cash_flow DOUBLE,
                net_cash_flow DOUBLE,
                capital_expenditure DOUBLE,
                free_cash_flow DOUBLE,
                dividends_paid DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, period)
            )
        """)
        
        # Financial ratios
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ratios (
                symbol VARCHAR,
                date DATE,
                period VARCHAR,
                pe_ratio DOUBLE,
                pb_ratio DOUBLE,
                ps_ratio DOUBLE,
                debt_to_equity DOUBLE,
                current_ratio DOUBLE,
                quick_ratio DOUBLE,
                roe DOUBLE,
                roa DOUBLE,
                profit_margin DOUBLE,
                operating_margin DOUBLE,
                dividend_yield DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, period)
            )
        """)
        
        # Stock list/universe
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_list (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                exchange VARCHAR,
                exchange_short_name VARCHAR,
                asset_type VARCHAR,
                price DOUBLE,
                sector VARCHAR,
                industry VARCHAR,
                country VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Strategy Audit Log (Change Governance)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_audit_log (
                strategy_hash VARCHAR PRIMARY KEY,
                config_json JSON,
                regime_snapshot JSON,
                llm_reasoning TEXT,
                human_rationale TEXT,
                approved_by VARCHAR,
                approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                stage VARCHAR, -- SHADOW, PAPER, CANARY, FULL
                capital_allocation DOUBLE DEFAULT 0.0,
                mlflow_run_id VARCHAR,
                ttl_expiry TIMESTAMP
            )
        """)
        
        # Strategy Drift Logs (Performance Tracking)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_drift_logs (
                strategy_hash VARCHAR,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name VARCHAR,
                expected_value DOUBLE,
                actual_value DOUBLE,
                drift_score DOUBLE,
                status VARCHAR, -- GREEN, YELLOW, RED
                PRIMARY KEY (strategy_hash, timestamp, metric_name)
            )
        """)

        # Trade Execution Log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR PRIMARY KEY,
                strategy_hash VARCHAR,
                symbol VARCHAR,
                side VARCHAR,
                quantity DOUBLE,
                fill_price DOUBLE,
                execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                commission DOUBLE,
                slippage_bps DOUBLE,
                order_type VARCHAR,
                account_id VARCHAR
            )
        """)

        # Real-Time Candle Truth Layer (Broadcasting)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS realtime_candles (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                is_final BOOLEAN,
                PRIMARY KEY (symbol, timestamp)
            )
        """)

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(execution_time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_strat ON trades(strategy_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy_audit_stage ON strategy_audit_log(stage)")
        
        logger.info("Database schema initialized with Governance and Execution logs")
    
    # =====================
    # Query Methods
    # =====================
    
    def query(self, sql: str) -> pl.DataFrame:
        """
        Execute a SQL query and return results as Polars DataFrame.
        
        Args:
            sql: SQL query string
            
        Returns:
            Polars DataFrame with query results
        """
        conn = self.connect()
        try:
            result = conn.execute(sql).pl()
            return result
        finally:
            if self.auto_close:
                self.close()
    
    def query_pandas(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as Pandas DataFrame."""
        conn = self.connect()
        try:
            return conn.execute(sql).df()
        finally:
            if self.auto_close:
                self.close()
    
    # =====================
    # Upsert Methods
    # =====================
    
    def upsert_prices(self, df: pl.DataFrame) -> int:
        """
        Upsert price data into the database.
        
        Args:
            df: Polars DataFrame with price data
            
        Returns:
            Number of rows affected
        """
        if df.is_empty():
            return 0
        
        conn = self.connect()
        
        # Standardize column names
        column_mapping = {
            "adjClose": "adj_close",
            "changePercent": "change_percent",
        }
        
        for old, new in column_mapping.items():
            if old in df.columns:
                df = df.rename({old: new})
        
        # Required columns
        required_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in required_cols if c in df.columns]
        
        df_subset = df.select(available_cols)
        
        # Register DataFrame and insert
        conn.register("temp_prices", df_subset)
        
        conn.execute("""
            INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, volume)
            SELECT symbol, date, open, high, low, close, volume
            FROM temp_prices
        """)
        
        conn.unregister("temp_prices")
        
        logger.info(f"Upserted {len(df)} price records")
        return len(df)
    
    def upsert_fundamentals(self, statement_type: str, df: pl.DataFrame) -> int:
        """
        Upsert fundamental data into the appropriate table.
        
        Args:
            statement_type: Type of financial statement
            df: Polars DataFrame with fundamental data
            
        Returns:
            Number of rows affected
        """
        if df.is_empty():
            return 0
        
        conn = self.connect()
        
        # Map statement type to table
        table_mapping = {
            "income-statement": "income_statements",
            "balance-sheet-statement": "balance_sheets",
            "cash-flow-statement": "cash_flows",
            "ratios": "ratios",
        }
        
        table_name = table_mapping.get(statement_type)
        if not table_name:
            logger.warning(f"Unknown statement type: {statement_type}")
            return 0
        
        # Register and insert
        conn.register("temp_fundamentals", df)
        
        # Get column intersection
        table_cols = [r[0] for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
        df_cols = df.columns
        common_cols = [c for c in df_cols if c.lower() in [t.lower() for t in table_cols]]
        
        if common_cols:
            cols_str = ", ".join(common_cols)
            conn.execute(f"""
                INSERT OR REPLACE INTO {table_name} ({cols_str})
                SELECT {cols_str} FROM temp_fundamentals
            """)
        
        conn.unregister("temp_fundamentals")
        
        logger.info(f"Upserted {len(df)} {statement_type} records")
        return len(df)
    
    def upsert_stock_list(self, df: pd.DataFrame) -> int:
        """Upsert stock list data."""
        if df.empty:
            return 0
        
        conn = self.connect()
        conn.register("temp_stocks", df)
        
        conn.execute("""
            INSERT OR REPLACE INTO stock_list (symbol, name, exchange, exchange_short_name, asset_type, price)
            SELECT symbol, name, exchange, exchangeShortName, type, price
            FROM temp_stocks
        """)
        
        conn.unregister("temp_stocks")
        logger.info(f"Upserted {len(df)} stock records")
        return len(df)
    
    # =====================
    # Data Retrieval
    # =====================
    
    def get_prices(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get price data from database.
        
        Args:
            symbols: Optional list of symbols to filter
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            Polars DataFrame with price data
        """
        where_clauses = []
        
        if symbols:
            # Sanitize symbols to prevent SQL injection
            sanitized = [s.replace("'", "").replace(";", "").upper()[:10] for s in symbols]
            symbols_str = ",".join([f"'{s}'" for s in sanitized])
            where_clauses.append(f"symbol IN ({symbols_str})")
        if start_date:
            where_clauses.append(f"date >= '{start_date}'")
        if end_date:
            where_clauses.append(f"date <= '{end_date}'")
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        sql = f"""
            SELECT * FROM prices
            WHERE {where_sql}
            ORDER BY symbol, date
        """
        
        return self.query(sql)
    
    def get_symbols(self) -> List[str]:
        """Get list of all symbols in the database."""
        result = self.query("SELECT DISTINCT symbol FROM prices ORDER BY symbol")
        return result["symbol"].to_list()
    
    def get_date_range(self) -> Dict[str, str]:
        """Get the date range of price data in the database."""
        result = self.query_pandas("""
            SELECT 
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(DISTINCT symbol) as num_symbols,
                COUNT(*) as num_records
            FROM prices
        """)
        
        if not result.empty:
            return {
                "min_date": str(result["min_date"].iloc[0]),
                "max_date": str(result["max_date"].iloc[0]),
                "num_symbols": int(result["num_symbols"].iloc[0]),
                "num_records": int(result["num_records"].iloc[0]),
            }
        return {"min_date": None, "max_date": None, "num_symbols": 0, "num_records": 0}
