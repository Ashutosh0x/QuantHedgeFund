"""
Database Initialization Script
Run this ONCE to create the database schema.
"""
import sys
import os
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qsconnect.database.duckdb_manager import DuckDBManager
from loguru import logger

def init_db():
    db_path = Path("data/quant.duckdb")
    if db_path.exists():
        logger.warning(f"Database already exists at {db_path}")
        # We don't delete it, just ensure schema is up to date
    
    logger.info("Initializing Database Schema...")
    # Initialize with read_only=False to trigger _init_schema()
    db_mgr = DuckDBManager(db_path, read_only=False)
    
    # Explicitly call _init_schema if needed (though __init__ does it)
    # The current DuckDBManager calls _init_schema() in __init__ if read_only=False
    
    # Verify tables
    tables = db_mgr.query("SHOW TABLES").to_dicts()
    table_names = [t['name'] for t in tables]
    logger.info(f"Existing Tables: {table_names}")
    
    # Check for core tables
    required = ['prices', 'realtime_candles', 'trades', 'strategy_audit_log']
    missing = [t for t in required if t not in table_names]
    
    if missing:
        logger.error(f"Missing tables: {missing}")
    else:
        logger.success("✅ Database Schema Verification Passed")
        
    db_mgr.close()

if __name__ == "__main__":
    init_db()
