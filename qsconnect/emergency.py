"""
Emergency Control System
Persists critical operational state (HALT) to disk for backend enforcement.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

# Path to shared state file
# We use a JSON file for simplicity (vs Redis) as requested in plan
STATE_FILE = Path(__file__).parent.parent / "data" / "system_state.json"

class EmergencyControl:
    """
    Manages the global Kill Switch state.
    Usage:
        EmergencyControl.halt("Op Request")
        EmergencyControl.resume()
        status = EmergencyControl.get_status()
    """
    
    @staticmethod
    def _ensure_dir():
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def halt(reason: str = "Manual Override"):
        """Trigger a system-wide halt."""
        EmergencyControl._ensure_dir()
        state = {
            "halted": True,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "triggered_by": "Dashboard"
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
            logger.warning(f"🚨 SYSTEM HALTED: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to persist halt state: {e}")
            return False

    @staticmethod
    def resume():
        """Resume normal operations."""
        EmergencyControl._ensure_dir()
        state = {
            "halted": False,
            "reason": "Resumed",
            "timestamp": datetime.utcnow().isoformat(),
            "triggered_by": "Dashboard"
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
            logger.info("✅ SYSTEM RESUMED")
            return True
        except Exception as e:
            logger.error(f"Failed to persist resume state: {e}")
            return False

    @staticmethod
    def is_halted() -> bool:
        """Check if system is currently halted."""
        if not STATE_FILE.exists():
            return False
            
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                return data.get("halted", False)
        except Exception:
            # If we can't read state, assume safe default (Review choice: Fail Open or Closed?)
            # Usually Fail Closed (Halted) is safer, but for now we return False to avoid lockouts during IO errors
            return False

    @staticmethod
    def get_status() -> dict:
        """Get full status object."""
        if not STATE_FILE.exists():
            return {"halted": False, "reason": "System Normal", "timestamp": ""}
            
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"halted": False, "reason": "State Read Error", "timestamp": ""}
