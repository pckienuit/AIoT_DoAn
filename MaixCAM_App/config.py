"""
Edge device configuration — reads from /root/config.json on SD card.
Defaults are suitable for local development/testing.
"""
import json
import os

CONFIG_PATH = "/root/config.json"

DEFAULTS = {
    "server_url": "http://10.154.35.1:8000",
    "flight_ids": [1],          # List of flight IDs to sync
    "sync_interval_sec": 300,   # Sync every 5 minutes
    "cache_dir": "/root/cache", # SD card cache directory
    "match_threshold": 0.045,   # Cosine distance threshold
    "api_timeout_sec": 5,       # HTTP timeout for API calls
    "fallback_enabled": True,   # Call server API on cache miss
}


def load_config() -> dict:
    """Load config from JSON file, falling back to defaults."""
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                overrides = json.load(f)
            cfg.update(overrides)
            print("[config] Loaded from", CONFIG_PATH)
        except Exception as e:
            print("[config] Failed to load config:", e, "— using defaults")
    else:
        print("[config] No config.json found, using defaults")
    return cfg
