"""
Edge device configuration — reads from /root/config.json on SD card.
Defaults are suitable for local development/testing.
"""
import json
import os

CONFIG_PATH = "/root/config.json"

DEFAULTS = {
    "server_url": "http://10.154.36.100:8000",
    "flight_ids": [1],          # List of flight IDs to sync
    "sync_interval_sec": 300,   # Sync every 5 minutes
    "cache_dir": "/root/cache", # SD card cache directory
    "match_threshold": 0.045,   # Cosine distance threshold
    "api_timeout_sec": 5,       # HTTP timeout for API calls
    "fallback_enabled": True,   # Call server API on cache miss
    "stream_fps_limit": 15,     # Max FPS for web streaming
    "stream_jpeg_quality": 70,  # JPEG compression quality (1-100)
    "ai_frame_interval": 3,     # Run AI once every N camera frames
    "recognition_frame_interval": 9, # Run ArcFace once every N camera frames
    "overlay_cache_ttl_sec": 1.0, # Keep last recognition overlay briefly
    "recognition_cache_ttl_sec": 1.5, # Reuse recent ArcFace match briefly
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
