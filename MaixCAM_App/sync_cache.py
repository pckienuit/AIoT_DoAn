"""
sync_cache.py — Sync flight face embeddings from server to local SD card cache.

Usage:
    from sync_cache import CacheManager
    cache = CacheManager(config)
    cache.sync(flight_id=1)         # Download & save to cache
    result = cache.match(embedding) # Search local cache
"""
import json
import math
import os
import time
import base64
import struct

# urequests is the MaixCAM/MicroPython-style HTTP client.
# On CPython it falls back to urllib.request.
try:
    import urequests as requests  # noqa: F401 — MaixPy runtime
    _USE_UREQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    _USE_UREQUESTS = False


# =====================================================================
# INTERNAL HTTP HELPERS
# =====================================================================

def _http_get(url: str, timeout: int) -> dict | None:
    """GET request → parsed JSON dict, or None on failure."""
    try:
        if _USE_UREQUESTS:
            r = requests.get(url, timeout=timeout)
            data = r.json()
            r.close()
            return data
        else:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
    except Exception as e:
        print("[sync] GET failed:", url, "->", e)
        return None


def _http_post(url: str, body: dict, timeout: int) -> dict | None:
    """POST JSON body → parsed JSON dict, or None on failure."""
    try:
        payload = json.dumps(body).encode()
        if _USE_UREQUESTS:
            r = requests.post(url, data=payload,
                              headers={"Content-Type": "application/json"},
                              timeout=timeout)
            data = r.json()
            r.close()
            return data
        else:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
    except Exception as e:
        print("[sync] POST failed:", url, "->", e)
        return None


# =====================================================================
# MATH & CRYPTOGRAPHY (no numpy — MaixCAM constraint)
# =====================================================================

def _l2_normalize(vec: list) -> list:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        return vec
    return [v / norm for v in vec]


def _cosine_distance(a: list, b: list) -> float:
    n = min(len(a), len(b))
    dot = 0.0
    for i in range(n):
        dot += a[i] * b[i]
    return 1.0 - dot


def _xtea_encrypt_block(key: tuple[int, int, int, int], block: tuple[int, int]) -> tuple[int, int]:
    y, z = block
    sum_val = 0
    delta = 0x9E3779B9
    mask = 0xFFFFFFFF
    for _ in range(32):
        y = (y + (((z << 4 ^ z >> 5) + z) ^ (sum_val + key[sum_val & 3]))) & mask
        sum_val = (sum_val + delta) & mask
        z = (z + (((y << 4 ^ y >> 5) + y) ^ (sum_val + key[(sum_val >> 11) & 3]))) & mask
    return y, z


def _xtea_crypt_ctr(key_bytes: bytes, nonce_bytes: bytes, data_bytes: bytes) -> bytes:
    key = struct.unpack(">4I", key_bytes)
    nonce = struct.unpack(">2I", nonce_bytes)
    
    out = bytearray()
    num_blocks = (len(data_bytes) + 7) // 8
    
    for i in range(num_blocks):
        ctr_block = (nonce[0], (nonce[1] + i) & 0xFFFFFFFF)
        keystream_block = _xtea_encrypt_block(key, ctr_block)
        keystream_bytes = struct.pack(">2I", *keystream_block)
        
        chunk = data_bytes[i*8 : (i+1)*8]
        for b, k in zip(chunk, keystream_bytes):
            out.append(b ^ k)
            
    return bytes(out)


def _encrypt_xtea_vector(key_bytes: bytes, vector: list[float]) -> tuple[str | None, str | None]:
    try:
        data_bytes = struct.pack("<128f", *vector)
        nonce_bytes = os.urandom(8)
        encrypted_bytes = _xtea_crypt_ctr(key_bytes, nonce_bytes, data_bytes)
        ciphertext_b64 = base64.b64encode(encrypted_bytes).decode("utf-8")
        nonce_b64 = base64.b64encode(nonce_bytes).decode("utf-8")
        return ciphertext_b64, nonce_b64
    except Exception as e:
        print("[encrypt] XTEA error:", e)
        return None, None


def _decrypt_xtea_vector(key_bytes: bytes, ciphertext_b64: str, nonce_b64: str) -> list[float] | None:
    try:
        ciphertext = base64.b64decode(ciphertext_b64)
        nonce = base64.b64decode(nonce_b64)
        decrypted_bytes = _xtea_crypt_ctr(key_bytes, nonce, ciphertext)
        if len(decrypted_bytes) != 512:
            return None
        return list(struct.unpack("<128f", decrypted_bytes))
    except Exception as e:
        print("[decrypt] XTEA error:", e)
        return None


# =====================================================================
# CACHE MANAGER
# =====================================================================

class CacheManager:
    """
    Manages the local face embedding cache on SD card.

    Cache structure (one file per flight):
        /root/cache/flight_<id>.json
        {
          "flight_id": 1,
          "synced_at": 1716553200.0,
          "items": [
            {
              "point_id": "uuid",
              "ciphertext": "base64",
              "iv": "base64 (nonce)",
              "payload": { booking_id, passenger_name, gate, seat, ... }
            }, ...
          ]
        }
    """

    def __init__(self, cfg: dict):
        self.server_url = cfg["server_url"].rstrip("/")
        self.cache_dir = cfg["cache_dir"]
        self.timeout = cfg["api_timeout_sec"]
        self.threshold = cfg["match_threshold"]
        self.fallback_enabled = cfg["fallback_enabled"]
        # Device secret key (16 bytes)
        raw_key = cfg.get("device_secret_key", "d3v1c3_s3cr3t_ke")
        self.device_key = raw_key.encode("utf-8")[:16].ljust(16, b"\x00")
        self._ensure_cache_dir()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def sync(self, flight_id: int) -> int:
        """
        Download embeddings for flight_id from server and save to cache.
        Returns number of items synced, or -1 on failure.
        """
        url = "{}/api/sync/{}".format(self.server_url, flight_id)
        print("[sync] Fetching flight {} from {}".format(flight_id, url))
        data = _http_get(url, self.timeout)
        if data is None:
            print("[sync] Sync failed for flight", flight_id)
            return -1

        count = data.get("count", 0)
        items = data.get("items", [])

        # Note: items are stored encrypted on SD card. No normalization on sync.
        cache_entry = {
            "flight_id": flight_id,
            "synced_at": time.time(),
            "count": count,
            "items": items,
        }
        self._write_cache(flight_id, cache_entry)
        print("[sync] Synced {} embeddings for flight {}".format(count, flight_id))
        return count

    def sync_all(self, flight_ids: list) -> dict:
        """Sync multiple flights. Returns {flight_id: count}."""
        results = {}
        for fid in flight_ids:
            results[fid] = self.sync(fid)
        return results

    def match_local(self, embedding: list, flight_id: int) -> dict | None:
        """
        Search local cache for matching face.
        Returns match dict with payload + distance, or None if no match.
        """
        cache = self._read_cache(flight_id)
        if cache is None:
            print("[cache] No local cache for flight", flight_id)
            return None

        query = _l2_normalize([float(v) for v in embedding])
        best_item = None
        best_dist = 999.0

        for item in cache.get("items", []):
            ref = item.get("embedding")
            if not ref:
                continue
            dist = _cosine_distance(query, ref)
            if dist < best_dist:
                best_dist = dist
                best_item = item

        if best_item is None or best_dist > self.threshold:
            return None

        return {
            "source": "cache",
            "distance": best_dist,
            "score": 1.0 - best_dist,
            "payload": best_item.get("payload", {}),
            "point_id": best_item.get("point_id"),
        }

    def match_server(self, embedding: list, flight_id: int) -> dict | None:
        """
        Fallback: call server /api/face/match.
        Returns match dict or None.
        """
        if not self.fallback_enabled:
            return None
        url = "{}/api/face/match".format(self.server_url)
        ciphertext, iv = _encrypt_xtea_vector(self.device_key, embedding)
        if not ciphertext or not iv:
            print("[fallback] Encryption failed for server match")
            return None
            
        body = {
            "flight_id": flight_id,
            "ciphertext": ciphertext,
            "iv": iv,
            "threshold": self.threshold,
        }
        print("[fallback] Calling server match (encrypted) for flight", flight_id)
        data = _http_post(url, body, self.timeout)
        if data is None or not data.get("matched"):
            return None

        booking = data.get("booking", {}) or {}
        return {
            "source": "server",
            "distance": data.get("distance"),
            "score": data.get("score"),
            "payload": {
                "passenger_name": booking.get("passenger_name", "Unknown"),
                "flight_code": booking.get("flight_code", ""),
                "gate": booking.get("gate", ""),
                "seat_number": booking.get("seat_number", ""),
                "departure_time": booking.get("departure_time", ""),
                "boarding_time": booking.get("boarding_time", ""),
                "destination": booking.get("destination", ""),
            },
            "booking_id": data.get("point_id"),
        }

    def match(self, embedding: list, flight_id: int) -> dict | None:
        """
        Full match pipeline:
        1. Try local cache first (fast, <1ms)
        2. Fallback to server API if cache miss
        Returns match dict or None if no match anywhere.
        """
        result = self.match_local(embedding, flight_id)
        if result is not None:
            return result
        return self.match_server(embedding, flight_id)

    def get_cache_info(self, flight_id: int) -> dict:
        """Return cache metadata for a flight (without embeddings)."""
        cache = self._read_cache(flight_id)
        if cache is None:
            return {"flight_id": flight_id, "cached": False}
        return {
            "flight_id": flight_id,
            "cached": True,
            "count": cache.get("count", 0),
            "synced_at": cache.get("synced_at"),
            "age_sec": time.time() - cache.get("synced_at", 0),
        }

    def clear_cache(self, flight_id: int) -> bool:
        """Delete cache file for a flight."""
        path = self._cache_path(flight_id)
        if os.path.exists(path):
            os.remove(path)
            print("[cache] Cleared cache for flight", flight_id)
            return True
        return False

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _ensure_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print("[cache] Created cache dir:", self.cache_dir)

    def _cache_path(self, flight_id: int) -> str:
        return os.path.join(self.cache_dir, "flight_{}.json".format(flight_id))

    def _write_cache(self, flight_id: int, data: dict):
        path = self._cache_path(flight_id)
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print("[cache] Write failed:", e)

    def _read_cache(self, flight_id: int) -> dict | None:
        path = self._cache_path(flight_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # Decrypt and normalize embeddings on read into memory
            valid_items = []
            for item in data.get("items", []):
                if "embedding" in item and item["embedding"]:
                    item["embedding"] = _l2_normalize(
                        [float(v) for v in item["embedding"]]
                    )
                    valid_items.append(item)
                elif "ciphertext" in item and "iv" in item:
                    decrypted = _decrypt_xtea_vector(
                        self.device_key, item["ciphertext"], item["iv"]
                    )
                    if decrypted is not None:
                        item["embedding"] = _l2_normalize(decrypted)
                        valid_items.append(item)
                        
            data["items"] = valid_items
            return data
        except Exception as e:
            print("[cache] Read failed:", e)
            return None
