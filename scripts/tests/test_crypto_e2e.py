#!/usr/bin/env python3
import httpx
import base64
import os
import struct
import paramiko
import json
import random

# Server URL (accessible locally)
SERVER_URL = "http://127.0.0.1:8000"
SSH_HOST = "10.154.35.1"
SSH_USER = "root"
SSH_PASS = "root"

def generate_aes_gcm_payload(vector, booking_id):
    # Hex key matches default AES_SECRET_KEY in server
    AES_SECRET_HEX = "94c8e763a8a3a31e2474db62c82e0fb58cc2a77ef7cb73f1d8c117b4abdc3d9d"
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aesgcm = AESGCM(bytes.fromhex(AES_SECRET_HEX))
    
    # Pack vector (128 float32 values)
    packed_bytes = struct.pack("<128f", *vector)
    iv = os.urandom(12)
    
    # Encrypt (tag is automatically appended)
    ciphertext = aesgcm.encrypt(iv, packed_bytes, None)
    
    return {
        "booking_id": booking_id,
        "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
        "iv": base64.b64encode(iv).decode("utf-8")
    }

def main():
    print("=" * 65)
    print("RUNNING END-TO-END CRYPTOGRAPHY SYSTEM INTEGRATION TEST")
    print("=" * 65)
    
    # 1. Setup metadata on Server via standard API
    # Create a unique flight code to avoid collision
    flight_code = f"VN-TEST-{random.randint(100, 999)}"
    print(f"[1] Creating temporary flight: {flight_code}")
    
    flight_res = httpx.post(f"{SERVER_URL}/api/flights", json={
        "flight_code": flight_code,
        "destination": "Paris",
        "departure_time": "2026-05-24T20:00:00+07:00",
        "boarding_time": "2026-05-24T19:20:00+07:00",
        "gate": "A12"
    })
    assert flight_res.status_code == 201, f"Failed to create flight: {flight_res.text}"
    flight = flight_res.json()
    flight_id = flight["id"]
    print(f"    Flight created successfully with ID: {flight_id}")
    
    # Create passenger
    passenger_res = httpx.post(f"{SERVER_URL}/api/passengers", json={
        "name": "E2E Test Crypto Passenger",
        "email": f"crypto_{random.randint(1000, 9999)}@test.com",
        "phone": "0987654321"
    })
    assert passenger_res.status_code == 201
    passenger = passenger_res.json()
    passenger_id = passenger["id"]
    
    # Create booking
    booking_res = httpx.post(f"{SERVER_URL}/api/bookings", json={
        "booking_code": f"BK-{random.randint(10000, 99999)}",
        "passenger_id": passenger_id,
        "flight_id": flight_id,
        "seat_number": "12A"
    })
    assert booking_res.status_code == 201
    booking = booking_res.json()
    booking_id = booking["id"]
    print(f"    Booking created successfully with ID: {booking_id}")
    
    # 2. Simulate Web Client: Generate and Encrypt a real 128D embedding
    print("[2] Generating dummy 128D face vector and encrypting using AES-GCM-256...")
    original_vector = [random.uniform(-0.1, 0.1) for _ in range(128)]
    # L2 normalize
    norm = sum(v*v for v in original_vector) ** 0.5
    original_vector = [v / norm for v in original_vector]
    
    # Generate encrypted payload
    register_payload = generate_aes_gcm_payload(original_vector, booking_id)
    
    # 3. Post to register endpoint
    print("[3] Submitting encrypted vector to /api/face/register...")
    register_res = httpx.post(f"{SERVER_URL}/api/face/register", json=register_payload)
    assert register_res.status_code == 201, f"Registration failed: {register_res.text}"
    register_data = register_res.json()
    print("    Face registered successfully!")
    print(f"    Qdrant Point ID: {register_data['point_id']}")
    
    # 4. SSH into MaixCAM and run sync & decrypt test
    print("[4] SSH connecting to MaixCAM to trigger sync and local decryption check...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, username=SSH_USER, password=SSH_PASS, timeout=10)
    
    # Remote verification script
    device_script = f"""
import sys
sys.path.insert(0, '/root')
from config import load_config
from sync_cache import CacheManager
import json

cfg = load_config()
cm = CacheManager(cfg)

# Sync flight cache
n_synced = cm.sync({flight_id})
print("[remote] Synced:", n_synced)

# Check SD card cache file content (verify it contains ciphertext and no plaintext embedding)
cache_path = cm._cache_path({flight_id})
with open(cache_path, 'r') as f:
    cache_data = json.load(f)

item = cache_data['items'][0]
print("[remote] Contains ciphertext:", 'ciphertext' in item)
print("[remote] Contains iv (nonce):", 'iv' in item)
print("[remote] Contains plaintext embedding:", 'embedding' in item)

# Match using cache (which triggers decryption into RAM)
result = cm.match_local({original_vector}, {flight_id})
print("[remote] Match result:", json.dumps(result))
"""
    
    sftp = client.open_sftp()
    with sftp.open('/root/_test_e2e_crypto.py', 'w') as f:
        f.write(device_script)
    sftp.close()
    
    _, stdout, stderr = client.exec_command(f'python3 /root/_test_e2e_crypto.py')
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    
    print("\n--- Device Execution Output ---")
    print(out)
    if err:
        print("--- Device Execution Errors ---")
        print(err)
        
    client.exec_command('rm -f /root/_test_e2e_crypto.py')
    client.close()
    
    # Parse output to verify correctness
    print("\n" + "=" * 65)
    print("VERIFICATION METRICS:")
    print("=" * 65)
    
    assert "[remote] Synced: 1" in out, "Failed to sync the registered face to device cache"
    assert "Contains ciphertext: True" in out, "Device cache file is missing encrypted ciphertext"
    assert "Contains iv (nonce): True" in out, "Device cache file is missing encrypted IV"
    assert "Contains plaintext embedding: False" in out, "CRITICAL SECURITY FAULT: Plaintext embedding written to device SD card cache!"
    assert '"source": "cache"' in out, "Device matching failed to retrieve the result from decrypted RAM cache"
    
    # Extract match score
    import re
    score_match = re.search(r'"score":\s*([0-9.]+)', out)
    if score_match:
        score = float(score_match.group(1))
        print(f"Cosine Similarity Score in RAM: {score:.5f} (Ideal: ~1.00000)")
        assert score > 0.99, f"Decrypted vector score {score} is too low (precision issue)"
        
    print("SUCCESS: End-to-End Cryptography Integration verified and 100% correct!")
    print("=" * 65)

if __name__ == "__main__":
    main()
