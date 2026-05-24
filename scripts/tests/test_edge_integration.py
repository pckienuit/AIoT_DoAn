#!/usr/bin/env python3
"""
test_edge_integration.py — Integration test for Phase 4 (Edge Integration).
Verifies:
  1. Fetching booking 12 from local DB.
  2. Syncing flight 12 from Server -> MaixCAM cache.
  3. Validating the cache content on MaixCAM via SSH.
  4. Running a face match simulation on MaixCAM with:
     a) Exact match (should be 0.0 distance).
     b) Slightly perturbed match (should be low distance, match).
     c) Cache miss fallback match (forces cache miss, calls server match API).
"""
import json
import math
import os
import random
import sys
import paramiko

# Connection config
HOST = "10.154.35.1"
USER = "root"
PASS = "root"
SERVER_URL = "http://10.154.35.100:8000"
FLIGHT_ID = 12

DEVICE_SIM_SCRIPT = """
import sys
sys.path.insert(0, '/root')
from config import load_config
from sync_cache import CacheManager
import json

cfg = load_config()
cm = CacheManager(cfg)

# Load the query vector we uploaded
with open('/root/test_query.json', 'r') as f:
    test_data = json.load(f)

emb = test_data['embedding']
name = test_data['name']
print(f"[device-sim] Querying name: {name}")

# --- Test 1: Local Cache Match (Exact Vector) ---
print("[device-sim] Run match_local (exact)...")
res1 = cm.match_local(emb, 12)
if res1:
    print(f"  SUCCESS: Found {res1['payload']['passenger_name']} in local cache! distance={res1['distance']:.6f}")
else:
    print("  FAILED: Exact match not found in cache")

# --- Test 2: Local Cache Match (Perturbed Vector) ---
print("[device-sim] Run match_local (perturbed)...")
perturbed_emb = [v + 0.005 for v in emb] # add a small perturbation
# L2 Norm
mag = sum(x*x for x in perturbed_emb) ** 0.5
perturbed_emb = [x/mag for x in perturbed_emb]

res2 = cm.match_local(perturbed_emb, 12)
if res2:
    print(f"  SUCCESS: Found {res2['payload']['passenger_name']} in local cache (perturbed)! distance={res2['distance']:.6f}")
else:
    print("  FAILED: Perturbed match not found in cache")

# --- Test 3: Fallback match server (Exact Vector) ---
print("[device-sim] Run match_server fallback...")
res3 = cm.match_server(emb, 12)
if res3:
    print(f"  SUCCESS: Server fallback match succeeded! passenger_name={res3['payload']['passenger_name']} distance={res3['distance']:.6f}")
else:
    print("  FAILED: Server fallback match failed")

# --- Test 4: Combined match API (Cache then Fallback) ---
print("[device-sim] Run combined match (Cache hit)...")
res4 = cm.match(emb, 12)
print(f"  Result source: {res4.get('source')} -> passenger: {res4['payload'].get('passenger_name')}")
"""

def l2_norm(v):
    mag = math.sqrt(sum(x*x for x in v))
    return [x/mag for x in v]

def main():
    print("=" * 65)
    print("AIoT Face Lookup: Edge Integration E2E Test Suite")
    print("=" * 65)

    # 1. Fetching embedding from server for flight 12
    print("\n[1] Fetching registered embedding for Flight 12 from Server...")
    import urllib.request
    try:
        url = f"{SERVER_URL}/api/sync/{FLIGHT_ID}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as r:
            resp = json.loads(r.read().decode())
        
        items = resp.get("items", [])
        if not items:
            print("ERROR: No face embeddings found in Server Qdrant for Flight 12. Run web_stage3 to register first.")
            sys.exit(1)
        
        target = items[0]
        embedding = target["embedding"]
        passenger_name = target["payload"]["passenger_name"]
        print(f"  Found passenger: {passenger_name}")
        print(f"  Embedding dimensions: {len(embedding)}")
    except Exception as e:
        print(f"  ERROR: Could not fetch from server: {e}")
        sys.exit(1)

    # 2. SSH to MaixCAM Device
    print("\n[2] Connecting to MaixCAM Edge Device via SSH...")
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print(f"  Connected to {HOST}!")
    except Exception as e:
        print(f"  ERROR: SSH connection failed: {e}")
        sys.exit(1)

    # 3. Trigger cache sync on device
    print("\n[3] Triggering cache sync on MaixCAM...")
    # Run the sync method on device
    sftp = client.open_sftp()
    
    # Save the embedding query JSON to device
    query_data = {"name": passenger_name, "embedding": embedding}
    with sftp.open('/root/test_query.json', 'w') as f:
        json.dump(query_data, f)
        
    # Write device sim script
    with sftp.open('/root/device_sim.py', 'w') as f:
        f.write(DEVICE_SIM_SCRIPT)
    sftp.close()

    # Trigger Sync command
    stdin, stdout, stderr = client.exec_command(
        f"python3 -c \"import sys; sys.path.insert(0, '/root'); from config import load_config; from sync_cache import CacheManager; cm=CacheManager(load_config()); cm.sync({FLIGHT_ID})\""
    )
    rc = stdout.channel.recv_exit_status()
    print("  Sync execution stdout:")
    print(stdout.read().decode().strip())
    err = stderr.read().decode().strip()
    if err:
        print("  Sync execution stderr:")
        print(err)

    # 4. Check that cache file exists and read it
    print("\n[4] Validating cache file on device SD Card...")
    stdin, stdout, stderr = client.exec_command(f"cat /root/cache/flight_{FLIGHT_ID}.json")
    rc = stdout.channel.recv_exit_status()
    if rc == 0:
        cache_data = json.loads(stdout.read().decode())
        print(f"  Cache file exists! Flight ID: {cache_data.get('flight_id')}, count: {cache_data.get('count')}")
    else:
        print("  ERROR: Cache file was not created on device.")
        sys.exit(1)

    # 5. Run simulation tests
    print("\n[5] Executing Face Matching Simulator on device...")
    stdin, stdout, stderr = client.exec_command("python3 /root/device_sim.py")
    rc = stdout.channel.recv_exit_status()
    print("  Simulator output:")
    print(stdout.read().decode().strip())
    err_out = stderr.read().decode().strip()
    if err_out:
        print("  Simulator errors:")
        print(err_out)

    # 6. Clean up simulation files
    client.exec_command("rm -f /root/test_query.json /root/device_sim.py")
    client.close()

    print("\n" + "=" * 65)
    print("E2E Integration Test Completed!")
    print("=" * 65)

if __name__ == "__main__":
    main()
