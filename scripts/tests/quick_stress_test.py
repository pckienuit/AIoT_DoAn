#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIoT Quick Stress Test - Test nhanh he thong
"""
import io
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import httpx
import base64
import os
import struct
import time
import random
import uuid
random.seed()
import statistics

SERVER_URL = "http://127.0.0.1:8000"

BAR_WIDTH = 40


class ProgressBar:
    def __init__(self, total: int, prefix: str = ""):
        self.total = total
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()
        self._last_line_len = 0
    
    def update(self, current: int = None, suffix: str = ""):
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        percent = self.current / self.total if self.total > 0 else 0
        filled = int(BAR_WIDTH * percent)
        bar = "=" * filled + "-" * (BAR_WIDTH - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"ETA: {int(eta)}s"
        else:
            eta_str = "ETA: --s"
        
        line = f"\r{self.prefix} [{bar}] {percent*100:5.1f}% {eta_str} {suffix}"
        spaces = " " * max(0, self._last_line_len - len(line))
        sys.stdout.write(line + spaces + "\r")
        sys.stdout.flush()
        self._last_line_len = len(line)
    
    def finish(self, message: str = ""):
        elapsed = time.time() - self.start_time
        bar = "=" * BAR_WIDTH
        line = f"\r{self.prefix} [{bar}] 100.0% {message} ({elapsed:.1f}s)\n"
        spaces = " " * max(0, self._last_line_len - len(line))
        sys.stdout.write(line + spaces)
        sys.stdout.flush()


def generate_aes_gcm_payload(vector, booking_id):
    AES_SECRET_HEX = "94c8e763a8a3a31e2474db62c82e0fb58cc2a77ef7cb73f1d8c117b4abdc3d9d"
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aesgcm = AESGCM(bytes.fromhex(AES_SECRET_HEX))
    packed_bytes = struct.pack("<128f", *vector)
    iv = os.urandom(12)
    ciphertext = aesgcm.encrypt(iv, packed_bytes, None)
    return {
        "booking_id": booking_id,
        "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
        "iv": base64.b64encode(iv).decode("utf-8")
    }


def check_server():
    try:
        r = httpx.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False


def run_test(num_passengers: int):
    print(f"\n[TEST] Dang ky {num_passengers} hanh khach...\n")
    
    # Tao chuyen bay
    flight_code = f"QTEST-{num_passengers}-{random.randint(100, 999)}"
    flight_res = httpx.post(f"{SERVER_URL}/api/flights", json={
        "flight_code": flight_code,
        "destination": "Test City",
        "departure_time": "2026-05-25T20:00:00+07:00",
        "boarding_time": "2026-05-25T19:20:00+07:00",
        "gate": "T1"
    }, timeout=30.0)
    flight_id = flight_res.json()["id"]
    print(f"[OK] Chuyen bay: {flight_code} (ID: {flight_id})")
    
    register_times = []
    progress = ProgressBar(num_passengers, "     ")
    
    for i in range(num_passengers):
        try:
            # Tao passenger (server tra ve 201 Created)
            pass_res = httpx.post(f"{SERVER_URL}/api/passengers", json={
                "name": f"Test Pax {i}",
                "email": f"test_{num_passengers}_{i}_{random.randint(10000, 99999)}@test.com",
                "phone": f"09{random.randint(10000000, 99999999)}"
            }, timeout=10.0)
            if pass_res.status_code not in (200, 201):
                print(f"\n[!] Passenger create failed: {pass_res.status_code}")
                continue
            passenger_id = pass_res.json()["id"]
            
            # Tao booking (server tra ve 201 Created)
            # Dung timestamp + random de tao booking_code unique
            import uuid
            book_code = f"QB-{int(time.time())}-{uuid.uuid4().hex[:8].upper()}"
            book_res = httpx.post(f"{SERVER_URL}/api/bookings", json={
                "booking_code": book_code,
                "passenger_id": passenger_id,
                "flight_id": flight_id,
                "seat_number": f"{random.randint(1, 30)}{random.choice('ABCDEF')}"
            }, timeout=10.0)
            if book_res.status_code not in (200, 201):
                print(f"\n[!] Booking create failed: {book_res.status_code}")
                continue
            booking_id = book_res.json()["id"]
            
            # Encrypt & register face (tra ve 201 Created)
            vector = [random.uniform(-0.1, 0.1) for _ in range(128)]
            norm = sum(v*v for v in vector) ** 0.5
            vector = [v / norm for v in vector]
            payload = generate_aes_gcm_payload(vector, booking_id)
            
            start = time.perf_counter()
            reg_res = httpx.post(f"{SERVER_URL}/api/face/register", json=payload, timeout=10.0)
            end = time.perf_counter()
            
            if reg_res.status_code == 201:
                register_times.append((end - start) * 1000)
            else:
                print(f"\n[!] Face register failed: {reg_res.status_code} - {reg_res.text[:100]}")
        except Exception as e:
            print(f"\n[!] Error at passenger {i}: {e}")
        
        progress.update(suffix=f"Pax {i+1}/{num_passengers}")
    
    progress.finish("Xong")
    
    # Sync test
    sync_start = time.perf_counter()
    sync_res = httpx.get(f"{SERVER_URL}/api/sync/{flight_id}", timeout=30.0)
    sync_end = time.perf_counter()
    sync_time_ms = (sync_end - sync_start) * 1000
    sync_kb = len(sync_res.content) / 1024.0
    
    avg_reg = statistics.mean(register_times) if register_times else 0
    p95_reg = sorted(register_times)[int(len(register_times)*0.95)] if register_times else 0
    
    print(f"\n[KET QUA]")
    print(f"  - Dang ky thanh cong: {len(register_times)}/{num_passengers}")
    print(f"  - Server Reg Avg: {avg_reg:.2f}ms")
    print(f"  - Server Reg P95: {p95_reg:.2f}ms")
    print(f"  - Sync time: {sync_time_ms:.2f}ms")
    print(f"  - Sync size: {sync_kb:.2f} KB")
    if num_passengers > 0 and len(register_times) > 0:
        bytes_per_pax = len(sync_res.content) / len(register_times) if sync_res.content else 0
        print(f"  - Bytes/passenger: {bytes_per_pax:.1f}")
    
    return {
        "flight_id": flight_id,
        "num_passengers": num_passengers,
        "registered": len(register_times),
        "avg_reg_ms": avg_reg,
        "p95_reg_ms": p95_reg,
        "sync_time_ms": sync_time_ms,
        "sync_kb": sync_kb
    }


def main():
    print("="*70)
    print("  AIoT QUICK STRESS TEST")
    print("="*70)
    
    print("\n[CHECK] Kiem tra server...")
    if not check_server():
        print("[ERROR] Server khong phan hoi!")
        print("Vui long chay: uvicorn server.main:app --host 0.0.0.0 --port 8000")
        return
    
    print("[OK] Server san sang\n")
    
    test_sizes = [10, 50, 100, 200]
    results = []
    
    for size in test_sizes:
        result = run_test(size)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("  TONG HOP KET QUA")
    print("="*70)
    print(f"\n{'Size':>8} | {'Reg OK':>8} | {'Avg ms':>10} | {'P95 ms':>10} | {'Sync ms':>10} | {'KB':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['num_passengers']:>8} | {r['registered']:>8} | {r['avg_reg_ms']:>10.2f} | {r['p95_reg_ms']:>10.2f} | {r['sync_time_ms']:>10.2f} | {r['sync_kb']:>8.2f}")
    print("-"*70)
    
    print("\n[SUCCESS] Quick stress test hoan tat!")


if __name__ == "__main__":
    main()
