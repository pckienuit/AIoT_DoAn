"""
benchmark_models.py - Benchmark SFace (FP32/INT8) vs V9 (Landmark) vs P3 (ArcFace) ONNX models.
"""
import os
import time
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "SFace (FP32)": os.path.join(SCRIPT_DIR, "models", "face_recognition_sface_2021dec.onnx"),
    "SFace (INT8)": os.path.join(SCRIPT_DIR, "models", "face_recognition_sface_2021dec_int8.onnx"),
    "V9 Landmarks": os.path.join(SCRIPT_DIR, "models", "face_detect_v9.onnx"),
    "ArcFace P3": os.path.join(SCRIPT_DIR, "models", "face_recognize_arcface_p3.onnx"),
}

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

def get_input_shape_and_type(session):
    input_meta = session.get_inputs()[0]
    return input_meta.shape, input_meta.type

def benchmark_model(name, path):
    if not os.path.exists(path):
        print(f"[-] {name} not found at {path}. Skipping.")
        return None

    # Load session (CPU execution provider)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(path, opts, providers=['CPUExecutionProvider'])

    shape, dtype = get_input_shape_and_type(session)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    # Replace dynamic batch size with 1
    run_shape = [1 if isinstance(s, str) or s is None or s < 0 else s for s in shape]
    
    print(f"\n[*] Benchmarking {name}...")
    print(f"    Path: {path}")
    print(f"    Input: {input_name} {run_shape} ({dtype})")
    print(f"    Outputs: {', '.join(output_names)}")

    # Generate dummy input data
    if "float" in dtype:
        dummy_input = np.random.randn(*run_shape).astype(np.float32)
    elif "int8" in dtype:
        dummy_input = np.random.randint(-128, 127, size=run_shape, dtype=np.int8)
    else:
        dummy_input = np.random.randn(*run_shape).astype(np.float32)

    # Warmup
    for _ in range(WARMUP_RUNS):
        session.run(output_names, {input_name: dummy_input})

    # Benchmark loop
    latencies = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        session.run(output_names, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0) # convert to ms

    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency

    return {
        "name": name,
        "avg": avg_latency,
        "min": min_latency,
        "max": max_latency,
        "std": std_latency,
        "fps": fps,
        "input_shape": f"{run_shape}",
    }

def main():
    print("=" * 70)
    print("           ONNX MODEL PERFORMANCE BENCHMARK (PYTHON/CPU)")
    print("=" * 70)
    
    results = []
    for name, path in MODELS.items():
        res = benchmark_model(name, path)
        if res:
            results.append(res)

    print("\n" + "=" * 78)
    print(f"{'Model Name':<18} | {'Input Shape':<15} | {'Avg (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10} | {'FPS':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<18} | {r['input_shape']:<15} | {r['avg']:10.2f} | {r['min']:10.2f} | {r['max']:10.2f} | {r['fps']:8.1f}")
    print("=" * 78)
    print(f"Warmup runs: {WARMUP_RUNS} | Benchmark runs: {BENCHMARK_RUNS}")

if __name__ == "__main__":
    main()
