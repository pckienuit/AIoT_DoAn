# Plan: Benchmark & Pipeline Integration (V9 + P3)

This plan outlines the integration of our custom-trained Face Detection/Landmarks (V9) and ArcFace (P3) models into the `EmbededFaceDetect` project, replacing the SFace models. It details how we will run both Web-based and Python-based performance benchmarks.

---

## 📋 Success Criteria

1. **Model Assets**: `face_detect_v9.onnx` and `face_recognize_arcface_p3.onnx` are successfully copied into `models/`.
2. **Python Benchmark**: A script `benchmark_models.py` runs SFace (FP32/INT8), V9, and P3 models on CPU via `onnxruntime`, measuring average latency and throughput.
3. **Web Pipeline**: `index.html` is updated to load V9 and P3, run the full pipeline (MediaPipe Box -> V9 Landmarks -> Alignment -> P3 Embedding), and match identities.
4. **Web Benchmark**: A benchmark panel in `index.html` allows automatic execution of WASM-based ONNX inference on the 4 models, displaying comparative latencies.

---

## 🏗️ File Structure Changes

```plaintext
d:\EmbededFaceDetect\
├── models/
│   ├── face_recognition_sface_2021dec.onnx (Existing)
│   ├── face_recognition_sface_2021dec_int8.onnx (Existing)
│   ├── face_detect_v9.onnx (New - copied)
│   └── face_recognize_arcface_p3.onnx (New - copied)
├── benchmark_models.py (New Python script)
└── index.html (Modified)
```

---

## 📊 Task Breakdown

### Phase 1: Setup & Copying (P0)
- **Task ID**: `setup-models`
- **Action**: Copy the `.onnx` files from `d:\AIoT_DoAn\models\exports\` to `d:\EmbededFaceDetect\models\`.
- **Verify**: Confirm the files exist and check their sizes.

### Phase 2: Python-based Benchmark (P1)
- **Task ID**: `python-benchmark`
- **Action**: Create `benchmark_models.py` which:
  - Loads all 4 models using Python `onnxruntime`.
  - Runs warmup iterations, then measures latency over 100 runs.
  - Outputs a formatted comparison table.
- **Verify**: Run `python benchmark_models.py` and review output.

### Phase 3: Web Application Pipeline Integration (P2)
- **Task ID**: `web-pipeline-integration`
- **Action**: Update `index.html` to:
  - Load `face_detect_v9.onnx` and `face_recognize_arcface_p3.onnx`.
  - Update crop & pre-process routines for V9 (RGB, scaled by 1/255, 224x224).
  - Implement 5-point alignment logic in JavaScript to crop face to 112x112.
  - Update crop & pre-process routines for P3 (RGB, normalized by `(val - 127.5) / 128.0`, 112x112).
- **Verify**: Open `index.html` and verify that face detection and matching still work using V9 + P3.

### Phase 4: Web-based Benchmark Panel (P2)
- **Task ID**: `web-benchmark`
- **Action**: Add a "Benchmark Panel" in `index.html` that:
  - Runs inference on dummy tensors of appropriate sizes for SFace, SFace INT8, V9, and P3.
  - Iterates 50 times per model using `ort.InferenceSession`.
  - Displays a clean visual table/chart comparing the WASM performance (latencies in ms) of all 4 models.
- **Verify**: Trigger benchmark in the browser and confirm it completes and reports results.

---

## 🏁 Phase X: Final Verification

- [ ] Run `python benchmark_models.py` -> Success and output printed.
- [ ] Run a local web server (e.g. `python -m http.server`) -> App loads without ORW issues.
- [ ] Run Web Benchmark -> All 4 models evaluated, results displayed.
- [ ] Verify matching functionality on camera and ref image.
