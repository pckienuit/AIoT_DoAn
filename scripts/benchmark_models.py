import time
import numpy as np
import onnxruntime as ort
import os

# --- Dummy L1Dist layer for Siamese ---
try:
    import tensorflow as tf
    class L1Dist(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__()
        def call(self, inputs):
            input_embedding, validation_embedding = inputs
            return tf.math.abs(input_embedding - validation_embedding)
except ImportError:
    pass

def benchmark_keras_model(model_path, input_shape, is_siamese=False):
    print(f"\n[Keras] Loading {os.path.basename(model_path)}...")
    try:
        if is_siamese:
            model = tf.keras.models.load_model(model_path, custom_objects={'L1Dist': L1Dist})
            dummy_input = [np.random.rand(*input_shape).astype(np.float32), np.random.rand(*input_shape).astype(np.float32)]
        else:
            model = tf.keras.models.load_model(model_path)
            dummy_input = np.random.rand(*input_shape).astype(np.float32)
        
        # Warm-up
        for _ in range(5):
            model.predict(dummy_input, verbose=0)
        
        print("Running benchmark (100 iterations)...")
        start_time = time.time()
        for _ in range(100):
            model.predict(dummy_input, verbose=0)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        fps = 1.0 / avg_time
        print(f"--> Average Time: {avg_time*1000:.2f} ms")
        print(f"--> FPS: {fps:.2f}")
    except Exception as e:
        print(f"Failed to benchmark {os.path.basename(model_path)}: {e}")

def benchmark_onnx_model(model_path, input_shape):
    print(f"\n[ONNX] Loading {os.path.basename(model_path)}...")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        
        # Warm-up
        for _ in range(5):
            session.run(None, {input_name: dummy_input})
            
        print("Running benchmark (100 iterations)...")
        start_time = time.time()
        for _ in range(100):
            session.run(None, {input_name: dummy_input})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        fps = 1.0 / avg_time
        print(f"--> Average Time: {avg_time*1000:.2f} ms")
        print(f"--> FPS: {fps:.2f}")
    except Exception as e:
        print(f"Failed to benchmark {os.path.basename(model_path)}: {e}")

if __name__ == "__main__":
    print("="*50)
    print(" STAGE 1: FACE DETECTION BENCHMARK")
    print("="*50)
    # Keras FaceTracker
    benchmark_keras_model("C:/Users/phanc/Downloads/test_model/test_model/facetracker.keras", (1, 120, 120, 3))
    # YOLOv8 ONNX
    benchmark_onnx_model("d:/AIoT_DoAn/models/exports/face_detect_v9.onnx", (1, 3, 320, 320))
    
    print("\n" + "="*50)
    print(" STAGE 2: FACE RECOGNITION BENCHMARK")
    print("="*50)
    # Siamese Keras
    # Shape for Nicholas Renotte tutorial is usually 100x100 or 105x105, let's try 100x100 and then 105x105 if it fails.
    benchmark_keras_model("C:/Users/phanc/Downloads/Face_Regconition/Face_Regconition/siamesemodel.keras", (1, 100, 100, 3), is_siamese=True)
    # ArcFace ONNX
    benchmark_onnx_model("d:/AIoT_DoAn/models/exports/face_recognize_arcface_p3.onnx", (1, 3, 112, 112))
    
    print("\nBenchmark completed!")
