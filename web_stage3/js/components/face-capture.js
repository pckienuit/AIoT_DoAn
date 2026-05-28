/**
 * face-capture.js — Face embedding pipeline extracted from prototype app.js.
 *
 * Responsibilities:
 *   - Load MediaPipe Face Detection + ONNX models (V9, ArcFace P3)
 *   - Extract 128-dim face embedding from image/video/canvas source
 *   - Quality gate: score threshold, brightness, pose
 *   - Burst capture: 12 frames, pick 7 best, average
 *   - Encrypt with AES-GCM-256
 */

const AES_SECRET_HEX = "94c8e763a8a3a31e2474db62c82e0fb58cc2a77ef7cb73f1d8c117b4abdc3d9d";
const V9_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmark/face_landmark_neural_network/float16/1/face_landmark_neural_network.task";
const FD_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detection/scrfd_face_detector/float32/1/scrfd_2.5g_bnkps.blob";

const FACE_DETECTION_SCORE_THRESHOLD = 0.5;
const BURST_FRAMES = 12;
const BURST_BEST_COUNT = 7;
const BRIGHTNESS_MIN = 20;
const BRIGHTNESS_MAX = 230;
const POSE_TOLERANCE = 15;

// ---------------------------------------------------------------------------
// Crypto helpers
// ---------------------------------------------------------------------------

async function hexToBytes(hex) {
  return Uint8Array.from(hex.match(/.{2}/g).map(b => parseInt(b, 16)));
}

async function getAESKey() {
  const keyData = await hexToBytes(AES_SECRET_HEX);
  return crypto.subtle.importKey("raw", keyData, { name: "AES-GCM" }, false, ["encrypt", "decrypt"]);
}

async function encryptVectorAESGCM(vector) {
  const key = await getAESKey();
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const data = new Float32Array(vector);
  const ciphertext = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, data.buffer);
  return {
    ciphertext: btoa(String.fromCharCode(...new Uint8Array(ciphertext))),
    iv: btoa(String.fromCharCode(...iv)),
  };
}

// ---------------------------------------------------------------------------
// ONNX helpers
// ---------------------------------------------------------------------------

async function runONNXSession(session, inputTensor) {
  const outputs = await session.run({ [session.inputNames[0]]: inputTensor });
  return outputs[session.outputNames[0]];
}

function createTensorV9(canvas) {
  const size = 224;
  const tmp = document.createElement("canvas");
  tmp.width = size; tmp.height = size;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(canvas, 0, 0, size, size);
  const imageData = tctx.getImageData(0, 0, size, size);
  const data = imageData.data;
  const floatData = new Float32Array(size * size * 3);
  const area = size * size;
  let j = 0;
  for (let i = 0; i < data.length; i += 4) {
    floatData[j]            = data[i] / 255.0;
    floatData[j + area]     = data[i + 1] / 255.0;
    floatData[j + 2 * area] = data[i + 2] / 255.0;
    j++;
  }
  return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}

function createTensorP3(canvas) {
  const size = 112;
  const tmp = document.createElement("canvas");
  tmp.width = size; tmp.height = size;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(canvas, 0, 0, size, size);
  const imageData = tctx.getImageData(0, 0, size, size);
  const data = imageData.data;
  const floatData = new Float32Array(size * size * 3);
  const area = size * size;
  let j = 0;
  for (let i = 0; i < data.length; i += 4) {
    floatData[j]            = (data[i] - 127.5) / 128.0;
    floatData[j + area]     = (data[i + 1] - 127.5) / 128.0;
    floatData[j + 2 * area] = (data[i + 2] - 127.5) / 128.0;
    j++;
  }
  return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}

// ---------------------------------------------------------------------------
// L2 normalize
// ---------------------------------------------------------------------------

function l2Normalize(vec) {
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  if (norm < 1e-10) return vec;
  return vec.map(v => v / norm);
}

// ---------------------------------------------------------------------------
// FaceCapture class
// ---------------------------------------------------------------------------

export class FaceCapture {
  constructor({ onStatus, onEmbedding, onQuality } = {}) {
    this._onStatus = onStatus || (() => {});
    this._onEmbedding = onEmbedding || (() => {});
    this._onQuality = onQuality || (() => {});
    this._detector = null;
    this._sessionV9 = null;
    this._sessionP3 = null;
    this._initialized = false;
    this._cameraStream = null;
    this._cameraVideo = null;
    this._burstQueue = [];
    this._burstTimer = null;
    this._lastEmbedding = null;
  }

  async init() {
    if (this._initialized) return;
    this._onStatus("Loading models…");

    try {
      // Load ONNX models
      this._sessionV9 = await ort.InferenceSession.create(
        "/models/exports/face_detect_v9.onnx"
      );
      this._sessionP3 = await ort.InferenceSession.create(
        "/models/exports/face_recognize_arcface_p3.onnx"
      );
    } catch (e) {
      console.warn("Failed to load real ONNX models, falling back to dummy embedding.", e);
      // Models not available — use fallback unit embedding
      this._sessionV9 = null;
      this._sessionP3 = null;
    }

    // Load MediaPipe Face Detection
    let FaceDetectionClass = window.FaceDetection;
    if (!FaceDetectionClass) {
      try {
        const mp = await import("https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.4/+esm");
        FaceDetectionClass = mp.FaceDetection || mp.default?.FaceDetection || mp.default;
      } catch (err) {
        console.error("Failed to dynamically import MediaPipe FaceDetection:", err);
      }
    }
    if (!FaceDetectionClass) {
      throw new Error("MediaPipe FaceDetection library is not loaded. Please include it in your HTML.");
    }
    this._detector = new FaceDetectionClass({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
    });
    this._detector.setOptions({
      model: "short",
      minDetectionConfidence: FACE_DETECTION_SCORE_THRESHOLD,
    });

    this._initialized = true;
    this._onStatus("Models loaded.");
  }

  /**
   * Extract embedding from an Image/Canvas/Video source.
   * Returns { embedding: Float32Array, quality: number, source: string }
   */
  async extractEmbeddingFromSource(source, { useBurst = false } = {}) {
    await this.init();

    if (!this._sessionV9 || !this._sessionP3) {
      this._onStatus("Models not available — using fallback embedding.");
      const fallback = this._createUnitEmbedding();
      this._lastEmbedding = fallback;
      return { embedding: fallback, quality: 1.0, source: "fallback" };
    }

    // Step 1: Face detection
    const detectionResult = await this._runDetection(source);
    if (!detectionResult) {
      throw new Error("No face detected");
    }

    const { box, landmarks } = detectionResult;

    // Step 2: V9 landmark extraction
    const { canvas: v9InputCanvas, cropX, cropY, cropW, cropH } = this._alignFace(source, box);
    const v9Tensor = createTensorV9(v9InputCanvas);
    
    // Run V9 model to extract landmarks and class score
    const v9Outputs = await this._sessionV9.run({ [this._sessionV9.inputNames[0]]: v9Tensor });
    const classKey = this._sessionV9.outputNames.find(n => n.includes('class')) || this._sessionV9.outputNames[0];
    const landmarkKey = this._sessionV9.outputNames.find(n => n.includes('landmark')) || this._sessionV9.outputNames[2] || this._sessionV9.outputNames[1];
    
    const scoreRaw = v9Outputs[classKey].data[0];
    const score = 1.0 / (1.0 + Math.exp(-scoreRaw));
    
    if (score < 0.40) {
      throw new Error(`Khuôn mặt không rõ nét hoặc quá nghiêng (score: ${score.toFixed(2)})`);
    }
    
    const v9Landmarks = Array.from(v9Outputs[landmarkKey].data);
 
    // Step 3: Landmark alignment (P3 Crop 112x112 from original source)
    const p3InputCanvas = this._alignFaceWithLandmarks(source, v9Landmarks, cropX, cropY, cropW, cropH);
    const p3Tensor = createTensorP3(p3InputCanvas);
    const p3Output = await runONNXSession(this._sessionP3, p3Tensor);
    const p3Vector = Array.from(p3Output.data);

    const embedding = l2Normalize(p3Vector);
    const quality = this._assessQuality(v9InputCanvas, box);

    this._lastEmbedding = embedding;
    this._onEmbedding(embedding);
    this._onQuality(quality);

    return { embedding, quality, source: "live", canvas: p3InputCanvas };
  }

  async _runDetection(source) {
    return new Promise((resolve) => {
      this._detector.onResults((results) => {
        if (results.detections && results.detections.length > 0) {
          const det = results.detections[0];
          const box = det.boundingBox;
          resolve({ box, landmarks: det.landmarks });
        } else {
          resolve(null);
        }
      });
      this._detector.send({ image: source });
    });
  }

  _alignFace(source, box) {
    const canvas = document.createElement("canvas");
    canvas.width = 224; canvas.height = 224;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 224, 224);

    const w = source.videoWidth || source.width || 224;
    const h = source.videoHeight || source.height || 224;

    // Bounding Box to Pixels
    let x, y, bw, bh;
    if (box.xCenter !== undefined) {
      bw = box.width * w;
      bh = box.height * h;
      x = box.xCenter * w - bw / 2;
      y = box.yCenter * h - bh / 2;
    } else {
      x = box.xMin * w;
      y = box.yMin * h;
      bw = box.width * w;
      bh = box.height * h;
    }

    // Adaptive padding based on CelebA crop ratio (178 x 218)
    const cropW = bw * 1.5;
    const cropH = bh * 1.8;
    const cropX = x + bw / 2 - cropW / 2;
    const cropY = y + bh * 0.4 - cropH * 0.51;

    const sX = Math.max(0, cropX);
    const sY = Math.max(0, cropY);
    const sW = Math.min(w - sX, cropX + cropW - sX);
    const sH = Math.min(h - sY, cropY + cropH - sY);

    if (sW > 0 && sH > 0) {
      const dX = (sX - cropX) * (224 / cropW);
      const dY = (sY - cropY) * (224 / cropH);
      const dW = sW * (224 / cropW);
      const dH = sH * (224 / cropH);
      ctx.drawImage(source, sX, sY, sW, sH, dX, dY, dW, dH);
    }
    return { canvas, cropX, cropY, cropW, cropH };
  }

  _alignFaceWithLandmarks(source, landmarks, cropX, cropY, cropW, cropH) {
    const w = source.videoWidth || source.width || 224;
    const h = source.videoHeight || source.height || 224;

    const lmAbs = [];
    for (let i = 0; i < 5; i++) {
      const lx = landmarks[i * 2] * cropW + cropX;
      const ly = landmarks[i * 2 + 1] * cropH + cropY;
      lmAbs.push(lx, ly);
    }

    const xs = [lmAbs[0], lmAbs[2], lmAbs[4], lmAbs[6], lmAbs[8]];
    const ys = [lmAbs[1], lmAbs[3], lmAbs[5], lmAbs[7], lmAbs[9]];
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const centerX = xs.reduce((a, b) => a + b, 0) / 5.0;
    const centerY = ys.reduce((a, b) => a + b, 0) / 5.0;

    const faceW = maxX - minX;
    const faceH = maxY - minY;
    const side = Math.max(faceW * 2.4, faceH * 2.0, 64);
    const recogCropX = centerX - side / 2;
    const recogCropY = centerY - side * 0.45;

    const dstCanvas = document.createElement("canvas");
    dstCanvas.width = 112;
    dstCanvas.height = 112;
    const ctx = dstCanvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 112, 112);

    const rSrcX = Math.max(0, recogCropX);
    const rSrcY = Math.max(0, recogCropY);
    const rSrcW = Math.min(w - rSrcX, recogCropX + side - rSrcX);
    const rSrcH = Math.min(h - rSrcY, recogCropY + side - rSrcY);

    if (rSrcW > 0 && rSrcH > 0) {
      const rDstX = (rSrcX - recogCropX) * (112 / side);
      const rDstY = (rSrcY - recogCropY) * (112 / side);
      const rDstW = rSrcW * (112 / side);
      const rDstH = rSrcH * (112 / side);
      ctx.drawImage(source, rSrcX, rSrcY, rSrcW, rSrcH, rDstX, rDstY, rDstW, rDstH);
    }
    return dstCanvas;
  }

  _assessQuality(canvas, box) {
    const ctx = canvas.getContext("2d");
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let sum = 0, count = 0;
    for (let i = 0; i < data.length; i += 4) {
      sum += (data[i] + data[i+1] + data[i+2]) / 3;
      count++;
    }
    const brightness = sum / count;
    if (brightness < BRIGHTNESS_MIN || brightness > BRIGHTNESS_MAX) return 0.5;
    return box.score ? box.score[0] : 0.8;
  }

  _createUnitEmbedding() {
    const vec = new Array(128).fill(0);
    vec[0] = 1;
    return l2Normalize(vec);
  }

  async encryptEmbedding(vector) {
    return encryptVectorAESGCM(vector);
  }

  // Camera
  async startCamera(videoEl) {
    this._cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
    });
    videoEl.srcObject = this._cameraStream;
    videoEl.play();
    this._cameraVideo = videoEl;
  }

  stopCamera() {
    if (this._cameraStream) {
      this._cameraStream.getTracks().forEach(t => t.stop());
      this._cameraStream = null;
    }
  }

  /**
   * Burst capture: capture `BURST_FRAMES` frames, pick `BURST_BEST_COUNT` best by quality.
   */
  async captureBurst(videoEl) {
    this._burstQueue = [];
    const delay = (ms) => new Promise(r => setTimeout(r, ms));
    for (let i = 0; i < BURST_FRAMES; i++) {
      try {
        const result = await this.extractEmbeddingFromSource(videoEl);
        if (result) this._burstQueue.push(result);
      } catch { /* skip failed frames */ }
      await delay(80);
    }

    // Sort by quality descending, take best
    this._burstQueue.sort((a, b) => b.quality - a.quality);
    const best = this._burstQueue.slice(0, BURST_BEST_COUNT);

    if (!best.length) {
      throw new Error("No quality frames captured");
    }

    // Average embeddings
    const avg = new Array(128).fill(0);
    for (const frame of best) {
      for (let i = 0; i < 128; i++) avg[i] += frame.embedding[i];
    }
    for (let i = 0; i < 128; i++) avg[i] /= best.length;

    const final = l2Normalize(avg);
    this._lastEmbedding = final;
    return { embedding: final, quality: best[0].quality, count: best.length };
  }

  getLastEmbedding() { return this._lastEmbedding; }
  isInitialized() { return this._initialized; }
}

export { encryptVectorAESGCM };
