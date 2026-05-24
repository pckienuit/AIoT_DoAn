const API_BASE = "http://127.0.0.1:8010";
const MP_URL = (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
const DATASET_SAMPLES = [
  "/data/images/calib_000_029185.jpg",
  "/data/images/calib_001_006557.jpg",
  "/data/images/calib_002_072098.jpg",
  "/data/images/calib_003_064197.jpg",
  "/data/images/calib_004_058514.jpg",
];

const elements = {
  form: document.querySelector("#registrationForm"),
  checkHealthBtn: document.querySelector("#checkHealthBtn"),
  healthText: document.querySelector("#healthText"),
  bookingText: document.querySelector("#bookingText"),
  pointText: document.querySelector("#pointText"),
  syncText: document.querySelector("#syncText"),
  logOutput: document.querySelector("#logOutput"),
  matchBtn: document.querySelector("#matchBtn"),
  resetBtn: document.querySelector("#resetBtn"),
  submitBtn: document.querySelector("#submitBtn"),
  passengerEmail: document.querySelector("#passengerEmail"),
  flightCode: document.querySelector("#flightCode"),
  registerFace: document.querySelector("#registerFace"),
  embeddingStatus: document.querySelector("#embeddingStatus"),
  faceFileInput: document.querySelector("#faceFileInput"),
  cameraVideo: document.querySelector("#cameraVideo"),
  facePreviewCanvas: document.querySelector("#facePreviewCanvas"),
  facePreview: document.querySelector(".face-preview"),
  startCameraBtn: document.querySelector("#startCameraBtn"),
  captureCameraBtn: document.querySelector("#captureCameraBtn"),
  loadDatasetBtn: document.querySelector("#loadDatasetBtn"),
  sourceTabs: document.querySelectorAll(".source-tab"),
  uploadControl: document.querySelector("#uploadControl"),
  cameraControl: document.querySelector("#cameraControl"),
  datasetControl: document.querySelector("#datasetControl"),
  submitReadiness: document.querySelector("#submitReadiness"),
  readinessTitle: document.querySelector("#readinessTitle"),
  readinessDetail: document.querySelector("#readinessDetail"),
  submitModeText: document.querySelector("#submitModeText"),
};

let lastFlightId = null;
let lastEmbedding = null;
let selectedEmbedding = null;
let ortSessionV9 = null;
let ortSessionP3 = null;
let detector = null;
let cameraStream = null;
let activeSource = "upload";

function sampleSuffix() {
  return String(Date.now()).slice(-6);
}

function resetSample() {
  const suffix = sampleSuffix();
  elements.passengerEmail.value = `stage3+${suffix}@example.com`;
  elements.flightCode.value = `ST${suffix.slice(-4)}`;
}

function log(message, data = null) {
  const timestamp = new Date().toLocaleTimeString();
  const payload = data ? `\n${JSON.stringify(data, null, 2)}` : "";
  elements.logOutput.textContent = `[${timestamp}] ${message}${payload}\n\n${elements.logOutput.textContent}`;
}

function setEmbeddingStatus(message) {
  elements.embeddingStatus.textContent = message;
}

function updateSubmitReadiness() {
  elements.submitReadiness.classList.remove("submit-readiness--real", "submit-readiness--fallback", "submit-readiness--skip");

  if (!elements.registerFace.checked) {
    elements.submitReadiness.classList.add("submit-readiness--skip");
    elements.readinessTitle.textContent = "Face registration skipped";
    elements.readinessDetail.textContent = "Booking will be created without sending any embedding to Qdrant.";
    elements.submitModeText.textContent = "Skip face";
    return;
  }

  if (selectedEmbedding) {
    elements.submitReadiness.classList.add("submit-readiness--real");
    elements.readinessTitle.textContent = "Real ONNX embedding ready";
    elements.readinessDetail.textContent = "Submit will send the selected 128D ArcFace P3 vector as plaintext JSON.";
    elements.submitModeText.textContent = "Real ONNX";
    return;
  }

  elements.submitReadiness.classList.add("submit-readiness--fallback");
  elements.readinessTitle.textContent = "Fallback vector mode";
  elements.readinessDetail.textContent = "Face registration is ON, but no real embedding is selected yet. Submit will use a deterministic test vector.";
  elements.submitModeText.textContent = "Fallback vector";
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(data?.detail || `HTTP ${response.status}`);
  }
  return data;
}

function createUnitEmbedding(seedText) {
  let seed = 2166136261;
  for (const char of seedText) {
    seed ^= char.charCodeAt(0);
    seed = Math.imul(seed, 16777619);
  }

  const values = [];
  let sum = 0;
  for (let index = 0; index < 128; index += 1) {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    const value = ((seed >>> 0) / 4294967295) * 2 - 1;
    values.push(value);
    sum += value * value;
  }

  const norm = Math.sqrt(sum) || 1;
  return values.map((value) => value / norm);
}

function l2Normalize(vector) {
  let sum = 0;
  for (const value of vector) sum += value * value;
  const norm = Math.sqrt(sum) || 1;
  return Array.from(vector, (value) => value / norm);
}

function getTensorFromCanvasV9(canvas) {
  const imageData = canvas.getContext("2d").getImageData(0, 0, 224, 224).data;
  const area = 224 * 224;
  const data = new Float32Array(3 * area);
  let pixel = 0;
  for (let index = 0; index < imageData.length; index += 4) {
    data[pixel] = imageData[index] / 255.0;
    data[pixel + area] = imageData[index + 1] / 255.0;
    data[pixel + 2 * area] = imageData[index + 2] / 255.0;
    pixel += 1;
  }
  return new ort.Tensor("float32", data, [1, 3, 224, 224]);
}

function getTensorFromCanvasP3(canvas) {
  const imageData = canvas.getContext("2d").getImageData(0, 0, 112, 112).data;
  const area = 112 * 112;
  const data = new Float32Array(3 * area);
  let pixel = 0;
  for (let index = 0; index < imageData.length; index += 4) {
    data[pixel] = (imageData[index] - 127.5) / 128.0;
    data[pixel + area] = (imageData[index + 1] - 127.5) / 128.0;
    data[pixel + 2 * area] = (imageData[index + 2] - 127.5) / 128.0;
    pixel += 1;
  }
  return new ort.Tensor("float32", data, [1, 3, 112, 112]);
}

async function initModels() {
  if (ortSessionV9 && ortSessionP3 && detector) return;
  setEmbeddingStatus("Loading V9 + ArcFace P3 models...");
  ort.env.wasm.numThreads = 1;
  [ortSessionV9, ortSessionP3] = await Promise.all([
    ort.InferenceSession.create("/models/exports/face_detect_v9.onnx", { executionProviders: ["wasm"] }),
    ort.InferenceSession.create("/models/exports/face_recognize_arcface_p3.onnx", { executionProviders: ["wasm"] }),
  ]);
  detector = new FaceDetection({ locateFile: MP_URL });
  detector.setOptions({ model: "short", minDetectionConfidence: 0.5 });
  await detector.initialize();
  setEmbeddingStatus("Models ready. Choose upload, webcam, or dataset image.");
}

function bboxToPixels(bbox, width, height) {
  if (bbox.xCenter !== undefined) {
    const w = bbox.width * width;
    const h = bbox.height * height;
    return { x: bbox.xCenter * width - w / 2, y: bbox.yCenter * height - h / 2, w, h };
  }
  return { x: bbox.xMin * width, y: bbox.yMin * height, w: bbox.width * width, h: bbox.height * height };
}

function drawContain(source, canvas) {
  const context = canvas.getContext("2d");
  context.fillStyle = "#030403";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const width = source.videoWidth || source.naturalWidth || source.width;
  const height = source.videoHeight || source.naturalHeight || source.height;
  const scale = Math.min(canvas.width / width, canvas.height / height);
  const drawWidth = width * scale;
  const drawHeight = height * scale;
  context.drawImage(source, (canvas.width - drawWidth) / 2, (canvas.height - drawHeight) / 2, drawWidth, drawHeight);
}

async function extractEmbeddingFromImageSource(source) {
  await initModels();
  const width = source.videoWidth || source.naturalWidth || source.width;
  const height = source.videoHeight || source.naturalHeight || source.height;
  const detectionCanvas = document.createElement("canvas");
  detectionCanvas.width = width;
  detectionCanvas.height = height;
  detectionCanvas.getContext("2d").drawImage(source, 0, 0, width, height);

  const results = await new Promise((resolve) => {
    detector.onResults(resolve);
    detector.send({ image: detectionCanvas });
  });
  if (!results.detections || results.detections.length === 0) {
    throw new Error("Face not found in selected image");
  }

  const box = bboxToPixels(results.detections[0].boundingBox, width, height);
  const cropW = box.w * 1.5;
  const cropH = box.h * 1.8;
  const cropX = box.x + box.w / 2 - cropW / 2;
  const cropY = box.y + box.h * 0.4 - cropH * 0.51;

  const v9Canvas = document.createElement("canvas");
  v9Canvas.width = 224;
  v9Canvas.height = 224;
  const v9Context = v9Canvas.getContext("2d");
  v9Context.fillStyle = "black";
  v9Context.fillRect(0, 0, 224, 224);

  const sX = Math.max(0, cropX);
  const sY = Math.max(0, cropY);
  const sW = Math.min(width - sX, cropX + cropW - sX);
  const sH = Math.min(height - sY, cropY + cropH - sY);
  if (sW <= 0 || sH <= 0) throw new Error("Invalid face crop");
  v9Context.drawImage(source, sX, sY, sW, sH, (sX - cropX) * (224 / cropW), (sY - cropY) * (224 / cropH), sW * (224 / cropW), sH * (224 / cropH));

  const feedsV9 = { [ortSessionV9.inputNames[0]]: getTensorFromCanvasV9(v9Canvas) };
  const outV9 = await ortSessionV9.run(feedsV9);
  const classKey = ortSessionV9.outputNames.find((name) => name.includes("class")) || ortSessionV9.outputNames[0];
  const landmarkKey = ortSessionV9.outputNames.find((name) => name.includes("landmark")) || ortSessionV9.outputNames[2] || ortSessionV9.outputNames[1];
  const score = 1.0 / (1.0 + Math.exp(-outV9[classKey].data[0]));
  if (score <= 0.4) throw new Error(`Low landmark confidence: ${score.toFixed(2)}`);

  const landmarks = outV9[landmarkKey].data;
  const points = [];
  for (let index = 0; index < 5; index += 1) {
    points.push(landmarks[index * 2] * cropW + cropX, landmarks[index * 2 + 1] * cropH + cropY);
  }

  const xs = [points[0], points[2], points[4], points[6], points[8]];
  const ys = [points[1], points[3], points[5], points[7], points[9]];
  const centerX = xs.reduce((a, b) => a + b, 0) / 5;
  const centerY = ys.reduce((a, b) => a + b, 0) / 5;
  const side = Math.max((Math.max(...xs) - Math.min(...xs)) * 2.4, (Math.max(...ys) - Math.min(...ys)) * 2.0, 64);
  const recogCropX = centerX - side / 2;
  const recogCropY = centerY - side * 0.45;

  const p3Canvas = document.createElement("canvas");
  p3Canvas.width = 112;
  p3Canvas.height = 112;
  const p3Context = p3Canvas.getContext("2d");
  p3Context.fillStyle = "black";
  p3Context.fillRect(0, 0, 112, 112);

  const rX = Math.max(0, recogCropX);
  const rY = Math.max(0, recogCropY);
  const rW = Math.min(width - rX, recogCropX + side - rX);
  const rH = Math.min(height - rY, recogCropY + side - rY);
  if (rW <= 0 || rH <= 0) throw new Error("Invalid recognition crop");
  p3Context.drawImage(source, rX, rY, rW, rH, (rX - recogCropX) * (112 / side), (rY - recogCropY) * (112 / side), rW * (112 / side), rH * (112 / side));

  const outP3 = await ortSessionP3.run({ [ortSessionP3.inputNames[0]]: getTensorFromCanvasP3(p3Canvas) });
  const vector = l2Normalize(outP3[ortSessionP3.outputNames[0]].data);
  const preview = elements.facePreviewCanvas.getContext("2d");
  preview.clearRect(0, 0, 224, 224);
  preview.imageSmoothingEnabled = true;
  preview.drawImage(p3Canvas, 0, 0, 224, 224);
  return { vector, score };
}

async function loadImageFromUrl(url) {
  const image = new Image();
  image.crossOrigin = "anonymous";
  image.src = `${url}?t=${Date.now()}`;
  await image.decode();
  return image;
}

async function processImageSource(source, label) {
  try {
    setEmbeddingStatus(`Extracting embedding from ${label}...`);
    const result = await extractEmbeddingFromImageSource(source);
    selectedEmbedding = result.vector;
    lastEmbedding = selectedEmbedding;
    updateSubmitReadiness();
    elements.facePreview.classList.remove("has-video");
    setEmbeddingStatus(`Real embedding ready from ${label}. V9 score=${result.score.toFixed(3)} · dims=${selectedEmbedding.length}`);
    log("Real face embedding extracted", { source: label, score: result.score, dims: selectedEmbedding.length });
  } catch (error) {
    selectedEmbedding = null;
    updateSubmitReadiness();
    setEmbeddingStatus(`Embedding failed: ${error.message}`);
    log("Embedding extraction failed", { source: label, error: error.message });
  }
}

async function checkBackend() {
  elements.healthText.textContent = "Checking...";
  try {
    const [health, vector] = await Promise.all([
      requestJson("/health"),
      requestJson("/health/vector", { method: "POST" }),
    ]);
    elements.healthText.textContent = `API OK · DB tables: ${health.database.tables.length} · Vector: ${vector.vector.collection}`;
    log("Backend is ready", { health, vector });
  } catch (error) {
    elements.healthText.textContent = `Backend error: ${error.message}`;
    log("Backend check failed", { error: error.message });
  }
}

function getFormPayload() {
  const formData = new FormData(elements.form);
  return {
    passenger: {
      name: formData.get("passengerName").trim(),
      email: formData.get("passengerEmail").trim(),
      phone: formData.get("passengerPhone").trim(),
    },
    flight: {
      flight_code: formData.get("flightCode").trim(),
      destination: formData.get("destination").trim(),
      departure_time: formData.get("departureTime").trim(),
      boarding_time: formData.get("boardingTime").trim(),
      gate: formData.get("gate").trim(),
      status: "boarding",
    },
    seatNumber: formData.get("seatNumber").trim(),
  };
}

async function createBookingFlow(event) {
  event.preventDefault();
  elements.submitBtn.disabled = true;
  elements.matchBtn.disabled = true;
  log("Starting booking flow...");

  try {
    const payload = getFormPayload();
    const passenger = await requestJson("/api/passengers", {
      method: "POST",
      body: JSON.stringify(payload.passenger),
    });
    const flight = await requestJson("/api/flights", {
      method: "POST",
      body: JSON.stringify(payload.flight),
    });
    const booking = await requestJson("/api/bookings", {
      method: "POST",
      body: JSON.stringify({
        booking_code: `WEB-${sampleSuffix()}`,
        passenger_id: passenger.id,
        flight_id: flight.id,
        seat_number: payload.seatNumber,
      }),
    });

    lastFlightId = flight.id;
    elements.bookingText.textContent = `${booking.booking_code} · ${booking.flight_code} · ${booking.seat_number}`;
    log("Booking created", { passenger, flight, booking });

    if (elements.registerFace.checked) {
      lastEmbedding = selectedEmbedding || createUnitEmbedding(`${booking.booking_code}:${passenger.email}`);
      const registered = await requestJson("/api/face/register", {
        method: "POST",
        body: JSON.stringify({ booking_id: booking.id, embedding: lastEmbedding }),
      });
      const synced = await requestJson(`/api/sync/${flight.id}`);
      elements.pointText.textContent = registered.point_id;
      elements.syncText.textContent = String(synced.count);
      elements.matchBtn.disabled = false;
      log("Face registered and synced", {
        embedding_source: selectedEmbedding ? "real_onnx" : "fallback_test_vector",
        registered,
        synced_count: synced.count,
      });
    } else {
      lastEmbedding = null;
      elements.pointText.textContent = "Skipped by user";
      elements.syncText.textContent = "0";
      log("Face registration skipped by user");
    }
  } catch (error) {
    log("Flow failed", { error: error.message });
  } finally {
    elements.submitBtn.disabled = false;
  }
}

async function runMatchTest() {
  if (!lastFlightId || !lastEmbedding) {
    log("No registered face available for match test");
    return;
  }
  elements.matchBtn.disabled = true;
  try {
    const result = await requestJson("/api/face/match", {
      method: "POST",
      body: JSON.stringify({ flight_id: lastFlightId, embedding: lastEmbedding }),
    });
    log("Match test complete", result);
  } catch (error) {
    log("Match test failed", { error: error.message });
  } finally {
    elements.matchBtn.disabled = false;
  }
}

function setSource(source) {
  activeSource = source;
  selectedEmbedding = null;
  updateSubmitReadiness();
  elements.sourceTabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.source === source));
  elements.uploadControl.classList.toggle("is-hidden", source !== "upload");
  elements.cameraControl.classList.toggle("is-hidden", source !== "camera");
  elements.datasetControl.classList.toggle("is-hidden", source !== "dataset");
  elements.facePreview.classList.toggle("has-video", source === "camera");
}

async function startCamera() {
  cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  elements.cameraVideo.srcObject = cameraStream;
  await elements.cameraVideo.play();
  elements.captureCameraBtn.disabled = false;
  elements.facePreview.classList.add("has-video");
  setEmbeddingStatus("Webcam started. Capture a frame to extract embedding.");
}

async function captureCameraFrame() {
  if (!elements.cameraVideo.videoWidth) return;
  drawContain(elements.cameraVideo, elements.facePreviewCanvas);
  await processImageSource(elements.cameraVideo, "webcam");
}

elements.checkHealthBtn.addEventListener("click", checkBackend);
elements.form.addEventListener("submit", createBookingFlow);
elements.matchBtn.addEventListener("click", runMatchTest);
elements.resetBtn.addEventListener("click", resetSample);
elements.registerFace.addEventListener("change", updateSubmitReadiness);
elements.sourceTabs.forEach((tab) => tab.addEventListener("click", () => setSource(tab.dataset.source)));
elements.startCameraBtn.addEventListener("click", startCamera);
elements.captureCameraBtn.addEventListener("click", captureCameraFrame);
elements.faceFileInput.addEventListener("change", async () => {
  const file = elements.faceFileInput.files[0];
  if (!file) return;
  const image = new Image();
  image.src = URL.createObjectURL(file);
  await image.decode();
  drawContain(image, elements.facePreviewCanvas);
  await processImageSource(image, "upload");
  URL.revokeObjectURL(image.src);
});
elements.loadDatasetBtn.addEventListener("click", async () => {
  const url = DATASET_SAMPLES[Math.floor(Math.random() * DATASET_SAMPLES.length)];
  const image = await loadImageFromUrl(url);
  drawContain(image, elements.facePreviewCanvas);
  await processImageSource(image, url);
});

resetSample();
setSource(activeSource);
updateSubmitReadiness();
initModels().catch((error) => {
  setEmbeddingStatus(`Model loading failed: ${error.message}. Fallback test vectors remain available.`);
  log("Model loading failed", { error: error.message });
});
log("Stage 3 console loaded");
