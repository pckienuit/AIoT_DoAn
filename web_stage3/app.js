const API_BASE = "http://127.0.0.1:8010";

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
};

let lastFlightId = null;
let lastEmbedding = null;

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
      lastEmbedding = createUnitEmbedding(`${booking.booking_code}:${passenger.email}`);
      const registered = await requestJson("/api/face/register", {
        method: "POST",
        body: JSON.stringify({ booking_id: booking.id, embedding: lastEmbedding }),
      });
      const synced = await requestJson(`/api/sync/${flight.id}`);
      elements.pointText.textContent = registered.point_id;
      elements.syncText.textContent = String(synced.count);
      elements.matchBtn.disabled = false;
      log("Face registered and synced", { registered, synced_count: synced.count });
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

elements.checkHealthBtn.addEventListener("click", checkBackend);
elements.form.addEventListener("submit", createBookingFlow);
elements.matchBtn.addEventListener("click", runMatchTest);
elements.resetBtn.addEventListener("click", resetSample);

resetSample();
log("Stage 3 console loaded");
