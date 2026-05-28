/**
 * api.js — Centralized API client with JWT auth headers.
 */
const API_BASE = (() => {
  // Allow override, otherwise call the same FastAPI origin that serves the web.
  return window.__API_BASE__ || window.location.origin;
})();

// ---------------------------------------------------------------------------
// Low-level fetch
// ---------------------------------------------------------------------------

async function apiRequest(path, options = {}) {
  const token = window._authToken;
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {}),
  };

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  } catch (err) {
    throw new Error(`Network error: ${err.message}`);
  }

  const text = await response.text();
  let data = null;
  try { data = JSON.parse(text); } catch { /* noop */ }

  if (!response.ok) {
    const msg = data?.detail || data?.message || `HTTP ${response.status}`;
    const err = new Error(msg);
    err.status = response.status;
    err.data = data;
    throw err;
  }

  return data;
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

async function apiRegister(payload) {
  return apiRequest("/api/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function apiLogin(email, password) {
  return apiRequest("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

async function apiGetMe() {
  return apiRequest("/api/auth/me");
}

async function apiUpdateMe(payload) {
  return apiRequest("/api/auth/me", {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------------------------
// Airports
// ---------------------------------------------------------------------------

async function apiGetAirports() {
  return apiRequest("/api/airports");
}

async function apiGetStats() {
  return apiRequest("/api/stats");
}

// ---------------------------------------------------------------------------
// Flights
// ---------------------------------------------------------------------------

async function apiSearchFlights({ origin, destination, date, passengers = 1 } = {}) {
  const params = new URLSearchParams();
  if (origin) params.set("origin", origin);
  if (destination) params.set("destination", destination);
  if (date) params.set("flight_date", date);
  if (passengers) params.set("passengers", String(passengers));
  const qs = params.toString();
  return apiRequest(`/api/flights/search${qs ? "?" + qs : ""}`);
}

async function apiGetFlight(id) {
  return apiRequest(`/api/flights/${id}`);
}

async function apiGetFlightSeats(id) {
  return apiRequest(`/api/flights/${id}/seats`);
}

async function apiHoldSeats(payload) {
  return apiRequest("/api/seat-holds", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function apiReleaseSeatHold(holdToken, options = {}) {
  return apiRequest(`/api/seat-holds/${encodeURIComponent(holdToken)}`, {
    method: "DELETE",
    ...options,
  });
}

// ---------------------------------------------------------------------------
// Bookings
// ---------------------------------------------------------------------------

async function apiCreateBooking(payload) {
  return apiRequest("/api/bookings", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function apiGetMyBookings() {
  return apiRequest("/api/bookings");
}

async function apiGetBooking(code) {
  return apiRequest(`/api/bookings/code/${code}`);
}

async function apiCancelBooking(id) {
  return apiRequest(`/api/bookings/${id}/cancel`, { method: "PATCH" });
}

async function apiChangeSeat(bookingId, newSeatNumber) {
  return apiRequest(`/api/bookings/${bookingId}/change-seat`, {
    method: "POST",
    body: JSON.stringify({ new_seat_number: newSeatNumber }),
  });
}

async function apiCheckin(bookingId) {
  return apiRequest(`/api/bookings/${bookingId}/checkin`, { method: "PATCH" });
}

// ---------------------------------------------------------------------------
// Payments
// ---------------------------------------------------------------------------

async function apiInitPayment(bookingId, method) {
  return apiRequest("/api/payments/init", {
    method: "POST",
    body: JSON.stringify({ booking_id: bookingId, method }),
  });
}

async function apiPaymentCallback(txId, status) {
  return apiRequest(`/api/payments/callback?tx_id=${txId}&status=${status}`, {
    method: "POST",
  });
}

async function apiPaymentStatus(bookingId) {
  return apiRequest(`/api/payments/${bookingId}`);
}

// ---------------------------------------------------------------------------
// Face (proto endpoints)
// ---------------------------------------------------------------------------

async function apiFaceRegister(payload) {
  return apiRequest("/api/face/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function apiFaceMatch(flightId, embedding, threshold = 0.020) {
  return apiRequest("/api/face/match", {
    method: "POST",
    body: JSON.stringify({ flight_id: flightId, embedding, threshold }),
  });
}

async function apiSyncFlight(flightId) {
  return apiRequest(`/api/sync/${flightId}`);
}

// ---------------------------------------------------------------------------
// Admin
// ---------------------------------------------------------------------------

async function apiAdminLogin(email, password) {
  return apiRequest("/api/admin/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

async function apiAdminGetStats() {
  return apiRequest("/api/admin/stats");
}

async function apiAdminGetFlights(filters = {}) {
  const params = new URLSearchParams();
  if (filters.status) params.set("status", filters.status);
  if (filters.date) params.set("date", filters.date);
  if (filters.search) params.set("search", filters.search);
  const qs = params.toString();
  return apiRequest(`/api/admin/flights${qs ? "?" + qs : ""}`);
}

async function apiAdminGetFlight(id) {
  return apiRequest(`/api/admin/flights/${id}`);
}

async function apiAdminUpdateFlightStatus(id, status) {
  return apiRequest(`/api/admin/flights/${id}/status`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });
}

async function apiAdminDeleteFlight(id) {
  return apiRequest(`/api/admin/flights/${id}`, { method: "DELETE" });
}

async function apiAdminRestoreFlight(id) {
  return apiRequest(`/api/admin/flights/${id}/restore`, { method: "PATCH" });
}

async function apiAdminGetBookings(filters = {}) {
  const params = new URLSearchParams();
  if (filters.flight_id) params.set("flight_id", filters.flight_id);
  if (filters.status) params.set("status", filters.status);
  if (filters.payment_status) params.set("payment_status", filters.payment_status);
  if (filters.face_registered !== undefined) params.set("face_registered", filters.face_registered);
  const qs = params.toString();
  return apiRequest(`/api/admin/bookings${qs ? "?" + qs : ""}`);
}

async function apiAdminGetBooking(id) {
  return apiRequest(`/api/admin/bookings/${id}`);
}

async function apiAdminUpdateBookingStatus(id, status) {
  return apiRequest(`/api/admin/bookings/${id}/status`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });
}

async function apiAdminDeleteFaceRegistration(bookingId) {
  return apiRequest(`/api/admin/bookings/${bookingId}/face`, { method: "DELETE" });
}

async function apiAdminGetRegisteredFlights() {
  return apiRequest("/api/admin/sync/registered-flights");
}

async function apiAdminSyncTrigger(payload) {
  return apiRequest("/api/admin/sync/trigger", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function apiAdminSyncStatus() {
  return apiRequest("/api/admin/sync/status");
}

async function apiAdminDeleteFlightCache(flightId) {
  return apiRequest(`/api/admin/sync/cache/${flightId}`, { method: "DELETE" });
}

async function apiAdminRunFlightStatusUpdate() {
  return apiRequest("/api/admin/flight-status/update", { method: "POST" });
}

async function apiAdminHealth() {
  return apiRequest("/api/admin/health");
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

async function apiHealth() {
  return apiRequest("/health");
}

async function apiVectorHealth() {
  return apiRequest("/health/vector", { method: "POST" });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Format VND currency.
 */
function formatVND(amount) {
  if (amount == null) return "—";
  return new Intl.NumberFormat("vi-VN").format(Math.round(amount)) + " ₫";
}

/**
 * Format date for display.
 */
function formatDate(dateStr) {
  if (!dateStr) return "—";
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString("vi-VN", { day: "2-digit", month: "short", year: "numeric" });
  } catch { return dateStr; }
}

/**
 * Format time string HH:MM:SS → HH:MM
 */
function formatTime(timeStr) {
  if (!timeStr) return "—";
  return String(timeStr).substring(0, 5);
}

/**
 * Format datetime ISO → HH:MM on date
 */
function formatDateTime(dt) {
  if (!dt) return "—";
  return formatDate(dt) + " · " + formatTime(dt);
}

export {
  apiRequest,
  apiRegister,
  apiLogin,
  apiGetMe,
  apiUpdateMe,
  apiGetAirports,
  apiGetStats,
  apiSearchFlights,
  apiGetFlight,
  apiGetFlightSeats,
  apiHoldSeats,
  apiReleaseSeatHold,
  apiCreateBooking,
  apiGetMyBookings,
  apiGetBooking,
  apiCancelBooking,
  apiChangeSeat,
  apiCheckin,
  apiInitPayment,
  apiPaymentCallback,
  apiPaymentStatus,
  apiFaceRegister,
  apiFaceMatch,
  apiSyncFlight,
  apiHealth,
  apiVectorHealth,
  formatVND,
  formatDate,
  formatTime,
  formatDateTime,
  // Admin
  apiAdminLogin,
  apiAdminGetStats,
  apiAdminGetFlights,
  apiAdminGetFlight,
  apiAdminUpdateFlightStatus,
  apiAdminDeleteFlight,
  apiAdminRestoreFlight,
  apiAdminGetBookings,
  apiAdminGetBooking,
  apiAdminUpdateBookingStatus,
  apiAdminDeleteFaceRegistration,
  apiAdminGetRegisteredFlights,
  apiAdminSyncTrigger,
  apiAdminSyncStatus,
  apiAdminDeleteFlightCache,
  apiAdminRunFlightStatusUpdate,
  apiAdminHealth,
};
