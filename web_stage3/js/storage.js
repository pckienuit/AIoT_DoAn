/**
 * storage.js — localStorage helpers for session/persist data.
 */

const PREFIX = "aiot_";

function get(key, defaultValue = null) {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw ? JSON.parse(raw) : defaultValue;
  } catch { return defaultValue; }
}

function set(key, value) {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch { /* storage full or unavailable */ }
}

function remove(key) {
  localStorage.removeItem(PREFIX + key);
}

// Booking flow state (persisted across pages)
const BOOKING_FLOW_KEY = "booking_flow";

function saveBookingFlow(data) {
  set(BOOKING_FLOW_KEY, { ...getBookingFlow(), ...data, _ts: Date.now() });
}

function getBookingFlow() {
  return get(BOOKING_FLOW_KEY, {});
}

function clearBookingFlow() {
  remove(BOOKING_FLOW_KEY);
}

// Flow is considered stale if older than 30 minutes — prevents old search data
// from leaking between sessions while preserving flow across login redirects.
const FLOW_TTL_MS = 30 * 60 * 1000;

function clearStaleFlow() {
  const flow = getBookingFlow();
  if (flow && flow._ts && Date.now() - flow._ts < FLOW_TTL_MS) return;
  clearBookingFlow();
}

export { get, set, remove, saveBookingFlow, getBookingFlow, clearBookingFlow, clearStaleFlow };
