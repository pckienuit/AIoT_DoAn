/**
 * auth.js — Auth state management (JWT).
 */
import { apiLogin, apiRegister, apiGetMe } from "./api.js";

// Module state
let _currentUser = null;
let _authToken = null;
const _listeners = new Set();

// ---------------------------------------------------------------------------
// Token management
// ---------------------------------------------------------------------------

function getToken() {
  if (_authToken) return _authToken;
  // Try localStorage
  const saved = localStorage.getItem("aiot_token");
  if (saved) {
    _authToken = saved;
    window._authToken = saved;  // expose for api.js
    // Don't auto-restore user — verify on page load
  }
  return _authToken;
}

function setToken(token) {
  _authToken = token;
  window._authToken = token;  // expose for api.js
  if (token) {
    localStorage.setItem("aiot_token", token);
  } else {
    localStorage.removeItem("aiot_token");
  }
  _notify();
}

function getUser() { return _currentUser; }

function isLoggedIn() { return !!_currentUser && !!_authToken; }

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async function login(email, password) {
  const data = await apiLogin(email, password);
  setToken(data.access_token);
  _currentUser = data.user;
  _notify();
  return data.user;
}

async function register(payload) {
  const data = await apiRegister(payload);
  setToken(data.access_token);
  _currentUser = data.user;
  _notify();
  return data.user;
}

async function logout() {
  _currentUser = null;
  setToken(null);
  _notify();
}

async function restoreSession() {
  const token = getToken();
  if (!token) return null;
  try {
    _currentUser = await apiGetMe();
    _notify();
    return _currentUser;
  } catch {
    setToken(null);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Event system
// ---------------------------------------------------------------------------

function subscribe(fn) {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

function _notify() {
  for (const fn of _listeners) {
    try { fn({ user: _currentUser, loggedIn: isLoggedIn() }); } catch { /* noop */ }
  }
}

// ---------------------------------------------------------------------------
// Init: restore session from stored token
// ---------------------------------------------------------------------------

restoreSession();

export {
  getToken,
  getUser,
  isLoggedIn,
  login,
  register,
  logout,
  restoreSession,
  subscribe,
};
