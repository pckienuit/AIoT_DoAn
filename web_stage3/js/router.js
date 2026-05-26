/**
 * router.js — Minimal SPA router using History API.
 *
 * Routes are defined as { path, title, render, requiresAuth } objects.
 * Call router.init() once on load.
 */

import { isLoggedIn, subscribe as authSubscribe } from "./auth.js";

let _routes = [];
let _currentPath = "";
let _onNavigateCallbacks = new Set();

function addRoute(route) {
  _routes.push(route);
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

function navigate(path, replace = false) {
  if (replace) {
    history.replaceState(null, "", path);
  } else {
    history.pushState(null, "", path);
  }
  _handleRoute();
}

function _handleRoute() {
  const path = location.pathname;
  _currentPath = path;

  // Find matching route (longest prefix first)
  let matched = null;
  let params = {};

  for (const route of _routes) {
    const match = _matchRoute(route.path, path);
    if (match) {
      matched = route;
      params = match.params;
      break;
    }
  }

  if (!matched) {
    // Fallback to home
    navigate("/", true);
    return;
  }

  // Auth guard
  if (matched.requiresAuth && !isLoggedIn()) {
    const returnUrl = encodeURIComponent(path);
    navigate(`/login?return=${returnUrl}`, true);
    return;
  }

  // Update page title
  document.title = matched.title
    ? `${matched.title} — AIoT Flight`
    : "AIoT Flight";

  // Notify listeners
  for (const cb of _onNavigateCallbacks) {
    try { cb(matched, params); } catch { /* noop */ }
  }

  // Render page
  matched.render(params, matched);
}

function _matchRoute(pattern, path) {
  const pParts = pattern.split("/").filter(Boolean);
  const pathParts = path.split("/").filter(Boolean);

  if (pParts.length !== pathParts.length) return null;

  const params = {};
  for (let i = 0; i < pParts.length; i++) {
    if (pParts[i].startsWith(":")) {
      params[pParts[i].slice(1)] = decodeURIComponent(pathParts[i]);
    } else if (pParts[i] !== pathParts[i]) {
      return null;
    }
  }
  return { params };
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function init() {
  // Handle browser back/forward
  window.addEventListener("popstate", _handleRoute);

  // Initial route
  _handleRoute();

  // Auth state changes — re-evaluate protected routes
  authSubscribe(() => {
    if (_currentPath.startsWith("/login") || _currentPath.startsWith("/register")) return;
    // Re-check current route auth requirement
    const route = _routes.find(r => _matchRoute(r.path, _currentPath));
    if (route?.requiresAuth && !isLoggedIn()) {
      navigate("/", true);
    }
  });
}

function onNavigate(cb) {
  _onNavigateCallbacks.add(cb);
  return () => _onNavigateCallbacks.delete(cb);
}

function getCurrentPath() { return _currentPath; }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function queryParam(key) {
  return new URLSearchParams(location.search).get(key);
}

export { addRoute, init, navigate, onNavigate, getCurrentPath, queryParam };
