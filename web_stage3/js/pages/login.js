/**
 * login.js — Login page logic.
 */
import { login, isLoggedIn } from "../auth.js";
import { navigate } from "../router.js";

export async function initLogin(params) {
  if (isLoggedIn()) {
    const returnUrl = new URLSearchParams(location.search).get("return") || "/";
    navigate(returnUrl);
    return;
  }

  const form = document.getElementById("loginForm");
  const loginBtn = document.getElementById("loginBtn");
  const emailInput = document.getElementById("email");
  const passwordInput = document.getElementById("password");
  const errorEl = document.getElementById("formError");

  function showError(msg) {
    errorEl.className = "alert alert--error";
    errorEl.textContent = msg;
    errorEl.style.display = "block";
  }

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorEl.style.display = "none";
    const email = emailInput.value.trim();
    const password = passwordInput.value;

    if (!email || !password) return;

    loginBtn.disabled = true;
    loginBtn.innerHTML = `<span class="spinner"></span> Đang đăng nhập…`;

    try {
      await login(email, password);
      const returnUrl = new URLSearchParams(location.search).get("return") || "/";
      navigate(returnUrl);
    } catch (err) {
      showError(err.message || "Đăng nhập thất bại.");
      loginBtn.disabled = false;
      loginBtn.textContent = "Đăng nhập";
    }
  });
}
