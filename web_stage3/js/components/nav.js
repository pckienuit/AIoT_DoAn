/**
 * nav.js — Navigation bar component.
 * Renders into #nav-placeholder in every page.
 */

import { getUser, isLoggedIn, logout, subscribe as authSubscribe } from "../auth.js";
import { navigate } from "../router.js";

const NAV_LINKS = [
  { href: "/", label: "Trang chủ", icon: "⌂" },
  { href: "/lookup", label: "Tra cứu vé", icon: "🔍" },
  { href: "/my-tickets", label: "Vé của tôi", icon: "⊞", auth: true },
];

function render() {
  const user = getUser();
  const loggedIn = isLoggedIn();

  const linksHtml = NAV_LINKS
    .filter(l => !l.auth || loggedIn)
    .map(l => {
      const active = location.pathname === l.href ? " is-active" : "";
      return `<a href="${l.href}" class="nav__link${active}">${l.label}</a>`;
    }).join("");

  const authHtml = loggedIn && user
    ? `
      <div class="nav__user">
        <div class="nav__dropdown">
          <div class="nav__avatar" id="navAvatar" tabindex="0" role="button" aria-haspopup="true">
            ${user.full_name?.charAt(0)?.toUpperCase() || "U"}
          </div>
          <div class="nav__dropdown-menu" id="navDropdown">
            <div class="nav__dropdown-item" style="pointer-events:none; font-size:0.8rem; color:var(--ink);">
              ${user.full_name || user.email}
            </div>
            <a href="/my-tickets" class="nav__dropdown-item">⊞ Vé của tôi</a>
            <button class="nav__dropdown-item nav__dropdown-item--danger" id="navLogoutBtn">
              ⏻ Đăng xuất
            </button>
          </div>
        </div>
      </div>`
    : `
      <div class="nav__actions">
        <a href="/login" class="btn btn--ghost btn--sm">Đăng nhập</a>
        <a href="/register" class="btn btn--primary btn--sm">Đăng ký</a>
      </div>`;

  const navHtml = `
    <nav class="nav" role="navigation" aria-label="Main navigation">
      <div class="nav__inner shell">
        <a href="/" class="nav__brand">
          <div class="nav__logo" aria-hidden="true">✈</div>
          <span class="nav__brand-name">AIoT Flight</span>
        </a>

        <div class="nav__links">
          ${linksHtml}
        </div>

        ${authHtml}

        <button class="nav__hamburger" id="navHamburger" aria-label="Toggle menu" aria-expanded="false">
          <span></span><span></span><span></span>
        </button>
      </div>

      <!-- Mobile menu -->
      <div class="nav__mobile-menu" id="navMobileMenu">
        ${linksHtml}
        ${loggedIn
          ? `<button class="nav__link" id="navLogoutBtnMobile">⏻ Đăng xuất</button>`
          : `<a href="/login" class="nav__link">Đăng nhập</a><a href="/register" class="nav__link">Đăng ký</a>`
        }
      </div>
    </nav>`;

  const placeholder = document.getElementById("nav-placeholder");
  if (placeholder) {
    placeholder.innerHTML = navHtml;
    attachNavEvents();
  }
}

function attachNavEvents() {
  const avatar = document.getElementById("navAvatar");
  const dropdown = document.getElementById("navDropdown");
  const hamburger = document.getElementById("navHamburger");
  const mobileMenu = document.getElementById("navMobileMenu");

  if (avatar && dropdown) {
    avatar.addEventListener("click", () => {
      dropdown.classList.toggle("is-open");
    });
    avatar.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") dropdown.classList.toggle("is-open");
    });
    document.addEventListener("click", e => {
      if (!avatar.contains(e.target)) dropdown.classList.remove("is-open");
    });
  }

  if (hamburger && mobileMenu) {
    hamburger.addEventListener("click", () => {
      const isOpen = mobileMenu.classList.toggle("is-open");
      hamburger.setAttribute("aria-expanded", String(isOpen));
    });
  }

  const logoutBtn = document.getElementById("navLogoutBtn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", async () => {
      await logout();
      navigate("/");
    });
  }

  const logoutBtnMobile = document.getElementById("navLogoutBtnMobile");
  if (logoutBtnMobile) {
    logoutBtnMobile.addEventListener("click", async () => {
      await logout();
      navigate("/");
    });
  }
}

// Auto-render on auth changes
authSubscribe(() => render());

export { render };
