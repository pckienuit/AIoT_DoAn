/**
 * search.js — Search results page.
 */
import { apiSearchFlights } from "../api.js";
import { navigate } from "../router.js";
import { saveBookingFlow } from "../storage.js";
import { createFlightCard } from "../components/flight-card.js";

let selectedFlightId = null;

export async function initSearch(params, route) {
  const url = new URL(location.href);
  const origin = url.searchParams.get("origin") || "";
  const destination = url.searchParams.get("destination") || "";
  const date = url.searchParams.get("date") || "";
  const passengers = parseInt(url.searchParams.get("passengers") || "1");

  // Update header
  const headerEl = document.getElementById("searchHeader");
  if (headerEl) {
    headerEl.innerHTML = `
      <div class="eyebrow">Kết quả tìm kiếm</div>
      <h1>${origin ? origin + " → " : ""}${destination ? destination : "Tất cả"} · ${date ? new Date(date).toLocaleDateString("vi-VN", {day:"2-digit",month:"short"}) : "Ngày"}</h1>
    `;
  }

  const container = document.getElementById("flightResults");
  if (!container) return;

  container.innerHTML = `
    <div style="text-align:center; padding:40px; color:var(--muted);">
      <div class="spinner" style="display:inline-block;width:24px;height:24px;border:3px solid var(--line);border-top-color:var(--signal);border-radius:50%;animation:spin 0.7s linear infinite;margin-bottom:12px;"></div>
      <p>Đang tìm chuyến bay…</p>
    </div>`;

  try {
    const flights = await apiSearchFlights({ origin, destination, date, passengers });

    if (!flights.length) {
      container.innerHTML = `
        <div class="empty-state">
          <svg viewBox="0 0 24 24"><path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg>
          <h3>Không tìm thấy chuyến bay</h3>
          <p>Không có chuyến bay nào phù hợp với tiêu chí tìm kiếm của bạn. Thử thay đổi điểm đến hoặc ngày khác.</p>
          <a href="/" class="btn btn--primary">Quay lại tìm kiếm</a>
        </div>`;
      return;
    }

    // Sort options
    const sortEl = document.getElementById("sortSelect");
    let sorted = [...flights];
    function applySort() {
      const sort = sortEl?.value || "price";
      if (sort === "price") sorted.sort((a, b) => (a.price_per_person || 0) - (b.price_per_person || 0));
      if (sort === "time") sorted.sort((a, b) => (a.departure_time || "").localeCompare(b.departure_time || ""));
      if (sort === "duration") sorted.sort((a, b) => (a.duration_minutes || 0) - (b.duration_minutes || 0));
      renderCards(sorted);
    }

    sortEl?.addEventListener("change", applySort);

    function renderCards(list) {
      container.innerHTML = `<div class="flight-results">${list.map(f => createFlightCard(f, {
        onSelect: () => selectFlight(f),
        selected: f.id === selectedFlightId,
      })).join("")}</div>`;

      container.querySelectorAll(".flight-card").forEach(card => {
        card.addEventListener("click", (e) => {
          if (e.target.closest("button")) return; // Don't select if clicking button
          const id = parseInt(card.dataset.flightId);
          const flight = list.find(f => f.id === id);
          if (flight) selectFlight(flight);
        });
        card.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            const id = parseInt(card.dataset.flightId);
            const flight = list.find(f => f.id === id);
            if (flight) selectFlight(flight);
          }
        });
      });
    }

    applySort();

    // Update stats
    const statsEl = document.getElementById("searchStats");
    if (statsEl) {
      const minPrice = flights.reduce((m, f) => Math.min(m, f.price_per_person || Infinity), Infinity);
      statsEl.innerHTML = `
        <div class="stat-item"><div class="stat-item__value">${flights.length}</div><div class="stat-item__label">Chuyến bay</div></div>
        <div class="stat-item"><div class="stat-item__value">${formatVND(minPrice === Infinity ? 0 : minPrice)}</div><div class="stat-item__label">Từ / khách</div></div>
        <div class="stat-item"><div class="stat-item__value">${passengers}</div><div class="stat-item__label">Hành khách</div></div>
        <div class="stat-item"><div class="stat-item__value">${date ? new Date(date).toLocaleDateString("vi-VN", {day:"2-digit",month:"short"}) : "—"}</div><div class="stat-item__label">Ngày đi</div></div>`;
    }

  } catch (err) {
    container.innerHTML = `
      <div class="alert alert--error">
        <strong>Lỗi:</strong> ${err.message}
      </div>`;
  }
}

function selectFlight(flight) {
  selectedFlightId = flight.id;
  saveBookingFlow({
    flightId: flight.id,
    flightNumber: flight.flight_number,
    pricePerPerson: flight.price_per_person,
    totalPrice: flight.total_price || flight.price_per_person,
    passengers: parseInt(new URL(location.href).searchParams.get("passengers") || "1"),
  });
  navigate(`/booking?flight=${flight.id}`);
}

function formatVND(amount) {
  if (!amount) return "—";
  return new Intl.NumberFormat("vi-VN").format(Math.round(amount)) + " ₫";
}
