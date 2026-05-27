/**
 * flight-card.js — Flight result card component.
 *
 * Usage:
 *   import { createFlightCard } from "./flight-card.js";
 *   container.innerHTML = createFlightCard(flight, { onSelect });
 */

const AIRLINE_LOGOS = {
  VN: `<svg viewBox="0 0 24 24"><path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg>`,
  default: `<svg viewBox="0 0 24 24"><path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg>`,
};

function getAirlineLogo(flightNumber) {
  const prefix = String(flightNumber || "").substring(0, 2).toUpperCase();
  return AIRLINE_LOGOS[prefix] || AIRLINE_LOGOS.default;
}

function getAirlineName(flightNumber) {
  const prefix = String(flightNumber || "").substring(0, 2).toUpperCase();
  const names = { VN: "Vietnam Airlines", VJ: "VietJet Air", QH: "Bamboo Airways" };
  return names[prefix] || "AIoT Airways";
}

function getDurationMinutes(flight) {
  return flight.duration_minutes || flight._duration_minutes || 0;
}

function formatDuration(minutes) {
  if (!minutes) return "—";
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${h}h ${m.toString().padStart(2, "0")}m`;
}

/**
 * Create HTML string for a flight card.
 *
 * @param {object} flight - Flight data from /api/flights/search
 * @param {object} opts
 * @param {Function} opts.onSelect - Called when card is clicked
 * @param {boolean} opts.selected - Whether this card is selected
 * @returns {string} HTML string
 */
function createFlightCard(flight, { onSelect, selected = false } = {}) {
  const code = flight.flight_number || "—";
  const airline = getAirlineName(code);
  const logo = getAirlineLogo(code);

  const depTime = formatTime(flight.departure_time);
  const arrTime = formatTime(flight.arrival_time);
  const duration = formatDuration(getDurationMinutes(flight));

  const statusLabel = flight.status || "scheduled";
  const statusBadge = {
    scheduled: `<span class="badge badge--muted">Scheduled</span>`,
    boarding:  `<span class="badge badge--signal">Boarding</span>`,
    departed:  `<span class="badge badge--muted">Departed</span>`,
    arrived:   `<span class="badge badge--ok">Arrived</span>`,
    cancelled: `<span class="badge badge--danger">Cancelled</span>`,
  }[statusLabel] || `<span class="badge badge--muted">${statusLabel}</span>`;

  const seats = flight.available_seats || 0;
  const seatsClass = seats < 10 ? "ticket-card__seats--low" : "";
  const seatsText = seats < 10 ? `Chỉ còn ${seats} ghế` : `${seats} ghế trống`;

  const price = flight.price_per_person || flight.total_price || 0;
  const priceFormatted = formatVND(price);

  const selectedClass = selected ? " flight-card--selected" : "";

  return `
    <article class="flight-card${selectedClass}" data-flight-id="${flight.id}" role="button" tabindex="0">
      <div class="flight-card__logo">
        <div class="flight-card__logo-icon" aria-hidden="true">
          ${logo}
        </div>
        <span class="flight-card__logo-name">${airline}</span>
      </div>

      <div class="flight-card__route">
        <div class="flight-card__times">
          <div>
            <div class="flight-card__time">${depTime}</div>
            <div class="flight-card__code">${flight.origin_code}</div>
          </div>
          <div class="flight-card__route-line">
            <span>${duration}</span>
            <hr />
            <svg class="flight-icon" viewBox="0 0 24 24"><path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg>
          </div>
          <div>
            <div class="flight-card__time">${arrTime}</div>
            <div class="flight-card__code">${flight.dest_code}</div>
          </div>
        </div>
        <div class="flight-card__info">
          <div class="flight-card__badges">
            ${statusBadge}
            <span class="badge badge--muted">${flight.plane_type || "—"}</span>
            ${flight.distance_km ? `<span class="badge badge--muted">${flight.distance_km} km</span>` : ""}
          </div>
          <div style="color: var(--muted); font-size: var(--text-xs); font-family: 'JetBrains Mono', monospace;">
            ${flight.origin_city || flight.origin_name || ""} → ${flight.dest_city || flight.dest_name || ""}
          </div>
        </div>
      </div>

      <div class="flight-card__price">
        <div class="flight-card__price-value">${priceFormatted}</div>
        <div class="flight-card__price-unit">/ khách</div>
        <div class="flight-card__seats ${seatsClass}">${seatsText}</div>
        <button class="btn btn--primary btn--sm btn--full" style="margin-top: 8px;">
          Chọn
        </button>
      </div>
    </article>`;
}

function formatTime(t) {
  if (!t) return "—";
  return String(t).substring(0, 5);
}

function formatVND(amount) {
  if (amount == null) return "—";
  return new Intl.NumberFormat("vi-VN").format(Math.round(amount)) + " ₫";
}

export { createFlightCard, formatVND, formatTime };
