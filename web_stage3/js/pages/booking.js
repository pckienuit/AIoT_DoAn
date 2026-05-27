/**
 * booking.js — Passenger info + seat selection booking flow.
 */
import { apiGetFlight, apiCreateBooking } from "../api.js";
import { isLoggedIn } from "../auth.js";
import { navigate } from "../router.js";
import { getBookingFlow, saveBookingFlow, clearBookingFlow } from "../storage.js";

export async function initBooking(params) {
  const flow = getBookingFlow();
  const flightId = parseInt(new URL(location.href).searchParams.get("flight") || flow.flightId || "0");

  if (!flightId) {
    navigate("/");
    return;
  }

  // Load flight details
  let flight;
  try {
    flight = await apiGetFlight(flightId);
    saveBookingFlow({ flightId, flight, ...flow });
  } catch {
    navigate("/");
    return;
  }

  renderFlightSummary(flight);
  restoreForm();

  document.getElementById("bookingForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const name = form.passengerName?.value?.trim();
    const email = form.passengerEmail?.value?.trim();
    const phone = form.passengerPhone?.value?.trim();
    const seat = form.seatNumber?.value?.trim() || null;

    if (!name) {
      alert("Vui lòng nhập tên hành khách.");
      return;
    }

    saveBookingFlow({ passengerName: name, passengerEmail: email, passengerPhone: phone, seatNumber: seat });

    if (!isLoggedIn()) {
      // Save flow and go to login
      navigate(`/login?return=${encodeURIComponent(`/booking?flight=${flightId}`)}`);
      return;
    }

    await submitBooking();
  });
}

function renderFlightSummary(flight) {
  const el = document.getElementById("flightSummary");
  if (!el) return;

  const dep = flight.departure_time ? String(flight.departure_time).substring(0, 5) : "—";
  const arr = flight.arrival_time ? String(flight.arrival_time).substring(0, 5) : "—";
  const price = formatVND(flight.price_per_person || flight.base_price);

  el.innerHTML = `
    <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:center;">
      <div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;">${flight.origin_code}</div>
        <div style="color:var(--muted);font-size:0.8rem;">${flight.origin_city || ""}</div>
        <div style="font-size:0.85rem;margin-top:4px;color:var(--muted);">${dep}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;">Bay thẳng</div>
        <div style="color:var(--signal);font-size:1.5rem;">→</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--muted);">${flight.flight_number}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;">${flight.dest_code}</div>
        <div style="color:var(--muted);font-size:0.8rem;">${flight.dest_city || ""}</div>
        <div style="font-size:0.85rem;margin-top:4px;color:var(--muted);">${arr}</div>
      </div>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:16px;padding-top:16px;border-top:1px solid var(--border);">
      <span style="color:var(--muted);font-size:0.8rem;">Giá 1 khách</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:700;color:var(--signal);">${price}</span>
    </div>`;
}

function restoreForm() {
  const flow = getBookingFlow();
  const form = document.getElementById("bookingForm");
  if (!form) return;
  if (flow.passengerName) form.passengerName.value = flow.passengerName;
  if (flow.passengerEmail) form.passengerEmail.value = flow.passengerEmail;
  if (flow.passengerPhone) form.passengerPhone.value = flow.passengerPhone;
  if (flow.seatNumber) form.seatNumber.value = flow.seatNumber;
}

async function submitBooking() {
  const flow = getBookingFlow();
  const btn = document.getElementById("submitBookingBtn");
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Đang đặt…'; }

  try {
    const booking = await apiCreateBooking({
      flight_id: flow.flightId,
      passenger_name: flow.passengerName,
      passenger_email: flow.passengerEmail,
      passenger_phone: flow.passengerPhone,
      seat_number: flow.seatNumber,
      total_price: flow.totalPrice || flow.pricePerPerson,
    });

    saveBookingFlow({ bookingId: booking.id, bookingCode: booking.booking_code, booking });
    clearBookingFlow();

    // Go to payment
    navigate(`/payment?booking=${booking.booking_code}`);
  } catch (err) {
    if (btn) { btn.disabled = false; btn.textContent = "Tiếp tục thanh toán"; }
    const errorEl = document.getElementById("bookingError");
    if (errorEl) errorEl.innerHTML = `<div class="alert alert--error"><strong>Lỗi:</strong> ${err.message}</div>`;
  }
}

function formatVND(amount) {
  if (!amount) return "—";
  return new Intl.NumberFormat("vi-VN").format(Math.round(amount)) + " ₫";
}
