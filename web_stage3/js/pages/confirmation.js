/**
 * confirmation.js — Post-payment confirmation page.
 */
import { apiGetBooking } from "../api.js";
import { navigate } from "../router.js";
import { clearBookingFlow } from "../storage.js";
import { formatVND } from "../api.js";

export async function initConfirmation(params) {
  const bookingCode = params.code
    || new URLSearchParams(location.search).get("code");

  if (!bookingCode) { navigate("/"); return; }

  try {
    const booking = await apiGetBooking(bookingCode);
    clearBookingFlow();
    renderConfirmation(booking);
  } catch {
    navigate("/");
  }
}

function renderConfirmation(booking) {
  const depTime = String(booking.departure_time || "").substring(0, 5);
  const arrTime = String(booking.arrival_time || "").substring(0, 5);
  const dateStr = formatDate(booking.flight_date);

  // Fill in ticket elements if they exist
  const flightNum = document.getElementById("ticketFlightNum");
  if (flightNum) flightNum.textContent = booking.flight_number || "—";

  const seat = document.getElementById("ticketSeat");
  if (seat) seat.textContent = booking.seat_number || "—";

  const passenger = document.getElementById("ticketPassenger");
  if (passenger) passenger.textContent = booking.passenger_name || booking.user_name || "—";

  const price = document.getElementById("ticketPrice");
  if (price) price.textContent = formatVND(booking.total_price);
}

function formatDate(d) {
  if (!d) return "—";
  try {
    return new Date(d).toLocaleDateString("vi-VN", { day: "2-digit", month: "short", year: "numeric" });
  } catch { return d; }
}
