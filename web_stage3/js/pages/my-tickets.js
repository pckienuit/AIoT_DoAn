/**
 * my-tickets.js — Booking management page logic.
 */
import { apiGetMyBookings, apiCancelBooking } from "../api.js";
import { navigate } from "../router.js";
import { createTicketCard } from "../components/ticket-card.js";

export async function initMyTickets(params) {
  let allBookings = [];

  async function load() {
    try {
      allBookings = await apiGetMyBookings();
      renderBookings(allBookings);
    } catch (err) {
      if (err.status === 401) {
        navigate("/login?return=/my-tickets");
      }
    }
  }

  function renderBookings(bookings) {
    const container = document.getElementById("ticketsContainer");
    if (!container) return;
    if (!bookings.length) {
      document.getElementById("emptyState").style.display = "block";
      return;
    }
    document.getElementById("ticketsContainer").style.display = "grid";
    container.innerHTML = bookings.map(b => createTicketCard(b, {
      onCancel: (b) => openCancelModal(b),
    })).join("");
  }

  load();
}
