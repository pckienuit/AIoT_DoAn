/**
 * ticket-card.js — Booking ticket display component.
 */

function createTicketCard(booking, { onCancel, onChangeSeat, onCheckin } = {}) {
  const code = booking.booking_code || "—";
  const status = booking.status || "confirmed";

  const statusMap = {
    pending:    `<span class="badge badge--warn">Đang chờ</span>`,
    confirmed:  `<span class="badge badge--ok">Đã xác nhận</span>`,
    checked_in: `<span class="badge badge--accent">Đã check-in</span>`,
    cancelled: `<span class="badge badge--danger">Đã hủy</span>`,
    refunded:   `<span class="badge badge--muted">Hoàn tiền</span>`,
  };

  const paymentBadge = booking.payment_status === "paid"
    ? `<span class="badge badge--ok">Đã thanh toán</span>`
    : `<span class="badge badge--warn">Chưa thanh toán</span>`;

  const depTime = formatDateTime(booking.departure_time, booking.flight_date);
  const origin = booking.origin_code || "—";
  const dest = booking.dest_code || "—";
  const flightNum = booking.flight_number || "—";
  const seat = booking.seat_number || "—";
  const passenger = booking.passenger_name || booking.user_name || "—";
  const date = formatDate(booking.flight_date);

  const canCancel = status === "confirmed" && booking.payment_status !== "paid";
  const canCheckin = status === "confirmed" && booking.payment_status === "paid";
  const canChangeSeat = status === "confirmed" && !seat;
  const canPrint = true;

  const hasFace = !!booking.face_registered || !!booking.qdrant_point_id;
  const canRegisterFace = status === "confirmed" && booking.payment_status === "paid" && !hasFace;

  const faceBadge = hasFace
    ? `<span class="badge" style="background:var(--ok-dim); color:var(--ok); border:1px solid var(--ok);">✓ Gương mặt</span>`
    : "";

  const registerFaceBtn = canRegisterFace
    ? `<a href="/register-face?booking=${code}" class="btn btn--primary btn--sm">Đăng ký mặt</a>`
    : "";
  const cancelBtn = canCancel
    ? `<button class="btn btn--danger btn--sm" data-action="cancel">Hủy vé</button>`
    : "";
  const checkinBtn = canCheckin
    ? `<a href="/checkin?booking=${code}" class="btn btn--accent btn--sm">Check-in</a>`
    : "";
  const seatBtn = canChangeSeat
    ? `<a href="/seat-map?booking=${code}" class="btn btn--ghost btn--sm">Đổi ghế</a>`
    : "";
  const printBtn = status !== "cancelled"
    ? `<button class="btn btn--ghost btn--sm" data-action="print">In vé</button>`
    : "";

  return `
    <article class="ticket-card animate-fade-up">
      <div class="ticket-card__header">
        <div>
          <div class="ticket-card__route">
            <div class="ticket-card__route-codes">
              <span>${origin}</span>
              <span class="ticket-card__route-arrow">→</span>
              <span>${dest}</span>
            </div>
            <div class="ticket-card__flight-num">${flightNum} · ${date}</div>
          </div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          ${statusMap[status] || ""}
          ${paymentBadge}
          ${faceBadge}
        </div>
      </div>

      <div class="ticket-card__body">
        <div class="ticket-card__field">
          <span class="ticket-card__field-label">Khách</span>
          <span class="ticket-card__field-value">${passenger}</span>
        </div>
        <div class="ticket-card__field">
          <span class="ticket-card__field-label">Khởi hành</span>
          <span class="ticket-card__field-value">${depTime}</span>
        </div>
        <div class="ticket-card__field">
          <span class="ticket-card__field-label">Ghế</span>
          <span class="ticket-card__field-value ticket-card__seat">${seat}</span>
        </div>
        <div class="ticket-card__actions">
          <div class="booking-code">${code}</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;margin-top:8px;">
            ${registerFaceBtn}
            ${checkinBtn}
            ${seatBtn}
            ${printBtn}
            ${cancelBtn}
          </div>
        </div>
      </div>
    </article>`;
}

function formatDateTime(timeStr, dateStr) {
  if (!timeStr) return "—";
  const t = String(timeStr).substring(0, 5);
  if (!dateStr) return t;
  return formatDate(dateStr) + " · " + t;
}

function formatDate(d) {
  if (!d) return "—";
  try {
    return new Date(d).toLocaleDateString("vi-VN", { day: "2-digit", month: "short", year: "numeric" });
  } catch { return d; }
}

export { createTicketCard };
