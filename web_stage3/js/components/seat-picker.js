/**
 * seat-picker.js — Interactive SVG-based seat map engine.
 *
 * Usage:
 *   const picker = createSeatPicker({ container, seats, onSelect });
 *   picker.render();
 *   picker.setSelected("12A");
 */

const SEAT_COLORS = {
  available: "#d6ff3f",
  selected:  "#ff6b1a",
  booked:    "#3a3a30",
  extra:     "#4a6fa5",
};

function createSeatPicker({ container, seats = [], onSelect, selectedSeats = [] } = {}) {
  // seats: array of { seat, row, col, status, extra_legroom, surcharge }
  const _selected = new Set(selectedSeats);
  let _seats = seats;
  let _container = null;

  function getLayout(seats) {
    if (!seats.length) return { rows: 30, cols: 6, aisleAfter: 3 };
    const maxRow = Math.max(...seats.map(s => s.row));
    const maxCol = Math.max(...seats.map(s => s.col));
    const aisleAfter = maxCol === 6 ? 3 : Math.ceil(maxCol / 2);
    return { rows: maxRow, cols: maxCol, aisleAfter };
  }

  function getSeatStatus(seat) {
    if (_selected.has(seat.seat)) return "selected";
    return seat.status || "available";
  }

  function getSeatColor(seat) {
    return SEAT_COLORS[getSeatStatus(seat)] || SEAT_COLORS.available;
  }

  function render() {
    if (!_container) return;

    const { rows, cols, aisleAfter } = getLayout(_seats);
    const colLetters = Array.from({ length: cols }, (_, i) =>
      `<span>${String.fromCharCode(65 + i)}</span>`
    ).join("");

    // Build rows
    const rowsData = [];
    for (let r = 1; r <= rows; r++) {
      const rowSeats = _seats.filter(s => s.row === r);
      rowsData.push({ rowNum: r, seats: rowSeats, aisleAfter });
    }

    const seatEls = rowsData.map(({ rowNum, seats, aisleAfter: aa }) => {
      const cells = [];
      for (let c = 1; c <= cols; c++) {
        const seat = seats.find(s => s.col === c);
        const isAisle = c === aa + 1 && aa > 0;
        if (isAisle) {
          cells.push(`<div class="seat-map__aisle"></div>`);
        }

        if (seat) {
          const status = getSeatStatus(seat);
          const color = getSeatColor(seat);
          const extra = seat.extra_legroom ? " seat--extra" : "";
          const booked = status === "booked" ? " seat--booked" : "";
          const selected = status === "selected" ? " seat--selected" : "";
          const tooltip = seat.surcharge > 0 ? ` (+${formatVND(seat.surcharge)})` : "";
          cells.push(`
            <div class="seat seat--${status}${extra}${booked}${selected}"
                 data-seat="${seat.seat}"
                 title="${seat.seat}${tooltip}"
                 role="button"
                 tabindex="${status !== "booked" ? 0 : -1}"
                 aria-label="Ghế ${seat.seat}${tooltip}, ${status}">
              <span class="seat__number">${seat.seat}</span>
              ${seat.surcharge > 0 ? `<span class="seat__price">+</span>` : ""}
            </div>`);
        } else {
          cells.push(`<div></div>`);
        }
      }

      return `
        <div class="seat-map__row">
          <div class="seat-map__row-label">${rowNum}</div>
          <div class="seat-map__row-seats">${cells.join("")}</div>
        </div>`;
    }).join("");

    _container.innerHTML = `
      <div class="seat-map-cabin">
        <div class="seat-map__legend">
          <div class="seat-map__legend-item">
            <div class="seat-map__legend-dot seat-map__legend-dot--available"></div> Trống
          </div>
          <div class="seat-map__legend-item">
            <div class="seat-map__legend-dot seat-map__legend-dot--selected"></div> Đã chọn
          </div>
          <div class="seat-map__legend-item">
            <div class="seat-map__legend-dot seat-map__legend-dot--booked"></div> Đã đặt
          </div>
          <div class="seat-map__legend-item">
            <div class="seat-map__legend-dot seat-map__legend-dot--extra"></div> Extra legroom (+)
          </div>
        </div>

        <div class="seat-map__plane-nose">Cửa ra máy bay ↑</div>

        <div class="seat-map__header">
          <span></span>
          ${colLetters.split("").map(c => `<span>${c}</span>`).join("")}
        </div>

        ${seatEls}

        <div class="seat-map__plane-nose" style="margin-top: 16px;">← Màn hình ghế trước →</div>
      </div>`;

    // Attach click events
    _container.querySelectorAll(".seat:not(.seat--booked)").forEach(el => {
      el.addEventListener("click", () => {
        const seat = el.dataset.seat;
        toggleSeat(seat);
      });
      el.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggleSeat(el.dataset.seat);
        }
      });
    });
  }

  function toggleSeat(seatLabel) {
    if (_selected.has(seatLabel)) {
      _selected.delete(seatLabel);
    } else {
      _selected.add(seatLabel);
    }
    render();
    if (onSelect) onSelect(Array.from(_selected));
  }

  function setSelected(seats) {
    _selected.clear();
    seats.forEach(s => _selected.add(s));
    render();
  }

  function getSelected() {
    return Array.from(_selected);
  }

  function mount(el) {
    _container = el;
    render();
  }

  return { mount, render, setSelected, getSelected, toggleSeat };
}

function formatVND(amount) {
  return new Intl.NumberFormat("vi-VN").format(Math.round(amount)) + " ₫";
}

export { createSeatPicker, formatVND };
