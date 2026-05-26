/**
 * home.js — Home page logic.
 */
import { apiGetAirports, apiSearchFlights } from "../api.js";
import { navigate } from "../router.js";
import { saveBookingFlow } from "../storage.js";

let airports = [];

export async function initHome(params) {
  render(await loadAirports());
  attachEvents();
}

async function loadAirports() {
  try {
    airports = await apiGetAirports();
  } catch {
    // Fallback: Vietnam airports
    airports = [
      { code: "HAN", name: "Nội Bài", city: "Hà Nội" },
      { code: "SGN", name: "Tân Sơn Nhất", city: "TP. Hồ Chí Minh" },
      { code: "DAD", name: "Đà Nẵng", city: "Đà Nẵng" },
      { code: "CXR", name: "Cam Ranh", city: "Nha Trang" },
      { code: "VHHH", name: "Tân Sơn Nhất HK", city: "Hồng Kông" },
    ];
  }
  return airports;
}

function render(airportList) {
  const selectHtml = `<option value="">— Chọn sân bay —</option>` +
    airportList.map(a => `<option value="${a.code}">${a.city} (${a.code})</option>`).join("");

  document.getElementById("originSelect").innerHTML = selectHtml;
  document.getElementById("destSelect").innerHTML = selectHtml;
}

function attachEvents() {
  // Swap button
  document.getElementById("swapBtn")?.addEventListener("click", () => {
    const origin = document.getElementById("originSelect");
    const dest = document.getElementById("destSelect");
    const tmp = origin.value;
    origin.value = dest.value;
    dest.value = tmp;
  });

  // Search form
  document.getElementById("searchForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const origin = document.getElementById("originSelect")?.value;
    const dest = document.getElementById("destSelect")?.value;
    const date = document.getElementById("dateInput")?.value;
    const passengers = parseInt(document.getElementById("passengerCount")?.textContent || "1");

    if (!date) {
      alert("Vui lòng chọn ngày khởi hành.");
      return;
    }

    saveBookingFlow({ origin, destination: dest, date, passengers });

    navigate(`/search?origin=${encodeURIComponent(origin || "")}&destination=${encodeURIComponent(dest || "")}&date=${date}&passengers=${passengers}`);
  });

  // Passenger counter
  const countEl = document.getElementById("passengerCount");
  document.getElementById("paxMinus")?.addEventListener("click", () => {
    const n = Math.max(1, parseInt(countEl.textContent) - 1);
    countEl.textContent = n;
  });
  document.getElementById("paxPlus")?.addEventListener("click", () => {
    const n = Math.min(9, parseInt(countEl.textContent) + 1);
    countEl.textContent = n;
  });

  // Trip type tabs
  document.querySelectorAll(".search-card__tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".search-card__tab").forEach(t => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      const isRound = tab.dataset.trip === "roundtrip";
      document.getElementById("returnDateRow").classList.toggle("is-hidden", !isRound);
    });
  });

  // Quick route chips
  document.querySelectorAll(".route-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const origin = chip.dataset.origin;
      const dest = chip.dataset.dest;
      if (origin) document.getElementById("originSelect").value = origin;
      if (dest) document.getElementById("destSelect").value = dest;
    });
  });

  // Set default date to tomorrow
  const dateInput = document.getElementById("dateInput");
  if (dateInput && !dateInput.value) {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    dateInput.value = tomorrow.toISOString().split("T")[0];
  }
}
