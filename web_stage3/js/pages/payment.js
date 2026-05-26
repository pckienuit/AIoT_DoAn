/**
 * payment.js — Payment page logic.
 * (For future router-based use; payment.html uses inline script.)
 */
import { apiGetBooking, apiInitPayment, apiPaymentCallback } from "../api.js";
import { navigate } from "../router.js";
import { getBookingFlow, saveBookingFlow } from "../storage.js";
import { formatVND } from "../api.js";

export async function initPayment(params) {
  const bookingCode = params.code
    || new URLSearchParams(location.search).get("booking")
    || getBookingFlow().bookingCode;

  if (!bookingCode) { navigate("/"); return; }

  let booking;
  try {
    booking = await apiGetBooking(bookingCode);
    saveBookingFlow({ bookingCode, booking });
  } catch {
    navigate("/");
    return;
  }

  if (booking.payment_status === "paid") {
    navigate(`/confirmation?code=${bookingCode}`);
    return;
  }

  const payBtn = document.getElementById("payNowBtn");
  const radios = document.querySelectorAll('input[name="method"]');
  let selectedMethod = null;

  radios.forEach(radio => {
    radio.addEventListener("change", () => {
      radios.forEach(r => r.closest(".payment-option").classList.remove("is-selected"));
      radio.closest(".payment-option").classList.add("is-selected");
      selectedMethod = radio.value;
      payBtn.disabled = false;
    });
  });

  if (payBtn) {
    payBtn.addEventListener("click", async () => {
      if (!selectedMethod) return;
      payBtn.disabled = true;
      payBtn.innerHTML = `<span class="spinner"></span> Đang xử lý…`;

      try {
        const result = await apiInitPayment(booking.id, selectedMethod);

        if (selectedMethod === "cash") {
          await apiPaymentCallback(result.transaction_id, "success");
          navigate(`/confirmation?code=${bookingCode}`);
        } else {
          if (result.payment_url && confirm("Demo mode — simulate successful payment?")) {
            await apiPaymentCallback(result.transaction_id, "success");
            navigate(`/confirmation?code=${bookingCode}`);
          }
        }
      } catch (err) {
        payBtn.disabled = false;
        payBtn.textContent = "Thử lại";
        const errEl = document.getElementById("paymentError");
        if (errEl) {
          errEl.className = "alert alert--error";
          errEl.textContent = err.message;
          errEl.style.display = "block";
        }
      }
    });
  }
}
