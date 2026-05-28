/**
 * checkin.js — Face check-in page logic.
 * (This module is for future router-based use; checkin.html uses inline script.)
 */
import { apiGetBooking, apiCheckin, apiFaceMatch } from "../api.js";
import { navigate } from "../router.js";
import { FaceCapture } from "../components/face-capture.js";

export async function initCheckin(params) {
  const bookingCode = params.booking || new URLSearchParams(location.search).get("booking");

  if (!bookingCode) {
    navigate("/my-tickets");
    return;
  }

  let booking;
  try {
    booking = await apiGetBooking(bookingCode);
  } catch {
    navigate("/my-tickets");
    return;
  }

  const faceCapture = new FaceCapture({
    onStatus: (msg) => {
      const el = document.getElementById("cameraStatus");
      if (el) el.textContent = msg;
    },
  });

  const video = document.getElementById("cameraVideo");
  if (!video) return;

  // Start camera
  const startBtn = document.getElementById("startCameraBtn");
  if (startBtn) {
    startBtn.addEventListener("click", async () => {
      startBtn.disabled = true;
      try {
        await faceCapture.startCamera(video);
        document.getElementById("captureBtn").disabled = false;
        document.getElementById("stepCamera").classList.add("is-done");
      } catch (err) {
        startBtn.disabled = false;
        alert("Không thể bật camera: " + err.message);
      }
    });
  }

  // Capture & match
  const captureBtn = document.getElementById("captureBtn");
  if (captureBtn) {
    captureBtn.addEventListener("click", async () => {
      captureBtn.disabled = true;
      captureBtn.textContent = "Đang xử lý…";
      try {
        const result = await faceCapture.extractEmbeddingFromSource(video);
        const match = await apiFaceMatch(booking.flight_id, result.embedding);
        if (match.matched) {
          document.getElementById("stepCapture").classList.add("is-done");
          document.getElementById("stepVerify").classList.add("is-active");
          const distance = Number(match.distance).toFixed(4);
          const threshold = Number(match.threshold ?? 0.02).toFixed(4);
          if (confirm(`Khuôn mặt đã xác thực (distance ${distance} <= ${threshold}). Xác nhận check-in?`)) {
            await apiCheckin(booking.id);
            alert("Check-in thành công!");
            navigate("/my-tickets");
          }
        } else {
          alert("Không tìm thấy khuôn mặt. Vui lòng thử lại.");
        }
      } catch (err) {
        alert("Lỗi: " + err.message);
      } finally {
        captureBtn.disabled = false;
        captureBtn.textContent = "Chụp & Xác thực";
      }
    });
  }
}
