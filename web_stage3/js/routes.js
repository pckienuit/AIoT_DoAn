/**
 * routes.js — Central SPA route registry.
 *
 * Import this to register all routes with the router.
 * Call `registerRoutes()` before `router.init()`.
 */

import { addRoute } from "./router.js";

import { initHome } from "./pages/home.js";
import { initSearch } from "./pages/search.js";
import { initBooking } from "./pages/booking.js";
import { initPayment } from "./pages/payment.js";
import { initConfirmation } from "./pages/confirmation.js";
import { initMyTickets } from "./pages/my-tickets.js";
import { initCheckin } from "./pages/checkin.js";

export function registerRoutes() {
  addRoute({
    path: "/",
    title: "Trang chủ",
    render: initHome,
  });

  addRoute({
    path: "/search",
    title: "Tìm chuyến bay",
    render: initSearch,
  });

  addRoute({
    path: "/booking",
    title: "Thông tin hành khách",
    render: initBooking,
  });

  addRoute({
    path: "/payment",
    title: "Thanh toán",
    render: initPayment,
  });

  addRoute({
    path: "/confirmation",
    title: "Xác nhận đặt vé",
    render: initConfirmation,
  });

  addRoute({
    path: "/my-tickets",
    title: "Vé của tôi",
    render: initMyTickets,
    requiresAuth: true,
  });

  addRoute({
    path: "/checkin",
    title: "Check-in khuôn mặt",
    render: initCheckin,
    requiresAuth: true,
  });

  addRoute({
    path: "/register-face",
    title: "Đăng ký khuôn mặt",
    render: () => {},
  });

  addRoute({
    path: "/kiosk",
    title: "Bảng tra cứu sân bay",
    render: () => {},
  });
}
