"""
display.py — Result display helpers for MaixCAM LCD.

Shows flight info overlay on the camera frame after a successful match,
or an error/status banner on no-match / processing states.
"""
import math
import time

try:
    from maix import image as mx_image
    _HAS_MAIX = True
except ImportError:
    _HAS_MAIX = False


# =====================================================================
# COLOR CONSTANTS (synchronized with web design system)
# =====================================================================
if _HAS_MAIX:
    COLOR_PRIMARY = mx_image.Color(255, 107, 26)  # Neon Orange (--signal)
    COLOR_ACCENT  = mx_image.Color(214, 255, 63)  # Neon Lime (--accent)
    COLOR_OK      = mx_image.Color(86, 240, 138)  # Neon Green (--ok)
    COLOR_DANGER  = mx_image.Color(255, 79, 79)  # Neon Red (--danger)
    COLOR_WARN    = mx_image.Color(245, 197, 24)  # Yellow (--warn)
    COLOR_TEXT    = mx_image.Color(245, 241, 223)  # Cream Text (--ink)
    COLOR_MUTED   = mx_image.Color(169, 162, 139)  # Gray Text (--muted)
    COLOR_BLACK   = mx_image.Color(17, 19, 13)     # Dark Gray Background (--panel)
    COLOR_BORDER  = mx_image.Color(57, 53, 34)     # Dark Gray Border (--border)
    COLOR_WHITE   = mx_image.COLOR_WHITE
else:
    COLOR_PRIMARY = COLOR_ACCENT = COLOR_OK = COLOR_DANGER = COLOR_WARN = None
    COLOR_TEXT = COLOR_MUTED = COLOR_BLACK = COLOR_BORDER = COLOR_WHITE = None


# =====================================================================
# VIETNAMESE ACCENT STRIPPER (fixes font drawing errors on MaixCAM)
# =====================================================================
_VIETNAMESE_MAP = {
    ord('à'): 'a', ord('á'): 'a', ord('ả'): 'a', ord('ã'): 'a', ord('ạ'): 'a',
    ord('ă'): 'a', ord('ằ'): 'a', ord('ắ'): 'a', ord('ẳ'): 'a', ord('ẵ'): 'a', ord('ặ'): 'a',
    ord('â'): 'a', ord('ầ'): 'a', ord('ấ'): 'a', ord('ẩ'): 'a', ord('ẫ'): 'a', ord('ậ'): 'a',
    ord('À'): 'A', ord('Á'): 'A', ord('Ả'): 'A', ord('Ã'): 'A', ord('Ạ'): 'A',
    ord('Ă'): 'A', ord('Ằ'): 'A', ord('Ắ'): 'A', ord('Ẳ'): 'A', ord('Ẵ'): 'A', ord('Ặ'): 'A',
    ord('Â'): 'A', ord('Ầ'): 'A', ord('Ấ'): 'A', ord('Ẩ'): 'A', ord('Ẫ'): 'A', ord('Ậ'): 'A',
    ord('è'): 'e', ord('é'): 'e', ord('ẻ'): 'e', ord('ẽ'): 'e', ord('ẹ'): 'e',
    ord('ê'): 'e', ord('ề'): 'e', ord('ế'): 'e', ord('ể'): 'e', ord('ễ'): 'e', ord('ệ'): 'e',
    ord('È'): 'E', ord('É'): 'E', ord('Ẻ'): 'E', ord('Ẽ'): 'E', ord('Ẹ'): 'E',
    ord('Ê'): 'E', ord('Ề'): 'E', ord('Ế'): 'E', ord('Ể'): 'E', ord('Ễ'): 'E', ord('Ệ'): 'E',
    ord('ì'): 'i', ord('í'): 'i', ord('ỉ'): 'i', ord('ĩ'): 'i', ord('ị'): 'i',
    ord('Ì'): 'I', ord('Í'): 'I', ord('Ỉ'): 'I', ord('Ĩ'): 'I', ord('Ị'): 'I',
    ord('ò'): 'o', ord('ó'): 'o', ord('ỏ'): 'o', ord('õ'): 'o', ord('ọ'): 'o',
    ord('ô'): 'o', ord('ồ'): 'o', ord('ố'): 'o', ord('ổ'): 'o', ord('ỗ'): 'o', ord('ộ'): 'o',
    ord('ơ'): 'o', ord('ờ'): 'o', ord('ớ'): 'o', ord('ở'): 'o', ord('ỡ'): 'o', ord('ợ'): 'o',
    ord('Ò'): 'O', ord('Ó'): 'O', ord('Ỏ'): 'O', ord('Õ'): 'O', ord('Ọ'): 'O',
    ord('Ô'): 'O', ord('Ồ'): 'O', ord('Ố'): 'O', ord('Ổ'): 'O', ord('Ỗ'): 'O', ord('Ộ'): 'O',
    ord('Ơ'): 'O', ord('Ờ'): 'O', ord('Ớ'): 'O', ord('Ở'): 'O', ord('Ỡ'): 'O', ord('Ợ'): 'O',
    ord('ù'): 'u', ord('ú'): 'u', ord('ủ'): 'u', ord('ũ'): 'u', ord('ụ'): 'u',
    ord('ư'): 'u', ord('ừ'): 'u', ord('ứ'): 'u', ord('ử'): 'u', ord('ữ'): 'u', ord('ự'): 'u',
    ord('Ù'): 'U', ord('Ú'): 'U', ord('Ủ'): 'U', ord('Ũ'): 'U', ord('Ụ'): 'U',
    ord('Ư'): 'U', ord('Ừ'): 'U', ord('Ứ'): 'U', ord('Ử'): 'U', ord('Ữ'): 'U', ord('Ự'): 'U',
    ord('ỳ'): 'y', ord('ý'): 'y', ord('ỷ'): 'y', ord('ỹ'): 'y', ord('ỵ'): 'y',
    ord('Ỳ'): 'Y', ord('Ý'): 'Y', ord('Ỷ'): 'Y', ord('Ỹ'): 'Y', ord('Ỵ'): 'Y',
    ord('đ'): 'd', ord('Đ'): 'D'
}

def strip_accents(text: str) -> str:
    """Replace Vietnamese accented characters with their ASCII equivalents."""
    if not text:
        return ""
    return str(text).translate(_VIETNAMESE_MAP)

def draw_string_ascii(img, x: int, y: int, text: str, color):
    """Draw text on the image after removing all Vietnamese accents."""
    img.draw_string(x, y, strip_accents(text), color)


# =====================================================================
# OVERLAY DRAWING HELPERS
# =====================================================================

def draw_target_brackets(img, x: int, y: int, w: int, h: int, color, bracket_len=18, thickness=3):
    """Draw holographic corner brackets around the face target box."""
    # Top-left corner
    img.draw_rect(x, y, bracket_len, thickness, color=color, thickness=-1)
    img.draw_rect(x, y, thickness, bracket_len, color=color, thickness=-1)
    
    # Top-right corner
    img.draw_rect(x + w - bracket_len, y, bracket_len, thickness, color=color, thickness=-1)
    img.draw_rect(x + w - thickness, y, thickness, bracket_len, color=color, thickness=-1)
    
    # Bottom-left corner
    img.draw_rect(x, y + h - thickness, bracket_len, thickness, color=color, thickness=-1)
    img.draw_rect(x, y + h - bracket_len, thickness, bracket_len, color=color, thickness=-1)
    
    # Bottom-right corner
    img.draw_rect(x + w - bracket_len, y + h - thickness, bracket_len, thickness, color=color, thickness=-1)
    img.draw_rect(x + w - thickness, y + h - bracket_len, thickness, bracket_len, color=color, thickness=-1)


# =====================================================================
# PRIMARY EXPORTED DRAW OVERLAYS
# =====================================================================

def draw_match_result(img, result: dict | None, x: int, y: int, w: int, h: int):
    """
    Draw flight info overlay on the frame after identity match.

    Args:
        img     : maix image frame
        result  : dict from CacheManager.match() or None
        x,y,w,h : bounding box of detected face
    """
    if result is None:
        draw_no_match(img, x, y, w, h)
        return

    payload = result.get("payload", {})
    source  = result.get("source", "?")
    dist    = result.get("distance", 0.0)

    name      = payload.get("passenger_name", "Unknown")
    gate      = payload.get("gate", "?")
    seat      = payload.get("seat_number", "?")
    flight    = payload.get("flight_number") or payload.get("flight_code") or "?"
    departure = payload.get("departure_time", "")
    boarding  = payload.get("boarding_time", "")
    dest      = payload.get("dest_city") or payload.get("destination") or ""

    # Neon Green brackets for successful match
    draw_target_brackets(img, x, y, w, h, COLOR_OK, thickness=3)

    # Name + distance tag above box
    label = "{} ({:.3f})".format(name, dist)
    draw_string_ascii(img, x, max(0, y - 22), label, COLOR_OK)

    # Virtual Boarding Pass Info Panel (bottom-left area of frame)
    panel_x = 12
    panel_y = img.height() - 150
    panel_w = img.width() - 2 * panel_x
    panel_h = 124

    # Dark background card with standard dark border (matches web style)
    img.draw_rect(panel_x, panel_y, panel_w, panel_h, color=COLOR_BLACK, thickness=-1)
    img.draw_rect(panel_x, panel_y, panel_w, panel_h, color=COLOR_BORDER, thickness=2)

    # Header title bar - Orange background (matches web Kiosk style)
    img.draw_rect(panel_x, panel_y, panel_w, 22, color=COLOR_PRIMARY, thickness=-1)
    draw_string_ascii(img, panel_x + 8, panel_y + 3, "BOARDING PASS / CHECK-IN SUCCESS", COLOR_BLACK)

    # Content columns
    col1_x = panel_x + 12
    col2_x = panel_x + 320
    start_y = panel_y + 32
    line_h = 20

    # Column 1: Passenger profile
    draw_string_ascii(img, col1_x, start_y, "PASSENGER: {}".format(name.upper()), COLOR_TEXT)
    draw_string_ascii(img, col1_x, start_y + line_h, "FLIGHT   : {}".format(flight), COLOR_PRIMARY)
    draw_string_ascii(img, col1_x, start_y + line_h * 2, "ROUTE    : {}".format(dest), COLOR_TEXT)
    draw_string_ascii(img, col1_x, start_y + line_h * 3, "SOURCE   : {}".format("LOCAL CACHE" if source == "cache" else "SERVER DB"), COLOR_MUTED)

    # Column 2: Flight details
    draw_string_ascii(img, col2_x, start_y, "GATE : {}".format(gate), COLOR_PRIMARY)
    draw_string_ascii(img, col2_x, start_y + line_h, "SEAT : {}".format(seat), COLOR_ACCENT)
    draw_string_ascii(img, col2_x, start_y + line_h * 2, "BOARD: {}".format(boarding[:16] if boarding else "N/A"), COLOR_TEXT)
    draw_string_ascii(img, col2_x, start_y + line_h * 3, "DEP  : {}".format(departure[:16] if departure else "N/A"), COLOR_TEXT)


def draw_no_match(img, x: int, y: int, w: int, h: int):
    """Red brackets + system warning block."""
    # Red brackets
    draw_target_brackets(img, x, y, w, h, COLOR_DANGER, thickness=3)
    draw_string_ascii(img, x, max(0, y - 22), "NO BOOKING", COLOR_DANGER)

    # Warning Card
    panel_x = 12
    panel_y = img.height() - 110
    panel_w = img.width() - 2 * panel_x
    panel_h = 80

    # Dark background card with standard dark border
    img.draw_rect(panel_x, panel_y, panel_w, panel_h, color=COLOR_BLACK, thickness=-1)
    img.draw_rect(panel_x, panel_y, panel_w, panel_h, color=COLOR_BORDER, thickness=2)

    # Header title bar - Red background
    img.draw_rect(panel_x, panel_y, panel_w, 20, color=COLOR_DANGER, thickness=-1)
    draw_string_ascii(img, panel_x + 8, panel_y + 2, "SYSTEM ALERT: IDENTITY UNKNOWN", COLOR_BLACK)

    # Warnings
    draw_string_ascii(img, panel_x + 12, panel_y + 28, "No boarding registration found for this face.", COLOR_TEXT)
    draw_string_ascii(img, panel_x + 12, panel_y + 50, "Please verify your ticket or register face at counter.", COLOR_MUTED)


def draw_status(img, msg: str, color=None):
    """Top-left status pill text."""
    if color is None:
        color = COLOR_WARN
    draw_string_ascii(img, 12, 12, msg, color)


def draw_hud(img, db_count: int, cache_count: int, flight_id: int, threshold: float, flight_status: str = None):
    """Bottom HUD bar showing system status."""
    hud_y = img.height() - 20
    # draw a dark background bar
    img.draw_rect(0, hud_y, img.width(), 20, color=COLOR_BLACK, thickness=-1)

    status_str = flight_status.upper() if flight_status else "ACTIVE"
    hud_text = " [SYS: OK] | CACHE: {} PAX | FLIGHT: {} [{}] | THRES: {:.3f}".format(
        cache_count, flight_id, status_str, threshold
    )
    draw_string_ascii(img, 8, hud_y + 3, hud_text, COLOR_MUTED)
