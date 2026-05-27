"""
display.py — Result display helpers for MaixCAM LCD.

Shows flight info overlay on the camera frame after a successful match,
or an error/status banner on no-match / processing states.
"""
try:
    from maix import image as mx_image
    _HAS_MAIX = True
except ImportError:
    _HAS_MAIX = False


# =====================================================================
# COLOR CONSTANTS (safe fallback for non-maix environments)
# =====================================================================
if _HAS_MAIX:
    COLOR_GREEN  = mx_image.COLOR_GREEN
    COLOR_RED    = mx_image.COLOR_RED
    COLOR_YELLOW = mx_image.COLOR_YELLOW
    COLOR_WHITE  = mx_image.COLOR_WHITE
    COLOR_BLACK  = mx_image.COLOR_BLACK
    COLOR_CYAN   = mx_image.Color(0, 220, 220)
    COLOR_ORANGE = mx_image.Color(255, 165, 0)
else:
    COLOR_GREEN = COLOR_RED = COLOR_YELLOW = COLOR_WHITE = COLOR_BLACK = None
    COLOR_CYAN = COLOR_ORANGE = None


# =====================================================================
# OVERLAY DRAWING
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
        _draw_no_match(img, x, y, w, h)
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

    # Green bounding box on match
    img.draw_rect(x, y, w, h, color=COLOR_GREEN, thickness=3)

    # Name + distance label above box
    label = "{} ({:.3f})".format(name, dist)
    img.draw_string(x, max(0, y - 22), label, COLOR_GREEN)

    # Info panel — bottom-left of frame (draw as an outline border box to not block camera)
    panel_x = 8
    panel_y = img.height() - 145
    line_h  = 20

    lines = [
        "[{}] {}".format("C" if source == "cache" else "S", flight),
        "Dest : {}".format(dest),
        "Gate : {}".format(gate),
        "Seat : {}".format(seat),
        "Dep  : {}".format(departure[:16] if departure else ""),
        "Board: {}".format(boarding[:16] if boarding else ""),
    ]

    # Outline border box (holographic/clean design) instead of solid black block
    bar_h = line_h * len(lines) + 8
    img.draw_rect(panel_x - 4, panel_y - 4,
                  img.width() - 2 * panel_x + 8, bar_h,
                  color=COLOR_GREEN,
                  thickness=2)

    for i, line in enumerate(lines):
        img.draw_string(panel_x, panel_y + i * line_h, line, COLOR_WHITE)


def draw_no_match(img, x: int, y: int, w: int, h: int):
    """Red box + 'No booking' label."""
    _draw_no_match(img, x, y, w, h)


def draw_status(img, msg: str, color=None):
    """Top-center status banner (sync state, errors, etc.)."""
    if color is None:
        color = COLOR_YELLOW
    img.draw_string(10, 10, msg, color)


def draw_hud(img, db_count: int, cache_count: int, flight_id: int, threshold: float, flight_status: str = None):
    """Bottom HUD bar showing pipeline status."""
    status_str = " [{}]".format(flight_status) if flight_status else ""
    hud = "v9+P3 | cache:{} | flight:{}{} | th:{:.3f}".format(
        cache_count, flight_id, status_str, threshold
    )
    img.draw_string(8, img.height() - 18, hud, COLOR_WHITE)


# =====================================================================
# PRIVATE
# =====================================================================

def _draw_no_match(img, x: int, y: int, w: int, h: int):
    img.draw_rect(x, y, w, h, color=COLOR_RED, thickness=2)
    img.draw_string(x, max(0, y - 18), "No booking", COLOR_RED)
