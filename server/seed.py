"""Seed script — populate airports, planes, routes, schedules, and flights.

Usage:
    python -m server.seed          # seed SQLite (default)
    DB_TYPE=mysql python -m server.seed   # seed MySQL

Run AFTER server is started or after init_db() has created the tables.
"""
from __future__ import annotations

import random
import secrets
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.database import get_connection, USE_MYSQL, row_to_dict, init_db


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

AIRPORTS = [
    {"code": "HAN", "name": "Nội Bài",        "city": "Hà Nội",      "country": "Vietnam"},
    {"code": "SGN", "name": "Tân Sơn Nhất",   "city": "TP.HCM",      "country": "Vietnam"},
    {"code": "DAD", "name": "Đà Nẵng",        "city": "Đà Nẵng",     "country": "Vietnam"},
    {"code": "CXR", "name": "Cam Ranh",        "city": "Nha Trang",    "country": "Vietnam"},
    {"code": "VII", "name": "Vinh",            "city": "Vinh",         "country": "Vietnam"},
]

PLANES = [
    {
        "plane_code": "VN-A320",
        "plane_type": "Airbus A320",
        "total_seats": 180,
        "seat_layout": {"rows": 30, "cols": 6, "aisle": 2, "extra_legroom_cols": [1, 6]},
    },
    {
        "plane_code": "VN-B787",
        "plane_type": "Boeing 787-9",
        "total_seats": 270,
        "seat_layout": {"rows": 36, "cols": 9, "aisle": 3, "extra_legroom_cols": [1, 9]},
    },
    {
        "plane_code": "VN-ATR7",
        "plane_type": "ATR 72-600",
        "total_seats": 72,
        "seat_layout": {"rows": 12, "cols": 6, "aisle": 2, "extra_legroom_cols": [1]},
    },
]

# Routes: (origin_code, destination_code, distance_km)
ROUTES = [
    ("HAN", "SGN", 1160),
    ("SGN", "HAN", 1160),
    ("HAN", "DAD",  760),
    ("DAD", "HAN",  760),
    ("SGN", "DAD",  960),
    ("DAD", "SGN",  960),
    ("HAN", "CXR", 1270),
    ("CXR", "HAN", 1270),
    ("SGN", "CXR",  440),
    ("CXR", "SGN",  440),
    ("HAN", "VII",  320),
    ("VII", "HAN",  320),
    ("VII", "SGN",  900),
    ("SGN", "VII",  900),
]

# Schedules: (plane_code, dep_time, arr_time, base_price_vnd)
# Departure times vary throughout the day
SCHEDULE_TEMPLATES = [
    # HAN → SGN (long haul)
    ("VN-B787", "06:00", "08:30", 1_890_000),
    ("VN-A320", "08:30", "10:50", 1_650_000),
    ("VN-A320", "11:00", "13:20", 1_720_000),
    ("VN-B787", "14:00", "16:20", 2_100_000),
    ("VN-A320", "17:00", "19:20", 1_800_000),
    ("VN-A320", "20:00", "22:20", 1_500_000),
    # SGN → HAN
    ("VN-B787", "06:30", "09:00", 1_890_000),
    ("VN-A320", "09:00", "11:20", 1_650_000),
    ("VN-A320", "12:00", "14:20", 1_720_000),
    ("VN-B787", "15:30", "17:50", 2_100_000),
    ("VN-A320", "18:30", "20:50", 1_800_000),
    # HAN ↔ DAD
    ("VN-A320", "07:00", "08:30", 1_200_000),
    ("VN-A320", "10:30", "12:00", 1_250_000),
    ("VN-A320", "15:00", "16:30", 1_350_000),
    ("VN-A320", "19:30", "21:00", 1_100_000),
    # SGN ↔ DAD
    ("VN-A320", "07:30", "09:30", 1_350_000),
    ("VN-A320", "13:00", "15:00", 1_400_000),
    ("VN-A320", "18:00", "20:00", 1_200_000),
    # HAN ↔ CXR
    ("VN-B787", "08:00", "10:00", 2_200_000),
    ("VN-B787", "16:00", "18:00", 2_400_000),
    # SGN ↔ CXR (short)
    ("VN-ATR7", "06:00", "07:30",   850_000),
    ("VN-ATR7", "10:00", "11:30",   890_000),
    ("VN-ATR7", "14:00", "15:30",   920_000),
    ("VN-ATR7", "18:00", "19:30",   850_000),
    # HAN ↔ VII (short)
    ("VN-ATR7", "06:30", "08:00",   600_000),
    ("VN-ATR7", "11:00", "12:30",   650_000),
    ("VN-ATR7", "16:30", "18:00",   620_000),
    ("VN-ATR7", "20:00", "21:30",   550_000),
    # VII ↔ SGN
    ("VN-A320", "08:30", "10:30", 1_400_000),
    ("VN-A320", "14:00", "16:00", 1_500_000),
]


def _execute(query: str, params: tuple) -> int:
    """Execute a write query and return rows affected."""
    with get_connection() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.rowcount


def _fetch_one(query: str, params: tuple) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    return row_to_dict(row)


def _fetch_all(query: str, params: tuple = ()) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    result = []
    for r in rows:
        if isinstance(r, dict):
            result.append(r)
        else:
            result.append(row_to_dict(r))
    return result
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [row_to_dict(r) for r in rows]


def _is_empty(table: str) -> bool:
    row = _fetch_one(f"SELECT COUNT(*) FROM {table}", ())
    # SQLite row_factory may return dict with numeric keys
    if row is None:
        return True
    if isinstance(row, dict):
        # SQLite dict keys are column names; handle both 'COUNT(*)' and 0
        vals = list(row.values())
        return vals[0] == 0 if vals else True
    if isinstance(row, (list, tuple)):
        return (row[0] if row else 1) == 0
    return True


def _generate_flight_number(origin: str, seq: int) -> str:
    """Generate a realistic flight number like VN1234."""
    prefix = random.choice(["VN", "VJ", "QH", "BL"])
    return f"{prefix}{seq % 9000 + 1000}"


def _generate_unique_flight_number() -> str:
    """Generate a unique prototype flight number."""
    for _ in range(200):
        prefix = random.choice(["VN", "VJ", "QH", "BL"])
        flight_number = f"{prefix}{random.randint(1000, 9999)}"
        if not _fetch_one("SELECT id FROM flights WHERE flight_number = ?", (flight_number,)):
            return flight_number
    return f"VN{secrets.randbelow(900000) + 100000}"


# ---------------------------------------------------------------------------
# Seed steps
# ---------------------------------------------------------------------------

def seed_airports() -> int:
    if not _is_empty("airports"):
        print("  [SKIP] airports — already seeded")
        return 0
    count = 0
    for ap in AIRPORTS:
        _execute(
            "INSERT INTO airports (code, name, city, country) VALUES (?, ?, ?, ?)",
            (ap["code"], ap["name"], ap["city"], ap["country"]),
        )
        count += 1
    print(f"  [OK]   airports: {count} rows")
    return count


def seed_planes() -> int:
    if not _is_empty("planes"):
        print("  [SKIP] planes — already seeded")
        return 0
    import json
    count = 0
    for p in PLANES:
        layout = json.dumps(p["seat_layout"])
        _execute(
            "INSERT INTO planes (plane_code, plane_type, total_seats, seat_layout) VALUES (?, ?, ?, ?)",
            (p["plane_code"], p["plane_type"], p["total_seats"], layout),
        )
        count += 1
    print(f"  [OK]   planes: {count} rows")
    return count


def seed_routes() -> int:
    if not _is_empty("routes"):
        print("  [SKIP] routes — already seeded")
        return 0

    def airport_id(code: str) -> int:
        row = _fetch_one("SELECT id FROM airports WHERE code = ?", (code,))
        return row["id"]

    count = 0
    for (orig, dest, dist) in ROUTES:
        o_id = airport_id(orig)
        d_id = airport_id(dest)
        _execute(
            "INSERT OR IGNORE INTO routes (origin_id, destination_id, distance_km) VALUES (?, ?, ?)",
            (o_id, d_id, dist),
        )
        count += 1
    print(f"  [OK]   routes: {count} rows")
    return count


def seed_schedules() -> int:
    if not _is_empty("schedules"):
        print("  [SKIP] schedules — already seeded")
        return 0

    def route_id(origin: str, dest: str) -> int | None:
        o_row = _fetch_one("SELECT id FROM airports WHERE code = ?", (origin,))
        d_row = _fetch_one("SELECT id FROM airports WHERE code = ?", (dest,))
        if not o_row or not d_row:
            return None
        r = _fetch_one(
            "SELECT id FROM routes WHERE origin_id = ? AND destination_id = ?",
            (o_row["id"], d_row["id"]),
        )
        return r["id"] if r else None

    def plane_id(code: str) -> int | None:
        r = _fetch_one("SELECT id FROM planes WHERE plane_code = ?", (code,))
        return r["id"] if r else None

    count = 0
    for (orig, dest, dist) in ROUTES:
        r_id = route_id(orig, dest)
        if r_id is None:
            continue
        # Find matching schedules for this route direction
        for (pcode, dep, arr, price) in SCHEDULE_TEMPLATES:
            p_id = plane_id(pcode)
            if p_id is None:
                continue
            # Rough check: plane type matches route distance
            _execute(
                """
                INSERT OR IGNORE INTO schedules
                (route_id, plane_id, departure_time, arrival_time, days_of_week, base_price)
                VALUES (?, ?, ?, ?, '1,2,3,4,5,6,7', ?)
                """,
                (r_id, p_id, dep, arr, price),
            )
            count += 1
    print(f"  [OK]   schedules: {count} rows")
    return count


def seed_flights(days_ahead: int = 14, start_date: date | None = None) -> int:
    """Generate missing flights for the next N days for each active schedule."""
    today = start_date or date.today()
    count = 0

    schedules = _fetch_all(
        """
        SELECT s.*, p.total_seats
        FROM schedules s
        JOIN planes p ON p.id = s.plane_id
        """
    )
    for sched in schedules:
        s_id = sched["id"]
        total_seats = int(sched.get("total_seats") or 180)

        for day_offset in range(days_ahead):
            flight_date = today + timedelta(days=day_offset)
            departure_time = str(sched.get("departure_time") or "")
            if flight_date == date.today() and departure_time <= datetime.now().strftime("%H:%M:%S"):
                continue

            existing = _fetch_one(
                "SELECT id FROM flights WHERE schedule_id = ? AND flight_date = ?",
                (s_id, str(flight_date)),
            )
            if existing:
                continue

            flight_num = _generate_unique_flight_number()

            multiplier = round(random.uniform(0.9, 1.25), 2)
            booked_preview = random.randint(0, min(35, max(0, total_seats - 1)))
            avail_seats = max(1, total_seats - booked_preview)

            _execute(
                """
                INSERT INTO flights
                (schedule_id, flight_date, flight_number, status, available_seats, price_multiplier)
                VALUES (?, ?, ?, 'scheduled', ?, ?)
                """,
                (s_id, str(flight_date), flight_num, avail_seats, multiplier),
            )
            count += 1

    print(f"  [OK]   flights: {count} rows ({days_ahead} days × {len(schedules)} schedules)")
    return count


def cleanup_old_flights(now: datetime | None = None) -> int:
    """Delete past prototype flights that have no bookings attached."""
    now = now or datetime.now()
    today = now.date().isoformat()
    current_time = now.strftime("%H:%M:%S")
    rows = _fetch_all(
        """
        SELECT fl.id
        FROM flights fl
        JOIN schedules s ON s.id = fl.schedule_id
        LEFT JOIN bookings b ON b.flight_id = fl.id
        WHERE b.id IS NULL
          AND (
            fl.flight_date < ?
            OR (fl.flight_date = ? AND s.departure_time <= ?)
          )
        """,
        (today, today, current_time),
    )
    ids = [int(row["id"]) for row in rows]
    if not ids:
        print("  [SKIP] old flights - none to delete")
        return 0

    deleted = 0
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        placeholders = ",".join("?" for _ in chunk)
        _execute(f"DELETE FROM seat_holds WHERE flight_id IN ({placeholders})", tuple(chunk))
        deleted += _execute(f"DELETE FROM flights WHERE id IN ({placeholders})", tuple(chunk))

    print(f"  [OK]   old flights deleted: {deleted} rows")
    return deleted


def maintain_prototype_flights(days_ahead: int = 14) -> dict[str, int]:
    """Keep SQLite prototype data fresh on server startup."""
    seed_airports()
    seed_planes()
    seed_routes()
    seed_schedules()
    deleted = cleanup_old_flights()
    created = seed_flights(days_ahead=days_ahead)
    return {"deleted": deleted, "created": created}


# ---------------------------------------------------------------------------
# Demo user
# ---------------------------------------------------------------------------

def seed_demo_user() -> None:
    """Create a demo account for testing (no external deps)."""
    import hashlib, base64
    existing = _fetch_one("SELECT id FROM users WHERE email = ?", ("demo@aiot.flight",))
    if existing:
        print("  [SKIP] demo user — already exists")
        return

    # Simple bcrypt hash of "demo1234" (pre-computed)
    # $2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.8rDGPW1xdWlZS2
    bcrypt_hash = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.8rDGPW1xdWlZS2"

    _execute(
        """
        INSERT INTO users
        (email, password_hash, full_name, phone, date_of_birth, id_card)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "demo@aiot.flight",
            bcrypt_hash,
            "Nguyen Van Demo",
            "0909123456",
            "1990-01-15",
            "012345678",
        ),
    )
    print("  [OK]   demo user: demo@aiot.flight / demo1234")


# ---------------------------------------------------------------------------
# Stats report
# ---------------------------------------------------------------------------

def print_stats() -> None:
    tables = ["airports", "planes", "routes", "schedules", "flights", "users"]
    print("\n  Database stats:")
    for t in tables:
        row = _fetch_one(f"SELECT COUNT(*) FROM {t}", ())
        if isinstance(row, dict):
            n = list(row.values())[0] if row else 0
        elif isinstance(row, (list, tuple)):
            n = row[0] if row else 0
        else:
            n = 0
        print(f"    {t:<12} {n:>5} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    db_type = "MySQL" if USE_MYSQL else "SQLite"
    print(f"\n  Seeding database ({db_type})…\n")

    # Ensure tables exist first
    init_db()

    seed_airports()
    seed_planes()
    seed_routes()
    seed_schedules()
    seed_flights()
    seed_demo_user()

    print_stats()
    print(f"\n  Done. {db_type} database seeded successfully.\n")


if __name__ == "__main__":
    run()
