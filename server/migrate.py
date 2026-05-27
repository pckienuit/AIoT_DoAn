"""Migrate prototype.db from old schema to new schema.

This script:
1. Backs up the old DB to prototype.db.bak
2. Alters bookings table (add missing columns, drop passenger_id)
3. Recreates the flights table to match new schedule-based schema

Usage: python -m server.migrate
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import date, timedelta

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "prototype.db"
BACKUP_PATH = DATA_DIR / "prototype.db.bak"


def _conn():
    return sqlite3.connect(DB_PATH)


def backup():
    if DB_PATH.exists():
        shutil.copy2(DB_PATH, BACKUP_PATH)
        print("  [OK]   Backup created -> prototype.db.bak")


def migrate_bookings():
    conn = _conn()
    cur = conn.execute("PRAGMA table_info(bookings)")
    cols = {r[1] for r in cur.fetchall()}

    needed = {
        "user_id", "passenger_name", "passenger_email", "passenger_phone",
        "qdrant_point_id", "face_registered", "face_registered_at",
        "total_price", "payment_status",
    }
    missing = needed - cols

    if not missing:
        print("  [SKIP] bookings — already migrated")
        conn.close()
        return

    # Add missing columns first
    for col in missing:
        if col == "user_id":
            conn.execute("ALTER TABLE bookings ADD COLUMN user_id INTEGER")
        elif col == "passenger_name":
            conn.execute("ALTER TABLE bookings ADD COLUMN passenger_name TEXT NOT NULL DEFAULT ''")
        elif col == "passenger_email":
            conn.execute("ALTER TABLE bookings ADD COLUMN passenger_email TEXT")
        elif col == "passenger_phone":
            conn.execute("ALTER TABLE bookings ADD COLUMN passenger_phone TEXT")
        elif col == "face_registered":
            conn.execute("ALTER TABLE bookings ADD COLUMN face_registered INTEGER NOT NULL DEFAULT 0")
        elif col == "face_registered_at":
            conn.execute("ALTER TABLE bookings ADD COLUMN face_registered_at TEXT")
        elif col == "total_price":
            conn.execute("ALTER TABLE bookings ADD COLUMN total_price REAL DEFAULT 0")
        elif col == "payment_status":
            conn.execute("ALTER TABLE bookings ADD COLUMN payment_status TEXT NOT NULL DEFAULT 'unpaid'")
    conn.commit()

    # Map old passenger_id → passenger data
    passenger_map = {}
    rows = conn.execute("SELECT id, name, email, phone FROM passengers").fetchall()
    for r in rows:
        passenger_map[r[0]] = {"name": r[1], "email": r[2], "phone": r[3]}

    # Migrate each booking
    bookings = conn.execute("SELECT * FROM bookings").fetchall()
    col_names = [r[1] for r in conn.execute("PRAGMA table_info(bookings)").fetchall()]

    for b in bookings:
        row = dict(zip(col_names, b))
        pid = row.get("passenger_id")
        pax = passenger_map.get(pid, {})

        conn.execute("""
            UPDATE bookings SET
                passenger_name  = ?,
                passenger_email = ?,
                passenger_phone = ?,
                face_registered = 0,
                total_price     = 0,
                payment_status  = CASE WHEN status = 'confirmed' THEN 'unpaid' ELSE status END
            WHERE id = ?
        """, (
            pax.get("name", ""),
            pax.get("email", ""),
            pax.get("phone", ""),
            row["id"],
        ))

    # Drop old column (only if SQLite version supports it; SQLite 3.35+)
    try:
        conn.execute("DROP COLUMN passenger_id")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Older SQLite — column stays (harmless)
    conn.close()
    print(f"  [OK]   bookings — migrated {len(bookings)} rows")


def migrate_flights():
    """Recreate flights table to new schedule-based schema."""
    conn = _conn()
    cur = conn.execute("PRAGMA table_info(flights)")
    cols = {r[1] for r in cur.fetchall()}

    if "schedule_id" in cols:
        print("  [SKIP] flights — already migrated")
        conn.close()
        return

    # Read old flights data
    old_flights = conn.execute("SELECT * FROM flights").fetchall()
    old_cols = [r[1] for r in conn.execute("PRAGMA table_info(flights)").fetchall()]
    old_data = [dict(zip(old_cols, f)) for f in old_flights]

    # Drop old table and recreate with new schema
    conn.execute("DROP TABLE flights")

    conn.execute("""
        CREATE TABLE flights (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_id     INTEGER NOT NULL,
            flight_date     TEXT NOT NULL,
            flight_number   TEXT NOT NULL UNIQUE,
            status          TEXT NOT NULL DEFAULT 'scheduled',
            available_seats INTEGER NOT NULL,
            price_multiplier REAL DEFAULT 1.00,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()

    # Import old flight_code as flight_number
    today = date.today()
    import random
    for i, f in enumerate(old_data):
        code = f.get("flight_code", f"OLD{i+1:04d}")
        # Parse destination to guess price (rough)
        dest = f.get("destination", "SGN")
        # Map destination to base price
        prices = {"SGN": 1650000, "HAN": 1650000, "DAD": 1250000, "CXR": 2200000}
        price_mult = random.uniform(0.85, 1.3)
        conn.execute("""
            INSERT INTO flights (schedule_id, flight_date, flight_number, status, available_seats, price_multiplier)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            1,  # dummy schedule_id — needs manual fix
            str(today + timedelta(days=random.randint(0, 14))),
            code,
            f.get("status", "scheduled"),
            random.randint(50, 180),
            round(price_mult, 2),
        ))
    conn.commit()
    conn.close()

    print(f"  [WARN] flights — recreated with {len(old_data)} rows (schedule_id=1 is placeholder — run seed.py to regenerate)")


def migrate_airports():
    """Add missing country column if needed."""
    conn = _conn()
    cur = conn.execute("PRAGMA table_info(airports)")
    cols = {r[1] for r in cur.fetchall()}
    if "country" not in cols:
        conn.execute("ALTER TABLE airports ADD COLUMN country TEXT NOT NULL DEFAULT 'Vietnam'")
        conn.commit()
        print("  [OK]   airports — added country column")
    else:
        print("  [SKIP] airports — already has country column")
    conn.close()


def migrate_users():
    """Ensure users table has id_card column."""
    conn = _conn()
    cur = conn.execute("PRAGMA table_info(users)")
    cols = {r[1] for r in cur.fetchall()}
    if "id_card" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN id_card TEXT")
        conn.commit()
        print("  [OK]   users — added id_card column")
    else:
        print("  [SKIP] users — already has id_card column")
    conn.close()


def migrate_flights_deleted_at():
    """Add deleted_at column to flights table for soft delete support."""
    conn = _conn()
    cur = conn.execute("PRAGMA table_info(flights)")
    cols = {r[1] for r in cur.fetchall()}
    if "deleted_at" not in cols:
        conn.execute("ALTER TABLE flights ADD COLUMN deleted_at TEXT")
        conn.commit()
        print("  [OK]   flights — added deleted_at column for soft delete")
    else:
        print("  [SKIP] flights — already has deleted_at column")
    conn.close()


def main():
    print("\n  Migrating prototype.db …\n")
    backup()
    migrate_bookings()
    migrate_flights()
    migrate_airports()
    migrate_users()
    migrate_flights_deleted_at()
    print("\n  Done. Run 'python -m server.seed' to populate data.\n")


if __name__ == "__main__":
    main()
