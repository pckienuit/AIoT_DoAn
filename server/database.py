"""Database layer supporting SQLite (prototype) and MySQL (production)."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

try:
    import mysql.connector
    from mysql.connector import pooling
    _HAS_MYSQL = True
except ImportError:
    _HAS_MYSQL = False

try:
    import sqlite3
    _HAS_SQLITE = True
except ImportError:
    _HAS_SQLITE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "prototype.db"

# Determine DB mode from env
USE_MYSQL = os.getenv("DB_TYPE", "sqlite").lower() == "mysql"
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "aiot_flight"),
}

_connection_pool: pooling.MySQLConnectionPool | None = None


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _sqlite_row_factory(cursor, row):
    """Return rows as dicts, converting JSON/seat-layout columns."""
    col_names = [desc[0] for desc in cursor.description]
    raw = dict(zip(col_names, row, strict=False))
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            # JSON columns used in SQLite prototype
            if k == "seat_layout":
                try:
                    out[k] = json.loads(v)
                    continue
                except Exception:
                    out[k] = v
                    continue
        # Convert datetime strings back to proper types
        if k in ("created_at", "updated_at", "face_registered_at") and isinstance(v, str):
            if v:
                try:
                    out[k] = datetime.fromisoformat(v.replace("Z", "+00:00"))
                    continue
                except Exception:
                    pass
        out[k] = v
    return out


@contextmanager
def _get_sqlite_conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = _sqlite_row_factory
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _execute_sqlite(conn, query: str, params: tuple[Any, ...] = ()):
    return conn.execute(query, params)


# ---------------------------------------------------------------------------
# MySQL helpers
# ---------------------------------------------------------------------------

def _get_mysql_pool() -> pooling.MySQLConnectionPool:
    global _connection_pool
    if _connection_pool is None:
        if not _HAS_MYSQL:
            raise RuntimeError("MySQL connector not installed")
        _connection_pool = pooling.MySQLConnectionPool(
            pool_name="aiot_pool",
            pool_size=5,
            pool_reset_session=True,
            **MYSQL_CONFIG,
        )
    return _connection_pool


@contextmanager
def _get_mysql_conn():
    pool = _get_mysql_pool()
    conn = pool.get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _execute_mysql(conn, query: str, params: tuple[Any, ...] = ()):
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, params)
    return cursor


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

def get_connection():
    """Return a context-manager that yields a DB connection (SQLite or MySQL)."""
    if USE_MYSQL:
        return _get_mysql_conn()
    return _get_sqlite_conn()


def row_to_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "_row"):
        return dict(row._row)
    if hasattr(row, "keys"):
        return dict(row)
    return dict(row)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SQLITE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    phone TEXT,
    date_of_birth TEXT,
    id_card TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS airports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL DEFAULT 'Vietnam'
);

CREATE TABLE IF NOT EXISTS planes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plane_code TEXT NOT NULL UNIQUE,
    plane_type TEXT NOT NULL,
    total_seats INTEGER NOT NULL,
    seat_layout TEXT NOT NULL DEFAULT '{"rows":30,"cols":6}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    origin_id INTEGER NOT NULL,
    destination_id INTEGER NOT NULL,
    distance_km INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (origin_id) REFERENCES airports(id),
    FOREIGN KEY (destination_id) REFERENCES airports(id),
    UNIQUE(origin_id, destination_id)
);

CREATE TABLE IF NOT EXISTS schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    route_id INTEGER NOT NULL,
    plane_id INTEGER NOT NULL,
    departure_time TEXT NOT NULL,
    arrival_time TEXT NOT NULL,
    days_of_week TEXT NOT NULL DEFAULT '1,2,3,4,5,6,7',
    base_price REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (route_id) REFERENCES routes(id),
    FOREIGN KEY (plane_id) REFERENCES planes(id)
);

CREATE TABLE IF NOT EXISTS flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schedule_id INTEGER NOT NULL,
    flight_date TEXT NOT NULL,
    flight_number TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'scheduled',
    available_seats INTEGER NOT NULL,
    price_multiplier REAL DEFAULT 1.00,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (schedule_id) REFERENCES schedules(id)
);

CREATE TABLE IF NOT EXISTS passengers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_code TEXT NOT NULL UNIQUE,
    user_id INTEGER,
    flight_id INTEGER NOT NULL,
    passenger_id INTEGER,
    seat_number TEXT,
    passenger_name TEXT NOT NULL,
    passenger_email TEXT,
    passenger_phone TEXT,
    qdrant_point_id TEXT UNIQUE,
    face_registered INTEGER NOT NULL DEFAULT 0,
    face_registered_at TEXT,
    status TEXT NOT NULL DEFAULT 'confirmed',
    total_price REAL,
    payment_status TEXT NOT NULL DEFAULT 'unpaid',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (passenger_id) REFERENCES passengers(id),
    FOREIGN KEY (flight_id) REFERENCES flights(id)
);

CREATE TABLE IF NOT EXISTS seat_holds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hold_token TEXT NOT NULL,
    flight_id INTEGER NOT NULL,
    seat_number TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (flight_id) REFERENCES flights(id)
);

CREATE TABLE IF NOT EXISTS payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    method TEXT NOT NULL,
    transaction_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (booking_id) REFERENCES bookings(id)
);

CREATE INDEX IF NOT EXISTS idx_bookings_user_id ON bookings(user_id);
CREATE INDEX IF NOT EXISTS idx_bookings_flight_id ON bookings(flight_id);
CREATE INDEX IF NOT EXISTS idx_bookings_status ON bookings(status);
CREATE INDEX IF NOT EXISTS idx_bookings_code ON bookings(booking_code);
CREATE INDEX IF NOT EXISTS idx_seat_holds_token ON seat_holds(hold_token);
CREATE INDEX IF NOT EXISTS idx_seat_holds_flight_expires ON seat_holds(flight_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_flights_date ON flights(flight_date);
CREATE INDEX IF NOT EXISTS idx_payments_booking_id ON payments(booking_id);
"""

MYSQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    date_of_birth DATE,
    id_card VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS airports (
    id INT PRIMARY KEY AUTO_INCREMENT,
    code VARCHAR(3) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    country VARCHAR(100) NOT NULL DEFAULT 'Vietnam'
);

CREATE TABLE IF NOT EXISTS planes (
    id INT PRIMARY KEY AUTO_INCREMENT,
    plane_code VARCHAR(20) NOT NULL UNIQUE,
    plane_type VARCHAR(50) NOT NULL,
    total_seats INT NOT NULL,
    seat_layout JSON NOT NULL DEFAULT '{"rows":30,"cols":6}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS routes (
    id INT PRIMARY KEY AUTO_INCREMENT,
    origin_id INT NOT NULL,
    destination_id INT NOT NULL,
    distance_km INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_route (origin_id, destination_id),
    FOREIGN KEY (origin_id) REFERENCES airports(id),
    FOREIGN KEY (destination_id) REFERENCES airports(id)
);

CREATE TABLE IF NOT EXISTS schedules (
    id INT PRIMARY KEY AUTO_INCREMENT,
    route_id INT NOT NULL,
    plane_id INT NOT NULL,
    departure_time TIME NOT NULL,
    arrival_time TIME NOT NULL,
    days_of_week VARCHAR(20) NOT NULL DEFAULT '1,2,3,4,5,6,7',
    base_price DECIMAL(10,0) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (route_id) REFERENCES routes(id),
    FOREIGN KEY (plane_id) REFERENCES planes(id)
);

CREATE TABLE IF NOT EXISTS flights (
    id INT PRIMARY KEY AUTO_INCREMENT,
    schedule_id INT NOT NULL,
    flight_date DATE NOT NULL,
    flight_number VARCHAR(10) NOT NULL UNIQUE,
    status ENUM('scheduled','boarding','departed','arrived','cancelled') DEFAULT 'scheduled',
    available_seats INT NOT NULL,
    price_multiplier DECIMAL(3,2) DEFAULT 1.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (schedule_id) REFERENCES schedules(id),
    INDEX idx_flight_date (flight_date)
);

CREATE TABLE IF NOT EXISTS passengers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bookings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    booking_code VARCHAR(20) NOT NULL UNIQUE,
    user_id INT,
    flight_id INT NOT NULL,
    passenger_id INT,
    seat_number VARCHAR(4),
    passenger_name VARCHAR(255) NOT NULL,
    passenger_email VARCHAR(255),
    passenger_phone VARCHAR(20),
    qdrant_point_id VARCHAR(36) UNIQUE,
    face_registered BOOLEAN DEFAULT FALSE,
    face_registered_at TIMESTAMP NULL,
    status ENUM('pending','confirmed','checked_in','cancelled','refunded') DEFAULT 'confirmed',
    total_price DECIMAL(12,0),
    payment_status ENUM('unpaid','paid','refunded') DEFAULT 'unpaid',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (passenger_id) REFERENCES passengers(id),
    FOREIGN KEY (flight_id) REFERENCES flights(id),
    INDEX idx_booking_user (user_id),
    INDEX idx_booking_flight (flight_id),
    INDEX idx_booking_status (status),
    INDEX idx_booking_code (booking_code)
);

CREATE TABLE IF NOT EXISTS seat_holds (
    id INT PRIMARY KEY AUTO_INCREMENT,
    hold_token VARCHAR(64) NOT NULL,
    flight_id INT NOT NULL,
    seat_number VARCHAR(255) NOT NULL,
    expires_at VARCHAR(32) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (flight_id) REFERENCES flights(id),
    INDEX idx_seat_holds_token (hold_token),
    INDEX idx_seat_holds_flight_expires (flight_id, expires_at)
);

CREATE TABLE IF NOT EXISTS payments (
    id INT PRIMARY KEY AUTO_INCREMENT,
    booking_id INT NOT NULL,
    amount DECIMAL(12,0) NOT NULL,
    method ENUM('vnpay','momo','stripe','cash') NOT NULL,
    transaction_id VARCHAR(100),
    status ENUM('pending','success','failed') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (booking_id) REFERENCES bookings(id),
    INDEX idx_payment_booking (booking_id)
);
"""


def _ensure_sqlite_booking_guards(conn) -> None:
    try:
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_bookings_active_flight_seat_unique
            ON bookings(flight_id, seat_number)
            WHERE seat_number IS NOT NULL
              AND trim(seat_number) != ''
              AND status NOT IN ('cancelled', 'refunded')
            """
        )
    except Exception as exc:
        print(
            "  [WARN] bookings active seat uniqueness not enabled; "
            f"clean duplicate active seats first ({exc})"
        )


def init_db() -> None:
    """Run schema migrations."""
    if USE_MYSQL:
        # Create database if not exists (connect without db name first)
        try:
            import mysql.connector
            tmp = mysql.connector.connect(
                host=MYSQL_CONFIG["host"],
                port=MYSQL_CONFIG["port"],
                user=MYSQL_CONFIG["user"],
                password=MYSQL_CONFIG["password"],
            )
            cur = tmp.cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
            cur.close()
            tmp.close()
        except Exception:
            pass  # DB might already exist or MySQL not available

        with _get_mysql_conn() as conn:
            for stmt in MYSQL_SCHEMA.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.cursor().execute(stmt)
            conn.commit()
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with _get_sqlite_conn() as conn:
            conn.executescript(SQLITE_SCHEMA)
            _ensure_sqlite_booking_guards(conn)


def iter_tables() -> Iterator[str]:
    if USE_MYSQL:
        with _get_mysql_conn() as conn:
            cur = conn.cursor()
            cur.execute("SHOW TABLES")
            for (name,) in cur:
                yield name
            cur.close()
    else:
        with _get_sqlite_conn() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        for row in rows:
            yield row["name"]
