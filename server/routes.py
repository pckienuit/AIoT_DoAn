from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from server.database import get_connection

router = APIRouter(prefix="/api", tags=["prototype"])


class PassengerCreate(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    phone: str | None = None


class FlightCreate(BaseModel):
    flight_code: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    departure_time: str = Field(min_length=1)
    boarding_time: str | None = None
    gate: str | None = None
    status: str = "scheduled"


class BookingCreate(BaseModel):
    booking_code: str = Field(min_length=1)
    passenger_id: int
    flight_id: int
    seat_number: str | None = None
    status: str = "booked"


def row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def fetch_one_or_404(query: str, params: tuple[Any, ...], name: str) -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"{name} not found")
    return row_to_dict(row)


@router.post("/passengers", status_code=201)
def create_passenger(payload: PassengerCreate) -> dict[str, Any]:
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO passengers (name, email, phone) VALUES (?, ?, ?)",
                (payload.name, payload.email, payload.phone),
            )
            passenger_id = cursor.lastrowid
    except Exception as exc:
        raise HTTPException(status_code=409, detail="Passenger email already exists") from exc

    return get_passenger(passenger_id)


@router.get("/passengers")
def list_passengers() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM passengers ORDER BY id DESC").fetchall()
    return [row_to_dict(row) for row in rows]


@router.get("/passengers/{passenger_id}")
def get_passenger(passenger_id: int) -> dict[str, Any]:
    return fetch_one_or_404(
        "SELECT * FROM passengers WHERE id = ?",
        (passenger_id,),
        "Passenger",
    )


@router.post("/flights", status_code=201)
def create_flight(payload: FlightCreate) -> dict[str, Any]:
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO flights (flight_code, destination, departure_time, boarding_time, gate, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.flight_code,
                    payload.destination,
                    payload.departure_time,
                    payload.boarding_time,
                    payload.gate,
                    payload.status,
                ),
            )
            flight_id = cursor.lastrowid
    except Exception as exc:
        raise HTTPException(status_code=409, detail="Flight code already exists") from exc

    return get_flight(flight_id)


@router.get("/flights")
def list_flights() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM flights ORDER BY departure_time ASC").fetchall()
    return [row_to_dict(row) for row in rows]


@router.get("/flights/{flight_id}")
def get_flight(flight_id: int) -> dict[str, Any]:
    return fetch_one_or_404(
        "SELECT * FROM flights WHERE id = ?",
        (flight_id,),
        "Flight",
    )


@router.post("/bookings", status_code=201)
def create_booking(payload: BookingCreate) -> dict[str, Any]:
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO bookings (booking_code, passenger_id, flight_id, seat_number, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload.booking_code,
                    payload.passenger_id,
                    payload.flight_id,
                    payload.seat_number,
                    payload.status,
                ),
            )
            booking_id = cursor.lastrowid
    except Exception as exc:
        raise HTTPException(status_code=409, detail="Booking cannot be created") from exc

    return get_booking(booking_id)


@router.get("/bookings")
def list_bookings() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                b.*,
                p.name AS passenger_name,
                p.email AS passenger_email,
                f.flight_code,
                f.destination,
                f.gate,
                f.departure_time,
                f.boarding_time,
                f.status AS flight_status
            FROM bookings b
            JOIN passengers p ON p.id = b.passenger_id
            JOIN flights f ON f.id = b.flight_id
            ORDER BY b.id DESC
            """
        ).fetchall()
    return [row_to_dict(row) for row in rows]


@router.get("/bookings/{booking_id}")
def get_booking(booking_id: int) -> dict[str, Any]:
    return fetch_one_or_404(
        """
        SELECT
            b.*,
            p.name AS passenger_name,
            p.email AS passenger_email,
            f.flight_code,
            f.destination,
            f.gate,
            f.departure_time,
            f.boarding_time,
            f.status AS flight_status
        FROM bookings b
        JOIN passengers p ON p.id = b.passenger_id
        JOIN flights f ON f.id = b.flight_id
        WHERE b.id = ?
        """,
        (booking_id,),
        "Booking",
    )
