"""Update flight statuses based on current time.

Run this periodically (e.g., via cron or scheduler) to mark flights as:
- 'departed' if past departure time
- 'arrived' if past arrival time

Usage: python -m server.update_flight_status
"""
from datetime import datetime, timedelta

from server.database import get_connection


def update_departed_flights():
    """Mark flights as 'departed' if past departure time."""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = now.date().isoformat()

    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE flights
            SET status = 'departed'
            WHERE status IN ('scheduled', 'boarding')
              AND flight_date = ?
              AND (
                SELECT departure_time FROM schedules WHERE id = flights.schedule_id
              ) <= ?
            """,
            (today, current_time),
        )
        conn.commit()
        return cur.rowcount


def update_arrived_flights():
    """Mark flights as 'arrived' if past arrival time."""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = now.date().isoformat()

    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE flights
            SET status = 'arrived'
            WHERE status = 'departed'
              AND flight_date = ?
              AND (
                SELECT arrival_time FROM schedules WHERE id = flights.schedule_id
              ) <= ?
            """,
            (today, current_time),
        )
        conn.commit()
        return cur.rowcount


def main():
    print("\n  Updating flight statuses…\n")

    departed = update_departed_flights()
    print(f"  [OK]   {departed} flight(s) marked as departed")

    arrived = update_arrived_flights()
    print(f"  [OK]   {arrived} flight(s) marked as arrived")

    if departed == 0 and arrived == 0:
        print("  [INFO] No flights to update")

    print()


if __name__ == "__main__":
    main()
