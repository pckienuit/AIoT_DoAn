"""Clear all flights and demo users for reseeding."""
import sys
sys.path.insert(0, '.')
from server.database import get_connection

with get_connection() as conn:
    cur = conn.execute("DELETE FROM flights")
    conn.commit()
    print(f"Cleared {cur.rowcount} flights")
    
    cur = conn.execute("DELETE FROM users WHERE email = 'demo@aiot.flight'")
    conn.commit()
    print(f"Cleared {cur.rowcount} demo users")

print("Ready to seed.")
