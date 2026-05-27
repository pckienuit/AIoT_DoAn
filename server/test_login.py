"""Direct test of login."""
import sys
sys.path.insert(0, '.')

from server.database import get_connection, row_to_dict

with get_connection() as conn:
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        ("demo@aiot.flight",)
    ).fetchone()
    if user:
        d = row_to_dict(user)
        print("User found:", d.get("email"), d.get("password_hash")[:20], "...")
        from server.auth import verify_password
        result = verify_password("demo1234", d["password_hash"])
        print("Password verify:", result)
    else:
        print("User NOT found")
