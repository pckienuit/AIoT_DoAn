"""Update demo user password hash using bcrypt directly."""
import sys, bcrypt
sys.path.insert(0, '.')
from server.database import get_connection

# Generate proper bcrypt hash
pw = "demo1234".encode("utf-8")
salt = bcrypt.gensalt(rounds=12)
new_hash = bcrypt.hashpw(pw, salt).decode("utf-8")
print(f"New hash: {new_hash}")

# Verify it works
assert bcrypt.checkpw(pw, new_hash.encode("utf-8"))
print("Hash verified OK")

with get_connection() as conn:
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE email = ?",
        (new_hash, "demo@aiot.flight")
    )
    conn.commit()
    print("Updated demo@aiot.flight in database")
