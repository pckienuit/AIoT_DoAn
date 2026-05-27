import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.database import get_connection
from server.seed import run as run_seed
from server.vector_service import get_qdrant_client, ensure_face_collection, COLLECTION_NAME

def main():
    print("=== Clearing SQLite Database Tables ===")
    tables = ["bookings", "payments", "passengers", "flights", "schedules", "routes", "planes", "airports", "users"]
    
    try:
        with get_connection() as conn:
            # Disable foreign key constraints temporarily to allow clearing tables in any order
            conn.execute("PRAGMA foreign_keys = OFF")
            for table in tables:
                try:
                    conn.execute(f"DELETE FROM {table}")
                    conn.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
                    print(f"  [OK] Cleared table: {table}")
                except Exception as e:
                    print(f"  [ERR] Failed to clear {table}: {e}")
            # Re-enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            conn.commit()
        print("SQLite Database tables cleared successfully.")
    except Exception as e:
        print(f"Error connecting to SQLite database: {e}")

    print("\n=== Reinitializing & Seeding Database ===")
    try:
        run_seed()
        print("Database seeded successfully.")
    except Exception as e:
        print(f"Error seeding Database: {e}")

    print("\n=== Clearing Qdrant Vector Database ===")
    try:
        client = get_qdrant_client()
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted Qdrant collection: {COLLECTION_NAME}")
        else:
            print(f"Qdrant collection '{COLLECTION_NAME}' did not exist.")
        
        ensure_face_collection()
        print(f"Created empty Qdrant collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Error clearing Qdrant Vector DB: {e}")

    print("\n=== All Databases Cleared and Reinitialized ===")

if __name__ == "__main__":
    main()
