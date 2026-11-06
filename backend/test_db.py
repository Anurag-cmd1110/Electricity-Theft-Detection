
from app.database import check_database_connection

if __name__ == "__main__":
    try:
        result = check_database_connection()
        print("MongoDB Atlas connection successful!")
        print(result)
    except Exception as exc:
        print("MongoDB connection failed.")
        print(f"{type(exc).__name__}: {exc}")
