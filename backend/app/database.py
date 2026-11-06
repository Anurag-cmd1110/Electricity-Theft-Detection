import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "electricity_theft_db",
)

if not MONGODB_URI:
    raise RuntimeError(
        "MONGODB_URI is missing. Add it to the project-root .env file."
    )

client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi("1"),
    serverSelectionTimeoutMS=10000,
)

db = client[MONGODB_DATABASE]
meter_readings = db["meter_readings"]
investigations = db["investigations"]
meters_collection = db["meters"]


def check_database_connection():
    client.admin.command("ping")
    return {
        "status": "connected",
        "database": MONGODB_DATABASE,
    }