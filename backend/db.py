import os
import datetime
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "deepshield"
COLLECTION_NAME = "detections"

_db_client = None

def get_db():
    global _db_client
    if _db_client is None:
        try:
            # Connect to MongoDB Atlas or local MongoDB
            _db_client = MongoClient(MONGO_URI, server_api=ServerApi('1') if "mongodb.net" in MONGO_URI else None)
            # Send a ping to confirm a successful connection
            _db_client.admin.command('ping')
            print("Successfully connected to MongoDB!")
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            return None
    
    return _db_client[DB_NAME]

def save_detection(session_id, filename, modality, verdict, confidence, details=None):
    db = get_db()
    if db is None:
        return False
        
    try:
        record = {
            "session_id": session_id,
            "filename": filename,
            "modality": modality,
            "verdict": verdict,
            "confidence": confidence,
            "details": details or {},
            "timestamp": datetime.datetime.utcnow()
        }
        db[COLLECTION_NAME].insert_one(record)
        return True
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return False

def get_recent_detections(limit=50):
    db = get_db()
    if db is None:
        return []
        
    try:
        cursor = db[COLLECTION_NAME].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"Error reading from MongoDB: {e}")
        return []
