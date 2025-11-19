from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

# MongoDB connection - read from env if provided
MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)

# Use `real_estate` DB
db = client["real_estate"]
