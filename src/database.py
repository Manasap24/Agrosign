import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

uri = os.getenv("MONGODB_URI")
database_name = os.getenv("MONGODB_DATABASE")

client = MongoClient(uri)

db = client[database_name]

try:
    client.admin.command("ping")
    print("MongoDB connection successful!")
except Exception as e:
    print("MongoDB connection failed:")
    print(e)
