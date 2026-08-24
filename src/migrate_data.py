from pathlib import Path
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

uri = os.getenv("MONGODB_URI")
database_name = os.getenv("MONGODB_DATABASE")

client = MongoClient(uri)
db = client[database_name]

processes_collection = db["agro_processes"]
terms_collection = db["agro_terms"]

processes_path = BASE_DIR / "dataset" / "agro_processes.csv"
terms_path = BASE_DIR / "dataset" / "agro_terms.csv"

process_df = pd.read_csv(processes_path)
terms_df = pd.read_csv(terms_path)

process_df = process_df.where(pd.notnull(process_df), None)
terms_df = terms_df.where(pd.notnull(terms_df), None)

process_records = process_df.to_dict("records")
term_records = terms_df.to_dict("records")

processes_collection.delete_many({})
terms_collection.delete_many({})

if process_records:
    processes_collection.insert_many(process_records)

if term_records:
    terms_collection.insert_many(term_records)

processes_collection.create_index("process_id", unique=True)
terms_collection.create_index("keyword", unique=True)

print(f"Inserted {len(process_records)} agricultural processes.")
print(f"Inserted {len(term_records)} agricultural terms.")

client.close()
