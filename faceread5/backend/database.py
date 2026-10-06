import os

from pymongo import MongoClient



MONGO_HOST = os.getenv("MONGO_HOST_NAME")
MONGO_PORT = int(os.getenv("MONGO_PORT"))
MONGO_DATABASE = os.getenv("MONGO_DB")
MONGO_USERNAME = os.getenv("MONGO_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_ROOT_PASSWORD")

client = MongoClient(
    f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"
)

db = client[MONGO_DATABASE]

collection = db["analyses"]