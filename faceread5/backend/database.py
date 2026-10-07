import os

from pymongo import MongoClient

from dotenv import load_dotenv


load_dotenv()



MONGO_HOST = faceread-mongodb
MONGO_PORT = int(os.getenv("MONGO_PORT"))
MONGO_DATABASE = os.getenv("MONGO_DATABASE")
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")

client = MongoClient(
    f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"
)

db = client[MONGO_DATABASE]

collection = db["analyses"]  