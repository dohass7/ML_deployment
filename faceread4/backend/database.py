import os

from pymongo import MongoClient

MONGO_HOST = os.getenv(
    "MONGO_HOST",
    "172.17.0.1"
    #"host.docker.internal"
)

MONGO_PORT = int(
    os.getenv(
        "MONGO_PORT",
        "27017"
    )
)

MONGO_DB = os.getenv(
    "MONGO_DB",
    "faceread"
)

client = MongoClient(
    f"mongodb://{MONGO_HOST}:{MONGO_PORT}"
)

db = client[MONGO_DB]

analyses = db["analyses"]