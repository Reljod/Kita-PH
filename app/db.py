import os
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database as MongoDatabase

class Database:
    client: MongoClient = None
    db: MongoDatabase = None

    @classmethod
    def connect(cls):
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGO_DB_NAME", "kita_db")
        cls.client = MongoClient(mongo_uri)
        cls.db = cls.client[db_name]

    @classmethod
    def close(cls):
        if cls.client:
            cls.client.close()

    @classmethod
    def get_chats_collection(cls) -> Collection:
        return cls.db["chats"]

db = Database()
