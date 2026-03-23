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

    @classmethod
    def get_rag_collection(cls) -> Collection:
        return cls.db["rag"]

    @classmethod
    def get_llms_collection(cls) -> Collection:
        return cls.db["llms"]

    @classmethod
    def get_agents_collection(cls) -> Collection:
        return cls.db["agents"]

    @classmethod
    def get_users_collection(cls) -> Collection:
        return cls.db["users"]

    @classmethod
    def get_organizations_collection(cls) -> Collection:
        return cls.db["organizations"]

    @classmethod
    def get_tokens_collection(cls) -> Collection:
        return cls.db["tokens"]

    @classmethod
    def get_tools_collection(cls) -> Collection:
        return cls.db["tools"]

db = Database()

from typing import Mapping, Any, Optional

class TenantCollection:
    def __init__(self, collection: Collection, org_id: str):
        """
        Wraps a PyMongo Collection to auto-inject org_id in queries and insertions.
        """
        self._collection = collection
        self.org_id = org_id

    def _inject(self, filter_query: Optional[Mapping[str, Any]]) -> dict:
        """Helper to inject the org_id into the query filter."""
        query = dict(filter_query) if filter_query else {}
        query["org_id"] = self.org_id
        return query

    def find(self, filter=None, *args, **kwargs):
        return self._collection.find(self._inject(filter), *args, **kwargs)

    def find_one(self, filter=None, *args, **kwargs):
        return self._collection.find_one(self._inject(filter), *args, **kwargs)

    def insert_one(self, document, *args, **kwargs):
        document["org_id"] = self.org_id
        return self._collection.insert_one(document, *args, **kwargs)

    def insert_many(self, documents, *args, **kwargs):
        for doc in documents:
            doc["org_id"] = self.org_id
        return self._collection.insert_many(documents, *args, **kwargs)

    def update_one(self, filter, update, **kwargs):
        return self._collection.update_one(self._inject(filter), update, **kwargs)

    def update_many(self, filter, update, **kwargs):
        return self._collection.update_many(self._inject(filter), update, **kwargs)

    def delete_one(self, filter, **kwargs):
        return self._collection.delete_one(self._inject(filter), **kwargs)

    def delete_many(self, filter, **kwargs):
        return self._collection.delete_many(self._inject(filter), **kwargs)

    def count_documents(self, filter, **kwargs):
        return self._collection.count_documents(self._inject(filter), **kwargs)

    def aggregate(self, pipeline, *args, **kwargs):
        """Inject an org_id match into the aggregation pipeline."""
        pipeline = list(pipeline) if pipeline else []
        org_match = {"$match": {"org_id": self.org_id}}

        if not pipeline:
            pipeline.append(org_match)
        elif "$vectorSearch" in pipeline[0] or "$search" in pipeline[0]:
            # MongoDB Atlas special stages must be first. Inject org_id filter as second stage.
            pipeline.insert(1, org_match)
        elif "$match" in pipeline[0]:
            pipeline[0]["$match"]["org_id"] = self.org_id
        else:
            pipeline.insert(0, org_match)
            
        return self._collection.aggregate(pipeline, *args, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes."""
        return getattr(self._collection, name)
