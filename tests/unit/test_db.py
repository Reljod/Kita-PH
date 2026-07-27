"""Tests for app.db — Database accessors and the TenantCollection wrapper.

TenantCollection is the entire multi-tenant isolation boundary: if it ever
fails to inject org_id, one organization reads another's data. These tests
treat that as a security property, not a convenience.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mongomock
import pytest

from app.db import Database, TenantCollection, db

OTHER_ORG = "org_someone_else"


# --- Database ------------------------------------------------------------


class TestDatabaseConnect:
    def test_connect_uses_env_uri_and_db_name(self, monkeypatch):
        monkeypatch.setenv("MONGO_URI", "mongodb://example:27017")
        monkeypatch.setenv("MONGO_DB_NAME", "custom_db")
        with patch("app.db.MongoClient") as client_cls:
            client_cls.return_value = {"custom_db": "sentinel-db"}
            Database.connect()
        client_cls.assert_called_once_with("mongodb://example:27017")
        assert Database.db == "sentinel-db"

    def test_connect_falls_back_to_localhost_defaults(self, monkeypatch):
        monkeypatch.delenv("MONGO_URI", raising=False)
        monkeypatch.delenv("MONGO_DB_NAME", raising=False)
        with patch("app.db.MongoClient") as client_cls:
            client_cls.return_value = {"kita_db": "sentinel-db"}
            Database.connect()
        client_cls.assert_called_once_with("mongodb://localhost:27017")

    def test_close_closes_an_open_client(self):
        fake_client = MagicMock()
        with patch.object(Database, "client", fake_client):
            Database.close()
        fake_client.close.assert_called_once()

    def test_close_is_a_noop_when_never_connected(self):
        with patch.object(Database, "client", None):
            Database.close()  # must not raise

    @pytest.mark.parametrize(
        ("accessor", "expected_name"),
        [
            ("get_chats_collection", "chats"),
            ("get_rag_collection", "rag"),
            ("get_llms_collection", "llms"),
            ("get_agents_collection", "agents"),
            ("get_users_collection", "users"),
            ("get_organizations_collection", "organizations"),
            ("get_tokens_collection", "tokens"),
            ("get_tools_collection", "tools"),
            ("get_files_collection", "files"),
            ("get_file_parse_collection", "file_parse"),
            ("get_file_parsed_flattened_collection", "file_parsed_flattened"),
        ],
    )
    def test_accessor_maps_to_its_collection(self, accessor, expected_name):
        fake_db = MagicMock()
        with patch.object(Database, "db", fake_db):
            getattr(Database, accessor)()
        fake_db.__getitem__.assert_called_once_with(expected_name)

    def test_module_level_db_is_a_database_instance(self):
        assert isinstance(db, Database)


# --- TenantCollection: read isolation -------------------------------------


class TestTenantCollectionReads:
    def test_find_one_injects_org_id(self, mock_collection):
        tenant = TenantCollection(mock_collection, "org_a")
        tenant.find_one({"name": "x"})
        mock_collection.find_one.assert_called_once_with(
            {"name": "x", "org_id": "org_a"}
        )

    def test_find_one_with_no_filter_still_scopes_to_the_org(self, mock_collection):
        TenantCollection(mock_collection, "org_a").find_one()
        mock_collection.find_one.assert_called_once_with({"org_id": "org_a"})

    def test_find_injects_org_id(self, mock_collection):
        TenantCollection(mock_collection, "org_a").find({"kind": "note"})
        mock_collection.find.assert_called_once_with(
            {"kind": "note", "org_id": "org_a"}
        )

    def test_count_documents_injects_org_id(self, mock_collection):
        TenantCollection(mock_collection, "org_a").count_documents({})
        mock_collection.count_documents.assert_called_once_with({"org_id": "org_a"})

    def test_caller_cannot_read_another_org_by_forging_org_id(self, mock_collection):
        """A caller-supplied org_id must never win over the tenant's own."""
        TenantCollection(mock_collection, "org_a").find_one({"org_id": OTHER_ORG})
        mock_collection.find_one.assert_called_once_with({"org_id": "org_a"})

    def test_positional_and_keyword_args_are_forwarded(self, mock_collection):
        tenant = TenantCollection(mock_collection, "org_a")
        tenant.find({"a": 1}, {"_id": 0}, limit=5)
        mock_collection.find.assert_called_once_with(
            {"a": 1, "org_id": "org_a"}, {"_id": 0}, limit=5
        )

    def test_find_does_not_mutate_the_callers_filter(self, mock_collection):
        original = {"name": "x"}
        TenantCollection(mock_collection, "org_a").find_one(original)
        assert original == {"name": "x"}, "caller's dict was mutated"


class TestTenantCollectionRealIsolation:
    """End-to-end isolation against a real in-memory Mongo."""

    @pytest.fixture
    def seeded(self, raw_collection):
        raw_collection.insert_many(
            [
                {"name": "mine-1", "org_id": "org_a"},
                {"name": "mine-2", "org_id": "org_a"},
                {"name": "theirs", "org_id": OTHER_ORG},
            ]
        )
        return raw_collection

    def test_find_returns_only_this_orgs_documents(self, seeded):
        names = {d["name"] for d in TenantCollection(seeded, "org_a").find({})}
        assert names == {"mine-1", "mine-2"}

    def test_find_one_cannot_reach_another_orgs_document(self, seeded):
        assert TenantCollection(seeded, "org_a").find_one({"name": "theirs"}) is None

    def test_count_documents_counts_only_this_org(self, seeded):
        assert TenantCollection(seeded, "org_a").count_documents({}) == 2

    def test_delete_many_cannot_delete_another_orgs_documents(self, seeded):
        TenantCollection(seeded, "org_a").delete_many({})
        remaining = list(seeded.find({}))
        assert len(remaining) == 1 and remaining[0]["name"] == "theirs"

    def test_update_many_cannot_touch_another_orgs_documents(self, seeded):
        TenantCollection(seeded, "org_a").update_many({}, {"$set": {"touched": True}})
        theirs = seeded.find_one({"name": "theirs"})
        assert "touched" not in theirs


# --- TenantCollection: writes ---------------------------------------------


class TestTenantCollectionWrites:
    def test_insert_one_stamps_org_id(self, mock_collection):
        TenantCollection(mock_collection, "org_a").insert_one({"name": "x"})
        assert mock_collection.insert_one.call_args[0][0]["org_id"] == "org_a"

    def test_insert_one_overrides_a_forged_org_id(self, mock_collection):
        TenantCollection(mock_collection, "org_a").insert_one(
            {"name": "x", "org_id": OTHER_ORG}
        )
        assert mock_collection.insert_one.call_args[0][0]["org_id"] == "org_a"

    def test_insert_many_stamps_every_document(self, mock_collection):
        docs = [{"n": 1}, {"n": 2}, {"n": 3}]
        TenantCollection(mock_collection, "org_a").insert_many(docs)
        written = mock_collection.insert_many.call_args[0][0]
        assert all(d["org_id"] == "org_a" for d in written)

    def test_insert_many_with_empty_list_is_forwarded(self, mock_collection):
        TenantCollection(mock_collection, "org_a").insert_many([])
        mock_collection.insert_many.assert_called_once_with([])

    def test_update_one_injects_org_id_into_the_filter_only(self, mock_collection):
        TenantCollection(mock_collection, "org_a").update_one(
            {"_id": 1}, {"$set": {"n": 2}}
        )
        args = mock_collection.update_one.call_args[0]
        assert args[0] == {"_id": 1, "org_id": "org_a"}
        assert args[1] == {"$set": {"n": 2}}, (
            "the update document must not be rewritten"
        )

    def test_update_many_injects_org_id(self, mock_collection):
        TenantCollection(mock_collection, "org_a").update_many({}, {"$set": {"n": 1}})
        assert mock_collection.update_many.call_args[0][0] == {"org_id": "org_a"}

    def test_delete_one_injects_org_id(self, mock_collection):
        TenantCollection(mock_collection, "org_a").delete_one({"_id": 1})
        mock_collection.delete_one.assert_called_once_with(
            {"_id": 1, "org_id": "org_a"}
        )

    def test_delete_many_injects_org_id(self, mock_collection):
        TenantCollection(mock_collection, "org_a").delete_many({"k": "v"})
        mock_collection.delete_many.assert_called_once_with(
            {"k": "v", "org_id": "org_a"}
        )

    def test_update_one_forwards_kwargs(self, mock_collection):
        TenantCollection(mock_collection, "org_a").update_one({}, {}, upsert=True)
        assert mock_collection.update_one.call_args.kwargs == {"upsert": True}


# --- TenantCollection: aggregation ----------------------------------------


class TestTenantCollectionAggregate:
    def test_empty_pipeline_becomes_an_org_match(self, mock_collection):
        TenantCollection(mock_collection, "org_a").aggregate([])
        assert mock_collection.aggregate.call_args[0][0] == [
            {"$match": {"org_id": "org_a"}}
        ]

    def test_none_pipeline_becomes_an_org_match(self, mock_collection):
        TenantCollection(mock_collection, "org_a").aggregate(None)
        assert mock_collection.aggregate.call_args[0][0] == [
            {"$match": {"org_id": "org_a"}}
        ]

    def test_org_match_is_prepended_to_a_non_match_first_stage(self, mock_collection):
        TenantCollection(mock_collection, "org_a").aggregate([{"$sort": {"n": 1}}])
        sent = mock_collection.aggregate.call_args[0][0]
        assert sent[0] == {"$match": {"org_id": "org_a"}}
        assert sent[1] == {"$sort": {"n": 1}}

    def test_existing_first_match_is_extended_with_org_id(self, mock_collection):
        TenantCollection(mock_collection, "org_a").aggregate([{"$match": {"n": 1}}])
        sent = mock_collection.aggregate.call_args[0][0]
        assert sent == [{"$match": {"n": 1, "org_id": "org_a"}}]

    @pytest.mark.parametrize("atlas_stage", ["$vectorSearch", "$search"])
    def test_atlas_search_stays_first_with_org_match_second(
        self, mock_collection, atlas_stage
    ):
        """Atlas requires $vectorSearch/$search to be stage 0, so the tenant
        filter has to follow it rather than displace it."""
        pipeline = [{atlas_stage: {"query": "hi"}}, {"$limit": 5}]
        TenantCollection(mock_collection, "org_a").aggregate(pipeline)
        sent = mock_collection.aggregate.call_args[0][0]
        assert atlas_stage in sent[0]
        assert sent[1] == {"$match": {"org_id": "org_a"}}
        assert sent[2] == {"$limit": 5}

    def test_a_forged_org_id_in_a_first_stage_match_is_overridden(
        self, mock_collection
    ):
        TenantCollection(mock_collection, "org_a").aggregate(
            [{"$match": {"org_id": OTHER_ORG}}]
        )
        sent = mock_collection.aggregate.call_args[0][0]
        assert sent[0]["$match"]["org_id"] == "org_a"

    def test_aggregate_forwards_kwargs(self, mock_collection):
        TenantCollection(mock_collection, "org_a").aggregate([], allowDiskUse=True)
        assert mock_collection.aggregate.call_args.kwargs == {"allowDiskUse": True}

    def test_aggregate_does_not_mutate_the_callers_pipeline(self, mock_collection):
        """A module-level pipeline constant reused across calls must not
        accumulate org_id from whichever tenant used it first."""
        shared_pipeline = [{"$match": {"kind": "note"}}]
        TenantCollection(mock_collection, "org_a").aggregate(shared_pipeline)
        assert shared_pipeline == [{"$match": {"kind": "note"}}], (
            "aggregate() mutated the caller's pipeline; a shared pipeline "
            "constant would leak org_id across tenants"
        )

    def test_second_tenant_does_not_inherit_the_first_tenants_org_id(
        self, mock_collection
    ):
        """The concrete cross-tenant leak the previous test guards against."""
        shared_pipeline = [{"$match": {"kind": "note"}}]
        TenantCollection(mock_collection, "org_a").aggregate(shared_pipeline)
        TenantCollection(mock_collection, "org_b").aggregate(shared_pipeline)
        second_call = mock_collection.aggregate.call_args_list[1][0][0]
        assert second_call[0]["$match"]["org_id"] == "org_b"


# --- TenantCollection: delegation -----------------------------------------


class TestTenantCollectionDelegation:
    def test_unknown_attributes_delegate_to_the_wrapped_collection(
        self, mock_collection
    ):
        tenant = TenantCollection(mock_collection, "org_a")
        mock_collection.create_index.return_value = "idx_name"
        assert tenant.create_index("field") == "idx_name"
        mock_collection.create_index.assert_called_once_with("field")

    def test_delegated_calls_are_not_org_scoped(self, mock_collection):
        """A documented sharp edge: only the explicitly wrapped methods inject
        org_id. find_one_and_update and friends pass straight through."""
        tenant = TenantCollection(mock_collection, "org_a")
        tenant.find_one_and_update({"_id": 1}, {"$set": {"n": 1}})
        mock_collection.find_one_and_update.assert_called_once_with(
            {"_id": 1}, {"$set": {"n": 1}}
        )

    def test_org_id_is_exposed_as_an_attribute(self, mock_collection):
        assert TenantCollection(mock_collection, "org_a").org_id == "org_a"

    def test_unknown_attribute_resolves_to_a_subcollection(self, mock_collection):
        """PyMongo maps `collection.foo` to the `collection.foo` namespace
        rather than raising, and TenantCollection inherits that."""
        real = mongomock.MongoClient()["d"]["c"]
        sub = TenantCollection(real, "org_a").some_namespace
        assert sub.name == "c.some_namespace"

    def test_dunder_lookups_do_not_fall_through_to_the_collection(
        self, mock_collection
    ):
        tenant = TenantCollection(mock_collection, "org_a")
        assert tenant.__class__ is TenantCollection
