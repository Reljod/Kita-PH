"""Tests for app.services.organization_service.

Organizations are addressed by ObjectId *or* by an opaque string id, so the
id-resolution paths get most of the attention here — that dual lookup is where
"org not found" bugs hide.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from app.models.organization import (
    OrganizationResponse,
    OrgCreate,
    OrgIntegrationUpdate,
    OrgMemberUpdate,
    OrgRole,
    OrgUpdate,
)
from app.services.organization_service import OrganizationService

OID = ObjectId("64b7f1c2e4b0a1a2b3c4d5e6")
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)
CREATOR = "user_creator_1"


def org_doc(**overrides) -> dict:
    doc = {
        "_id": OID,
        "org_name": "Acme",
        "org_code": "acme",
        "org_members": [{"user_id": CREATOR, "role": "ADMIN"}],
        "integrations": {"facebook_page_id": None},
        "status": "completed",
        "created_at": NOW,
        "updated_at": NOW,
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def orgs() -> MagicMock:
    collection = MagicMock(name="organizations")
    collection.find_one.return_value = None
    collection.find_one_and_update.return_value = None
    collection.find.return_value = iter([])
    collection.insert_one.return_value = MagicMock(inserted_id=OID)
    collection.update_one.return_value = MagicMock(modified_count=1)
    return collection


@pytest.fixture
def service(orgs) -> OrganizationService:
    with patch("app.services.organization_service.Database") as database:
        database.get_organizations_collection.return_value = orgs
        yield OrganizationService()


# --- Create ---------------------------------------------------------------


class TestCreateOrg:
    def test_returns_the_created_org(self, service):
        result = service.create_org(
            OrgCreate(org_name="Acme", org_code="acme"), CREATOR
        )
        assert isinstance(result, OrganizationResponse)
        assert result.org_name == "Acme" and result.id == str(OID)

    def test_the_creator_becomes_an_admin(self, service):
        result = service.create_org(
            OrgCreate(org_name="Acme", org_code="acme"), CREATOR
        )
        assert result.org_members[0].user_id == CREATOR
        assert result.org_members[0].role == OrgRole.ADMIN

    def test_a_new_org_starts_initializing(self, service):
        """Org creation kicks off async provisioning; the status must not
        claim 'completed' before that work has run."""
        result = service.create_org(
            OrgCreate(org_name="Acme", org_code="acme"), CREATOR
        )
        assert result.status == "initializing"

    def test_the_document_is_persisted(self, service, orgs):
        service.create_org(OrgCreate(org_name="Acme", org_code="acme"), CREATOR)
        stored = orgs.insert_one.call_args[0][0]
        assert stored["org_name"] == "Acme" and stored["org_code"] == "acme"

    def test_timestamps_are_stamped(self, service, orgs):
        service.create_org(OrgCreate(org_name="Acme", org_code="acme"), CREATOR)
        stored = orgs.insert_one.call_args[0][0]
        assert isinstance(stored["created_at"], datetime)


class TestUpdateOrgStatus:
    def test_returns_true_when_a_document_changed(self, service, orgs):
        assert service.update_org_status(str(OID), "completed") is True

    def test_returns_false_when_nothing_changed(self, service, orgs):
        orgs.update_one.return_value = MagicMock(modified_count=0)
        assert service.update_org_status(str(OID), "completed") is False

    def test_queries_by_object_id_for_a_valid_id(self, service, orgs):
        service.update_org_status(str(OID), "completed")
        assert orgs.update_one.call_args[0][0] == {"_id": OID}

    def test_falls_back_to_a_string_id(self, service, orgs):
        """Some orgs are keyed by an opaque string rather than an ObjectId."""
        service.update_org_status("org-string-key", "completed")
        assert orgs.update_one.call_args[0][0] == {"_id": "org-string-key"}

    def test_sets_the_status_and_refreshes_updated_at(self, service, orgs):
        service.update_org_status(str(OID), "failed")
        update = orgs.update_one.call_args[0][1]["$set"]
        assert update["status"] == "failed"
        assert isinstance(update["updated_at"], datetime)


# --- Read -----------------------------------------------------------------


class TestGetOrg:
    def test_returns_the_org_found_by_object_id(self, service, orgs):
        orgs.find_one.return_value = org_doc()
        assert service.get_org(str(OID)).id == str(OID)

    def test_returns_none_for_an_empty_id(self, service, orgs):
        assert service.get_org("") is None

    def test_an_empty_id_never_reaches_the_database(self, service, orgs):
        service.get_org("")
        orgs.find_one.assert_not_called()

    def test_returns_none_when_absent(self, service, orgs):
        orgs.find_one.return_value = None
        assert service.get_org(str(OID)) is None

    def test_falls_back_to_a_string_id_lookup(self, service, orgs):
        # A non-hex id makes ObjectId() raise before any query runs, so the
        # string lookup is the only round trip.
        orgs.find_one.return_value = org_doc(_id="org-string-key")
        assert service.get_org("org-string-key").id == "org-string-key"
        orgs.find_one.assert_called_once_with({"_id": "org-string-key"})

    def test_a_valid_object_id_that_misses_retries_as_a_string(self, service, orgs):
        """A 24-hex id parses fine but may still be stored as a string."""
        orgs.find_one.side_effect = [None, org_doc(_id=str(OID))]
        assert service.get_org(str(OID)).id == str(OID)
        assert orgs.find_one.call_args_list[0][0][0] == {"_id": OID}
        assert orgs.find_one.call_args_list[1][0][0] == {"_id": str(OID)}

    def test_a_malformed_id_still_tries_the_string_lookup(self, service, orgs):
        """ObjectId() raises for a non-hex id; that must degrade to the string
        query rather than escaping as a 500."""
        orgs.find_one.return_value = None
        assert service.get_org("not-an-object-id") is None
        assert orgs.find_one.call_args[0][0] == {"_id": "not-an-object-id"}


class TestGetOrgByCode:
    def test_returns_the_org(self, service, orgs):
        orgs.find_one.return_value = org_doc()
        assert service.get_org_by_code("acme").org_code == "acme"

    def test_queries_on_org_code(self, service, orgs):
        service.get_org_by_code("acme")
        orgs.find_one.assert_called_once_with({"org_code": "acme"})

    def test_returns_none_when_absent(self, service, orgs):
        assert service.get_org_by_code("nope") is None


class TestGetOrgByIntegrationId:
    def test_builds_a_dotted_integration_query(self, service, orgs):
        service.get_org_by_integration_id("facebook_page_id", "12345")
        orgs.find_one.assert_called_once_with(
            {"integrations.facebook_page_id": "12345"}
        )

    def test_returns_the_org(self, service, orgs):
        orgs.find_one.return_value = org_doc()
        assert service.get_org_by_integration_id("facebook_page_id", "12345").id == str(
            OID
        )

    def test_returns_none_when_no_org_owns_that_integration(self, service, orgs):
        assert service.get_org_by_integration_id("facebook_page_id", "nope") is None


class TestGetOrgByIdOrCode:
    def test_prefers_an_id_match(self, service, orgs):
        orgs.find_one.return_value = org_doc()
        assert service.get_org_by_id_or_code(str(OID)).id == str(OID)

    def test_falls_through_to_the_code_lookup(self, service, orgs):
        # "acme" isn't a valid ObjectId, so get_org() spends one query on the
        # string lookup and get_org_by_code() spends the second.
        orgs.find_one.side_effect = [None, org_doc()]
        assert service.get_org_by_id_or_code("acme").org_code == "acme"

    def test_returns_none_when_neither_matches(self, service, orgs):
        orgs.find_one.return_value = None
        assert service.get_org_by_id_or_code("nope") is None


class TestGetUserOrgs:
    def test_returns_every_org_the_user_belongs_to(self, service, orgs):
        orgs.find.return_value = iter([org_doc(), org_doc(org_code="other")])
        assert len(service.get_user_orgs(CREATOR)) == 2

    def test_queries_on_membership(self, service, orgs):
        service.get_user_orgs(CREATOR)
        orgs.find.assert_called_once_with({"org_members.user_id": CREATOR})

    def test_returns_an_empty_list_for_a_user_with_no_orgs(self, service, orgs):
        orgs.find.return_value = iter([])
        assert service.get_user_orgs("nobody") == []


# --- Update ---------------------------------------------------------------


class TestUpdateOrg:
    def test_returns_the_updated_org(self, service, orgs):
        orgs.find_one_and_update.return_value = org_doc(org_name="Renamed")
        assert (
            service.update_org(str(OID), OrgUpdate(org_name="Renamed")).org_name
            == "Renamed"
        )

    def test_only_set_fields_are_written(self, service, orgs):
        orgs.find_one_and_update.return_value = org_doc()
        service.update_org(str(OID), OrgUpdate(org_name="Renamed"))
        update = orgs.find_one_and_update.call_args[0][1]["$set"]
        assert "org_code" not in update

    def test_an_empty_update_short_circuits_to_a_read(self, service, orgs):
        orgs.find_one.return_value = org_doc()
        service.update_org(str(OID), OrgUpdate())
        orgs.find_one_and_update.assert_not_called()

    def test_returns_none_when_the_org_is_absent(self, service, orgs):
        orgs.find_one_and_update.return_value = None
        assert service.update_org(str(OID), OrgUpdate(org_name="Renamed")) is None


class TestUpdateIntegrations:
    def test_writes_dotted_integration_keys(self, service, orgs):
        orgs.find_one_and_update.return_value = org_doc()
        service.update_integrations(
            str(OID), OrgIntegrationUpdate(facebook_page_id="42")
        )
        update = orgs.find_one_and_update.call_args[0][1]["$set"]
        assert update["integrations.facebook_page_id"] == "42"

    def test_does_not_clobber_sibling_integrations(self, service, orgs):
        """A dotted $set is what keeps updating one integration from wiping
        the rest of the integrations subdocument."""
        orgs.find_one_and_update.return_value = org_doc()
        service.update_integrations(
            str(OID), OrgIntegrationUpdate(facebook_page_id="42")
        )
        update = orgs.find_one_and_update.call_args[0][1]["$set"]
        assert "integrations" not in update

    def test_an_empty_update_short_circuits_to_a_read(self, service, orgs):
        orgs.find_one.return_value = org_doc()
        service.update_integrations(str(OID), OrgIntegrationUpdate())
        orgs.find_one_and_update.assert_not_called()

    def test_returns_none_when_the_org_is_absent(self, service, orgs):
        orgs.find_one_and_update.return_value = None
        assert (
            service.update_integrations(
                str(OID), OrgIntegrationUpdate(facebook_page_id="42")
            )
            is None
        )


# --- Membership -----------------------------------------------------------


class TestAddOrUpdateMember:
    def test_returns_none_when_the_org_is_absent(self, service, orgs):
        orgs.find_one.return_value = None
        assert (
            service.add_or_update_member(
                str(OID), OrgMemberUpdate(user_id="u2", role=OrgRole.MEMBER)
            )
            is None
        )

    def test_a_new_member_is_pushed(self, service, orgs):
        orgs.find_one.return_value = org_doc()
        service.add_or_update_member(
            str(OID), OrgMemberUpdate(user_id="u2", role=OrgRole.MEMBER)
        )
        update = orgs.update_one.call_args[0][1]
        assert "$push" in update
        assert update["$push"]["org_members"]["user_id"] == "u2"

    def test_an_existing_member_has_their_role_updated_in_place(self, service, orgs):
        """Pushing an existing member again would duplicate them."""
        orgs.find_one.return_value = org_doc()
        service.add_or_update_member(
            str(OID), OrgMemberUpdate(user_id=CREATOR, role=OrgRole.DEV)
        )
        query, update = orgs.update_one.call_args[0]
        assert query["org_members.user_id"] == CREATOR
        assert update["$set"]["org_members.$.role"] == OrgRole.DEV
        assert "$push" not in update

    def test_the_org_is_re_read_and_returned(self, service, orgs):
        orgs.find_one.return_value = org_doc()
        assert service.add_or_update_member(
            str(OID), OrgMemberUpdate(user_id="u2", role=OrgRole.MEMBER)
        ).id == str(OID)


class TestRevokeMember:
    def test_pulls_the_member(self, service, orgs):
        orgs.find_one_and_update.return_value = org_doc(org_members=[])
        service.revoke_member(str(OID), "u2")
        update = orgs.find_one_and_update.call_args[0][1]
        assert update["$pull"]["org_members"] == {"user_id": "u2"}

    def test_returns_the_updated_org(self, service, orgs):
        orgs.find_one_and_update.return_value = org_doc(org_members=[])
        assert service.revoke_member(str(OID), "u2").org_members == []

    def test_returns_none_when_the_org_is_absent(self, service, orgs):
        orgs.find_one_and_update.return_value = None
        assert service.revoke_member(str(OID), "u2") is None

    def test_updated_at_is_refreshed(self, service, orgs):
        orgs.find_one_and_update.return_value = org_doc()
        service.revoke_member(str(OID), "u2")
        update = orgs.find_one_and_update.call_args[0][1]
        assert isinstance(update["$set"]["updated_at"], datetime)
