"""Tests for app.models.agent — agent id parsing and response formatting.

Agents are versioned and addressed as either `<base_id>` (meaning "latest")
or `<base_id>-v<n>` (a pinned version), so the id grammar is load-bearing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from bson import ObjectId

from app.models.agent import (
    AgentCreateRequest,
    AgentResponse,
    AgentUpdateRequest,
    format_agent_response,
    parse_agent_id,
)

OID = ObjectId("64b7f1c2e4b0a1a2b3c4d5e6")
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)


def agent_doc(**overrides) -> dict:
    doc = {
        "_id": OID,
        "base_id": str(OID),
        "version": 1,
        "name": "Researcher",
        "role": "analyst",
        "goal": "find things",
        "backstory": "trained on the archive",
        "personalities": ["curious"],
        "llm_id": "llm_1",
        "tools": ["web_search"],
        "created_at": NOW,
        "updated_at": NOW,
    }
    doc.update(overrides)
    return doc


class TestParseAgentId:
    def test_a_bare_id_has_no_version(self):
        assert parse_agent_id("abc123") == ("abc123", None)

    def test_a_versioned_id_splits_into_base_and_version(self):
        assert parse_agent_id("abc123-v2") == ("abc123", 2)

    def test_a_multi_digit_version_is_parsed(self):
        assert parse_agent_id("abc123-v42") == ("abc123", 42)

    def test_only_the_last_v_marker_is_treated_as_the_version(self):
        """Base ids can themselves contain '-v', so the split must be from
        the right."""
        assert parse_agent_id("my-v1-agent-v3") == ("my-v1-agent", 3)

    @pytest.mark.parametrize(
        "agent_id",
        ["abc-vX", "abc-v", "abc-v2x", "abc-v-1", "abc-v1.5"],
    )
    def test_a_non_numeric_version_is_not_a_version(self, agent_id):
        base, version = parse_agent_id(agent_id)
        assert version is None and base == agent_id

    def test_an_empty_string_round_trips(self):
        assert parse_agent_id("") == ("", None)

    def test_version_zero_is_parsed(self):
        assert parse_agent_id("abc-v0") == ("abc", 0)


class TestFormatAgentResponse:
    def test_maps_the_document_fields(self):
        result = format_agent_response(agent_doc())
        assert isinstance(result, AgentResponse)
        assert result.name == "Researcher" and result.llm_id == "llm_1"

    def test_base_id_falls_back_to_the_document_id(self):
        doc = agent_doc(base_id=None)
        assert format_agent_response(doc).base_id == str(OID)

    def test_version_defaults_to_one(self):
        doc = agent_doc()
        del doc["version"]
        assert format_agent_response(doc).version == 1

    def test_the_system_prompt_is_attached_when_supplied(self):
        assert (
            format_agent_response(agent_doc(), system_prompt="you are…").system_prompt
            == "you are…"
        )

    def test_the_system_prompt_is_omitted_by_default(self):
        """List views skip prompt construction for performance."""
        assert format_agent_response(agent_doc()).system_prompt is None

    def test_tools_default_to_empty(self):
        doc = agent_doc()
        del doc["tools"]
        assert format_agent_response(doc).tools == []

    def test_personalities_may_be_absent(self):
        doc = agent_doc()
        del doc["personalities"]
        assert format_agent_response(doc).personalities is None

    def test_missing_timestamps_are_defaulted_rather_than_raising(self):
        doc = agent_doc()
        del doc["created_at"]
        del doc["updated_at"]
        assert isinstance(format_agent_response(doc).created_at, datetime)

    def test_a_missing_required_field_raises(self):
        doc = agent_doc()
        del doc["role"]
        with pytest.raises(KeyError):
            format_agent_response(doc)

    def test_id_is_the_base_id_for_version_one(self):
        assert format_agent_response(agent_doc(version=1)).id == str(OID)

    def test_id_stays_unversioned_for_later_versions(self):
        """Current behaviour: `id` is always the bare base_id, so a v3 agent
        and its v1 predecessor report the same id and are told apart only by
        the separate `version` field. parse_agent_id() understands the
        `<base>-v<n>` form on the way in, so this is an in/out asymmetry
        rather than a crash — pinning the response id would change every
        client URL, so it is deliberately left alone here."""
        result = format_agent_response(agent_doc(version=3))
        assert result.id == str(OID)
        assert result.version == 3

    def test_the_round_trip_from_id_back_through_the_parser_is_stable(self):
        result = format_agent_response(agent_doc(version=3))
        assert parse_agent_id(result.id) == (str(OID), None)


class TestAgentRequestValidation:
    def test_a_valid_create_request_is_accepted(self):
        req = AgentCreateRequest(
            name="A", role="r", goal="g", backstory="b", llm_id="llm_1"
        )
        assert req.tools == []

    @pytest.mark.parametrize("field", ["name", "role", "goal", "backstory", "llm_id"])
    def test_required_fields_reject_the_empty_string(self, field):
        payload = {
            "name": "A",
            "role": "r",
            "goal": "g",
            "backstory": "b",
            "llm_id": "llm_1",
        }
        payload[field] = ""
        with pytest.raises(ValueError):
            AgentCreateRequest(**payload)

    def test_an_over_long_name_is_rejected(self):
        with pytest.raises(ValueError):
            AgentCreateRequest(
                name="x" * 101, role="r", goal="g", backstory="b", llm_id="l"
            )

    def test_too_many_tools_are_rejected(self):
        with pytest.raises(ValueError):
            AgentCreateRequest(
                name="A",
                role="r",
                goal="g",
                backstory="b",
                llm_id="l",
                tools=[f"t{i}" for i in range(51)],
            )

    def test_an_update_request_may_be_entirely_empty(self):
        """PATCH semantics: omitting every field is a no-op, not an error."""
        assert AgentUpdateRequest().model_dump(exclude_unset=True) == {}

    def test_an_update_request_tracks_which_fields_were_set(self):
        req = AgentUpdateRequest(name="Renamed")
        assert req.model_dump(exclude_unset=True) == {"name": "Renamed"}

    def test_an_update_request_rejects_an_empty_name(self):
        with pytest.raises(ValueError):
            AgentUpdateRequest(name="")
