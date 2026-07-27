"""Tests for the app.exceptions hierarchy.

Every exception here becomes an HTTP response shape, so the invariants that
matter are: a stable machine-readable `code`, a sensible `status_code`, and a
`to_dict()` payload that never loses the details a client needs.
"""

from __future__ import annotations

import inspect
import pkgutil

import pytest

import app.exceptions as exceptions_pkg
from app.exceptions import (
    AgentNotFoundError,
    AgentRunFailedError,
    AgentRunStreamFailedError,
    AgentVersionNotFoundError,
    AuthSessionExpiredError,
    ChatNotFoundError,
    ForbiddenError,
    InvalidApiKeyOrClientError,
    KitaDatabaseError,
    KitaException,
    KitaRedisError,
    KitaValidationError,
    SystemConfigurationError,
    ToolWebSearchError,
    UnauthorizedError,
)


def all_exception_classes() -> list[type[KitaException]]:
    """Every KitaException subclass exported by the package."""
    found: dict[str, type[KitaException]] = {}
    for module in pkgutil.iter_modules(exceptions_pkg.__path__):
        mod = __import__(f"app.exceptions.{module.name}", fromlist=["*"])
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, KitaException) and obj is not KitaException:
                found[obj.__name__] = obj
    return sorted(found.values(), key=lambda c: c.__name__)


class TestKitaExceptionBase:
    def test_message_is_preserved(self):
        assert KitaException("something broke").message == "something broke"

    def test_str_renders_the_message(self):
        assert str(KitaException("something broke")) == "something broke"

    def test_it_is_a_real_exception(self):
        with pytest.raises(KitaException):
            raise KitaException("boom")

    def test_details_default_to_an_empty_dict(self):
        assert KitaException("x").details == {}

    def test_details_are_kept(self):
        assert KitaException("x", details={"k": "v"}).details == {"k": "v"}

    def test_the_default_status_is_500(self):
        assert KitaException("x").status_code == 500

    def test_the_status_code_can_be_overridden_per_instance(self):
        assert KitaException("x", status_code=418).status_code == 418

    def test_overriding_the_status_does_not_leak_to_the_class(self):
        """A per-instance override that mutated the class would silently
        change the status of every later exception of that type."""
        KitaException("x", status_code=418)
        assert KitaException("y").status_code == 500

    def test_to_dict_has_the_wire_shape(self):
        payload = KitaException("broke", details={"k": "v"}).to_dict()
        assert payload == {
            "error": {
                "code": "SYSTEM_INTERNAL_ERROR",
                "message": "broke",
                "details": {"k": "v"},
            }
        }

    def test_to_dict_includes_details_even_when_empty(self):
        assert KitaException("broke").to_dict()["error"]["details"] == {}


class TestExceptionRegistry:
    @pytest.mark.parametrize("cls", all_exception_classes(), ids=lambda c: c.__name__)
    def test_every_exception_declares_a_code(self, cls):
        assert isinstance(cls.code, str) and cls.code

    @pytest.mark.parametrize("cls", all_exception_classes(), ids=lambda c: c.__name__)
    def test_every_exception_declares_a_plausible_status(self, cls):
        assert 400 <= cls.status_code <= 599

    @pytest.mark.parametrize("cls", all_exception_classes(), ids=lambda c: c.__name__)
    def test_codes_are_screaming_snake_case(self, cls):
        assert cls.code == cls.code.upper()
        assert " " not in cls.code

    def test_codes_are_unique_across_concrete_exceptions(self):
        """Two different failures sharing a code makes them indistinguishable
        to a client that branches on it."""
        by_code: dict[str, list[str]] = {}
        for cls in all_exception_classes():
            # The parent category classes deliberately share codes with the
            # concrete error they stand in for; only compare leaves.
            if cls.__subclasses__():
                continue
            by_code.setdefault(cls.code, []).append(cls.__name__)
        collisions = {code: names for code, names in by_code.items() if len(names) > 1}
        assert not collisions, f"duplicate error codes: {collisions}"


class TestAgentExceptions:
    def test_not_found_builds_a_message_from_the_id(self):
        assert "agent_1" in AgentNotFoundError("agent_1").message

    def test_not_found_is_a_404(self):
        assert AgentNotFoundError("agent_1").status_code == 404

    def test_not_found_carries_the_agent_id_in_details(self):
        assert AgentNotFoundError("agent_1").details["agent_id"] == "agent_1"

    def test_not_found_accepts_a_custom_message(self):
        assert AgentNotFoundError("agent_1", message="gone").message == "gone"

    def test_not_found_merges_extra_details(self):
        err = AgentNotFoundError("agent_1", details={"org_id": "org_1"})
        assert err.details == {"agent_id": "agent_1", "org_id": "org_1"}

    def test_caller_details_can_override_the_derived_ones(self):
        err = AgentNotFoundError("agent_1", details={"agent_id": "override"})
        assert err.details["agent_id"] == "override"

    def test_version_not_found_names_the_version(self):
        err = AgentVersionNotFoundError("agent_1", 3)
        assert "3" in err.message and err.details["version"] == 3

    def test_run_failed_embeds_the_underlying_error(self):
        err = AgentRunFailedError("agent_1", "upstream 503")
        assert "upstream 503" in err.message
        assert err.details["error"] == "upstream 503"

    def test_run_failed_is_a_500(self):
        assert AgentRunFailedError("agent_1", "x").status_code == 500

    def test_stream_failed_has_its_own_code(self):
        assert AgentRunStreamFailedError("a", "x").code == "AGENT_RUN_STREAM_FAILED"

    def test_chat_not_found_carries_the_chat_id(self):
        assert ChatNotFoundError("chat_1").details["chat_id"] == "chat_1"


class TestAuthExceptions:
    @pytest.mark.parametrize(
        ("cls", "expected_status"),
        [
            (UnauthorizedError, 401),
            (ForbiddenError, 403),
            (InvalidApiKeyOrClientError, 401),
            (AuthSessionExpiredError, 401),
        ],
    )
    def test_status_codes(self, cls, expected_status):
        assert cls("nope").status_code == expected_status

    def test_session_expired_has_its_own_code(self):
        assert AuthSessionExpiredError("expired").code == "AUTH_SESSION_EXPIRED"

    def test_an_auth_error_message_is_not_forced_to_a_default(self):
        assert UnauthorizedError("custom reason").message == "custom reason"


class TestSystemExceptions:
    def test_validation_errors_are_4xx_not_5xx(self):
        """A bad request is the caller's fault; reporting it as a 500 makes
        every client retry something that can never succeed."""
        assert 400 <= KitaValidationError("bad input").status_code < 500

    def test_database_errors_are_5xx(self):
        assert KitaDatabaseError("mongo down").status_code >= 500

    def test_redis_errors_are_5xx(self):
        assert KitaRedisError("redis down").status_code >= 500

    def test_configuration_errors_are_5xx(self):
        assert SystemConfigurationError("missing env").status_code >= 500

    def test_each_system_error_has_a_distinct_code(self):
        codes = {
            KitaDatabaseError("x").code,
            KitaRedisError("x").code,
            KitaValidationError("x").code,
        }
        assert len(codes) == 3


class TestToolExceptions:
    def test_web_search_error_keeps_its_details(self):
        err = ToolWebSearchError("search failed", details={"query": "kita"})
        assert err.details["query"] == "kita"

    def test_a_tool_error_is_still_a_kita_exception(self):
        assert isinstance(ToolWebSearchError("x"), KitaException)

    def test_to_dict_round_trips_for_a_subclass(self):
        payload = ToolWebSearchError("search failed", details={"query": "k"}).to_dict()
        assert payload["error"]["code"] == ToolWebSearchError.code
        assert payload["error"]["details"] == {"query": "k"}
