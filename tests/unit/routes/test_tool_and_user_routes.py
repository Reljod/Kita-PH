"""Tests for app.routes.tool and app.routes.user."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

from app.dependencies import get_agent_service, get_tool_service
from app.models.agent import AgentResponse
from app.models.user import UserResponse
from app.routes import tool as tool_routes
from app.routes import user as user_routes
from app.security import get_current_user, get_user_service

from .conftest import override

TOOL_OID = ObjectId("64b7f1c2e4b0a1a2b3c4d5e6")
TOOL_ID = str(TOOL_OID)
USER_ID = "user_1"
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)


def tool_doc(**overrides) -> dict:
    doc = {
        "_id": TOOL_OID,
        "name": "web_search",
        "description": "Search the web",
        "created_at": NOW,
        "updated_at": NOW,
    }
    doc.update(overrides)
    return doc


def an_agent() -> AgentResponse:
    return AgentResponse(
        id="agent_1",
        base_id="agent_1",
        version=1,
        name="Researcher",
        role="analyst",
        goal="find things",
        backstory="trained",
        llm_id="llm_1",
        tools=["web_search"],
        created_at=NOW,
        updated_at=NOW,
    )


def a_user(**overrides) -> UserResponse:
    payload = {
        "id": USER_ID,
        "email": "person@example.com",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "created_at": NOW,
        "updated_at": NOW,
    }
    payload.update(overrides)
    return UserResponse(**payload)


# --- tool routes ----------------------------------------------------------


@pytest.fixture
def tool_service() -> MagicMock:
    service = MagicMock(name="tool_service")
    service.get_tools = AsyncMock(return_value=[tool_doc()])
    service.get_tool = AsyncMock(return_value=tool_doc())
    service.get_tool_by_name = AsyncMock(return_value=tool_doc())
    service.register_tool = AsyncMock(return_value=True)
    service.deregister_tool = AsyncMock(return_value=True)
    return service


@pytest.fixture
def agent_service() -> MagicMock:
    service = MagicMock(name="agent_service")
    service.get_agents_by_tool.return_value = [an_agent()]
    return service


@pytest.fixture
def tool_client(make_client, tool_service, agent_service):
    return make_client(
        tool_routes.router,
        {
            get_tool_service: override(tool_service),
            get_agent_service: override(agent_service),
        },
    )


class TestListTools:
    def test_tools_are_listed(self, tool_client):
        response = tool_client.get("/tool/")
        assert response.status_code == 200 and len(response.json()) == 1

    def test_an_empty_registry_lists_nothing(self, tool_client, tool_service):
        tool_service.get_tools = AsyncMock(return_value=[])
        assert tool_client.get("/tool/").json() == []

    def test_a_service_failure_is_a_500(self, tool_client, tool_service):
        tool_service.get_tools = AsyncMock(side_effect=RuntimeError("mongo down"))
        assert tool_client.get("/tool/").status_code == 500


class TestAvailableTools:
    def test_the_built_in_catalogue_is_returned(self, tool_client):
        with patch.object(
            tool_routes, "get_available_tools", return_value=["web_search"]
        ):
            response = tool_client.get("/tool/available")
        assert response.status_code == 200 and response.json() == ["web_search"]

    def test_a_catalogue_failure_is_a_500(self, tool_client):
        with patch.object(
            tool_routes,
            "get_available_tools",
            side_effect=RuntimeError("import blew up"),
        ):
            assert tool_client.get("/tool/available").status_code == 500


class TestRegisterTool:
    def test_a_tool_is_registered(self, tool_client):
        response = tool_client.post("/tool/", json={"name": "web_search"})
        assert response.status_code == 201

    def test_the_name_reaches_the_service(self, tool_client, tool_service):
        tool_client.post("/tool/", json={"name": "web_search"})
        tool_service.register_tool.assert_awaited_once_with("web_search")

    def test_a_refused_registration_is_a_400_not_a_500(self, tool_client, tool_service):
        """The handler's own HTTPException(400) must not be re-wrapped by the
        broad except below it."""
        tool_service.register_tool = AsyncMock(return_value=False)
        assert tool_client.post("/tool/", json={"name": "nope"}).status_code == 400

    def test_an_unknown_tool_name_is_a_400(self, tool_client, tool_service):
        tool_service.register_tool = AsyncMock(side_effect=ValueError("no such tool"))
        assert tool_client.post("/tool/", json={"name": "nope"}).status_code == 400

    def test_a_service_failure_is_a_500(self, tool_client, tool_service):
        tool_service.register_tool = AsyncMock(side_effect=RuntimeError("mongo down"))
        assert tool_client.post("/tool/", json={"name": "x"}).status_code == 500

    @pytest.mark.parametrize("body", [{"name": ""}, {}, {"name": "x" * 101}])
    def test_invalid_input_is_rejected(self, tool_client, tool_service, body):
        assert tool_client.post("/tool/", json=body).status_code == 422
        tool_service.register_tool.assert_not_awaited()


class TestGetTool:
    def test_a_tool_is_returned_by_id(self, tool_client):
        assert tool_client.get(f"/tool/{TOOL_ID}").status_code == 200

    def test_a_missing_tool_is_a_404_not_a_500(self, tool_client, tool_service):
        tool_service.get_tool = AsyncMock(return_value=None)
        assert tool_client.get(f"/tool/{TOOL_ID}").status_code == 404

    def test_a_tool_is_returned_by_name(self, tool_client):
        assert tool_client.get("/tool/name/web_search").status_code == 200

    def test_a_missing_name_is_a_404_not_a_500(self, tool_client, tool_service):
        tool_service.get_tool_by_name = AsyncMock(return_value=None)
        assert tool_client.get("/tool/name/nope").status_code == 404

    def test_a_service_failure_is_a_500(self, tool_client, tool_service):
        tool_service.get_tool = AsyncMock(side_effect=RuntimeError("mongo down"))
        assert tool_client.get(f"/tool/{TOOL_ID}").status_code == 500


class TestDeregisterTool:
    def test_a_tool_is_deregistered(self, tool_client):
        assert tool_client.delete(f"/tool/{TOOL_ID}").status_code == 200

    def test_a_missing_tool_is_a_404_not_a_500(self, tool_client, tool_service):
        tool_service.deregister_tool = AsyncMock(return_value=False)
        assert tool_client.delete(f"/tool/{TOOL_ID}").status_code == 404

    def test_a_service_failure_is_a_500(self, tool_client, tool_service):
        tool_service.deregister_tool = AsyncMock(side_effect=RuntimeError("down"))
        assert tool_client.delete(f"/tool/{TOOL_ID}").status_code == 500


class TestToolAgents:
    def test_agents_using_a_tool_are_listed(self, tool_client):
        response = tool_client.get(f"/tool/{TOOL_ID}/agents")
        assert response.status_code == 200 and len(response.json()) == 1

    def test_the_lookup_uses_the_path_id(self, tool_client, agent_service):
        tool_client.get(f"/tool/{TOOL_ID}/agents")
        agent_service.get_agents_by_tool.assert_called_once_with(TOOL_ID)

    def test_no_agents_yields_an_empty_list(self, tool_client, agent_service):
        agent_service.get_agents_by_tool.return_value = []
        assert tool_client.get(f"/tool/{TOOL_ID}/agents").json() == []

    def test_a_service_failure_is_a_500(self, tool_client, agent_service):
        agent_service.get_agents_by_tool.side_effect = RuntimeError("mongo down")
        assert tool_client.get(f"/tool/{TOOL_ID}/agents").status_code == 500


# --- user routes ----------------------------------------------------------


@pytest.fixture
def user_service() -> MagicMock:
    service = MagicMock(name="user_service")
    service.update_user.return_value = a_user()
    service.update_password.return_value = True
    return service


@pytest.fixture
def user_client(make_client, user_service):
    return make_client(
        user_routes.router,
        {
            get_current_user: override(a_user()),
            get_user_service: override(user_service),
        },
    )


class TestGetMe:
    def test_the_current_user_is_returned(self, user_client):
        response = user_client.get("/user/me")
        assert response.status_code == 200
        assert response.json()["id"] == USER_ID

    def test_the_password_is_never_exposed(self, user_client):
        """UserResponse omits it, and this pins that it stays omitted."""
        assert "password" not in user_client.get("/user/me").json()


class TestUpdateMe:
    def test_the_profile_is_updated(self, user_client):
        response = user_client.patch("/user/me", json={"first_name": "Grace"})
        assert response.status_code == 200

    def test_the_update_targets_the_authenticated_user(self, user_client, user_service):
        """The id comes from the token, never from the request body, so one
        user cannot edit another's profile."""
        user_client.patch(
            "/user/me", json={"first_name": "Grace", "id": "someone_else"}
        )
        assert user_service.update_user.call_args[0][0] == USER_ID

    def test_only_the_supplied_fields_are_sent(self, user_client, user_service):
        user_client.patch("/user/me", json={"first_name": "Grace"})
        submitted = user_service.update_user.call_args[0][1]
        assert submitted.model_dump(exclude_unset=True) == {"first_name": "Grace"}

    def test_an_empty_update_is_allowed(self, user_client):
        assert user_client.patch("/user/me", json={}).status_code == 200

    def test_a_blank_name_is_rejected(self, user_client, user_service):
        assert user_client.patch("/user/me", json={"first_name": ""}).status_code == 422
        user_service.update_user.assert_not_called()

    def test_a_vanished_user_is_a_404(self, user_client, user_service):
        """update_user returns None when the record went away between
        authentication and the write. Returning that under
        response_model=UserResponse produced a serialisation failure — a 500
        for what is really a not-found."""
        user_service.update_user.return_value = None
        response = user_client.patch("/user/me", json={"first_name": "Grace"})
        assert response.status_code == 404


class TestUpdatePassword:
    def test_the_password_is_updated(self, user_client):
        response = user_client.patch(
            "/user/me/password",
            json={"old_password": "old-password", "new_password": "new-password"},
        )
        assert response.status_code == 200

    def test_the_change_targets_the_authenticated_user(self, user_client, user_service):
        user_client.patch(
            "/user/me/password",
            json={"old_password": "old-password", "new_password": "new-password"},
        )
        assert user_service.update_password.call_args[0][0] == USER_ID

    def test_a_wrong_current_password_is_a_400(self, user_client, user_service):
        user_service.update_password.return_value = False
        response = user_client.patch(
            "/user/me/password",
            json={"old_password": "wrong-password", "new_password": "new-password"},
        )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        "body",
        [
            {"old_password": "short", "new_password": "new-password"},
            {"old_password": "old-password", "new_password": "short"},
            {"new_password": "new-password"},
            {"old_password": "old-password"},
        ],
    )
    def test_invalid_input_is_rejected(self, user_client, user_service, body):
        assert user_client.patch("/user/me/password", json=body).status_code == 422
        user_service.update_password.assert_not_called()
