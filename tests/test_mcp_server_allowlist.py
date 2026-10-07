"""Per-user MCP server grants: a non-admin reaches only the servers an admin listed."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import core.auth
from src.agent_tools import ToolBlock
from src.tool_security import is_public_blocked_tool, mcp_servers_for_owner

GRANTED = "a1b2c3d4"


class _FakeAuth:
    grants = {"house": [GRANTED, "email", "builtin_browser", 7]}

    def is_admin(self, owner):
        return owner == "root"

    def get_privileges(self, owner):
        return {"allowed_mcp_servers": self.grants.get(owner, [])}


class _BrokenAuth:
    def __init__(self):
        raise RuntimeError("auth store unreadable")


@pytest.fixture
def fake_auth(monkeypatch):
    monkeypatch.setattr(core.auth, "AuthManager", _FakeAuth)


@pytest.mark.parametrize(
    "tool, owner, blocked",
    [
        (f"mcp__{GRANTED}__turn_on", "house", False),
        ("mcp__deadbeef__turn_on", "house", True),
        ("mcp__email__send_email", "house", True),
        ("mcp__builtin_browser__browser_click", "house", True),
        ("bash", "house", True),
        ("send_email", "house", True),
        (f"mcp__{GRANTED}__turn_on", None, True),
        (f"mcp__{GRANTED}__turn_on", "stranger", True),
        ("web_search", "house", False),
    ],
)
def test_gate(fake_auth, tool, owner, blocked):
    assert is_public_blocked_tool(tool, owner) is blocked


def test_builtin_and_malformed_ids_are_never_granted(fake_auth):
    assert mcp_servers_for_owner("house") == {GRANTED}
    assert mcp_servers_for_owner("root") == set()


def test_unreadable_grants_fail_closed(monkeypatch):
    monkeypatch.setattr(core.auth, "AuthManager", _BrokenAuth)
    assert mcp_servers_for_owner("house") == set()
    assert is_public_blocked_tool(f"mcp__{GRANTED}__turn_on", "house") is True


def _manager(servers):
    from src.agent_runtime.remote_resources import configuration_incarnation, endpoint_identity
    from src.mcp_manager import McpManager

    manager = McpManager()
    for server in servers:
        session = SimpleNamespace(call_tool=AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(text="ok")], isError=False)))
        url = f"https://{server}.test/mcp"
        manager._sessions[server] = session
        manager._tools[server] = [{"name": "turn_on"}]
        manager._resource_endpoints[server] = (endpoint_identity(url), configuration_incarnation(url))
        manager._register_resource_connection(server, session)

    return manager


@pytest.mark.parametrize(
    "owner, admin, granted",
    [
        ("house", False, {f"mcp__{GRANTED}__turn_on"}),
        ("root", True, {f"mcp__{GRANTED}__turn_on", "mcp__deadbeef__turn_on"}),
        ("stranger", False, set()),
    ],
)
def test_request_authority_admits_owner_mcp_tools(fake_auth, monkeypatch, owner, admin, granted):
    import src.tool_security as ts
    from src import agent_tools
    from src.agent_runtime.authority import create_request_authority

    monkeypatch.setattr(agent_tools, "get_mcp_manager", lambda: _manager([GRANTED, "deadbeef", "email"]))
    monkeypatch.setattr(ts, "owner_is_admin_or_single_user", lambda o: admin)

    authority = create_request_authority("turn on the lamp", owner=owner)

    assert {g.tool for g in authority.grants if g.tool.startswith("mcp__")} == granted


@pytest.mark.asyncio
async def test_granted_server_is_dispatched_for_non_admin(fake_auth, monkeypatch):
    # Resolve from the live module: other tests re-import src.tool_execution
    # (see test_edit_file.py), so a top-level reference could miss the patch.
    import src.tool_execution as te
    import src.tool_security as ts
    from src import agent_tools
    from src.agent_runtime.authority import create_request_authority

    manager = _manager([GRANTED, "deadbeef"])
    monkeypatch.setattr(te, "_owner_is_admin", lambda owner: False)
    monkeypatch.setattr(ts, "owner_is_admin_or_single_user", lambda owner: False)
    monkeypatch.setattr(te, "get_mcp_manager", lambda: manager)
    monkeypatch.setattr(agent_tools, "get_mcp_manager", lambda: manager)
    authority = create_request_authority("turn on the lamp", owner="house")

    for server, reached in ((GRANTED, True), ("deadbeef", False)):
        _desc, result = await te.execute_tool_block(
            ToolBlock(f"mcp__{server}__turn_on", "{}"), owner="house",
            request_authority=authority, security_context=te.NO_TOOL_SECURITY_CONTEXT,
        )
        assert manager._sessions[server].call_tool.await_count == int(reached), (server, result)
