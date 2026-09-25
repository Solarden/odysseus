"""Per-user MCP server grants: a non-admin reaches only the servers an admin listed."""
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


@pytest.mark.asyncio
async def test_granted_server_is_dispatched_for_non_admin(fake_auth, monkeypatch):
    # Resolve from the live module: other tests re-import src.tool_execution
    # (see test_edit_file.py), so a top-level reference could miss the patch.
    import src.tool_execution as te

    calls = []

    class _Mcp:
        async def call_tool(self, name, args):
            calls.append(name)
            return {"output": "ok", "exit_code": 0}

    monkeypatch.setattr(te, "_owner_is_admin", lambda owner: False)
    monkeypatch.setattr(te, "get_mcp_manager", lambda: _Mcp())

    for tool, reached in ((f"mcp__{GRANTED}__turn_on", True), ("mcp__deadbeef__turn_on", False)):
        _desc, result = await te.execute_tool_block(
            ToolBlock(tool, "{}"), owner="house", security_context=te.NO_TOOL_SECURITY_CONTEXT
        )
        assert (tool in calls) is reached, (tool, result)
