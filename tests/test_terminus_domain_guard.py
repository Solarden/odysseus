import pytest

from src.agent_loop import _classify_agent_request, _wants_terminus_toolset


@pytest.mark.parametrize("query", [
    "generate a photo of an old fisherman with a sign reading MARTA on his boat",
    "generate a picture of a cat sleeping on her lap",
    "show me the email from Anna",
    "add a reminder on Monday",
])
def test_named_domain_keeps_its_tools(query):
    domains = _classify_agent_request([{"role": "user", "content": query}], query).get("domains")

    assert _wants_terminus_toolset(query, None, domains) is False


@pytest.mark.parametrize("query, workspace, domains", [
    ("check disk space on petel", None, set()),
    ("list the files on this computer", None, {"files"}),
    ("fix the failing test", "/work/repo", set()),
])
def test_local_computer_and_workspace_requests_still_switch(query, workspace, domains):
    assert _wants_terminus_toolset(query, workspace, domains) is True
