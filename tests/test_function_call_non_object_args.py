import pytest
from src.tool_schemas import function_call_to_tool_block


@pytest.mark.parametrize("arguments", [
    '["ls -la"]',   # JSON array
    '"ls -la"',     # bare JSON string
    '42',            # JSON number
    'true',          # JSON bool
    'null',          # JSON null
])
def test_non_object_arguments_do_not_crash(arguments):
    """A native function call whose arguments are valid JSON but not an object
    must not raise (it used to throw AttributeError: 'list' object has no
    attribute 'get', aborting the entire agent stream)."""
    block = function_call_to_tool_block("bash", arguments)
    # Coerced to empty args -> empty bash command, but importantly NO crash.
    assert block is not None
    assert block.tool_type == "bash"
    assert block.content == ""


@pytest.mark.parametrize("tool_name", ["list_emails", "mcp__email__list_emails"])
def test_email_mcp_non_object_arguments_are_rejected(tool_name):
    block = function_call_to_tool_block(tool_name, '["INBOX"]')

    assert block is None


def test_write_file_recovers_legacy_command_path_content_shape():
    block = function_call_to_tool_block(
        "write_file",
        '{"command":"/workspace/output.html\\n<html>ok</html>"}',
    )

    assert block is not None
    assert block.tool_type == "write_file"
    assert block.content == "/workspace/output.html\n<html>ok</html>"


def test_write_file_rejects_one_line_command_alias():
    block = function_call_to_tool_block(
        "write_file",
        '{"command":"not a path-plus-content payload"}',
    )

    assert block is None


def test_edit_document_skips_non_object_edit_items():
    block = function_call_to_tool_block(
        "edit_document",
        '{"edits": ["bad", 42, null, {"find": "old", "replace": "new"}]}',
    )

    assert block is not None
    assert block.tool_type == "edit_document"
    assert block.content == "<<<FIND>>>\nold\n<<<REPLACE>>>\nnew\n<<<END>>>"


def test_suggest_document_skips_non_object_suggestion_items():
    block = function_call_to_tool_block(
        "suggest_document",
        '{"suggestions": ["bad", 42, null, {"find": "old", "replace": "new", "reason": "clearer"}]}',
    )

    assert block is not None
    assert block.tool_type == "suggest_document"
    assert block.content == (
        "<<<FIND>>>\nold\n<<<SUGGEST>>>\nnew\n<<<REASON>>>\nclearer\n<<<END>>>"
    )


def test_ui_control_open_email_reply_preserves_structured_body():
    block = function_call_to_tool_block(
        "ui_control",
        '{"action":"open_email_reply","uid":"3228","folder":"INBOX","mode":"reply","body":"Hi Andy,\\n\\nNo thank you.\\n\\nBest,"}',
    )

    assert block is not None
    assert block.tool_type == "ui_control"
    assert block.content == "open_email_reply 3228 INBOX reply Hi Andy,\n\nNo thank you.\n\nBest,"


class _FakeMcpManager:
    def __init__(self, qualified_names):
        self._names = qualified_names

    def get_all_tools(self):
        return [{"qualified_name": n} for n in self._names]


@pytest.mark.parametrize("tool_name", [
    "mcp__list_emails",
    "mcp__9114454f__list_emails",
    "mcp__builtin_browser__list_emails",
])
def test_misprefixed_email_call_is_routed_to_email_server(tool_name, monkeypatch):
    monkeypatch.setitem(function_call_to_tool_block.__globals__, "get_mcp_manager", lambda: _FakeMcpManager([]))

    block = function_call_to_tool_block(tool_name, '{"max_results": 5}')

    assert block is not None
    assert block.tool_type == "mcp__email__list_emails"
    assert block.content == '{"max_results": 5}'


@pytest.mark.parametrize("tool_name", [
    "mcp__9114454f__HassTurnOn",
    "mcp__acme_mail__send_email",
])
def test_real_mcp_tool_is_not_rerouted(tool_name, monkeypatch):
    monkeypatch.setitem(function_call_to_tool_block.__globals__, "get_mcp_manager", lambda: _FakeMcpManager([tool_name]))

    block = function_call_to_tool_block(tool_name, "{}")

    assert block is not None
    assert block.tool_type == tool_name
