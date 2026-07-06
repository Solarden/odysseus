"""Open-weight models trained on the LangChain "structured chat" agent often
fall back to emitting the ReAct envelope as raw JSON instead of using the
fenced/native tool channel:

    {"action": "generate_image", "action_input": "{\\"prompt\\": \\"a cat\\"}"}

Older parsing only recognized fenced blocks, [TOOL_CALL], XML invoke, tool_code,
StepFun/Gemma tokens, DSML, and bare web_search JSON — so this envelope neither
fired the tool nor got stripped, and leaked into the chat reply. It is now
recovered as Pattern 5b (keyed on `action_input`, only when `action` names a
real tool).
"""
import json
import sys
from unittest.mock import MagicMock

for mod in ['src.agent_tools', 'src.tool_parsing', 'src.tool_schemas', 'src.tool_execution']:
    sys.modules.pop(mod, None)
for mod in [
    'sqlalchemy', 'sqlalchemy.orm', 'sqlalchemy.ext', 'sqlalchemy.ext.declarative',
    'sqlalchemy.ext.hybrid', 'sqlalchemy.sql', 'sqlalchemy.sql.expression',
    'src.database', 'core.models', 'core.database', 'core.auth'
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import src.agent_tools  # noqa: E402, F401
from src.tool_parsing import parse_tool_blocks, strip_tool_blocks  # noqa: E402


# action_input as a stringified JSON blob (the exact reported shape).
_STRINGIFIED = '{"action": "generate_image", "action_input": "{ \\"prompt\\": \\"a cat\\" }"}'
# action_input as a nested object.
_OBJECT = '{"action": "generate_image", "action_input": {"prompt": "a cat"}}'


def test_react_envelope_fires_generate_image():
    for text in (_STRINGIFIED, _OBJECT):
        blocks = parse_tool_blocks(text)
        assert len(blocks) == 1, text
        assert blocks[0].tool_type == "generate_image"
        assert json.loads(blocks[0].content)["prompt"] == "a cat"


def test_bare_string_action_input_fires_generate_image():
    # LangChain's single-input form: action_input is the bare prompt string,
    # NOT a JSON object. This regressed (json.loads of the prompt failed -> the
    # call didn't fire and leaked as text). It must map to the tool's primary
    # arg (prompt) and fire.
    text = (
        '{"action": "generate_image", "action_input": '
        '"A mystical forest scene with hanging vines and glowing lanterns."}'
    )
    blocks = parse_tool_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "generate_image"
    assert "mystical forest" in json.loads(blocks[0].content)["prompt"]
    assert "action_input" not in strip_tool_blocks(text)


def test_bare_string_action_input_maps_web_search_query():
    text = '{"action": "web_search", "action_input": "best espresso machines 2026"}'
    blocks = parse_tool_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"


def test_react_envelope_recovered_even_when_native_gate_skips_fenced():
    # Never an illustrative example — recover regardless of skip_fenced.
    blocks = parse_tool_blocks(_STRINGIFIED, skip_fenced=True)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "generate_image"


def test_react_envelope_inside_json_fence():
    text = "Here you go:\n```json\n" + _STRINGIFIED + "\n```"
    blocks = parse_tool_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "generate_image"


def test_strip_removes_react_envelope():
    cleaned = strip_tool_blocks(_STRINGIFIED)
    assert "action_input" not in cleaned
    assert "generate_image" not in cleaned


def test_strip_removes_react_envelope_and_wrapping_fence():
    text = "Sure:\n```json\n" + _STRINGIFIED + "\n```"
    cleaned = strip_tool_blocks(text)
    assert "action_input" not in cleaned
    assert "```" not in cleaned  # no dangling empty fence
    assert "Sure:" in cleaned


def test_action_key_without_action_input_is_not_misparsed():
    # `action` alone is a legit arg key for manage_notes / manage_calendar etc.
    # No action_input => not a ReAct envelope => left untouched.
    text = '{"action": "add", "title": "buy milk"}'
    assert parse_tool_blocks(text) == []
    assert strip_tool_blocks(text) == text


def test_unknown_action_never_fires_but_is_stripped():
    # An unknown/stray action ("something_else") must NOT fire a tool, but IS
    # stripped from display — leaving it as raw JSON both looks broken and (in
    # agent mode) resets the loop-breaker, feeding the runaway regeneration loop.
    text = '{"action": "definitely_not_a_tool", "action_input": "{}", "thought": "hmm"}'
    assert parse_tool_blocks(text) == []
    assert "definitely_not_a_tool" not in strip_tool_blocks(text)
    assert "action_input" not in strip_tool_blocks(text)


def test_final_answer_sentinel_is_unwrapped_not_deleted():
    # LangChain's terminal action; its action_input IS the user-facing answer.
    # It must not fire a tool, and strip must UNWRAP it (keep the answer),
    # never delete it.
    text = '{"action": "Final Answer", "action_input": "The capital of France is Paris."}'
    assert parse_tool_blocks(text) == []
    cleaned = strip_tool_blocks(text)
    assert cleaned.strip() == "The capital of France is Paris."


def test_session_meta_tools_never_fire_from_react_blob():
    # A confused model emitting a session/meta action in a ReAct blob must NOT
    # spawn/fork a chat (the "new session appears in the sidebar" bug). Not
    # fired, but still stripped so it neither leaks nor feeds the loop.
    for tool in ("create_session", "manage_session", "send_to_session",
                 "list_sessions", "chat_with_model"):
        text = f'{{"action": "{tool}", "action_input": "{{}}", "thought": "x"}}'
        assert parse_tool_blocks(text) == [], tool
        assert tool not in strip_tool_blocks(text), tool


def test_content_tool_fires_even_when_a_meta_blob_is_present():
    # A blocked meta blob before a content blob must not shadow the content one:
    # generate_image still fires; the create_session blob is ignored + stripped.
    text = (
        '{"action": "create_session", "action_input": "{}", "thought": "spawn"}'
        '{"action": "generate_image", "action_input": "{\\"prompt\\": \\"a cat\\"}"}'
    )
    blocks = parse_tool_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "generate_image"
    assert "create_session" not in strip_tool_blocks(text)


def test_multiple_blobs_fire_once_and_all_stripped():
    # Mirrors the observed runaway: a weak model emits two identical
    # generate_image blobs + a stray something_else in one response. Exactly ONE
    # generate_image fires (no double image); ALL blobs are stripped so the
    # displayed/looped text is empty (lets the loop-breaker converge).
    text = (
        '{"action": "generate_image", "action_input": "{\\"prompt\\": \\"a cat\\"}", "thought": "t1"}'
        '{"action": "generate_image", "action_input": "{\\"prompt\\": \\"a cat\\"}", "thought": "t2"}'
        '{"action": "something_else", "action_input": "{}", "thought": "t3"}'
    )
    blocks = parse_tool_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "generate_image"
    assert strip_tool_blocks(text).strip() == ""
