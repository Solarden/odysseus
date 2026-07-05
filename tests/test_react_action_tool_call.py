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


def test_unknown_action_is_ignored():
    text = '{"action": "definitely_not_a_tool", "action_input": "{}"}'
    assert parse_tool_blocks(text) == []
    # Nothing recognized -> nothing stripped.
    assert "definitely_not_a_tool" in strip_tool_blocks(text)


def test_final_answer_sentinel_is_left_alone():
    # LangChain's terminal action; its action_input is the user-facing answer,
    # not a tool. function_call_to_tool_block returns None -> untouched.
    text = '{"action": "Final Answer", "action_input": "The capital of France is Paris."}'
    assert parse_tool_blocks(text) == []
    assert "Paris" in strip_tool_blocks(text)
