"""Gemma 4 sometimes writes its native call as text, without the <|tool_call>
markers that Pattern 4b keys on: ```mcp__…__HassTurnOff{name:<|"|>x<|"|>}```.
The samples below are verbatim replies from huihui_ai/gemma-4-abliterated:26b
as `house` on 2026-11-15/16; none of them ran, and HA was never called.
"""
import json

import pytest

from src.agent_loop import _resolve_tool_blocks
from src.tool_parsing import parse_tool_blocks, strip_tool_blocks

_TURN_OFF = "mcp__9114454f__HassTurnOff"
_LIVE_CONTEXT = "mcp__9114454f__GetLiveContext"


@pytest.mark.parametrize("text, tool, args", [
    (
        '```mcp__9114454f__HassTurnOff{name:<|"|>living room light<|"|>}```',
        _TURN_OFF,
        {"name": "living room light"},
    ),
    (
        '```mcp__9114454f__HassTurnOff{area:<|"|>living room<|"|>,name:<|"|>ceiling light<|"|>}\n```',
        _TURN_OFF,
        {"area": "living room", "name": "ceiling light"},
    ),
    (
        '```mcp__9114454f__HassTurnOff{area:<|"|>living room<|"|>,name:<|"|>ceiling light<|"|>}```Done.',
        _TURN_OFF,
        {"area": "living room", "name": "ceiling light"},
    ),
    (
        '```mcp__9114454f__HassTurnOff{area:<|"|>living room<|"|>,domain:[<|"|>light<|"|>],'
        'name:<|"|>living room light<|"|>}```',
        _TURN_OFF,
        {"area": "living room", "domain": ["light"], "name": "living room light"},
    ),
    (
        '```mcp__9114454f__GetLiveContext{area:<|"|>living room<|"|>,domain:[<|"|>light<|"|>]}```'
        "get_live_context(area='living room', domain=['light'])",
        _LIVE_CONTEXT,
        {"area": "living room", "domain": ["light"]},
    ),
])
@pytest.mark.parametrize("skip_fenced", [True, False])
def test_leaked_gemma_call_parses_with_its_name_case(text, tool, args, skip_fenced):
    blocks = parse_tool_blocks(text, skip_fenced=skip_fenced)

    assert [(b.tool_type, json.loads(b.content)) for b in blocks] == [(tool, args)]


@pytest.mark.parametrize("text", [
    "Use a dict like config{name: 'x'} to set it.",
    '```json\n{"name": "living room light"}\n```',
    'The token is <|"|> and nothing else.',
    "call:HassTurnOff{name: living room}",
])
def test_no_gemma_string_token_in_the_body_is_not_a_call(text):
    assert parse_tool_blocks(text, skip_fenced=True) == []


def test_unclosed_body_is_not_a_call():
    assert parse_tool_blocks('mcp__x__HassTurnOff{name:<|"|>living room<|"|>', skip_fenced=True) == []


def test_call_after_many_brace_openers_is_still_found():
    text = "x{1} " * 60 + '```mcp__x__HassTurnOff{name:<|"|>lamp<|"|>}```'

    assert [b.tool_type for b in parse_tool_blocks(text, skip_fenced=True)] == ["mcp__x__HassTurnOff"]


@pytest.mark.parametrize("text", [
    'See config{ ```mcp__x__HassTurnOff{name:<|"|>lamp<|"|>}``` }',
    'note{text: mcp__x__HassTurnOff{name:<|"|>lamp<|"|>}}',
    'Call:mcp__x__HassTurnOff{name:<|"|>lamp<|"|>}',
])
def test_the_call_itself_is_found_whatever_wraps_or_prefixes_it(text):
    assert [b.tool_type for b in parse_tool_blocks(text, skip_fenced=True)] == ["mcp__x__HassTurnOff"]


def test_brace_text_inside_a_value_does_not_hide_the_call():
    text = 'mcp__x__HassBroadcast{message:<|"|>set{a:1} done<|"|>}'

    blocks = parse_tool_blocks(text, skip_fenced=True)

    assert [(b.tool_type, json.loads(b.content)) for b in blocks] == [
        ("mcp__x__HassBroadcast", {"message": "set{a:1} done"}),
    ]


def test_comma_and_colon_inside_a_value_stay_in_the_value():
    text = 'mcp__x__HassListAddItem{item:<|"|>milk, note: organic<|"|>,name:<|"|>Shopping<|"|>}'

    blocks = parse_tool_blocks(text, skip_fenced=True)

    assert json.loads(blocks[0].content) == {"item": "milk, note: organic", "name": "Shopping"}


def test_strip_removes_the_call_and_its_fence():
    text = '```mcp__9114454f__HassTurnOff{area:<|"|>living room<|"|>,name:<|"|>ceiling light<|"|>}```Done.'

    assert strip_tool_blocks(text, skip_fenced=True).strip() == "Done."


def test_resolve_runs_an_offered_leaked_call_for_an_api_model():
    text = '```mcp__9114454f__HassTurnOff{name:<|"|>living room light<|"|>}```'

    blocks, used_native, _ = _resolve_tool_blocks(
        text, [], 1, is_api_model=True, offered_tool_names={_TURN_OFF, "web_search"},
    )

    assert used_native is False
    assert [(b.tool_type, json.loads(b.content)) for b in blocks] == [(_TURN_OFF, {"name": "living room light"})]


def test_resolve_drops_a_leaked_call_that_was_not_offered():
    text = '```mcp__9114454f__HassTurnOff{name:<|"|>living room light<|"|>}```'

    blocks, _, _ = _resolve_tool_blocks(
        text, [], 1, is_api_model=True, offered_tool_names={"web_search"},
    )

    assert blocks == []
