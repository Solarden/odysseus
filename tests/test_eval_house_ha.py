import pytest

from scripts.eval_house_ha import ha_call_ok, pass_k

TOOL = "mcp__9114454f__HassTurnOn"
AUDIT = {"type": "tool_resolution_audit", "accepted_tools": [TOOL], "used_native": True}
OUTPUT = {"type": "tool_output", "tool": TOOL, "exit_code": 0}


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([AUDIT, OUTPUT], True),
        ([{**AUDIT, "used_native": False}, OUTPUT], False),
        ([OUTPUT], False),
        ([AUDIT], False),
        ([AUDIT, {**OUTPUT, "exit_code": 1}], False),
        ([AUDIT, {**OUTPUT, "exit_code": None}], False),
        ([AUDIT, {**OUTPUT, "tool": "mcp__9114454f__HassTurnOff"}], False),
        ([{**AUDIT, "accepted_tools": ["ask_user"]}, OUTPUT], False),
        ([AUDIT, OUTPUT, {"type": "error", "error": "timeout"}], False),
        # A later native round does not excuse a first call recovered from text.
        ([{**AUDIT, "used_native": False}, OUTPUT, AUDIT, OUTPUT], False),
    ],
)
def test_ha_call_ok(events, expected):
    assert ha_call_ok(events, "HassTurnOn") is expected


def test_pass_k_needs_every_run():
    summary = pass_k({"on": [True, True, True], "off": [True, False, True], "never": []})

    assert summary == {"pass_k": "1/3", "per_case": {"on": "3/3", "off": "2/3", "never": "0/0"}}
