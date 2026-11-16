#!/usr/bin/env python3
"""Measure whether a model makes native Home Assistant calls as the `house` user.

Each case goes through the real chat route, so the model sees odysseus's full request (every
tool, the system prompt), which is what decides native calls. A run counts only when the model
called the HA tool natively and the call returned exit_code 0. The fork also runs calls it
recovers from text, and those run fine, so `tool_resolution_audit.used_native` is the only signal
that tells them apart. A case passes only when all of its ``--repeat`` runs count (pass^k).

The cases switch real lights: each "on" case is followed by its "off" case, so the lights end off.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_odysseus_tool_use import (
    _cookie,
    _raise_for_status_with_body,
    _sse_events,
)

CASES = [
    ("kitchen_on", "Turn on the Kitchen Light.", "HassTurnOn"),
    ("kitchen_off", "Turn off the Kitchen Light.", "HassTurnOff"),
    ("hall_on", "Can you switch the Hall Light on?", "HassTurnOn"),
    ("hall_off", "Please switch the Hall Light off.", "HassTurnOff"),
]


def ha_call_ok(events: list[dict], ha_tool: str) -> bool:
    suffix = f"__{ha_tool}"
    audits = [
        e for e in events
        if e.get("type") == "tool_resolution_audit"
        and any(str(name).endswith(suffix) for name in e.get("accepted_tools") or [])
    ]
    native = bool(audits) and bool(audits[0].get("used_native"))
    ran = any(
        e.get("type") == "tool_output" and str(e.get("tool") or "").endswith(suffix) and e.get("exit_code") == 0
        for e in events
    )
    errors = any(e.get("type") == "error" for e in events)

    return native and ran and not errors


def pass_k(results: dict[str, list[bool]]) -> dict:
    per_case = {name: f"{sum(runs)}/{len(runs)}" for name, runs in results.items()}
    passed = sum(bool(runs) and all(runs) for runs in results.values())

    return {"pass_k": f"{passed}/{len(results)}", "per_case": per_case}


def run_turn(client: httpx.Client, args, name: str, message: str) -> list[dict]:
    base = args.base_url.rstrip("/")
    create = client.post(
        base + "/api/session",
        data={
            "name": "[eval] " + name,
            "endpoint_url": args.endpoint,
            **({"endpoint_id": args.endpoint_id} if args.endpoint_id else {}),
            "model": args.model,
            "skip_validation": "true",
            "rag": "false",
        },
        timeout=30,
    )
    _raise_for_status_with_body(create)
    session_id = create.json()["id"]
    events: list[dict] = []
    try:
        with client.stream(
            "POST",
            base + "/api/chat_stream",
            data={"message": message, "session": session_id, "mode": "agent", "agent_prompt_mode": "auto"},
            headers={"Accept": "text/event-stream"},
            timeout=args.timeout,
        ) as response:
            _raise_for_status_with_body(response)
            events.extend(_sse_events(response))
    except httpx.HTTPError as exc:
        events.append({"type": "error", "error": repr(exc)})
    finally:
        try:
            client.delete(base + f"/api/session/{session_id}", timeout=15)
        except httpx.HTTPError:
            pass

    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7011")
    parser.add_argument("--endpoint", required=True, help="The endpoint URL `house` chats through.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint-id", default="")
    parser.add_argument("--cookie-file", default="data/sessions.json")
    parser.add_argument("--username", default="house")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", required=True, help="JSONL file with every run's events.")
    args = parser.parse_args()

    results: dict[str, list[bool]] = {name: [] for name, _, _ in CASES}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(
        cookies={"odysseus_session": _cookie(Path(args.cookie_file), args.username)},
        follow_redirects=False,
    )
    with client, output.open("a", encoding="utf-8") as out:
        for run in range(args.repeat):
            for name, message, ha_tool in CASES:
                started = time.monotonic()
                events = run_turn(client, args, name, message)
                ok = ha_call_ok(events, ha_tool)
                results[name].append(ok)
                record = {"run": run, "case": name, "ok": ok, "seconds": round(time.monotonic() - started, 1)}
                out.write(json.dumps({**record, "events": events}, ensure_ascii=True) + "\n")
                print(json.dumps(record), flush=True)

    print(json.dumps({"model": args.model, "repeat": args.repeat, **pass_k(results)}))


if __name__ == "__main__":
    main()
