"""POST /api/codex/tasks/{pause,resume}: a scoped API token pauses and resumes a
built-in task by action name, so automation outside the UI can keep the GPU quiet.
"""
import asyncio
import tempfile

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from starlette.requests import Request

from tests.helpers.import_state import clear_fake_database_modules

clear_fake_database_modules()

import core.database as cdb
from core.database import ScheduledTask
import routes.codex_routes as codex_routes

_TMPDB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_ENGINE = create_engine(
    f"sqlite:///{_TMPDB.name}",
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)
cdb.Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)


@pytest.fixture(autouse=True)
def _db(monkeypatch):
    monkeypatch.setattr(codex_routes, "SessionLocal", _TS)
    db = _TS()
    try:
        db.query(ScheduledTask).delete()
        db.add(ScheduledTask(
            id="t1", owner="alice", name="Email Summary", prompt="",
            task_type="action", action="summarize_emails", trigger_type="schedule",
            schedule="cron", cron_expression="0 */2 * * *", status="active",
            output_target="session",
        ))
        db.commit()
    finally:
        db.close()


def _endpoint(path):
    for route in codex_routes.setup_codex_routes().routes:
        if route.path == path and "POST" in route.methods:
            return route.endpoint
    raise AssertionError(f"POST {path} route not found")


def _request(scopes):
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": [], "state": {}})
    request.state.api_token = True
    request.state.api_token_owner = "alice"
    request.state.api_token_scopes = list(scopes)
    return request


def _call(verb, scopes=("tasks:write",), action="summarize_emails"):
    return asyncio.run(_endpoint(f"/api/codex/tasks/{verb}")(_request(scopes), action))


def _status():
    db = _TS()
    try:
        return db.query(ScheduledTask).filter(ScheduledTask.id == "t1").first().status
    finally:
        db.close()


@pytest.mark.parametrize("scopes", [(), ("tasks:read",), ("chat",)])
def test_requires_tasks_write(scopes):
    with pytest.raises(HTTPException) as exc:
        _call("pause", scopes)

    assert exc.value.status_code == 403
    assert _status() == "active"


def test_pause_then_resume():
    assert _call("pause")["status"] == "paused"
    assert _status() == "paused"

    resumed = _call("resume")

    assert resumed["status"] == "active" and resumed["next_run"]
    assert _status() == "active"


@pytest.mark.parametrize("action", ["no_such_task", "run_local"])
def test_only_builtin_actions(action):
    db = _TS()
    try:
        db.add(ScheduledTask(
            id="t2", owner="alice", name="Shell", prompt="", task_type="action",
            action="run_local", trigger_type="schedule", schedule="cron",
            cron_expression="0 * * * *", status="paused", output_target="session",
        ))
        db.commit()
    finally:
        db.close()

    with pytest.raises(HTTPException) as exc:
        _call("resume", action=action)

    assert exc.value.status_code == 404
