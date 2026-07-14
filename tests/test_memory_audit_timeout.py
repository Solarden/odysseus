from pathlib import Path


def test_memory_audit_uses_its_own_llm_timeout():
    source = Path("app.py").read_text()
    start = source.index("_TIMEOUT_EXEMPT_PREFIXES =")
    end = source.index("\n)\n", start)
    timeout_exemptions = source[start:end]

    assert '"/api/memory/audit"' in timeout_exemptions


def test_sync_chat_is_timeout_exempt():
    # /api/v1/chat (cron sync-chat) must be exempt: a cold model load can exceed
    # REQUEST_HARD_TIMEOUT and 504 before the response with session_id reaches the
    # client, orphaning the session. NOTE "/api/chat" does NOT cover "/api/v1/chat".
    source = Path("app.py").read_text()
    start = source.index("_TIMEOUT_EXEMPT_PREFIXES =")
    end = source.index("\n)\n", start)
    timeout_exemptions = source[start:end]

    assert '"/api/v1/chat"' in timeout_exemptions
