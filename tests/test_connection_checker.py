"""Tests for app.services.health.connection_checker."""

from unittest.mock import MagicMock, patch

import requests

from app.services.health.connection_checker import (
    ConnectionResult,
    _check_groq_llm,
    _check_langsmith,
    _check_logfire,
    _check_portkey_gateway,
    _check_qdrant,
    _check_supabase_postgres,
    _print_cli_report,
    check_all_connections,
    log_connection_summary,
)


def _ok(name: str) -> ConnectionResult:
    return ConnectionResult(name, True, "ok")


def _fail(name: str, message: str = "fail") -> ConnectionResult:
    return ConnectionResult(name, False, message)


# CONNECTION RESULT DATA CLASS TESTS
def test_connection_result_to_dict_ok():
    """Verify dictionary formatting for healthy results."""
    res = ConnectionResult("test_service", True, "all good")
    d = res.to_dict()
    assert d["healthy"] is True
    assert d["message"] == "all good"
    assert d["status"] == "ok: all good"


def test_connection_result_to_dict_unavailable():
    """Verify dictionary formatting for unhealthy results."""
    res = ConnectionResult("test_service", False, "timeout error")
    d = res.to_dict()
    assert d["healthy"] is False
    assert d["message"] == "timeout error"
    assert d["status"] == "unavailable: timeout error"


# SUPABASE POSTGRES TESTS
def test_check_supabase_postgres_success():
    """Verify healthy status when Postgres pool connects and executes query."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_pool.getconn.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with (
        patch(
            "app.services.health.connection_checker.ConnectionPool",
            return_value=mock_pool,
        ),
        patch(
            "app.services.health.connection_checker.settings.SUPABASE_URI",
            "postgresql://user:pass@localhost:5432/db",
        ),
    ):
        result = _check_supabase_postgres()

    assert result.healthy is True
    assert result.name == "postgres"
    assert "reachable" in result.message
    mock_cursor.execute.assert_called_once_with("SELECT 1")
    mock_pool.putconn.assert_called_once_with(mock_conn)
    mock_pool.close.assert_called_once()


def test_check_supabase_postgres_missing_uri():
    """Verify failure when SUPABASE_URI environment variable is missing."""
    with patch("app.services.health.connection_checker.settings.SUPABASE_URI", None):
        result = _check_supabase_postgres()

    assert result.healthy is False
    assert result.name == "postgres"
    assert "SUPABASE_URI not set" in result.message


def test_check_supabase_postgres_failure():
    """Verify unhealthy status when Postgres pool throws a connection exception."""
    with (
        patch(
            "app.services.health.connection_checker.ConnectionPool",
            side_effect=Exception("Connection refused by Supabase"),
        ),
        patch(
            "app.services.health.connection_checker.settings.SUPABASE_URI",
            "postgresql://user:pass@localhost:5432/db",
        ),
    ):
        result = _check_supabase_postgres()

    assert result.healthy is False
    assert result.name == "postgres"
    assert "Connection refused by Supabase" in result.message


# QDRANT VECTOR DB TESTS
def test_check_qdrant_success():
    """Verify healthy status when Qdrant client fetches collections successfully."""
    mock_client = MagicMock()
    with (
        patch(
            "app.services.health.connection_checker.QdrantClient",
            return_value=mock_client,
        ),
        patch(
            "app.services.health.connection_checker.settings.QDRANT_URL",
            "https://qdrant.example.com",
        ),
    ):
        result = _check_qdrant()

    assert result.healthy is True
    assert result.name == "qdrant"
    mock_client.get_collections.assert_called_once()


def test_check_qdrant_missing_url():
    """Verify failure when QDRANT_URL environment variable is missing."""
    with patch("app.services.health.connection_checker.settings.QDRANT_URL", None):
        result = _check_qdrant()

    assert result.healthy is False
    assert result.name == "qdrant"
    assert "QDRANT_URL not set" in result.message


def test_check_qdrant_failure():
    """Verify unhealthy status when Qdrant client raises an exception."""
    with (
        patch(
            "app.services.health.connection_checker.QdrantClient",
            side_effect=Exception("Unauthorized vector DB access"),
        ),
        patch(
            "app.services.health.connection_checker.settings.QDRANT_URL",
            "https://qdrant.example.com",
        ),
    ):
        result = _check_qdrant()

    assert result.healthy is False
    assert "Unauthorized vector DB access" in result.message


# PORTKEY LLM GATEWAY TESTS
def test_check_portkey_gateway_success():
    """Verify healthy status when Portkey gateway completion returns valid text."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Hello"))]

    with patch(
        "app.services.health.connection_checker.portkey_client.chat.completions.create",
        return_value=mock_resp,
    ):
        result = _check_portkey_gateway()

    assert result.healthy is True
    assert result.name == "llm_gateway"


def test_check_portkey_gateway_empty_response():
    """Verify failure when Portkey gateway returns choices with no content."""
    mock_resp = MagicMock()
    mock_resp.choices = []

    with patch(
        "app.services.health.connection_checker.portkey_client.chat.completions.create",
        return_value=mock_resp,
    ):
        result = _check_portkey_gateway()

    assert result.healthy is False
    assert result.name == "llm_gateway"
    assert "empty response" in result.message


# GROQ LLM TESTS
def test_check_groq_llm_success():
    """Verify healthy status when direct HTTP call to Groq succeeds."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with (
        patch(
            "app.services.health.connection_checker.requests.post",
            return_value=mock_response,
        ),
        patch(
            "app.services.health.connection_checker.settings.GROQ_API_KEY",
            "gsk_fake_key",
        ),
    ):
        result = _check_groq_llm()

    assert result.healthy is True
    assert result.name == "groq_llm"


def test_check_groq_llm_missing_key():
    """Verify failure when GROQ_API_KEY environment variable is not set."""
    with patch("app.services.health.connection_checker.settings.GROQ_API_KEY", None):
        result = _check_groq_llm()

    assert result.healthy is False
    assert "GROQ_API_KEY not set" in result.message


def test_check_groq_llm_failure():
    """Verify unhealthy status when Groq HTTP call raises HTTPError."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")

    with (
        patch(
            "app.services.health.connection_checker.requests.post",
            return_value=mock_response,
        ),
        patch(
            "app.services.health.connection_checker.settings.GROQ_API_KEY",
            "gsk_fake_key",
        ),
    ):
        result = _check_groq_llm()

    assert result.healthy is False
    assert "401 Unauthorized" in result.message


# LOGFIRE MONITORING TESTS
def test_check_logfire_success():
    """Verify healthy status when LOGFIRE_TOKEN is present."""
    with patch(
        "app.services.health.connection_checker.settings.LOGFIRE_TOKEN",
        "logfire_token_123",
    ):
        result = _check_logfire()

    assert result.healthy is True
    assert result.name == "logfire"


def test_check_logfire_missing_token():
    """Verify failure status when LOGFIRE_TOKEN is missing."""
    with patch("app.services.health.connection_checker.settings.LOGFIRE_TOKEN", None):
        result = _check_logfire()

    assert result.healthy is False
    assert "LOGFIRE_TOKEN not set" in result.message


# LANGSMITH TRACING TESTS
def test_check_langsmith_success():
    """Verify healthy status when LangSmith /ok endpoint returns 200."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with (
        patch(
            "app.services.health.connection_checker.requests.get",
            return_value=mock_response,
        ),
        patch(
            "app.services.health.connection_checker.settings.LANGSMITH_API_KEY",
            "lsv2_fake_key",
        ),
        patch(
            "app.services.health.connection_checker.settings.LANGSMITH_ENDPOINT",
            "https://api.smith.langchain.com",
        ),
    ):
        result = _check_langsmith()

    assert result.healthy is True
    assert result.name == "langsmith"


def test_check_langsmith_missing_key():
    """Verify failure when LANGSMITH_API_KEY environment variable is missing."""
    with patch(
        "app.services.health.connection_checker.settings.LANGSMITH_API_KEY",
        None,
    ):
        result = _check_langsmith()

    assert result.healthy is False
    assert "LANGSMITH_API_KEY not set" in result.message


def test_check_langsmith_failure():
    """Verify unhealthy status when LangSmith endpoint is unreachable."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Internal Error")

    with (
        patch(
            "app.services.health.connection_checker.requests.get",
            return_value=mock_response,
        ),
        patch(
            "app.services.health.connection_checker.settings.LANGSMITH_API_KEY",
            "lsv2_fake_key",
        ),
        patch(
            "app.services.health.connection_checker.settings.LANGSMITH_ENDPOINT",
            "https://api.smith.langchain.com",
        ),
    ):
        result = _check_langsmith()

    assert result.healthy is False
    assert "500 Internal Error" in result.message


# AGGREGATOR & CLI REPORT TESTS
def test_check_all_connections_runs_all_six_checkers():
    """Verify check_all_connections aggregates all 6 external service results."""
    mock_checkers = [
        lambda: _ok("postgres"),
        lambda: _ok("qdrant"),
        lambda: _ok("llm_gateway"),
        lambda: _ok("groq_llm"),
        lambda: _ok("logfire"),
        lambda: _fail("langsmith"),
    ]

    with patch("app.services.health.connection_checker._CHECKERS", mock_checkers):
        results = check_all_connections()

    assert len(results) == 6
    assert results["postgres"].healthy is True
    assert results["langsmith"].healthy is False


def test_log_connection_summary_returns_boolean():
    """Verify log_connection_summary returns True when all healthy and False otherwise."""
    all_ok = {"p": _ok("postgres"), "q": _ok("qdrant")}
    one_failed = {"p": _ok("postgres"), "q": _fail("qdrant")}

    assert log_connection_summary(all_ok) is True
    assert log_connection_summary(one_failed) is False


def test_print_cli_report_exit_codes():
    """Verify CLI reporter returns exit code 0 on success and 1 on failure."""
    all_ok = {"p": _ok("postgres"), "q": _ok("qdrant")}
    one_failed = {"p": _ok("postgres"), "q": _fail("qdrant")}

    assert _print_cli_report(all_ok) == 0
    assert _print_cli_report(one_failed) == 1
