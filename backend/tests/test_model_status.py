from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from app.model_status import status_from_logs


def entry(message: str, task: str, timestamp: datetime):
    return SimpleNamespace(message=message, context_ids=[task], timestamp=timestamp)


def test_model_status_is_optional_when_modal_is_not_configured():
    with TestClient(create_app()) as client:
        response = client.get("/api/model-status")
    assert response.status_code == 200
    assert response.json() == {"stage": "unavailable"}


def test_startup_status_never_exposes_private_log_data():
    now = datetime.now(UTC)
    logs = [
        entry(
            '[cold-start] {"event":"serve_entered","modal_cloud_provider":"CLOUD_PROVIDER_OCI",'
            '"modal_region":"us-east","secret":"private"}',
            "ta-1",
            now,
        ),
        entry(
            '[cold-start] {"event":"gpu","gpu":"NVIDIA L40S, 46068 MiB",'
            '"modal_cloud_provider":"CLOUD_PROVIDER_OCI","modal_region":"us-east"}',
            "ta-1",
            now,
        ),
        entry('[cold-start] {"event":"vllm_spawned"}', "ta-1", now),
    ]
    result = status_from_logs(logs)
    assert {key: result[key] for key in ("stage", "gpu", "provider", "region")} == {
        "stage": "loading",
        "gpu": "NVIDIA L40S",
        "provider": "Oracle Cloud",
        "region": "us-east",
    }
    assert "private" not in str(result)


def test_multiple_active_containers_are_not_misattributed():
    now = datetime.now(UTC)
    logs = [
        entry('[cold-start] {"event":"serve_entered"}', "ta-1", now),
        entry('[cold-start] {"event":"serve_entered"}', "ta-2", now),
    ]
    assert status_from_logs(logs) == {"stage": "waiting"}


def test_snapshot_restore_and_ready_are_reported_from_real_events():
    now = datetime.now(UTC)
    logs = [
        entry("Restoring Function from memory snapshot.", "ta-new", now),
        entry(
            '[cold-start] {"event":"restored","modal_cloud_provider":"CLOUD_PROVIDER_AWS",'
            '"modal_region":"us-east"}',
            "ta-new",
            now,
        ),
        entry('[cold-start] {"event":"awake"}', "ta-new", now),
        entry('[cold-start] {"event":"gpu","gpu":"NVIDIA L40S, 46068 MiB"}', "ta-new", now),
    ]
    result = status_from_logs(logs)
    assert {key: result[key] for key in ("stage", "gpu", "provider", "region")} == {
        "stage": "ready",
        "gpu": "NVIDIA L40S",
        "provider": "AWS",
        "region": "us-east",
    }


def test_irrelevant_or_malformed_log_lines_are_ignored():
    now = datetime.now(UTC)
    logs = [entry("user text: secret", "ta-1", now), entry("[cold-start] {not json", "ta-1", now)]
    assert status_from_logs(logs) == {"stage": "waiting"}


def test_unexpected_event_types_are_ignored():
    now = datetime.now(UTC)
    logs = [
        entry('[cold-start] {"event":[],"modal_cloud_provider":{}}', "ta-1", now),
        entry('[cold-start] {"event":"gpu","modal_cloud_provider":{}}', "ta-1", now),
    ]
    assert status_from_logs(logs) == {"stage": "waiting"}


def test_creation_wake_is_not_reported_as_a_snapshot_restore():
    now = datetime.now(UTC)
    logs = [
        entry('[cold-start] {"event":"sleeping"}', "ta-new", now),
        entry('[cold-start] {"event":"restored","snapshot_task_id":"ta-new"}', "ta-new", now),
    ]
    assert status_from_logs(logs)["stage"] == "creating_snapshot"


def test_restore_is_identified_by_creator_container():
    now = datetime.now(UTC)
    logs = [entry('[cold-start] {"event":"restored","snapshot_task_id":"ta-old"}', "ta-new", now)]
    assert status_from_logs(logs)["stage"] == "restoring"


def test_status_does_not_cross_chat_runs_or_guess_when_requests_overlap(monkeypatch):
    import time

    app = create_app()
    calls = []

    async def fake_status(started):
        calls.append(started)
        return {"stage": "loading", "gpu": "NVIDIA L4"}

    monkeypatch.setattr("app.main.get_model_status", fake_status)
    with TestClient(app) as client:
        assert client.get("/api/model-status?message_id=new").json() == {"stage": "unavailable"}
        app.state.model_requests["token-one"] = ("old", time.time())
        assert client.get("/api/model-status?message_id=new").json() == {"stage": "waiting"}
        assert not calls
        app.state.model_requests["token-two"] = ("new", time.time())
        assert client.get("/api/model-status?message_id=new").json() == {"stage": "waiting"}
        assert not calls
        del app.state.model_requests["token-one"]
        assert client.get("/api/model-status?message_id=new").json() == {
            "stage": "loading",
            "gpu": "NVIDIA L4",
        }
        assert len(calls) == 1


def test_status_rechecks_if_another_run_starts_while_logs_are_fetched(monkeypatch):
    import time

    app = create_app()
    app.state.model_requests["first"] = ("message-one", time.time())

    async def fake_status(_started):
        app.state.model_requests["second"] = ("message-two", time.time())
        return {"stage": "loading", "gpu": "NVIDIA L4"}

    monkeypatch.setattr("app.main.get_model_status", fake_status)
    with TestClient(app) as client:
        assert client.get("/api/model-status?message_id=message-one").json() == {"stage": "waiting"}


def test_model_loading_shows_safe_progress_and_source():
    now = datetime.now(UTC)
    logs = [
        entry(
            '[cold-start] {"event":"serve_entered","modal_cloud_provider":"CLOUD_PROVIDER_OCI",'
            '"modal_region":"us-east"}',
            "ta-1",
            now,
        ),
        entry(
            '[cold-start] {"event":"cache_state","safetensors":{"exists":true,'
            '"files":4,"bytes":19306310880}}',
            "ta-1",
            now,
        ),
        entry('[cold-start] {"event":"gpu","gpu":"NVIDIA L40S, 46068 MiB"}', "ta-1", now),
        entry(
            "[vllm +42.5s] Loading safetensors checkpoint shards: "
            "50% Completed | 2/4 [00:04<00:04, 2.3s/it]",
            "ta-1",
            now,
        ),
    ]
    result = status_from_logs(logs)
    assert result["stage"] == "loading"
    assert result["detail"] == "Loading model weights: 2 of 4 shards"
    assert result["weights_source"] == "Cached on Modal Volume"
    assert result["gpu"] == "NVIDIA L40S"
    assert result["gpu_memory_mib"] == 46068
    assert any(
        event["label"] == "Loading model weights: 2 of 4 shards" for event in result["events"]
    )
    assert "ta-1" not in str(result)


def test_forged_vllm_lines_do_not_appear_in_telemetry():
    now = datetime.now(UTC)
    logs = [entry("[vllm +4.0s] some private data: secret-token", "ta-1", now)]
    assert status_from_logs(logs) == {"stage": "waiting"}


def test_service_stats_are_available_even_if_logs_fail(monkeypatch):
    import asyncio
    import sys
    import time
    from types import SimpleNamespace

    from app.model_status import get_model_status

    async def lookup(_name):
        raise OSError("logs unavailable")

    async def stats():
        return SimpleNamespace(backlog=2, num_total_runners=1, num_running_inputs=0)

    server = SimpleNamespace(serve=SimpleNamespace(get_current_stats=SimpleNamespace(aio=stats)))
    fake_modal = SimpleNamespace(
        App=SimpleNamespace(lookup=SimpleNamespace(aio=lookup)),
        Cls=SimpleNamespace(from_name=lambda *_args: lambda: server),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setenv("MODAL_TOKEN_ID", "test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test")
    result = asyncio.run(get_model_status(time.time() - 5))
    assert result == {
        "stage": "unavailable",
        "service": {"queued": 2, "containers": 1, "running": 0},
    }


def test_quantization_is_shown_only_when_reported():
    now = datetime.now(UTC)
    logs = [
        entry(
            "[vllm +14.5s] non-default args: {'quantization': 'bitsandbytes', 'private': 'x'}",
            "ta-1",
            now,
        ),
    ]
    result = status_from_logs(logs)
    assert result["quantization"] == "BitsAndBytes 4-bit"
    assert "private" not in str(result)
