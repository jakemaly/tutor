"""Checks for cold-start timing helpers; they run without Modal, Docker, or a GPU."""

import json

import pytest
from cold_start_benchmark import compute_stages, describe_trial, read_env, snapshot_mode
from cold_start_timing import (
    classify_data_stream_line,
    classify_sse_data,
    engine_metrics,
    metric_total,
    parse_timing_lines,
    parse_vllm_milestones,
    process_clock,
    received_network_bytes,
    summarize,
    timing_line,
)


def test_timing_line_records_ids_but_never_identity_token(monkeypatch):
    monkeypatch.setenv("MODAL_TASK_ID", "ta-123")
    monkeypatch.setenv("MODAL_IDENTITY_TOKEN", "secret-token")
    line = timing_line("serve_entered", rx_bytes=5)

    assert "secret-token" not in line
    [event] = parse_timing_lines(
        [f"2026-11-14T19:00:00Z ta-123 {line}", "unrelated", "[cold-start] {"]
    )
    assert event["event"] == "serve_entered"
    assert event["modal_task_id"] == "ta-123"
    assert event["rx_bytes"] == 5


def test_vllm_milestones_use_first_occurrence_and_reported_durations():
    lines = [
        "[vllm +31.200s] INFO non-default args: {...}",
        "[vllm +60.000s] (EngineCore pid=9) INFO Loading weights took 12.50 seconds",
        "[vllm +61.000s] INFO Model loading took 6.1 GiB memory and 14.25 seconds",
        "[vllm +90.000s] INFO init engine (profile, create kv cache, warmup model) took 29.00 s",
        "[vllm +99.000s] INFO Application startup complete.",
        "[vllm +120.000s] INFO Application startup complete.",
    ]
    milestones = parse_vllm_milestones(lines)

    assert milestones["cli_imported"] == {"offset": 31.2, "reported_seconds": None}
    assert milestones["weights_loaded"]["reported_seconds"] == 12.5
    assert milestones["model_loaded"] == {"offset": 61.0, "reported_seconds": 14.25}
    assert milestones["engine_initialized"]["reported_seconds"] == 29.0
    assert milestones["api_started"]["offset"] == 99.0


def test_proc_helpers_parse_network_and_process_start():
    net_dev = (
        "Inter-|   Receive\n face |bytes packets\n"
        "    lo: 900 1 0 0 0 0 0 0 900 1 0 0 0 0 0 0\n"
        "  eth0: 1500 2 0 0 0 0 0 0 10 1 0 0 0 0 0 0\n"
    )
    assert received_network_bytes(net_dev) == 1500

    # The command name contains spaces and a parenthesis; starttime is 250 ticks after boot.
    stat = "42 (python (x) y) S " + " ".join(["0"] * 18) + " 250 0 0"
    clock = process_clock(stat, "10.00 20.00", now_wall=1000.0, ticks_per_second=100)
    assert clock == {"boot_wall": 990.0, "process_start_wall": 992.5}


def test_metric_total_sums_label_sets_for_exact_name():
    metrics = "\n".join(
        [
            "# HELP vllm:time_to_first_token_seconds TTFT",
            'vllm:time_to_first_token_seconds_count{engine="0"} 1.0',
            'vllm:time_to_first_token_seconds_sum{engine="0"} 0.4',
            'vllm:time_to_first_token_seconds_sum{engine="1"} 0.1',
        ]
    )
    assert metric_total(metrics, "vllm:time_to_first_token_seconds_sum") == pytest.approx(0.5)
    assert metric_total(metrics, "vllm:time_to_first_token_seconds_count") == 1.0
    assert metric_total(metrics, "vllm:missing") is None


@pytest.mark.parametrize(
    ("delta", "finish", "kind"),
    [
        ({"role": "assistant", "content": ""}, None, "role"),
        ({"content": "A"}, None, "content"),
        ({"tool_calls": [{"index": 0}]}, None, "tool_call"),
        ({"reasoning_content": "hm"}, None, "reasoning"),
        ({}, "stop", "finish"),
    ],
)
def test_sse_classification_separates_role_only_chunks(delta, finish, kind):
    data = json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]})
    assert classify_sse_data(data) == kind


def test_sse_classification_handles_done_and_bad_payloads():
    assert classify_sse_data(" [DONE]") == "done"
    assert classify_sse_data("not json") == "unparsed"


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ('0:"Hello"', "text"),
        ('0:""', "empty_text"),
        ('b:{"toolCallId":"1","toolName":"load_skill"}', "tool_call_begin"),
        ('a:{"toolCallId":"1","result":{}}', "tool_result"),
        ('3:"failed"', "error"),
        ('d:{"finishReason":"stop"}', "finish"),
        ("", "keepalive"),
    ],
)
def test_data_stream_classification(line, kind):
    assert classify_data_stream_line(line)[0] == kind


def synthetic_trial() -> dict:
    """A trial whose container clock reads 2 s ahead of the local clock's NTP-corrected time."""
    t0_wall = 1000.0
    local_minus_ntp = 0.5  # local clock is 0.5 s fast; true t0 is 999.5
    serve_wall = 999.5 + 40.0
    return {
        "label": "baseline",
        "trial": 1,
        "t0_wall": t0_wall,
        "clock": {"local_minus_ntp": local_minus_ntp},
        "containers_before": 0,
        "new_task_id": True,
        "client": {"first": {"text": 200.0}, "errors": []},
        "relay": [{"received": 0.05, "first": {"role": 199.0, "content": 199.5}}],
        "container_events": [
            {
                "event": "serve_entered",
                "wall": serve_wall,
                "mono": 500.0,
                "process_start_wall": 999.5 + 35.0,
                "rx_bytes": 1_000_000,
                "modal_task_id": "ta-1",
            },
            {"event": "cache_state", "mono": 500.1, "gguf": {"exists": True}},
            {"event": "tokenizer_config_ready", "mono": 501.0},
            {"event": "vllm_spawned", "mono": 501.0, "spawned_mono": 501.0},
            {"event": "gpu", "mono": 501.1, "gpu": "NVIDIA L4, 23034 MiB"},
            {"event": "vllm_ready", "mono": 640.0, "rx_bytes": 3_000_000},
            {"event": "first_token_observed", "mono": 658.0, "ttft_seconds": 1.2},
        ],
        "vllm_milestones": {
            "cli_imported": {"offset": 20.0, "reported_seconds": None},
            "model_loaded": {"offset": 80.0, "reported_seconds": 30.0},
            "engine_initialized": {"offset": 130.0, "reported_seconds": 50.0},
        },
    }


def test_stages_telescope_to_time_to_first_text():
    stages = {stage["key"]: stage["seconds"] for stage in compute_stages(synthetic_trial())}

    assert stages["local_backend"] == 0.05
    assert stages["provisioning"] == pytest.approx(34.95)
    assert stages["runtime_init"] == pytest.approx(5.0)
    assert stages["vllm_import"] == pytest.approx(20.0)
    assert stages["engine_spawn"] == pytest.approx(30.0)  # 50 s load start minus 20 s import
    assert stages["model_load"] == pytest.approx(30.0)
    assert stages["engine_init"] == pytest.approx(50.0)
    assert stages["backend_stream"] == pytest.approx(0.5)
    assert sum(stages.values()) == pytest.approx(200.0)


def test_describe_trial_reports_gpu_cache_and_cold_checks():
    summary = describe_trial(synthetic_trial())

    assert summary["gpu"].startswith("NVIDIA L4")
    assert summary["ttft_seconds"] == 200.0
    assert all(summary["cold"].values())
    # The synthetic trial has the historical GGUF cache fields.
    assert summary["cache"]["weights_cached"] is True
    assert summary["cache"]["weights_format"] == "gguf"
    assert summary["cache"]["network_mb_until_ready"] == 2.0
    assert summary["tool_call_before_answer"] is False


def test_safetensors_trial_uses_model_files_event_and_weight_cache():
    """Safetensors runs log model_files_ready and a safetensors cache entry instead."""
    trial = synthetic_trial()
    for event in trial["container_events"]:
        if event["event"] == "cache_state":
            event.pop("gguf")
            event["safetensors"] = {"exists": False, "reason": "No Safetensors weights"}
        elif event["event"] == "tokenizer_config_ready":
            event["event"] = "model_files_ready"

    stages = {stage["key"]: stage["seconds"] for stage in compute_stages(trial)}
    summary = describe_trial(trial)

    assert stages["tokenizer_sync"] == pytest.approx(0.9)  # cache_state 500.1 -> files 501.0
    assert sum(stages.values()) == pytest.approx(200.0)
    assert summary["cache"]["weights_format"] == "safetensors"
    assert summary["cache"]["weights_cached"] is False


def test_missing_stage_boundary_is_reported_not_guessed():
    trial = synthetic_trial()
    del trial["vllm_milestones"]["cli_imported"]
    stages = {stage["key"]: stage["seconds"] for stage in compute_stages(trial)}

    assert stages["vllm_import"] is None
    assert stages["engine_spawn"] is None
    assert stages["model_load"] == pytest.approx(30.0)


def test_read_env_strips_inline_comments(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# note\nLLM_BASE_URL=https://x.modal.run/v1 # comment\nLLM_MODEL='m'\n")
    assert read_env(env) == {"LLM_BASE_URL": "https://x.modal.run/v1", "LLM_MODEL": "m"}


def test_summarize_reports_median_and_range():
    assert summarize([3.0, 1.0, None, 2.0]) == {
        "n": 3,
        "median": 2.0,
        "min": 1.0,
        "max": 3.0,
    }
    assert summarize([]) is None


def test_log_prefix_task_id_wins_over_the_process_environment(monkeypatch):
    # After a snapshot restore the process environment can still hold the creator's task ID.
    monkeypatch.setenv("MODAL_TASK_ID", "ta-CREATOR")
    line = timing_line("restored", snapshot_task_id="ta-CREATOR")
    [event] = parse_timing_lines([f"2026-11-14T20:00:00Z ta-RESTORED1 {line}"])

    assert event["modal_task_id"] == "ta-CREATOR"
    assert event["log_task_id"] == "ta-RESTORED1"


def test_engine_metrics_subtract_warmup_baseline():
    before = "\n".join(
        [
            "vllm:time_to_first_token_seconds_count 2.0",
            "vllm:time_to_first_token_seconds_sum 0.5",
            "vllm:prompt_tokens_total 40.0",
        ]
    )
    after = "\n".join(
        [
            "vllm:time_to_first_token_seconds_count 3.0",
            "vllm:time_to_first_token_seconds_sum 0.8",
            "vllm:prompt_tokens_total 657.0",
        ]
    )
    baseline = engine_metrics(before)
    assert baseline["requests"] == 2.0
    assert engine_metrics(before, baseline)["requests"] == 0

    first = engine_metrics(after, baseline)
    assert first["requests"] == 1.0
    assert first["ttft_seconds"] == pytest.approx(0.3)
    assert first["prompt_tokens"] == 617.0
    assert first["queue_seconds"] is None
    assert engine_metrics("")["requests"] == 0


def restored_trial() -> dict:
    """A restored trial; the container's monotonic clock is unrelated to the snapshot's."""
    return {
        "label": "snapshot",
        "trial": 2,
        "t0_wall": 2000.0,
        "clock": {"local_minus_ntp": 0.0},
        "containers_before": 0,
        "new_task_id": True,
        "client": {"first": {"text": 30.0}, "errors": []},
        "relay": [{"received": 0.04, "first": {"role": 29.5, "content": 29.6}}],
        "container_events": [
            {
                "event": "restored",
                "wall": 2012.0,
                "mono": 9000.0,
                "rx_bytes": 100,
                "snapshot_id": "abc",
                "snapshot_task_id": "ta-CREATOR",
                "log_task_id": "ta-NEW",
            },
            {"event": "awake", "mono": 9015.0, "rx_bytes": 100},
            {"event": "gpu", "mono": 9015.1, "gpu": "NVIDIA L40S, 46068 MiB"},
            {"event": "first_token_observed", "mono": 9017.0, "requests": 1.0},
        ],
        "vllm_milestones": {},
        "snapshot_log": ["ta-NEW Restoring Function from memory snapshot."],
    }


def test_restored_trial_uses_restore_stages_that_sum_to_ttft():
    trial = restored_trial()
    stages = {stage["key"]: stage["seconds"] for stage in compute_stages(trial)}

    assert snapshot_mode(trial) == "restored"
    assert list(stages) == [
        "local_backend",
        "provisioning_restore",
        "wake_up",
        "first_token",
        "return_path",
        "backend_stream",
    ]
    assert stages["provisioning_restore"] == pytest.approx(11.96)
    assert stages["wake_up"] == pytest.approx(15.0)
    assert stages["first_token"] == pytest.approx(2.0)
    assert sum(stages.values()) == pytest.approx(30.0)


def test_describe_restored_trial_verifies_restore_from_another_container():
    summary = describe_trial(restored_trial())

    assert summary["task_id"] == "ta-NEW"
    assert summary["snapshot"]["mode"] == "restored"
    assert summary["snapshot"]["restored_from_other_container"] is True
    assert summary["cold"] == {
        "no_containers_before_request": True,
        "new_task_id": True,
        "restore_began_after_request": True,
    }
    assert summary["cache"]["network_mb_until_ready"] == 0.0


def test_snapshot_creation_trial_includes_capture_stages():
    trial = synthetic_trial()
    trial["container_events"] = trial["container_events"][:-1] + [
        {"event": "warmup_done", "mono": 642.0},
        {"event": "sleeping", "mono": 645.0},
        # A different monotonic origin after the restore must not shift the stages.
        {"event": "restored", "wall": 999.5 + 200.0, "mono": 10.0, "snapshot_task_id": "ta-1"},
        {"event": "awake", "mono": 12.0},
        {"event": "first_token_observed", "mono": 13.0},
    ]
    for event in trial["container_events"]:
        event["log_task_id"] = "ta-1"
    trial["client"]["first"]["text"] = 215.0
    trial["relay"][0]["first"]["content"] = 214.5
    stages = {stage["key"]: stage["seconds"] for stage in compute_stages(trial)}
    summary = describe_trial(trial)

    assert snapshot_mode(trial) == "created"
    assert stages["warmup"] == pytest.approx(2.0)
    assert stages["sleep"] == pytest.approx(3.0)
    assert stages["snapshot"] == pytest.approx(200.0 - 185.0)  # sleeping at 40 + 145 s
    assert stages["wake_up"] == pytest.approx(2.0)
    assert stages["first_token"] == pytest.approx(1.0)
    assert sum(stages.values()) == pytest.approx(215.0)
    assert summary["snapshot"]["restored_from_other_container"] is False
