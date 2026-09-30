import sys
import types

import pytest

from rl_trace_observer.context import RUN_ID_ENV, SCHEMA_VERSION, TraceContext, current_run_id


def _fake_ray(monkeypatch, *, initialized=True, session="session_2026-09-30_17-36-58_296392_620", job="01000000"):
    context = types.SimpleNamespace(get_session_name=lambda: session, get_job_id=lambda: job)
    ray = types.SimpleNamespace(is_initialized=lambda: initialized, get_runtime_context=lambda: context)
    monkeypatch.setitem(sys.modules, "ray", ray)


def test_run_id_comes_from_the_ray_session_and_job(monkeypatch):
    monkeypatch.delenv(RUN_ID_ENV, raising=False)
    _fake_ray(monkeypatch)

    assert current_run_id() == "session_2026-09-30_17-36-58_296392_620-job-01000000"


def test_run_id_env_overrides_ray(monkeypatch):
    monkeypatch.setenv(RUN_ID_ENV, "ppo-run-001")
    _fake_ray(monkeypatch)

    assert current_run_id() == "ppo-run-001"


def test_no_run_id_outside_ray(monkeypatch):
    monkeypatch.delenv(RUN_ID_ENV, raising=False)
    _fake_ray(monkeypatch, initialized=False)

    assert current_run_id() is None


def test_context_round_trips_and_names_its_profile_session(monkeypatch):
    monkeypatch.setenv(RUN_ID_ENV, "ppo-run-001")
    context = TraceContext.current(global_step=42, role="actor", framework_rank=3)

    data = context.to_json()
    assert data["profile_session_id"] == "ppo-run-001-step-42"
    assert TraceContext.from_json(data) == context
    assert TraceContext(run_id="ppo-run-001").profile_session_id is None


def test_context_of_another_schema_version_is_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        TraceContext.from_json({"schema_version": SCHEMA_VERSION + 1, "run_id": "x"})


def test_callers_can_supply_a_propagated_run_id(monkeypatch):
    # e.g. a rollout server outside Ray, given the run id by the trainer.
    monkeypatch.delenv(RUN_ID_ENV, raising=False)

    context = TraceContext.current(run_id="ppo-run-001", global_step=3)

    assert context.run_id == "ppo-run-001"
    assert context.profile_session_id == "ppo-run-001-step-3"
