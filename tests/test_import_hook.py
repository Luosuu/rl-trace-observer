"""Post-import hooks: several patches of one module all run when it is imported."""

import sys
import textwrap

from rl_trace_observer.integrations.verl.import_hook import when_imported


def test_every_callback_of_a_module_runs_once_in_order(tmp_path, monkeypatch):
    (tmp_path / "hooked_module.py").write_text(textwrap.dedent("VALUE = 1\n"))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "hooked_module", raising=False)
    calls = []

    def first(module):
        calls.append(("first", module.VALUE))

    def second(module):
        calls.append(("second", module.VALUE))

    when_imported("hooked_module", first)
    when_imported("hooked_module", second)
    when_imported("hooked_module", first)
    import hooked_module  # noqa: F401

    assert calls == [("first", 1), ("second", 1)]
    # Once imported, a new callback runs right away.
    when_imported("hooked_module", lambda module: calls.append(("late", module.VALUE)))
    assert calls[-1] == ("late", 1)


def test_the_plugin_patches_verl_when_it_registers():
    import subprocess

    code = (
        "import rl_trace_observer.integrations.verl.register\n"
        "from verl.checkpoint_engine import CheckpointEngineManager\n"
        "assert getattr(CheckpointEngineManager.update_weights, '_rl_trace_observer_records_weight_updates', False)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
