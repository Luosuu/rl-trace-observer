"""Loaded by every Python process of `tokenspeed serve` through PYTHONPATH.

See ..weight_group, ..proton_graphs and ..profile_saving.
"""

import sys

for _module in ("weight_group", "proton_graphs", "profile_saving"):
    try:
        __import__(f"rl_trace_observer.integrations.tokenspeed.{_module}", fromlist=["install"]).install()
    except Exception as error:  # never break the server's interpreter startup
        print(f"rl-trace-observer: {_module} patch not installed: {error!r}", file=sys.stderr)
