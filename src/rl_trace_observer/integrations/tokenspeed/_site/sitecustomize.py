"""Loaded by every Python process of `tokenspeed serve` through PYTHONPATH; see ..weight_group and ..proton_graphs."""

import sys

for _module in ("weight_group", "proton_graphs"):
    try:
        __import__(f"rl_trace_observer.integrations.tokenspeed.{_module}", fromlist=["install"]).install()
    except Exception as error:  # never break the server's interpreter startup
        print(f"rl-trace-observer: {_module} patch not installed: {error!r}", file=sys.stderr)
