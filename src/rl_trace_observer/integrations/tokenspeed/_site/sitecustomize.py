"""Loaded by every Python process of `tokenspeed serve` through PYTHONPATH; see ..weight_group."""

try:
    from rl_trace_observer.integrations.tokenspeed.weight_group import install

    install()
except Exception as error:  # never break the server's interpreter startup
    import sys

    print(f"rl-trace-observer: weight-group patch not installed: {error!r}", file=sys.stderr)
