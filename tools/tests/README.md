tools/tests/ — pytest suites (numpy only).

Run:  uv run --no-project --with numpy --with pytest python -m pytest tools/tests -q

test_exl3_prefill_ref.py — the EXL3 prefill oracle's tests: the decode vs the vendored
numpy decoder tile for tile, the 16/64-row block row agreement, the rotation's
self-inverse, the per-row svh scale applied once, the roundings, and the CLI.
