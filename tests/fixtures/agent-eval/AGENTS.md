# AGENTS.md — playground

A tiny dependency-free Python geometry package.

- Layout: `src/shapes/` holds one module per shape (`circle.py`, `rect.py`,
  `polygon.py`) with the exports listed alphabetically in `__init__.py`;
  `src/cli.py` is the command line; `tests/` is unittest; `docs/` are notes;
  `data/` are fixtures for the file tools.
- Conventions: functions are pure and take floats; validate inputs and raise
  `ValueError` with a short lowercase message; every module and function has
  a one-line docstring; files end with a newline.
- Tests: `make test` (`python3 -m unittest discover -s tests`). There is no
  pytest.
- Do not edit `data/`; `scripts/gen_fixtures.py` generates it.
