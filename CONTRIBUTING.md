# Contributing

Use Python 3.11+ and a virtual environment. Install with `python -m pip install -e ".[dev]"`, run `python -m pytest`, and optionally `ruff check src tests`. Keep production runtime lightweight and offline; place training-only packages in the `training` extra. Add deterministic synthetic tests for image logic. Never add captured game/user screenshots, model weights, credentials, or personal datasets. Keep capture read-only and do not add real-game input execution.

By contributing, you agree that contributions are offered under this project's AGPL-3.0-only license.
