# Contributing

Thank you for investing time in Sleep Pattern Storyboard!

## Development environment

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install '.[dev]'
   ```

2. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

3. Build the curated dataset to exercise analytics and app flows:

   ```bash
   python -m sleep_analysis.cli ingest --source oura --path data/raw/vendor_samples/oura.csv --out data/processed/parquet
   python -m sleep_analysis.cli build-features --parquet-dir data/processed/parquet --hrv data/raw/signals/hrv_sample.csv --steps data/raw/signals/steps_sample.csv --tags data/raw/signals/tags_sample.csv --screen-time data/raw/signals/screen_time_sample.csv --out data/processed/features
   ```

## Coding guidelines

- Run `pre-commit run -a` before opening a PR.
- Keep unit tests green (`pytest -q`).
- Favour small, focused commits with descriptive messages.
- Surface telemetry impacts when adding new CLI commands.

## Pull request checklist

- [ ] Tests written/updated
- [ ] Lint passes locally (`pre-commit run -a`)
- [ ] README updated for user-facing changes
- [ ] Sensitive data checked (use `--privacy local-only` if needed)

Happy shipping! 🌙
