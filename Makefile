# Makefile — Reddit Meter
test-runner:
	APP_ENV=test PYTHONPATH=. uv run python app/jobs/runner.py --method hot --posts 3 --comments 3 

# Format code
format:
	uv run black .

# Lint code (optional)
lint:
	ruff check . --extend-exclude '*.ipynb'

lint-fix:
	ruff check . --fix
