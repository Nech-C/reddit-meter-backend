# Makefile — Reddit Meter
runner-test:
	APP_ENV=test PYTHONPATH=. uv run python app/jobs/runner.py --method hot --posts 3 --comments 3

runner-dev:
	APP_ENV=dev PYTHONPATH=. uv run python app/jobs/runner.py --method hot --posts 15 --comments 5

# Format code
format:
	uv run black .

# Lint code (optional)
lint:
	ruff check . --extend-exclude '*.ipynb'

lint-fix:
	ruff check . --fix

build-pipeline:
	docker build --progress=plain -t reddit-meter-pipeline -f Dockerfile.pipeline .

test-pipeline:
	docker run \
		--env-file .env.test \
		-v $(pwd)/creds.json:/app/creds.json \
		-v $(pwd)/test_subreddits.json:/app/test_subreddits.json \
		reddit-meter-pipeline \
		make runner-test

push-pipeline:
	docker tag reddit-meter-pipeline \
  	us-central1-docker.pkg.dev/reddit-sentiment-meter/reddit-meter/reddit-meter-pipeline

	docker push us-central1-docker.pkg.dev/reddit-sentiment-meter/reddit-meter/reddit-meter-pipeline

