# Makefile — Reddit Meter
runner-test:
	APP_ENV=test PYTHONPATH=. uv run python app/jobs/runner.py --method hot --posts 3 --comments 3

runner-dev:
	APP_ENV=dev PYTHONPATH=. uv run python app/jobs/runner.py --method hot --posts 15 --comments 5

format:
	uv run black .

lint:
	ruff check . --extend-exclude '*.ipynb'

lint-fix:
	ruff check . --fix

build-pipeline:
	docker build --progress=plain -t reddit-meter-pipeline -f Dockerfile.pipeline .

test-pipeline:
	docker run \
		--env-file .env.test \
		-v $(PWD)/creds.json:/app/creds.json \
		-v $(PWD)/test_subreddits.json:/app/test_subreddits.json \
		reddit-meter-pipeline \
		make runner-test

push-pipeline:
	docker tag reddit-meter-pipeline \
  	us-central1-docker.pkg.dev/reddit-sentiment-meter/reddit-meter/reddit-meter-pipeline

	docker push us-central1-docker.pkg.dev/reddit-sentiment-meter/reddit-meter/reddit-meter-pipeline

test-api-local:
	PYTHONPATH=. uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8080

build-api:
	docker build --progress=plain -t reddit-meter-api -f Dockerfile.api .

test-api-image:
	docker run \
		--rm \
		--env-file .env.test \
		-v ${PWD}/creds.json:/reddit-meter-api/creds.json \
		-p 8080:8080 \
		reddit-meter-api \

run-api-image:
	docker run \
		--rm \
		--env-file .env.dev \
		-v ${PWD}/creds.json:/reddit-meter-api/creds.json \
		-p 8080:8080 \
		reddit-meter-api

push-api:
	docker tag reddit-meter-api \
		us-central1-docker.pkg.dev/reddit-sentiment-meter/reddit-meter/reddit-meter-api

	docker push us-central1-docker.pkg.dev/reddit-sentiment-meter/reddit-meter/reddit-meter-api
