# Reddit Sentiment Meter – Backend

This repository powers the data collection pipeline and API for my Reddit sentiment meter project. It fetches posts and comments from a curated list of subreddits, runs them through a DistilBERT emotion classifier, aggregates the results and stores them in Firestore. A small FastAPI application exposes the latest sentiment data.

## Features
- Fetches posts and comments from multiple subreddits
- Emotion classification using `bhadresh-savani/distilbert-base-uncased-emotion`
- Weighted aggregation of sentiment scores
- Firestore and Google Cloud Storage integration
- Docker images for the pipeline and API
- Makefile helpers for local development

## Quick Start

### 1. Install dependencies
This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
uv sync
```

### 2. Environment variables
Create a `.env.dev` file with the following variables:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password
REDDIT_USER_AGENT=RedditMeter/1.0
REDDIT_RATELIMIT_SECONDS=600
GOOGLE_APPLICATION_CREDENTIALS=creds.json
GOOGLE_BUCKET_NAME=your_bucket
FIRESTORE_DATABASE_ID=your_db
SUBREDDIT_JSON_PATH=subreddits.json
```

A `.env.test` file is used by the Makefile for pipeline tests.

### 3. Running the pipeline
Run the collection and sentiment pipeline locally:

```bash
make runner-dev
```

This processes the subreddits defined in `subreddits.json` and writes aggregated results to Firestore and GCS. Use `make runner-test` to run against the smaller `test_subreddits.json` sample.

### 4. Starting the API

```bash
make test-api-local
```

The API runs on `http://localhost:8080` with these endpoints:

- `GET /` – basic check
- `GET /sentiment/current` – latest aggregated sentiment
- `GET /sentiment/day` – history for the past day
- `GET /sentiment/week` – past 7 days
- `GET /sentiment/month` – past 31 days

### 5. Docker
Build the images locally:

```bash
make build-api       # Dockerfile.api
make build-pipeline  # Dockerfile.pipeline
```

Use the `push-*` targets to tag and push to Google Artifact Registry.

## Repository layout
```
app/
├── api/          # FastAPI application
├── jobs/         # Collection pipeline runner
├── ml/           # BERT-based sentiment inference
├── processing/   # Aggregation utilities
├── reddit/       # Reddit fetch helpers
└── storage/      # Firestore and GCS wrappers
```

Other important files:

- `Dockerfile.api` – API container
- `Dockerfile.pipeline` – pipeline container with the ML model
- `Makefile` – convenience tasks
- `subreddits.json` – subreddit list grouped by topic

## Contributing
Source files are formatted with `black` and linted with `ruff`.

```bash
black .
ruff check .
```

---
This backend forms part of my personal portfolio project. Feedback is welcome!
