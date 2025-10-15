# Reddit Sentiment Meter – Backend

## Introduction
The Reddit Sentiment Meter is a portfolio project that surfaces how online communities feel in near real time. This backend
collects Reddit submissions and comments from curated subreddits, evaluates emotion with a fine-tuned DistilBERT model, and
exposes clean, aggregated sentiment data through a public API. The goal is to demonstrate end-to-end ownership: data acquisition,
machine learning inference, cloud storage, and production-ready API design.

### Project Outcomes
- **High-signal data**: Daily emotion scores grouped by subreddit topic, stored in Firestore for 31-day trend analysis and
  exported to BigQuery for deep dives.
- **ML in production**: Batched inference with Hugging Face transformers optimized for pipeline throughput.
- **APIs for visualization**: FastAPI service designed to power dashboards, experimentation notebooks, or recruiter demos.
- **Cloud-native deployment**: Backend runs on Google App Engine and serves the React frontend in
  [`Nech-C/reddit-meter-frontend`](https://github.com/Nech-C/reddit-meter-frontend).

## Technical Overview & Design Choices

| Area | Decision | Rationale |
| --- | --- | --- |
| **Language** | Python 3.11 | Rich ecosystem for async APIs, data processing, and ML inference. |
| **Frameworks** | FastAPI, Pydantic | Async-first, type-safe, automatic docs for a recruiter-friendly API story. |
| **ML Model** | `bhadresh-savani/distilbert-base-uncased-emotion` | Compact transformer delivering strong zero-shot emotion detection. |
| **Data Pipeline** | Custom jobs orchestrated via Makefile targets | Keeps infra-light while showcasing reproducible data workflows. |
| **Storage & Analytics** | Google Firestore, Cloud Storage, BigQuery | Managed services with generous free tier, analytics-friendly exports, and simple integration. |
| **Dependency Mgmt** | [uv](https://github.com/astral-sh/uv) | Fast, deterministic Python environments that mirror prod containers. |
| **Containerization** | Dockerfiles for API & pipeline | Portable deployment story for recruiters and interview demos. |
| **Hosting** | Google App Engine | Fully managed autoscaling for the FastAPI backend with minimal ops overhead. |
| **Testing** | Pytest suite (unit + integration) | Highlights code quality focus and CI readiness. |

### Architecture Snapshot
```
          +-----------------+
          | Reddit API      |
          +--------+--------+
                   |
                   v
        +----------+-----------+
        | Pipeline Jobs (app/) |
        | - reddit fetchers    |
        | - ml inference       |
        | - aggregation        |
        +----------+-----------+
                   |
                   v
          +-----------------------------+        +-----------------------------+
          | Firestore · Cloud Storage   |<-------| FastAPI Service (app/api)   |
          | · BigQuery                  |------->| (Google App Engine)         |
          +-----------------------------+        +---------------+-------------+
                                                           |           \
                                                           |            \
                                                           v             v
                                            +-------------------+   +-----------------------------+
                                            | React Frontend &  |   | Analysts & notebooks        |
                                            | other UIs         |   | (Colab, Kaggle, etc.)       |
                                            | (Nech-C/reddit-   |   +-----------------------------+
                                            |  meter-frontend)  |
                                            +-------------------+
```

## Feature Highlights
- Fetches posts and comments from multiple subreddits with rate-limit controls.
- Emotion classification using DistilBERT with topic-aware weighting.
- Aggregates scores into daily, weekly, and monthly rollups.
- Persists results to Firestore and Google Cloud Storage for analytics.
- Exports curated aggregates into BigQuery for SQL storytelling and recruiter demos.
- Exposes FastAPI endpoints with automatic OpenAPI docs.
- Dockerized jobs and API for reproducible deployments.
- Makefile tasks streamline local development and CI steps.

## Getting Started

### 1. Clone & install dependencies
```bash
git clone https://github.com/<your-username>/reddit-meter-backend.git
cd reddit-meter-backend
uv sync
```

### 2. Configure environment variables
Environment is managed via `.env.<APP_ENV>` files. Create a `.env.dev` (or `.env.production`, etc.) and populate every variable your workflow needs using the reference below.

#### Core application & API
| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `APP_ENV` | No | `dev` | Selects which `.env.<APP_ENV>` file to load. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | – | Service-account JSON path for Firestore, GCS, and BigQuery clients. |
| `LOG_LEVEL` | No | `INFO` | Adjust logging verbosity for pipeline/API runs. |

#### Reddit ingestion
| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `REDDIT_CLIENT_ID` | Yes | – | OAuth client for Reddit API access. |
| `REDDIT_CLIENT_SECRET` | Yes | – | OAuth secret for Reddit API access. |
| `REDDIT_USERNAME` | Yes | – | Reddit username tied to the script application. |
| `REDDIT_PASSWORD` | Yes | – | Password for the ingest account. |
| `REDDIT_USER_AGENT` | Yes | – | Identifies the app to Reddit’s API. |
| `REDDIT_RATELIMIT_SECONDS` | No | `600` | Cooldown window to respect API limits. |
| `REDDIT_SUBREDDIT_JSON_PATH` | Yes | – | Path to the curated subreddit configuration file. |

#### ML inference tuning
| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `BATCH_MAX_TOKENS` | No | `512` | Upper bound of tokens processed per inference batch. |
| `SENTIMENT_MODEL_ID` | No | `bhadresh-savani/distilbert-base-uncased-emotion` | Allows swapping the deployed transformer. |

#### Firestore, GCS & BigQuery configuration
| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `FIRESTORE_POST_ARCHIVE_COLLECTION_NAME` | Yes | – | Firestore collection storing raw post archives. |
| `FIRESTORE_SENTIMENT_HISTORY_COLLECTION_NAME` | Yes | – | Collection containing historical sentiment rollups. |
| `FIRESTORE_CURRENT_SENTIMENT_COLLECTION_NAME` | Yes | – | Collection holding the latest snapshot document. |
| `FIRESTORE_HISTORY_RETRIEVAL_LIMIT` | No | `180` | Number of historical documents fetched by default. |
| `FIRESTORE_DATABASE_ID` | Yes | – | Firestore database name shared by pipeline and annotation runs. |
| `FIRESTORE_GOOGLE_BUCKET_NAME` | Yes | – | Backing Cloud Storage bucket used for JSON archives. |
| `BIGQUERY_DATASET_ID` | Yes | – | Dataset receiving structured sentiment history. |
| `BIGQUERY_GLOBAL_SENTIMENT_HISTORY_TABLE` | Yes | – | Table ID where aggregated sentiment rows land. |

#### Distributed annotation workers (`app/llm_annotation/annotation_worker.py`)
| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `RUN_ID` | Yes | – | Identifier for the annotation run stored in Firestore. |
| `WORKER_ID` | No | `worker` | Distinguishes notebook instances competing for leases. |
| `LEASE_MIN` | No | `30` | Minutes before a Firestore lease expires. |
| `LOAD_8BT` | No | `True` | Toggle 8-bit quantization for GPU-constrained notebooks. |
| `MAX_PROMPT_LEN` | No | `1024` | Prompt length for the Qwen model. |
| `MAX_NEW_TOKENS` | No | `64` | Controls response size. |
| `CHUNK_SIZE` | No | `1024` | Records per chunk when annotating a shard. |
| `BATCH_SIZE` | No | `8` | Base generation batch size; auto-tunes on OOM. |
| `SOURCE_HF_REPO` | No | `Nech-C/reddit-sentiment` | Hugging Face dataset providing raw Reddit posts. |
| `HF_TOKEN` | Yes | – | Hugging Face token for private dataset/model access. |
| `ANN_MODEL_ID` | No | `Qwen/Qwen3-4B-Instruct-2507` | Base instruct model used to label data. |
| `GCS_BUCKET` | Yes | – | Cloud Storage bucket storing shard inputs/outputs. |
| `GCS_PREFIX` | No | `annotations` | Bucket prefix for the annotation run. |
| `FIRESTORE_ANNO_COLLECTIONS` | Yes | – | Firestore collection managing annotation runs. |
| `FIRESTORE_TASKS_SUBCOLLECTIONS` | Yes | – | Subcollection containing shard lease documents. |

#### Dataset tooling (Colab/Kaggle utilities)
| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `SOURCE_REPO_ID` | No | `Nech-C/reddit-sentiment` | Source HF dataset for shard creation. |
| `TARGET_REPO_ID` | No | `Nech-C/reddit-sentiment-annotated` | Destination HF dataset for labeled exports. |
| `SHARD_SIZE` | No | `2048` | Records per Firestore shard task. |
| `CHUNK_SIZE` | No | `256` | Records per inference chunk when creating shard tasks. |
| `MODEL_ID` | No | `Qwen/Qwen2.5-7B-Instruct` | Model ID logged for provenance. |
| `FIRESTORE_DATABASE_ID` | Yes | – | Firestore database leveraged during shard orchestration. |
| `FIRESTORE_ANNO_COLLECTION_NAME` | No | `annotation_runs` | Firestore collection storing annotation run metadata. |
| `FIRESTORE_ANNO_TASKS_SUBCOLLECTION_NAME` | No | `tasks` | Subcollection storing shard task documents. |
| `GOOGLE_BUCKET_NAME` | Yes | – | Bucket containing raw archives to download before dataset upload. |
| `GCS_PREFIX` | No | `(empty)` | Optional prefix filter when reading archives from GCS. |
| `MIN_ARCHIVE_COUNT` | No | `1` | Guardrail for minimum archives before upload. |
| `DELETE_AFTER_UPLOAD` | No | `False` | Remove blobs post-upload to Hugging Face when true. |
| `DL_MAX_WORKERS` | No | `16` | Thread pool size for parallel downloads. |
| `DL_CHUNK_MB` | No | `8` | Chunk size for GCS downloads in MB. |
| `TMPDIR` | No | `/tmp` | Directory used for temporary extraction. |

### 3. Run the data pipeline locally
```bash
make runner-dev
```
This command orchestrates fetch → classify → aggregate steps, then persists the results. Use `make runner-test` for a fast run on
`test_subreddits.json`.

### 4. Start the API for dashboards or demos
```bash
make test-api-local
```
The service runs at `http://localhost:8080` and exposes:

- `GET /` – health probe
- `GET /sentiment/current` – latest aggregated sentiment
- `GET /sentiment/day` – past 24 hours
- `GET /sentiment/week` – past 7 days
- `GET /sentiment/month` – past 31 days

### 5. Build Docker images (optional)
```bash
make build-api       # Dockerfile.api
make build-pipeline  # Dockerfile.pipeline
```
Use the `push-*` targets to publish to Google Artifact Registry or any OCI registry.

## Repository Layout
```
.
├── app/
│   ├── api/              # FastAPI application (routers, schemas, dependencies)
│   ├── jobs/             # End-to-end ingestion and aggregation runners
│   ├── llm_annotation/   # Distributed labeling tooling (workers, shard scripts)
│   ├── ml/               # Transformer inference helpers
│   ├── processing/       # Sentiment aggregation & normalization logic
│   ├── reddit/           # Reddit API clients and fetch orchestration
│   ├── storage/          # Firestore, Cloud Storage, and BigQuery adapters
│   └── utils/            # Shared environment helpers and logging
├── requirements/         # Locked dependency exports per service
├── tests/                # Pytest suites for jobs, API, and utilities
├── Dockerfile.api        # Google App Engine-ready API container
├── Dockerfile.pipeline   # Batch pipeline container with CUDA/HF stack
├── Makefile              # Developer workflows and CI entry points
├── app.yaml              # App Engine deployment manifest
├── pyproject.toml        # Project metadata & tooling configuration
├── uv.lock               # Deterministic dependency lockfile
└── README.md             # You are here
```

Other notable artifacts:
- `subreddits.json` / `test_subreddits.json` – curated subreddit lists for prod vs. tests
- `requirements/*.txt` – ready-to-install requirements snapshots for App Engine builds
- `Dockerfile.*` – container blueprints for both the API and pipeline stacks

## Engineering Practices Recruiters Care About
- **Quality & Testing**: Run `pytest` via `make test` for the pipeline and API. Linting is enforced with `ruff`, formatting with
  `black`, and type hints are used across the codebase.
- **Observability Ready**: Structured logging is implemented in the jobs and API to plug into Stackdriver or any JSON log collector.
- **Security & Secrets**: Local development relies on `.env` files; production containers expect credentials to be mounted at
  runtime via secret managers.
- **Scalability**: Pipeline supports batch sizes and concurrency tuning through environment variables. API is stateless and ready
  for horizontal scaling on Cloud Run or similar platforms, and currently runs on Google App Engine standard.
- **Product Impact**: Aggregated sentiment trends have been used in interviews to discuss community health monitoring and content
  strategy insights.

## Roadmap & Future Enhancements
- Extend ML model to detect sarcasm and toxicity for richer sentiment analysis.
- Introduce Supabase/Postgres sink for SQL-friendly analytics.
- Add CI/CD pipeline (GitHub Actions) that runs tests, builds images, and deploys to Cloud Run.
- Provide a public demo dashboard built with SvelteKit (frontend companion project).

## Distributed Training & Smart Data Labeling
- **Annotation worker**: `app/llm_annotation/annotation_worker.py` shards Reddit data into prompts and coordinates distributed
  labeling jobs. Each worker checks out tasks from Firestore, processes batches locally on Colab or Kaggle GPUs, and writes
  completions back through Cloud Storage.
- **GCS coordination**: Intermediate artifacts (claim checks, prompt shards, completion payloads) live under a shared GCS prefix so
  you can mix-and-match compute from different notebooks without race conditions.
- **BigQuery landing zone**: Curated exports from Cloud Storage are mirrored into BigQuery for long-term analytics, making it
  easy to validate model drift or demo SQL storytelling to recruiters.

## Contributing & Local Tooling
```bash
black .
ruff check .
pytest
```

Bug reports and feature suggestions are always welcome—open an issue or reach out on LinkedIn.

---
This backend powers the data layer for the Reddit Sentiment Meter portfolio project. Feedback and collaboration opportunities are
greatly appreciated.
