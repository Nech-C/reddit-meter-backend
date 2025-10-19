# app/api/main.py
from datetime import datetime, timedelta

import secure
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from app.storage.firestore import default_repo, FirestoreRepo
from app.storage.bigquery import default_bq_repo, BigQueryRepo
from app.constants import TIMEZONE

app = FastAPI()
app.state.limiter = Limiter(key_func=get_remote_address)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Secure headers middleware using secure.Secure
secure_headers = secure.Secure.with_default_headers()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        await secure_headers.set_headers_async(response)
        return response


app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SlowAPIMiddleware)


def get_repo() -> FirestoreRepo:
    return default_repo()


def get_bq_repo() -> BigQueryRepo:
    return default_bq_repo()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


# TODO: convert timestamps into ISO strings before returning results. do it in pydantic?


@app.get("/sentiment/current")
@app.state.limiter.limit("10/minute")
def get_current_sentiment(request: Request, repo=Depends(get_repo)):
    return repo.get_latest_sentiment()


@app.get("/sentiment/day")
@app.state.limiter.limit("2/minute")
def get_past_day_sentiment(request: Request, repo=Depends(get_repo)):
    # TODO: make this a variable
    return repo.get_recent_sentiment_history(1)


@app.get("/sentiment/week")
@app.state.limiter.limit("2/minute")
def get_past_week_sentiment(request: Request, repo=Depends(get_repo)):
    return repo.get_recent_sentiment_history(7)


@app.get("/sentiment/v2/week")
@app.state.limiter.limit("2/minute")
def get_past_week_sentiment_v2(request: Request, repo=Depends(get_bq_repo)):
    """get sentiment from 7 days ago to today from bigquery"""
    now = datetime.now(TIMEZONE)
    seven_days_ago = now - timedelta(days=7)
    return repo.get_global_sentiment_history_by_day_range(
        seven_days_ago.date(), now.date()
    )


@app.get("/sentiment/month")
@app.state.limiter.limit("2/minute")
def get_past_month_sentiment(request: Request, repo=Depends(get_repo)):
    return repo.get_recent_sentiment_history(31)


@app.get("/_ah/warmup")
@app.state.limiter.exempt
def warmup(repo=Depends(get_repo)):
    """
    Called by App Engine before routing real traffic to a new instance.
    Do lightweight tasks that pay the one-time cold costs:
      - import paths executed
      - gRPC channels to Firestore opened
      - first query compiled
    """
    try:
        repo.healthcheck()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
