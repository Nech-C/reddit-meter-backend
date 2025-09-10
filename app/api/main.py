# app/api/main.py
import secure
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from app.storage.firestore import (
    db,
    CURRENT_SENTIMENT_COLLECTION_NAME,
    get_latest_sentiment,
    get_recent_sentiment_history,
)

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


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/sentiment/current")
@app.state.limiter.limit("10/minute")
def get_current_sentiment(request: Request):
    return get_latest_sentiment()


@app.get("/sentiment/day")
@app.state.limiter.limit("2/minute")
def get_past_day_sentiment(request: Request):
    # TODO: make this a variable
    return get_recent_sentiment_history(1)


@app.get("/sentiment/week")
@app.state.limiter.limit("2/minute")
def get_past_week_sentiment(request: Request):
    return get_recent_sentiment_history(7)


@app.get("/sentiment/month")
@app.state.limiter.limit("2/minute")
def get_past_month_sentiment(request: Request):
    return get_recent_sentiment_history(31)


@app.get("/_ah/warmup")
@app.state.limiter.exempt
def warmup():
    """
    Called by App Engine before routing real traffic to a new instance.
    Do lightweight tasks that pay the one-time cold costs:
      - import paths executed
      - gRPC channels to Firestore opened
      - first query compiled
    """
    try:
        list(db.collection(CURRENT_SENTIMENT_COLLECTION_NAME).limit(1).stream())
        return {"status": "ok"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
