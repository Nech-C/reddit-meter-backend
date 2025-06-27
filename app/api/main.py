# file: app/api/main.pys
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/sentiment/current")
def get_current_sentiment():
    return {
        "joy": 0.8,
        "sadness": 0.1,
        "anger": 0.05,
        "fear": 0.02,
        "love": 0.03,
        "surprise": 0.01,
    }
