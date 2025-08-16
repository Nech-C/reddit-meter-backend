# file: app/utils/utils.py
import os


def getenv_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("true", "1", "yes", "on")


def getenv_int(key: str, default: int) -> int:
    v = os.getenv(key)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def getenv_str(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def getenv_app_env():
    return getenv_str("APP_ENV", "test")


def get_dotenv_name():
    return f".env.{getenv_app_env()}"
