import os

from app.utils import utils


def test_getenv_bool(monkeypatch):
    monkeypatch.setenv("FLAG_TRUE", "True")
    monkeypatch.setenv("FLAG_FALSE", "no")

    assert utils.getenv_bool("FLAG_TRUE") is True
    assert utils.getenv_bool("FLAG_FALSE", default=True) is False
    assert utils.getenv_bool("MISSING", default=True) is True


def test_getenv_int(monkeypatch):
    monkeypatch.setenv("INT_VALUE", "123")
    monkeypatch.setenv("INT_BAD", "not-int")

    assert utils.getenv_int("INT_VALUE", 0) == 123
    assert utils.getenv_int("INT_BAD", 7) == 7
    assert utils.getenv_int("INT_MISSING", 5) == 5


def test_getenv_app_env(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    assert utils.getenv_app_env() == "test"

    monkeypatch.setenv("APP_ENV", "prod")
    assert utils.getenv_app_env() == "prod"


def test_get_dotenv_name(monkeypatch):
    monkeypatch.setenv("APP_ENV", "staging")
    assert utils.get_dotenv_name() == ".env.staging"
