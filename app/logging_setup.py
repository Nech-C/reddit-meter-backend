# app/logging_setup.py
import logging
import logging.config
import os


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO")
    fmt_console = "%(asctime)s %(levelname)s %(name)s — %(message)s"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {"format": fmt_console},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "console",
                }
            },
            "root": {
                "level": level,
                "handlers": ["console"],
            },
            "loggers": {
                "urllib3": {"level": "WARNING", "propagate": True},
                "google": {"level": "WARNING", "propagate": True},
            },
        }
    )
