import logging
from datetime import datetime
from functools import cache

from google.cloud import bigquery
from google.api_core import exceptions as google_exceptions
from pydantic import ValidationError

from app import constants
from app.config import BigQuerySettings, get_bigquery_settings
from app.models.post import SentimentSummary
from app.logging_setup import setup_logging

setup_logging()
log = logging.getLogger("storage.bigquery")


class BigQueryRepo:
    """A wrapper for google.cloud.bigquery.Client"""

    def __init__(self, s: BigQuerySettings = None):
        self.s: BigQuerySettings = s if s is not None else get_bigquery_settings()
        self.client: bigquery.Client = bigquery.Client()

    def insert_global_sentiment_history(
        self, aggregated_sentiment: SentimentSummary | dict
    ) -> list[dict]:
        """
        Insert one SentimentSummary row into BigQuery.

        Returns a list of insert errors (empty list on success), or raises on API-level failures.
        """
        try:
            summary = (
                aggregated_sentiment
                if isinstance(aggregated_sentiment, SentimentSummary)
                else SentimentSummary.model_validate(aggregated_sentiment)
            )
        except ValidationError:
            log.exception(
                "Provided aggregated_sentiment failed validation; aborting BQ insert."
            )
            # return a sentinel error list so caller can decide; you could also raise
            return [{"error": "validation_failed"}]

        row = summary.to_bq_dict()
        table_id = f"{self.s.bq_dataset}.{self.s.bq_global_sentiment_history_table}"
        now = datetime.now(constants.TIMEZONE).isoformat()
        row["timestamp"] = now
        row["updated_at"] = now
        try:
            errors = self.client.insert_rows_json(table_id, [row], retry=self.s.retry)
            if errors:
                log.error("BigQuery insert returned errors: %s", errors)
            else:
                log.info("Inserted sentiment snapshot to BigQuery table %s", table_id)
            return errors
        except google_exceptions.GoogleAPICallError as e:
            log.exception("Google API error while inserting rows to BigQuery: %s", e)
            raise
        except Exception as e:
            log.exception("Unexpected error while inserting rows to BigQuery: %s", e)
            raise


@cache
def default_bq_repo() -> BigQueryRepo:
    return BigQueryRepo()
