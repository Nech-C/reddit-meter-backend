import logging
from datetime import datetime, date
from functools import cache
from typing import List, Callable

from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter
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

    def __init__(
        self,
        settings: BigQuerySettings = None,
        client: bigquery.Client = None,
        now_fn: Callable[..., datetime] = datetime.now,
    ):
        self.s: BigQuerySettings = (
            settings if settings is not None else get_bigquery_settings()
        )
        self.client: bigquery.Client = bigquery.Client()
        self._now_fn = now_fn

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
        now = self._now_fn(constants.TIMEZONE).isoformat()
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

    def get_global_sentiment_history_by_day_range(
        self, start: date, end: date
    ) -> List[SentimentSummary]:
        """Get global sentiment history between the specified day range from BigQuery

        Args:
            start (date): The earliest date allowed for SentimentSummay
            end (date): The latest date allowed for SentimentSummary

        Returns:
            List[SentimentSummary]: A list of SentimentSummary models
        """
        query = (
            "SELECT * "
            f"FROM `{self.s.bq_dataset}.{self.s.bq_global_sentiment_history_table}` "
            "WHERE DATE(timestamp) BETWEEN @start_date AND @end_date "
            "ORDER BY timestamp ASC "
            f"LIMIT {self.s.bq_global_sentiment_history_limit};"
        )

        job_config = QueryJobConfig(
            query_parameters=[
                ScalarQueryParameter("start_date", "DATE", start),
                ScalarQueryParameter("end_date", "DATE", end),
            ]
        )

        job = self.client.query(query, job_config=job_config)
        rows = job.result()

        results = []

        for row in rows:
            try:
                row = dict(row.items())
                validated = SentimentSummary.model_validate(row)
                results.append(validated)
            except Exception:
                log.exception("fail to validate a sentiment summary from bq")
                continue

        return results


@cache
def default_bq_repo() -> BigQueryRepo:
    return BigQueryRepo()
