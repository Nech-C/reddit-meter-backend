# test_bigquery.py
from unittest.mock import MagicMock

import pytest
from google.api_core import exceptions as google_exceptions


def test_insert_global_sentiment_history_sucess(
    bigquery_repo, sample_summary, get_constant_datetime
):
    bigquery_repo.client.insert_rows_json = MagicMock()
    bigquery_repo.client.insert_rows_json.return_value = []

    errors = bigquery_repo.insert_global_sentiment_history(sample_summary)

    bigquery_repo.client.insert_rows_json.assert_called_once()

    args, kwargs = bigquery_repo.client.insert_rows_json.call_args
    assert (
        args[0]
        == f"{bigquery_repo.s.bq_dataset}.{bigquery_repo.s.bq_global_sentiment_history_table}"
    )

    row = args[1][0]
    assert row["timestamp"] == get_constant_datetime.isoformat()
    assert row["timestamp"] == row["updated_at"]

    assert errors == []


def test_insert_global_sentiment_history_failure(
    bigquery_repo, sample_summary, get_constant_datetime
):
    # fail to validate
    error = bigquery_repo.insert_global_sentiment_history({"joy": "no joy"})
    assert error == [{"error": "validation_failed"}]

    # bigquery client.insert_rows_json returns error
    bigquery_repo.client.insert_rows_json = MagicMock()
    bigquery_repo.client.insert_rows_json.return_value = ["!"]
    error = bigquery_repo.insert_global_sentiment_history(sample_summary)
    assert error == ["!"]

    # trigger GoogleAPICallError
    bigquery_repo.client.insert_rows_json = MagicMock()
    bigquery_repo.client.insert_rows_json.return_value = ["!"]
    bigquery_repo.client.insert_rows_json.side_effect = (
        google_exceptions.GoogleAPICallError("boom")
    )
    with pytest.raises(google_exceptions.GoogleAPICallError):
        _ = bigquery_repo.insert_global_sentiment_history(sample_summary)

    # trigger other error
    bigquery_repo.client.insert_rows_json = MagicMock()
    bigquery_repo.client.insert_rows_json.return_value = ["!"]
    bigquery_repo.client.insert_rows_json.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        _ = bigquery_repo.insert_global_sentiment_history(sample_summary)
