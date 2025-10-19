"""Centralized constants used across the Reddit meter backend."""

from datetime import timezone

# NOTE:
# Keeping all temporal calculations in UTC ensures consistent ordering when
# persisting timestamps to Firestore and JSON archives.
TIMEZONE = timezone.utc

# Default configuration values used when interacting with Reddit and storing
# inference metadata. They are defined here to avoid scattering magic numbers
# and strings throughout the codebase.
DEFAULT_MAX_POST_AGE_DAYS = 7
DEFAULT_SENTIMENT_SOURCE = "bert"
DEFAULT_FETCH_SLEEP_SECONDS = 1
DEFAULT_COMMENT_AUTHOR_PLACEHOLDER = "[deleted]"
DEFAULT_BQ_TEXT_PREVIEW_MAX = 1024
