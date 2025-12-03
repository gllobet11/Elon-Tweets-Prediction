"""
api_ingestor.py

This module provides the ApiIngestor class, responsible for fetching tweet data
directly from the xtracker.polymarket.com API, based on the cURL command.
"""

from datetime import datetime

import httpx
import pandas as pd
from loguru import logger


class ApiIngestor:
    """
    Fetches tweet data for a user from the x-tracker API using date ranges.
    """

    API_ENDPOINT = "https://xtracker.polymarket.com/api/users/elonmusk/posts"
    HEADERS = {
        "accept": "*/*",
        "referer": "https://xtracker.polymarket.com/user/elonmusk",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetches tweets from the API for a specific date range.
        """
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        params = {"startDate": start_str, "endDate": end_str}
        logger.info(f"Fetching from {start_str} to {end_str}")

        try:
            with httpx.Client() as client:
                response = client.get(
                    self.API_ENDPOINT,
                    params=params,
                    headers=self.HEADERS,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                json_response = response.json()

            tweets_data = json_response.get("data", [])

            if not tweets_data:
                logger.info("API returned no tweet data for this period.")
                return pd.DataFrame()

            logger.debug(f"Type of json_response['data']: {type(tweets_data)}")
            if isinstance(tweets_data, dict):
                tweets_data = [tweets_data]

            df = pd.DataFrame(tweets_data)
            logger.debug(f"Step 1 PASSED. DF created with {len(df)} rows.")

            # --- FIX: Handle duplicate 'id' column ---
            # The API returns an 'id' (database ID) and 'platformId' (Twitter ID).
            # We want 'platformId' to be our 'id' column.
            if "id" in df.columns and "platformId" in df.columns:
                logger.debug(
                    "Found both 'id' and 'platformId' columns. Dropping 'id' (database ID).",
                )
                df = df.drop(columns=["id"])
            elif "id" in df.columns and "platformId" not in df.columns:
                logger.warning(
                    "No 'platformId' found in API response, using 'id' as default.",
                )
            # --- End FIX ---

            logger.debug("Step 2: Renaming columns...")
            df = df.rename(
                columns={
                    "platformId": "id",
                    "content": "text",
                    "createdAt": "created_at",
                },
            )
            logger.debug("Step 2 PASSED. Columns renamed.")

            logger.debug("Step 3: Slicing DataFrame...")
            df = df[["id", "text", "created_at"]]
            logger.debug("Step 3 PASSED. Slicing successful.")

            logger.debug("Step 4: Converting ID to numeric...")
            df["id"] = pd.to_numeric(df["id"], errors="coerce")
            logger.debug("Step 4 PASSED. ID converted to numeric.")

            logger.debug("Step 5: Dropping NA IDs...")
            df = df.dropna(subset=["id"])
            logger.debug("Step 5 PASSED. NA IDs dropped.")

            logger.debug("Step 6: Converting ID to int64...")
            df["id"] = df["id"].astype("int64")
            logger.debug("Step 6 PASSED. ID converted to int64.")

            logger.debug("Step 7: Converting created_at to datetime...")
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
            logger.debug("Step 7 PASSED. created_at converted to datetime.")

            logger.success(
                f"Successfully processed {len(df)} tweets. Dates: {df['created_at'].min().date()} to {df['created_at'].max().date()}.",
            )
            return df

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred in fetch: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return pd.DataFrame()
