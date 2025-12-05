import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from transformers import pipeline
from loguru import logger

# --- NewsAPI Configuration ---
NEWS_API_KEY = "804ac9ca97264fc59870104765fc3d0c"
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# --- Sentiment Analysis Model ---
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)


def fetch_news_sentiment_daily(query: str, date: datetime, api_key: str) -> float:
    """
    Fetches news articles for a specific query and date, calculates average sentiment.
    """
    date_str = date.strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": date_str,
        "to": date_str,
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": api_key,
        "pageSize": 50,
    }

    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        if not articles:
            logger.info(f"No articles found for {date_str}.")
            return 0.0

        titles = [article["title"] for article in articles if article.get("title")]
        if not titles:
            return 0.0

        results = sentiment_pipeline(titles)

        sentiment_scores = [
            result["score"] if result["label"] == "POSITIVE" else -result["score"]
            for result in results
        ]

        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        logger.info(
            f"Fetched {len(articles)} articles for {date_str}. Avg sentiment: {avg_score:.4f}"
        )
        return avg_score

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news for {date_str}: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error processing sentiment for {date_str}: {e}")
        return 0.0


def calculate_news_sentiment_spike(
    dates_series: pd.DatetimeIndex, query: str, api_key: str
) -> pd.Series:
    """
    Calculates the Z-score deviation of daily news sentiment.
    """
    sentiment_scores = [
        fetch_news_sentiment_daily(query, date.to_pydatetime(), api_key)
        for date in dates_series
    ]
    daily_sentiment = pd.Series(sentiment_scores, index=dates_series)

    rolling_mean = daily_sentiment.rolling(window=7, min_periods=1).mean()
    rolling_std = daily_sentiment.rolling(window=7, min_periods=1).std()

    safe_rolling_std = rolling_std.replace(0, pd.NA)
    news_sentiment_spike = (daily_sentiment - rolling_mean) / safe_rolling_std

    return news_sentiment_spike.fillna(0)


if __name__ == "__main__":
    logger.add("file_news_sentiment_retest_{time}.log")
    logger.info("--- Re-testing News Sentiment Spike (Nov 15-22) ---")

    test_start_date = datetime(2025, 11, 15)
    test_end_date = datetime(2025, 11, 22)
    test_dates = pd.date_range(
        start=test_start_date, end=test_end_date, freq="D", tz="UTC"
    )

    news_sentiment_spike_feature = calculate_news_sentiment_spike(
        dates_series=test_dates,
        query="Elon Musk OR Tesla OR SpaceX",
        api_key=NEWS_API_KEY,
    )

    print("\n--- Results for Nov 15-22, 2025 ---")
    print(news_sentiment_spike_feature.to_string())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    news_sentiment_spike_feature.plot(
        title="News Sentiment Spike (Z-Score) Nov 15-22",
        xlabel="Date",
        ylabel="Z-Score",
    )
    plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if (news_sentiment_spike_feature == 0).all():
        logger.warning(
            "All sentiment spike values are zero. This indicates the API is likely blocking future date requests as expected."
        )
    else:
        logger.success(
            "Non-zero sentiment values were generated, which is unexpected for future dates."
        )
