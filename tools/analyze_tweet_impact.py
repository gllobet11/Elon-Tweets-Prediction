import json
import os
import sys
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from loguru import logger

# --- Add project root to sys.path ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.ingestion.poly_feed import PolymarketFeed
    from config.settings import MARKET_KEYWORDS
    from config.bins_definition import MARKET_BINS
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    sys.exit(1)


# --- Constants ---
TWEET_DATA_PATH = "data/processed/merged_elon_tweets.csv"
BINS_TO_PLOT = ["280-299", "380-399"]  # Plot price for these two bins

# --- Analysis Window Configuration ---
ANALYSIS_DATE = "2025-12-06"
START_TIME_ET = "06:00:00"
END_TIME_ET = "11:00:00"
TIMEZONE_ET = "America/New_York"

# Analysis Parameters
TIME_AGGREGATION_MINS = "1min"


def get_analysis_timestamps() -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Calculates start and end timestamps for the analysis window in UTC and Unix format."""
    start_str = f"{ANALYSIS_DATE} {START_TIME_ET}"
    end_str = f"{ANALYSIS_DATE} {END_TIME_ET}"

    start_et = pd.to_datetime(start_str).tz_localize(TIMEZONE_ET)
    end_et = pd.to_datetime(end_str).tz_localize(TIMEZONE_ET)

    start_utc = start_et.tz_convert("UTC")
    end_utc = end_et.tz_convert("UTC")
    start_unix = int(start_utc.timestamp())

    logger.info(f"Window Start (ET): {start_et} | Window End (ET): {end_et}")

    return start_utc, end_utc, start_unix


def load_data(
    start_unix_ts: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]] | tuple[None, None]:
    """
    Loads tweet data and fetches price data for multiple bins of the specific market.
    """
    # 1. Load Tweet Data
    logger.info("Loading tweet data...")
    if not os.path.exists(TWEET_DATA_PATH):
        logger.error(f"Tweet data not found at {TWEET_DATA_PATH}")
        return None, None

    df_tweets = pd.read_csv(TWEET_DATA_PATH)
    df_tweets["created_at"] = pd.to_datetime(df_tweets["created_at"], format="mixed")

    # Robust timezone handling
    if df_tweets["created_at"].dt.tz is None:
        df_tweets["created_at"] = df_tweets["created_at"].dt.tz_localize("UTC")
    else:
        df_tweets["created_at"] = df_tweets["created_at"].dt.tz_convert("UTC")

    df_tweets = df_tweets.set_index("created_at")
    logger.info(f"Loaded {len(df_tweets)} tweets.")

    # 2. Fetch Price Data for the CURRENT market's bins
    logger.info("Connecting to Polymarket to fetch price data...")
    poly_feed = PolymarketFeed()
    if not poly_feed.valid:
        logger.error("Could not initialize PolymarketFeed.")
        return df_tweets, None

    logger.info(f"Fetching market info for keywords: {MARKET_KEYWORDS}")
    updated_bins = poly_feed.fetch_market_ids_automatically(
        keywords=MARKET_KEYWORDS, bins_dict=MARKET_BINS
    )

    price_dataframes = {}
    for bin_label in BINS_TO_PLOT:
        if bin_label not in updated_bins or not updated_bins[bin_label].get("id_yes"):
            logger.error(
                f"Could not find a valid token ID for bin '{bin_label}'. Skipping."
            )
            continue

        market_id_to_fetch = updated_bins[bin_label]["id_yes"]
        logger.info(
            f"Fetching prices for bin '{bin_label}' (Token ID: ...{market_id_to_fetch[-10:]})"
        )

        history = poly_feed.get_price_history(
            market_id=market_id_to_fetch, fidelity=1, start_timestamp=start_unix_ts
        )
        if not history:
            logger.warning(f"No price history found for bin '{bin_label}'.")
            continue

        df_prices = pd.DataFrame(history)
        df_prices.rename(columns={"t": "timestamp", "p": "price"}, inplace=True)
        df_prices["timestamp"] = pd.to_datetime(
            df_prices["timestamp"], unit="s", utc=True
        )
        df_prices = df_prices.set_index("timestamp")

        df_prices = df_prices.loc[~df_prices.index.duplicated(keep="last")]
        logger.success(
            f"Loaded {len(df_prices)} unique price points for bin '{bin_label}'."
        )
        price_dataframes[bin_label] = df_prices

    if not price_dataframes:
        logger.error("Failed to load price data for any of the specified bins.")
        return df_tweets, None

    return df_tweets, price_dataframes


def analyze_and_plot(
    df_tweets: pd.DataFrame,
    df_prices_dict: dict[str, pd.DataFrame],
    start_window: pd.Timestamp,
    end_window: pd.Timestamp,
):
    """
    Generates a plot of cumulative tweets and multiple price series for a specific time window.
    """
    logger.info(
        f"Correctly calculating daily cumulative tweets before filtering for window: {start_window} to {end_window}"
    )

    day_start_et = pd.Timestamp(ANALYSIS_DATE).tz_localize(TIMEZONE_ET).normalize()
    day_start_utc = day_start_et.tz_convert("UTC")

    full_day_resampled = (
        df_tweets.loc[day_start_utc:end_window]
        .resample(TIME_AGGREGATION_MINS)
        .size()
        .fillna(0)
    )
    daily_cumulative = full_day_resampled.cumsum()
    cumulative_tweets = daily_cumulative[start_window:end_window]

    # --- VISUAL DIAGNOSTIC: Get individual tweet times ---
    tweet_events = df_tweets.loc[start_window:end_window].index

    # Resample price data
    prices_in_window_dict = {}
    for bin_label, df_prices in df_prices_dict.items():
        prices_in_window_dict[bin_label] = (
            df_prices["price"]
            .resample(TIME_AGGREGATION_MINS)
            .ffill()[start_window:end_window]
        )

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()

    et_tz = pytz.timezone(TIMEZONE_ET)

    # Plot Cumulative Tweets line
    ax1.set_xlabel(f"Time (ET) on {start_window.astimezone(et_tz).date()}")
    ax1.set_ylabel("Total Cumulative Tweets", color="blue")
    ax1.plot(
        cumulative_tweets.index,
        cumulative_tweets.values,
        color="blue",
        linestyle="-",
        label="Cumulative Tweets",
    )
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(bottom=0, top=80)

    # --- VISUAL DIAGNOSTIC: Plot individual tweet events as a rug plot ---
    ax1.plot(
        tweet_events,
        [0.5] * len(tweet_events),
        "|",
        color="black",
        markersize=10,
        label="Individual Tweet",
    )

    # Plot each price series
    colors = ["green", "red", "purple", "orange"]
    ax2.set_ylabel(f"Price of Bins", color=colors[0])
    for i, (bin_label, prices_in_window) in enumerate(prices_in_window_dict.items()):
        color = colors[i % len(colors)]
        ax2.plot(
            prices_in_window.index,
            prices_in_window.values,
            color=color,
            marker=".",
            linestyle="-",
            markersize=4,
            label=f"Price Bin {bin_label}",
        )
    ax2.tick_params(axis="y", labelcolor=colors[0])

    plt.title(
        f"Cumulative Tweet and Price Activity between {START_TIME_ET} and {END_TIME_ET} ET"
    )

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=et_tz))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    fig.tight_layout()

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    output_filename = f"multi_bin_cumulative_impact_{ANALYSIS_DATE}.png"
    plt.savefig(output_filename)
    logger.info(f"Plot saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    start_utc, end_utc, start_unix = get_analysis_timestamps()
    df_tweets, df_prices_dict = load_data(start_unix)
    if df_tweets is not None and df_prices_dict is not None:
        analyze_and_plot(df_tweets, df_prices_dict, start_utc, end_utc)
    else:
        logger.error("Failed to load data. Aborting analysis.")
