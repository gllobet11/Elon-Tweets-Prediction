import yfinance as yf
import pandas as pd
from arch import arch_model
from datetime import datetime
import matplotlib.pyplot as plt
from loguru import logger


def calculate_tesla_garch_volatility(start_date: str, end_date: str) -> pd.Series:
    """
    Calculates the conditional volatility of TSLA stock price using a GARCH(1,1) model.

    Args:
        start_date (str): Start date for fetching historical data (YYYY-MM-DD).
        end_date (str): End date for fetching historical data (YYYY-MM-DD).

    Returns:
        pd.Series: A daily series of conditional volatilities for TSLA, indexed by date.
                   Returns an empty Series if data fetching or GARCH fitting fails.
    """
    try:
        logger.info(f"Fetching TSLA historical data from {start_date} to {end_date}...")
        tsla_data = yf.download("TSLA", start=start_date, end=end_date)

        if tsla_data.empty:
            logger.warning("No TSLA data fetched. Returning empty series.")
            return pd.Series(dtype=float)

        # Calculate daily returns
        returns = tsla_data["Close"].pct_change().dropna()

        if returns.empty:
            logger.warning("No returns to calculate. Returning empty series.")
            return pd.Series(dtype=float)

        # Fit GARCH(1,1) model
        logger.info("Fitting GARCH(1,1) model to TSLA returns...")
        # Using vol='Garch' for GARCH(1,1) is standard.
        # Mean model 'Constant' is typical for returns.
        garch_model = arch_model(
            100 * returns, vol="Garch", p=1, q=1, mean="Constant", dist="t"
        )  # Using 't' distribution for robustness
        res = garch_model.fit(
            update_freq=5, disp="off"
        )  # disp='off' suppresses verbose output
        logger.info("GARCH(1,1) model fitted successfully.")

        # Extract conditional volatility
        conditional_volatility = res.conditional_volatility

        # Reindex to full daily range and forward fill missing values
        full_idx = pd.date_range(
            start=returns.index.min(), end=returns.index.max(), freq="D"
        )
        daily_volatility = conditional_volatility.reindex(full_idx, method="ffill")

        logger.info("TSLA GARCH volatility calculated.")
        return daily_volatility

    except Exception as e:
        logger.error(f"Error calculating TSLA GARCH volatility: {e}")
        return pd.Series(dtype=float)


if __name__ == "__main__":
    logger.add("file_{time}.log", rotation="500 MB")  # Log to file
    logger.info("--- Testing Tesla GARCH Volatility Feature Engineering ---")

    # Define a test period that covers your backtest timeframe (Sep-Nov 2025)
    # Plus some buffer for initial GARCH fitting
    test_start_date = "2025-08-01"
    test_end_date = "2025-11-30"

    tesla_volatility = calculate_tesla_garch_volatility(test_start_date, test_end_date)

    if not tesla_volatility.empty:
        print("\n--- Sample Tesla GARCH Volatility Data (first 5 days) ---")
        print(tesla_volatility.head().to_string())

        print("\n--- Sample Tesla GARCH Volatility Data (last 5 days) ---")
        print(tesla_volatility.tail().to_string())

        # Plotting the volatility for visual inspection
        plt.figure(figsize=(12, 6))
        tesla_volatility.plot(
            title="Tesla GARCH(1,1) Conditional Volatility (2025)",
            xlabel="Date",
            ylabel="Volatility (%)",
        )
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        logger.info("Tesla GARCH volatility feature generation test successful.")
    else:
        logger.warning("Tesla GARCH volatility could not be generated. Test failed.")
