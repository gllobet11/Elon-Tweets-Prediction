import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
from loguru import logger
import numpy as np

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.strategy.prob_math import DistributionConverter
    from config.bins_definition import MARKET_BINS
    from config.settings import ALPHA_CANDIDATES, WEEKS_TO_VALIDATE  # Corrected import, added WEEKS_TO_VALIDATE
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from tools.generate_historical_performance import generate_backtest_predictions # Add this import
except Exception as e:
    logger.error(f"Error import: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
HISTORICAL_PERFORMANCE_PATH = os.path.join(
    project_root, "data", "processed", "historical_performance.csv"
)
OUTPUT_PLOT_PATH = os.path.join(project_root, "historical_predictions_plot.png")
REGIME_CHANGE_PLOT_PATH = os.path.join(project_root, "regime_change_visualization.png")
# LOOKBACK_WEEKS = 52 # REMOVE THIS LINE
NUM_CHANGES = 3
MIN_WEEKS_SEPARATION = 8

OPTIMAL_ALPHA = ALPHA_CANDIDATES[0]
OPTIMAL_DISTRIBUTION = "nbinom"
BINS_CONFIG_LIST = [(k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]


def calculate_log_loss_for_row(row: pd.Series) -> float:
    """Calculates Log Loss for a single row of predictions."""
    mu_pred = row["y_pred"]
    y_true = row["y_true"]

    try:
        probs = DistributionConverter.get_bin_probabilities(
            mu_remainder=mu_pred,
            current_actuals=0,
            model_type=OPTIMAL_DISTRIBUTION,
            alpha=OPTIMAL_ALPHA,
            bins_config=BINS_CONFIG_LIST,
        )
    except ValueError:
        return np.nan

    correct_bin = None
    for label, lower, upper in BINS_CONFIG_LIST:
        if lower <= y_true < upper:
            correct_bin = label
            break

    if correct_bin is None:
        return np.nan

    prob_correct = probs.get(correct_bin, 0) + 1e-9
    return -np.log(prob_correct)


def visualize_predictions_and_metrics():
    logger.info("📊 Starting visualization of historical predictions...")

    # Load historical predictions directly from the generation function
    df_hist = generate_backtest_predictions(weeks_to_validate=WEEKS_TO_VALIDATE)

    if df_hist.empty:
        logger.error("No historical prediction data generated for visualization. Exiting.")
        sys.exit(1)

    df_hist["week_start_date"] = pd.to_datetime(df_hist["week_start_date"])
    df_hist = df_hist.sort_values("week_start_date").set_index("week_start_date")

    df_hist["log_loss"] = df_hist.apply(calculate_log_loss_for_row, axis=1)
    avg_log_loss = df_hist["log_loss"].mean()

    rmse = np.sqrt(np.mean((df_hist["y_true"] - df_hist["y_pred"]) ** 2))

    logger.info(f"Visualizing predictions for {len(df_hist)} weeks.")
    logger.info(f"Average Log Loss: {avg_log_loss:.4f}")
    logger.info(f"RMSE: {rmse:.2f}")

    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=df_hist,
        x=df_hist.index,
        y="y_true",
        label="Actual Tweets (y_true)",
        marker="o",
        color="blue",
    )
    sns.lineplot(
        data=df_hist,
        x=df_hist.index,
        y="y_pred",
        label="Predicted Tweets (y_pred)",
        marker="x",
        linestyle="--",
        color="orange",
    )

    plt.fill_between(
        df_hist.index,
        df_hist["y_pred_lower"],
        df_hist["y_pred_upper"],
        color="orange",
        alpha=0.2,
        label="Confidence Interval",
    )

    plt.title("Historical Tweet Predictions vs. Actuals with Confidence Interval")
    plt.xlabel("Week Start Date")
    plt.ylabel("Number of Tweets")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_PATH)
    logger.success(f"Plot saved to: {OUTPUT_PLOT_PATH}")


def visualize_regime_change():
    """
    Loads unified data, calculates regime intensity (Z-score) using FeatureEngineer,
    and generates a visualization showing monthly tweet counts and the Z-score time series.
    """
    print("--- Z-Score Based Regime Change Visualization ---")

    try:
        # 1. Load unified data
        df_tweets = load_unified_data()

        # 2. Generate features, including the Z-score
        feat_eng = FeatureEngineer()
        all_features = feat_eng.process_data(df_tweets)

        # 3. Prepare data for visualization
        monthly_counts = all_features["n_tweets"].resample("MS").sum()
        z_score_series = all_features["regime_intensity"].dropna()

        # 4. Generate the visualization
        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

        # --- Plot 1: Monthly Tweets ---
        ax1.plot(
            monthly_counts.index,
            monthly_counts.values,
            marker="o",
            linestyle="-",
            label="Tweets per Month",
        )
        ax1.set_title("Elon Musk's Monthly Tweets", fontsize=16)
        ax1.set_ylabel("Number of Tweets")
        ax1.legend()

        # --- Plot 2: Z-Score (Regime Intensity) ---
        ax2.plot(
            z_score_series.index,
            z_score_series.values,
            color="purple",
            label="Regime Intensity (Z-score)",
        )
        # Add threshold lines
        ax2.axhline(
            2.0, color="r", linestyle="--", lw=1, label="High Regime Threshold (+2.0)"
        )
        ax2.axhline(
            -2.0, color="r", linestyle=":", lw=1, label="Low Regime Threshold (-2.0)"
        )

        ax2.set_title("Daily Regime Intensity (Z-score of Tweet Count)", fontsize=16)
        ax2.set_ylabel("Z-score")
        ax2.set_xlabel("Date")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        print(
            "\n📊 Plot generated. The top plot shows monthly tweet volume. The bottom plot shows the daily Z-score, where values above +2.0 or below -2.0 indicate significant regime shifts.",
        )

    except Exception as e:
        print(f"\n❌ A fatal error occurred during visualization: {e}")


def find_and_visualize_top_changes():
    print("Loading unified tweet data...")
    df_tweets = load_unified_data()

    if df_tweets.empty:
        return

    # Use FeatureEngineer to get regime intensity (Z-score)
    feat_eng = FeatureEngineer()
    all_features = feat_eng.process_data(df_tweets)  # Corrected call

    # Consider only the last year for selecting top changes
    recent_features = all_features.tail(LOOKBACK_WEEKS * 7)  # Daily data

    # Greedy algorithm to find top N distinct changes based on Z-score
    candidates = recent_features.reindex(
        recent_features["regime_intensity"].abs().sort_values(ascending=False).index
    )

    top_changes = {}
    selected_dates = []

    for date, row in candidates.iterrows():
        if len(top_changes) >= NUM_CHANGES:
            break

        is_far_enough = True
        for sel_date in selected_dates:
            if abs((date - sel_date).days) < (MIN_WEEKS_SEPARATION * 7):
                is_far_enough = False
                break

        if is_far_enough:
            top_changes[date] = row["regime_intensity"]
            selected_dates.append(date)

    print(
        f"\nTop {NUM_CHANGES} distinct regime changes (>{MIN_WEEKS_SEPARATION} weeks apart) based on Z-score:",
    )
    for date, z_score in top_changes.items():
        print(f"- Week of {date.date()}: Z-score of {z_score:.2f}")

    # --- Visualization ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plotting weekly tweet counts for context
    weekly_counts = df_tweets.resample("W-MON", on="created_at").size()
    weekly_counts.tail(LOOKBACK_WEEKS).plot(
        ax=ax,
        label="Weekly Tweet Count",
        color="cornflowerblue",
        lw=2,
    )

    for date, z_score in top_changes.items():
        color = "green" if z_score > 0 else "red"
        ax.axvline(
            date,
            color=color,
            linestyle="--",
            lw=2,
            label=f"Shift on {date.date()} (Z-score: {z_score:+.2f})",
        )

    ax.set_title(
        f"Top {NUM_CHANGES} Distinct Regime Changes (Min Separation: {MIN_WEEKS_SEPARATION} weeks)",
        fontsize=16,
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(REGIME_CHANGE_PLOT_PATH)
    print(f"\n✅ Visualization saved to: {REGIME_CHANGE_PLOT_PATH}")


def visualize_multi_model_backtest():
    logger.info("📊 Starting visualization of multi-model backtest performance...")

    multi_model_performance_path = os.path.join(
        project_root, "data", "processed", "all_models_historical_performance.csv"
    )
    if not os.path.exists(multi_model_performance_path):
        logger.error(
            f"Multi-model performance data not found at '{multi_model_performance_path}'."
        )
        logger.error(
            "Please run `tools/model_analysis.py --task train_and_evaluate` first."
        )
        sys.exit(1)

    df = pd.read_csv(multi_model_performance_path)
    df["week_start"] = pd.to_datetime(df["week_start"])

    # Melt the dataframe to make it suitable for seaborn lineplot
    id_vars = ["week_start", "y_true"]
    value_vars = [col for col in df.columns if "y_pred" in col]
    df_melted = df.melt(
        id_vars=id_vars, value_vars=value_vars, var_name="model", value_name="y_pred"
    )
    df_melted["model"] = df_melted["model"].str.replace("y_pred_", "")

    plt.figure(figsize=(15, 8))

    # Plot y_true
    sns.lineplot(
        data=df,
        x="week_start",
        y="y_true",
        label="Actual Tweets (y_true)",
        color="black",
        marker="o",
        linewidth=2.5,
    )

    # Plot predictions for each model
    sns.lineplot(
        data=df_melted,
        x="week_start",
        y="y_pred",
        hue="model",
        marker="x",
        linestyle="--",
    )

    plt.title("Multi-Model Backtest Performance", fontsize=16)
    plt.xlabel("Week Start Date")
    plt.ylabel("Number of Tweets")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(project_root, "multi_model_backtest_plot.png")
    plt.savefig(output_path)
    logger.success(f"Multi-model backtest plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualization tools.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "visualize_predictions",
            "visualize_regime_change",
            "visualize_top_changes",
            "visualize_multi_model_backtest",
        ],
        help="The visualization task to execute.",
    )
    args = parser.parse_args()

    if args.task == "visualize_predictions":
        visualize_predictions_and_metrics()
    elif args.task == "visualize_regime_change":
        visualize_regime_change()
    elif args.task == "visualize_top_changes":
        find_and_visualize_top_changes()
    elif args.task == "visualize_multi_model_backtest":
        visualize_multi_model_backtest()


if __name__ == "__main__":
    main()
