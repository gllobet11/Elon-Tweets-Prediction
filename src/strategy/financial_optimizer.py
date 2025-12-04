import hashlib
import os
import pickle
import sys
from datetime import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config.bins_definition import MARKET_BINS
    from config.settings import WEEKS_TO_VALIDATE
    from src.processing.feature_eng import FeatureEngineer
    from src.strategy.utils import get_last_complete_friday  # Corrected import
except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE) # Set numpy seed for reproducibility
INITIAL_CAPITAL = 1000.0
BINS = [(v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]


def get_file_hash(filepath):
    """Returns SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    with open(filepath, "rb") as f:
        bytes = f.read()
        return hashlib.sha256(bytes).hexdigest()


def get_market_prices_simulation(mu_market: float) -> np.ndarray:
    """Simulates market prices based on a Poisson distribution + spread."""
    probs = []
    for l, h in BINS:
        p = stats.poisson.cdf(h, mu_market) - stats.poisson.cdf(l - 1, mu_market)
        probs.append(p)

    prices = np.array(probs) + 0.015
    return prices / prices.sum()


def simulate_trading_run(
    df: pd.DataFrame, alpha_nb: float, kelly_fraction: float,
    week_to_market_map: dict, price_histories: dict
) -> tuple[pd.Series, float, float, list]:
    """Simulates a full trading trajectory using real prices where available and logs trades."""
    df = df.copy()
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    peak = capital
    max_drawdown = 0
    ev_values = []
    trades = [] # List to log each trade

    for _, row in df.iterrows():
        week_start_str = pd.to_datetime(row['week_start_date']).strftime('%Y-%m-%d')
        mu_mypred = row["y_pred"]
        y_true = row["y_true"]

        alpha_nb = max(alpha_nb, 1e-5)
        n_param = 1.0 / alpha_nb
        p_param = 1.0 / (1.0 + alpha_nb * mu_mypred)

        my_probs = []
        winning_bin = -1
        market_prices = None

        for i, (l, h) in enumerate(BINS):
            prob = stats.nbinom.cdf(h, n_param, p_param) - stats.nbinom.cdf(
                l - 1, n_param, p_param,
            )
            my_probs.append(prob)
            if l <= y_true <= h:
                winning_bin = i
        
        winning_bin_range = f"{BINS[winning_bin][0]}-{BINS[winning_bin][1]}" if winning_bin != -1 else "N/A"

        market_id = week_to_market_map.get(week_start_str)
        if market_id and market_id in price_histories and price_histories[market_id]:
            initial_price = price_histories[market_id][0]['p']
            most_likely_bin_index = np.argmax(my_probs)
            simulated_prices = get_market_prices_simulation(mu_mypred)
            if most_likely_bin_index < len(simulated_prices):
                adjustment_factor = initial_price / simulated_prices[most_likely_bin_index]
                market_prices = simulated_prices * adjustment_factor
                market_prices /= market_prices.sum()
                logger.debug(f"Week {week_start_str}: Using REAL price data (adjusted). Price: {initial_price:.3f}")

        if market_prices is None:
            market_prices = get_market_prices_simulation(mu_mypred)
            logger.debug(f"Week {week_start_str}: Using SIMULATED price data.")

        edges = np.array(my_probs) - market_prices
        best_idx = np.argmax(edges)
        edge = edges[best_idx]

        if edge > 0.05:
            price = market_prices[best_idx]
            my_prob = my_probs[best_idx]
            odds = 1.0 / price
            b = odds - 1

            if b > 0:
                ev_bet = my_prob * b - (1 - my_prob) * 1
                ev_values.append(ev_bet)
                f_star = (my_prob * (b + 1) - 1) / b
                bet_size = max(0, f_star) * kelly_fraction
                bet_size = min(bet_size, 0.20)
                wager = capital * bet_size
                
                outcome = "Win" if best_idx == winning_bin else "Loss"
                pnl = wager * b if outcome == "Win" else -wager

                trade_log = {
                    "week_start": week_start_str,
                    "selected_bin": f"{BINS[best_idx][0]}-{BINS[best_idx][1]}",
                    "actual_bin": winning_bin_range,
                    "model_prob": my_prob,
                    "market_price": price,
                    "edge": edge,
                    "bet_size_usd": wager,
                    "outcome": outcome,
                    "pnl": pnl
                }
                trades.append(trade_log)

                capital += pnl

        equity_curve.append(capital)
        peak = max(peak, capital)
        dd = (peak - capital) / peak
        max_drawdown = max(max_drawdown, dd)

        if capital < 50:
            return pd.Series(equity_curve), 1.0, np.mean(ev_values) if ev_values else 0.0, trades

    return pd.Series(equity_curve), max_drawdown, np.mean(ev_values) if ev_values else 0.0, trades


def optimize_risk_params(
    df_backtest: pd.DataFrame, week_to_market_map: dict, price_histories: dict
) -> tuple[float, float, pd.DataFrame]:
    """Optimizes risk parameters (alpha and kelly_fraction)."""
    logger.info(f"⚡ Optimizing over {len(df_backtest)} weeks...")

    alphas = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    kellys = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = []

    for a in alphas:
        for k in kellys:
            equity_curve, max_drawdown, avg_ev, _ = simulate_trading_run(df_backtest, a, k, week_to_market_map, price_histories)

            final_capital = equity_curve.iloc[-1]
            pnl = final_capital - INITIAL_CAPITAL
            roi = pnl / INITIAL_CAPITAL

            if max_drawdown == 0:
                calmar_ratio = roi * 10
            elif max_drawdown >= 1.0:
                calmar_ratio = -1.0
            else:
                calmar_ratio = roi / max_drawdown

            results.append(
                {"alpha": a, "kelly": k, "score": calmar_ratio, "pnl": pnl, "roi": roi, "avg_ev": avg_ev},
            )

    df_res = pd.DataFrame(results)

    best = df_res.sort_values(by=["score", "alpha", "kelly"], ascending=[False, True, True]).iloc[0]
    logger.info("\n🏆 WINNING CONFIGURATION (Max Calmar Ratio):")
    logger.info(f"   Alpha (NB): {best['alpha']}")
    logger.info(f"   Kelly Mul : {best['kelly']}")
    logger.info(f"   Calmar    : {best['score']:.2f}")
    logger.info(f"   PnL ($)   : ${best['pnl']:.2f}")
    logger.info(f"   ROI       : {best['roi']:.2%}")
    logger.info(f"   Avg EV    : {best['avg_ev']:.4f}")

    return best["alpha"], best["kelly"], df_res


if __name__ == "__main__":
    # --- Load Data ---
    backtest_data_path = os.path.join(
        project_root, "data", "processed", "historical_performance.csv",
    )
    week_map_path = os.path.join(project_root, "data", "markets", "week_to_market_id_map.json")
    price_history_path = os.path.join(project_root, "data", "markets", "all_price_histories.json")

    logger.info(
        f"--- Input File Hash (SHA-256): {get_file_hash(backtest_data_path)} ---",
    )

    if not os.path.exists(backtest_data_path):
        logger.error(f"Backtest data not found at '{backtest_data_path}'.")
        sys.exit(1)

    logger.info(f"Loading pre-computed backtest data from '{os.path.basename(backtest_data_path)}'...")
    df_backtest_real = pd.read_csv(backtest_data_path)

    # --- Apply Bias Correction ---
    try:
        with open("bias_correction.txt", "r") as f:
            bias_correction_factor = float(f.read())
        
        df_backtest_real['y_pred'] = df_backtest_real['y_pred'] + bias_correction_factor
        logger.info(f"Applying bias correction of {bias_correction_factor:.2f} to all predictions.")
    except (FileNotFoundError, ValueError):
        logger.warning("bias_correction.txt not found or invalid. Running optimizer without bias correction.")

    # --- Load Real Price Data ---
    try:
        with open(week_map_path, "r") as f:
            week_to_market_map = json.load(f)
        logger.success(f"Loaded {len(week_to_market_map)} week-to-market ID mappings.")
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Week-to-market map not found or invalid. Proceeding with simulation only.")
        week_to_market_map = {}

    try:
        with open(price_history_path, "r") as f:
            price_histories = json.load(f)
        logger.success(f"Loaded price histories for {len(price_histories)} markets.")
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Price histories not found or invalid. Proceeding with simulation only.")
        price_histories = {}

    # --- Run Optimization ---
    optimal_alpha, optimal_kelly, df_results = optimize_risk_params(
        df_backtest_real, week_to_market_map, price_histories
    )

    params = {
        "alpha": optimal_alpha,
        "kelly": optimal_kelly,
        "timestamp": datetime.now(),
    }
    with open("risk_params.pkl", "wb") as f:
        pickle.dump(params, f)
    logger.info("\n💾 Optimal risk parameters saved to 'risk_params.pkl'")

    # --- Run Final Simulation ---
    logger.info("\n--- Final Simulation with Optimal Parameters ---")
    best_equity, best_mdd, best_avg_ev, trades_log = simulate_trading_run(
        df_backtest_real, optimal_alpha, optimal_kelly, week_to_market_map, price_histories
    )

    final_capital = best_equity.iloc[-1]
    final_pnl = final_capital - INITIAL_CAPITAL
    final_roi = final_pnl / INITIAL_CAPITAL

    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    logger.info(f"Final Capital  : ${final_capital:.2f}")
    logger.info(f"Final PnL      : ${final_pnl:.2f}")
    logger.info(f"Final ROI      : {final_roi:.2%}")
    logger.info(f"Max Drawdown   : {best_mdd:.2%}")
    logger.info(f"Avg EV (final) : {best_avg_ev:.4f}")

    # --- Display Trades Log ---
    if trades_log:
        logger.info("\n--- Trade Operations Log ---")
        df_trades = pd.DataFrame(trades_log)
        # Convert numeric columns to appropriate types for formatting
        for col in ['model_prob', 'market_price', 'edge', 'bet_size_usd', 'pnl']:
            df_trades[col] = pd.to_numeric(df_trades[col])
        
        # Use tabulate for clean printing
        from tabulate import tabulate
        print(tabulate(df_trades, headers='keys', tablefmt='psql', floatfmt=(".4f", ".4f", ".4f", ".2f", ".2f")))
    else:
        logger.info("No trades were executed in the final simulation.")

    # Plotting section
    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(
        df_results.pivot(index="alpha", columns="kelly", values="score"),
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
    )
    ax.set_title("Calmar Ratio (Score)")
    plt.xlabel("Kelly Fraction")
    plt.ylabel("Alpha (NBinom)")
    # plt.show()

    if "week_start_date" in df_backtest_real.columns:
        start_date = pd.to_datetime(df_backtest_real["week_start_date"].iloc[0])
    else:
        start_date = datetime.now() - pd.DateOffset(weeks=len(df_backtest_real))

    equity_index = pd.date_range(
        start=start_date, periods=len(best_equity), freq="W-FRI",
    )
    equity_series_to_plot = pd.Series(best_equity.values, index=equity_index)

    plt.figure(figsize=(12, 6))
    equity_series_to_plot.plot(
        label=f"Alpha={optimal_alpha}, Kelly={optimal_kelly}\nFinal ROI: {final_roi:.2%}",
        drawstyle="steps-post",
    )
    plt.title(f"Optimal Equity Curve (Max Drawdown: {best_mdd * 100:.1f}%)")
    plt.ylabel("Capital ($)")
    plt.xlabel("Backtest Weeks")
    plt.grid(True, which="major", linestyle="--")
    plt.legend()
    plt.tight_layout()
    # plt.show()
