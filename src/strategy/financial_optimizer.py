import hashlib
import os
import pickle
import sys
from datetime import datetime

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
) -> tuple[pd.Series, float]:
    """Simulates a full trading trajectory."""
    df = df.copy()
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    peak = capital
    max_drawdown = 0

    for _, row in df.iterrows():
        mu_mypred = row["y_pred"]
        y_true = row["y_true"]

        alpha_nb = max(alpha_nb, 1e-5)
        n_param = 1.0 / alpha_nb
        p_param = 1.0 / (1.0 + alpha_nb * mu_mypred)

        my_probs = []
        winning_bin = -1

        for i, (l, h) in enumerate(BINS):
            prob = stats.nbinom.cdf(h, n_param, p_param) - stats.nbinom.cdf(
                l - 1, n_param, p_param,
            )
            my_probs.append(prob)
            if l <= y_true <= h:
                winning_bin = i

        mu_market = row["y_pred"]
        market_prices = get_market_prices_simulation(mu_market)

        edges = np.array(my_probs) - market_prices
        best_idx = np.argmax(edges)
        edge = edges[best_idx]

        if edge > 0.05:
            price = market_prices[best_idx]
            my_prob = my_probs[best_idx]

            odds = 1.0 / price
            b = odds - 1
            if b <= 0:
                continue

            f_star = (my_prob * (b + 1) - 1) / b
            bet_size = max(0, f_star) * kelly_fraction
            bet_size = min(bet_size, 0.20)
            wager = capital * bet_size

            if best_idx == winning_bin:
                capital += wager * b
            else:
                capital -= wager

        equity_curve.append(capital)
        peak = max(peak, capital)
        dd = (peak - capital) / peak
        max_drawdown = max(max_drawdown, dd)

        if capital < 50:
            return pd.Series(equity_curve), 1.0

    return pd.Series(equity_curve), max_drawdown


def optimize_risk_params(
    df_backtest: pd.DataFrame,
) -> tuple[float, float, pd.DataFrame]:
    """Optimizes risk parameters (alpha and kelly_fraction)."""
    logger.info(f"⚡ Optimizing over {len(df_backtest)} weeks...")

    alphas = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    kellys = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = []

    for a in alphas:
        for k in kellys:
            equity_curve, max_drawdown = simulate_trading_run(df_backtest, a, k)

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
                {"alpha": a, "kelly": k, "score": calmar_ratio, "pnl": pnl, "roi": roi},
            )

    df_res = pd.DataFrame(results)

    best = df_res.loc[df_res["score"].idxmax()]
    logger.info("\n🏆 WINNING CONFIGURATION (Max Calmar Ratio):")
    logger.info(f"   Alpha (NB): {best['alpha']}")
    logger.info(f"   Kelly Mul : {best['kelly']}")
    logger.info(f"   Calmar    : {best['score']:.2f}")
    logger.info(f"   PnL ($)   : ${best['pnl']:.2f}")
    logger.info(f"   ROI       : {best['roi']:.2%}")

    return best["alpha"], best["kelly"], df_res


if __name__ == "__main__":
    backtest_data_path = os.path.join(
        project_root, "data", "processed", "historical_performance.csv",
    )

    logger.info(
        f"--- Input File Hash (SHA-256): {get_file_hash(backtest_data_path)} ---",
    )

    if not os.path.exists(backtest_data_path):
        logger.error(f"Backtest data not found at '{backtest_data_path}'.")
        logger.error(
            "Please run `tools/generate_historical_performance.py` first to create this file.",
        )
        sys.exit(1)

    logger.info(
        f"Loading pre-computed backtest data from '{os.path.basename(backtest_data_path)}'...",
    )
    df_backtest_real = pd.read_csv(backtest_data_path)

    optimal_alpha, optimal_kelly, df_results = optimize_risk_params(df_backtest_real)

    params = {
        "alpha": optimal_alpha,
        "kelly": optimal_kelly,
        "timestamp": datetime.now(),
    }
    with open("risk_params.pkl", "wb") as f:
        pickle.dump(params, f)
    logger.info("\n💾 Optimal risk parameters saved to 'risk_params.pkl'")

    logger.info("\n--- Final Simulation with Optimal Parameters ---")
    best_equity, best_mdd = simulate_trading_run(
        df_backtest_real, optimal_alpha, optimal_kelly,
    )

    final_capital = best_equity.iloc[-1]
    final_pnl = final_capital - INITIAL_CAPITAL
    final_roi = final_pnl / INITIAL_CAPITAL

    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    logger.info(f"Final Capital  : ${final_capital:.2f}")
    logger.info(f"Final PnL      : ${final_pnl:.2f}")
    logger.info(f"Final ROI      : {final_roi:.2%}")
    logger.info(f"Max Drawdown   : {best_mdd:.2%}")

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
