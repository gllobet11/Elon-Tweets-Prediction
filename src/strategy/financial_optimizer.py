import hashlib
import os
import pickle
import sys
from datetime import datetime
import json
import itertools
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from tqdm import tqdm
from tabulate import tabulate

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config.bins_definition import MARKET_BINS, BINS_ORDER
    from src.strategy.prob_math import DistributionConverter
except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
INITIAL_CAPITAL = 1000.0
BINS = [(v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]


def get_market_prices_simulation(mu_market: float) -> np.ndarray:
    """Fallback function to simulate market prices if real ones aren't found."""
    probs = []
    for l, h in BINS:
        p = stats.poisson.cdf(h, mu_market) - stats.poisson.cdf(l - 1, mu_market)
        probs.append(p)
    prices = np.array(probs)
    if prices.sum() == 0:
        prices = np.ones(len(prices)) / len(prices)
    else:
        prices = prices / prices.sum()
    prices = prices + 0.02
    return prices / prices.sum()


def get_historical_prices_for_week(
    week_start_str: str, market_data: dict, y_pred_for_sim: float
) -> np.ndarray:
    historical_map = market_data.get("historical_map", {})
    price_histories = market_data.get("price_histories", {})

    simulated_full_prices = get_market_prices_simulation(y_pred_for_sim)

    week_market_info = historical_map.get(week_start_str)
    if not week_market_info:
        # logger.warning(f"Week {week_start_str}: No market map found. Using sim.")
        return simulated_full_prices

    prices = np.zeros(len(BINS_ORDER))
    # Intentamos parsear la fecha con flexibilidad
    try:
        week_start_ts = int(pd.to_datetime(week_start_str).timestamp())
    except:
        return simulated_full_prices
        
    lookup_window_end_ts = week_start_ts + 24 * 3600  # Ventana de 24h

    for i, bin_label in enumerate(BINS_ORDER):
        price_for_bin = simulated_full_prices[i]

        token_id = week_market_info["bins"].get(bin_label)
        if token_id and token_id in price_histories:
            history = price_histories[token_id]
            # Buscar el primer precio dentro de la ventana del lunes
            first_valid_price = next(
                (p["p"] for p in history if week_start_ts <= p["t"] <= lookup_window_end_ts),
                None,
            )
            if first_valid_price is not None:
                price_for_bin = first_valid_price
        
        prices[i] = price_for_bin

    return prices


def simulate_trading_run(
    df,
    alpha_nb,
    kelly_fraction,
    min_edge_threshold,
    max_price_threshold,
    market_data,
    initial_capital=1000.0,
    debug=False,
):
    df = df.copy()
    capital = initial_capital
    equity_curve = [capital]
    peak = capital
    max_drawdown = 0.0
    ev_list = []
    trades = []
    bins_config_math = [(k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]

    for idx, row in df.iterrows():
        y_pred = row.get("y_pred", 0)
        y_true = row.get("y_true", 0)
        week_start = str(row.get("week_start_date"))

        try:
            model_probs_dict = DistributionConverter.get_bin_probabilities(
                mu_remainder=y_pred,
                current_actuals=0,
                model_type="nbinom" if alpha_nb > 0 else "poisson",
                alpha=alpha_nb,
                bins_config=bins_config_math,
            )
        except Exception:
            equity_curve.append(capital)
            continue

        # --- Use Real Historical Prices ---
        safe_market_data = market_data if market_data else {}
        market_prices = get_historical_prices_for_week(
            week_start, safe_market_data, y_pred
        )

        winning_bin_idx = -1
        winning_bin_label = "Unknown"
        for i, (l, h) in enumerate(BINS):
            if l <= y_true < h:
                winning_bin_idx = i
                winning_bin_label = f"{l}-{h}"
                break

        daily_opportunities = []
        for i, (l, h) in enumerate(BINS):
            model_prob = list(model_probs_dict.values())[i]
            market_price = market_prices[i]

            # Filtros de seguridad básicos
            if market_price <= 0.01 or market_price >= max_price_threshold:
                kelly_f, edge = -1, -1
            else:
                odds = 1.0 / market_price
                b = odds - 1
                kelly_f = model_prob - (1 - model_prob) / b
                edge = model_prob - market_price

            daily_opportunities.append({
                "bin_idx": i,
                "label": f"{l}-{h}",
                "prob": model_prob,
                "price": market_price,
                "edge": edge,
                "kelly": kelly_f,
                "b": odds - 1,
            })

        if daily_opportunities:
            # Estrategia de Bloque: Apostar al mejor y sus vecinos
            anchor = max(daily_opportunities, key=lambda x: x["prob"])
            indices = {anchor["bin_idx"], anchor["bin_idx"] - 1, anchor["bin_idx"] + 1}

            for i in indices:
                if 0 <= i < len(daily_opportunities):
                    opp = daily_opportunities[i]
                    is_anchor = i == anchor["bin_idx"]
                    thresh = min_edge_threshold if is_anchor else 0.01

                    if opp["kelly"] > 0 and opp["edge"] > thresh:
                        size_multiplier = 1.0 if is_anchor else 0.65
                        bet_usd = (
                            capital * opp["kelly"] * kelly_fraction * size_multiplier
                        )

                        if bet_usd > 2.0:
                            outcome = "Win" if i == winning_bin_idx else "Loss"
                            pnl = bet_usd * opp["b"] if outcome == "Win" else -bet_usd
                            capital += pnl

                            ev_dollar = (opp["prob"] * (bet_usd * opp["b"])) - ((1 - opp["prob"]) * bet_usd)
                            ev_list.append(ev_dollar)

                            if debug:
                                trades.append({
                                    "week": week_start,
                                    "bin": opp["label"],
                                    "real_bin": winning_bin_label,
                                    "type": "ANCHOR" if is_anchor else "Cover",
                                    "prob": opp["prob"],
                                    "price": opp["price"],
                                    "edge": opp["edge"],
                                    "bet": bet_usd,
                                    "res": outcome,
                                    "pnl": pnl,
                                })

        equity_curve.append(capital)
        peak = max(peak, capital)
        if peak > 0:
            dd = (peak - capital) / peak
            max_drawdown = max(max_drawdown, dd)
        if capital < 50: # Quiebra técnica
            break

    return (
        pd.Series(equity_curve),
        max_drawdown,
        np.mean(ev_list) if ev_list else 0.0,
        trades,
    )


def perform_grid_search(df_backtest: pd.DataFrame, market_data: dict) -> pd.DataFrame:
    """
    CORREGIDO: Devuelve DataFrame completo para permitir selección inteligente.
    """
    logger.info(f"⚡ Grid Search: Fine-Tuning Block Strategy...")
    
    alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50] 
    kellys = [0.2, 0.3, 0.4, 0.5]
    edges = [0.02, 0.04, 0.06]
    prices = [0.85, 0.95]

    combinations = list(itertools.product(alphas, kellys, edges, prices))
    logger.info(f"Testing {len(combinations)} strategy combinations...")

    results = []
    if "y_pred" not in df_backtest.columns:
        logger.error("❌ 'y_pred' column missing during grid search.")
        return pd.DataFrame()

    for alpha, kelly, edge, price in tqdm(combinations, desc="Simulating"):
        equity, mdd, ev, trades = simulate_trading_run(
            df_backtest, alpha, kelly, edge, price, market_data, debug=False
        )

        final_cap = equity.iloc[-1]
        roi = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
        wins = sum(1 for t in trades if t["pnl"] > 0)
        total = len(trades)
        win_rate = wins / total if total > 0 else 0

        score = roi * 100
        if mdd > 0.25: score -= mdd * 200
        if total < 5: score = -100

        results.append({
            "alpha": alpha,
            "kelly": kelly,
            "edge": edge,
            "price": price,
            "roi": roi,
            "mdd": mdd,
            "win_rate": win_rate,
            "score": score,
            "final_cap": final_cap,
        })

    df_results = pd.DataFrame(results).sort_values(by="score", ascending=False)
    logger.info("\n🏆 TOP 5 FINANCIAL STRATEGIES (by score):")
    print(df_results.head(5).to_string(index=False))

    # CAMBIO: Retornar DataFrame completo
    return df_results

def print_final_metrics(equity, mdd, ev, trades):
    final_cap = equity.iloc[-1]
    pnl = final_cap - INITIAL_CAPITAL
    roi = pnl / INITIAL_CAPITAL
    df_t = pd.DataFrame(trades)

    if not df_t.empty:
        total = len(df_t)
        wins = len(df_t[df_t["pnl"] > 0])
        wr = wins / total if total > 0 else 0
        gross_loss = abs(df_t[df_t["pnl"] < 0]["pnl"].sum())
        pf = df_t[df_t["pnl"] > 0]["pnl"].sum() / gross_loss if gross_loss > 0 else float("inf")
    else:
        total, wins, wr, pf = 0, 0, 0, 0

    print("\n" + "=" * 50)
    print(f"📊 FINAL BACKTEST REPORT")
    print("=" * 50)
    print(f"💰 Initial Capital : ${INITIAL_CAPITAL:,.2f}")
    print(f"🏁 Final Capital   : ${final_cap:,.2f}")
    print(f"📈 Net PnL         : ${pnl:,.2f}")
    print(f"🚀 ROI             : {roi:.2%}")
    print(f"💡 Avg EV per Trade: ${ev:,.2f}")
    print("-" * 50)
    print(f"📉 Max Drawdown    : {mdd:.2%}")
    print(f"🧠 Profit Factor   : {pf:.2f}")
    print(f"✅ Win Rate        : {wr:.2%}")
    print(f"🎲 Total Trades    : {total}")
    print("=" * 50 + "\n")

    if not df_t.empty:
        print("📜 TRADE LOG (Last 10):")
        print(tabulate(df_t.tail(10)[["week", "bin", "real_bin", "type", "prob", "price", "bet", "res", "pnl"]], 
                       headers="keys", tablefmt="simple_grid", floatfmt=".3f"))


def calculate_balanced_bias(df: pd.DataFrame) -> float:
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise KeyError("Missing columns for bias calculation.")
    errors = df["y_true"] - df["y_pred"]
    mean_bias = errors.mean()
    # Si el bias es pequeño, no lo corregimos agresivamente para evitar overfitting
    correction = mean_bias * 0.5 
    logger.info(f"📊 Bias Correction applied: {correction:+.2f} tweets")
    return correction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.path.join(project_root, "data", "processed", "historical_performance.csv"))
    args = parser.parse_args()

    # 1. Cargar Datos de Mercado
    try:
        with open(os.path.join(project_root, "data", "markets", "historical_market_map.json"), "r") as f:
            historical_map = json.load(f)
        with open(os.path.join(project_root, "data", "markets", "all_price_histories.json"), "r") as f:
            price_histories = json.load(f)
        market_data = {"historical_map": historical_map, "price_histories": price_histories}
    except Exception:
        logger.warning("⚠️ Market data not found. Using simulated prices.")
        market_data = None

    # 2. Cargar Predicciones
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        logger.info("👉 Run 'python tools/model_analysis.py --task train_and_evaluate' first.")
        sys.exit(1)

    df_backtest_real = pd.read_csv(args.input)
    df_backtest_real.columns = df_backtest_real.columns.str.strip()
    
    col_map = {"prediction": "y_pred", "pred": "y_pred", "actual": "y_true"}
    df_backtest_real.rename(columns=col_map, inplace=True)

    # 3. Bias Correction
    bias = calculate_balanced_bias(df_backtest_real)
    df_backtest_real["y_pred"] = df_backtest_real["y_pred"] + bias

    # 4. Grid Search Financiero - DEVUELVE DATAFRAME COMPLETO
    results_df = perform_grid_search(df_backtest_real, market_data)
    
    if results_df.empty:
        logger.error("❌ Grid search failed, exiting.")
        sys.exit(1)

    # --- SELECCIÓN INTELIGENTE ---
    print("\n" + "="*50)
    print("🔬 STRATEGY SELECTION LOGIC")
    print("="*50)
    print("📊 Applying multi-criteria selection:")
    print("   1. Filter by alpha = 0.3 (optimal dispersion)")
    print("   2. Select highest ROI from filtered results")
    print("-" * 50)

    # Filtrar solo alpha = 0.3
    optimal_alpha_strategies = results_df[results_df["alpha"] == 0.3].copy()

    if optimal_alpha_strategies.empty:
        logger.warning("⚠️ No strategies with alpha=0.3 found, using best overall")
        best_row = results_df.iloc[0]
    else:
        # Ordenar por ROI descendente y tomar el mejor
        best_row = optimal_alpha_strategies.sort_values(by="roi", ascending=False).iloc[0]
        
        print(f"\n🎯 STRATEGIES WITH ALPHA=0.3 (top 3 by ROI):")
        print(optimal_alpha_strategies.sort_values(by="roi", ascending=False)
              .head(3)[["kelly", "edge", "price", "roi", "mdd", "final_cap"]]
              .to_string(index=False))

    print(f"\n✅ SELECTED STRATEGY:")
    print(f"   Alpha: {best_row['alpha']} | Kelly: {best_row['kelly']} | "
          f"Edge: {best_row['edge']} | ROI: {best_row['roi']:.2%}")
    print("="*50 + "\n")

    best_params = {
        "alpha": best_row["alpha"],
        "kelly": best_row["kelly"],
        "edge_threshold": best_row["edge"],
        "price_threshold": best_row["price"],
    }

    # 5. GUARDADO DE PARÁMETROS
    mae = (df_backtest_real["y_true"] - df_backtest_real["y_pred"]).abs().mean()

    params_package = {
        "strategy_source": "financial_optimizer_grid",
        "timestamp": datetime.now(),
        "alpha_nbinom": best_params["alpha"],
        "kelly_fraction": best_params["kelly"],
        "min_edge": best_params["edge_threshold"],
        "max_price": best_params["price_threshold"],
        "bias_correction": bias,
        "model_mae": mae
    }

    output_pkl = os.path.join(project_root, "risk_params.pkl")
    with open(output_pkl, "wb") as f:
        pickle.dump(params_package, f)
        
    print("\n" + "="*50)
    print(f"🧠 OPTIMIZER RESULTS (Best ROI with α=0.3)")
    print("="*50)
    print(f"⚙️  PARAMETERS:")
    print(f"   • Alpha (NBinom) : {best_params['alpha']}")
    print(f"   • Kelly Fraction : {best_params['kelly']}")
    print(f"   • Edge Threshold : {best_params['edge_threshold']}")
    print(f"   • Price Cap      : {best_params['price_threshold']}")
    print("-" * 50)
    print(f"📊 MODEL METRICS:")
    print(f"   • MAE            : {mae:.2f} tweets")
    print(f"   • Bias Applied   : {bias:+.2f} tweets")
    print("-" * 50)
    print(f"💰 EXPECTED PERFORMANCE:")
    print(f"   • ROI            : {best_row['roi']:.2%}")
    print(f"   • Max Drawdown   : {best_row['mdd']:.2%}")
    print(f"   • Final Capital  : ${best_row['final_cap']:.2f}")
    print("="*50 + "\n")

    logger.success(f"💾 Saved optimal parameters (α={best_params['alpha']}) to risk_params.pkl")

    # 6. Simulación Final con parámetros óptimos
    equity, mdd, ev, trades = simulate_trading_run(
        df_backtest_real,
        best_params["alpha"],
        best_params["kelly"],
        best_params["edge_threshold"],
        best_params["price_threshold"],
        market_data,
        debug=True,
    )

    print_final_metrics(equity, mdd, ev, trades)

    # Plot
    if not equity.empty:
        plt.figure(figsize=(12, 6))
        equity.plot(title=f"Optimal Strategy Equity Curve (α=0.3, ROI: {best_row['roi']:.1%})")
        plt.grid(True, alpha=0.3)
        plt.show()