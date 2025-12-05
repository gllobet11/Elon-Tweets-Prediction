import hashlib
import os
import pickle
import sys
from datetime import datetime
import json
import itertools

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
    from config.bins_definition import MARKET_BINS
    from src.strategy.prob_math import DistributionConverter
except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
INITIAL_CAPITAL = 1000.0
BINS = [(v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]

def get_file_hash(filepath):
    if not os.path.exists(filepath): return "FILE_NOT_FOUND"
    with open(filepath, "rb") as f: return hashlib.sha256(f.read()).hexdigest()

def get_market_prices_simulation(mu_market: float) -> np.ndarray:
    probs = []
    for l, h in BINS:
        p = stats.poisson.cdf(h, mu_market) - stats.poisson.cdf(l - 1, mu_market)
        probs.append(p)
    prices = np.array(probs)
    if prices.sum() == 0: prices = np.ones(len(prices)) / len(prices)
    else: prices = prices / prices.sum()
    prices = prices + 0.02 
    return prices / prices.sum()

def simulate_trading_run(df, alpha_nb, kelly_fraction, min_edge_threshold, max_price_threshold, initial_capital=1000.0, debug=False):
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
        week_start = str(row.get("week_start_date", idx))

        try:
            model_probs_dict = DistributionConverter.get_bin_probabilities(
                mu_remainder=y_pred, current_actuals=0, model_type="nbinom" if alpha_nb > 0 else "poisson",
                alpha=alpha_nb, bins_config=bins_config_math
            )
        except:
            equity_curve.append(capital)
            continue

        simulated_prices = get_market_prices_simulation(y_pred)
        winning_bin_idx = -1
        winning_bin_label = "Unknown"
        for i, (l, h) in enumerate(BINS):
            if l <= y_true < h:
                winning_bin_idx = i
                winning_bin_label = f"{l}-{h}"
                break

        # --- OPPORTUNITIES ---
        daily_opportunities = []
        for i, (l, h) in enumerate(BINS):
            model_prob = list(model_probs_dict.values())[i]
            market_price = simulated_prices[i]
            
            if market_price <= 0.01 or market_price >= max_price_threshold:
                kelly_f, edge = -1, -1
            else:
                odds = 1.0 / market_price
                b = odds - 1
                kelly_f = model_prob - (1 - model_prob) / b
                edge = model_prob - market_price

            daily_opportunities.append({
                "bin_idx": i, "label": f"{l}-{h}", "prob": model_prob,
                "price": market_price, "edge": edge, "kelly": kelly_f, "b": odds - 1
            })

        # --- BLOCK STRATEGY (The Logic that worked best) ---
        if daily_opportunities:
            anchor = max(daily_opportunities, key=lambda x: x['prob'])
            # Look at Anchor + Neighbors
            indices = {anchor['bin_idx'], anchor['bin_idx']-1, anchor['bin_idx']+1}
            
            for i in indices:
                if 0 <= i < len(daily_opportunities):
                    opp = daily_opportunities[i]
                    is_anchor = (i == anchor['bin_idx'])
                    
                    # Strict threshold for anchor, permissive (1%) for neighbors (This was the key fix)
                    thresh = min_edge_threshold if is_anchor else 0.01 
                    
                    if opp['kelly'] > 0 and opp['edge'] > thresh:
                        # Sizing logic from the successful run
                        size_multiplier = 1.0 if is_anchor else 0.65 
                        bet_usd = capital * opp['kelly'] * kelly_fraction * size_multiplier
                        
                        if bet_usd > 2.0:
                            outcome = "Win" if i == winning_bin_idx else "Loss"
                            pnl = bet_usd * opp['b'] if outcome == "Win" else -bet_usd
                            capital += pnl
                            
                            ev_dollar = (opp['prob'] * (bet_usd * opp['b'])) - ((1 - opp['prob']) * bet_usd)
                            ev_list.append(ev_dollar)

                            if debug:
                                trades.append({
                                    "week": week_start, "bin": opp['label'], "real_bin": winning_bin_label,
                                    "type": "ANCHOR" if is_anchor else "Cover", 
                                    "prob": opp['prob'], "price": opp['price'], "edge": opp['edge'],
                                    "bet": bet_usd, "res": outcome, "pnl": pnl
                                })

        equity_curve.append(capital)
        peak = max(peak, capital)
        if peak > 0:
            dd = (peak - capital) / peak
            max_drawdown = max(max_drawdown, dd)
        if capital < 50: break

    return pd.Series(equity_curve), max_drawdown, np.mean(ev_list) if ev_list else 0.0, trades

def perform_grid_search(df_backtest: pd.DataFrame) -> dict:
    logger.info(f"⚡ Grid Search: Fine-Tuning Block Strategy...")

    # --- RANGOS ORIGINALES QUE DIERON EL MEJOR RESULTADO ---
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    kellys = [0.3, 0.4, 0.5, 0.6, 0.7]
    edges = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    prices = [0.90, 0.95]

    combinations = list(itertools.product(alphas, kellys, edges, prices))
    logger.info(f"Testing {len(combinations)} strategies...")

    results = []
    for alpha, kelly, edge, price in tqdm(combinations, desc="Simulating"):
        equity, mdd, ev, trades = simulate_trading_run(df_backtest, alpha, kelly, edge, price, debug=False)
        
        final_cap = equity.iloc[-1]
        roi = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
        
        wins = sum(1 for t in trades if t['pnl'] > 0)
        total = len(trades)
        win_rate = wins / total if total > 0 else 0

        # Score Balanceado Original
        score = roi * 100
        if win_rate > 0.25: score += 30 
        if mdd > 0.20: score -= (mdd * 100)
        if total < 5: score = -100

        results.append({
            "alpha": alpha, "kelly": kelly, "edge": edge, "price": price,
            "roi": roi, "mdd": mdd, "win_rate": win_rate, "score": score,
            "final_cap": final_cap
        })

    df_results = pd.DataFrame(results).sort_values(by="score", ascending=False)
    logger.info("\n🏆 TOP 5 STRATEGIES:")
    print(df_results.head(5).to_string(index=False))
    
    best = df_results.iloc[0]
    return {
        "alpha": best["alpha"], "kelly": best["kelly"],
        "edge_threshold": best["edge"], "price_threshold": best["price"],
    }

def print_final_metrics(equity, mdd, ev, trades):
    final_cap = equity.iloc[-1]
    pnl = final_cap - INITIAL_CAPITAL
    roi = pnl / INITIAL_CAPITAL
    df_t = pd.DataFrame(trades)
    
    if not df_t.empty:
        total = len(df_t)
        wins = len(df_t[df_t['pnl'] > 0])
        wr = wins / total
        gross_loss = abs(df_t[df_t['pnl'] < 0]['pnl'].sum())
        pf = df_t[df_t['pnl'] > 0]['pnl'].sum() / gross_loss if gross_loss > 0 else float('inf')
    else:
        total, wins, wr, pf = 0,0,0,0

    print("\n" + "="*50)
    print(f"📊 FINAL PERFORMANCE REPORT (Restored Best Strategy)")
    print("="*50)
    print(f"💰 Initial Capital : ${INITIAL_CAPITAL:,.2f}")
    print(f"🏁 Final Capital   : ${final_cap:,.2f}")
    print(f"📈 Net PnL         : ${pnl:,.2f}")
    print(f"🚀 ROI             : {roi:.2%}")
    print("-" * 50)
    print(f"📉 Max Drawdown    : {mdd:.2%}")
    print(f"🧠 Profit Factor   : {pf:.2f}")
    print(f"✅ Win Rate        : {wr:.2%}")
    print("="*50 + "\n")

    if not df_t.empty:
        print("📜 FULL TRADE LOG:")
        print(tabulate(df_t[['week', 'bin', 'real_bin', 'type', 'prob', 'price', 'bet', 'res', 'pnl']], 
                       headers="keys", tablefmt="simple_grid", floatfmt=".3f"))

def calculate_balanced_bias(df: pd.DataFrame) -> float:
    # Balanced Bias (0.25 MAE) - This was the sweet spot
    errors = df["y_true"] - df["y_pred"]
    mean_bias = errors.mean()
    mae = errors.abs().mean()
    correction = mean_bias + (mae * 0.25)
    logger.info(f"📊 Bias Correction: {correction:+.2f} tweets")
    return correction

if __name__ == "__main__":
    backtest_data_path = os.path.join(project_root, "data", "processed", "historical_performance.csv")
    if not os.path.exists(backtest_data_path): sys.exit(1)
    df_backtest_real = pd.read_csv(backtest_data_path)

    # 1. BIAS
    bias = calculate_balanced_bias(df_backtest_real)
    df_backtest_real["y_pred"] = df_backtest_real["y_pred"] + bias

    # 2. OPTIMIZACIÓN
    best_params = perform_grid_search(df_backtest_real)

    # Guardar
    mae = (df_backtest_real["y_true"] - df_backtest_real["y_pred"]).abs().mean()
    params = {
        "alpha": best_params["alpha"], "kelly": best_params["kelly"],
        "edge_threshold": best_params["edge_threshold"], "price_threshold": best_params["price_threshold"],
        "mae": mae, "timestamp": datetime.now(), "strategy": "RESTORED_BLOCK"
    }
    with open("risk_params.pkl", "wb") as f: pickle.dump(params, f)

    # 3. SIMULACIÓN
    equity, mdd, ev, trades = simulate_trading_run(
        df_backtest_real,
        best_params["alpha"], best_params["kelly"],
        best_params["edge_threshold"], best_params["price_threshold"],
        debug=True
    )

    print_final_metrics(equity, mdd, ev, trades)

    plt.figure(figsize=(12, 6))
    pd.Series(equity).plot(title=f"Restored Best Strategy (ROI: {(equity.iloc[-1]-1000)/10:.1f}%)")
    plt.grid(True, alpha=0.3)
    # plt.show()