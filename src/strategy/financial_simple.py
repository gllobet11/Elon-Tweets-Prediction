import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from loguru import logger
from tabulate import tabulate

# --- Configuración ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config.bins_definition import MARKET_BINS
except ImportError:
    pass

INITIAL_CAPITAL = 1000.0
BINS = [(v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]

def get_market_prices_simulation(mu_market: float) -> np.ndarray:
    """Simula precios de mercado para saber a cuánto compramos."""
    probs = []
    for l, h in BINS:
        p = stats.poisson.cdf(h, mu_market) - stats.poisson.cdf(l - 1, mu_market)
        probs.append(p)
    prices = np.array(probs)
    prices = prices / (prices.sum() + 1e-9)
    prices = prices + 0.02 # Spread
    return prices / prices.sum()

def calculate_simple_bias(df_historical: pd.DataFrame) -> float:
    """Calculates a simple bias from historical y_true and y_pred."""
    err = df_historical["y_true"] - df_historical["y_pred"]
    return err.mean() + (err.abs().mean() * 0.25)

def get_simple_strategy_recommendations(
    y_pred_current: float,
    current_capital: float,
    mode: str = "BLOCK",
    bet_pct: float = 0.10,
    historical_bias: float = 0.0,
    market_prices_override: np.ndarray = None,
) -> list[dict]:
    """
    Generates a list of recommended trades for a single prediction period
    based on a simplified strategy.

    Args:
        y_pred_current (float): The current week's prediction from the model.
        current_capital (float): The current available capital for betting.
        mode (str): "FOCUS" (bet on predicted bin) or "BLOCK" (predicted bin + neighbors).
        bet_pct (float): Percentage of current_capital to risk per main bet.
        historical_bias (float): Pre-calculated bias to apply to y_pred_current.
        market_prices_override (np.ndarray): Optional. Array of current market prices for bins.
                                            If None, prices are simulated.

    Returns:
        list[dict]: A list of trade recommendations, each with 'bin_label', 'wager', 'price', etc.
    """
    recommendations = []
    
    # 1. Aplicar Bias a la predicción actual
    y_pred_biased = y_pred_current + historical_bias

    # 2. Identificar Bin Objetivo (Target)
    target_idx = -1
    target_label = ""
    
    # Encontrar dónde cae la predicción numéricamente
    for i, (l, h) in enumerate(BINS):
        if l <= y_pred_biased < h:
            target_idx = i
            target_label = f"{l}-{h}"
            break
    
    # Si la predicción se sale de los bins (muy bajo o muy alto), ajustar al límite
    if target_idx == -1:
        if y_pred_biased < BINS[0][0]: target_idx = 0
        else: target_idx = len(BINS) - 1
        target_label = f"{BINS[target_idx][0]}-{BINS[target_idx][1]}"

    # 3. Definir Apuestas según Modo
    bets_to_consider = []
    
    if mode == "FOCUS":
        # Solo al bin predicho
        bets_to_consider.append({"idx": target_idx, "weight": 1.0, "type": "MAIN"})
        
    elif mode == "BLOCK":
        # Target (100%) + Vecinos (50%)
        bets_to_consider.append({"idx": target_idx, "weight": 1.0, "type": "MAIN"})
        # Vecino Abajo
        if target_idx > 0:
            bets_to_consider.append({"idx": target_idx - 1, "weight": 0.5, "type": "COVER"})
        # Vecino Arriba
        if target_idx < len(BINS) - 1:
            bets_to_consider.append({"idx": target_idx + 1, "weight": 0.5, "type": "COVER"})

    # 4. Obtener Precios de Mercado
    if market_prices_override is not None:
        market_prices = market_prices_override
    else:
        market_prices = get_market_prices_simulation(y_pred_biased)
    
    for bet in bets_to_consider:
        idx_bet = bet["idx"]
        price = market_prices[idx_bet]
        
        # Filtro de cordura: No comprar si es absurdo (> 85 cts)
        if price > 0.85: continue 
        
        # Tamaño apuesta: % del Capital actual * Peso
        wager = current_capital * bet_pct * bet["weight"]
        
        # Cap de seguridad ($150 max por trade individual)
        if wager > 150: wager = 150 # Max bet per bin
        
        if wager > 5: # Only recommend if wager is significant
            recommendations.append({
                "bin_idx": idx_bet,
                "bin_label": f"{BINS[idx_bet][0]}-{BINS[idx_bet][1]}",
                "type": bet["type"],
                "price": price,
                "wager_usd": wager,
                "implied_odds": 1.0 / price
            })
            
    return recommendations


def run_simple_strategy(df_historical: pd.DataFrame, mode: str = "BLOCK", bet_pct: float = 0.10):
    """
    Runs a backtest of the simplified strategy over historical data.
    """
    capital = INITIAL_CAPITAL
    equity = [capital]
    all_trades = []
    
    # Pre-calcular bias
    bias = calculate_simple_bias(df_historical)
    logger.info(f"Modo: {mode} | Bet Size: {bet_pct*100}% | Bias Applied: {bias:.2f}")

    for idx, row in df_historical.iterrows():
        y_pred = row["y_pred"]
        y_true = row["y_true"]
        week_start_str = str(row["week_start_date"])

        # Generate recommendations using the new function
        recommendations = get_simple_strategy_recommendations(
            y_pred_current=y_pred,
            current_capital=capital, # Use current capital for bet sizing
            mode=mode,
            bet_pct=bet_pct,
            historical_bias=bias,
            market_prices_override=None # Simulate prices for backtest
        )
        
        # Identify the real winning bin for this week (for trade evaluation)
        real_bin_label = "Unknown"
        for i, (l, h) in enumerate(BINS):
            if l <= y_true < h:
                real_bin_label = f"{l}-{h}"
                break

        # Execute simulated trades based on recommendations
        for trade_rec in recommendations:
            idx_bet = trade_rec["bin_idx"]
            price = trade_rec["price"]
            wager = trade_rec["wager_usd"]
            
            is_win = (trade_rec["bin_label"] == real_bin_label)
            
            if is_win:
                revenue = wager / price
                pnl = revenue - wager
                res = "WIN"
            else:
                pnl = -wager
                res = "LOSS"
            
            capital += pnl
            
            all_trades.append({
                "week": week_start_str,
                "pred": int(y_pred),
                "pred_biased": int(y_pred + bias), # Store biased prediction too
                "bet_bin": trade_rec["bin_label"],
                "real_bin": real_bin_label,
                "type": trade_rec["type"],
                "price": price,
                "wager": wager,
                "res": res,
                "pnl": pnl
            })
            
        equity.append(capital)
        if capital < 100: # Stop if capital falls too low
            logger.warning(f"Backtest stopped due to low capital (< $100) at week {week_start_str}")
            break

    return pd.Series(equity), all_trades

def print_results(equity, trades, title):
    final = equity.iloc[-1]
    roi = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    df_t = pd.DataFrame(trades)
    
    wins = len(df_t[df_t['pnl'] > 0])
    total = len(df_t)
    wr = wins/total if total > 0 else 0
    
    print(f"\n--- {title} ---")
    print(f"Final Capital: ${final:.2f}")
    print(f"ROI          : {roi:.2%}")
    print(f"Win Rate     : {wr:.2%}")
    print(f"Total Trades : {total}")
    
    if not df_t.empty:
        # Mostrar resumen agrupado por semana para ver el efecto neto
        weekly_pnl = df_t.groupby("week")['pnl'].sum()
        print("\nWeekly Net PnL:")
        print(weekly_pnl.tail(5)) # Ultimas 5 semanas
        
        print("\nFull Log (Last 10):")
        cols = ['week', 'pred', 'bet_bin', 'real_bin', 'type', 'price', 'res', 'pnl']
        print(tabulate(df_t[cols].tail(10), headers="keys", tablefmt="simple", floatfmt=".2f"))

if __name__ == "__main__":
    file_path = os.path.join(project_root, "data", "processed", "historical_performance.csv")
    if not os.path.exists(file_path):
        print("Data file not found.")
        sys.exit(1)
        
    df_historical = pd.read_csv(file_path)
    
    # Escenario 1: Focus (Solo apostar a lo que dice el modelo)
    # Arriesgamos 15% del banco por apuesta (Agresivo porque es solo 1 apuesta)
    eq_focus, tr_focus = run_simple_strategy(df_historical, mode="FOCUS", bet_pct=0.15)
    print_results(eq_focus, tr_focus, "STRATEGY: FOCUS (Single Bin)")
    
    # Escenario 2: Bloque (Target + Vecinos)
    # Arriesgamos 10% base (el central lleva 10%, vecinos 5%). Total riesgo ~20%
    eq_block, tr_block = run_simple_strategy(df_historical, mode="BLOCK", bet_pct=0.10)
    print_results(eq_block, tr_block, "STRATEGY: BLOCK (Target + Neighbors)")

    plt.figure(figsize=(10,6))
    pd.Series(eq_focus).plot(label="Focus Strategy")
    pd.Series(eq_block).plot(label="Block Strategy")
    plt.legend()
    plt.title("Simple Directional Strategies")
    plt.grid(True, alpha=0.3)
    plt.show() # Keep show for standalone execution