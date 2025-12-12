import os
import sys
import json
import argparse
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
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
INITIAL_CAPITAL = 1000.0
BINS = [(v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]

# Cargar parámetros de riesgo optimizados (si existen) para obtener el Alpha correcto
RISK_PARAMS_PATH = os.path.join(project_root, "risk_params.pkl")
try:
    with open(RISK_PARAMS_PATH, "rb") as f:
        RISK_PARAMS = pickle.load(f)
    # Valores por defecto si falla la carga parcial
    ALPHA_NB = RISK_PARAMS.get("alpha_nbinom", RISK_PARAMS.get("alpha", 0.3))
    BIAS_CORRECTION = RISK_PARAMS.get("bias_correction", RISK_PARAMS.get("bias", 0.0))
except Exception:
    logger.warning("⚠️ risk_params.pkl not found. Using defaults.")
    ALPHA_NB = 0.3
    BIAS_CORRECTION = 0.0


def get_market_prices_simulation(mu_market: float) -> np.ndarray:
    from scipy import stats
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


def get_historical_prices_for_week(week_start_str: str, market_data: dict, y_pred_sim: float) -> np.ndarray:
    """Intenta obtener precios reales, fallback a simulación."""
    if not market_data:
        return get_market_prices_simulation(y_pred_sim)
        
    hist_map = market_data.get("historical_map", {})
    price_hist = market_data.get("price_histories", {})
    
    # Fallback simulation
    sim_prices = get_market_prices_simulation(y_pred_sim)
    
    week_info = hist_map.get(week_start_str)
    if not week_info:
        return sim_prices

    prices = np.zeros(len(BINS_ORDER))
    try:
        week_ts = int(pd.to_datetime(week_start_str).timestamp())
    except:
        return sim_prices
        
    for i, label in enumerate(BINS_ORDER):
        prices[i] = sim_prices[i] # Default
        tid = week_info["bins"].get(label)
        if tid and tid in price_hist:
            # Buscar precio cercano al lunes
            for p in price_hist[tid]:
                if p["t"] >= week_ts:
                    prices[i] = p["p"]
                    break
    return prices


def run_simple_strategy(df, mode="conservative", bet_size_pct=0.10, market_data=None):
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    
    # Aplicar corrección de bias globalmente
    df = df.copy()
    
    # --- CORRECCIÓN DE NOMBRES DE COLUMNAS ---
    df.columns = df.columns.str.strip()
    col_map = {
        "week_start_date": "week_start",
        "prediction": "y_pred", 
        "pred": "y_pred",
        "actual": "y_true"
    }
    df.rename(columns=col_map, inplace=True)
    # -----------------------------------------

    # Aplicar Bias del modelo (si existe)
    if "y_pred" in df.columns:
        df["y_pred"] = df["y_pred"] + BIAS_CORRECTION
    
    # Configuración según modo
    if mode == "aggressive":
        edge_thresh = 0.02
        kelly_fraction = 0.5
    elif mode == "focus":
        edge_thresh = 0.05
        kelly_fraction = 0.4
    else: # Conservative
        edge_thresh = 0.08
        kelly_fraction = 0.2

    bins_config_math = [(k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]

    logger.info(f"Mode: {mode.upper()} | Bias Applied: {BIAS_CORRECTION:.2f}")

    for _, row in df.iterrows():
        if "week_start" not in row:
            continue
            
        week_start_str = str(row["week_start"])
        y_pred = row["y_pred"]
        y_true = row["y_true"]

        # 1. Calcular Probabilidades del Modelo (Tu Edge)
        try:
            probs = DistributionConverter.get_bin_probabilities(
                mu_remainder=y_pred,
                current_actuals=0,
                model_type="nbinom",
                alpha=ALPHA_NB, # Usamos el alpha optimizado
                bins_config=bins_config_math
            )
        except:
            equity_curve.append(capital)
            continue

        # 2. Obtener Precios del Mercado (El Oponente)
        market_prices = get_historical_prices_for_week(week_start_str, market_data, y_pred)

        # 3. Encontrar Oportunidades
        opportunities = []
        winning_bin_idx = -1
        
        # Identificar bin ganador real
        for i, (l, h) in enumerate(BINS):
            if l <= y_true < h:
                winning_bin_idx = i
                break

        for i, (l, h) in enumerate(BINS):
            model_p = list(probs.values())[i]
            market_p = market_prices[i]
            
            if market_p < 0.01 or market_p > 0.98: continue
            
            edge = model_p - market_p
            # Kelly Simple
            odds = (1.0 / market_p) - 1
            kelly = model_p - (1 - model_p) / odds
            
            if edge > edge_thresh and kelly > 0:
                opportunities.append({
                    "idx": i,
                    "bin": f"{l}-{h}",
                    "edge": edge,
                    "kelly": kelly,
                    "odds": odds,
                    "prob": model_p
                })

        # 4. Ejecutar Trades (Solo el mejor de la semana)
        if opportunities:
            # Ordenar por Kelly
            best_opp = sorted(opportunities, key=lambda x: x["kelly"], reverse=True)[0]
            
            # Tamaño de apuesta
            wager = capital * best_opp["kelly"] * kelly_fraction
            wager = min(wager, capital * 0.20) 

            if wager > 5.0:
                is_win = (best_opp["idx"] == winning_bin_idx)
                pnl = (wager * best_opp["odds"]) if is_win else -wager
                capital += pnl
                
                trades.append({
                    "Week": week_start_str,
                    "Bin": best_opp["bin"],
                    "Bet": wager,
                    "Result": "WIN" if is_win else "LOSS",
                    "PnL": pnl,
                    "Capital": capital
                })

        equity_curve.append(capital)

    # Métricas
    equity_series = pd.Series(equity_curve)
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    mdd = 0.0
    peak = equity_series.cummax()
    if not peak.empty and peak.max() > 0:
        drawdown = (equity_series - peak) / peak
        mdd = abs(drawdown.min())
    
    pf = float('inf')
    if trades:
        df_t = pd.DataFrame(trades)
        gross_win = df_t[df_t["PnL"] > 0]["PnL"].sum()
        gross_loss = abs(df_t[df_t["PnL"] < 0]["PnL"].sum())
        if gross_loss > 0:
            pf = gross_win / gross_loss

    return equity_series, total_return, mdd, pf, trades


def calculate_simple_bias(df):
    """Alias para compatibilidad con el dashboard."""
    # Simplemente calcula media de errores
    if "y_true" in df.columns and "y_pred" in df.columns:
        return (df["y_true"] - df["y_pred"]).mean()
    return 0.0


def get_simple_strategy_recommendations(
    y_pred_current: float,
    current_capital: float,
    mode: str = "focus",
    bet_pct: float = 0.10,
    historical_bias: float = 0.0,
    market_prices_override: np.ndarray = None
) -> list:
    """
    Calcula las recomendaciones de apuestas en vivo para el Dashboard.
    """
    # 1. Configuración de Estrategia
    mode = mode.lower()
    if mode == "aggressive":
        edge_thresh = 0.02
        kelly_fraction = 0.5
    elif mode == "focus":
        edge_thresh = 0.05
        kelly_fraction = 0.4
    else:  # Conservative
        edge_thresh = 0.08
        kelly_fraction = 0.2

    # Override de bet_pct si se pasa explícitamente (aunque aquí usamos Kelly)
    # Si quisieras usar bet_pct fijo, descomenta o ajusta la lógica abajo.

    bins_config_math = [(k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()]
    
    # 2. Aplicar Bias
    y_pred_adj = y_pred_current + historical_bias

    # 3. Calcular Probabilidades
    try:
        probs_dict = DistributionConverter.get_bin_probabilities(
            mu_remainder=y_pred_adj,
            current_actuals=0,
            model_type="nbinom",
            alpha=ALPHA_NB, # Usa la global cargada al inicio del script
            bins_config=bins_config_math
        )
    except Exception as e:
        logger.error(f"Error calculating probabilities: {e}")
        return []

    # 4. Analizar Oportunidades
    recommendations = []
    
    for i, (bin_label, _, _) in enumerate(bins_config_math):
        model_p = probs_dict.get(bin_label, 0.0)
        
        if market_prices_override is not None:
            market_p = market_prices_override[i]
        else:
            market_p = 0.0 # No podemos calcular edge sin precio
            
        if market_p < 0.01 or market_p > 0.98:
            continue

        edge = model_p - market_p
        odds = (1.0 / market_p) - 1
        kelly = model_p - (1 - model_p) / odds

        if edge > edge_thresh and kelly > 0:
            # Tamaño de apuesta
            wager = current_capital * kelly * kelly_fraction
            wager = min(wager, current_capital * 0.20) # Cap de seguridad

            if wager > 2.0:
                recommendations.append({
                    "bin_label": bin_label,
                    "wager_usd": wager,
                    "edge": edge,
                    "kelly_full": kelly,
                    "model_prob": model_p,
                    "market_price": market_p
                })
    
    # Ordenar por Kelly (mejores oportunidades primero)
    recommendations = sorted(recommendations, key=lambda x: x['kelly_full'], reverse=True)
    
    return recommendations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.path.join(project_root, "data", "processed", "historical_performance.csv"))
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error("Input file not found.")
        sys.exit(1)

    # Cargar datos de mercado
    try:
        with open(os.path.join(project_root, "data", "markets", "historical_market_map.json"), "r") as f:
            hm = json.load(f)
        with open(os.path.join(project_root, "data", "markets", "all_price_histories.json"), "r") as f:
            ph = json.load(f)
        market_data = {"historical_map": hm, "price_histories": ph}
        logger.info("✅ Successfully loaded real historical market price data.")
    except:
        market_data = None

    logger.info(f"\n--- Running Backtest on '{os.path.basename(args.input)}' ---")
    df_raw = pd.read_csv(args.input)

    results = []
    strategies = ["conservative", "focus", "aggressive"]
    all_trades_history = {}  # <--- NUEVO: Almacén temporal de trades
    
    plt.figure(figsize=(10, 6))
    
    for strat in strategies:
        # <--- CAMBIO AQUÍ: Capturamos 'trades_list' en lugar de usar '_'
        eq, ret, mdd, pf, trades_list = run_simple_strategy(df_raw, mode=strat, market_data=market_data)
        
        # Guardamos los trades para consultarlos luego
        all_trades_history[strat] = trades_list
        
        label = f"{strat.title()} (ROI: {ret:.1%} | MDD: {mdd:.1%})"
        plt.plot(eq, label=label)
        results.append({"Strategy": strat, "ROI": ret, "MDD": mdd, "PF": pf})

    plt.title("Estrategias de Apuestas: Curvas de Capital")
    plt.ylabel("Capital ($)")
    plt.xlabel("Semanas")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    print("\n" + pd.DataFrame(results).to_markdown(index=False, floatfmt=".2f"))

    
    # --- Calcular MAE antes del reporte ---
    df_stats = df_raw.copy()
    df_stats.columns = df_stats.columns.str.strip()
    col_map = {
        "week_start_date": "week_start",
        "prediction": "y_pred", 
        "pred": "y_pred",
        "actual": "y_true"
    }
    df_stats.rename(columns=col_map, inplace=True)

    mae = 0.0
    if "y_true" in df_stats.columns and "y_pred" in df_stats.columns:
        mae = (df_stats["y_true"] - df_stats["y_pred"]).abs().mean()

    # --- SELECCIÓN DEL GANADOR ---
    best_strat = max(results, key=lambda x: x["ROI"])
    strat_name = best_strat["Strategy"]

    presets = {
        "conservative": {"kelly": 0.2, "edge": 0.08},
        "focus":        {"kelly": 0.4, "edge": 0.05},
        "aggressive":   {"kelly": 0.5, "edge": 0.02}
    }
    winner_config = presets[strat_name]

    # --- NUEVO: REPORTE FINAL DETALLADO ---
    print("\n" + "="*50)
    print(f"🏆 WINNING STRATEGY SELECTED: {strat_name.upper()}")
    print("="*50)
    print(f"⚙️  PARAMETERS:")
    print(f"   • Kelly Fraction : {winner_config['kelly']}")
    print(f"   • Min Edge       : {winner_config['edge']}")
    print(f"   • Alpha (NBinom) : {ALPHA_NB:.2f}")
    print("-" * 50)
    print(f"📊 MODEL METRICS:")
    print(f"   • MAE            : {mae:.2f} tweets")
    print(f"   • Bias Applied   : {BIAS_CORRECTION:+.2f} tweets")
    print("-" * 50)

    winning_trades = all_trades_history[strat_name]
    if winning_trades:
        df_win = pd.DataFrame(winning_trades)
        # Mostrar TODAS las operaciones de la estrategia ganadora
        print(f"📜 FULL TRADE LOG ({len(df_win)} trades):")
        print(tabulate(df_win[["Week", "Bin", "Bet", "Result", "PnL", "Capital"]], 
                    headers="keys", tablefmt="simple_grid", floatfmt=".2f"))
    else:
        print("⚠️ No trades executed by the winning strategy.")
    print("="*50 + "\n")

    # Guardar Configuración
    params_package = {
        "strategy_source": f"financial_simple_{strat_name}",
        "timestamp": datetime.now(),
        "alpha_nbinom": ALPHA_NB,
        "kelly_fraction": winner_config["kelly"],
        "min_edge": winner_config["edge"],
        "max_price": 0.95,
        "bias_correction": BIAS_CORRECTION,
        "model_mae": mae  # ✅ Ahora incluye MAE real
    }

    output_pkl = os.path.join(project_root, "risk_params.pkl")
    with open(output_pkl, "wb") as f:
        pickle.dump(params_package, f)
        
    logger.success(f"💾 Saved '{strat_name}' parameters to risk_params.pkl")
    