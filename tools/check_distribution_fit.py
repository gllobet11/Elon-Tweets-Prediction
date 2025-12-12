import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from loguru import logger

# Path Configuration
project_root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.ingestion.unified_feed import load_unified_data
from src.processing.feature_eng import FeatureEngineer

# Rutas de Polymarket
MARKETS_DIR = os.path.join(project_root, "data", "markets")
HISTORICAL_MAP_PATH = os.path.join(MARKETS_DIR, "historical_market_map.json")
PRICE_HISTORIES_PATH = os.path.join(MARKETS_DIR, "all_price_histories.json")

def load_polymarket_consensus():
    """
    Calcula la distribución promedio histórica que ha tenido el mercado.
    Devuelve un diccionario {rango_tuits: probabilidad_media}.
    """
    if not os.path.exists(HISTORICAL_MAP_PATH) or not os.path.exists(PRICE_HISTORIES_PATH):
        logger.warning("⚠️ No se encontraron datos de Polymarket para comparar.")
        return None

    with open(HISTORICAL_MAP_PATH, "r") as f:
        market_map = json.load(f)
    with open(PRICE_HISTORIES_PATH, "r") as f:
        price_histories = json.load(f)

    # Acumulador de probabilidades por etiqueta de bin
    bin_probs_sum = {}
    bin_counts = {}

    for week_start, meta in market_map.items():
        bins = meta.get("bins", {})
        for bin_label, token_id in bins.items():
            history = price_histories.get(token_id, [])
            if not history:
                continue
            
            # Calculamos el precio promedio histórico de este bin
            # (Usamos todos los ticks disponibles para tener una visión global)
            prices = [p["p"] for p in history]
            if not prices: continue
            
            # Normalización rápida: Si > 1 es bps, si < 1 es decimal
            avg_raw = np.mean(prices)
            avg_price = avg_raw / 10000.0 if avg_raw > 1.0 else avg_raw
            
            bin_probs_sum[bin_label] = bin_probs_sum.get(bin_label, 0.0) + avg_price
            bin_counts[bin_label] = bin_counts.get(bin_label, 0) + 1

    # Calcular promedio
    avg_market_dist = {}
    for label, total_prob in bin_probs_sum.items():
        count = bin_counts[label]
        avg_market_dist[label] = total_prob / count

    # Normalizar para que sume 1.0 (aproximadamente)
    total_mass = sum(avg_market_dist.values())
    if total_mass > 0:
        avg_market_dist = {k: v/total_mass for k, v in avg_market_dist.items()}
        
    return avg_market_dist

def parse_bin_range(label):
    """Convierte '200-219' a (200, 219). '400+' a (400, 450)."""
    try:
        if "+" in label:
            start = int(label.replace("+", "").replace(",", ""))
            return start, start + 50 # Asumimos ancho 50 para el último bin
        if "-" in label:
            parts = label.split("-")
            return int(parts[0]), int(parts[1])
    except:
        pass
    return None, None

def analyze_tails():
    # 1. Cargar Datos Reales (Ground Truth)
    df_tweets = load_unified_data()
    df_features = FeatureEngineer().process_data(df_tweets)
    
    # Agrupar por semana real
    df_features["week_start"] = df_features.index.to_period("W-MON").start_time
    weekly_counts = df_features.groupby("week_start")["n_tweets"].sum()
    
    mu_real = weekly_counts.mean()
    var_real = weekly_counts.var()
    
    # 2. Métricas Estadísticas
    alpha_ideal = (var_real - mu_real) / (mu_real ** 2)
    
    print("\n" + "="*60)
    print("📊 DIAGNÓSTICO DE COLAS: Realidad vs Modelo vs Mercado")
    print("="*60)
    print(f"Media Real (Tweets/Semana) : {mu_real:.1f}")
    print(f"Varianza Real              : {var_real:.1f}")
    print(f"Alpha Ideal (Teórico)      : {max(0, alpha_ideal):.4f}")
    print("-" * 60)

    # 3. Preparar Plot
    plt.figure(figsize=(12, 7))
    x_max = int(max(weekly_counts) * 1.3)
    x = np.arange(min(weekly_counts) - 50, x_max)
    x = x[x >= 0] # No tweets negativos

    # --- A) REALIDAD (Histograma Gris) ---
    plt.hist(weekly_counts, bins=15, density=True, alpha=0.3, color='black', label='Realidad (Histórico)')

    # --- B) MODELO ACTUAL (Línea Roja) ---
    # CAMBIA ESTE VALOR POR EL QUE TE HAYA DADO OPTUNA O MANUALMENTE
    CURRENT_ALPHA = 0.0223  # <--- Tu valor actual (cámbialo si Optuna dio otro)
    
    def get_n_p(mu, alpha):
        if alpha <= 1e-9: return np.inf, 1.0 # Poisson limit
        n = 1 / alpha
        p = n / (n + mu)
        return n, p

    if CURRENT_ALPHA > 0:
        n_curr, p_curr = get_n_p(mu_real, CURRENT_ALPHA)
        pmf_curr = stats.nbinom.pmf(x, n_curr, p_curr)
        plt.plot(x, pmf_curr, 'r-', linewidth=2.5, label=f'Modelo Actual (alpha={CURRENT_ALPHA:.3f})')

    # --- C) MODELO IDEAL (Línea Azul Punteada) ---
    if alpha_ideal > 0:
        n_ideal, p_ideal = get_n_p(mu_real, alpha_ideal)
        pmf_ideal = stats.nbinom.pmf(x, n_ideal, p_ideal)
        plt.plot(x, pmf_ideal, 'b--', linewidth=2, label=f'Ajuste Ideal (alpha={alpha_ideal:.3f})')

    # --- D) POLYMARKET (Barras Verdes) ---
    market_dist = load_polymarket_consensus()
    if market_dist:
        # Convertir bins a densidad para plotear
        # Densidad = Probabilidad / Ancho_del_Bin
        x_poly = []
        y_poly = []
        widths = []
        
        sorted_bins = sorted(market_dist.keys(), key=lambda k: parse_bin_range(k)[0] if parse_bin_range(k)[0] else 0)
        
        for label in sorted_bins:
            low, high = parse_bin_range(label)
            if low is not None and high is not None:
                prob = market_dist[label]
                width = high - low + 1
                density = prob / width
                
                # Crear puntos para el step plot o barras
                x_poly.append(low)
                y_poly.append(density)
                widths.append(width)
        
        # Usamos bar plot transparente para ver el "bloque"
        plt.bar(x_poly, y_poly, width=widths, align='edge', color='green', alpha=0.2, edgecolor='green', label='Consenso Polymarket (Avg)')

    # Decoración
    plt.title("Comparativa de Distribuciones: ¿Quién entiende mejor el riesgo?", fontsize=14)
    plt.xlabel("Tweets Semanales")
    plt.ylabel("Densidad de Probabilidad")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.2)
    
    # Texto informativo
    plt.text(0.02, 0.95, f'Si la línea roja es más estrecha que las barras verdes,\nsubestimas el riesgo de mercado.', 
             transform=plt.gca().transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    plt.show()

if __name__ == "__main__":
    analyze_tails()