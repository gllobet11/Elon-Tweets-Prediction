import os
import json
import numpy as np
import pandas as pd
from loguru import logger

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == "tools":
    PROJECT_ROOT = os.path.dirname(current_dir)
else:
    PROJECT_ROOT = current_dir

MARKETS_DIR = os.path.join(PROJECT_ROOT, "data", "markets")
HISTORICAL_MAP_PATH = os.path.join(MARKETS_DIR, "historical_market_map.json")
PRICE_HISTORIES_PATH = os.path.join(MARKETS_DIR, "all_price_histories.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "eda")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "poly_daily_features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    if not os.path.exists(HISTORICAL_MAP_PATH) or not os.path.exists(PRICE_HISTORIES_PATH):
        logger.error("Missing historical data files.")
        return None, None
    
    with open(HISTORICAL_MAP_PATH, "r") as f:
        market_map = json.load(f)
    with open(PRICE_HISTORIES_PATH, "r") as f:
        price_histories = json.load(f)
        
    return market_map, price_histories

def _bin_label_to_midpoint(bin_label: str) -> float:
    try:
        if "+" in bin_label:
            return float(bin_label.replace("+", "")) + 5.0
        if "-" in bin_label:
            l, h = bin_label.split("-")
            return (float(l) + float(h)) / 2.0
        return float(bin_label)
    except:
        return np.nan

def generate_daily_features():
    market_map, price_histories = load_data()
    if not market_map:
        return

    logger.info("🚀 Starting generation of DAILY Polymarket features (Fixing Scale)...")

    all_ticks = []
    
    for week_start, meta in market_map.items():
        bins = meta.get("bins", {})
        for bin_label, token_id in bins.items():
            history = price_histories.get(token_id, [])
            if history:
                df_h = pd.DataFrame(history)
                df_h["t"] = pd.to_datetime(df_h["t"], unit="s", utc=True)
                
                # --- CORRECCIÓN DE ESCALA ---
                # Convertimos a float primero
                df_h["p"] = df_h["p"].astype(float)
                # Si el promedio es mayor a 1.0, asumimos BPS (5300) y dividimos.
                # Si es menor (0.53), es precio directo.
                if df_h["p"].mean() > 1.0:
                    df_h["p"] = df_h["p"] / 10000.0
                
                df_h["token_id"] = token_id
                df_h["week_start"] = week_start
                df_h["bin_label"] = bin_label
                all_ticks.append(df_h)

    if not all_ticks:
        logger.error("No price history found.")
        return

    logger.info("Creating Unified Tick DataFrame...")
    df_main = pd.concat(all_ticks, ignore_index=True)
    df_main["date"] = df_main["t"].dt.date
    
    # 2. Daily Means
    logger.info("Calculating Daily Mean Prices per Bin...")
    daily_means = df_main.groupby(["date", "week_start", "bin_label"])["p"].mean().unstack("bin_label")
    
    # 3. Volatility (Intraday)
    logger.info("Calculating Intraday Volatility...")
    daily_std = df_main.groupby(["date", "token_id"])["p"].std().groupby("date").mean()
    
    # 4. Features
    features_list = []
    for (date, week_start), row in daily_means.iterrows():
        row = row.fillna(0)
        if row.sum() > 0:
            probs = row / row.sum()
        else:
            continue 

        expected_val = 0
        for bin_label, prob in probs.items():
            mid = _bin_label_to_midpoint(bin_label)
            if not np.isnan(mid):
                expected_val += prob * mid
        
        p_valid = probs[probs > 0]
        entropy = -np.sum(p_valid * np.log(p_valid))
        max_prob = probs.max()
        min_prob = probs.min()
        vol = daily_std.get(date, 0.0)

        features_list.append({
            "date": pd.to_datetime(date).tz_localize("UTC"),
            "poly_implied_mean_tweets": expected_val,
            "poly_entropy": entropy,
            "poly_max_prob": max_prob,
            "poly_daily_vol": vol,
            "poly_prob_spread": max_prob - min_prob
        })

    # 5. Export
    df_features = pd.DataFrame(features_list)
    
    if df_features.empty:
        logger.warning("No features generated.")
        return

    if df_features["date"].duplicated().any():
        logger.info("⚠️ Overlapping dates detected. Aggregating by mean.")
        df_features = df_features.groupby("date").mean()
    else:
        df_features = df_features.set_index("date")

    df_features = df_features.sort_index()
    # Rellenar fines de semana o huecos
    df_features = df_features.asfreq("D").ffill(limit=2)
    
    df_features.to_csv(OUTPUT_FILE)
    logger.success(f"✅ Daily Granular Features saved to: {OUTPUT_FILE}")
    logger.info("\n" + df_features.tail().to_string())

if __name__ == "__main__":
    generate_daily_features()