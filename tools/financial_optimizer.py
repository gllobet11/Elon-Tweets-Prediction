"""
financial_optimizer.py

Este script es el corazromUtf8bfn de la estrategia de optimizaciromUtf83n financiera.
Realiza un backtest walk-forward utilizando el mejor modelo de Prophet
(previamente seleccionado por `models_evals.py`) para simular la trayectoria
de un capital.

El objetivo principal es encontrar la combinaciromUtf83n romUtf83ptima de dos hiperparromUtf83metros:
1. `alpha` (parromUtf83metro de dispersiromUtf83n para la distribuciromUtf83n Negative Binomial).
2. `kelly_fraction` (fracciromUtf83n del Criterio de Kelly para el tamaromUtf83o de la apuesta).

Estos parromUtf83metros se optimizan para maximizar el "Calmar Ratio" (Retorno / MromUtf83ximo Drawdown),
una mromUtf83trica clave de rendimiento ajustado al riesgo. Los parromUtf83metros romUtf83ptimos se guardan
para ser utilizados en el dashboard de producciromUtf83n (`main.py`).

El script tambiromUtf83n genera visualizaciones (heatmaps y curva de equidad) para
entender el rendimiento a travromUtf83s de diferentes combinaciones de parromUtf83metros.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from prophet import Prophet
from datetime import timedelta, datetime
import pickle
import glob
from loguru import logger

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config.bins_definition import MARKET_BINS
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"Error de importaciromUtf83n en financial_optimizer.py: {e}")
    sys.exit(1)

# --- CONFIGURACIromUtf83N ---
INITIAL_CAPITAL = 1000.0
BINS = [(v['lower'], v['upper']) for k, v in MARKET_BINS.items()]
WEEKS_TO_VALIDATE = 12

def get_last_complete_friday(last_data_date: pd.Timestamp) -> datetime:
    """
    Encuentra el romUtf83ltimo viernes completo que puede iniciar una ventana de pronromUtf83stico de 7 dromUtf83as,
    asegurando que se disponga de datos de verdad fundamental.

    Args:
        last_data_date (pd.Timestamp): La fecha del romUtf83ltimo punto de datos disponible.

    Returns:
        datetime: Un objeto datetime que representa el romUtf83ltimo viernes completo.
    """
    if isinstance(last_data_date, pd.Timestamp):
        last_data_date = last_data_date.to_pydatetime()
    if last_data_date.tzinfo is not None:
        last_data_date = last_data_date.replace(tzinfo=None)

    last_possible_forecast_start = last_data_date - timedelta(days=6)
    days_since_friday = (last_possible_forecast_start.weekday() - 4) % 7
    return last_possible_forecast_start - timedelta(days=days_since_friday)


def generate_backtest_predictions(weeks_to_validate: int) -> pd.DataFrame:
    """
    Genera predicciones de backtest utilizando la configuraciromUtf83n del MEJOR modelo
    encontrado por `models_evals.py`.
    """
    logger.info("⚙️  Generando predicciones de backtest para optimizaciromUtf83n financiera...")
    
    try:
        model_files = glob.glob('best_prophet_model_*.pkl')
        if not model_files:
            raise FileNotFoundError("No se encontrromUtf83 ninguna archivo de modelo .pkl. Ejecuta `tools/models_evals.py` primero.")
        
        latest_model_path = max(model_files, key=os.path.getmtime)
        with open(latest_model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        regressors = model_data.get('regressors', [])
        model_name = model_data.get('model_name', 'Desconocido')
        logger.info(f"   -> Usando configuraciromUtf83n del modelo '{model_name}' desde '{os.path.basename(latest_model_path)}'")
        if regressors:
            logger.info(f"   -> Regresores: {regressors}")
        else:
            logger.info("   -> Modelo sin regresores (baseline).")
            
    except Exception as e:
        logger.error(f"   ❌ Error al cargar el modelo: {e}")
        logger.warning("   -> Usando configuraciromUtf83n de regresores por defecto como fallback.")
        regressors = ['lag_1', 'last_burst', 'roll_sum_7', 'momentum']

    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)
    
    for col in regressors:
        if col not in all_features.columns:
            logger.warning(f"   ⚠️ La columna del regresor '{col}' no se encontrromUtf83 en los datos. Se rellenarromUtf83 con 0.")
            if 'momentum' in col and 'momentum' not in all_features.columns:
                 roll_3 = all_features['n_tweets'].rolling(3).mean().shift(1)
                 roll_7 = all_features['n_tweets'].rolling(7).mean().shift(1)
                 all_features['momentum'] = (roll_3 - roll_7).fillna(0)
            else:
                 all_features[col] = 0.0

    last_data_date = all_features.index.max()
    last_complete_friday = get_last_complete_friday(last_data_date)
    validation_fridays = sorted([last_complete_friday - timedelta(weeks=i) for i in range(weeks_to_validate)])
    
    prophet_df = all_features.reset_index().rename(columns={'date': 'ds', 'n_tweets': 'y'})
    if prophet_df['ds'].dt.tz is not None:
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    
    if regressors:
        prophet_df[regressors] = prophet_df[regressors].fillna(0)
    
    predictions = []
    logger.info(f"   -> Validando {len(validation_fridays)} semanas desde {validation_fridays[0].date()} hasta {validation_fridays[-1].date()}")

    for friday_date in validation_fridays:
        week_start, week_end = friday_date, friday_date + timedelta(days=6)
        df_train = prophet_df[prophet_df['ds'] < week_start]
        test_dates = pd.date_range(week_start, week_end, freq='D')
        
        if len(df_train) < 90: continue

        m = Prophet(growth='linear', yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)
        if regressors:
            for reg in regressors: 
                m.add_regressor(reg)
        
        m.fit(df_train)
        
        future = pd.DataFrame({'ds': test_dates})
        if regressors:
            future = future.merge(prophet_df[['ds'] + regressors], on='ds', how='left').fillna(0)
        
        forecast = m.predict(future)
        result_week = forecast[['ds', 'yhat']].merge(prophet_df[['ds', 'y']], on='ds', how='left')
        predictions.append(result_week)

    if not predictions:
        raise ValueError("No se pudieron generar predicciones de backtest.")
        
    df_pred = pd.concat(predictions)
    df_weekly = df_pred.groupby(pd.Grouper(key='ds', freq='W-FRI')).agg(
        y_true=('y', 'sum'),
        y_pred=('yhat', 'sum')
    ).dropna()
    
    logger.success(f"✅ Predicciones de backtest generadas para {len(df_weekly)} semanas.")
    return df_weekly


def get_market_prices_simulation(mu_market: float) -> np.ndarray:
    """
    Simula precios de mercado.
    """
    probs = []
    for l, h in BINS:
        p = stats.poisson.cdf(h, mu_market) - stats.poisson.cdf(l-1, mu_market)
        probs.append(p)
    
    prices = np.array(probs) + 0.015 
    return prices / prices.sum()

def simulate_trading_run(df: pd.DataFrame, alpha_nb: float, kelly_fraction: float) -> tuple[pd.Series, float]:
    """
    Simula una trayectoria de trading completa.
    """
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    peak = capital
    max_drawdown = 0
    
    for _, row in df.iterrows():
        mu_mypred = row['y_pred']
        y_true = row['y_true']
        
        if alpha_nb < 1e-5: alpha_nb = 1e-5
        n_param = 1.0 / alpha_nb
        p_param = 1.0 / (1.0 + alpha_nb * mu_mypred)
        
        my_probs = []
        winning_bin = -1
        
        for i, (l, h) in enumerate(BINS):
            prob = stats.nbinom.cdf(h, n_param, p_param) - stats.nbinom.cdf(l-1, n_param, p_param)
            my_probs.append(prob)
            if l <= y_true <= h:
                winning_bin = i
                
        mu_market = row['y_pred'] * np.random.uniform(0.95, 1.05)
        market_prices = get_market_prices_simulation(mu_market)
        
        edges = np.array(my_probs) - market_prices
        best_idx = np.argmax(edges)
        edge = edges[best_idx]
        
        if edge > 0.05:
            price = market_prices[best_idx]
            my_prob = my_probs[best_idx]
            
            odds = 1.0 / price
            b = odds - 1
            if b <= 0: continue
            
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

def optimize_risk_params(df_backtest: pd.DataFrame) -> tuple[float, float]:
    """
    Optimiza los parromUtf83metros de riesgo (alpha y kelly_fraction).
    """
    logger.info(f"⚡ Optimizando sobre {len(df_backtest)} semanas...")
    
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
            
            results.append({
                'alpha': a, 
                'kelly': k, 
                'score': calmar_ratio,
                'pnl': pnl,
                'roi': roi
            })
            
    df_res = pd.DataFrame(results)
    
    # --- Visualizaciones ---
    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(df_res.pivot(index='alpha', columns='kelly', values='score'), annot=True, fmt=".2f", cmap="RdYlGn")
    ax.set_title("Calmar Ratio (Score)")
    plt.xlabel("Kelly Fraction")
    plt.ylabel("Alpha (NBinom)")
    plt.show()

    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(df_res.pivot(index='alpha', columns='kelly', values='pnl'), annot=True, fmt=".0f", cmap="viridis")
    ax.set_title("Profit & Loss ($)")
    plt.xlabel("Kelly Fraction")
    plt.ylabel("Alpha (NBinom)")
    plt.show()

    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(df_res.pivot(index='alpha', columns='kelly', values='roi'), annot=True, fmt=".2%", cmap="plasma")
    ax.set_title("Return on Investment (ROI)")
    plt.xlabel("Kelly Fraction")
    plt.ylabel("Alpha (NBinom)")
    plt.show()
    
    best = df_res.loc[df_res['score'].idxmax()]
    logger.info("\n🏆 CONFIGURACIromUtf83N GANADORA (Max Calmar Ratio):")
    logger.info(f"   Alpha (NB): {best['alpha']}")
    logger.info(f"   Kelly Mul : {best['kelly']}")
    logger.info(f"   Calmar    : {best['score']:.2f}")
    logger.info(f"   PnL ($)   : ${best['pnl']:.2f}")
    logger.info(f"   ROI       : {best['roi']:.2%}")
    
    return best['alpha'], best['kelly']

if __name__ == "__main__":
    df_backtest_real = generate_backtest_predictions(weeks_to_validate=WEEKS_TO_VALIDATE)
    
    optimal_alpha, optimal_kelly = optimize_risk_params(df_backtest_real)
    
    with open('risk_params.pkl', 'wb') as f:
        pickle.dump({'alpha': optimal_alpha, 'kelly': optimal_kelly}, f)
    logger.info(f"\n💾 ParromUtf83metros de riesgo romUtf83ptimos guardados en 'risk_params.pkl'")
    
    logger.info("\n--- SimulaciromUtf83n Final con ParromUtf83metros romUtf83ptimos ---")
    best_equity, best_mdd = simulate_trading_run(df_backtest_real, optimal_alpha, optimal_kelly)
    
    final_capital = best_equity.iloc[-1]
    final_pnl = final_capital - INITIAL_CAPITAL
    final_roi = final_pnl / INITIAL_CAPITAL

    logger.info(f"Capital Inicial: ${INITIAL_CAPITAL:.2f}")
    logger.info(f"Capital Final  : ${final_capital:.2f}")
    logger.info(f"PnL Final      : ${final_pnl:.2f}")
    logger.info(f"ROI Final      : {final_roi:.2%}")
    logger.info(f"Max Drawdown   : {best_mdd:.2%}")

    start_date = df_backtest_real.index.min() - pd.DateOffset(weeks=1)
    equity_index = pd.date_range(start=start_date, periods=len(best_equity), freq='W-FRI')
    equity_series_to_plot = pd.Series(best_equity.values, index=equity_index)
    
    plt.figure(figsize=(12, 6))
    equity_series_to_plot.plot(
        label=f'Alpha={optimal_alpha}, Kelly={optimal_kelly}\nFinal ROI: {final_roi:.2%}', 
        drawstyle="steps-post"
    )
    plt.title(f"Curva de Equidad romUtf83ptima (Max Drawdown: {best_mdd*100:.1f}%)")
    plt.ylabel("Capital ($)")
    plt.xlabel("Semanas del Backtest")
    plt.grid(True, which='major', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()
