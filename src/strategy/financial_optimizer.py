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

# Set a random seed for reproducibility
np.random.seed(42)

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config.bins_definition import MARKET_BINS
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
except (ImportError, ModuleNotFoundError) as e:
    # Fallback para ejecución local directa
    print(f"Nota importación: {e}")

# --- CONFIGURACIÓN ---
INITIAL_CAPITAL = 1000.0
# Usar los bins reales del proyecto
BINS = [(v['lower'], v['upper']) for k, v in MARKET_BINS.items()]
WEEKS_TO_VALIDATE = 12 

def get_last_complete_friday(last_data_date):
    """
    Finds the last Friday that can start a complete 7-day forecast window.
    """
    if isinstance(last_data_date, pd.Timestamp):
        last_data_date = last_data_date.to_pydatetime()
    if last_data_date.tzinfo is not None:
        last_data_date = last_data_date.replace(tzinfo=None)

    last_possible_forecast_start = last_data_date - timedelta(days=6)
    days_since_friday = (last_possible_forecast_start.weekday() - 4) % 7
    return last_possible_forecast_start - timedelta(days=days_since_friday)

def generate_backtest_predictions(weeks_to_validate: int):
    """
    Genera predicciones de backtest usando el mejor modelo Prophet (Dynamic_AR).
    """
    print("⚙️  Generando predicciones de backtest...")
    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)
    
    # Feature Engineering 'on the fly' si falta
    if 'momentum' not in all_features.columns:
        roll_3 = all_features['n_tweets'].rolling(3).mean().shift(1)
        roll_7 = all_features['n_tweets'].rolling(7).mean().shift(1)
        all_features['momentum'] = (roll_3 - roll_7).fillna(0)

    last_data_date = all_features.index.max()
    last_complete_friday = get_last_complete_friday(last_data_date)
    validation_fridays = sorted([last_complete_friday - timedelta(weeks=i) for i in range(weeks_to_validate)])
    
    regressors = ['lag_1', 'last_burst', 'roll_sum_7', 'momentum']
    
    prophet_df = all_features.reset_index().rename(columns={'date': 'ds', 'n_tweets': 'y'})
    if prophet_df['ds'].dt.tz is not None:
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    
    for col in [r for r in regressors if r not in prophet_df.columns]:
        prophet_df[col] = 0.0
    prophet_df[regressors] = prophet_df[regressors].fillna(0)
    
    predictions = []
    print(f"   -> Validando {len(validation_fridays)} semanas desde {validation_fridays[0].date()}")

    for friday_date in validation_fridays:
        week_start = friday_date
        # week_end = friday_date + timedelta(days=6) # No se usa explícitamente en el filtro
        
        df_train = prophet_df[prophet_df['ds'] < week_start]
        test_dates = pd.date_range(week_start, periods=7, freq='D')
        
        if len(df_train) < 90: continue

        m = Prophet(growth='linear', yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)
        for reg in regressors: m.add_regressor(reg)
        m.fit(df_train)
        
        future = pd.DataFrame({'ds': test_dates})
        future = future.merge(prophet_df[['ds'] + regressors], on='ds', how='left').fillna(0)
        
        forecast = m.predict(future)
        result_week = forecast[['ds', 'yhat']].merge(prophet_df[['ds', 'y']], on='ds', how='left')
        predictions.append(result_week)

    if not predictions:
        raise ValueError("No se pudieron generar predicciones de backtest.")
        
    df_pred = pd.concat(predictions)
    # Agrupar por semana (empezando el viernes de cada grupo)
    df_weekly = df_pred.groupby(pd.Grouper(key='ds', freq='W-THU')).agg(
        y_true=('y', 'sum'),
        y_pred=('yhat', 'sum')
    ).dropna()
    
    # Filtrar semanas incompletas (menos de 7 dias suma real muy baja sospechosa)
    df_weekly = df_weekly[df_weekly['y_true'] > 100] 
    
    print(f"✅ Predicciones generadas para {len(df_weekly)} semanas validas.")
    return df_weekly

def get_market_prices_simulation(mu_market):
    """
    Simula precios de mercado basándose en una Poisson + Spread.
    """
    probs = []
    for l, h in BINS:
        p = stats.poisson.cdf(h, mu_market) - stats.poisson.cdf(l-1, mu_market)
        probs.append(p)
    
    prices = np.array(probs) + 0.015 
    return prices / prices.sum()

def simulate_trading_run(df, alpha_nb, kelly_fraction):
    """
    Simula la trayectoria de capital.
    """
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    peak = capital
    max_drawdown = 0
    
    for _, row in df.iterrows():
        mu_mypred = row['y_pred']
        y_true = row['y_true']
        
        if alpha_nb < 1e-5: alpha_nb = 1e-5
        # Parametrización scipy nbinom: n = 1/alpha, p = 1/(1+alpha*mu)
        n_param = 1.0 / alpha_nb
        p_param = 1.0 / (1.0 + alpha_nb * mu_mypred)
        
        my_probs = []
        winning_bin = -1
        
        for i, (l, h) in enumerate(BINS):
            prob = stats.nbinom.cdf(h, n_param, p_param) - stats.nbinom.cdf(l-1, n_param, p_param)
            my_probs.append(prob)
            if l <= y_true <= h:
                winning_bin = i
                
        # Simulación de Mercado (con ruido para realismo)
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
            bet_size = min(bet_size, 0.20) # Max 20% portfolio
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

def optimize_risk_params(df_backtest):
    print(f"⚡ Optimizando sobre {len(df_backtest)} semanas...")
    
    alphas = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
    kellys = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = []
    
    for a in alphas:
        for k in kellys:
            equity_curve, max_drawdown = simulate_trading_run(df_backtest, a, k)
            total_return = (equity_curve.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
            
            if max_drawdown == 0:
                score = total_return * 10 
            elif max_drawdown >= 1.0:
                score = -1.0 
            else:
                score = total_return / max_drawdown
            
            results.append({'alpha': a, 'kelly': k, 'score': score})
            
    df_res = pd.DataFrame(results)
    best = df_res.loc[df_res['score'].idxmax()]
    
    print(f"\n🏆 CONFIGURACIÓN GANADORA:")
    print(f"   Alpha (NB): {best['alpha']}")
    print(f"   Kelly Mul : {best['kelly']}")
    print(f"   Calmar    : {best['score']:.2f}")
    
    return best['alpha'], best['kelly'], df_res

if __name__ == "__main__":
    # Flujo completo
    try:
        df_backtest_real = generate_backtest_predictions(weeks_to_validate=WEEKS_TO_VALIDATE)
        optimal_alpha, optimal_kelly, df_results = optimize_risk_params(df_backtest_real)
        
        # Guardar parámetros
        params = {'alpha': optimal_alpha, 'kelly': optimal_kelly, 'timestamp': datetime.now()}
        with open('risk_params.pkl', 'wb') as f:
            pickle.dump(params, f)
        print("💾 Parámetros guardados en risk_params.pkl")
        
        # Plot Heatmap
        pivot = df_results.pivot(index='alpha', columns='kelly', values='score')
        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn")
        plt.title("Optimization Heatmap (Calmar Ratio)")
        plt.savefig("optimization_heatmap.png")
        print("📊 Heatmap guardado como optimization_heatmap.png")
        
    except Exception as e:
        print(f"Error en ejecución: {e}")