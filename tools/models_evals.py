"""
models_evals.py

Este script se encarga de evaluar diferentes configuraciones de modelos de Prophet
para predecir la cantidad semanal de tweets. Utiliza un enfoque de validación
walk-forward para simular el rendimiento del modelo a lo largo del tiempo.

El objetivo principal es identificar la configuración de modelo (conjunto de regresores)
y el parámetro de dispersión (`alpha` para la distribución Negative Binomial)
que minimizan el Log Loss promedio.

El script genera:
- Un resumen tabular de las métricas de rendimiento para cada configuración evaluada.
- Un gráfico comparativo de los Log Loss para las diferentes combinaciones.
- Guarda el modelo con la mejor configuración en un archivo `.pkl` para su uso posterior
  en la optimización financiera y el dashboard de producción.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate
import logging
from prophet import Prophet
import pickle
from datetime import datetime, timedelta
from loguru import logger

# --- SUPRESIÓN DE LOGS ---
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

# --- Path Configuration & Imports ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from src.strategy.prob_math import DistributionConverter
    from config.bins_definition import MARKET_BINS
except Exception as e:
    logger.error(f"Error import: {e}")
    sys.exit(1)

# --- HYPERPARAMETERS ---
ALPHA_CANDIDATES = [0.01, 0.05, 0.10, 0.15, 0.20]

def get_last_complete_friday(last_data_date: pd.Timestamp) -> datetime:
    """
    Encuentra el último viernes completo que puede iniciar una ventana de pronóstico de 7 días,
    asegurando que se disponga de datos de verdad fundamental.
    """
    if isinstance(last_data_date, pd.Timestamp):
        last_data_date = last_data_date.to_pydatetime()
    if last_data_date.tzinfo is not None:
        last_data_date = last_data_date.replace(tzinfo=None)

    last_possible_forecast_start = last_data_date - timedelta(days=6)
    days_since_friday = (last_possible_forecast_start.weekday() - 4) % 7
    last_possible_friday = last_possible_forecast_start - timedelta(days=days_since_friday)
    
    return last_possible_friday

def run_weekly_walk_forward(all_features_df: pd.DataFrame, regressors: list, validation_fridays: list) -> pd.DataFrame:
    """
    Simula predicciones semanales utilizando un enfoque de validación walk-forward.
    """
    prophet_df = all_features_df.reset_index().rename(columns={'date': 'ds', 'n_tweets': 'y'})
    if prophet_df['ds'].dt.tz is not None:
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    
    if regressors:
        for col in [r for r in regressors if r not in prophet_df.columns]:
            prophet_df[col] = 0.0
        prophet_df[regressors] = prophet_df[regressors].fillna(0)
    
    predictions = []
    logger.info(f"   -> Validando semanas: {[d.strftime('%Y-%m-%d') for d in validation_fridays]}")

    for friday_date in validation_fridays:
        week_start, week_end = friday_date, friday_date + timedelta(days=6)
        df_train = prophet_df[prophet_df['ds'] < week_start]
        test_dates = pd.date_range(week_start, week_end, freq='D')
        
        if len(df_train) < 90:
            logger.warning(f"   ⚠️ Insuficientes datos para {friday_date.date()}"); continue

        try:
            m = Prophet(growth='linear', yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)
            for reg in regressors: m.add_regressor(reg)
            m.fit(df_train)
            
            future = pd.DataFrame({'ds': test_dates})
            if regressors:
                future = future.merge(prophet_df[['ds'] + regressors], on='ds', how='left').fillna(0)
            
            forecast = m.predict(future)
            result_week = forecast[['ds', 'yhat']].merge(prophet_df[['ds', 'y']], on='ds', how='left')
            
            for _, row in result_week.iterrows():
                predictions.append({'ds': row['ds'], 'y_pred': max(0, row['yhat']), 'y_true': row['y'], 'week_start': friday_date})
        except Exception as e:
            logger.error(f"   ❌ Error en semana {friday_date.date()}: {e}")

    return pd.DataFrame(predictions).set_index('ds') if predictions else pd.DataFrame()

def train_best_model(all_features_df: pd.DataFrame, best_config: dict) -> dict:
    """
    Entrena el mejor modelo Prophet final con la configuración óptima.
    """
    logger.info(f"\n🏆 Entrenando modelo final: {best_config['name']} con distribución {best_config['distribution']} y alpha {best_config['alpha']:.4f}")
    
    prophet_df = all_features_df.reset_index().rename(columns={'date': 'ds', 'n_tweets': 'y'})
    if prophet_df['ds'].dt.tz is not None: prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    
    regressors = best_config['regressors']
    if regressors:
        for col in [r for r in regressors if r not in prophet_df.columns]: prophet_df[col] = 0.0
        prophet_df[regressors] = prophet_df[regressors].fillna(0)
    
    m = Prophet(growth='linear', yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)
    for reg in regressors: m.add_regressor(reg)
    m.fit(prophet_df)
    
    return {'model': m, 'model_name': best_config['name'], 'regressors': regressors, 'trained_on': prophet_df['ds'].max(), 'training_samples': len(prophet_df), 'metrics': best_config['metrics']}

def get_bin_for_value(value: float, bins_config: list) -> str | None:
    """
    Determina en qué bin cae un valor.
    """
    for label, lower, upper in bins_config:
        if lower <= value < upper: return label
    return None

def compare_prophet_feature_sets_weekly(weeks_to_validate: int = 4):
    """
    Realiza una comparación semanal walk-forward de diferentes configuraciones de modelos.
    """
    logger.info(f"\n{'='*80}\n   VALIDACIÓN DE DISTRIBUCIÓN Y MODELOS (ÚLTIMAS {weeks_to_validate} SEMANAS)   \n{'='*80}\n")
    
    logger.info("📡 Cargando y procesando datos...")
    df_tweets = load_unified_data()
    all_features = FeatureEngineer().process_data(df_tweets)
    
    if 'momentum' not in all_features.columns:
        logger.warning("   ⚠️ Calculando 'momentum'..."); roll_3 = all_features['n_tweets'].rolling(3).mean().shift(1); roll_7 = all_features['n_tweets'].rolling(7).mean().shift(1); all_features['momentum'] = (roll_3 - roll_7).fillna(0)

    last_data_date = all_features.index.max()
    logger.info(f"📅 Último dato disponible: {last_data_date.date()}")
    last_complete_friday = get_last_complete_friday(last_data_date)
    logger.info(f"📅 Último viernes para iniciar validación: {last_complete_friday.date()}")
    
    validation_fridays = sorted([last_complete_friday - timedelta(weeks=i) for i in range(weeks_to_validate)])
    
    model_candidates = {'Baseline': [], 'Dynamic_AR': ['lag_1', 'last_burst', 'roll_sum_7', 'momentum'], 'External': ['reply_ratio', 'hour_std_feature']}
    dist_candidates = ['nbinom', 'poisson']
    bins_config = [(k, v['lower'], v['upper']) for k, v in MARKET_BINS.items()]
    metrics = []
    weekly_diagnostics_all = []

    for model_name, regressors in model_candidates.items():
        logger.info(f"\n🔍 Evaluando Modelo de Regresión: {model_name}")
        results_df = run_weekly_walk_forward(all_features, regressors, validation_fridays)
        if results_df.empty: logger.warning("   ❌ Sin resultados de predicción."); continue
            
        weekly_agg = results_df.dropna().groupby('week_start').agg(y_true=('y_true', 'sum'), y_pred=('y_pred', 'sum')).reset_index()

        for dist_type in dist_candidates:
            alphas_to_test = ALPHA_CANDIDATES if dist_type == 'nbinom' else [None]
            
            for alpha in alphas_to_test:
                log_losses = []
                for _, week in weekly_agg.iterrows():
                    mu, y_true = week['y_pred'], week['y_true']
                    try:
                        probs = DistributionConverter.get_bin_probabilities(
                            mu_remainder=mu, current_actuals=0,
                            model_type=dist_type, alpha=alpha, bins_config=bins_config
                        )
                    except ValueError as e: 
                        logger.error(f"     - Error al generar probs para {dist_type}: {e}"); continue

                    correct_bin = get_bin_for_value(y_true, bins_config)
                    prob_correct = (probs.get(correct_bin, 0) + 1e-9) if correct_bin else 1e-9
                    loss = -np.log(prob_correct)
                    log_losses.append(loss)
                    
                    weekly_diagnostics_all.append({
                        'week_start': week['week_start'], 'model_name': model_name, 
                        'distribution': dist_type, 'alpha': alpha if alpha is not None else 'N/A', 
                        'y_true': y_true, 'y_pred': mu, 'correct_bin': correct_bin, 
                        'prob_of_correct_bin': prob_correct, 'log_loss_for_week': loss
                    })

                if log_losses:
                    metrics.append({
                        "Model": model_name, "Distribution": dist_type,
                        "Alpha": alpha if alpha is not None else 'N/A',
                        "Avg Log Loss": np.mean(log_losses), "Weeks Validated": len(log_losses)
                    })

    if not metrics: logger.error("❌ No se generaron métricas."); return
        
    logger.info(f"\n{'='*80}\n   RESULTADOS DE VALIDACIÓN (Log Loss)   \n{'='*80}")
    df_metrics = pd.DataFrame(metrics).sort_values('Avg Log Loss')
    logger.info("\n" + tabulate(df_metrics, headers='keys', tablefmt='simple_grid', floatfmt=".4f", showindex=False))
    
    best = df_metrics.iloc[0]
    
    best_text = f"🏆 MEJOR COMBINACIÓN: Modelo '{best['Model']}' con Distribución '{best['Distribution']}'"
    if best['Distribution'] == 'nbinom': best_text += f" y Alpha = {best['Alpha']:.4f}"
    best_text += f"\n   • Log Loss Promedio: {best['Avg Log Loss']:.4f}"
    logger.success(f"\n{best_text}\n")
    
    logger.info(f"\n{'='*80}\n   ANÁLISIS DETALLADO: Mejor Modelo y Distribución Ganadora   \n{'='*80}")
    try:
        diag_df = pd.DataFrame(weekly_diagnostics_all)
        best_model_name, winning_dist = best['Model'], best['Distribution']
        winning_alpha = best['Alpha'] if winning_dist == 'nbinom' else 'N/A'

        winner_condition = ((diag_df['model_name'] == best_model_name) & (diag_df['distribution'] == winning_dist))
        if winning_dist == 'nbinom' and winning_alpha != 'N/A': winner_condition &= (diag_df['alpha'] == winning_alpha)
            
        df_winner = diag_df[winner_condition]

        logger.info(f"Desempeño semanal para el Modelo ganador '{best_model_name}' y Distribución '{winning_dist}' (Alpha: {winning_alpha}):\n")
        
        table_data = df_winner[['week_start', 'y_true', 'y_pred', 'correct_bin', 'prob_of_correct_bin', 'log_loss_for_week']].copy()
        table_data['week_start'] = table_data['week_start'].dt.date
        logger.info("\n" + tabulate(table_data, headers='keys', tablefmt='simple_grid', floatfmt=".4f", showindex=False))
        
        if winning_dist == 'nbinom':
            logger.info("\nComparación con un Alpha conservador (si aplica):")
            conservative_alpha = 0.15 if 0.15 in ALPHA_CANDIDATES else None
            if conservative_alpha and conservative_alpha != winning_alpha:
                 df_conserv = diag_df[(diag_df['model_name'] == best_model_name) & (diag_df['distribution'] == 'nbinom') & (diag_df['alpha'] == conservative_alpha)]
                 df_compare_alpha = pd.merge(
                    df_winner[['week_start', 'log_loss_for_week', 'prob_of_correct_bin']], 
                    df_conserv[['week_start', 'log_loss_for_week', 'prob_of_correct_bin']], 
                    on='week_start', suffixes=(f'_{winning_alpha}', f'_{conservative_alpha}')
                )
                 df_compare_alpha = df_compare_alpha.rename(columns={
                     f'prob_of_correct_bin_{winning_alpha}': f'Prob (α={winning_alpha})', 
                     f'log_loss_for_week_{winning_alpha}': f'LogLoss (α={winning_alpha})',
                     f'prob_of_correct_bin_{conservative_alpha}': f'Prob (α={conservative_alpha})', 
                     f'log_loss_for_week_{conservative_alpha}': f'LogLoss (α={conservative_alpha})'
                 })
                 df_compare_alpha['week_start'] = df_compare_alpha['week_start'].dt.date
                 logger.info("\n" + tabulate(df_compare_alpha, headers='keys', tablefmt='simple_grid', floatfmt=".4f", showindex=False))
            else:
                logger.info("No hay un alpha conservador diferente para comparar o el ganador ya es conservador.")

    except Exception as e:
        logger.error(f"\n❌ No se pudo generar el análisis de diagnóstico: {e}")

    best_config = {
        'name': best['Model'], 'regressors': model_candidates[best['Model']], 
        'distribution': best['Distribution'], 'alpha': best['Alpha'] if best['Distribution'] == 'nbinom' else None,
        'metrics': best.to_dict()
    }
    
    final_model_package = train_best_model(all_features, best_config)
    
    final_model_package['best_distribution'] = best_config['distribution']
    if best_config['distribution'] == 'nbinom':
        final_model_package['best_alpha'] = best_config['alpha']

    model_filename = f'best_prophet_model_{datetime.now().strftime("%Y%m%d")}.pkl'
    with open(model_filename, 'wb') as f: pickle.dump(final_model_package, f)
    
    logger.success(f"\n💾 Modelo guardado: {model_filename}")
    logger.info(f"   • Distribución: {final_model_package['best_distribution']}")
    if 'best_alpha' in final_model_package:
        logger.info(f"   • Mejor Alpha: {final_model_package['best_alpha']:.4f}")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_plot = df_metrics.copy()
    df_plot['Combined'] = df_plot['Model'] + ' (' + df_plot['Distribution'] + \
                          np.where(df_plot['Distribution'] == 'nbinom', ' - α=' + df_plot['Alpha'].astype(str), '') + ')'
    
    df_plot = df_plot.sort_values(['Distribution', 'Model', 'Alpha'])
    
    colors = plt.cm.tab20.colors
    model_color_map = {model: colors[i % len(colors)] for i, model in enumerate(df_plot['Model'].unique())}

    x_labels, x_positions, bar_width, current_position = [], [], 0.35, 0
    
    for i, row in df_plot.iterrows():
        x_labels.append(row['Combined'])
        x_positions.append(current_position)
        
        if row['Distribution'] == 'nbinom':
            ax.bar(current_position, row['Avg Log Loss'], width=bar_width, 
                   color=model_color_map[row['Model']], 
                   label=f"{row['Model']} (NBinom, α={row['Alpha']:.2f})" if row['Alpha'] != 'N/A' else f"{row['Model']} (NBinom)",
                   edgecolor='black')
        elif row['Distribution'] == 'poisson':
            ax.plot(current_position, row['Avg Log Loss'], 'o', 
                    color=model_color_map[row['Model']], markersize=8, 
                    label=f"{row['Model']} (Poisson)", markeredgecolor='black')
            
        current_position += 1

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_title('Comparación de Modelos y Distribuciones (Avg Log Loss)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Avg Log Loss (Menor es Mejor)', fontsize=12)
    ax.set_xlabel('Configuración de Modelo y Distribución', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(l)] for l in unique_labels]
    ax.legend(unique_handles, unique_labels, title="Leyenda", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    output_file = 'dist_model_validation.png'
    plt.savefig(output_file, dpi=300)
    logger.success(f"📊 Gráfico de validación guardado: {output_file}")

if __name__ == "__main__":
    compare_prophet_feature_sets_weekly(weeks_to_validate=4)
