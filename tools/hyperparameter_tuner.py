import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta, datetime
from loguru import logger
from prophet import Prophet
import pickle

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from src.strategy.prob_math import DistributionConverter
    from config.bins_definition import MARKET_BINS
    from config.settings import WEEKS_TO_VALIDATE
    from tools.models_evals import get_last_complete_friday, run_weekly_walk_forward, ALPHA_CANDIDATES, dist_candidates, get_bin_for_value # Import necessary components from models_evals
except Exception as e:
    logger.error(f"Error import: {e}")
    sys.exit(1)

# --- GLOBAL CONFIGURATION ---
TARGET_FEATURES = ['lag_1', 'roll_sum_7', 'momentum', 'last_burst', 'is_high_regime', 'is_regime_change']
BINS_CONFIG_LIST = [(k, v['lower'], v['upper']) for k, v in MARKET_BINS.items()]

def tune_prophet_hyperparameters(
    df_tweets: pd.DataFrame, 
    features: list, 
    weeks_to_validate: int,
    alpha_candidates: list,
    dist_candidates: list,
    bins_config: list,
) -> dict:
    """
    Performs hyperparameter tuning for Prophet model using walk-forward validation.
    Searches for the best combination of Prophet hyperparameters and NBinom alpha.
    """
    logger.info("🚀 Starting Prophet Hyperparameter Tuning (Walk-Forward Validation)")

    all_features_df = FeatureEngineer().process_data(df_tweets)
    
    # Define hyperparameter search space for Prophet
    prophet_param_grid = {
        'changepoint_prior_scale': [0.005, 0.01, 0.05, 0.1],
        'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0],
    }

    best_loss = float('inf')
    best_params = {}
    
    # Generate all combinations of Prophet hyperparameters
    from itertools import product
    prophet_param_combinations = [dict(zip(prophet_param_grid.keys(), v)) for v in product(*prophet_param_grid.values())]

    total_combinations = len(prophet_param_combinations) * len(alpha_candidates) * len(dist_candidates)
    logger.info(f"Total combinations to evaluate: {total_combinations}")

    current_combination_num = 0

    for prophet_params in prophet_param_combinations:
        for dist_type in dist_candidates:
            alphas_to_test = alpha_candidates if dist_type == 'nbinom' else [None]

            for alpha in alphas_to_test:
                current_combination_num += 1
                logger.info(f"Evaluating combination {current_combination_num}/{total_combinations}: Prophet: {prophet_params}, Dist: {dist_type}, Alpha: {alpha}")

                # --- Walk-forward evaluation logic ---
                # This part is similar to evaluate_model_cv but for a single set of Prophet params
                
                last_data_date = all_features_df.index.max()
                last_complete_friday = get_last_complete_friday(last_data_date)
                validation_fridays = sorted([last_complete_friday - timedelta(weeks=i) for i in range(weeks_to_validate)])
                
                predictions = []
                # Prepare data for Prophet format once for this combination
                df_prophet_format = all_features_df.reset_index().rename(columns={'date': 'ds', 'n_tweets': 'y'})

                for friday_date in validation_fridays:
                    week_start, week_end = friday_date, friday_date + timedelta(days=6)
                    df_train = df_prophet_format[df_prophet_format['ds'] < week_start]
                    test_dates = pd.date_range(week_start, week_end, freq='D')
                    
                    if len(df_train) < 90:
                        logger.warning(f"   ⚠️ Insuficientes datos para {friday_date.date()}"); continue

                    try:
                        m = Prophet(
                            growth='linear',
                            yearly_seasonality=False,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            **prophet_params # Apply current Prophet hyperparameters
                        )
                        for reg in features: m.add_regressor(reg)
                        m.fit(df_train)
                        
                        future = pd.DataFrame({'ds': test_dates})
                        if features:
                            future = future.merge(all_features_df.reset_index().rename(columns={'date': 'ds'})[['ds'] + features], on='ds', how='left').fillna(0)
                        
                        forecast = m.predict(future)
                        result_week = forecast[['ds', 'yhat']].merge(all_features_df.reset_index().rename(columns={'date': 'ds', 'n_tweets': 'y'}), on='ds', how='left')
                        
                        for _, row in result_week.iterrows():
                            predictions.append({'ds': row['ds'], 'y_pred': max(0, row['yhat']), 'y_true': row['y'], 'week_start': friday_date})
                    except Exception as e:
                        logger.error(f"   ❌ Error en semana {friday_date.date()} con Prophet params {prophet_params}, Dist {dist_type}, Alpha {alpha}: {e}")

                if not predictions:
                    current_loss = float('inf')
                else:
                    results_df = pd.DataFrame(predictions).set_index('ds')
                    weekly_agg = results_df.dropna().groupby('week_start').agg(y_true=('y_true', 'sum'), y_pred=('y_pred', 'sum')).reset_index()

                    log_losses = []
                    for _, week in weekly_agg.iterrows():
                        mu, y_true = week['y_pred'], week['y_true']
                        try:
                            probs = DistributionConverter.get_bin_probabilities(
                                mu_remainder=mu, current_actuals=0,
                                model_type=dist_type, alpha=alpha, bins_config=bins_config
                            )
                        except ValueError:
                            continue # Skip if probability calculation fails

                        correct_bin = get_bin_for_value(y_true, bins_config)
                        prob_correct = (probs.get(correct_bin, 0) + 1e-9) if correct_bin else 1e-9
                        log_losses.append(-np.log(prob_correct))
                    
                    current_loss = np.mean(log_losses) if log_losses else float('inf')

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = {
                        'prophet_params': prophet_params,
                        'distribution': dist_type,
                        'alpha': alpha,
                        'log_loss': best_loss,
                        'features': features # The features are fixed for this tuning run
                    }
                    logger.success(f"🏆 New best found! Log Loss: {best_loss:.4f} with params: {best_params}")

    logger.info("="*30)
    logger.info(f"✨ Tuning complete. Best Log Loss: {best_loss:.4f}")
    logger.info(f"Parameters: {best_params}")
    logger.info("="*30)

    return best_params

if __name__ == "__main__":
    df_tweets_data = load_unified_data()
    best_tuned_params = tune_prophet_hyperparameters(
        df_tweets=df_tweets_data,
        features=TARGET_FEATURES,
        weeks_to_validate=WEEKS_TO_VALIDATE,
        alpha_candidates=ALPHA_CANDIDATES,
        dist_candidates=dist_candidates,
        bins_config=BINS_CONFIG_LIST
    )
    
    # Optionally save the best parameters
    # with open('best_prophet_hyperparams.pkl', 'wb') as f:
    #     pickle.dump(best_tuned_params, f)
    # logger.success(f"Best hyperparameters saved to 'best_prophet_hyperparams.pkl'")
