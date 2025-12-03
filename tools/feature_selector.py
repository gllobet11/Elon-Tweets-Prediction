import os
import sys

from loguru import logger

# Adjust imports to match your project structure
# --- Path Configuration & Imports ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from config.bins_definition import MARKET_BINS  # Added import
    from config.settings import WEEKS_TO_VALIDATE
    from src.ingestion.unified_feed import load_unified_data
    from src.processing.feature_eng import FeatureEngineer
    from tools.models_evals import (
        ALPHA_CANDIDATES,
        dist_candidates,
        evaluate_model_cv,
    )
except Exception as e:
    logger.error(f"Error import: {e}")
    sys.exit(1)


# CONFIG
# WEEKS_TO_TEST = 12  # Trust the long term (Already imported as WEEKS_TO_VALIDATE)
BASELINE_FEATURES = ["lag_1", "roll_sum_7", "momentum", "last_burst"]
CANDIDATE_FEATURES = [
    "reply_ratio",
    "hour_std_feature",  # External
    "is_high_regime",
    "regime_intensity",
    "is_regime_change",  # Regime
    "is_weekend",
    "dow",  # Calendar
    "cv_7",  # Volatility
]


def run_forward_selection():
    logger.info("🚀 Starting Forward Feature Selection (Greedy Approach)")

    # 1. Load Data
    df_tweets = load_unified_data()
    all_features_df = FeatureEngineer().process_data(df_tweets)

    # 2. Baseline Evaluation
    logger.info(f"Evaluating Baseline: {BASELINE_FEATURES}")
    baseline_loss, _ = evaluate_model_cv(
        all_features_df=all_features_df,
        regressors=BASELINE_FEATURES,
        weeks_to_validate=WEEKS_TO_VALIDATE,
        alpha_candidates=ALPHA_CANDIDATES,
        dist_candidates=dist_candidates,
        bins_config=[(k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()],
    )
    logger.success(f"📉 Baseline Log Loss: {baseline_loss:.4f}")

    best_loss = baseline_loss
    best_features = BASELINE_FEATURES.copy()

    # 3. Iterative Testing
    for feature in CANDIDATE_FEATURES:
        current_features = best_features + [feature]
        logger.info(
            f"Testing candidate: + {feature} (Current features: {current_features})",
        )

        try:
            current_loss, _ = evaluate_model_cv(
                all_features_df=all_features_df,
                regressors=current_features,
                weeks_to_validate=WEEKS_TO_VALIDATE,
                alpha_candidates=ALPHA_CANDIDATES,
                dist_candidates=dist_candidates,
                bins_config=[
                    (k, v["lower"], v["upper"]) for k, v in MARKET_BINS.items()
                ],
            )

            improvement = best_loss - current_loss

            if improvement > 0.005:  # Threshold: Must improve by at least 0.005
                logger.success(
                    f"✅ KEEP {feature}: Improved Loss by {improvement:.4f} (New: {current_loss:.4f})",
                )
                best_loss = current_loss
                best_features.append(feature)
            else:
                logger.warning(
                    f"❌ DROP {feature}: No significant improvement (Diff: {improvement:.4f})",
                )

        except Exception as e:
            logger.error(f"Error testing {feature}: {e}")

    # 4. Final Verdict
    logger.info("=" * 30)
    logger.info(f"🏆 FINAL WINNING MODEL FEATURES (Loss: {best_loss:.4f})")
    logger.info(best_features)
    logger.info("=" * 30)


if __name__ == "__main__":
    run_forward_selection()
