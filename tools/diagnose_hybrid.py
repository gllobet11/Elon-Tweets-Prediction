import argparse
import os
import sys
import pandas as pd
from loguru import logger
import pickle

# --- Path Configuration ---
try:
    # FIX: Go up only one level to reach project root from tools/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logger.debug(f"Calculated project_root: {project_root}")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    logger.debug(f"Current sys.path: {sys.path}")

    from src.strategy.hybrid_predictor import get_hybrid_prediction

    # FIX: Import the class so pickle knows what object this is
    from src.models.prophet_inference import ProphetInferenceModel
    from src.processing.feature_eng import FeatureEngineer

    # Pointing to the specific dated file seen in your screenshot
    PROPHET_MODEL_PATH = os.path.join(project_root, "best_prophet_model_20251205.pkl")

except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)


def diagnose_prediction(
    debug_mode: bool,
    debug_dir: str,
    feature_file: str,
    days_forward: int,
):
    logger.info("--- Starting Hybrid Prediction Diagnostic ---")

    # 1. Load the Prophet Model
    logger.info(f"Loading Prophet model from: {PROPHET_MODEL_PATH}...")

    try:
        with open(PROPHET_MODEL_PATH, "rb") as f:
            loaded_content = pickle.load(f)

        # FIX: Check if we loaded a dictionary wrapper or the model directly
        if isinstance(loaded_content, dict) and "model" in loaded_content:
            logger.info("Detected dictionary wrapper. Extracting 'model' key...")
            prophet_model = loaded_content["model"]
        else:
            prophet_model = loaded_content

        logger.info(f"Model loaded successfully. Type: {type(prophet_model)}")

    except FileNotFoundError:
        logger.error(f"Model file NOT found at: {PROPHET_MODEL_PATH}")
        return
    except Exception as e:
        logger.error(f"Failed to load model pickle: {e}")
        return
    # 2. Load the initial features DataFrame
    logger.info(f"Loading initial features from: {feature_file}...")
    if not os.path.exists(feature_file):
        logger.error(
            f"Feature file not found at '{feature_file}'. Please ensure it exists."
        )
        return

    try:
        all_features_df = pd.read_csv(feature_file)

        # 1. Normalize column names
        all_features_df.columns = [c.lower() for c in all_features_df.columns]

        # 2. FIX: Handle common date column naming issues
        if "date_utc" in all_features_df.columns:
            all_features_df.rename(columns={"date_utc": "date"}, inplace=True)

        # NEW FIX: Handle 'unnamed: 0' (often the Date index read incorrectly)
        if "unnamed: 0" in all_features_df.columns:
            logger.info("Renaming 'unnamed: 0' to 'date'...")
            all_features_df.rename(columns={"unnamed: 0": "date"}, inplace=True)

        # 3. Validation
        if "date" not in all_features_df.columns:
            logger.error(f"CRITICAL: Column 'date' not found in CSV.")
            logger.error(f"Available columns are: {all_features_df.columns.tolist()}")
            return

        # 4. Set up the Index
        all_features_df["date"] = pd.to_datetime(all_features_df["date"])
        all_features_df.set_index("date", inplace=True)
        all_features_df.index = all_features_df.index.normalize()

        # 5. Create 'ds' for Prophet
        all_features_df["ds"] = all_features_df.index

        # 6. Safety: Inject missing regressors (like 'momentum') if they are absent
        # (Your logs show 'momentum' is present now, but this is good safety)
        if hasattr(prophet_model, "extra_regressors"):
            for reg in prophet_model.extra_regressors:
                if reg not in all_features_df.columns:
                    logger.warning(
                        f"MISSING REGRESSOR: '{reg}' not found. Filling with 0.0."
                    )
                    all_features_df[reg] = 0.0

    except Exception as e:
        logger.error(f"Failed to read feature CSV: {e}")
        return

    if all_features_df.empty:
        logger.error("Loaded features DataFrame is empty. Exiting.")
        return

    # 3. Run the hybrid prediction with debug mode
    logger.info(
        f"Running get_hybrid_prediction with debug_mode={debug_mode} for {days_forward} days..."
    )

    try:
        predictions_df, metrics = get_hybrid_prediction(
            prophet_model=prophet_model,
            all_features_df=all_features_df,
            days_forward=days_forward,
            debug_mode=debug_mode,
            debug_path=debug_dir,
        )

        logger.info("\n--- Prediction Summary ---")
        if not predictions_df.empty:
            logger.info(
                f"Weekly Total Prediction: {metrics.get('weekly_total_prediction'):.2f}"
            )
            logger.info("Daily Predictions:")
            print(predictions_df)
            if debug_mode:
                logger.info(f"Detailed debug snapshots saved to: {debug_dir}")
        else:
            logger.warning("No predictions generated.")

    except Exception as e:
        logger.exception(f"Error during prediction execution: {e}")

    logger.info("--- Hybrid Prediction Diagnostic Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose the Hybrid Prediction process with snapshot logging."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode and save intermediate snapshots to disk.",
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="debug_dumps",
        help="Directory to save debug snapshots. Relative to project root.",
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        default="data/processed/verified_features.csv",
        help="Path to the CSV file containing the initial historical features.",
    )
    parser.add_argument(
        "--days_forward",
        type=int,
        default=7,
        help="Number of days to predict forward (e.g., a week). Defaults to 7.",
    )
    args = parser.parse_args()

    # Adjust debug_dir to be relative to the project root
    abs_debug_dir = os.path.join(project_root, args.debug_dir)
    os.makedirs(abs_debug_dir, exist_ok=True)  # Ensure dir exists

    diagnose_prediction(
        debug_mode=args.debug,
        debug_dir=abs_debug_dir,
        feature_file=os.path.join(project_root, args.feature_file),
        days_forward=args.days_forward,
    )
