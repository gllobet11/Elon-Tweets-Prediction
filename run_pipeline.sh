#!/bin/bash

# --- Script to run the Elon Tweets Prediction Pipeline ---

# Default values for optional flags
OPTIMIZE_FLAG=false
VISUALS_FLAG=false

# --- Help Message Function ---
show_help() {
    echo "Usage: bash run_pipeline.sh [OPTIONS]"
    echo ""
    echo "Runs the complete Elon Musk Tweet Prediction Pipeline, from data ingestion to launching the dashboard."
    echo ""
    echo "Options:"
    echo "  --optimize    Run the extensive hyperparameter tuning process. This is computationally intensive."
    echo "  --visuals     Generate and save the multi-model backtest visualization plot."
    echo "  --help, -h    Show this help message and exit."
}

# --- Parse command-line arguments ---
for arg in "$@"; do
    case $arg in
        --optimize)
        OPTIMIZE_FLAG=true
        shift # Remove --optimize from processing
        ;;
        --visuals)
        VISUALS_FLAG=true
        shift # Remove --visuals from processing
        ;;
        --help|-h)
        show_help
        exit 0
        ;;
        *)
        # Unknown option
        ;;
    esac
done

echo "Starting Elon Tweets Prediction Pipeline..."

# 1. Ingest and Unify Latest Tweet Data
echo ">>> Step 1: Ingesting and unifying latest tweet data..."
./.venv/Scripts/python.exe run_ingest.py
if [ $? -ne 0 ]; then echo "Error in ingestion step. Exiting."; exit 1; fi

# 2 & 3. Model Training (Conditional)
if [ "$OPTIMIZE_FLAG" = true ]; then
    echo ">>> Step 2: Running hyperparameter tuning (as --optimize was provided)..."
    ./.venv/Scripts/python.exe tools/model_analysis.py --task tune_hyperparameters
    if [ $? -ne 0 ]; then echo "Error in hyperparameter tuning step. Exiting."; exit 1; fi

    echo ">>> Step 3: Training model with tuned hyperparameters..."
    ./.venv/Scripts/python.exe tools/model_analysis.py --task train_with_tuned_hps
    if [ $? -ne 0 ]; then echo "Error in training with tuned HPs. Exiting."; exit 1; fi
else
    echo ">>> Step 2 & 3: Evaluating models and training the best production model..."
    ./.venv/Scripts/python.exe tools/model_analysis.py --task train_and_evaluate
    if [ $? -ne 0 ]; then echo "Error in model evaluation step. Exiting."; exit 1; fi
fi


# 4. Generate Historical Performance Data
echo ">>> Step 4: Generating historical performance data..."
./.venv/Scripts/python.exe tools/generate_historical_performance.py
if [ $? -ne 0 ]; then echo "Error in historical performance generation. Exiting."; exit 1; fi

# 5. Optimize Financial Parameters
echo ">>> Step 5: Optimizing financial parameters..."
./.venv/Scripts/python.exe src/strategy/financial_optimizer.py
if [ $? -ne 0 ]; then echo "Error in financial optimization step. Exiting."; exit 1; fi

# 6. (Optional) Run Multi-Model Backtest Visualization
if [ "$VISUALS_FLAG" = true ]; then
    echo ">>> Step 6: Generating multi-model backtest visualization (as --visuals was provided)..."
    ./.venv/Scripts/python.exe tools/visualization.py --task visualize_multi_model_backtest
    if [ $? -ne 0 ]; then echo "Error in visualization step. Exiting."; exit 1; fi
else
    echo ">>> Step 6: Skipping multi-model backtest visualization (add --visuals to include)."
fi

# 7. Launch the Live Dashboard
echo ">>> Step 7: Launching the live dashboard..."
./.venv/Scripts/streamlit.exe run main.py

echo "Pipeline execution complete."
