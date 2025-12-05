import subprocess
import sys
import argparse
from loguru import logger

# --- Configuration ---
# Define the sequence of steps in the pipeline
PIPELINE_STEPS = {
    "ingest": ["run_ingest.py"],
    "train_evaluate": ["tools/model_analysis.py", "--task", "train_and_evaluate"],
    "optimize_financials": ["src/strategy/financial_optimizer.py"],
    "visualize_backtest": ["tools/visualization.py", "--task", "visualize_multi_model_backtest"],
    "launch_dashboard": ["-m", "streamlit", "run", "main.py"]
}

def run_step(command, step_name):
    """Executes a pipeline step using the active Python interpreter."""
    logger.info(f">>> Running Step: {step_name}...")
    
    # Use sys.executable to ensure we're using the Python from the activated virtual environment
    # The command list is constructed with the python interpreter followed by the script and its arguments
    full_command = [sys.executable] + command
    
    # Use Popen for real-time output, especially for long-running processes like streamlit
    process = subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')

    # Read and print output line-by-line in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    # Wait for the process to complete and get the return code
    rc = process.poll()

    if rc != 0:
        logger.error(f"--- ERROR in step: {step_name} ---")
        logger.error(f"Command '{' '.join(full_command)}' failed with return code {rc}.")
        sys.exit(1)
    else:
        logger.success(f"--- Step '{step_name}' completed successfully. ---")

def main(args):
    """Main function to run the pipeline steps."""
    logger.info("🚀 Starting Elon Tweets Prediction Pipeline...")

    # --- Run Mandatory Steps ---
    run_step(PIPELINE_STEPS["ingest"], "Data Ingestion")
    
    if args.optimize:
        logger.info("Optimization flag set. Running full hyperparameter tuning.")
        # Replace the standard train_evaluate with the more intensive tuning tasks
        run_step(["tools/model_analysis.py", "--task", "tune_hyperparameters"], "Hyperparameter Tuning")
        run_step(["tools/model_analysis.py", "--task", "train_with_tuned_hps"], "Train with Tuned HPs")
    else:
        run_step(PIPELINE_STEPS["train_evaluate"], "Model Training & Evaluation")

    run_step(PIPELINE_STEPS["optimize_financials"], "Financial Parameter Optimization")
    
    # --- Run Optional Steps ---
    if args.visuals:
        run_step(PIPELINE_STEPS["visualize_backtest"], "Generate Visuals")
    else:
        logger.info("Skipping visualization step. Use --visuals flag to include.")
        
    # --- Launch Dashboard ---
    if args.dashboard:
        logger.info("Launching the live dashboard...")
        # The dashboard step is run last and will keep running
        run_step(PIPELINE_STEPS["launch_dashboard"], "Launch Dashboard")
    else:
        logger.info("Skipping dashboard launch. Use --dashboard flag to launch.")

    logger.success("✅ Pipeline execution complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete Elon Musk Tweet Prediction Pipeline.")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run the extensive hyperparameter tuning process. This is computationally intensive."
    )
    parser.add_argument(
        "--visuals",
        action="store_true",
        help="Generate and save the multi-model backtest visualization plot."
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard after the pipeline completes."
    )
    
    args = parser.parse_args()
    main(args)
