# Elon Musk Tweet Prediction & Quantitative Analysis Pipeline

## 1. Project Overview

This project implements a quantitative pipeline to predict Elon Musk's weekly tweet count and identify potential trading opportunities on Polymarket. The core of the project is a statistical arbitrage strategy that compares model-driven predictions against market-implied probabilities to find a profitable "edge".

-   **Prediction Model**: A time-series forecasting model using **Facebook Prophet**, trained on historical tweet data.
-   **Probability Model**: The model's raw prediction is converted into a full probability distribution for each market outcome (e.g., "200-219 tweets") using the **Negative Binomial (NBinom)** distribution.
-   **Financial Strategy**: Bet sizes are determined using the **Kelly Criterion**, a formula for bet sizing that balances risk and reward. The system's parameters are optimized to maximize the **Calmar Ratio**, ensuring a focus on risk-adjusted returns.

## 2. Architecture & Workflow

The project is divided into two main phases: an **Offline Pipeline** for data processing and model training, and a **Live Dashboard** for analysis and trading signals. The entire offline process can be executed with a single automated script.

### The Automated Pipeline (`run_pipeline.sh`)

A bash script is provided to run the entire offline workflow. It automates all the necessary steps, from data ingestion to financial optimization.

**How to Run:**
```bash
# Activate your virtual environment first
# On Windows (Git Bash)
source .venv/Scripts/activate
# On macOS/Linux
# source .venv/bin/activate

# Run the full pipeline
bash run_pipeline.sh
```

**Optional Flags:**
-   `--optimize`: Runs a comprehensive hyperparameter tuning process for the Prophet model and Negative Binomial distribution. This is computationally intensive and should be used periodically, not on every run.
-   `--visuals`: Generates a plot comparing the backtest performance of all candidate models against the actual tweet counts.

**Example with flags:**
```bash
bash run_pipeline.sh --optimize --visuals
```

### Manual Workflow Steps

The automated script executes the following steps, which can also be run manually for debugging or granular control.

**Step 1: Ingest and Unify Data (`run_ingest.py`)**
-   **Purpose**: Fetches the latest tweets and merges them with the historical dataset.
-   **Command**: `python run_ingest.py`
-   **Output**: `data/processed/merged_elon_tweets.csv`

**Step 2: Model Evaluation and Training (`tools/model_analysis.py`)**
-   **Purpose**: Compares different model configurations (feature sets), trains the best one, and saves it as a `.pkl` file. It also generates the data for the multi-model backtest visualization.
-   **Command**: `python tools/model_analysis.py --task train_and_evaluate`
-   **Outputs**: 
    -   `best_prophet_model_YYYYMMDD.pkl`
    -   `data/processed/all_models_historical_performance.csv`

**Step 3: Generate Historical Performance (`tools/generate_historical_performance.py`)**
-   **Purpose**: Uses the saved production model to generate backtested predictions for financial optimization.
-   **Command**: `python tools/generate_historical_performance.py`
-   **Output**: `data/processed/historical_performance.csv`

**Step 4: Optimize Financial Parameters (`src/strategy/financial_optimizer.py`)**
-   **Purpose**: Finds the optimal risk parameters (Kelly fraction) by maximizing the Calmar Ratio based on the historical performance data.
-   **Command**: `python src/strategy/financial_optimizer.py`
-   **Output**: `risk_params.pkl`

### The Live Dashboard (`main.py`)

Once all the artifacts are generated, the interactive Streamlit dashboard provides the final analysis and trading signals.

**How to Launch:**
```bash
streamlit run main.py
```

The dashboard loads the latest data, the trained model, and the optimized risk parameters to provide real-time insights.

## 3. Installation and Usage

1.  **Clone the repository and navigate into it.**

2.  **Create and activate a virtual environment:**
    ```bash
    # We recommend using 'uv' for faster performance
    pip install uv

    # Create venv
    uv venv

    # Activate venv
    # Windows (in Git Bash)
    source .venv/Scripts/activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Run the Automated Pipeline:**
    To update all data, train the model, and optimize parameters, simply run the pipeline script.
    ```bash
    bash run_pipeline.sh
    ```
    *(Optional: Add `--optimize` or `--visuals` flags as needed.)*

5.  **Launch the Dashboard:**
    After the pipeline completes, start the interactive application.
    ```bash
    streamlit run main.py
    ```

## 4. Optional Tools

The `tools/` directory contains consolidated scripts for advanced analysis, verification, and visualization.

-   **`tools/model_analysis.py`**:
    -   `--task display_feature_importance`: Shows Prophet regressor coefficients.
    -   `--task run_forward_selection`: Performs a greedy feature selection.
    -   `--task tune_hyperparameters`: Runs an extensive hyperparameter search.
-   **`tools/visualization.py`**:
    -   `--task visualize_predictions`: Plots the single best model's predictions with confidence intervals.
    -   `--task visualize_multi_model_backtest`: Plots all candidate models' backtest predictions.
    -   `--task visualize_regime_change`: Visualizes tweet activity Z-scores over time.
-   **`tools/data_verification.py`**: A suite of tools to check data integrity at various stages.
-   **`tools/utilities.py`**: Helper functions for inspecting saved `.pkl` files and testing the Polymarket feed.