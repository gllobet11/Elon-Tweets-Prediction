# Elon Musk Tweet Prediction & Quantitative Analysis Pipeline

## 1. Project Overview

This project implements a quantitative pipeline to predict Elon Musk's weekly tweet count and identify potential trading opportunities on Polymarket. The core of the project is a statistical arbitrage strategy that compares model-driven predictions against market-implied probabilities to find a profitable "edge".

-   **Prediction Model**: A time-series forecasting model using **Facebook Prophet**, trained on historical tweet data.
-   **Probability Model**: The model's raw prediction is converted into a full probability distribution for each market outcome (e.g., "200-219 tweets") using the **Negative Binomial (NBinom)** distribution.
-   **Financial Strategy**: Bet sizes are determined using the **Kelly Criterion**, a formula for bet sizing that balances risk and reward. The system's parameters are optimized to maximize the **Calmar Ratio**, ensuring a focus on risk-adjusted returns.

## 2. Architecture & Workflow

The project is divided into two main phases: **Offline Analysis & Optimization** (steps you run manually to prepare the models) and the **Live Dashboard** (the interactive Streamlit application).

### Phase 1: Offline Analysis & Model Generation

This phase involves preparing and optimizing the predictive model and financial strategy parameters. These steps are typically run when you want to update your model, re-evaluate its performance, or fine-tune its parameters.

**Step 1: Ingest and Unify Data (`run_ingest.py`)**
This script fetches the latest tweets and merges them with the historical dataset. **Run this every time you want to update the data.**
```bash
python run_ingest.py
```
-   **Output**: `data/processed/merged_elon_tweets.csv`

**Step 2: Hyperparameter Tuning (Optional, Advanced) (`tools/hyperparameter_tuner.py`)**
This script performs an extensive search for the optimal Prophet model hyperparameters (e.g., `changepoint_prior_scale`, `seasonality_prior_scale`) and Negative Binomial distribution parameters (`alpha`) using walk-forward validation. This is generally a **one-time optimization step** after feature engineering, or when you suspect model performance has degraded.
```bash
python tools/hyperparameter_tuner.py
```
-   **Output**: Logs showing the best found hyperparameters. No direct file output, but guides subsequent steps.

**Step 3: Evaluate Models & Train/Save the Best Production Model (`tools/models_evals.py`)**
This script performs a walk-forward validation to evaluate different model configurations and, crucially, trains and saves the **production-ready Prophet model** with the optimal features and hyperparameters (including those found by `hyperparameter_tuner.py`).
```bash
python tools/models_evals.py
```
-   **Output**: `best_prophet_model_YYYYMMDD.pkl` (e.g., `best_prophet_model_20251203.pkl`)

**Step 4: Generate Historical Performance (`tools/generate_historical_performance.py`)**
This script uses the **saved production model** (`best_prophet_model_YYYYMMDD.pkl`) to generate backtested predictions (`y_pred`, `y_pred_lower`, `y_pred_upper`) versus actual outcomes (`y_true`) for a specified number of past weeks. This output is critical for the financial optimizer and for visualizing historical performance with confidence bands.
```bash
python tools/generate_historical_performance.py
```
-   **Output**: `data/processed/historical_performance.csv`

**Step 5: Optimize Financial Parameters (`src/strategy/financial_optimizer.py`)**
This script uses the `historical_performance.csv` data to find the optimal `alpha` (for the Negative Binomial distribution) and `kelly_fraction` that maximize the Calmar Ratio. This process is fully deterministic and also calculates the Expected Value (EV) of the strategy.
```bash
python src/strategy/financial_optimizer.py
```
-   **Output**: `risk_params.pkl` (containing the optimized financial parameters).

**Step 6: Visualize Historical Predictions (Optional) (`tools/visualize_predictions.py`)**
This script generates a plot comparing the model's historical predictions against actuals, now including confidence bands, along with RMSE and Log Loss metrics. Useful for quick visual checks of model performance.
```bash
python tools/visualize_predictions.py
```
-   **Output**: `historical_predictions_plot.png`

### Phase 2: The Streamlit Dashboard (`main.py`)

Once all the artifacts (`.csv`, `.pkl`) are generated, launch the interactive dashboard to see the final analysis and trading signals.

**How to Launch:**
```bash
streamlit run main.py
```

The dashboard is built with a modular architecture that delegates tasks to specialized processors:
-   **`DashboardDataLoader`**: Loads all required data, including tweets, the Prophet model, risk parameters, and market data from Polymarket.
-   **`DashboardLogicProcessor`**: Executes the core business logic, such as calculating KPIs, generating the hybrid prediction for the current week, and calculating trading opportunities by comparing model probabilities to market prices.
-   **`DashboardChartGenerator`**: Creates all visualizations, such as statistical charts and the probability comparison graph.

**Dashboard Features:**
-   **Current Week View**: The default view, showing:
    -   A statistical analysis of recent tweet activity.
    -   A "Hybrid Prediction" that combines actual tweets from the current week with model predictions for the remaining days.
    -   A detailed "Trading Opportunities" table with the model's edge and the recommended Kelly Criterion bet size for each market outcome.
    -   A probability chart comparing the model's predictions (a line) against the market's prices (bars).
-   **Historical Performance View**:
    -   A dropdown menu allows you to select any previously backtested week.
    -   Displays the model's prediction (`y_pred`) versus the actual outcome (`y_true`), now with confidence bands.
    -   Shows a daily bar chart of tweet activity for the selected historical week, providing clear visual context on performance.

## 3. Installation and Usage

1.  **Clone the repository and navigate into it.**

2.  **Create and activate a virtual environment:**
    ```bash
    # We recommend using 'uv' for faster performance
    pip install uv

    # Create venv
    uv venv

    # Activate venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Full Pipeline Execution (Daily / Weekly Refresh)**:
    To update all predictions and parameters for the live dashboard:
    ```bash
    # Step 1: Ingest latest tweet data
    python run_ingest.py

    # Step 2: Evaluate models, train and save the best production model
    python tools/models_evals.py

    # Step 3: Generate historical performance data using the best model
    python tools/generate_historical_performance.py

    # Step 4: Optimize financial parameters based on historical performance
    python src/strategy/financial_optimizer.py

    # Step 5: (Optional) Visualize historical predictions with confidence bands
    python tools/visualize_predictions.py

    # Step 6: Launch the interactive Streamlit dashboard
    streamlit run main.py
    ```
    **Note on Hyperparameter Tuning**: The `tools/hyperparameter_tuner.py` script is for **advanced, one-time optimization** of model parameters, not part of the regular refresh cycle. It helps determine the best settings used by `tools/models_evals.py`.