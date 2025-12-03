# Elon Musk Tweet Prediction & Quantitative Analysis Pipeline

## 1. Project Overview

This project implements a quantitative pipeline to predict Elon Musk's weekly tweet count and identify potential trading opportunities on Polymarket. The core of the project is a statistical arbitrage strategy that compares model-driven predictions against market-implied probabilities to find a profitable "edge".

- **Prediction Model**: A time-series forecasting model using **Facebook Prophet**, trained on historical tweet data.
- **Probability Model**: The model's raw prediction is converted into a full probability distribution for each market outcome (e.g., "200-219 tweets") using the **Negative Binomial (NBinom)** distribution.
- **Financial Strategy**: Bet sizes are determined using the **Kelly Criterion**, a formula for bet sizing that balances risk and reward. The system's parameters are optimized to maximize the **Calmar Ratio**, ensuring a focus on risk-adjusted returns.

## 2. Architecture & Workflow

The project is divided into two main phases: **Offline Analysis & Optimization** (steps you run manually to prepare the models) and the **Live Dashboard** (the interactive Streamlit application).

### Phase 1: Offline Analysis & Model Generation

This is the complete workflow to prepare all the necessary artifacts for the dashboard.

**Step 1: Ingest and Unify Data (`run_ingest.py`)**
This script fetches the latest tweets and merges them with the historical dataset. **Run this every time you want to update the data.**
```bash
python run_ingest.py
```
- **Output**: `data/processed/merged_elon_tweets.csv`

**Step 2: Evaluate Models (`tools/models_evals.py`)**
This script performs a walk-forward validation to find the best-performing Prophet model based on Log Loss. **Run this when you want to retrain or validate a new model structure.**
```bash
python tools/models_evals.py
```
- **Output**: `best_prophet_model_YYYYMMDD.pkl`

**Step 3: Generate Historical Performance (`tools/generate_historical_performance.py`)**
This script creates a backtest of predictions versus actuals. **This is a necessary step before running the financial optimizer.**
```bash
python tools/generate_historical_performance.py
```
- **Output**: `data/processed/historical_performance.csv`

**Step 4: Optimize Financial Parameters (`src/strategy/financial_optimizer.py`)**
This script uses the historical performance data to find the optimal `alpha` and `kelly_fraction` that maximize the Calmar Ratio. **Run this after generating new historical performance data.**
```bash
python src/strategy/financial_optimizer.py
```
- **Output**: `risk_params.pkl`

### Phase 2: The Streamlit Dashboard (`main.py`)

Once all the artifacts (`.csv`, `.pkl`) are generated, launch the interactive dashboard to see the final analysis and trading signals.

**How to Launch:**
```bash
streamlit run main.py
```

The dashboard is built with a modular architecture that delegates tasks to specialized processors:
- **`DashboardDataLoader`**: Loads all required data, including tweets, the Prophet model, risk parameters, and market data from Polymarket.
- **`DashboardLogicProcessor`**: Executes the core business logic, such as calculating KPIs, generating the hybrid prediction for the current week, and calculating trading opportunities by comparing model probabilities to market prices.
- **`DashboardChartGenerator`**: Creates all visualizations, such as statistical charts and the probability comparison graph.

**Dashboard Features:**
- **Current Week View**: The default view, showing:
    - A statistical analysis of recent tweet activity.
    - A "Hybrid Prediction" that combines actual tweets from the current week with model predictions for the remaining days.
    - A detailed "Trading Opportunities" table with the model's edge and the recommended Kelly Criterion bet size for each market outcome.
    - A probability chart comparing the model's predictions (a line) against the market's prices (bars).
- **Historical Performance View**:
    - A dropdown menu allows you to select any previously backtested week.
    - Displays the model's prediction (`y_pred`) versus the actual outcome (`y_true`).
    - Shows a daily bar chart of tweet activity for the selected historical week, providing clear visual context on performance.

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

4.  **Run the pipeline:**
    ```bash
    # Step 1: Update your data
    python run_ingest.py

    # Step 2: (Recommended) Generate latest historical performance
    python tools/generate_historical_performance.py

    # Step 3: Launch the dashboard
    streamlit run main.py
    ```
