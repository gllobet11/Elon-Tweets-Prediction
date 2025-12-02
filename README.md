# Elon Musk Tweets Prediction & Polymarket Trading Strategy

## 1. Project Overview

This project implements a fully automated quantitative pipeline to predict the weekly tweet count of Elon Musk and execute a statistical arbitrage strategy on the corresponding Polymarket market.

- **Objective**: To find and capitalize on discrepancies between our model's tweet count predictions and the market's implied probabilities.
- **Strategy**: Statistical arbitrage. We compare our model's estimated probability distribution (Fair Value) against market prices to find a positive "Edge" (positive expected value).
- **Core Technology**: The prediction engine is built on **Facebook's Prophet** model, while the financial risk parameters are optimized to maximize the **Calmar Ratio**.

## 2. System Architecture

The project is structured into several key components:

```
Elon-tweets-Prediction/
│
├── config/                 # Configuration files (market bins, keywords)
│
├── data/                   # Raw and processed data
│
├── src/
│   ├── ingestion/          # Modules for data ingestion (Polymarket, Twitter)
│   ├── processing/         # Feature engineering scripts
│   ├── strategy/           # Core strategy logic (prediction, probability math, financial optimization)
│   └── dashboard/          # Modular components for the Streamlit UI
│
├── tools/                  # Offline scripts for model evaluation and optimization
│
├── main.py                 # Main entry point for the Streamlit dashboard
├── requirements.txt        # Project dependencies
├── best_prophet_model_...pkl # Saved best-performing Prophet model
├── risk_params.pkl         # Saved optimal risk parameters (alpha and kelly)
└── README.md
```

## 3. Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Elon-tweets-Prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 4. Usage: The Main Workflow

The project has a clear, four-step workflow to go from raw data to actionable trading signals.

### Step 1: Ingest Data

Run this script to download the latest market and tweet data.

```bash
python run_ingest.py
```

### Step 2: Evaluate and Select the Best Prediction Model

Run this script to perform a walk-forward validation on different Prophet model configurations. It evaluates them based on Log Loss and saves the best-performing model.

```bash
python tools/models_evals.py
```
**Output**: `best_prophet_model_YYYYMMDD.pkl`

### Step 3: Optimize Financial Risk Parameters

This script takes the best model from the previous step and runs a series of trading simulations to find the optimal `alpha` (for probability distribution) and `kelly_fraction` (for bet sizing) that maximize the Calmar Ratio.

```bash
python tools/financial_optimizer.py
```
**Output**: `risk_params.pkl`

### Step 4: Launch the Dashboard

With the latest data, the best model, and optimal risk parameters, launch the Streamlit dashboard to see the final analysis and trading signals.

```bash
streamlit run main.py
```

The dashboard will automatically load `best_prophet_model_*.pkl` and `risk_params.pkl` to provide real-time insights.