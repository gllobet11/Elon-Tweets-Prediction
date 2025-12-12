"""
main.py

Elon Musk Tweet Prediction Dashboard
Corrected Version: Fixes indentation, timezone comparison for tweet counts,
and adds specific audit tools for live debugging.
"""

import asyncio
import os
import sys
import subprocess
import pandas as pd
import streamlit as st

# FIX for asyncio conflict between Streamlit and Playwright on Windows
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Internal Imports ---
try:
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.dashboard.dashboard_chart_generator import DashboardChartGenerator
    from src.dashboard.dashboard_data_loader import DashboardDataLoader
    from src.dashboard.dashboard_logic_processor import DashboardLogicProcessor
    from src.processing.feature_eng import FeatureEngineer
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"Import error. Ensure the folder structure is correct. Error: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Elon Quant Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- MAIN UI ---
st.title("Elon Musk: Prediction and Quantitative Analysis Pipeline")

try:
    with st.spinner("Loading dashboard components and data... Please wait."):
        # Initialize Processors
        data_loader = DashboardDataLoader()
        logic_processor = DashboardLogicProcessor()
        chart_generator = DashboardChartGenerator()

        # Load all necessary data
        granular_data, daily_data = data_loader.load_and_prepare_tweets_data()
        model_data = data_loader.load_prophet_model()
        risk_params = data_loader.load_risk_parameters()
        market_info = data_loader.fetch_market_data()

    # --- FIX: Indentation was broken in original script here ---
    prophet_model = model_data["model"]
    model_name = model_data.get("model_name", "Unknown")
    optimal_alpha = risk_params["alpha"]
    optimal_kelly = risk_params["kelly"]

    st.success(
        f"Model '{model_name}' loaded. Alpha: **{optimal_alpha:.4f}** | Kelly: **{optimal_kelly:.2f}**"
    )
    
    # Check if we are in fallback mode
    is_fallback = "Fallback" in market_info.get('market_question', '')
    if is_fallback:
        st.warning(f"⚠️ Market not found via API. Using simulated/fallback mode: {market_info['market_question']}")
    else:
        st.info(f"**Analyzed Market:** {market_info['market_question']}")

    # --- Sidebar Controls ---
    st.sidebar.title("Pipeline Control")
    if st.sidebar.button("REFRESH PIPELINE (takes ~2 mins)"):
        st.sidebar.info("Pipeline refresh initiated...")
        pipeline_steps = {
            "Data Ingestion": [sys.executable, "run_ingest.py"],
            "Model Training": [sys.executable, "tools/model_analysis.py", "--task", "train_and_evaluate"],
            "Financial Optimization": [sys.executable, "src/strategy/financial_optimizer.py"],
        }
        with st.expander("Pipeline Logs", expanded=True):
            for step_name, command in pipeline_steps.items():
                st.write(f"--- Running: {step_name} ---")
                try:
                    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
                    if process.returncode != 0:
                        st.error(f"Error in {step_name}:\n{process.stderr}")
                        st.stop()
                    else:
                        st.code(process.stdout[-500:]) # Show last 500 chars
                        st.success(f"{step_name} OK.")
                except Exception as e:
                    st.error(f"Execution failed: {e}")
                    st.stop()
        st.success("Refresh complete! Reloading...")
        st.rerun()

    st.sidebar.title("View Options")
    selected_strategy = st.sidebar.selectbox(
        "Select Trading Strategy:",
        ["Optimal (Financial Optimizer)", "Simple Directional (Financial Simple)"]
    )

    simple_strat_mode = "BLOCK"
    simple_strat_bet = 0.10
    if selected_strategy == "Simple Directional (Financial Simple)":
        st.sidebar.markdown("---")
        simple_strat_mode = st.sidebar.radio("Mode:", ["BLOCK", "FOCUS"])
        simple_strat_bet = st.sidebar.slider("Bet %:", 0.01, 0.20, 0.10)

    # Historical Performance Data
    historical_performance_df = st.cache_data(ttl=3600)(data_loader.load_historical_performance)()
    week_options = ["Current Week"]
    if not historical_performance_df.empty:
        historical_weeks = historical_performance_df.index.strftime("%Y-%m-%d").tolist()
        week_options.extend(historical_weeks)
    selected_week = st.sidebar.selectbox("Select Week", week_options)

    if selected_week == "Current Week":
        st.sidebar.title("Trading Settings")
        bankroll = st.sidebar.number_input("Bankroll ($):", value=1000.0, step=100.0)

        # --- KPIs Section ---
        st.subheader("📈 Statistical Activity Analysis")
        kpis = logic_processor.calculate_kpis(daily_data)
        mae = model_data.get("mae", None)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Daily Mean (Current Month)", f"{kpis['monthly_mean']:.1f}")
        c2.metric("Yesterday vs Mean", f"{kpis['yesterday_val']:.0f}", f"{kpis['deviation']:+.1f}")
        c3.metric("Outlier Days (7d)", f"{kpis['outlier_days']}", delta_color="inverse")
        c4.metric("Model MAE", f"{mae:.2f}" if mae else "N/A")

        chart_dow, chart_dist = chart_generator.generate_statistical_charts(
            daily_data=daily_data, data_last_6_months=kpis["data_last_6_months"]
        )
        sc1, sc2 = st.columns(2)
        sc1.altair_chart(chart_dow, use_container_width=True)
        sc2.altair_chart(chart_dist, use_container_width=True)

        # --- Prediction Section ---
        st.divider()
        st.subheader("🤖 Market Prediction & Live Count")

        # Refactored: Call the logic processor to get the audit count and DataFrame
        current_audit_count, audit_df = logic_processor.get_live_tweet_audit(granular_data, market_info)

        with st.spinner("Calculating opportunities..."):
            feature_engineer = FeatureEngineer()
            all_features_df = feature_engineer.process_data(granular_data)
            
            df_opportunities, pred_metrics = logic_processor.calculate_trading_opportunities(
                prophet_model=prophet_model,
                optimal_alpha=optimal_alpha,
                optimal_kelly=optimal_kelly,
                market_info=market_info,
                granular_data=granular_data,
                all_features_df=all_features_df,
                bankroll=bankroll,
                historical_performance_df=historical_performance_df,
                selected_strategy=selected_strategy,
                simple_mode=simple_strat_mode,
                simple_bet_pct=simple_strat_bet,
            )

        # Display Metrics
        p1, p2, p3 = st.columns(3)
        p1.metric("Hybrid Forecast (Total)", f"{pred_metrics['weekly_total_prediction']:.2f}")
        
        # Use our audited count if available, otherwise use processor's
        display_count = current_audit_count if not audit_df.empty else pred_metrics['sum_of_actuals']
        p2.metric("Live Tweet Count (Current)", f"{display_count}")
        
        p3.metric(
            "Expected Future Tweets", 
            f"{pred_metrics['sum_of_predictions']:.2f}",
            delta=f"{pred_metrics['remaining_days_fraction']:.1f} days left"
        )

        # --- AUDIT EXPANDER ---
        m_start = market_info.get("market_start_date")
        with st.expander(f"🕵️ Tweet Count Audit ({display_count} tweets found)"):
            if not audit_df.empty and m_start:
                st.write(f"Showing tweets from **{m_start.strftime('%Y-%m-%d %H:%M UTC')}** to Now:")
                # Show newest first
                st.dataframe(
                    audit_df.sort_index(ascending=False)[['text']], 
                    use_container_width=True,
                    column_config={"text": "Tweet Content"}
                )
            else:
                st.warning("No tweets found in the granular data for the current market period.")

        st.markdown("##### Trading Opportunities")
        st.dataframe(logic_processor.style_opportunities_df(df_opportunities), use_container_width=True)

        st.divider()
        st.subheader("📊 Probability Edge Visualization")
        final_chart = chart_generator.generate_probability_comparison_chart(df_opportunities)
        st.altair_chart(final_chart, use_container_width=True)

        # --- Debugger Expander ---
        with st.expander("🐞 Price Calculation Debugger"):
            st.markdown("""
            This section shows the raw components of the live price calculation for **active bins only**.
            The `Mkt Price` is the midpoint between the best `Bid` (highest price to buy) and the best `Ask` (lowest price to sell).
            If a bin is not listed here, it's because it is closed or has no liquidity.
            """)
            
            # Filter for active bins to debug - only if Status column exists
            if "Status" in df_opportunities.columns:
                # Optimal strategy - filter out bins without liquidity
                debug_df = df_opportunities[
                    ~df_opportunities["Status"].isin(["NO_LIQUIDITY", "MISSING_ID"])
                ].copy()
                
                if not debug_df.empty:
                    st.dataframe(
                        debug_df[["Bin", "Mkt Price", "Status"]],
                        use_container_width=True
                    )
                else:
                    st.warning("No active bins with live prices were found to debug.")
            else:
                # Simple Directional strategy - show all bins (no Status column)
                st.info("Simple Directional Strategy - showing all bins with market prices:")
                st.dataframe(
                    df_opportunities[["Bin", "Mkt Price", "My Model", "Edge"]],
                    use_container_width=True
                )
    else:
        # --- Historical View ---
        st.subheader(f"🕰️ Performance: Week of {selected_week}")
        selected_week_dt = pd.to_datetime(selected_week).tz_localize("UTC")
        
        try:
            week_data = historical_performance_df.loc[selected_week_dt]
            col_hp1, col_hp2 = st.columns(2)
            col_hp1.metric("Model Prediction", f"{week_data['y_pred']:.2f}")
            col_hp2.metric("Actual Result", f"{week_data['y_true']:.0f}")
            
            # Chart logic for historical...
            end_of_hist_week = selected_week_dt + pd.Timedelta(days=7)
            mask = (daily_data.index >= selected_week_dt) & (daily_data.index < end_of_hist_week)
            df_hist_chart = daily_data[mask]
            
            if not df_hist_chart.empty:
                if isinstance(df_hist_chart, pd.Series): df_hist_chart = df_hist_chart.to_frame(name="n_tweets")
                hist_chart = chart_generator.generate_historical_week_chart(df_hist_chart)
                st.altair_chart(hist_chart, use_container_width=True)
            else:
                st.info("No daily data for this week.")
        except KeyError:
             st.warning("Data not found for this specific date in historical records.")

except Exception as e:
    st.error(f"Critical Dashboard Error: {e}")
    st.exception(e)