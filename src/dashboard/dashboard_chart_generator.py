"""
dashboard_chart_generator.py

This module handles the generation of Altair chart objects.
It centralizes visualization logic to keep the Streamlit UI clean.

Improvements:
- Interactive tooltips.
- Color-coded "Edge" visualization for decision making.
- Optimized bar sizing for historical views.
"""

import altair as alt
import pandas as pd


class DashboardChartGenerator:
    """
    Generates Altair chart objects for the dashboard.
    """

    def __init__(self):
        # Define a consistent color palette
        self.color_market = "#ff7f0e"  # Orange
        self.color_model = "#1f77b4"  # Blue
        self.color_edge_pos = "#2ca02c"  # Green
        self.color_edge_neg = "#d62728"  # Red
        self.color_bar_neutral = "#aec7e8"  # Light Blue

    def generate_statistical_charts(
        self, daily_data: pd.Series, data_last_6_months: pd.DataFrame
    ) -> tuple[alt.Chart, alt.Chart]:
        """
        Generates statistical charts for tweet activity.
        1. Day of Week Average.
        2. Weekly Sum Distribution.
        """
        # --- Chart 1: Day of Week Analysis ---
        data_last_6_months = data_last_6_months.copy()
        data_last_6_months["day_of_week"] = data_last_6_months.index.day_name()

        avg_by_day = (
            data_last_6_months.groupby("day_of_week")["n_tweets"].mean().reset_index()
        )

        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        # Sort logic
        avg_by_day["day_sort"] = avg_by_day["day_of_week"].apply(
            lambda x: day_order.index(x)
        )

        chart_dow = (
            alt.Chart(avg_by_day)
            .mark_bar(color=self.color_model, opacity=0.8)
            .encode(
                x=alt.X("day_of_week", sort=day_order, title=None),
                y=alt.Y("n_tweets", title="Avg Tweets"),
                tooltip=[
                    alt.Tooltip("day_of_week", title="Day"),
                    alt.Tooltip("n_tweets", title="Avg Tweets", format=".1f"),
                ],
            )
            .properties(title="Average Tweets by Day (Last 6 Months)", height=250)
        )

        # --- Chart 2: Weekly Distribution Histogram ---
        weekly_sums = (
            data_last_6_months.resample("W-MON")["n_tweets"].sum().reset_index()
        )

        chart_dist = (
            alt.Chart(weekly_sums)
            .mark_bar(color=self.color_bar_neutral)
            .encode(
                x=alt.X(
                    "n_tweets", bin=alt.Bin(maxbins=20), title="Weekly Tweet Count"
                ),
                y=alt.Y("count()", title="Frequency (Weeks)"),
                tooltip=[
                    alt.Tooltip("n_tweets", bin=True, title="Range"),
                    alt.Tooltip("count()", title="Weeks Count"),
                ],
            )
            .properties(
                title="Weekly Tweet Count Distribution (Last 6 Months)", height=250
            )
        )

        return chart_dow, chart_dist

    def generate_probability_comparison_chart(
        self, df_opportunities: pd.DataFrame
    ) -> alt.Chart:
        """
        Generates a layered chart comparing Model vs Market probabilities.

        IMPROVEMENT: Colors the Model points based on 'Edge' magnitude.
        Green = Positive Edge (Buy), Red/Gray = Negative Edge.
        """

        # Helper to parse bin midpoints for plotting
        def get_midpoint(bin_label: str) -> float:
            if "+" in str(bin_label):
                # Handle "320+" cases
                clean = str(bin_label).replace("+", "").replace(",", "")
                return int(clean) + 20
            parts = str(bin_label).replace(",", "").split("-")
            try:
                return (int(parts[0]) + int(parts[1])) / 2
            except:
                return 0

        df_opportunities = df_opportunities.copy()
        df_opportunities["Midpoint"] = df_opportunities["Bin"].apply(get_midpoint)

        # Calculate Edge for color coding if not present, though logic_processor usually provides it
        if "Edge" not in df_opportunities.columns:
            df_opportunities["Edge"] = (
                df_opportunities["My Model"] - df_opportunities["Mkt Price"]
            )

        base = alt.Chart(df_opportunities).encode(
            x=alt.X(
                "Midpoint:Q",
                title="Number of Tweets",
                axis=alt.Axis(labelAngle=0),  # Easier to read horizontal
            )
        )

        # Layer 1: Market Prices (Orange Bars)
        market_bars = base.mark_bar(
            width=25,
            opacity=0.3,
            color=self.color_market,
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3,
        ).encode(
            y=alt.Y("Mkt Price:Q", title="Probability", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("Bin", title="Bin Range"),
                alt.Tooltip("Mkt Price", title="Market Prob", format=".1%"),
            ],
        )

        # Layer 2: Model Line (Connecting the dots)
        model_line = base.mark_line(
            color=self.color_model,
            strokeDash=[5, 5],  # Dashed line to differentiate from bars
            opacity=0.6,
        ).encode(y=alt.Y("My Model:Q"))

        # Layer 3: Model Points (Colored by Edge)
        # We create a specific color scale for the edge
        model_points = base.mark_circle(size=100, opacity=1.0).encode(
            y=alt.Y("My Model:Q"),
            color=alt.condition(
                alt.datum.Edge > 0,
                alt.value(self.color_edge_pos),  # Green if Edge > 0
                alt.value(self.color_edge_neg),  # Red if Edge <= 0
            ),
            tooltip=[
                alt.Tooltip("Bin", title="Bin"),
                alt.Tooltip("My Model", title="My Prob", format=".1%"),
                alt.Tooltip("Mkt Price", title="Mkt Prob", format=".1%"),
                alt.Tooltip("Edge", title="Edge", format=".1%"),
                alt.Tooltip("Bet Size ($)", title="Kelly Bet", format="$.2f"),
            ],
        )

        final_chart = (
            (market_bars + model_line + model_points)
            .properties(
                title="Probability Analysis: Green points indicate a positive Edge (Buy Signal)",
                height=400,
            )
            .interactive()
        )

        return final_chart

    def generate_historical_week_chart(
        self, daily_data_for_week: pd.DataFrame
    ) -> alt.Chart:
        """
        Generates a bar chart for daily activity of a historical week.

        IMPROVEMENT: Uses Ordinal dates to fix 'thin bar' issues and adds a mean line.
        """
        if daily_data_for_week.empty or "n_tweets" not in daily_data_for_week.columns:
            return alt.Chart(pd.DataFrame()).mark_text(text="No data to display.")

        data_to_plot = daily_data_for_week.reset_index().rename(
            columns={"index": "date"}
        )
        # Create a formatted string for the axis to ensure discrete bars
        data_to_plot["date_str"] = data_to_plot["date"].dt.strftime("%Y-%m-%d (%a)")

        # Calculate mean for the rule line
        mean_val = data_to_plot["n_tweets"].mean()

        # Base Chart
        base = alt.Chart(data_to_plot).encode(
            x=alt.X("date_str:O", title="Date", axis=alt.Axis(labelAngle=-45))
        )

        # Layer 1: Bars
        bars = base.mark_bar(color=self.color_model).encode(
            y=alt.Y("n_tweets:Q", title="Tweets"),
            tooltip=[
                alt.Tooltip("date_str", title="Date"),
                alt.Tooltip("n_tweets", title="Count"),
            ],
        )

        # Layer 2: Text labels on bars
        text = base.mark_text(dy=-10).encode(
            y=alt.Y("n_tweets:Q"), text=alt.Text("n_tweets:Q")
        )

        # Layer 3: Average Rule
        rule = (
            alt.Chart(pd.DataFrame({"mean": [mean_val]}))
            .mark_rule(color="red", strokeDash=[4, 4])
            .encode(y="mean:Q")
        )

        return (bars + text + rule).properties(
            title=f"Daily Activity vs Weekly Avg ({mean_val:.1f})", height=300
        )

    def generate_feature_importance_chart(self, feature_importance_df: pd.DataFrame) -> alt.Chart:
        """
        Generates a horizontal bar chart for feature importance.
        """
        if feature_importance_df.empty:
            return alt.Chart(pd.DataFrame()).mark_text(text="No feature importance data available.").properties(height=150)

        chart = alt.Chart(feature_importance_df).mark_bar().encode(
            x=alt.X('Magnitude:Q', title='Coefficient Magnitude (Impact on Prediction)'),
            y=alt.Y('Feature:N', sort='-x', title='Feature'),
            color=alt.Color('Impact:N', 
                            scale=alt.Scale(
                                domain=['Positive', 'Negative'], 
                                range=[self.color_edge_pos, self.color_edge_neg]
                            ),
                            legend=alt.Legend(title="Impact")),
            tooltip=[
                alt.Tooltip('Feature', title='Feature'),
                alt.Tooltip('Coefficient', title='Coefficient Value', format='.4f')
            ]
        ).properties(
            title='Live Feature Influence on Prediction',
            height=200
        )
        return chart
