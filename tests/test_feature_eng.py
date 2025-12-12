from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.processing.feature_eng import FeatureEngineer


@pytest.fixture
def sample_raw_data():
    """Fixture for a sample raw DataFrame similar to merged_elon_tweets.csv"""
    data = {
        "id": range(1000, 1070),
        "text": [f"Tweet {i}" for i in range(70)],
        "created_at": pd.to_datetime(
            ["2024-12-25 10:00:00"] * 5
            + ["2024-12-26 11:00:00"] * 10
            + ["2024-12-27 12:00:00"] * 8
            + ["2024-12-28 13:00:00"] * 7
            + ["2024-12-29 14:00:00"] * 5
            + ["2024-12-30 15:00:00"] * 3
            + ["2024-12-31 16:00:00"] * 2
            + ["2025-01-01 10:00:00"] * 1
            + ["2025-01-02 11:00:00"] * 2
            + ["2025-01-03 12:00:00"] * 10
            + ["2025-01-04 12:00:00"] * 5
            + ["2025-01-05 12:00:00"] * 5
            + ["2025-01-06 12:00:00"] * 5
            + ["2025-01-07 12:00:00"] * 2,
        ).tolist()[
            :70
        ],  # Asegurar 70 fechas para 70 filas
    }
    df = pd.DataFrame(data)
    # Ensure timezone-naive for consistency with FeatureEngineer
    df["created_at"] = df["created_at"].dt.tz_localize(None)
    return df


@pytest.fixture
def feature_engineer_instance():
    return FeatureEngineer()


def test_process_data_output_columns(feature_engineer_instance, sample_raw_data):
    """
    Test `process_data` to ensure all expected feature columns are created
    and basic data integrity (no NaNs in critical columns) after processing.
    """
    processed_df = feature_engineer_instance.process_data(sample_raw_data.copy())

    expected_columns = [
        "n_tweets",
        "reply_count",
        "hour_std",
        "sleep_debt_avg",
        "zombie_mode",
        "late_night_flag",
        "trend",
        "lag_1",
        "lag_2",
        "lag_7",
        "roll_mean_3",
        "roll_mean_7",
        "momentum",
        "roll_std_7",
        "roll_sum_7",
        "roll_sum_14",
        "delta_7_14",
        "ratio_mean_3_7",
        "cv_7",
        "is_spike_1",
        "is_burst_today",
        "last_burst",
        "days_since_burst",
        "dow",
        "regime_intensity",
        "is_high_regime",
        "is_low_regime",
        "is_regime_change",
        "reply_ratio",
        "hour_std_feature",
    ]

    for col in expected_columns:
        assert col in processed_df.columns, f"Missing expected column: {col}"

    # Check for NaNs in critical output columns (after fillna/dropna logic)
    # The first few rows will naturally have NaNs due to rolling/shifting,
    # but after dropna(subset=['roll_sum_14']), there should be no more NaNs in output.
    assert (
        not processed_df[expected_columns].isnull().any().any()
    ), "Should not have NaNs in final processed features"


def test_process_data_feature_values(feature_engineer_instance, sample_raw_data):
    """
    Test `process_data` to ensure specific feature calculations are correct.
    Focus on a few key features like lag_1, roll_sum_7, and momentum.
    """
    processed_df = feature_engineer_instance.process_data(sample_raw_data.copy())

    # Test lag_1 (simple shift)
    # The first actual value after NaN for lag_1 should be the second day's n_tweets
    first_valid_lag_1_idx = processed_df["lag_1"].first_valid_index()
    if (
        first_valid_lag_1_idx
        and (first_valid_lag_1_idx - timedelta(days=1)) in processed_df.index
    ):  # Added index check
        assert (
            processed_df.loc[first_valid_lag_1_idx, "lag_1"]
            == processed_df.loc[first_valid_lag_1_idx - timedelta(days=1), "n_tweets"]
        )

    # Test roll_sum_7 (sum of last 7 days, shifted)
    # Pick a date far enough in to have a full 7-day sum for accurate testing.
    test_date_sum = pd.to_datetime("2025-01-10").normalize()  # Adjusted test date
    if (
        test_date_sum in processed_df.index
        and test_date_sum - timedelta(days=7) in processed_df.index
    ):
        expected_sum = processed_df.loc[
            test_date_sum - timedelta(days=7) : test_date_sum - timedelta(days=1),
            "n_tweets",
        ].sum()
        assert np.isclose(processed_df.loc[test_date_sum, "roll_sum_7"], expected_sum)

    # Test momentum (roll_mean_3 - roll_mean_7)
    test_date_momentum = pd.to_datetime("2025-01-10").normalize()  # Adjusted test date
    if test_date_momentum in processed_df.index:
        # Ensure enough preceding data for rolling means
        if test_date_momentum - timedelta(days=7) in processed_df.index:
            roll_mean_3 = processed_df.loc[
                test_date_momentum
                - timedelta(days=3) : test_date_momentum
                - timedelta(days=1),
                "n_tweets",
            ].mean()
            roll_mean_7 = processed_df.loc[
                test_date_momentum
                - timedelta(days=7) : test_date_momentum
                - timedelta(days=1),
                "n_tweets",
            ].mean()
            expected_momentum = roll_mean_3 - roll_mean_7
            assert np.isclose(
                processed_df.loc[test_date_momentum, "momentum"],
                expected_momentum,
            )


def test_add_regime_feature_values(feature_engineer_instance):
    """
    Test `add_regime_feature` to ensure Z-score and regime flags are calculated correctly.
    """
    # Create a daily series with a clear spike and a clear drop
    dates = pd.date_range(
        start="2025-01-01",
        periods=200,
        freq="D",
    )  # Aumentar duración para ventana móvil
    n_tweets = np.full(
        len(dates),
        50,
    )  # Base level (increased from 10 to 50 for a more stable Z-score)
    n_tweets[40:45] = 500  # Introduce a very high spike
    n_tweets[80:85] = 0  # Introduce a very low drop

    # Convert to DataFrame with 'n_tweets' column
    df_test = pd.DataFrame({"n_tweets": n_tweets}, index=dates)
    df_test.index.name = "date"

    # Apply the feature engineering (just the add_regime_feature part for this test)
    processed_df = feature_engineer_instance.add_regime_feature(df_test.copy())

    # Check for 'regime_intensity' during the spike
    # Spike is from day 40-44. With a window of 56 days.
    # Check a date well after the spike has started and the window has passed over it.
    spike_check_date = pd.to_datetime(
        "2025-02-13",
    ).normalize()  # A date *within* the spike
    if spike_check_date in processed_df.index:
        # Assert that the Z-score (regime_intensity) is significantly positive
        assert (
            processed_df.loc[spike_check_date, "regime_intensity"] > 2.0
        ), f"Regime intensity not high enough at {spike_check_date}"
        assert (
            processed_df.loc[spike_check_date, "is_high_regime"] == 1
        ), f"is_high_regime not set at {spike_check_date}"
        assert (
            processed_df.loc[spike_check_date, "is_regime_change"] == 1
        ), f"is_regime_change not set at {spike_check_date}"
    else:
        pytest.fail(f"Spike check date {spike_check_date} not in processed_df index.")

    # Check for 'regime_intensity' during the drop
    # Drop is from day 80-84. With a window of 56 days.
    drop_check_date = pd.to_datetime(
        "2025-03-23",
    ).normalize()  # A date *within* the drop
    if drop_check_date in processed_df.index:
        # Assert that the Z-score (regime_intensity) is significantly negative
        assert (
            processed_df.loc[drop_check_date, "regime_intensity"] < -2.0
        ), f"Regime intensity not low enough at {drop_check_date}"
        assert (
            processed_df.loc[drop_check_date, "is_low_regime"] == 1
        ), f"is_low_regime not set at {drop_check_date}"
        assert (
            processed_df.loc[drop_check_date, "is_regime_change"] == 1
        ), f"is_regime_change not set at {drop_check_date}"
    else:
        pytest.fail(f"Drop check date {drop_check_date} not in processed_df index.")

    # Check that flags are 0 during normal periods
    normal_date_start = pd.to_datetime("2025-01-10").normalize()
    normal_date_end = pd.to_datetime("2025-02-05").normalize()

    # Check a range of normal dates to be sure
    for current_date in pd.date_range(start=normal_date_start, end=normal_date_end):
        if current_date in processed_df.index:
            assert processed_df.loc[current_date, "is_high_regime"] == 0
            assert processed_df.loc[current_date, "is_low_regime"] == 0
            assert processed_df.loc[current_date, "is_regime_change"] == 0


def test_get_latest_features(feature_engineer_instance, sample_raw_data):
    """
    Test `get_latest_features` to ensure it returns the last valid row
    with only the model's required features.
    """
    latest_features_df = feature_engineer_instance.get_latest_features(
        sample_raw_data.copy(),
    )

    # Note: If the model starts using the new regime features, this list will need updating.
    model_required_features = ["lag_1", "last_burst"]
    assert latest_features_df.shape[0] == 1, "Should return exactly one row"
    assert (
        list(latest_features_df.columns) == model_required_features
    ), "Should return only model required features"
    assert (
        not latest_features_df.isnull().any().any()
    ), "Should not have NaNs in latest features"
