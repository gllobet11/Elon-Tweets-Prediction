import os
import sys
from datetime import datetime
from typing import Iterable

import requests
import numpy as np
import pandas as pd

# Añadir raíz del proyecto al path (si de verdad lo necesitas)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.processing.feature_eng import (
    calculate_spacex_features_vectorized,
)  # ya existe en tu proyecto


# -------------------- Config / Constantes --------------------

LL2_BASE_URL = "https://ll.thespacedevs.com/2.2.0/launch/"
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_TIMEOUT = 30
LAUNCHES_CACHE_PATH = "data/spacex_launches_2025_ll2.parquet"
LL2_LIMIT = 50  # tamaño de página razonable
MAX_LAUNCHES = 1000  # safety cap por si acaso


# -------------------- Lógica de extracción de fechas --------------------


def fetch_spacex_launches_from_ll2(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    max_launches: int = MAX_LAUNCHES,
) -> pd.DatetimeIndex:
    """
    Descarga lanzamientos SpaceX en [start, end] desde Launch Library 2
    y devuelve un DatetimeIndex UTC con las fechas 'net'.
    """
    session = requests.Session()
    all_dates: list[pd.Timestamp] = []
    offset = 0

    while True:
        params = {
            # Filtros de fecha
            "window_start__gte": start,
            "window_start__lte": end,
            # Buscar SpaceX en launch_service_provider / texto
            "search": "SpaceX",
            "mode": "detailed",
            "limit": LL2_LIMIT,
            "offset": offset,
        }

        resp = session.get(
            LL2_BASE_URL,
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()

        results = payload.get("results", [])
        if not results:
            break

        for launch in results:
            lsp = (launch.get("launch_service_provider") or {}).get("name", "") or ""
            if "SpaceX" not in lsp:
                continue

            net = launch.get("net")
            if not net:
                continue

            ts = pd.to_datetime(net, utc=True)
            all_dates.append(ts)

        offset += LL2_LIMIT

        # cortar si ya tenemos suficientes o si la API no trae más
        if len(results) < LL2_LIMIT or len(all_dates) >= max_launches:
            break

    if not all_dates:
        return pd.DatetimeIndex([], tz="UTC")

    return pd.DatetimeIndex(all_dates).unique().sort_values()


def get_spacex_launches_for_backtest_corrected(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    use_cache: bool = True,
) -> pd.DatetimeIndex:
    # 1) cache local
    if use_cache and os.path.exists(LAUNCHES_CACHE_PATH):
        launch_dates = pd.read_parquet(LAUNCHES_CACHE_PATH)["launch_date"]
        launch_dates = pd.DatetimeIndex(launch_dates)

        if launch_dates.tz is None:
            launch_dates = launch_dates.tz_localize("UTC")
        else:
            launch_dates = launch_dates.tz_convert("UTC")

        mask = (launch_dates >= pd.to_datetime(start, utc=True)) & (
            launch_dates <= pd.to_datetime(end, utc=True)
        )
        launch_dates = launch_dates[mask]

        print(f"✅ Loaded {len(launch_dates)} launches from cache.")
        return launch_dates

    # 2) llamada a LL2
    try:
        launch_dates = fetch_spacex_launches_from_ll2(start, end)
        print(f"✅ Found {len(launch_dates)} SpaceX launches from LL2.")

        if use_cache and len(launch_dates) > 0:
            pd.DataFrame({"launch_date": launch_dates}).to_parquet(LAUNCHES_CACHE_PATH)
            print(f"💾 Saved to {LAUNCHES_CACHE_PATH}")

        return launch_dates

    except Exception as e:
        print(f"❌ API Error: {e}")
        fallback_dates = pd.DatetimeIndex(
            [
                "2025-09-15 12:00:00+00:00",
                "2025-10-13 12:00:00+00:00",
                "2025-11-05 12:00:00+00:00",
                "2025-11-20 12:00:00+00:00",
            ],
            tz="UTC",
        )
        return fallback_dates


# -------------------- Cálculo de features (si quieres mantenerlo aquí) --------------------


def calculate_spacex_features_vectorized_local(
    target_dates: pd.DatetimeIndex,
    launch_dates: pd.DatetimeIndex,
    future_days: int = 7,
) -> tuple[pd.Series, pd.Series]:
    """
    Versión autocontenida de calculate_spacex_features_vectorized por si quieres testear este script solo.
    Si ya lo tienes en src.processing.feature_eng, puedes borrar esta función.
    """
    if launch_dates.empty:
        return (
            pd.Series(14, index=target_dates),
            pd.Series(0, index=target_dates),
        )

    # Targets
    targets = pd.DataFrame({"date": target_dates}).sort_values("date")
    if targets["date"].dt.tz is None:
        targets["date"] = targets["date"].dt.tz_localize("UTC")

    # Launches
    launches = pd.DataFrame({"launch_date": launch_dates}).sort_values("launch_date")
    if launches["launch_date"].dt.tz is None:
        launches["launch_date"] = launches["launch_date"].dt.tz_localize("UTC")

    # Proximidad (días hasta el siguiente lanzamiento)
    merged = pd.merge_asof(
        targets,
        launches,
        left_on="date",
        right_on="launch_date",
        direction="forward",
    )

    days_to_launch = (merged["launch_date"] - merged["date"]).dt.days
    proximity_series = days_to_launch.fillna(14).clip(upper=14).astype(int)
    proximity_series.index = targets["date"]

    # Conteo de lanzamientos en ventana futura
    target_vals = targets["date"].values
    launch_vals = launches["launch_date"].values

    idx_start = np.searchsorted(launch_vals, target_vals, side="right")
    target_plus_window = target_vals + np.timedelta64(future_days, "D")
    idx_end = np.searchsorted(launch_vals, target_plus_window, side="right")

    count_series = pd.Series(idx_end - idx_start, index=targets["date"])

    return (
        proximity_series.reindex(target_dates),
        count_series.reindex(target_dates),
    )


# -------------------- Script main --------------------


def main():
    print("=== SpaceX Feature Test (Refactored, Launch Library 2) ===")

    spacex_launches = get_spacex_launches_for_backtest_corrected(
        start="2025-01-01",
        end="2025-12-31",
    )

    if spacex_launches.empty:
        print("❌ No launches found to test.")
        return

    print("\n📅 Launches Loaded:")
    for date in spacex_launches:
        print(f"  • {date.strftime('%Y-%m-%d %H:%M UTC')}")

    test_dates = pd.date_range("2025-03-01", "2025-12-31", freq="D", tz="UTC")

    proximity_series, count_series = calculate_spacex_features_vectorized(
        test_dates, spacex_launches
    )

    results_df = pd.DataFrame(
        {
            "date": test_dates.strftime("%Y-%m-%d"),
            "proximity_days": proximity_series.values,
            "future_launches_7d": count_series.values,
        }
    )

    print("\n🔍 Feature Sample (First 15 days):")
    print(results_df.head(15).to_string(index=False))

    print("\n🚀 Critical Days (Proximity < 3 days):")
    critical = results_df[results_df["proximity_days"] < 3]
    if not critical.empty:
        print(critical.to_string(index=False))
    else:
        print("No critical days found in this period.")


if __name__ == "__main__":
    main()
