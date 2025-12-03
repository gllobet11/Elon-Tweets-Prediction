import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.ingestion.unified_feed import load_unified_data
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error during import: {e}")
    sys.exit(1)

# --- Configuration ---
LOOKBACK_WEEKS = 52
NUM_CHANGES = 3
MIN_WEEKS_SEPARATION = 8  # NUEVO PARAMETRO: Mínimo 2 meses entre cambios
OUTPUT_FILE = os.path.join(project_root, "regime_change_visualization.png")


def find_and_visualize_top_changes():
    print("Loading unified tweet data...")
    df_tweets = load_unified_data()

    if df_tweets.empty:
        return

    df_tweets["created_at"] = pd.to_datetime(df_tweets["created_at"], utc=True)
    weekly_counts = df_tweets.resample("W-MON", on="created_at").size()

    # --- LÓGICA MEJORADA: Ignorar la semana actual ---
    # Asegurarse de que solo se consideren semanas completas para evitar caídas falsas
    if not weekly_counts.empty:
        # Eliminar la última fila si corresponde a la semana actual (incompleta)
        today_utc = pd.to_datetime("today", utc=True)
        if weekly_counts.index[-1] + pd.Timedelta(weeks=1) > today_utc:
            weekly_counts = weekly_counts.iloc[:-1]

    # Diferencia absoluta
    weekly_diffs = weekly_counts.diff()
    recent_abs_diffs = weekly_diffs.abs().dropna().tail(LOOKBACK_WEEKS)

    # --- LOGICA MEJORADA DE FILTRADO ---
    # 1. Ordenar por magnitud (de mayor a menor)
    candidates = recent_abs_diffs.sort_values(ascending=False)

    top_changes = {}
    selected_dates = []

    for date, change_val in candidates.items():
        if len(top_changes) >= NUM_CHANGES:
            break

        # 2. Chequear si la fecha candidata está "lejos" de las ya seleccionadas
        is_far_enough = True
        for sel_date in selected_dates:
            # Calculamos distancia en días
            days_diff = abs((date - sel_date).days)
            if days_diff < (MIN_WEEKS_SEPARATION * 7):
                is_far_enough = False
                break

        # 3. Si está lejos, la guardamos
        if is_far_enough:
            top_changes[date] = weekly_diffs.loc[date]  # Guardamos el valor real (+/-)
            selected_dates.append(date)

    # Convertir a Series para facilitar el ploteo
    top_changes_series = pd.Series(top_changes)

    print(
        f"\nTop {NUM_CHANGES} distinct regime changes (>{MIN_WEEKS_SEPARATION} weeks apart):",
    )
    for date, change in top_changes.items():
        print(f"- Week of {date.date()}: Change of {change:.0f}")

    # --- Visualization (Igual que antes pero usando top_changes_series) ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))
    weekly_counts.tail(LOOKBACK_WEEKS).plot(
        ax=ax, label="Weekly Tweet Count", color="cornflowerblue", lw=2,
    )

    for date, change in top_changes.items():
        color = "green" if change > 0 else "red"  # Verde si sube, rojo si baja
        ax.axvline(
            date,
            color=color,
            linestyle="--",
            lw=2,
            label=f"Shift on {date.date()} ({change:+.0f})",
        )

    ax.set_title(
        f"Top {NUM_CHANGES} Distinct Regime Changes (Min Separation: {MIN_WEEKS_SEPARATION} weeks)",
        fontsize=16,
    )

    # Fix legend duplication
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"\n✅ Visualization saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    find_and_visualize_top_changes()
