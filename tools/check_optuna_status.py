import optuna
import os
import sys

# Path fix
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

# Carga el estudio guardado en la base de datos SQL
storage_url = "sqlite:///optuna_tuning.db"
study_name = "prophet-tuning-study"  # Asegúrate de que coincida con el de tu script

try:
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    print(f"Número de trials completados: {len(study.trials)}")
    print(f"Mejor valor actual (Log Loss): {study.best_value:.4f}")

    # 1. Gráfico de Historia de Optimización
    # Te dice si el modelo sigue aprendiendo o si ya se estancó
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.show()

    # 2. Importancia de Hiperparámetros
    # Te dice qué parámetros importan realmente. Si uno tiene < 0.05 de importancia,
    # puedes dejar de tunearlo y fijarlo en un valor medio.
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.show()

    # 3. Gráfico de Cortes (Slice Plot)
    # Muestra dónde se concentran los mejores puntos.
    fig3 = optuna.visualization.plot_slice(study)
    fig3.show()

except Exception as e:
    print(f"Error cargando el estudio: {e}")
    print("Asegúrate de haber ejecutado 'run_optuna' al menos una vez.")
