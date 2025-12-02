import sys
import os
import pandas as pd

# Añadir el root del proyecto al path para encontrar 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.unified_feed import load_unified_data

def verify_unified_data_ingestion():
    """
    Script de verificación aislado para el `unified_feed`.
    Llama a `load_unified_data` y reporta la estructura del DataFrame resultante.
    """
    print("--- Verificación de Unified Data Feed ---")
    
    try:
        # 1. Ejecutar la función de unificación
        df_unified = load_unified_data()

        # 2. Validar el resultado
        if df_unified is not None and not df_unified.empty:
            print("\n✅ Unificación completada con éxito.")
            print("\n--- Head del DataFrame Unificado ---")
            print(df_unified.head())
            print("\n--- Tail del DataFrame Unificado ---")
            print(df_unified.tail())
            print(f"\nFecha Mínima: {df_unified['created_at'].min()}")
            print(f"Fecha Máxima: {df_unified['created_at'].max()}")
        else:
            print("❌ La función `load_unified_data` retornó un DataFrame vacío o None.")

    except Exception as e:
        print(f"\n❌ Ocurrió un error fatal durante la verificación: {e}")

if __name__ == "__main__":
    verify_unified_data_ingestion()
