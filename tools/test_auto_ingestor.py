import os
import sys

# --- Path Configuration ---
try:
    # Asumimos que este script está en la carpeta 'tools', así que subimos un nivel para llegar a la raíz
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from loguru import logger

    from src.ingestion.auto_ingestor import AutoIngestor

except (ImportError, ModuleNotFoundError) as e:
    print(
        "Error de importación. Asegúrate de que la estructura de carpetas es correcta.",
    )
    print(f"Error: {e}")
    sys.exit(1)


def run_ingestor_test():
    """
    Ejecuta una prueba dedicada para el AutoIngestor para verificar
    que puede encontrar y descargar el archivo del mercado correcto.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("🧪 Iniciando prueba del AutoIngestor...")

    # --- 1. Configuración del Test ---
    # Directorio de descarga
    RAW_DATA_DIR = os.path.join(project_root, "data", "raw")

    # Palabras clave para el mercado que queremos descargar
    # Esto debe coincidir con el mercado que esperas encontrar en la página
    TARGET_KEYWORDS = ["elon musk", "tweets", "november 25", "december 2"]

    logger.info(f"Buscando mercado con las keywords: {TARGET_KEYWORDS}")

    # --- 2. Ejecución ---
    try:
        ingestor = AutoIngestor(keywords=TARGET_KEYWORDS, download_path=RAW_DATA_DIR)
        ingestor.run()
        logger.info("✅ Prueba del AutoIngestor completada.")
    except Exception as e:
        logger.error("❌ La prueba del AutoIngestor falló con una excepción.")
        logger.exception(e)


if __name__ == "__main__":
    run_ingestor_test()
