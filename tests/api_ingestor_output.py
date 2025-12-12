"""
test_api_ingestor_manual.py

Script de auditoría EXACTA contra el mercado activo.
1. Obtiene las fechas oficiales (metadata) de xTracker.
2. Descarga los tweets filtrando por ese rango exacto (al milisegundo).
3. Compara el resultado con lo que ves en la web de Polymarket.
"""

import os
import sys
import pandas as pd
from loguru import logger

# --- Configuración de Importación ---
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.ingestion.api_ingestor import ApiIngestor
except ImportError:
    logger.error("❌ No se pudo importar ApiIngestor. Verifica la ruta 'src/ingestion/api_ingestor.py'")
    sys.exit(1)

OUTPUT_FILE = "audit_market_exact.csv"
from config.settings import MARKET_KEYWORDS # Importa tus keywords reales

def run_market_audit():
    logger.info("🕵️ INICIANDO AUDITORÍA DE MERCADO (POR KEYWORDS)")
    logger.info(f"🔑 Keywords de settings.py: {MARKET_KEYWORDS}")
    
    ingestor = ApiIngestor()

    # PASAMOS LAS KEYWORDS EXPLICITAMENTE
    official_start, official_end, official_title = ingestor.get_official_market_dates(MARKET_KEYWORDS)

    if not official_start:
        logger.error("❌ No se encontró ningún mercado con esas palabras clave.")
        return

    logger.success(f"✅ Mercado Encontrado: '{official_title}'")
    try:
    # Descargar
        df = ingestor.fetch(official_start, official_end)
    except Exception as e:
        logger.error(f"❌ Error en fetch: {e}")
        return

    # 4. Resultados
    print("\n" + "="*60)
    print("📊 RESULTADO DE LA AUDITORÍA")
    print("="*60)

    count = len(df)
    print(f"🎯 RECUENTO EXACTO: {count}")
    print(f"   (Debe coincidir con el número grande en Polymarket)")
    
    if not df.empty:
        print("-" * 60)
        print(f"📅 Primer Tweet Contado (UTC): {df.iloc[0]['created_at']} | ID: {df.iloc[0]['id']}")
        print(f"📅 Último Tweet Contado (UTC):  {df.iloc[-1]['created_at']} | ID: {df.iloc[-1]['id']}")
        print("-" * 60)
        
        # Validación de límites
        min_date = df['created_at'].min()
        max_date = df['created_at'].max()
        
        if min_date < official_start:
            logger.error(f"❌ ALERTA: Hay tweets ANTERIORES al inicio oficial! {min_date} < {official_start}")
        elif max_date > official_end:
            logger.error(f"❌ ALERTA: Hay tweets POSTERIORES al final oficial! {max_date} > {official_end}")
        else:
            logger.success("✅ Todos los tweets están DENTRO de la ventana oficial.")

    # 5. Guardar CSV
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"💾 CSV de auditoría guardado: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_market_audit()