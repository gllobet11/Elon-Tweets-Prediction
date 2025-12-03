import os
import sys
from pprint import pprint

# Añadir el root del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import sys

# Añadir el root del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import sys

# Añadir el root del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import sys

from config.bins_definition import MARKET_BINS
from src.ingestion.poly_feed import PolymarketFeed

# Añadir el root del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



def test_final_price_logic():
    """
    Script de prueba final para verificar la lógica de obtención de precios.
    """
    print("--- Prueba Final de Lógica de Precios ---")

    MARKET_KEYWORDS = ["elon musk", "tweets", "november 25", "december 2"]

    try:
        poly_feed = PolymarketFeed()
        if not poly_feed.valid:
            print("❌ No se pudo inicializar ClobClient.")
            return

        # 1. Mapear IDs
        print("\n🔎 Mapeando IDs de tokens 'Yes' y 'No' para cada bin...")
        updated_bins = poly_feed.fetch_market_ids_automatically(
            keywords=MARKET_KEYWORDS, bins_dict=MARKET_BINS,
        )

        # 2. Obtener precios
        print("\n💰 Obteniendo precios con la lógica de valuación final...")
        price_snapshot = poly_feed.get_all_bins_prices(updated_bins)

        # 3. Imprimir resultados
        print("\n--- Snapshot de Precios Final Obtenido ---")
        pprint(price_snapshot)

    except Exception as e:
        print(f"\n❌ Ocurrió un error fatal durante la prueba: {e}")


if __name__ == "__main__":
    test_final_price_logic()
