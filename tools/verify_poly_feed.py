import sys
import os
from pprint import pprint

# Añadir el root del proyecto al path para encontrar 'src' y 'config'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.poly_feed import PolymarketFeed
from config.bins_definition import MARKET_BINS

def verify_poly_feed():
    """
    Script de verificación para PolymarketFeed.
    Lista todos los mercados activos de Elon Musk para encontrar el correcto.
    """
    print("--- Verificación de PolymarketFeed (Listado de Mercados) ---")

    try:
        # 1. Instanciar el feed
        poly_feed = PolymarketFeed()
        if not poly_feed.valid:
            print("❌ No se pudo inicializar ClobClient.")
            return

        # 2. Obtener todos los mercados
        print("\n🔎 Obteniendo la lista de mercados activos de Polymarket...")
        markets_resp = poly_feed._robust_api_call(poly_feed.client.get_markets, next_cursor="")
        
        if not markets_resp or not markets_resp.get('data'):
            print("❌ No se pudo obtener la lista de mercados de la API.")
            return

        # 3. Filtrar y mostrar los mercados relevantes
        print("\n--- Mercados de Elon Musk Encontrados ---")
        count = 0
        for market in markets_resp['data']:
            question = market.get('question', '')
            if 'elon' in question.lower() and 'tweet' in question.lower():
                print(f"  ----------------------------------------")
                print(f"  📌 Pregunta: {market.get('question')}")
                print(f"     Slug: {market.get('slug')}")
                print(f"     ID: {market.get('id')}")
                print(f"     Condition ID: {market.get('condition_id')}")
                count += 1
        
        if count == 0:
            print("  [❌] No se encontró ningún mercado activo que contenga 'elon' y 'tweet'.")
        
        print(f"\nResumen: Se encontraron {count} mercados relevantes.")

    except Exception as e:
        print(f"\n❌ Ocurrió un error fatal durante la verificación: {e}")

if __name__ == "__main__":
    verify_poly_feed()
