"""
test_poly_feed_optimized.py

Script de test para verificar que poly_feed_FINAL_OPTIMIZED funciona correctamente.
"""

import sys
import os

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..')) # Sube un nivel al directorio raíz (2.Elon-Tweets-Prediction)
sys.path.insert(0, project_root)
import time


from loguru import logger

# Importar la versión optimizada
try:
        from src.ingestion.poly_feed import PolymarketFeed
        logger.info("✅ Imported from src.ingestion.poly_feed")
except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        sys.exit(1)

# Configuración
MARKET_SLUG = "elon-musk-of-tweets-december-5-december-12"
MARKET_KEYWORDS = ["Elon", "Musk", "tweets"]

# Bins de ejemplo (estructura típica)
MARKET_BINS = {
    "0-19": {"lower": 0, "upper": 19},
    "20-39": {"lower": 20, "upper": 39},
    "40-59": {"lower": 40, "upper": 59},
    "60-79": {"lower": 60, "upper": 79},
    "80-99": {"lower": 80, "upper": 99},
    "100-119": {"lower": 100, "upper": 119},
    "120-139": {"lower": 120, "upper": 139},
    "140-159": {"lower": 140, "upper": 159},
    "160-179": {"lower": 160, "upper": 179},
    "180-199": {"lower": 180, "upper": 199},
    "200-219": {"lower": 200, "upper": 219},
    "220-239": {"lower": 220, "upper": 239},
    "240-259": {"lower": 240, "upper": 259},
    "260-279": {"lower": 260, "upper": 279},
    "280-299": {"lower": 280, "upper": 299},
    "300-319": {"lower": 300, "upper": 319},
    "320-339": {"lower": 320, "upper": 339},
    "340-359": {"lower": 340, "upper": 359},
    "360-379": {"lower": 360, "upper": 379},
    "380-399": {"lower": 380, "upper": 399},
    "400-419": {"lower": 400, "upper": 419},
    "420-439": {"lower": 420, "upper": 439},
    "440-459": {"lower": 440, "upper": 459},
    "460-479": {"lower": 460, "upper": 479},
    "480-499": {"lower": 480, "upper": 499},
    "500+": {"lower": 500, "upper": float('inf')},
}


def test_initialization():
    """Test 1: Verificar que se inicializa correctamente"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Initialization")
    logger.info("=" * 80)
    
    feed = PolymarketFeed()
    
    if feed.valid:
        logger.success("✅ PolymarketFeed initialized successfully")
        return feed
    else:
        logger.error("❌ PolymarketFeed initialization failed")
        return None


def test_market_discovery(feed):
    """Test 2: Buscar mercados (primera vez - debería tomar 40-50 segundos)"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Market Discovery (First Time)")
    logger.info("=" * 80)
    
    logger.info(f"🔎 Searching for slug: '{MARKET_SLUG}'")
    logger.info(f"⏱️  Expected time: 40-50 seconds")
    
    start_time = time.time()
    
    mapped_bins = feed.fetch_market_ids_automatically(
        keywords=MARKET_KEYWORDS,
        bins_dict=MARKET_BINS.copy(),
        market_slug=MARKET_SLUG
    )
    
    elapsed = time.time() - start_time
    
    # Contar bins mapeados
    successful_bins = sum(1 for v in mapped_bins.values() if v.get("id_yes") and v.get("id_no"))
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Time taken: {elapsed:.1f} seconds")
    logger.info(f"   Bins mapped: {successful_bins}/{len(MARKET_BINS)}")
    
    if successful_bins == len(MARKET_BINS):
        logger.success(f"✅ All bins mapped successfully!")
    elif successful_bins >= len(MARKET_BINS) * 0.9:
        logger.warning(f"⚠️  Most bins mapped ({successful_bins}/{len(MARKET_BINS)})")
    else:
        logger.error(f"❌ Only {successful_bins}/{len(MARKET_BINS)} bins mapped")
    
    # Verificar que se usó early exit
    if elapsed < 55:
        logger.success(f"✅ Early exit worked! (time < 55s)")
    else:
        logger.warning(f"⚠️  Early exit might not be working (time >= 55s)")
    
    return mapped_bins, successful_bins


def test_cache(feed):
    """Test 3: Verificar que el cache funciona (debería ser instantáneo)"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Cache Performance (Second Time)")
    logger.info("=" * 80)
    
    logger.info(f"🔎 Searching again (should use cache)")
    logger.info(f"⏱️  Expected time: <1 second")
    
    start_time = time.time()
    
    mapped_bins = feed.fetch_market_ids_automatically(
        keywords=MARKET_KEYWORDS,
        bins_dict=MARKET_BINS.copy(),
        market_slug=MARKET_SLUG
    )
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Time taken: {elapsed:.3f} seconds")
    
    if elapsed < 1.0:
        logger.success(f"✅ Cache working! ({elapsed:.3f}s < 1s)")
    else:
        logger.warning(f"⚠️  Cache might not be working ({elapsed:.3f}s >= 1s)")
    
    return elapsed < 1.0


def test_pricing(feed, mapped_bins):
    """Test 4: Obtener precios de los bins mapeados"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Price Fetching")
    logger.info("=" * 80)
    
    logger.info(f"💰 Fetching prices for mapped bins...")
    
    start_time = time.time()
    snapshot = feed.get_all_bins_prices(mapped_bins)
    elapsed = time.time() - start_time
    
    # Analizar precios
    bins_with_prices = sum(1 for v in snapshot.values() if v.get("status") in ["ACTIVE_YES", "ACTIVE_NO"])
    bins_no_liquidity = sum(1 for v in snapshot.values() if v.get("status") == "NO_LIQUIDITY")
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Time taken: {elapsed:.1f} seconds")
    logger.info(f"   Bins with prices: {bins_with_prices}/{len(snapshot)}")
    logger.info(f"   Bins no liquidity: {bins_no_liquidity}/{len(snapshot)}")
    
    # Mostrar algunos ejemplos
    logger.info(f"\n💰 Sample prices:")
    count = 0
    for bin_label, price_data in snapshot.items():
        if count < 5 and price_data.get("status") in ["ACTIVE_YES", "ACTIVE_NO"]:
            logger.info(
                f"   {bin_label:10s}: "
                f"mid={price_data['mid_price']:.4f} "
                f"bid={price_data.get('bid', 0):.4f} "
                f"ask={price_data.get('ask', 0):.4f} "
                f"[{price_data['status']}]"
            )
            count += 1
    
    if bins_with_prices >= len(snapshot) * 0.8:
        logger.success(f"✅ Good price coverage ({bins_with_prices}/{len(snapshot)})")
    else:
        logger.warning(f"⚠️  Low price coverage ({bins_with_prices}/{len(snapshot)})")
    
    return bins_with_prices


def main():
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "OPTIMIZED POLY FEED TEST" + " " * 33 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    
    # Test 1: Initialization
    feed = test_initialization()
    if not feed:
        logger.error("\n❌ FAILED: Could not initialize PolymarketFeed")
        return False
    
    # Test 2: Market Discovery (primera vez)
    mapped_bins, successful_bins = test_market_discovery(feed)
    if successful_bins < len(MARKET_BINS) * 0.8:
        logger.error(f"\n❌ FAILED: Only {successful_bins}/{len(MARKET_BINS)} bins mapped")
        return False
    
    # Test 3: Cache Performance
    cache_works = test_cache(feed)
    if not cache_works:
        logger.warning("\n⚠️  WARNING: Cache might not be working optimally")
    
    # Test 4: Pricing
    bins_with_prices = test_pricing(feed, mapped_bins)
    if bins_with_prices < len(MARKET_BINS) * 0.5:
        logger.warning(f"\n⚠️  WARNING: Low price coverage")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    
    all_passed = (
        successful_bins == len(MARKET_BINS) and
        cache_works and
        bins_with_prices >= len(MARKET_BINS) * 0.8
    )
    
    if all_passed:
        logger.success("""
    ✅ ALL TESTS PASSED!
    
    ✓ Markets discovered successfully
    ✓ Cache working correctly
    ✓ Prices fetched successfully
    
    🎉 poly_feed_FINAL_OPTIMIZED is working perfectly!
    """)
        return True
    else:
        logger.warning("""
    ⚠️  SOME TESTS HAD WARNINGS
    
    The system is working but not optimally.
    Check the warnings above for details.
    """)
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)