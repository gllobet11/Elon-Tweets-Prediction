import os

from loguru import logger
from playwright.sync_api import TimeoutError, sync_playwright


class AutoIngestor:
    """
    Automatiza la descarga del CSV de datos de tweets desde xTracker.
    """

    URL = "https://xtracker.polymarket.com/user/elonmusk"

    def __init__(self, keywords: list, download_path: str):
        self.keywords = [k.lower() for k in keywords]
        self.download_path = download_path
        os.makedirs(self.download_path, exist_ok=True)

    def _find_and_click_market_button(self, page):
        logger.info("🔎 Buscando estructura de mercado...")

        # Selector de la fila
        container_locator = "div[class*='trackingRow']"
        try:
            page.wait_for_selector(container_locator, state="attached", timeout=15000)
        except TimeoutError:
            logger.error("Timeout: No se encontraron filas de mercado.")
            return False

        market_containers = page.locator(container_locator)
        count = market_containers.count()
        logger.info(f"Filas detectadas: {count}")

        for i in range(count):
            container = market_containers.nth(i)
            container_text = container.inner_text().lower()

            if all(keyword in container_text for keyword in self.keywords):
                logger.success(f"✅ Mercado encontrado en la fila {i}.")

                # --- TÁCTICA 1: ACTIVAR LA FILA (CLICK + HOVER) ---
                logger.info("🖱️  Activando fila (Click + Hover)...")

                container.click(force=True, position={"x": 10, "y": 10})
                page.wait_for_timeout(500)
                container.hover()
                page.wait_for_timeout(1000)

                # --- TÁCTICA 2: BUSCAR POR ICONO SVG ---
                btn_by_icon = container.locator("svg.lucide-download").locator("..")

                candidates = [
                    container.locator(
                        "button[title='Export all tweets in this period']",
                    ),
                    btn_by_icon,
                    container.locator("button", has_text="Tweets"),
                ]

                target_btn = None

                logger.info("🕵️ Buscando botón de descarga...")
                for btn in candidates:
                    if btn.count() > 0 and btn.first.is_visible():
                        target_btn = btn.first
                        logger.info("   -> ¡Botón encontrado y visible!")
                        break

                if not target_btn:
                    logger.warning(
                        "   -> Botón no visible inmediatamente. Esperando renderizado...",
                    )
                    try:
                        btn_by_icon.first.wait_for(state="visible", timeout=3000)
                        target_btn = btn_by_icon.first
                        logger.info("   -> Botón apareció tras espera.")
                    except:
                        logger.error(
                            "❌ El botón no apareció. Dump del HTML de la fila:",
                        )
                        print(container.inner_html())

                if target_btn:
                    logger.info("⬇️ Iniciando descarga...")
                    try:
                        with page.expect_download(timeout=30000) as download_info:
                            target_btn.click(force=True)

                        download = download_info.value
                        destination_path = os.path.join(
                            self.download_path, download.suggested_filename,
                        )
                        download.save_as(destination_path)

                        logger.success(
                            f"📦 ¡Éxito! Archivo guardado: {destination_path}",
                        )
                        return True
                    except Exception as e:
                        logger.error(f"Error en la descarga: {e}")
                        return False

        logger.error("❌ No se encontró el botón en ninguna fila coincidente.")
        return False

    def run(self):
        logger.info(f"🚀 Iniciando navegador: {self.URL}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = context.new_page()

            try:
                page.goto(self.URL, wait_until="networkidle", timeout=60000)
                page.wait_for_timeout(3000)

                if not self._find_and_click_market_button(page):
                    raise RuntimeError("Fallo en la lógica de descarga.")

            except Exception as e:
                logger.error(f"❌ Error en el proceso: {e}")
                page.screenshot(path="debug_final_failure.png")
            finally:
                browser.close()
                logger.info("🚪 Navegador cerrado.")
