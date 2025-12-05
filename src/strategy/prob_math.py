from loguru import logger
from scipy import stats


class DistributionConverter:
    """
    Convierte una predicción (mu) en probabilidades, ajustando dinámicamente
    la incertidumbre según cuánto de la semana ya ha transcurrido.
    """

    @staticmethod
    def get_bin_probabilities(
        mu_remainder,
        current_actuals,
        model_type="nbinom",
        alpha=0.2,
        bins_config=None,
    ):
        logger.info("DEBUG prob_math.py: Inside get_bin_probabilities:")
        logger.info(f"  mu_remainder={mu_remainder}, type={type(mu_remainder)}")
        logger.info(
            f"  current_actuals={current_actuals}, type={type(current_actuals)}",
        )
        logger.info(f"  model_type={model_type}, type={type(model_type)}")
        logger.info(f"  alpha={alpha}, type={type(alpha)}")
        logger.info(f"  bins_config={bins_config}, type={type(bins_config)}")
        """
        Calcula probabilidades sobre el remanente y las proyecta a los bins totales.
        
        Args:
            mu_remainder (float): Predicción de tweets SOLO para los días/horas faltantes.
            current_actuals (int): Tweets ya confirmados (Ground Truth).
            model_type (str): 'nbinom' o 'poisson'.
            alpha (float): Parámetro de dispersión (0.2).
            bins_config (list): Lista de tuplas [(Label, Low, High), ...]
            
        Returns:
            dict: { 'BinLabel': probabilidad_calculada }
        """
        if bins_config is None:
            raise ValueError("Se requiere configuración de bins")

        probabilities = {}

        # Validación de seguridad: mu_remainder no puede ser negativo ni 0 absoluto para NBinom
        mu_rem = max(0.01, mu_remainder)

        # Configuración Distribución sobre el REMANENTE
        if model_type == "nbinom":
            alpha = max(alpha, 1e-5)
            # Parámetros aplicados solo a la parte incierta
            # Var = mu + alpha * mu^2
            n_param = 1.0 / alpha
            p_param = 1.0 / (1.0 + alpha * mu_rem)

        for bin_label, bin_low_total, bin_high_total in bins_config:
            # --- LÓGICA DE DESPLAZAMIENTO (SHIFT) ---
            # Restamos lo que ya llevamos (actuals) a los límites del bin

            # 1. Ajustar límites al remanente
            low_rem = bin_low_total - current_actuals
            high_rem = bin_high_total - current_actuals

            # Manejo de límite superior infinito (el último bin suele ser "340+")
            if bin_high_total == float("inf") or str(bin_high_total) == "inf":
                high_rem = 100000

            # 2. Caso Imposible (El bin ya quedó atrás)
            # Ej: Bin "200-219", Actuals 237 -> high_rem es negativo (-18).
            # Probabilidad es 0 porque ya nos pasamos.
            if high_rem < 0:
                probabilities[bin_label] = 0.0
                continue

            # 3. Ajuste de límite inferior negativo
            # Ej: Bin "220-239", Actuals 237 -> low_rem es negativo (-17).
            # Significa que "ya estamos dentro o por encima del suelo del bin".
            # Para el remanente, contamos desde 0.
            low_rem = max(0, low_rem)

            # 4. Cálculo de Probabilidad (CDF del remanente)
            if model_type == "poisson":
                prob = stats.poisson.cdf(high_rem, mu_rem) - stats.poisson.cdf(
                    low_rem - 1,
                    mu_rem,
                )
            else:  # NBinom
                prob = stats.nbinom.cdf(high_rem, n_param, p_param) - stats.nbinom.cdf(
                    low_rem - 1,
                    n_param,
                    p_param,
                )

            probabilities[bin_label] = prob

        return probabilities

    @staticmethod
    def calculate_kelly_bet(
        my_prob,
        market_price,
        bankroll,
        kelly_fraction=0.2,
        max_position=0.2,
    ):
        """
        Calcula el tamaño de la apuesta ($) según Kelly.
        """
        if market_price <= 0.001 or market_price >= 0.999:
            return 0.0

        edge = my_prob - market_price
        if edge <= 0:
            return 0.0

        b = (1.0 / market_price) - 1.0
        f_star = (my_prob * (b + 1) - 1) / b
        f_safe = f_star * kelly_fraction
        f_final = min(f_safe, max_position)

        return bankroll * f_final
