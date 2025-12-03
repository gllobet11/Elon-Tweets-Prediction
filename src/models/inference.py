import pandas as pd
import statsmodels.api as sm
from loguru import logger
from statsmodels.genmod.families import NegativeBinomial


class InferenceModel:
    def __init__(self):
        """
        Inicializa un contenedor para un modelo GLM que se entrenará on-the-fly.
        """
        self.model_res = None
        self.alpha = None
        self.feature_cols = None

    def _estimate_alpha(self, y_train: pd.Series) -> float:
        """Estima el parámetro alpha para el modelo Negative Binomial."""
        mu = y_train.mean()
        var = y_train.var()
        alpha = (var - mu) / (mu**2) if mu > 0 else 0.0
        return max(alpha, 1e-6)  # Evitar alpha cero o negativo

    def train(self, train_df: pd.DataFrame, feature_cols: list, verbose: bool = True):
        """
        Entrena el modelo GLM NegativeBinomial con los datos proporcionados.
        """
        if verbose:
            logger.info("🧠 Entrenando modelo NB on-the-fly...")
        self.feature_cols = feature_cols

        y_train = train_df["n_tweets"].astype(float)
        X_train = train_df[self.feature_cols].astype(float)
        X_train_const = sm.add_constant(X_train, has_constant="add")

        self.alpha = self._estimate_alpha(y_train)
        model = sm.GLM(
            y_train, X_train_const, family=NegativeBinomial(alpha=self.alpha),
        )
        self.model_res = model.fit()

        if verbose:
            logger.info(f"✅ Modelo entrenado. Alpha (dispersión): {self.alpha:.4f}")
            logger.info(f"📊 Coeficientes del nuevo modelo: \n{self.model_res.params}")

    def predict(self, features: pd.DataFrame):
        """
        Realiza la inferencia para la semana entrante usando el modelo entrenado.
        """
        if self.model_res is None:
            raise RuntimeError(
                "❌ El modelo no ha sido entrenado. Llama al método .train() primero.",
            )

        input_data = features.copy()
        input_data["const"] = 1.0

        # Asegurarse de que las columnas están en el orden correcto
        model_cols = self.model_res.params.index.tolist()
        input_data = input_data[model_cols]

        predicted_mu = self.model_res.predict(input_data).iloc[0]

        return predicted_mu, self.alpha
