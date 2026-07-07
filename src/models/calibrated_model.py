import pandas as pd


class CalibratedModel:
    """Wrapper generico attorno a un modello di base: allena il modello, poi
    misura il proprio bias sistematico (errore medio con segno) su una
    porzione recente di dati di training tenuta da parte per la validazione,
    e corregge le previsioni future di quella quantita'.

    Il bias non e' mai un valore fisso o scelto a mano: viene ri-misurato dai
    dati del ristorante ad ogni fit, quindi si adatta automaticamente se il
    comportamento del canale cambia nel tempo.
    """

    VALIDATION_DAYS = 30
    MIN_VALIDATION_DAYS = 5

    def __init__(self, base_factory):
        self._base_factory = base_factory
        self._model = None
        self._bias = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        validation_days = min(self.VALIDATION_DAYS, max(self.MIN_VALIDATION_DAYS, len(X) // 4))

        train_X, val_X = X.iloc[:-validation_days], X.iloc[-validation_days:]
        train_y, val_y = y.iloc[:-validation_days], y.iloc[-validation_days:]

        probe = self._base_factory()
        probe.fit(train_X, train_y)
        val_pred = probe.predict(val_X)
        self._bias = float((val_pred - val_y.values).mean())

        self._model = self._base_factory()
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self._model.predict(X) - self._bias


def create_calibrated_model(base_factory):
    def factory():
        return CalibratedModel(base_factory)

    return factory
