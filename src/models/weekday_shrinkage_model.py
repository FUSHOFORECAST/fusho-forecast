import pandas as pd


class WeekdayShrinkageModel:
    """Baseline robusto (mediana/media mobile) corretto con un effetto
    giorno-della-settimana attenuato (shrinkage), invece che sostituito del
    tutto: un canale rumoroso come il cash ha un pattern settimanale reale
    ma una stima per-giorno pura e' troppo rumorosa da sola (pochi campioni).

    Sia la colonna di base sia l'intensita' della correzione (shrinkage) sono
    scelte automaticamente su una porzione recente di dati di training tenuta
    da parte per la validazione -- generico per qualsiasi canale/ristorante,
    niente impostato a mano.
    """

    VALIDATION_DAYS = 30
    MIN_VALIDATION_DAYS = 5
    SHRINKAGE_GRID = [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]

    def __init__(self):
        self._base_column: str | None = None
        self._weekday_factor: dict[int, float] = {}
        self._shrinkage: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        channel = y.name

        base_candidates = [c for c in X.columns if c.startswith(f"{channel}_rolling_median_")]
        base_candidates += [c for c in X.columns if c.startswith(f"{channel}_rolling_") and "median" not in c]
        if not base_candidates:
            raise ValueError(f"Nessuna feature di rolling trovata per il canale '{channel}'.")

        if "dayofweek" not in X.columns:
            raise ValueError("WeekdayShrinkageModel richiede la colonna 'dayofweek' tra le feature.")

        validation_days = min(self.VALIDATION_DAYS, max(self.MIN_VALIDATION_DAYS, len(X) // 4))
        train_X, val_X = X.iloc[:-validation_days], X.iloc[-validation_days:]
        train_y, val_y = y.iloc[:-validation_days], y.iloc[-validation_days:]

        # Il fattore per giorno-settimana e' imparato solo dalla porzione di
        # training, non da quella di validazione usata per scegliere lo shrinkage.
        weekday_factor = self._compute_weekday_factor(train_X, train_y)

        best = None
        for base_col in base_candidates:
            for shrinkage in self.SHRINKAGE_GRID:
                preds = self._predict_with(val_X, base_col, weekday_factor, shrinkage)
                errors = (preds - val_y.values)
                mae = pd.Series(errors).abs().mean()
                if pd.isna(mae):
                    continue
                if best is None or mae < best[0]:
                    best = (mae, base_col, shrinkage)

        if best is None:
            base_col, shrinkage = base_candidates[0], 0.0
        else:
            _, base_col, shrinkage = best

        # Rifit del fattore giorno-settimana su tutti i dati (train+val) per il modello finale.
        self._weekday_factor = self._compute_weekday_factor(X, y)
        self._base_column = base_col
        self._shrinkage = shrinkage
        return self

    @staticmethod
    def _compute_weekday_factor(X: pd.DataFrame, y: pd.Series) -> dict[int, float]:
        overall_mean = y.mean()
        if not overall_mean:
            return {}
        by_weekday = y.groupby(X["dayofweek"].values).mean()
        return (by_weekday / overall_mean).to_dict()

    @staticmethod
    def _predict_with(X: pd.DataFrame, base_col: str, weekday_factor: dict[int, float], shrinkage: float):
        base = X[base_col].values
        factors = X["dayofweek"].map(weekday_factor).fillna(1.0).values
        adjusted_factor = 1 + shrinkage * (factors - 1)
        return base * adjusted_factor

    def predict(self, X: pd.DataFrame):
        return self._predict_with(X, self._base_column, self._weekday_factor, self._shrinkage)


def create_model():
    return WeekdayShrinkageModel()
