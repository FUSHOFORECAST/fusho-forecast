import pandas as pd


class SeasonalNaiveMedianModel:
    """Baseline robusto: predice usando la colonna di rolling-median/mean del
    canale stesso (gia' calcolata come feature) che ha performato meglio sugli
    ultimi giorni di training. Utile per canali rumorosi (es. cash) dove un
    modello ad albero con molte feature tende a overfittare rumore, mentre un
    aggregato robusto della storia recente del canale generalizza meglio.
    """

    VALIDATION_DAYS = 30
    MIN_VALIDATION_DAYS = 5

    def __init__(self):
        self._best_column: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        channel = y.name

        candidate_cols = [c for c in X.columns if c.startswith(f"{channel}_rolling_median_")]
        candidate_cols += [c for c in X.columns if c.startswith(f"{channel}_rolling_") and "median" not in c]

        if not candidate_cols:
            raise ValueError(
                f"Nessuna feature di rolling trovata per il canale '{channel}': "
                f"SeasonalNaiveMedianModel richiede colonne '{channel}_rolling_*'."
            )

        validation_days = min(self.VALIDATION_DAYS, max(self.MIN_VALIDATION_DAYS, len(X) // 4))
        val_X = X.iloc[-validation_days:]
        val_y = y.iloc[-validation_days:]

        best_col, best_mae = None, None
        for col in candidate_cols:
            mae = (val_X[col] - val_y).abs().mean()
            if pd.isna(mae):
                continue
            if best_mae is None or mae < best_mae:
                best_mae, best_col = mae, col

        self._best_column = best_col or candidate_cols[0]
        return self

    def predict(self, X: pd.DataFrame):
        return X[self._best_column].values


def create_model():
    return SeasonalNaiveMedianModel()
