import numpy as np

from src.forecast_candidates.base_candidate import BaseCandidate


class MemoryShortCandidate(BaseCandidate):
    name = "memory_short"

    def build(self, df, target="total"):
        return df[target].shift(1).rolling(7).mean()


class MemoryMediumCandidate(BaseCandidate):
    name = "memory_medium"

    def build(self, df, target="total"):
        return df[target].shift(1).rolling(30).mean()


class MemoryLongCandidate(BaseCandidate):
    name = "memory_long"

    def build(self, df, target="total"):
        return df[target].shift(1).rolling(90).mean()


class AdaptiveMemoryCandidate(BaseCandidate):
    name = "adaptive_memory"

    def build(self, df, target="total"):
        short_col = "memory_short"
        medium_col = "memory_medium"
        long_col = "memory_long"

        values = []

        for _, row in df.iterrows():
            momentum = row.get("business_momentum_30_90", np.nan)

            if np.isnan(momentum):
                values.append(np.nan)
            elif momentum > 1.05:
                values.append(row.get(short_col, np.nan))
            elif momentum < 0.98:
                values.append(row.get(long_col, np.nan))
            else:
                values.append(row.get(medium_col, np.nan))

        return values
