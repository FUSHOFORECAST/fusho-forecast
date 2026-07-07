from src.forecast_candidates.base_candidate import BaseCandidate


class MomentumAdjustedCandidate(BaseCandidate):
    name = "momentum_adjusted"

    def build(self, df):
        base = df["total_rolling_7"]

        momentum_lift = df["business_momentum_7_30"] - 1
        acceleration_lift = df["business_acceleration"]

        volatility_damper = 1 / (1 + df["volatility_ratio_14_60"].clip(lower=0))

        adjustment = (
            momentum_lift * 0.70
            + acceleration_lift * 0.30
        ) * volatility_damper

        adjustment = adjustment.clip(lower=-0.18, upper=0.18)

        return base * (1 + adjustment)
