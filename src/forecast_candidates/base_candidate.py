class BaseCandidate:
    name = None

    def build(self, df, target="total"):
        raise NotImplementedError("Each candidate must implement build(df, target)")
