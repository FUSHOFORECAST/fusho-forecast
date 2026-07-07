class BaseEngine:
    def __init__(self, name):
        self.name = name

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def score(self, *args, **kwargs):
        raise NotImplementedError
