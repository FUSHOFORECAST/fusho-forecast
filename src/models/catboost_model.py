from catboost import CatBoostRegressor


def create_model():

    model = CatBoostRegressor(

        iterations=600,

        learning_rate=0.03,

        depth=6,

        loss_function="MAE",

        eval_metric="MAE",

        random_seed=42,

        verbose=False,

    )

    return model
