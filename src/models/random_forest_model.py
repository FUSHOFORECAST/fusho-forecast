from sklearn.ensemble import RandomForestRegressor


def create_model():

    model = RandomForestRegressor(

        n_estimators=400,

        min_samples_leaf=3,

        random_state=42,

        n_jobs=-1,

    )

    return model
