import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from exercises.linear_regression.data import TARGET, load_frame, split_data

EXPERIMENT = "w01-models"

MODELS: dict[str, list[str]] = {
    "baseline-mean": [],
    "medinc": ["MedInc"],
    "no-location": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"],
    "medinc-location": ["MedInc", "Latitude", "Longitude"],
    "all": ["MedInc", "Latitude", "Longitude", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
}


def fit_predict(
    features: list[str], train: pd.DataFrame, other: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """ Bez cech: zawsze średnia celu z części treningowej. """
    y_train = train[TARGET]
    if not features:
        mean = y_train.mean()
        return np.full(len(train), mean), np.full(len(other), mean)
    model = LinearRegression().fit(train[features], y_train)
    return model.predict(train[features]), model.predict(other[features])


def main() -> None:
    train, val, _test = split_data(load_frame())
    train_ds = mlflow.data.from_pandas(train, name="california-train", targets=TARGET)
    val_ds = mlflow.data.from_pandas(val, name="california-val", targets=TARGET)
    mlflow.set_experiment(EXPERIMENT)
    for run_name, features in MODELS.items():
        with mlflow.start_run(run_name=run_name):
            mlflow.log_input(train_ds, context="training")
            mlflow.log_input(val_ds, context="validation")

            mlflow.log_params(
                {
                    "model": "mean" if not features else "LinearRegression",
                    "features": ",".join(features) or "none",
                    "n_features": len(features),
                }
            )

            train_pred, val_pred = fit_predict(features, train, val)
            train_mse = mean_squared_error(train[TARGET], train_pred)
            val_mse = mean_squared_error(val[TARGET], val_pred)
            mlflow.log_metrics({"train_mse": train_mse, "val_mse": val_mse})

            print(f"{run_name:16s} n={len(features)}  val_mse={val_mse:.4f}")


if __name__ == "__main__":
    main()
#Dla kontroli: baseline-mean ma MSE walidacyjne 1,3734, a all 0,5297