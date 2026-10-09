import mlflow
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from exercises.linear_regression.data import load_frame, split_data

THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.30]

train, val, test = split_data(load_frame()) #ten sam split co i w data.py
corr = train.corr() #Matrica korelacji, wyłącznie na części treningowej

r_target = corr["MedHouseVal"].drop("MedHouseVal")#drop the target itself, r = 1
print(r_target)

train_ds = mlflow.data.from_pandas(train, name="california-train", targets="MedHouseVal")
val_ds = mlflow.data.from_pandas(val, name="california-val", targets="MedHouseVal")
mlflow.set_experiment("w01-correlation")

with mlflow.start_run(run_name="correlation-selection"):
    mlflow.log_dict(r_target.to_dict(), "correlation_with_target.json")
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    names = list(corr.columns)
    ax.set_xticks(range(len(names)), names, rotation=45)
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, round(corr.iloc[i, j], 2), ha="center", va="center")
    fig.colorbar(image)
    fig.tight_layout()
    mlflow.log_figure(fig, "correlation_train.png")
    plt.close(fig)

    for t in THRESHOLDS:
        features = []
        for name in r_target.index:
            if abs(r_target[name]) >= t:
                features.append(name)

        with mlflow.start_run(run_name=f"t={t}", nested=True):
            mlflow.log_input(train_ds, context="training")
            mlflow.log_input(val_ds, context="validation")
            model = LinearRegression()
            model.fit(train[features], train["MedHouseVal"])
            val_pred = model.predict(val[features])
            val_mse = mean_squared_error(val["MedHouseVal"], val_pred)
            mlflow.log_param("threshold", t)
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("features", ",".join(features))
            mlflow.log_metric("val_mse", val_mse)

            print(t, len(features), round(val_mse, 4), features)
#Dla kontroli: przy t = 0 wybierasz wszystkie 8 cech i dostajesz 0,5297, jak w Z2(models.py).