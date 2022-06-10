# %%
# [markdown] ##
import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
import pickle

# %%
# [markdown] ## Read Data
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


df_train = read_dataframe("./data/green_tripdata_2021-01.parquet")
df_val = read_dataframe("./data/green_tripdata_2021-02.parquet")


# %%
# [markdown] ## Data preprocessing
df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]
categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
numerical = ["trip_distance"]

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)

target = "duration"
y_train = df_train[target].values
y_val = df_val[target].values
# %%
# [markdown] ##


# Q1. Install MLflow. version?
print(mlflow.__version__)


# %%
# [markdown] ## Set Tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# %%
# [markdown] ## Set New experiment
mlflow.set_experiment("hw-2-experiment")


# %%
# [markdown] ## Q3 - How many params auto logged by MLFlow?
# How many parameters are automatically logged by MLflow?

# This is from video 2.1
with mlflow.start_run():

    mlflow.set_tag("mad-scientist", "Alex")

    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.parquet")

    # We are intersted in alpha as the main
    # param to tune

    alpha = 0.01
    mlflow.log_param("alpha", alpha)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)

    mlflow.log_metric("rmse", rmse)

    # Save model locally
    with open("models/lin_reg.bin", "wb") as f_out:
        pickle.dump((dv, lr), f_out)

    # Log the model artifact from local
    mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")

# %%
# [markdown] ## Saving artifacts on MLFlow backend
with mlflow.start_run():

    mlflow.set_tag("mad-scientist", "Alex")

    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.parquet")

    # We are intersted in alpha as the main
    # param to tune

    alpha = 0.01
    mlflow.log_param("alpha", alpha)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)

    mlflow.log_metric("rmse", rmse)

    # Log model to backend
    # using sklearn
    # will give you many artifacts + info for free
    mlflow.sklearn.log_model(lr, "models_mflow")

    # Log the preprocessor also
    mlflow.log_artifact("output/dv.pkl", artifact_path="preprocessor")


# %%
# [markdown] ## Load model from backend run artifact

model_uri = "runs:/e6187cf829524d9880ef57b80166fefe/models_mflow"
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_uri)
loaded_model


# %%
# [markdown] ## Load model into sklearn
sklearn_model = mlflow.sklearn.load_model(model_uri)
sklearn_model

# %%
# [markdown] ## Use loaded model to make predictions
y_pred = sklearn_model.predict(X_val)
y_pred[:10]


# %%
# [markdown] ## Q5 Hyper param tuning on RandomForest Regressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {"loss": rmse, "status": STATUS_OK}


search_space = {
    "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "objective": "reg:linear",
    "seed": 42,
}

best_result = fmin(
    fn=objective, space=search_space, algo=tpe.suggest, max_evals=50, trials=Trials()
)

# %%
# [markdown] ## Promote the best model by RMSE
# First connect to the client
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# %% Search for the best run by RMSE
from mlflow.entities import ViewType

EXPERIMENT_ID = '5'
runs = client.search_runs(
    experiment_ids=EXPERIMENT_ID,    # Experiment ID we want
    filter_string="metrics.rmse < 7",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.rmse ASC"]
)

for run in runs:
    print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']:.4f}")
best_run_id = runs[0].info.run_id
best_run_id

# %%
# [markdown] ## Select the best model and add to registry
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_uri = f"runs:/{best_run_id}/models"
mlflow.register_model(model_uri=model_uri, name="test-best-rf-model")

# %%
# [markdown] ## Get model info and versions
model_name = "test-best-rf-model"
latest_versions = client.get_latest_versions(name=model_name)

for version in latest_versions:
    print(f"version: {version.version}, stage: {version.current_stage}")


# %%
# [markdown] ## Promote it to staging
model_version = 1
new_stage = 'Staging'

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)

# %% Update model description
from datetime import datetime
date = datetime.today().date()
client.update_model_version(
    name=model_name,
    version=model_version,
    description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
)

# %%
