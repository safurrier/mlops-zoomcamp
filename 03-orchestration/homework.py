from optparse import OptionConflictError
from re import A
from typing import Tuple
import pandas as pd
from pyparsing import Optional
import pickle
from typing import Optional as OptionalArg

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()

    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")

    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


def subtract_months_from_datetime_get_year_and_month(
    date: datetime, months: int
) -> Tuple[str, str]:
    # subtract a specified amount of
    # months from a datetime string
    # and return the month and year from result
    # date
    past_date = date - relativedelta(months=months)
    month = str(past_date.month)
    if len(month) < 2:
        month = f"0{month}"
    year = str(past_date.year)
    return year, month


@task
def get_paths(date: OptionalArg[str] = None, data_dir: str = "data") -> Tuple[str, str]:
    if not date:
        date = datetime.today().date()
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    # Determine training data
    # 2 months prior to date
    train_year, train_month = subtract_months_from_datetime_get_year_and_month(date, 2)
    train_path = f"fhv_tripdata_{train_year}-{train_month}.parquet"

    val_year, val_month = subtract_months_from_datetime_get_year_and_month(date, 1)
    val_path = f"fhv_tripdata_{val_year}-{val_month}.parquet"

    if data_dir:
        train_path = (Path(data_dir) / Path(train_path)).as_posix()
        val_path = (Path(data_dir) / Path(val_path)).as_posix()
    return train_path, val_path


@flow(task_runner=SequentialTaskRunner())
def main(
    date: OptionalArg[str] = None,
):
    train_path, val_path = get_paths(date).result()

    categorical = ["PUlocationID", "DOlocationID"]

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    # Export artifacts
    with open(f"artifacts/model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)

    with open(f"artifacts/dv-{date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    run_model(df_val_processed, categorical, dv, lr)


# if __name__ == "__main__":
#     main(date="2021-08-15")

# Deployment of flow

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"],
)
