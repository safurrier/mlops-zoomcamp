# %%
# [markdown] ## Imports
import pandas as pd
import sklearn

import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error


# %%
# [markdown] ## Read Data
df_train = pd.read_parquet("./data/fhv_tripdata_2021-01.parquet")
df_val = pd.read_parquet("./data/fhv_tripdata_2021-02.parquet")

# %% Q1 Number of records
df_train.shape[0]

# %%
# [markdown] ## Q2 Avg duration
df_train["duration"] = df_train.dropOff_datetime - df_train.pickup_datetime
df_train["duration"] = df_train["duration"].apply(lambda td: td.total_seconds() / 60)
df_train["duration"].mean()


# %%
# [markdown] ## Remove outliers
records_before = df_train.shape[0]
df_train = df_train[(df_train.duration >= 1) & (df_train.duration <= 60)]
records_after = df_train.shape[0]
records_removed = records_before - records_after
records_removed


# %%
# [markdown] ## % missing Pickup locationvalues
round(df_train["PUlocationID"].isnull().sum() / df_train.shape[0], 2)
# %%
# [markdown] ## Fill NaN
df_train["PUlocationID"] = df_train["PUlocationID"].fillna("-1")
df_train["DOlocationID"] = df_train["DOlocationID"].fillna("-1")


df_train = df_train[(df_train.duration >= 1) & (df_train.duration <= 60)]
# %%
# [markdown] ## Q4 1 hot encode categorical variables

categorical = ["PUlocationID", "DOlocationID"]

dv = DictVectorizer()

df_train[categorical] = df_train[categorical].astype(str)
train_dicts = df_train[categorical].to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)
X_train.shape


# Repeat processing for validation df
df_val["duration"] = df_val.dropOff_datetime - df_val.pickup_datetime
df_val["duration"] = df_val["duration"].apply(lambda td: td.total_seconds() / 60)
df_val = df_val[(df_val.duration >= 1) & (df_val.duration <= 60)]
df_val["PUlocationID"] = df_val["PUlocationID"].fillna("-1")
df_val["DOlocationID"] = df_val["DOlocationID"].fillna("-1")

# Validation
df_val[categorical] = df_val[categorical].astype(str)
val_dicts = df_val[categorical].to_dict(orient="records")
X_val = dv.transform(val_dicts)

# %%
# [markdown] ## Train OLS
# %%
# [markdown] ## Setup train/test vals
target = "duration"
y_train = df_train[target].values
y_val = df_val[target].values

# %%
# [markdown] ## Train model
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

mean_squared_error(y_train, y_pred, squared=False)


# %%
# [markdown] ## Scores against validation
y_pred = lr.predict(X_val)

mean_squared_error(y_val, y_pred, squared=False)
