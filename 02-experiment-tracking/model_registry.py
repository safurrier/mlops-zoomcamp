# %%
# [markdown] ##
from mlflow.tracking import MlflowClient


# %%
# [markdown] ##
ML_FLOW_TRACKING_URI = "sqlite:///mlflow.db"

# %%
# [markdown] ##
client = MlflowClient(tracking_uri=ML_FLOW_TRACKING_URI)
client
# %%
client.list_experiments()
# %%
# Lazy create if doesn't exist. pass if error
try:
    client.create_experiment("cool-experiment")
except:  # MlflowException
    pass


# %%
# [markdown] ## Search best runs
from mlflow.entities import ViewType

runs = client.search_runs(
    # Hyperopt runs
    experiment_ids="5",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.rmse ASC"],
)
runs

# %%
for run in runs:
    print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']:4f}")

# %%

# %%
# [markdown] ## Promote best model
import mlflow
mlflow.set_tracking_uri(ML_FLOW_TRACKING_URI)

# %%
# [markdown] ## Register model
mlflow.register_model(model_uri=, name='')

