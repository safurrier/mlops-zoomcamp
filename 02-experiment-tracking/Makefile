env:
	conda create -n mlopscamp python=3.9
setup:
	pip install -r requirements.txt

# Added arg for default artifact root folder
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns


preprocess-data:
	python homework/preprocess_data.py --raw_data_path data --dest_path ./output

# mlflow-ui step must be running in background
train:
	python homework/train.py

hyper-param-optimization:
	python homework/hpo.py

register-model:
	python homework/register_model.py