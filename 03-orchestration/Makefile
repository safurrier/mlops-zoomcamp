ml-flow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --serve-artifacts --artifacts-destination ./artifacts

# Create a local prefect storage instance
prefect-storage:
	prefect storage create
prefect-server:
	prefect config set PREFECT_ORION_UI_API_URL="http://localhost:4200/api"
	prefect orion start --host 0.0.0.0
prefect-deploy-flow:
	prefect deployment create homework.py
# hw:
# 	python homework.py