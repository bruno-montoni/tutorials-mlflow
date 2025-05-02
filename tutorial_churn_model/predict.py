# %%
# Importing Dependencies
import pandas as pd

import mlflow
import mlflow.client

# %%
# Setting the Tracking URI (just like in 'train.py')
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# %%
# Instantiating the client
client = mlflow.client.MlflowClient()
# Getting the aliased version ('get_latest_versions()' is deprecated)
model = client.get_model_version_by_alias(name= "tutorial_churn", # 'name': registered model name
                                          alias= "best") # 'alias': model alias
# Getting the version
version = model.version

# %%
# Loading the scikit-learn model from a run
model = mlflow.sklearn.load_model(f"models:/tutorial_churn/{version}") # We must make sure the flavor matches the registered model version

# %%
# Reading the CSV
df = pd.read_csv("data/churn_data.csv", sep= ",")
# (Optional) Confirming ouput
df.head()

# %%
# Sampling data
X = df.head()[model.feature_names_in_] # 'model' is a scikit-learn object
                                       # 'feature_names_in_': names of features seen during fit
                                       # 'X' only has predictor cols

# %%
# Predicting (for top 5 rows)
proba = model.predict_proba(X)
# (Optional) Confirming predictions
proba

# %%
