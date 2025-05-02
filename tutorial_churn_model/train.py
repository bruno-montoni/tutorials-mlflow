# %% ('tralha': allows to run '.py' file as cells in a Jupyter Notebook)
# Importing Dependencies
import pandas as pd

from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble

import mlflow

# %%
# Setting the Tracking URI
mlflow.set_tracking_uri(uri= "http://127.0.0.1:8080") # 'uri': must match our MLflow server
# Associating this code to an experiment 
mlflow.set_experiment(experiment_id= 288920102422483482)

# %%
# Reading the CSV
df = pd.read_csv("./data/churn_data.csv", sep= ",")
# Extracting features/target cols names
features = df.columns[2:-1]
target = "flag_churn"
# Individual df's
X = df[features]
y = df[target]
# (Optional) Confirming ouput
df.head()
# %%
# Splitting train/test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    test_size= 0.2,
                                                                    random_state= 42)
# Outputting train/test homogeneity
print("Mean y train:", y_train.mean()) 
print("Mean y test:", y_test.mean())

# %%
# Starting a new run
with mlflow.start_run():
    # Enabling autologging with 'sklearn' flavor
    mlflow.sklearn.autolog()
    
    # Model 1: Tree Classifier 
    clf = tree.DecisionTreeClassifier(min_samples_leaf= 10,
                                      min_samples_split= 10,
                                      random_state= 42)

    # Model 2: Random Forrest Classifier 
    #clf = ensemble.RandomForestClassifier(n_estimators= 500,
    #                                      min_samples_leaf= 35,
    #                                      random_state= 42)
    
    # Fitting the model
    clf.fit(X_train, y_train)
    # Predicting 
    y_train_predict = clf.predict(X_train)
    y_test_predict = clf.predict(X_test)
    # Model accuracy
    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    
    # Logging multiple custom metrics  
    mlflow.log_metrics(metrics= {"acc_train": acc_train, "acc_test": acc_test}) # 'metrics': dictionary of 'metric_name: value'
# %%
# Outputting results
print("Acurácia train:", acc_train)
print("Acurácia test:", acc_test)

# %%
