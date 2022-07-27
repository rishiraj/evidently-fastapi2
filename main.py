import pandas as pd
import numpy as np
import requests
import zipfile
import io

from datetime import datetime
from sklearn import datasets, ensemble

from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab, NumTargetDriftTab, RegressionPerformanceTab

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

"""## Bicycle Demand Data

This step automatically downloads the bike dataset from UCI. This version is slightly different from the dataset used in Kaggle competition. If you want the example to be identical to the one in the Evidently blog "How to break a model in 20 days", you can manually download the dataset from Kaggle: https://www.kaggle.com/c/bike-sharing-demand/data 

And add this code:

raw_data['mnth'] = raw_data.index.map(lambda x : x.month)

raw_data['hr'] = raw_data.index.map(lambda x : x.hour)

raw_data['weekday'] = raw_data.index.map(lambda x : x.weekday() + 1)
"""

content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday'], index_col='dteday')

raw_data.head()

"""## Regression Model

### Model training
"""

target = 'cnt'
prediction = 'prediction'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']

reference = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
current = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']

reference.head()

regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)

regressor.fit(reference[numerical_features + categorical_features], reference[target])

ref_prediction = regressor.predict(reference[numerical_features + categorical_features])
current_prediction = regressor.predict(current[numerical_features + categorical_features])

reference['prediction'] = ref_prediction
current['prediction'] = current_prediction

"""### Model Perfomance """

column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.prediction = prediction
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

regression_perfomance_dashboard = Dashboard(tabs=[RegressionPerformanceTab()])
regression_perfomance_dashboard.calculate(reference, None, column_mapping=column_mapping)

regression_perfomance_dashboard.save("./static/index.html")

"""##  Week 1"""

regression_perfomance_dashboard.calculate(reference, current.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00'], 
                                            column_mapping=column_mapping)

regression_perfomance_dashboard.save("./static/regression_performance_after_week1.html")

target_drift_dashboard = Dashboard(tabs=[NumTargetDriftTab()])
target_drift_dashboard.calculate(reference, current.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00'], 
                                   column_mapping=column_mapping)

target_drift_dashboard.save("./static/target_drift_after_week1.html")

"""## Week 2"""

regression_perfomance_dashboard.calculate(reference, current.loc['2011-02-07 00:00:00':'2011-02-14 23:00:00'], 
                                            column_mapping=column_mapping)

regression_perfomance_dashboard.save("./static/regression_performance_after_week2.html")

target_drift_dashboard.calculate(reference, current.loc['2011-02-07 00:00:00':'2011-02-14 23:00:00'], 
                                   column_mapping=column_mapping)

target_drift_dashboard.save("./static/target_drift_after_week2.html")

"""## Week 3"""

regression_perfomance_dashboard.calculate(reference, current.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00'], 
                                            column_mapping=column_mapping)

regression_perfomance_dashboard.save("./static/regression_performance_after_week3.html")

target_drift_dashboard.calculate(reference, current.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00'], 
                                   column_mapping=column_mapping)

target_drift_dashboard.save("./static/target_drift_after_week3.html")

"""## Data Drift"""

column_mapping = ColumnMapping()

column_mapping.numerical_features = numerical_features

data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
data_drift_dashboard.calculate(reference, current.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00'], 
                                   column_mapping=column_mapping)

data_drift_dashboard.save("./static/data_drift_dashboard_after_week1.html")

app = FastAPI()

app.mount("/", StaticFiles(directory="static",html = True), name="static")
