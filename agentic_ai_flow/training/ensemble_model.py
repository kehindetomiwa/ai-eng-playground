from statistics import linear_regression

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from config_setting import ConfigSetting

config = ConfigSetting()

ensemble_data = config.ensemble_data_path
linear_regression_model_path = config.linear_regression_model_path

data = pd.read_pickle(ensemble_data)

print(data.head())

# drop price from dataframe
X = data.drop(columns=["Price"])
y = data["Price"]

# train
lr = LinearRegression()
lr.fit(X, y)

feature_columns = X.columns.tolist()

for feature, coef in zip(feature_columns, lr.coef_):
    print(f"Feature: {feature}, Coefficient: {coef}")
print(f"Intercept: {lr.intercept_}")

joblib.dump(lr, linear_regression_model_path)
