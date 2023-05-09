import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error


data = pd.read_csv('data_C02_emission.csv')

input_variables = [
    "Fuel Consumption City (L/100km)",
    "Fuel Consumption Hwy (L/100km)",
    "Fuel Consumption Comb (L/100km)",
    "Fuel Consumption Comb (mpg)",
    "Engine Size (L)",
    "Cylinders",
    "Fuel Type",
]

output_variable = ["CO2 Emissions (g/km)"]


ohe = OneHotEncoder()
encoded_fuel_type = ohe.fit_transform(data[["FUEL_TYPE"]]).toarray()
data["Fuel Type"] = encoded_fuel_type

X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

plt.figure()
plt.scatter(x=X_train[:, 0], y=y_train, c="b")
plt.scatter(x=X_test[:, 0], y=y_test, c="r")
plt.show()

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)

y_test_p = linearModel.predict(X_test_n)

RMSE = mean_squared_error(y_test, y_test_p, squared=False)
MAE = mean_absolute_error(y_test, y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
R2 = r2_score(y_test, y_test_p)

plt.figure()
plt.scatter(x=X_test_n[:, 0], y=y_test, c="b")
plt.scatter(x=X_test_n[:, 0], y=y_test_p, c="r")
plt.show()

print("RMSE", RMSE)
print("MAE", MAE)
print("MAPE", MAPE)
print("R2", R2)