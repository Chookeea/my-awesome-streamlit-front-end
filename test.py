import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

taxi_data = pd.read_csv('yellow_tripdata_2021-01.csv')
regr = RandomForestRegressor(max_depth=10, n_estimators=100)

X = taxi_data[['PULocationID']]
y = taxi_data[['trip_distance']]
regr.fit(X, y)
prediction = regr.predict(y)
print(prediction)
print(taxi_data.columns.values)