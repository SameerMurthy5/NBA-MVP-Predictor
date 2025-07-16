import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


#load data
with open("pickle/dataframe/data_df.pkl", "rb") as f:
    data_df = pickle.load(f)

predictors = [
    'Age', 'G', 'GS', 'MP_tot', 'MPG',
    'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
    'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP',
    'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%',
    'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
    'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS'
]

x = data_df[predictors] # features
y = data_df["Share"] # target variable
print(x)
print(y)

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)


# make predictions
y_pred = model.predict(x_test)


single_data = x_test.iloc[0].values.reshape(1, -1)
predicted_value = model.predict(single_data)
print(f"Predicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")


# evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
# save the model
with open("pickle/models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)
    



