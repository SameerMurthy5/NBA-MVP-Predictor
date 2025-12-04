from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def find_ap(combination):
    actual = combination.sort_values("Share", ascending=False).head(5)
    predicted = combination.sort_values("predictions", ascending=False)
    ps, found, seen = [], 0, 1
    for _, row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            ps.append(found / seen)
        seen += 1
    return sum(ps) / len(ps)

def add_ranks(predictions):
    predictions = predictions.sort_values("predictions", ascending=False)
    predictions["Predicted_Rk"] = list(range(1, predictions.shape[0] + 1))
    predictions = predictions.sort_values("Share", ascending=False)
    predictions["Rk"] = list(range(1, predictions.shape[0] + 1))
    predictions["Diff"] = predictions["Rk"] - predictions["Predicted_Rk"]
    return predictions

def backtest(stats, years, predictors, model_constructor):
    aps, all_predictions = [], []
    for year in years[5:]:
        train = stats[stats["Year"] < year]
        test = stats[stats["Year"] == year]
        
        model = model_constructor()   # new fresh model
        model.fit(train[predictors], train["Share"])
        
        predictions = pd.DataFrame(model.predict(test[predictors]),
                                   columns=["predictions"], index=test.index)
        combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
        combination = add_ranks(combination)
        
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    
    mean_ap = sum(aps) / len(aps)
    return mean_ap, aps, pd.concat(all_predictions)

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
# Load data
def function():

    

    x = data_df[predictors]
    y = data_df["Share"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize and fit XGBRegressor
    bst = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    bst.fit(x_train, y_train)

    # Make predictions and evaluate
    preds = bst.predict(x_test)

    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    # Backtesting
    mean_ap, aps, all_predictions = backtest(
        stats=data_df,
        years=range(1991,2022),
        predictors=predictors,
        model_constructor=lambda: XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            objective='reg:squarederror',
            random_state=42
        )
    )

# print("Mean Average Precision (AP) for Backtesting:", mean_ap)
# print(all_predictions[all_predictions["Rk"] < 5].sort_values("Diff").head(10))


def predict_year(stats, year, predictors, model_constructor):
    # Train on all years before the target year
    train = stats[stats["Year"] < year]
    test = stats[stats["Year"] == year]
    
    # Create new fresh model
    model = model_constructor()
    
    # Fit on training data
    model.fit(train[predictors], train["Share"])
    
    # Predict for the target year
    predictions = pd.DataFrame(
        model.predict(test[predictors]),
        columns=["predictions"],
        index=test.index
    )
    
    # Combine with players + actual share
    combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
    combination = add_ranks(combination)
    
    return combination

year_pred = predict_year(
    stats=data_df,
    year=2021,
    predictors=predictors,
    model_constructor=lambda: XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective='reg:squarederror',
        random_state=42
    )
)

print(year_pred.sort_values("predictions", ascending=False).head(10))
