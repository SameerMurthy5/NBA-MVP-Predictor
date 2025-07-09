import pandas as pd
import pickle;
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def train_mvp_predictor(stats, predictors, year=2021):
    train = stats[stats["Year"] < year]
    test = stats[stats["Year"] == year]
    model = Ridge(alpha=.1)
    model.fit(train[predictors], train["Share"])
    
    predictions = pd.DataFrame(model.predict(test[predictors]), columns=["predictions"], index=test.index)
    results = pd.concat([test[["Player", "Share"]], predictions], axis=1)
    results = results.sort_values("Share", ascending=False)
    results["Rk"] = list(range(1, results.shape[0]+1))
    results = results.sort_values("predictions", ascending=False)
    results["Predicted_Rk"] = list(range(1, results.shape[0]+1))

    # save model as pkl file for deployment
    with open('mvp_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        print("Model saved as mvp_model.pkl")
        
    return model, results

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

def backtest(stats, model, years, predictors):
    aps, all_predictions = [], []
    for year in years[5:]:
        train = stats[stats["Year"] < year]
        test = stats[stats["Year"] == year]
        model.fit(train[predictors], train["Share"])
        predictions = pd.DataFrame(model.predict(test[predictors]), columns=["predictions"], index=test.index)
        combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
        combination = add_ranks(combination)
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    mean_ap = sum(aps) / len(aps)
    return mean_ap, aps, pd.concat(all_predictions)

# # for testing 
# stats = pd.read_csv("data_csv_files/player_mvp_stats.csv")
# stats = stats.fillna(0)

# # Define predictors
# predictors = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 
#                 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year', 
#                 'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS']

# # Train model and predict MVP for 2021
# model, results = train_mvp_predictor(stats, predictors)
# #print("MVP Predictions for 2021:")
# #print(results)

# # Backtesting
# mean_ap, aps, all_predictions = backtest(stats, model, range(1991, 2022), predictors)
# print("Mean Average Precision (AP) for Backtesting:", mean_ap)
# print(all_predictions[all_predictions["Rk"] < 5].sort_values("Diff").head(10))