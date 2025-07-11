import pandas as pd
from mvp_preprocessing.predictors import load_data, preprocess_data
from Data_Visualization.Data_Visualization import plot_highest_scoring_seasons, plot_highest_scoring_per_season, plot_correlation_with_share
from models.machine_learning import train_mvp_predictor, backtest

# Load and preprocess data
# mvps, players, teams = load_data()
# stats = preprocess_data(mvps, players, teams)

#load new stats
stats = pd.read_csv("data/combined_preprocessed.csv")


# Data visualization
#plot_highest_scoring_seasons(stats)
#plot_highest_scoring_per_season(stats)
#plot_correlation_with_share(stats)

# Define predictors
predictors = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 
                'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year', 
                'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS']

new_predictors = [
    'Age', 'G', 'GS', 'MP_tot', 'MPG',
    'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
    'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP',
    'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%',
    'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
    'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS'
]

# Train model and predict MVP for a year 
year = 2021
model, results = train_mvp_predictor(stats, new_predictors, year)
print("MVP Predictions for", year, ":") 
print(results)

# Backtesting
mean_ap, aps, all_predictions = backtest(stats, model, range(1991, 2022), new_predictors)
print("Mean Average Precision (AP) for Backtesting:", mean_ap)
print(all_predictions[all_predictions["Rk"] < 5].sort_values("Diff").head(10))

