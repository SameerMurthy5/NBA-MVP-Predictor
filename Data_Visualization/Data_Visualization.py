import pandas as pd
import matplotlib.pyplot as plt

def plot_highest_scoring_seasons(stats):
    highest_scoring = stats[stats["G"] > 70].sort_values("PTS", ascending=False).head(10)
    highest_scoring.plot.bar(x="Player", y="PTS")
    plt.title("Top 10 Highest Scoring Seasons")
    plt.show()

def plot_highest_scoring_per_season(stats):
    highest_scoring_per_season = stats.groupby("Year").apply(lambda x: x.sort_values("PTS", ascending=False).head(1))
    highest_scoring_per_season.plot.bar(x="Year", y="PTS")
    plt.title("Highest Scorer Per Season")
    plt.show()

def plot_correlation_with_share(stats):
    numeric_stats = stats.select_dtypes(include=["number"])
    numeric_stats.corr()["Share"].plot.bar()
    plt.title("Correlation of Features with MVP Share")
    plt.show()

# for testing 
# stats = pd.read_csv("data_csv_files/player_mvp_stats.csv")
# plot_highest_scoring_seasons(stats)
# plot_highest_scoring_per_season(stats)
# plot_correlation_with_share(stats)