import pandas as pd

def load_data():
    mvps = pd.read_csv("data_csv_files/mvps.csv")
    players = pd.read_csv("data_csv_files/players.csv")
    del players["Unnamed: 0"]
    del players["Rk"]
    teams = pd.read_csv("data_csv_files/teams.csv")
    return mvps, players, teams

def load_nicknames():
    nicknames = {}
    with open("data_csv_files/nicknames.csv") as f:
        lines = f.readlines()
        for line in lines[1:]:
            abbrev, name = line.strip().split(",")
            nicknames[abbrev] = name
    return nicknames

def single_team(df):
    if df.shape[0] == 1:
        return df
    else:
        row = df[df["Tm"] == "TOT"]
        row["Tm"] = df.iloc[-1, :]["Tm"]
        return row

def preprocess_data(mvps, players, teams):
    mvps = mvps[["Player", "Year", "Pts Won", "Pts Max", "Share"]]
    players["Player"] = players["Player"].str.replace("*", "", regex=False)
    players = players.groupby(["Player", "Year"], group_keys=True).apply(single_team)
    players.index = players.index.droplevel(0).droplevel(0)
    
    combined = players.merge(mvps, how="outer", on=["Player", "Year"])
    combined[["Pts Won", "Pts Max", "Share"]] = combined[["Pts Won", "Pts Max", "Share"]].fillna(0)
    
    teams = teams[~teams["W"].str.contains("Division")]
    teams["Team"] = teams["Team"].str.replace("*", "", regex=False)
    nicknames = load_nicknames()
    combined["Team"] = combined["Tm"].map(nicknames)
    
    stats = combined.merge(teams, how="outer", on=["Team", "Year"])
    del stats["Unnamed: 0"]
    stats["GB"] = stats["GB"].str.replace("—", "0")
    stats = stats.apply(pd.to_numeric, errors="ignore")
    stats.to_csv("data_csv_files/player_mvp_stats.csv", index=False)
    stats = stats.fillna(0)
    return stats

# for testing 
# mvps, players, teams = load_data()
# stats = preprocess_data(mvps, players, teams)
# print(stats.head())