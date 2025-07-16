import pandas as pd
import pickle

years = range(1991, 2022)

def preprocess_advanced_data():
    for year in years:
        data = pd.read_html("advanced_data/html/{}.html".format(year))
        print(f"Processing data for {year}...")
        table = data[0]
        table = table[table["Player"] != "Player"] # remove header rows
        table = table.drop(columns="Awards") # remove awards column
        table = table.drop(table.index[-1]) # drop last row
        table = table.drop(columns="Rk") # remove rank column
        table["Year"] = year # add year column

        players = table.groupby(["Player", "Year"], as_index=False).first()
        players.to_csv("advanced_data/csv/{}.csv".format(year), index=False)



def combine_advanced_data():
    combined = pd.DataFrame()
    for year in years:
        data = pd.read_csv("data/advanced_data/csv/{}.csv".format(year))
        combined = pd.concat([combined, data], ignore_index=True)
    
    combined.to_csv("data/advanced_data/combined_advanced_data.csv", index=False)
    print("Combined advanced data saved to advanced_data/combined_advanced_data.csv")



def combine_all_data(advanced, mvp_stats):
    mvp_stats["Player"] = mvp_stats["Player"].replace("J.J. Redick", "JJ Redick")
    combined = advanced.merge(mvp_stats, how="outer", on=["Player", "Year"])
    combined.to_csv("data/real_combined.csv", index=False)

def preprocess_combined_data():
    combined = pd.read_csv("data/real_combined.csv")
    print(combined.head())
    # Preprocess combined data
    combined.drop(columns=["Age_y", "Pos_y", "G_y", "GS_y"], inplace=True)
    combined.rename(columns={
        "Age_x": "Age",
        "Pos_x": "Pos",
        "G_x": "G",
        "GS_x": "GS",
        "Team_x": "Tm_abbr",
        "Team_y": "Team",
        "MP_x": "MP_tot",
        "MP_y": "MPG"
    }, inplace=True)

    # most columns with NaN are percentages or 0/0
    combined = combined.fillna(0)
    # Save the preprocessed combined data
    combined.to_csv("data/combined_preprocessed.csv", index=False)

