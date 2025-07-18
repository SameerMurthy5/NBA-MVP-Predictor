# NBA MVP Predictor:

## Additions

    7/10/25: adding advanced metrics to train on including (WS/48, VORP, etc) increasing model mean average precision of placing
            the proper players in the top 5 of MVP voting by 10 %

    3/13/25: storing model on Amazon S3 bucket and deplyed model on EC2 instance using fast API as an API endpoint to send data to frontend

    This project is a machine-learning based tool for predicting the NBA Most Valuable Player (MVP) award.
    By analyzing player statistics and historical MVP data, the goal of this predictor is to estimate
    the likelihood of a player winning the MVP award in a given season.
    It also includes data preprocessing, visualizations, and model evaluation to ensure accurate and meaningful results.

    This project began as an opportunity to learn and explore data preprocessing, visualization, and machine learning techniques in a real-world context through walkthroughs and resources online. I am now implementing additional models, feature engineering techniques, and custom improvements to make the predictor more accurate and adaptable.

# Next Steps:

    * apply more models to the dataset (ex. Random Forest)
    * dive more into feature engineering and create new stats or find new stats that will give more information to the models
    * rather than read from a csv, implement data retrieval through an API.

# Features:

    ## Data Preprocessing:

        * Combines player statistics, team data, and historical MVP information into a comprehensive dataset.

        * Cleans and normalizes data for use in machine learning models.

    ## Data Visualization:

        * Visualizes key trends in NBA history, such as top-scoring seasons and the correlation between player stats and MVP shares.

        * Generates insightful bar charts to highlight patterns and relationships.

    ## Machine Learning Prediction:

        * Uses Ridge linear regression to predict MVP shares based on player statistics.

        * Implements backtesting to evaluate model performance across multiple seasons.

        * uses Average Precision (AP) as an error metric to measure prediction accuracy.
