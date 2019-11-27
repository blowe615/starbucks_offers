# Starbucks Reward Offers

## Table of Contents
* [Installations](#installations)
* [Project Motivation](#project-motivation)
* [File Descriptions](#file-descriptions)
* [Instructions](#instructions)
* [Results](#results)
  - [Model](#model)
  - [Web App](#web-app)
* [Acknowledgements](#acknowledgements)

## Installations
All of the following packages were used for this project and can be found in the standard Anaconda distribution for Python 3.7:
* Jupyter notebook
* NumPy
* Pandas
* Scikit-learn
* Flask
* Plotly

## Project Motivation
This project serves as my Capstone Project for Udacity's [Data Scientist Nanodegree](https://www.udacity.com/school-of-data-science) program.  Students were tasked with applying data science principles to a real-world problem and sharing their solutions on GitHub and either a blog post or web app.  I decided to use a dataset provided to Udacity by Starbucks for my Capstone project.  As a longtime [Starbucks Rewards](https://www.starbucks.com/rewards/) member, I was excited to dig into how Starbucks makes decisions on reward offers because it is something that I use in my everyday life.  Plus I figured my familiarity with the subject matter would help with my analysis.

## File Descriptions
There are 4 main directories in this repository:
#### data
All of the data provided by Starbucks to complete this project.  Note that the data is simulated for a single product rather than actual data from real members.
* `portfolio.json`: JSON file containing information about the 10 types of reward offers sent to Starbucks Rewards members during the 30-day test period.
* `profile.json`: JSON file containing demographic information (age, income, enrollment date, gender, and id) about the 17,000 members included in the test period.
* `transcript.json`: JSON file containing the event log of when offers were sent, viewed, and completed by members and when transactions made by the members during the 30-day test period.

#### notebook
File containing the jupyter notebook in which I performed my investigations and analysis
* `starbucks_offers.ipynb`: Jupyter notebook in which I performed my analysis.  Contains the following sections: Introduction, Data Exploration, Data Cleaning, Feature Extraction, Modeling, Results, and Conclusion.
* `starbucks_offers.html`: HTML file of `starbucks_offers.ipynb` for viewing.

#### pickle_files
All of the pickle files that I saved to be used either in the web app or to save time when re-running `starbucks_offers.ipynb`.
* `learning_df.p`: Pandas DataFrame containing only the features to be used in training and testing ML models (input variables and output variable).
* `member_preds_df.p`: Pandas DataFrame containing member demographic data from `profile` data set along with the reward recommended for each member.  Used in plotting results in `starbucks_offers.ipynb` and in demographic distributions for the web app.
* `model.p`: Tuned scikit-learn Bagging Regressor model trained on the training subset of the `learning_df` DataFrame.
* `portfolio_encoded.p`: Pandas DataFrame containing the encoded values of the `portfolio` data set.
* `scaler.p`: Scikit-learn MinMaxScaler object fit to the training data from `learning_df`.  Used to scale the user inputs in the web app.
* `transactions_encoded.p`: Pandas DataFrame containing the encoded features of the `transcript` data set for only transaction events.  Used to avoid having to re-encode the DataFrame every time the Jupyter notebook was rerun.
* `transcript_encoded.p`: Pandas DataFrame containing the encoded features of the `transcript` data set for all event types.  Used to avoid having to re-encode the DataFrame every time the Jupyter was rerun.

#### app
All of the files required to run the Flask web app.
* `run.py`: Python script to run the web app displaying visualizations of the dataset and allowing users to input demographic data and view reward recommendations.
* `process_data.py`: Python script containing helper functions used in `run.py` to create Plotly figures and make reward recommendations on user input demographic data.
* `templates`: directory containing 2 html files:
  -  `index.html`: Main page of web app. Contains visualizations of member demographic data.
  -  `go.html`: Reward recommendation result page of web app.

## Instructions:
1. Run the following command in the `app` directory to launch the web app.

    `python run.py`

2. Go to http://0.0.0.0:3001/ to view the web app.

## Results
### Model
For a thorough discussion of the project results, please see the **Results Discussion** section of `starbucks_offers.ipynb`.

The goal of my project was to recommend a reward type to any Starbucks Rewards member based on demographic data such as their age, income, enrollment date in the program, and gender.  To accomplish this, I trained and tested a tree based Bagging regression model on the transactions included in `transcript` data set.  The purpose of the model was to predict the amount a member would spend in a given transaction based on their demographic data and the reward offer that was active (if any) when the transaction occurred. My final tuned regression model had a mean absolute error of $2.81 on the training data and $3.83 on the testing data.  The plot below shows the relationship between the predicted and actual transaction amounts in the training set.

![](https://github.com/blowe615/starbucks_offers/blob/master/preds_vs_actual.png)

With my trained model, I was able to make reward recommendations for any member based on their demographic data.  For a given member, I used my model to determine their predicted transaction amount for each of the reward offer types, resulting in 11 predicted transaction amounts (10 reward types plus the no reward option).  I then recommended the reward type that resulted in the maximum predicted transaction amount for that member.  The plot below shows the distribution of reward recommendations by reward type.

![](https://github.com/blowe615/starbucks_offers/blob/master/reward_rec_freqs.png)

Reward 0 (no reward), Reward 8 (10-day, 25% discount on a $20 purchase) Reward 10 (4-day informational offer), and Reward 9 (3-day informational offer) were the 4 most commonly recommended reward types to the 17,000 members in the data set.  For discussion as to why I think these reward types were recommended the most, please see the **Results Discussion** section of `starbucks_offers.ipynb`.

### Web App
The main page of the web app shows 4 plots about the demographic data of the members used during this project.

At the top of the page are a series of input boxes where the user can input any combination of age, income, enrollment_date and gender and view the recommended reward type for a member with those demographics.

If Age or Income are left blank or are not numerical values, the model will default to the mean values (54.39 years and $65,405) to make a reward recommendation.  If Enrollment Date is left blank, the model will default to today's date.  If Gender is left blank, no gender will be input into the model.  This is different than selecting 'Other'.

After clicking the 'Submit' button, the user will be taken to a second page where the recommended reward type will be displayed, along with a plot showing the predicted transaction amount for each Reward ID and a description of each reward type.  From there the user can continue entering demographic data and viewing reward recommendations, or can click on 'Starbucks Reward Offers' in the navbar to return to the main page.

![](https://github.com/blowe615/starbucks_offers/blob/master/reward_rec_results.png)

## Acknowledgements
This project would not have been possible without the partnership between Udacity and [Starbucks](https://www.starbucks.com/).  Thank you Starbucks for sharing this data set and allowing me to work on a project with real world implications.  Thank you to Alvira Swalin for writing this very helpful [blog](https://heartbeat.fritz.ai/how-to-make-your-machine-learning-models-robust-to-outliers-44d404067d07) post on dealing with outliers in Machine Learning models.  I also found the Stack Overflow posts and the documentation for each of the python packages extremely helpful in completing this project.
