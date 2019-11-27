# import libraries
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor
import plotly.graph_objs as go

# # read in profile dataset
# profile = pd.read_json('../data/profile.json',orient='records',lines=True)
#
# # Map 118 to NaN in 'Age' column
# profile['age'] = profile['age'].apply(lambda x: np.nan if x==118 else x)
#
# # Map 'None' to NaN in 'gender column'
# profile['gender'] = profile['gender'].apply(lambda x: np.nan if x==None else x)

# load member_preds_df
member_preds_df = pickle.load(open('../pickle_files/member_preds_df.p','rb'))

# load MinMaxScaler
scaler = pickle.load(open('../pickle_files/scaler.p','rb'))

# load portfolio_encoded
portfolio_encoded = pickle.load(open('../pickle_files/portfolio_encoded.p','rb'))

def return_figures():
    '''
    Creates 4 plotly visualizations of the demographic data

    Inputs:
    None

    Returns:
    figures (list of dicts): list of dictionaries containing the plotly figures and layouts
    '''
    # create the first plot - a histogram of ages
    graph_one = []
    graph_one.append(
     go.Histogram(
     x = member_preds_df['age'],
     xbins = dict(start=10,end=120,size=10),
     )
    )
    layout_one = dict(title='Distribution of Starbucks Rewards Member Ages',
                xaxis = dict(title = 'Age (years)'),
                yaxis = dict(title = 'Count')
                )

    # create the second plot - a histogram of income
    graph_two = []
    graph_two.append(
     go.Histogram(
     x = member_preds_df['income'],
     xbins = dict(start=0,end=140000,size=20000),
     )
    )
    layout_two = dict(title='Distribution of Starbucks Rewards Member Incomes',
                xaxis = dict(title = 'Income (USD)'),
                yaxis = dict(title = 'Count')
                )

    # create the third plot - a histogram of enrollment_date
    years = member_preds_df['enrollment_date'].apply(lambda x: x.year)
    graph_three = []
    graph_three.append(
     go.Histogram(
     x = years,
     xbins = dict(start=2013,end=2020,size=1),
     )
    )
    layout_three = dict(title='Distribution of Starbucks Rewards Member Enrollment Dates',
                xaxis = dict(title = 'Enrollment Year'),
                yaxis = dict(title = 'Count')
                )

    # create the fourth plot - a pie chart of the genders
    gender_labels = member_preds_df['gender'].value_counts().index.map({'M':'Male','F':'Female','O':'Other'})
    graph_four = []
    graph_four.append(
     go.Pie(
     values = member_preds_df['gender'].value_counts(),
     labels = gender_labels
     )
    )
    layout_four = dict(title='Breakdown of Starbucks Rewards Member Genders')

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))

    return figures

def return_reward_figure(preds):
    '''
    Creates a bar chart with the predicted transaction amounts for each reward type

    Inputs:
    preds (1D array): predicted transaction amounts for Rewards 0-10

    Returns:
    reward_figure (list of 1 dict): list containing one dictionary with data and layout of reward chart
    '''
    graph = []
    graph.append(
     go.Bar(
     x = range(11),
     y = preds
     )
    )
    layout = dict(title='Predicted Transaction Amount vs Reward ID',
                xaxis = dict(title = 'Reward ID'),
                yaxis = dict(title = 'Reward Adjusted Predicted Transaction Amount (USD)')
                )

    reward_figure = []
    reward_figure.append(dict(data=graph, layout=layout))
    return reward_figure

def transform_demographic_data(age,income,enrollment_date,gender):
    '''
    Take raw demographic data (age, income, enrollment date, gender) and transform it:
    - convert date to timestamp, encode gender, scale all values
    - add 11 zeros in front of data to create input array for `make_member_predictions`

    Inputs:
    age (int or float): member age in years or NaN if age is unknown (if unknown will use mean age)
    income (float): member annual income in USD or NaN if unknown (if unknown will use mean income)
    enrollment_date (datetime object): the date (mm-dd-YYYY) that a member enrolled
    gender (string): 'F','M','O', or 'U' (for unknown)

    Returns:
    member_inputs (1x17 array): normalized array containing values for:
    reward_ids (0-10), age, income, enrollment date, and genders (F,M,O)
    '''
    mean_age = member_preds_df['age'].mean() # calculate the mean age from the dataset (used to replace nulls)
    mean_income = member_preds_df['income'].mean() # calculate mean income from the dataset (used to replace nulls)
    if pd.isnull(age):
        age = mean_age # if age is null, replace with mean age
    if pd.isnull(income):
        income=mean_income # if income is null, replace with mean income
    tstamp = enrollment_date.timestamp() # convert enrollment date from datetime object to timestamp (for scaling)
    # create the gender encoding array based on the gender provided (if any)
    if gender == 'Female':
        gen_array = np.array([1,0,0])
    elif gender == 'Male':
        gen_array = np.array([0,1,0])
    elif gender == 'Other':
        gen_array = np.array([0,0,1])
    else:
        gen_array = np.array([0,0,0])
    member_inputs_raw = np.zeros((1,17)) # create 1x17 array of zeros to initialize the member inputs
    member_inputs_raw[0,11:14] = [age, income, tstamp] # replace elements 11-13 with the age, income and tstamp
    member_inputs_raw[0,14:] = gen_array # replace the last 3 elements with the gender encoding array
    member_inputs = scaler.transform(member_inputs_raw) # scale the inputs using the MinMaxScaler
    return member_inputs.reshape(1,17) # return the inputs in a 1x17 array

def make_member_predictions(model, member_inputs):
    '''
    Makes predictions for the amount a member will spend based on each of the 10 reward offers or no reward offer
    Deducts the reward amount if the predicted amount is greater than the difficulty to complete the reward

    Inputs:
    model: an sklearn model fit to training data
    member_inputs (2D array): normalized values for the input features with shape (# members, 17 features)
        the features (columns) must be in this order: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'age','income',
                                                     'enrollment_tstamp','gender_F','gender_M','gender_O'

    Returns:
    reward_adjusted_preds (2D array): the reward_adjusted predicted transaction amounts for the 11 reward options (no reward + 10 reward types)
    - shape (# members, 11 columns)
    '''
    reward = portfolio_encoded['reward'].values # extract reward values from portfolio_encoded
    reward = np.insert(reward,0,0) # insert a reward of 0 at the beginning for Reward 0 (no offer)
    difficulty = portfolio_encoded['difficulty'].values # extract difficulty values from portfolio_encoded (minimum spend)
    difficulty = np.insert(difficulty, 0, 0) # insert a difficulty of 0 at the beginning for Reward 0 (no offer)
    member_inputs = member_inputs.flatten() # flatten member_inputs into a 1D array
    member_preds = np.zeros(11) # initialize array to store member predictions
    member_inputs[:10] = 0 # reset the reward booleans to 0 so that only 1 reward is active at a time
    for idx in range(11): # loop through each reward_id (0 through 10)
        member_inputs[idx]=1 # set the reward_id boolean to 1 (True)
        # append the prediction to the member_preds array
        member_preds[idx] = model.predict(member_inputs.reshape(1,17))
        member_inputs[idx]=0 # set the reward_id back to 0
    # create an boolean array to determine if the predicted transaction is greater than the difficulty for each reward
    offer_met = np.greater_equal(member_preds,difficulty)
    # subtract the reward amount from the predicted transactions if the offer is met
    reward_adjusted_preds = np.subtract(member_preds, reward, out=member_preds, where=offer_met)
    return reward_adjusted_preds
