# import libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
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

# read in member_preds_df
member_preds_df = pickle.load(open('pickle_files/member_preds_df.p','rb'))

def return_figures():
    '''
    Creates 9 plotly visualizations:
    - 4 visualizations of the demographic data
    - 5 visualizations showing the frequency of offer recommendations by demographic

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
     xbins = range(10,120,10),
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
     xbins = range(0,140000,20000),
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
     xbins = range(2013,2020),
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
     ids = gender_labels
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
