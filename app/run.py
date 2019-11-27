# import libraries
import numpy as np
import pickle
import json, plotly
from datetime import datetime
from flask import Flask
from flask import render_template, request
from process_data import return_figures, return_reward_figure, transform_demographic_data, make_member_predictions
from sklearn.ensemble import BaggingRegressor

app = Flask(__name__)


# define routes
@app.route('/')
@app.route('/index')
def index():

    figures = return_figures() # get the list of demographic figures

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON)

# web page that handles user inputs of demographic data and displays model results and reward recommendation
@app.route('/go')
def go():
    # save user inputs
    age_string = request.args.get('age', '')
    try: # check if age is a float or int
        age = float(age_string)
        age_string = age_string + ' years'
    except: # if not, set to NaN
        age = np.nan
        age_string = '54.39 years'
    income_string = request.args.get('income', '')
    try: # check if income is a float or int
        income = float(income_string)
        income_string = '$' + income_string
    except: # if not, set to NaN
        income = np.nan
        income_string = '$65405'
    enrollment_date_string = request.args.get('enrollment_date', '')
    if enrollment_date_string == '': # check if enrollment date is blank
        enrollment_date = datetime.today() # if so, set to today
        enrollment_date_string = enrollment_date.strftime('%Y-%m-%d')
    else:
        enrollment_date = datetime.strptime(enrollment_date_string, '%Y-%m-%d')
    gender = request.args.get('gender', '') # get gender from radio buttons
    if gender == '': # check if gender is blank
        gender = 'None' # if so, set to 'None' (string not object)

    # load model
    model = pickle.load(open('../pickle_files/model.p','rb'))
    # use model to make reward recommendation
    preds = make_member_predictions(model,transform_demographic_data(age,income,enrollment_date,gender))
    best_reward = np.argmax(preds.flatten()) # identify best reward id

    reward_figure = return_reward_figure(preds) # get the reward_figure based on the preds
    id = ['figure-0'] # create the id for the html id tag
    # Convert the plotly figures to JSON for javascript in html template
    figureJSON = json.dumps(reward_figure, cls=plotly.utils.PlotlyJSONEncoder)

    # create reward_dict for displaying reward descriptions
    reward_dict = {0:'No reward',
                    1:'BOGO: $5, 5 days',
                    2:'BOGO: $5, 7 days',
                    3:'BOGO: $10, 5 days',
                    4:'BOGO: $10, 7 days',
                    5:'Discount: spend $7 get $3, 7 days',
                    6:'Discount: spend $10 get $2, 7 days',
                    7:'Discount: spend $10 get $2, 10 days',
                    8:'Discount: spend $20 get $5, 10 days',
                    9:'Informational: 3 days',
                    10:'Informational: 4 days'}

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        age=age_string,
        income=income_string,
        enrollment_date=enrollment_date_string,
        gender=gender,
        best_reward=best_reward,
        id=id,
        figureJSON=figureJSON,
        reward_dict=reward_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
