# import libraries
import numpy as np
import pickle
import json, plotly
from datetime import datetime
from flask import Flask
from flask import render_template, request
from process_data import return_figures, transform_demographic_data, make_member_predictions
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
    age = request.args.get('age', '')
    try: # check if age is a float or int
        float(age)
    except: # if not, set to NaN
        age = np.nan
    income = request.args.get('income', '')
    try: # check if income is a float or int
        float(income)
    except: # if not, set to NaN
        income = np.nan
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

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        age=age,
        income=income,
        enrollment_date=enrollment_date_string,
        gender=gender,
        preds=preds,
        best_reward=best_reward)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
