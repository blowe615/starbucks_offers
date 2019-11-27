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

    figures = return_figures()

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
    age = request.args.get('age', np.nan)
    income = request.args.get('income', np.nan)
    enrollment_date = request.args.get('enrollment_date', datetime(2017,1,1))
    gender = request.args.get('gender', '')

    # # load model
    # model = pickle.load(open('../pickle_files/model.p','rb'))
    # # use model to make reward recommendation
    # preds = make_member_predictions(model,transform_demographic_data(age,income,enrollment_date,gender))
    # best_reward = np.argmax(preds.flatten())
    preds=[1,2]
    best_reward=5

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        age=age,
        income=income,
        enrollment_date=enrollment_date,
        gender=gender,
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
