# import libraries
import numpy as np
import json, plotly
from datetime import datetime
from flask import Flask
from flask import render_template
from process_data import return_figures


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

# web page that handles user query and displays model results
# @app.route('/go')
def go():
    # save user inputs
    # query = request.args.get('query', '')
    age = request.args.get('age', np.nan)
    income = request.args.get('income', np.nan)
    enrollment_date = request.args.get('enrollment_date', datetime(2017,1,1))
    gender = request.args.get('gender', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
# go

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
