# import libraries
import json, plotly
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
# def go():
# go

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
