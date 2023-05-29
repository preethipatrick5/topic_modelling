import os
import os.path
from os.path import isfile
import logging
import serving
import csv
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask_caching import Cache
from flask import Flask, request, render_template

config = {"DEBUG": True,  # some Flask specific configs
          "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
          "CACHE_DEFAULT_TIMEOUT": 0
          }
app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)
file_name = 'data.csv'


# various Flask explanations available at: https://flask.palletsprojects.com/en/1.1.x/quickstart/

def doRender(tname, values={}):
    if not os.path.isfile(os.path.join(os.getcwd(), 'templates/' + tname)):  # No such file
        return render_template('index.htm')
    return render_template(tname, **values)


# Defines a POST supporting estimate route
@app.route('/classify', methods=['POST'])
def classificationHandler():
    if request.method == 'POST':
        abstract = request.form.get('abstract')
        body = serving.main(abstract)
        label = body['labels']
        cf = body['confidenceScores']
        print("ConfidenceScores:")
        print(cf)
        classification = body['classification']
        keys = cf.keys()
        print("keys:", keys)
        values = cf.values()
        print("values:", values)
        plt.bar(keys, values)
        plt.xticks(rotation="45", fontsize=8)
        plt.savefig('static/images/plot.png')
        plt.close()
        if not os.path.isfile(file_name):
            with open(file_name, 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["Abstract", "Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology",
                     "Quantitative Finance"])
                file.close()
        with open('data.csv', 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([abstract, classification[0], classification[1], classification[2], classification[3],
                             classification[4], classification[5]])

        return render_template('result.htm', label=label)
    return 'Should not ever get here'


# catch all other page requests - doRender checks if a page is available (shows it) or not (index)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def mainPage(path):
    return doRender(path)


@app.errorhandler(500)
# A small bit of error handling
def server_error(e):
    logging.exception('ERROR!')
    return """
	An error occurred: <pre>{}</pre>
	""".format(e), 500


if __name__ == '__main__':
    # Entry point for running on the local machine
    # On GAE, endpoints (e.g. /) would be called.
    # Called as: gunicorn -b :$PORT index:app,
    # host is localhost; port is 8080; this file is index (.py)

    with app.app_context():
        cache.clear()

    app.run(host='127.0.0.1', port=8080, debug=True)
