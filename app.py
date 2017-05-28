import os

from flask import Flask, request
import pandas as pd
import numpy as np


app = Flask(__name__)


def RMSE(true, pred, axis=None):
    """
    Function computes RMSE.

    :param true: expected values
    :param pred: predicted values
    :return: computed RMSE
    """
    if hasattr(true, 'values'):
        true = true.values
    if hasattr(pred, 'values'):
        pred = pred.values
    diff = true - pred
    mse = np.mean(diff ** 2, axis=axis)
    rmse = np.sqrt(mse)
    return rmse


@app.route('/submission', methods=['POST'])
def submit():
    content = request.json
    df = pd.DataFrame.from_dict(content)
    return str(RMSE(df, df))


@app.route('/', methods=['GET'])
def index():
  return "HELLO WORLD"

if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port, debug=True)
