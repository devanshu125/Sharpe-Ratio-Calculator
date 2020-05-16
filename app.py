from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sharpe', methods=['POST'])

# Calculate sharpe ratio and return graph

def predict():
    stock_list = [x for x in request.form.values()]
    stock_1 = stock_list[0]
    stock_2 = stock_list[1]

    # Reading in the data
    stock_data = pd.read_csv('datasets/stock_data.csv', parse_dates=['Date'], index_col='Date').dropna()
    benchmark_data = pd.read_csv('datasets/benchmark_data.csv', parse_dates=['Date'], index_col='Date').dropna()

    stock_data = stock_data[[stock_1, stock_2]]

    stock_returns = stock_data.pct_change()
    sp_returns = benchmark_data['S&P 500'].pct_change()

    # difference in daily returns
    excess_returns = stock_returns.sub(sp_returns, axis=0)

    # mean of excess returns
    avg_excess_return = excess_returns.mean()

    # std deviation
    sd_excess_return = excess_returns.std()

    # calculate the daily sharpe ratio
    daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

    # annualize the sharpe ratio
    annual_factor = np.sqrt(252)
    annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

    sharpe_0 = annual_sharpe_ratio[0]
    sharpe_1 = annual_sharpe_ratio[1]

    return render_template('predict.html', stock_1=stock_1, stock_2=stock_2, sharpe_0=sharpe_0, sharpe_1=sharpe_1)


if __name__ == "__main__":
    app.run(debug=True)
