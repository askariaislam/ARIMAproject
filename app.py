import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np

def get_user_input():
    # User input for stock selection
    ticker = input("Enter the ticker symbol of the stock you want to forecast: ")

    # User input for timeframe
    start_date_input = input("Enter the start date in the format 'YYYY-MM-DD' (or enter 'start' to fetch data from the earliest date): ")
    end_date_input = input("Enter the end date in the format 'YYYY-MM-DD' (or enter 'now' to fetch data until the most recent day): ")

    # User input for future prediction period
    while True:
        try:
            n_periods = int(input("Enter the number of days in the future you want to predict: "))
            break
        except ValueError:
            print("Please enter a valid number.")
    return ticker, start_date_input, end_date_input, n_periods

def get_stock_data(ticker, start_date_input, end_date_input):
    # Getting stock data
    tickerData = yf.Ticker(ticker)

    # Determine start date
    if start_date_input.lower() == 'start':
        start_date = None
    else:
        start_date = start_date_input

    # Determine end date
    if end_date_input.lower() == 'now':
        end_date = None
    else:
        end_date = end_date_input

    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
    tickerDf = tickerDf[['Close']]
    tickerDf = tickerDf.asfreq('B').fillna(method='ffill') # filling NaNs using forward fill method
    return tickerDf

def check_stationarity(df):
    # Check if 'df' is empty
    if df.empty:
        print("The 'df' DataFrame is empty. Please check your data.")
        return False
    else:
        # Check for stationarity with Augmented Dickey-Fuller test
        result = adfuller(df.dropna())
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        return True

def fit_arima_model(df):
    # Fitting the ARIMA model using auto_arima
    model = auto_arima(df, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    model_fit = model.fit(df)
    return model_fit

def forecast(model_fit, n_periods):
    # Making forecast
    future_forecast, conf_int = model_fit.predict(n_periods=n_periods, return_conf_int=True)
    return future_forecast, conf_int

def plot_forecast(train, future_forecast, conf_int):
    # Create a date range for future dates
    future_dates = pd.date_range(start=train.index[-1], periods=len(future_forecast)+1, freq='B')[1:]  # start from the day after the last date in the training set

    # Create a pandas series for the future forecast and set the index to the future dates
    future_forecast_series = pd.Series(future_forecast, index=future_dates)

    # Define lower and upper series for the confidence interval
    lower_series = pd.Series(conf_int[:, 0], index=future_dates)
    upper_series = pd.Series(conf_int[:, 1], index=future_dates)

    # Plotting original data and forecast
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(future_forecast_series, label='future forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

def calculate_percentage_profit(train, future_forecast):
    last_known_price = train.iloc[-1, 0]
    forecasted_price = future_forecast[-1]
    
    profit = ((forecasted_price - last_known_price) / last_known_price) * 100
    return profit


def main():
    # get user input
    ticker, start_date_input, end_date_input, n_periods = get_user_input()
    # get stock data
    df = get_stock_data(ticker, start_date_input, end_date_input)
    if check_stationarity(df):
        model_fit = fit_arima_model(df)
        future_forecast, conf_int = forecast(model_fit, n_periods)
        plot_forecast(df, future_forecast, conf_int)

if __name__ == "__main__":
    main()
