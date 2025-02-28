import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load CSV file (assumes 'Date' and 'Price' columns)
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return df

# Test for stationarity using ADF test
def test_stationarity(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary. Differencing will be applied.")
        return series.diff().dropna()
    return series

# Find optimal ARIMA order using AIC
def find_best_arima_order(series):
    best_aic = float("inf")
    best_order = None
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                except Exception as e:
                    print(f"Skipping ARIMA({p},{d},{q}) due to error: {e}")
                    continue
    
    print(f'Best ARIMA order: {best_order} with AIC: {best_aic}')
    return best_order if best_order else (1, 1, 1)  # Fallback order

# Prepare data for ARIMA model
def prepare_data(df):
    df['Days'] = (df.index - df.index.min()).days
    X = df[['Days']]
    y = df['Price']
    return train_test_split(y, test_size=0.2, random_state=42)

# Train ARIMA model
def train_model(y_train, order):
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    return model_fit

# Predict and evaluate
def predict_and_evaluate(model, y_test):
    predictions = model.forecast(steps=len(y_test))
    error = mean_absolute_error(y_test, predictions)
    return predictions, error

# Plot results
def plot_results(df, model):
    future_steps = 30
    future_preds = model.forecast(steps=future_steps)
    future_dates = pd.date_range(df.index[-1], periods=future_steps+1, freq='D')[1:]
    
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df['Price'], label='Actual Prices')
    plt.plot(future_dates, future_preds, linestyle='dashed', label='Forecast')
    plt.legend()
    plt.show()

# Main function
def main():
    file_path = 'amd_stock_data.csv'
    df = load_data(file_path)
    test_stationarity(df['Price'])
    best_order = find_best_arima_order(df['Price'])
    y_train, y_test = prepare_data(df)
    model = train_model(y_train, best_order)
    predictions, error = predict_and_evaluate(model, y_test)
    print(f'Mean Absolute Error: {error}')
    plot_results(df, model)

main()