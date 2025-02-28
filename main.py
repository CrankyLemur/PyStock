import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load CSV file (assumes 'Date' and 'Close' columns)
def load_data(file_path):
    df_history = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return df_history

# Prepare data for ML model
def prepare_data(df):
    df['Days'] = (df.index - df.index.min()).days  # Convert dates to numerical values
    X = df[['Days']]
    y = df['Close']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Predict and evaluate
def predict_and_evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    return predictions, error

# Plot results
def plot_results(df, model):
    future_days = np.arange(df['Days'].max() + 1, df['Days'].max() + 30).reshape(-1, 1)
    future_preds = model.predict(future_days)
    
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df['Close'], label='Actual Prices')
    plt.plot(pd.date_range(df.index[-1], periods=30), future_preds, linestyle='dashed', label='Forecast')
    plt.legend()
    plt.show()

# Main function
def main():
    file_path = 'amd_stock_data.csv'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)
    predictions, error = predict_and_evaluate(model, X_test, y_test)
    print(f'Mean Absolute Error: {error}')
    plot_results(df, model)

if __name__ == '__main__':
    main()