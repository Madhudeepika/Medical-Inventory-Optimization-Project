import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_percentage_error

# Assuming df is your time series DataFrame
# Convert 'DateTime' to datetime type if not already done
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set 'DateTime' as the index for time series analysis
df_time_series = df.set_index('DateTime')

# Drop rows with NaN values in the target variable ('Final_Sales')
df_time_series = df_time_series.dropna(subset=['Final_Sales'])

# Choose your actual exogenous features
exog_columns = ['Typeofsales', 'Final_Cost']

# Ensure exogenous features have numeric data types
df_time_series[exog_columns] = df_time_series[exog_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in the exogenous features
df_time_series = df_time_series.dropna(subset=['Final_Sales'] + exog_columns)

# Model 5: Vector Autoregression (VAR)
model_var = VAR(df_time_series[['Final_Sales'] + exog_columns])

# Manually loop through potential lag orders and choose the one with the lowest AIC
best_aic = float('inf')
best_lag_order = None

# Ensure enough data points for fitting
for lag_order in range(1, min(13, len(df_time_series) - 1)):  # Adjust the range as needed
    try:
        results_var = model_var.fit(lag_order)
        aic = results_var.aic

        if aic < best_aic:
            best_aic = aic
            best_lag_order = lag_order
    except ValueError:
        continue

# Check if a valid lag order was found
if best_lag_order is not None:
    # Fit the VAR model with the selected lag order
    results_var = model_var.fit(best_lag_order)

    # Forecasting
    forecast_var = results_var.forecast(df_time_series[['Final_Sales'] + exog_columns].values[-best_lag_order:], steps=10)  # Adjust the number of steps as needed

    # Create DataFrame for forecast results
    df_forecast_var = pd.DataFrame(forecast_var, columns=['Final_Sales_forecast'] + exog_columns, index=pd.date_range(start=df_time_series.index[-1] + pd.Timedelta(days=1), periods=10, freq='D'))

    # Visualize the actual sales and VAR forecast
    plt.figure(figsize=(12, 6))
    plt.plot(df_time_series.index, df_time_series['Final_Sales'], label='Actual Sales', marker='o')
    plt.plot(df_forecast_var.index, df_forecast_var['Final_Sales_forecast'], label='VAR Forecast', linestyle='dashed', marker='o')
    plt.title('Actual Sales vs. VAR Forecast')
    plt.xlabel('Date')
    plt.ylabel('Final Sales')
    plt.legend()
    plt.show()

    # Calculate MAPE for VAR model
    mape_var = mean_absolute_percentage_error(df_time_series['Final_Sales'].iloc[-10:], df_forecast_var['Final_Sales_forecast'])
    print(f'MAPE for VAR model: {mape_var}')
else:
    print("Not enough data points to fit the VAR model with any lag order.")
