import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
data = pd.read_csv(r"A:\AA Project 155- Medical Inventary\New folder\preprocessed_data.csv")
df = data
# Assuming df is your time series DataFrame
# Convert 'DateTime' to datetime type if not already done
df['DateTime'] = pd.to_datetime(df['Dateofbill'], errors='coerce')

# Set 'DateTime' as the index for time series analysis
df_time_series = df.set_index('DateTime')

# Drop rows with NaN values in the target variable ('Final_Sales')
df_time_series = df_time_series.dropna(subset=['Final_Sales'])

# Fit ARIMA model
model_arima = ARIMA(df_time_series['Final_Sales'], order=(1, 1, 1))  # Adjust order as needed
result_arima = model_arima.fit()

# Forecast future values
forecast_arima = result_arima.get_forecast(steps=10)  # Adjust steps as needed

# Extract forecasted values and confidence intervals
forecast_values = forecast_arima.predicted_mean
confidence_intervals = forecast_arima.conf_int()

# Display the forecast values and confidence intervals
print("Forecasted Values:")
print(forecast_values)

# Plot the original time series and the forecast
plt.figure(figsize=(10, 6))
plt.plot(df_time_series['Final_Sales'], label='Actual')
plt.plot(forecast_values.index, forecast_values.values, color='red', label='ARIMA Forecast')
plt.fill_between(confidence_intervals.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='red', alpha=0.2, label='95% Confidence Interval')
plt.title('ARIMA Forecast vs Actual')
plt.legend()
plt.show()

# Calculate MAPE for the ARIMA model
mape_arima = mean_absolute_percentage_error(df_time_series['Final_Sales'][-10:], forecast_values)
print(f'MAPE for ARIMA model: {mape_arima}')
