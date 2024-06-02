import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Assuming df is your time series DataFrame
# Convert 'DateTime' to datetime type if not already done
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set 'DateTime' as the index
df_time_series = df.set_index('DateTime')






#################################################
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

# Assuming df is your time series DataFrame
# Convert 'DateTime' to datetime type if not already done
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set 'DateTime' as the index for time series analysis
df_time_series = df.set_index('DateTime')

# Drop rows with NaN values in the target variable ('Final_Sales')
df_time_series = df_time_series.dropna(subset=['Final_Sales'])

# Model 1: Moving Average (MA)
window_size = 3  # Adjust as needed
df_time_series['ma_predictions'] = df_time_series['Final_Sales'].rolling(window=window_size).mean()

# Drop rows with NaN values in the 'ma_predictions' column
df_time_series = df_time_series.dropna(subset=['ma_predictions'])

# Display the first few rows of the DataFrame with predictions
print(df_time_series[['Final_Sales', 'ma_predictions']].head())

# Calculate MAPE for the Moving Average (MA) model
mape_ma = mean_absolute_percentage_error(df_time_series['Final_Sales'], df_time_series['ma_predictions'])
print(f'MAPE for Moving Average (MA) model: {mape_ma}')




# Model 1: Moving Average (MA)
window_size = 3  # You can adjust the window size
df_time_series['MA_predictions'] = df_time_series['Final_Sales'].rolling(window=window_size).mean()

# Plot the original time series and MA predictions
plt.figure(figsize=(10, 6))
plt.plot(df_time_series['Final_Sales'], label='Original Time Series')
plt.plot(df_time_series['MA_predictions'], label=f'MA ({window_size}-period) Predictions', linestyle='--')
plt.legend()
plt.title('Moving Average (MA) Model')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Evaluate performance using MAPE
def calculate_mape(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / abs(y_true).mean() * 100
df_time_series['ma_predictions'] = df_time_series['Final_Sales'].rolling(window=window_size).mean()


#mape_ma = calculate_mape(df_time_series['Final_Sales'], df_time_series['MA_predictions'])
#print(f'MAPE for Moving Average (MA) model: {mape_ma}')


print(df_time_series[['Final_Sales', 'ma_predictions']].head())

mape_ma = mean_absolute_percentage_error(df_time_series['Final_Sales'], df_time_series['ma_predictions'])
print(f'MAPE for Moving Average (MA) model: {mape_ma}')

# Document findings, challenges, and optimizations
# Findings:
# - The MA model provides a smoothed representation of the time series.
# - It captures trends over the specified window size.

# Challenges:
# - The model may not handle abrupt changes or sudden spikes well.
# - Selection of the optimal window size is subjective and might require experimentation.

# Optimizations:
# - Experiment with different window sizes to find the optimal smoothing level.
# - Consider combining the MA model with other time series models for improved accuracy.
print(df_time_series[['Final_Sales', 'ma_predictions']])
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

# Assuming df is your time series DataFrame
# Convert 'DateTime' to datetime type if not already done
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set 'DateTime' as the index for time series analysis
df_time_series = df.set_index('DateTime')

# Drop rows with NaN values in the target variable ('Final_Sales')
df_time_series = df_time_series.dropna(subset=['Final_Sales'])

# Model 1: Moving Average (MA)
window_size = 3  # Adjust as needed
df_time_series['ma_predictions'] = df_time_series['Final_Sales'].rolling(window=window_size).mean()

# Drop rows with NaN values in the 'ma_predictions' column
df_time_series = df_time_series.dropna(subset=['ma_predictions'])

# Display the DataFrame for debugging
print(df_time_series[['Final_Sales', 'ma_predictions']])

# Calculate MAPE for the Moving Average (MA) model
mape_ma = mean_absolute_percentage_error(df_time_series['Final_Sales'], df_time_series['ma_predictions'])
print(f'MAPE for Moving Average (MA) model: {mape_ma}')


