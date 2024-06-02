
pip install gluonts


import pandas as pd
import matplotlib.pyplot as plt
from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.common import ListDataset
from sklearn.metrics import mean_absolute_percentage_error

# Assuming df is your time series DataFrame
# Convert 'DateTime' to datetime type if not already done
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set 'DateTime' as the index for time series analysis
df_time_series = df.set_index('DateTime')

# Drop rows with NaN values in the target variable ('Final_Sales')
df_time_series = df_time_series.dropna(subset=['Final_Sales'])

# Create GluonTS ListDataset
train_data = ListDataset([{"start": df_time_series.index[0], "target": df_time_series['Final_Sales'][:-10]}], freq="D")

# Model 6: DeepAR (Deep Autoregressive) Model
deepar_estimator = DeepAREstimator(freq="D", prediction_length=10, trainer__epochs=100)
deepar_predictor = deepar_estimator.train(train_data)
deepar_forecast = deepar_predictor.predict(df_time_series.index[-10:])

# Visualize the actual sales and DeepAR forecast
plt.figure(figsize=(12, 6))
plt.plot(df_time_series.index, df_time_series['Final_Sales'], label='Actual Sales', marker='o')
plt.plot(pd.to_datetime(deepar_forecast.index), deepar_forecast.mean, label='DeepAR Forecast', linestyle='dashed', marker='o')
plt.title('Actual Sales vs. DeepAR Forecast')
plt.xlabel('Date')
plt.ylabel('Final Sales')
plt.legend()
plt.show()

# Calculate MAPE for DeepAR model
mape_deepar = mean_absolute_percentage_error(df_time_series['Final_Sales'].iloc[-10:], deepar_forecast.mean)
print(f'MAPE for DeepAR model: {mape_deepar}')
